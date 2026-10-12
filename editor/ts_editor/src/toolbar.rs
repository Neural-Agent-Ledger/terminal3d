//! The top toolbar — the COMMAND BAR organ — rendered into the one-row
//! [`crate::layout::Layout::toolbar`] strip. Like a real 3D editor's command bar,
//! the buttons are grouped into named CLUSTERS divided by `│` rules so the eye
//! finds an action by function instead of scanning a flat row:
//!
//!   **File** \[New level\]\[Open\]\[Save\]\[Save As\]  │  **Edit** \[Undo\]\[Redo\]
//!   │  **View** \[2D view\]\[3D view\]\[Split view\]  │  **Help**
//!
//! The button COLUMN SPANS are the single source of truth ([`button_spans`]) used
//! by BOTH the painter ([`draw_toolbar`]) and the hit-test ([`toolbar_hit`]) so a
//! click always lands on the button drawn under it. The cluster dividers come
//! from the SAME [`CLUSTER_BREAK_AFTER`] table both functions read, so paint and
//! hit can never drift. Pure layout/hit math (no terminal); the Bevy mouse layer
//! routes clicks here via `mouse::apply_mouse`.
use crate::state::{EditorState, ViewMode};
use re_core::{text::draw_str, Cell, Framebuffer};

/// A toolbar button. Ordered by CLUSTER: File, then Edit, then View, then Help.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ToolbarButton {
    // File cluster.
    New,
    Open,
    Save,
    SaveAs,
    // Edit cluster.
    Undo,
    Redo,
    // View cluster.
    Mode2D,
    Mode3D,
    ModeSplit,
    LevelSettings,
    // Help (pinned last).
    Help,
}

/// The label drawn inside a button's brackets.
fn button_label(b: ToolbarButton) -> &'static str {
    match b {
        ToolbarButton::Mode2D => "2D view",
        ToolbarButton::Mode3D => "3D view",
        ToolbarButton::ModeSplit => "Split view",
        ToolbarButton::LevelSettings => "Level",
        ToolbarButton::Open => "Open",
        ToolbarButton::Save => "Save",
        ToolbarButton::SaveAs => "Save As",
        ToolbarButton::New => "New level",
        ToolbarButton::Undo => "Undo",
        ToolbarButton::Redo => "Redo",
        ToolbarButton::Help => "Help",
    }
}

/// The buttons in paint order (also hit-test order), grouped by cluster:
/// File (New/Open/Save/Save As) · Edit (Undo/Redo) · View (2D/3D/Split) · Help.
pub const BUTTONS: [ToolbarButton; 11] = [
    // File
    ToolbarButton::New,
    ToolbarButton::Open,
    ToolbarButton::Save,
    ToolbarButton::SaveAs,
    // Edit
    ToolbarButton::Undo,
    ToolbarButton::Redo,
    // View
    ToolbarButton::Mode2D,
    ToolbarButton::Mode3D,
    ToolbarButton::ModeSplit,
    ToolbarButton::LevelSettings,
    // Help
    ToolbarButton::Help,
];

/// The buttons AFTER which a cluster divider (`│`) is drawn — i.e. the last
/// button of each cluster except the final one. THE single source of truth for
/// the cluster breaks, read by both [`button_spans`] (which reserves [`SEP_W`]
/// columns for the divider) and [`draw_toolbar`] (which paints the `│` there), so
/// the painted dividers always line up with the gaps the hit-test skips.
///
/// Breaks after Save As (end of File), after Redo (end of Edit), and after
/// Level (end of View — the View cluster is 2D/3D/Split/Level — just before the
/// pinned Help button).
const CLUSTER_BREAK_AFTER: [ToolbarButton; 3] = [
    ToolbarButton::SaveAs,
    ToolbarButton::Redo,
    ToolbarButton::LevelSettings,
];

/// True when a cluster divider is drawn after `b`.
fn is_cluster_break(b: ToolbarButton) -> bool {
    CLUSTER_BREAK_AFTER.contains(&b)
}

/// Leading pad before the first button (one space so the row breathes).
const LEAD: usize = 1;
/// A `" │ "` cluster divider is drawn at each [`CLUSTER_BREAK_AFTER`] boundary.
/// Its width in columns (one pad + the rule + one pad).
const SEP_W: usize = 3;

/// The clickable column SPAN `[start, end)` of each button, relative to the
/// toolbar's left edge (column 0 = `toolbar.x`). A button's rendered text is
/// `"[LABEL]"`, so its span is `label.len() + 2` columns wide; a single space
/// separates adjacent buttons within a cluster and a `" │ "` ([`SEP_W`] wide)
/// divides the clusters at each [`CLUSTER_BREAK_AFTER`] boundary. THE single
/// source of truth shared by the painter and the hit-test.
///
/// Returns `(button, start, end)` tuples in paint order.
pub fn button_spans() -> Vec<(ToolbarButton, usize, usize)> {
    let mut out = Vec::with_capacity(BUTTONS.len());
    let mut col = LEAD;
    for (i, &b) in BUTTONS.iter().enumerate() {
        let w = button_label(b).len() + 2; // brackets
        out.push((b, col, col + w));
        col += w;
        // At a cluster boundary reserve the wider `│` divider; otherwise a single
        // space between adjacent buttons within the same cluster.
        if is_cluster_break(b) {
            col += SEP_W;
        } else if i + 1 < BUTTONS.len() {
            col += 1;
        }
    }
    out
}

/// Map an ABSOLUTE toolbar-row column (relative to the toolbar's left edge, i.e.
/// `col - toolbar.x`) to the button under it, or `None` for a gap / past the end.
/// Uses the SAME [`button_spans`] the painter uses, so paint and hit agree.
pub fn toolbar_hit(col: usize, _toolbar_w: usize) -> Option<ToolbarButton> {
    button_spans()
        .into_iter()
        .find(|&(_, s, e)| col >= s && col < e)
        .map(|(b, _, _)| b)
}

const BTN_FG: u32 = 0x00_AEC4DA;
const BTN_BG: u32 = 0x00_1A2230;
const ACTIVE_FG: u32 = 0x00_101622;
const ACTIVE_BG: u32 = 0x00_FFD24A; // highlighted active view mode
const BAR_BG: u32 = 0x00_141A26;
const DISABLED_FG: u32 = 0x00_3A4452; // dim grey for a no-op (empty undo/redo)

/// Is this button currently DISABLED (drawn dim, click is a no-op)? Undo when the
/// undo stack is empty, Redo when the redo stack is empty; others never.
pub fn is_disabled(state: &EditorState, b: ToolbarButton) -> bool {
    match b {
        ToolbarButton::Undo => state.history.undo_len() == 0,
        ToolbarButton::Redo => state.history.redo_len() == 0,
        _ => false,
    }
}

/// Is this button the currently-active view mode (highlighted)?
fn is_active(state: &EditorState, b: ToolbarButton) -> bool {
    matches!(
        (b, state.view_mode),
        (ToolbarButton::Mode2D, ViewMode::Full2D)
            | (ToolbarButton::Mode3D, ViewMode::Full3D)
            | (ToolbarButton::ModeSplit, ViewMode::Split)
    )
}

/// Paint the toolbar into the row at `(x0, y0)` of width `w`. Fills the bar
/// background, then draws each `"[LABEL]"` at its [`button_spans`] column, with
/// the active view-mode button highlighted and a `" │ "` divider at each cluster
/// boundary ([`CLUSTER_BREAK_AFTER`]). No-op if the row is zero-width.
pub fn draw_toolbar(state: &EditorState, fb: &mut Framebuffer, x0: usize, y0: usize, w: usize) {
    if w == 0 {
        return;
    }
    // Background fill across the whole bar.
    for rx in 0..w {
        fb.set(
            x0 + rx,
            y0,
            Cell {
                glyph: ' ' as u32,
                fg: BTN_FG,
                bg: BAR_BG,
                ..Cell::default()
            },
        );
    }
    for (b, s, e) in button_spans() {
        if s >= w {
            break;
        }
        let active = is_active(state, b);
        let disabled = is_disabled(state, b);
        let (fg, bg) = if active {
            (ACTIVE_FG, ACTIVE_BG)
        } else if disabled {
            (DISABLED_FG, BTN_BG)
        } else {
            (BTN_FG, BTN_BG)
        };
        let text = format!("[{}]", button_label(b));
        // Paint the button's own cell backgrounds first (draw_str only sets the
        // glyph cells it writes, which it does for the full bracketed text).
        for col in s..e.min(w) {
            fb.set(
                x0 + col,
                y0,
                Cell {
                    glyph: ' ' as u32,
                    fg,
                    bg,
                    ..Cell::default()
                },
            );
        }
        draw_str(fb, x0 + s, y0, &text, fg, bg);
        // Draw a cluster divider after each cluster's last button, if it fits.
        // The `│` sits centred in the reserved SEP_W gap (`e` .. `e + SEP_W`) and
        // is tinted the COMMAND-BAR accent so the clusters read as one organ.
        if is_cluster_break(b) {
            let sep_col = e + 1; // centre of the " │ " gap
            if sep_col < w {
                draw_str(
                    fb,
                    x0 + sep_col,
                    y0,
                    "\u{2502}",
                    crate::chrome::COMMAND_HEADER_FG,
                    BAR_BG,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_are_ordered_non_overlapping_and_cover_every_button() {
        let spans = button_spans();
        assert_eq!(spans.len(), BUTTONS.len());
        // Each maps to the matching button in order; spans are increasing and
        // non-overlapping.
        let mut prev_end = 0;
        for (i, &(b, s, e)) in spans.iter().enumerate() {
            assert_eq!(b, BUTTONS[i]);
            assert!(s >= prev_end, "span {i} starts after the previous ends");
            assert!(e > s, "span {i} is non-empty");
            prev_end = e;
        }
    }

    #[test]
    fn hit_test_uses_the_same_spans_as_paint() {
        // Every column inside a span hits that span's button; gaps hit None.
        let spans = button_spans();
        let total = spans.last().unwrap().2 + 5;
        for col in 0..total {
            let expected = spans
                .iter()
                .find(|&&(_, s, e)| col >= s && col < e)
                .map(|&(b, _, _)| b);
            assert_eq!(toolbar_hit(col, total), expected, "column {col}");
        }
    }

    #[test]
    fn each_button_has_at_least_one_hitting_column() {
        for &(b, s, e) in &button_spans() {
            for col in s..e {
                assert_eq!(toolbar_hit(col, e + 1), Some(b));
            }
        }
    }

    #[test]
    fn columns_before_first_and_in_lead_pad_miss() {
        // The leading pad column (0) is before the first button span.
        assert_eq!(toolbar_hit(0, 80), None);
        assert_eq!(button_spans()[0].1, LEAD);
    }

    #[test]
    fn undo_button_disabled_when_stack_empty() {
        let mut s = EditorState::new_empty(8, 8, 2);
        assert!(
            is_disabled(&s, ToolbarButton::Undo),
            "no undo history → disabled"
        );
        assert!(is_disabled(&s, ToolbarButton::Redo), "no redo → disabled");
        // After an edit there is undo history.
        s.push_undo();
        assert!(
            !is_disabled(&s, ToolbarButton::Undo),
            "undo enabled after a snapshot"
        );
    }

    #[test]
    fn draw_toolbar_paints_the_active_mode_highlighted() {
        let mut state = EditorState::new_empty(8, 8, 2);
        state.view_mode = ViewMode::Split;
        let mut fb = Framebuffer::new(80, 4);
        draw_toolbar(&state, &mut fb, 0, 0, 80);
        // The ModeSplit button is active → its first cell uses the active bg.
        let split_span = button_spans()
            .into_iter()
            .find(|&(b, _, _)| b == ToolbarButton::ModeSplit)
            .unwrap();
        assert_eq!(fb.get(split_span.1, 0).unwrap().bg, ACTIVE_BG);
        // The 2D button is NOT active → ordinary button bg.
        let m2d = button_spans()
            .into_iter()
            .find(|&(b, _, _)| b == ToolbarButton::Mode2D)
            .unwrap();
        assert_eq!(fb.get(m2d.1, 0).unwrap().bg, BTN_BG);
    }

    #[test]
    fn three_cluster_dividers_paint_in_the_reserved_gaps() {
        // The command bar has FOUR clusters (File · Edit · View · Help), so THREE
        // `│` dividers are painted — each in the SEP_W gap after a cluster's last
        // button, and never overlapping a button span (hit-test misses there).
        let state = EditorState::new_empty(8, 8, 2);
        let mut fb = Framebuffer::new(120, 4);
        draw_toolbar(&state, &mut fb, 0, 0, 120);
        let spans = button_spans();
        let mut dividers = 0;
        for &(b, _s, e) in &spans {
            if is_cluster_break(b) {
                let sep_col = e + 1;
                assert_eq!(
                    fb.get(sep_col, 0).map(|c| c.glyph),
                    Some('\u{2502}' as u32),
                    "a │ divider is painted after {b:?}"
                );
                // The divider column is a GAP for the hit-test (no button there).
                assert_eq!(
                    toolbar_hit(sep_col, 120),
                    None,
                    "divider column is not clickable"
                );
                dividers += 1;
            }
        }
        assert_eq!(dividers, 3, "four clusters → three dividers");
    }

    #[test]
    fn clusters_are_ordered_file_edit_view_help() {
        // The paint/hit order groups buttons by function so the bar reads as
        // File · Edit · View · Help.
        assert_eq!(
            BUTTONS,
            [
                ToolbarButton::New,
                ToolbarButton::Open,
                ToolbarButton::Save,
                ToolbarButton::SaveAs,
                ToolbarButton::Undo,
                ToolbarButton::Redo,
                ToolbarButton::Mode2D,
                ToolbarButton::Mode3D,
                ToolbarButton::ModeSplit,
                ToolbarButton::LevelSettings,
                ToolbarButton::Help,
            ]
        );
    }
}
