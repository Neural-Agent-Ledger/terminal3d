//! The persistent TOOLBOX: a clickable VERTICAL LEFT COLUMN that SETS the
//! active tool/brush, grouped by function (Geometry / Shapes / Objects / Paint /
//! Erase) like Blender's tool shelf or Tiled's tool column. Exactly one tool is
//! active (highlighted with a leading `▶` marker). Each button synthesizes the
//! SAME `EditorKey` the keyboard uses (via `edit::apply_key`), so the click
//! inherits every modal guard + undo snapshot and stays in parity with the
//! keyboard — no duplicated tool logic.
//!
//! Single source of truth: [`tool_button_rows`] is shared by the painter
//! ([`draw_toolbox`]) and the hit-test ([`toolbox_hit`]), so a click always lands
//! on the row drawn under it.
use crate::chrome::TOOLBOX_HEADER_FG as HEADER_FG;
use crate::layout::Rect;
use crate::state::{EditorState, EntityKind};
use re_core::{text::draw_str, Cell, Framebuffer};

/// A tool-palette button.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ToolButton {
    Solid,
    Floor,
    Ceil,
    Trace,
    Matrix,
    Hologram,
    Portal,
    Obj,
    Wedge,
    Stair,
    Material,
    Vox,
    Object,
    Pipe,
    Erase,
}

impl ToolButton {
    /// The full-word label drawn for this tool (never abbreviated).
    pub fn label(self) -> &'static str {
        match self {
            ToolButton::Solid => "Solid",
            ToolButton::Floor => "Floor",
            ToolButton::Ceil => "Ceiling",
            ToolButton::Trace => "Trace",
            ToolButton::Matrix => "Matrix",
            ToolButton::Hologram => "Hologram",
            ToolButton::Portal => "Portal",
            ToolButton::Obj => "Object",
            ToolButton::Wedge => "Wedge",
            ToolButton::Stair => "Stair",
            ToolButton::Material => "Material",
            ToolButton::Vox => "Voxel",
            ToolButton::Object => "Select",
            ToolButton::Pipe => "Pipe",
            ToolButton::Erase => "Erase",
        }
    }

    /// A single distinct SINGLE-WIDTH glyph for this tool, used by the COMPACT
    /// (icon-only) toolbox column at narrow terminal widths
    /// ([`crate::layout::TOOLBOX_ICON_W`]). Kept next to [`label`](Self::label) so
    /// the two stay discoverable together. Mirrors the top-down grid markers where
    /// one exists (`#` solid, `/` trace, `%` matrix, `H` hologram, `@` portal, `_`
    /// floor, `^` ceiling, `O` object, `◣` wedge, `▦` material/vox, `+` pipe, `×`
    /// erase) so the
    /// column reads the same language as the grid. Every glyph is distinct within the
    /// column.
    pub fn icon(self) -> char {
        match self {
            ToolButton::Solid => '#',
            ToolButton::Floor => '_',
            ToolButton::Ceil => '^',
            ToolButton::Trace => '/',
            ToolButton::Matrix => '%',
            ToolButton::Hologram => 'H',
            ToolButton::Portal => '@',
            ToolButton::Obj => 'O',
            ToolButton::Wedge => '\u{25E3}',    // ◣ solid SW wedge
            ToolButton::Stair => '\u{2261}',    // ≡ stacked treads
            ToolButton::Material => '\u{25A6}', // ▦ filled grid (paint)
            ToolButton::Vox => '\u{25A3}',      // ▣ box-model anchor
            ToolButton::Object => '\u{25C6}',   // ◆ select/object
            ToolButton::Pipe => '+',            // pipe/switch wiring
            ToolButton::Erase => '\u{00D7}',    // × erase
        }
    }

    /// The keyboard character this button synthesizes (routed through
    /// `edit::apply_key`). All are plain `Char(_)` keys with `shift: false`; the
    /// uppercase variants (`P`) are literal chars `edit.rs` matches.
    pub fn key_char(self) -> char {
        match self {
            ToolButton::Solid => '1',
            ToolButton::Floor => '2',
            ToolButton::Ceil => '3',
            ToolButton::Trace => '4',
            ToolButton::Matrix => '5',
            ToolButton::Hologram => '9',
            ToolButton::Portal => '0',
            ToolButton::Obj => '6',
            ToolButton::Wedge => '7',
            ToolButton::Stair => '8',
            ToolButton::Material => 'm',
            ToolButton::Vox => 'v',
            ToolButton::Object => 'o',
            ToolButton::Pipe => 'P',
            ToolButton::Erase => 'x',
        }
    }
}

/// Every tool button (used by tests to prove all are reachable in the toolbox).
#[cfg(test)]
const TOOL_BUTTONS: [ToolButton; 15] = [
    ToolButton::Solid,
    ToolButton::Floor,
    ToolButton::Ceil,
    ToolButton::Trace,
    ToolButton::Matrix,
    ToolButton::Hologram,
    ToolButton::Portal,
    ToolButton::Obj,
    ToolButton::Wedge,
    ToolButton::Stair,
    ToolButton::Material,
    ToolButton::Vox,
    ToolButton::Object,
    ToolButton::Pipe,
    ToolButton::Erase,
];

/// The toolbox GROUPS, in draw order, each a `(header, &[tools])`. This is the
/// authored grouping that gives the column its "divided by place" structure. The
/// `(header, members)` pairs feed BOTH the painter and the hit-test via
/// [`tool_button_rows`], so the layout is single-sourced.
const GROUPS: &[(&str, &[ToolButton])] = &[
    (
        "GEOMETRY",
        &[
            ToolButton::Solid,
            ToolButton::Floor,
            ToolButton::Ceil,
            ToolButton::Trace,
            ToolButton::Matrix,
            ToolButton::Hologram,
            ToolButton::Portal,
            ToolButton::Obj,
        ],
    ),
    ("SHAPES", &[ToolButton::Wedge, ToolButton::Stair]),
    ("OBJECTS", &[ToolButton::Object, ToolButton::Pipe]),
    ("PAINT", &[ToolButton::Material, ToolButton::Vox]),
    ("ERASE", &[ToolButton::Erase]),
];

/// One painted row of the vertical toolbox: either a dim GROUP HEADER or a TOOL.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ToolRow {
    /// A group header label (e.g. `GEOMETRY`), painted dim.
    Header(&'static str),
    /// A selectable tool.
    Tool(ToolButton),
}

/// The toolbox rows in paint (and hit-test) order: each group's header followed
/// by its tools. Returns `(row, row_y)` where `row_y` is the row's OFFSET from the
/// toolbox top (row 0 = `toolbox.y`). THE single source of truth shared by the
/// painter ([`draw_toolbox`]) and the hit-test ([`toolbox_hit`]).
pub fn tool_button_rows() -> Vec<(ToolRow, usize)> {
    let mut out = Vec::new();
    let mut y = 0usize;
    for (header, tools) in GROUPS {
        out.push((ToolRow::Header(header), y));
        y += 1;
        for &t in *tools {
            out.push((ToolRow::Tool(t), y));
            y += 1;
        }
    }
    out
}

const BTN_FG: u32 = 0x00_B0C4D8;
const ACTIVE_FG: u32 = 0x00_101622;
const ACTIVE_BG: u32 = 0x00_60D0A0; // highlighted active tool
const BOX_BG: u32 = 0x00_121822;
// Group-header rows use the shared TOOLBOX organ accent (`HEADER_FG`, imported at
// the top): a dim cool-neutral grey, a tool tray rather than an object hue.

/// Is `b` the currently-active tool? Mirrors the mode flags `edit.rs` sets so the
/// toolbox highlights exactly the tool the keyboard would consider active.
fn is_active(state: &EditorState, b: ToolButton) -> bool {
    use re_world::CellKind;
    // The plain-geometry-brush predicate: not in any special mode.
    let plain = !state.shape_mode
        && !state.stair_mode
        && state.place.is_none()
        && !state.material_mode
        && !state.vox_mode;
    match b {
        ToolButton::Solid => plain && state.brush == CellKind::WallSolid,
        ToolButton::Floor => plain && state.brush == CellKind::Floor,
        ToolButton::Ceil => plain && state.brush == CellKind::Ceiling,
        ToolButton::Trace => plain && state.brush == CellKind::WallTrace,
        ToolButton::Matrix => plain && state.brush == CellKind::WallMatrix,
        ToolButton::Hologram => plain && state.brush == CellKind::Hologram,
        ToolButton::Portal => plain && state.brush == CellKind::Portal,
        ToolButton::Obj => plain && state.brush == CellKind::Object,
        ToolButton::Wedge => state.shape_mode,
        ToolButton::Stair => state.stair_mode,
        ToolButton::Material => state.material_mode,
        ToolButton::Vox => state.vox_mode,
        // The object SELECTOR is active for any placement kind except Pipe.
        ToolButton::Object => {
            state.place.is_some() && state.place != Some(EntityKind::Pipe) && !state.vox_mode
        }
        ToolButton::Pipe => state.place == Some(EntityKind::Pipe),
        // Erase has no persistent active state.
        ToolButton::Erase => false,
    }
}

/// Paint the vertical TOOLBOX into `rect`, in the layout's WIDTH-DEGRADED form.
///
/// FULL mode (`rect.w >= TOOLBOX_W`): group headers dim ([`HEADER_FG`]), tools as
/// full words; the ACTIVE tool (per [`is_active`]) gets the [`ACTIVE_BG`] highlight
/// + a leading `▶` marker, inactive rows a 2-space pad so they align.
///
/// COMPACT (icon-only) mode (`rect.w < TOOLBOX_W`, i.e. the
/// [`crate::layout::TOOLBOX_ICON_W`] strip): each header becomes a dim 1-char tick
/// (`\u{00B7}`) separator and each tool its single distinct [`ToolButton::icon`]
/// glyph; the active tool keeps the highlight (+ `▶` only if the strip is wide
/// enough to also fit the icon, else just the highlight on the icon cell).
///
/// BOTH modes paint the same rows from [`tool_button_rows`] (SSOT with the
/// hit-test). Rows past the column height are clipped. No-op on an empty rect.
/// Single-width glyphs only.
pub fn draw_toolbox(state: &EditorState, fb: &mut Framebuffer, rect: Rect) {
    if rect.is_empty() {
        return;
    }
    let compact = rect.w < crate::layout::TOOLBOX_W;
    // Fill the whole column background first.
    for ry in 0..rect.h {
        for rx in 0..rect.w {
            fb.set(
                rect.x + rx,
                rect.y + ry,
                Cell {
                    glyph: ' ' as u32,
                    fg: BTN_FG,
                    bg: BOX_BG,
                    ..Cell::default()
                },
            );
        }
    }
    for (row, ry) in tool_button_rows() {
        if ry >= rect.h {
            break;
        }
        let y = rect.y + ry;
        match row {
            ToolRow::Header(h) => {
                if compact {
                    // A dim 1-char tick marks the group break in the icon column.
                    draw_str(fb, rect.x, y, "\u{00B7}", HEADER_FG, BOX_BG); // ·
                } else {
                    // Group header, dim. Clipped to the column width.
                    let text: String = h.chars().take(rect.w).collect();
                    draw_str(fb, rect.x, y, &text, HEADER_FG, BOX_BG);
                }
            }
            ToolRow::Tool(b) => {
                let active = is_active(state, b);
                let (fg, bg) = if active {
                    (ACTIVE_FG, ACTIVE_BG)
                } else {
                    (BTN_FG, BOX_BG)
                };
                // Repaint the row background (so the active highlight spans the column).
                for rx in 0..rect.w {
                    fb.set(
                        rect.x + rx,
                        y,
                        Cell {
                            glyph: ' ' as u32,
                            fg,
                            bg,
                            ..Cell::default()
                        },
                    );
                }
                let text = if compact {
                    // Icon column: `▶◆` (marker + icon) when 2+ cells fit, else the
                    // bare icon. The active highlight already distinguishes it.
                    if active && rect.w >= 2 {
                        format!("\u{25B6}{}", b.icon())
                    } else {
                        b.icon().to_string()
                    }
                } else {
                    // `▶ Label` for the active tool, `  Label` (2-space pad) otherwise,
                    // so active/inactive rows align. Fits in TOOLBOX_W (marker/pad + 8).
                    let marker = if active { '\u{25B6}' } else { ' ' };
                    format!("{marker} {}", b.label())
                };
                let shown: String = text.chars().take(rect.w).collect();
                draw_str(fb, rect.x, y, &shown, fg, bg);
            }
        }
    }
}

/// Map an absolute `(col, row)` click to the [`ToolButton`] under it in the
/// toolbox `layout`, or `None` for a header / gap / outside the column. Uses the
/// SAME [`tool_button_rows`] geometry as the painter (SSOT).
pub fn toolbox_hit(layout: &crate::layout::Layout, col: usize, row: usize) -> Option<ToolButton> {
    let rect = layout.toolbox;
    if rect.is_empty() {
        return None;
    }
    if col < rect.x || col >= rect.x + rect.w || row < rect.y || row >= rect.y + rect.h {
        return None;
    }
    let ry = row - rect.y;
    tool_button_rows().into_iter().find_map(|(r, y)| match r {
        ToolRow::Tool(b) if y == ry => Some(b),
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout;
    use crate::state::ViewMode;

    #[test]
    fn rows_are_grouped_in_order_with_headers() {
        let rows = tool_button_rows();
        // First row is the GEOMETRY header.
        assert_eq!(rows[0].0, ToolRow::Header("GEOMETRY"));
        // Headers appear in the authored order.
        let headers: Vec<&str> = rows
            .iter()
            .filter_map(|(r, _)| match r {
                ToolRow::Header(h) => Some(*h),
                _ => None,
            })
            .collect();
        assert_eq!(headers, ["GEOMETRY", "SHAPES", "OBJECTS", "PAINT", "ERASE"]);
        // Row offsets are contiguous and increasing.
        for (i, (_, y)) in rows.iter().enumerate() {
            assert_eq!(*y, i, "row {i} offset is contiguous");
        }
    }

    #[test]
    fn every_tool_button_is_reachable_in_the_toolbox() {
        let tools: Vec<ToolButton> = tool_button_rows()
            .iter()
            .filter_map(|(r, _)| match r {
                ToolRow::Tool(b) => Some(*b),
                _ => None,
            })
            .collect();
        for &b in &TOOL_BUTTONS {
            assert!(tools.contains(&b), "{b:?} reachable in the toolbox");
        }
        assert_eq!(
            tools.len(),
            TOOL_BUTTONS.len(),
            "no tool listed twice / missing"
        );
    }

    #[test]
    fn hit_test_matches_paint_rows() {
        // The hit-test resolves the SAME tool the painter draws on each row, and
        // returns None on header rows / outside the column.
        let l = layout::compute(ViewMode::Split, 120, 40, false);
        let rect = l.toolbox;
        assert!(!rect.is_empty());
        for (r, ry) in tool_button_rows() {
            if ry >= rect.h {
                break;
            }
            let row = rect.y + ry;
            let hit = toolbox_hit(&l, rect.x, row);
            match r {
                ToolRow::Tool(b) => assert_eq!(hit, Some(b), "tool row {ry}"),
                ToolRow::Header(_) => assert_eq!(hit, None, "header row {ry} is not a tool"),
            }
        }
        // Outside the column → None.
        assert_eq!(toolbox_hit(&l, rect.x + rect.w + 1, rect.y), None);
    }

    #[test]
    fn hit_test_none_when_toolbox_empty() {
        // Below the floor the toolbox is empty and never resolves a tool.
        let l = layout::compute(ViewMode::Split, 20, 6, false);
        assert!(l.toolbox.is_empty());
        assert_eq!(toolbox_hit(&l, 0, 0), None);
    }

    #[test]
    fn active_highlight_tracks_the_brush() {
        use re_world::CellKind;
        let mut s = EditorState::new_empty(8, 8, 2);
        s.brush = CellKind::WallSolid;
        assert!(is_active(&s, ToolButton::Solid));
        assert!(!is_active(&s, ToolButton::Floor));
        s.shape_mode = true;
        assert!(is_active(&s, ToolButton::Wedge));
        assert!(!is_active(&s, ToolButton::Solid), "shape mode is not plain");
    }

    #[test]
    fn longest_label_fits_the_toolbox_width() {
        // `▶ Material` / `▶ Ceiling` must fit within TOOLBOX_W (marker + space + word).
        let longest = TOOL_BUTTONS
            .iter()
            .map(|b| b.label().chars().count())
            .max()
            .unwrap();
        assert!(
            longest + 2 <= layout::TOOLBOX_W,
            "longest tool row fits the column"
        );
    }

    #[test]
    fn every_tool_icon_is_single_width_and_distinct() {
        // Each tool's compact icon is exactly one char and no two tools share one,
        // so the icon column is unambiguous.
        let mut seen = std::collections::HashSet::new();
        for &b in &TOOL_BUTTONS {
            let g = b.icon();
            assert_eq!(g.to_string().chars().count(), 1, "{b:?} icon is one char");
            assert!(seen.insert(g), "{b:?} icon {g:?} is distinct");
        }
    }

    #[test]
    fn hit_test_matches_paint_rows_in_icon_mode() {
        // At a narrow-but-above-floor width the toolbox is the 3-wide icon column;
        // the SAME hit-test (driven by `tool_button_rows`) still resolves every tool
        // row, proving SSOT for paint + hit across both modes. Find such a width.
        let mut found = false;
        for w in layout::MIN_COLS..=200 {
            let l = layout::compute(ViewMode::Split, w, 30, false);
            if l.toolbox.w != layout::TOOLBOX_ICON_W {
                continue;
            }
            found = true;
            let rect = l.toolbox;
            for (r, ry) in tool_button_rows() {
                if ry >= rect.h {
                    break;
                }
                let row = rect.y + ry;
                let hit = toolbox_hit(&l, rect.x, row);
                match r {
                    ToolRow::Tool(b) => assert_eq!(hit, Some(b), "icon-mode tool row {ry} (w={w})"),
                    ToolRow::Header(_) => {
                        assert_eq!(hit, None, "icon-mode header row {ry} (w={w})")
                    }
                }
            }
            break;
        }
        assert!(
            found,
            "an icon-mode toolbox width exists to exercise the hit-test"
        );
    }
}
