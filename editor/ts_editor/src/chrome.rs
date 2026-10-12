//! Editor chrome: top-down z-slice grid (integer zoom + pan + rulers + chunk
//! lines + height overlay), side panel (zoom/z-slider/brush/help), and the
//! elevation cross-section side-window. All painted as Framebuffer cells.
//!
//! Pure viewport math (zoom clamp, pan auto-follow, ruler ticks, z-slider,
//! cross-section extraction) lives in `crate::view` and is unit-tested there;
//! this module is the (thin) painting layer on top.
//!
//! This module does not dispatch keys itself — `edit.rs`/`keymap.rs` own key
//! handling and `mouse.rs` owns clicks; chrome exposes the geometry (box
//! rects, row/organ layout) both consume so a click lands on exactly what is
//! painted. It writes editor UI cells directly into a `re_core::Framebuffer`;
//! the 3D preview it frames is rendered by `re_render`/`re_present` and
//! composited into the same framebuffer by the caller, not by this module.
use crate::entity::io_role_label;
use crate::rowvalue::{RowValue, RowZone};
use crate::state::{facing_label, shape_label, EditorState, EntityKind};
use crate::tree::{self, TreeRow};
use crate::view;
use re_core::{text::draw_str, Cell, DiagXY, Framebuffer, RampDir, Region, Shape, Trigger};
use re_world::CellKind;

// ─── MODAL GEOMETRY (single source of truth for painter + hit-test) ──────────
//
// Each centered overlay (Help / Open / Wizard / Palette) computes its box rect
// from the framebuffer size in ONE place so the mouse hit-test in `mouse.rs`
// lands on exactly what `draw_*` paints. The geometry fns return `None` when the
// fb is too small to draw the box (matching the painters' early-out).

/// Rows of the HELP popup that are NOT body content: the top border (which
/// carries the `[X]`), the title band, the filter band, the position footer, and
/// the bottom border. The box height is `body rows + this`, and the body height
/// is `box height - this` — one constant, both directions, so a band can never
/// be given a row the body also thinks it owns.
const HELP_CHROME_ROWS: usize = 5;

/// The HELP popup laid out: the box rect plus the ABSOLUTE row of every band
/// inside it.
///
/// Two independent row calculations for one box is how a hint line ends up
/// landing on top of a clickable swatch row: the painter and the height
/// disagree because each does its own arithmetic. The rule here is
/// ONE running cursor: `title_y`, `filter_y`, `body_y`, `body_h` and `footer_y`
/// are all handed out by the walk below, and the box height is what the walk
/// consumed. [`help_box_geom`] (the `[X]` hit-test) reads the same struct.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HelpLayout {
    /// Box origin column (left border).
    pub x0: usize,
    /// Box origin row (top border).
    pub y0: usize,
    /// Box width including both borders.
    pub pw: usize,
    /// Box height including both borders.
    pub ph: usize,
    /// Absolute row of the title band.
    pub title_y: usize,
    /// Absolute row of the filter / match-count band.
    pub filter_y: usize,
    /// Absolute row of the FIRST body line.
    pub body_y: usize,
    /// How many body lines fit (always >= 1 when the layout exists).
    pub body_h: usize,
    /// Absolute row of the position footer, one row above the bottom border.
    pub footer_y: usize,
    /// Absolute text column shared by every band (inside the left border).
    pub text_x: usize,
    /// Usable text width for a band.
    pub text_w: usize,
    /// How many lines the (possibly filtered) sheet has right now.
    pub total: usize,
    /// The CLAMPED index of the first visible line.
    pub scroll: usize,
}

/// Clamp a help-sheet scroll offset so the LAST PAGE IS THE LAST PAGE: the final
/// body row shows the final line and nothing scrolls past it into blank rows.
/// `0` whenever the whole sheet fits. Pure; shared by the painter, the key
/// handlers (`edit::help_scroll_by`) and the wheel, so all three agree on where
/// the end is.
#[must_use]
pub fn clamp_help_scroll(scroll: usize, total: usize, visible: usize) -> usize {
    if visible == 0 || total <= visible {
        return 0;
    }
    scroll.min(total - visible)
}

/// Lay out the HELP popup for the current filter + scroll. `None` when the
/// framebuffer cannot host the box (matching the painter's early-out).
///
/// WIDTH comes from the UNFILTERED sheet on purpose: the box must not resize
/// horizontally under the pointer on every keystroke while a filter is being
/// typed. HEIGHT follows the filtered content (capped by the framebuffer), so a
/// nine-match filter is a nine-row box rather than a mostly-empty screen.
pub fn help_layout(state: &EditorState, w: usize, h: usize) -> Option<HelpLayout> {
    let lines = crate::keymap::help_lines_filtered(&state.help_filter);
    let full = help_lines();
    let content_w = full.iter().map(|l| l.chars().count()).max().unwrap_or(0);
    // The FOOTER must fit as well, or the one band that teaches the scroll keys
    // is the one band that gets clipped. Its floor is the WIDEST form it can
    // ever take (the last page of the unfiltered sheet, and the longer of the
    // two key hints) so it is a constant: neither scrolling nor typing a filter
    // may resize the box.
    let band_w = help_footer_text(full.len(), full.len(), full.len(), true)
        .chars()
        .count()
        .max(
            help_footer_text(full.len(), full.len(), full.len(), false)
                .chars()
                .count(),
        );
    let pw = (content_w.max(band_w) + 4).min(w);

    // ── the ONE running cursor ──
    let mut used = 1; // top border (+ the [X])
    let title_y = used;
    used += 1;
    let filter_y = used;
    used += 1;
    let body_y = used;
    // Body rows wanted, then the footer + the bottom border.
    let ph = (body_y + lines.len().max(1) + 2).min(h);
    if pw < 12 || ph < HELP_CHROME_ROWS + 1 {
        return None;
    }
    let footer_y = ph - 2;
    let body_h = footer_y - body_y;

    let (x0, y0) = ((w - pw) / 2, (h - ph) / 2);
    Some(HelpLayout {
        x0,
        y0,
        pw,
        ph,
        title_y: y0 + title_y,
        filter_y: y0 + filter_y,
        body_y: y0 + body_y,
        body_h,
        footer_y: y0 + footer_y,
        text_x: x0 + 2,
        text_w: pw.saturating_sub(4),
        total: lines.len(),
        scroll: clamp_help_scroll(state.help_scroll, lines.len(), body_h),
    })
}

/// The geometry of the HELP popup box: `(x0, y0, pw, ph)`. The `[X]` hit-test in
/// `mouse.rs` reads THIS, and it is a projection of [`help_layout`] — one
/// geometry function, no hand-copied coordinates.
pub fn help_box_geom(
    state: &EditorState,
    w: usize,
    h: usize,
) -> Option<(usize, usize, usize, usize)> {
    help_layout(state, w, h).map(|l| (l.x0, l.y0, l.pw, l.ph))
}

/// The help popup's FILTER band: the live query and the match count.
fn help_filter_text(state: &EditorState) -> String {
    let total = crate::keymap::help_match_count("");
    if state.help_filter.is_empty() {
        format!("filter: (type to filter)   {total} rows")
    } else {
        format!(
            "filter: {}_   {} of {total} rows",
            state.help_filter,
            crate::keymap::help_match_count(&state.help_filter)
        )
    }
}

/// The POSITION half of the help popup's footer: which slice of the sheet is on
/// screen. Pure, and the ONLY band that is allowed to widen the box — the key
/// hints beside it may truncate on a narrow terminal, but "where am I in 178
/// lines" must always be legible or the scrollbar has no readout at all.
fn help_position_text(scroll: usize, body_h: usize, total: usize) -> String {
    let first = if total == 0 { 0 } else { scroll + 1 };
    let last = (scroll + body_h).min(total);
    format!("-- {first}-{last} of {total} lines --")
}

/// The full help popup FOOTER: the position readout plus the keys that move it.
/// Sized through [`help_layout`], which reserves it its own row AND sizes the
/// box so the widest form of this string fits, so it can neither overpaint a
/// content line nor lose its own tail to the border.
fn help_footer_text(scroll: usize, body_h: usize, total: usize, filtering: bool) -> String {
    let keys = if filtering {
        "Up/Dn PgUp/PgDn scroll  Backspace edits  Esc clears filter"
    } else {
        "Up/Dn PgUp/PgDn Home/End  type to filter  ?/Esc close"
    };
    format!("{}  {keys}", help_position_text(scroll, body_h, total))
}

/// The geometry of the OPEN picker box: `(x0, y0, pw, ph)`. Mirrors `draw_open`.
pub fn open_box_geom(
    state: &EditorState,
    w: usize,
    h: usize,
) -> Option<(usize, usize, usize, usize)> {
    let o = state.open.as_ref()?;
    let title = "OPEN LEVEL";
    let footer = "Up/Dn  Enter load  Esc cancel  [Load][Cancel]";
    let longest = o
        .files
        .iter()
        .map(|f| f.len() + 2)
        .chain([title.len(), footer.len()])
        .max()
        .unwrap_or(footer.len());
    let pw = (longest + 4).min(w);
    let rows = o.files.len().max(1);
    let ph = (rows + 4).min(h);
    if pw < 12 || ph < 6 {
        return None;
    }
    Some(((w - pw) / 2, (h - ph) / 2, pw, ph))
}

/// Map an absolute `(col, row)` click to the OPEN picker FILE INDEX under it, or
/// `None`. Uses the exact same body geometry + `window_start` windowing as
/// `draw_open`, so a click selects the file drawn under the pointer.
pub fn open_file_at(
    state: &EditorState,
    w: usize,
    h: usize,
    col: usize,
    row: usize,
) -> Option<usize> {
    let o = state.open.as_ref()?;
    if o.files.is_empty() {
        return None;
    }
    let (x0, y0, pw, ph) = open_box_geom(state, w, h)?;
    let body_y0 = y0 + 2;
    let body_h = ph.saturating_sub(4);
    // Inside the box horizontally (between the borders)?
    if col <= x0 || col >= x0 + pw - 1 {
        return None;
    }
    if row < body_y0 || row >= body_y0 + body_h {
        return None;
    }
    let line = row - body_y0;
    let start = window_start(o.cursor.min(o.files.len() - 1), o.files.len(), body_h);
    let fi = start + line;
    if fi < o.files.len() {
        Some(fi)
    } else {
        None
    }
}

/// The `[Load]` and `[Cancel]` button spans on the OPEN box footer row:
/// `(load_span, cancel_span, row)` where each span is absolute `(start, end)`
/// columns. Shared by the painter + hit-test. `None` if too small.
#[allow(clippy::type_complexity)]
pub fn open_footer_buttons(
    state: &EditorState,
    w: usize,
    h: usize,
) -> Option<((usize, usize), (usize, usize), usize)> {
    let (x0, y0, _pw, ph) = open_box_geom(state, w, h)?;
    let row = y0 + ph - 2;
    let load_s = x0 + 2;
    let load_e = load_s + "[Load]".len();
    let cancel_s = load_e + 1;
    let cancel_e = cancel_s + "[Cancel]".len();
    Some(((load_s, load_e), (cancel_s, cancel_e), row))
}

/// The geometry of the WIZARD box: `(x0, y0, pw, ph)`. Mirrors `draw_wizard`.
pub fn wizard_box_geom(w: usize, h: usize) -> Option<(usize, usize, usize, usize)> {
    let pw = 34usize.min(w);
    let ph = 11usize.min(h);
    if pw < 10 || ph < 8 {
        return None;
    }
    Some(((w - pw) / 2, (h - ph) / 2, pw, ph))
}

/// The geometry of the LEVEL SETTINGS box: `(x0, y0, pw, ph)`. A generous
/// centered modal. Mirrors [`wizard_box_geom`]; returns `None` if the fb is too
/// small to host the form.
pub fn settings_box_geom(w: usize, h: usize) -> Option<(usize, usize, usize, usize)> {
    let pw = 52usize.min(w);
    // Tall enough to host Meta/Room/Effects/Fog/Sky plus the list sections; clamps
    // to the framebuffer height so a short terminal still gets a (scroll-clipped)
    // box rather than nothing. The row painter clips overflow, never panics.
    let ph = 34usize.min(h);
    if pw < 20 || ph < 12 {
        return None;
    }
    Some(((w - pw) / 2, (h - ph) / 2, pw, ph))
}

/// Compute the vertical SCROLL OFFSET for the LEVEL SETTINGS row list so the
/// focused row is always inside the visible window. Pure (no terminal) so it is
/// unit-testable: given the `focused` row index, the `total` number of rows, and
/// the number of `visible` content rows, returns the index of the FIRST row to
/// draw. Standard "keep focus visible" scrolling — when focus is below the window
/// it scrolls down just enough; when above, up just enough; otherwise the scroll
/// is clamped so the window never runs past the end of the list (no blank tail
/// while rows remain off-screen above). Returns `0` when everything fits or
/// `visible == 0`.
#[must_use]
pub fn settings_scroll(focused: usize, total: usize, visible: usize) -> usize {
    if visible == 0 || total <= visible {
        return 0;
    }
    let max_scroll = total - visible;
    // Keep `focused` within `[scroll, scroll + visible)`.
    let scroll = if focused < visible {
        0
    } else {
        focused + 1 - visible
    };
    scroll.min(max_scroll)
}

/// A clickable control inside the LEVEL SETTINGS box footer, returned by the hit-test.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SettingsHit {
    /// `[Apply]` → confirm + apply the form (same as Enter).
    Apply,
    /// `[Resize]` → apply the staged size (same as `R`).
    Resize,
    /// `[Cancel]` → close the form discarding edits (same as Esc).
    Cancel,
}

/// The LEVEL SETTINGS footer button spans: `((apply_s, apply_e),
/// (resize_s, resize_e), (cancel_s, cancel_e), row)` as absolute columns + the
/// absolute footer row. THE single source of truth shared by [`draw_settings`]
/// (painter) and [`settings_hit`]. `None` if the box does not fit.
#[allow(clippy::type_complexity)]
pub fn settings_footer_spans(
    w: usize,
    h: usize,
) -> Option<((usize, usize), (usize, usize), (usize, usize), usize)> {
    let (x0, y0, _pw, ph) = settings_box_geom(w, h)?;
    let row = y0 + ph - 2;
    let apply_s = x0 + 2;
    let apply_e = apply_s + "[Apply]".len();
    let resize_s = apply_e + 1;
    let resize_e = resize_s + "[Resize]".len();
    let cancel_s = resize_e + 1;
    let cancel_e = cancel_s + "[Cancel]".len();
    Some((
        (apply_s, apply_e),
        (resize_s, resize_e),
        (cancel_s, cancel_e),
        row,
    ))
}

/// Hit-test a click inside the LEVEL SETTINGS box footer. Returns the
/// [`SettingsHit`] under `(col, row)` or `None`. Mirrors [`saveas_hit`].
pub fn settings_hit(w: usize, h: usize, col: usize, row: usize) -> Option<SettingsHit> {
    let ((aps, ape), (rss, rse), (cas, cae), frow) = settings_footer_spans(w, h)?;
    if row != frow {
        return None;
    }
    if col >= aps && col < ape {
        return Some(SettingsHit::Apply);
    }
    if col >= rss && col < rse {
        return Some(SettingsHit::Resize);
    }
    if col >= cas && col < cae {
        return Some(SettingsHit::Cancel);
    }
    None
}

/// A clickable control inside the NEW-LEVEL wizard, returned by the hit-test.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WizardHit {
    /// Click a field row → focus that field.
    Focus(crate::wizard::WizardField),
    /// `[-]` on a row → adjust focused field down.
    Minus,
    /// `[+]` on a row → adjust focused field up.
    Plus,
    /// `[Clr]` → clear the name buffer.
    Clear,
    /// `[Make]` → create the level (same as Enter).
    Make,
    /// `[Cancel]` → close the wizard.
    Cancel,
}

/// The wizard field rows, in draw order, with their absolute `(label_x, row)` and
/// the `[-]` `[+]` button column spans. Single source of truth for painter +
/// hit-test. Rows are at `y0 + 2 + i`; the `[-]`/`[+]` sit at the right edge.
#[allow(clippy::type_complexity)]
pub fn wizard_rows(
    w: usize,
    h: usize,
) -> Option<(usize, usize, usize, usize, [crate::wizard::WizardField; 5])> {
    use crate::wizard::WizardField::*;
    let (x0, y0, pw, _ph) = wizard_box_geom(w, h)?;
    // [-] at pw-7..pw-4, [+] at pw-4..pw-1 (3 cells each, inside the right border).
    let minus_x = x0 + pw.saturating_sub(7);
    let plus_x = x0 + pw.saturating_sub(4);
    Some((x0, y0, minus_x, plus_x, [SizeX, SizeY, SizeZ, Name, Shell]))
}

/// Hit-test a click inside the wizard box. Returns the [`WizardHit`] under
/// `(col, row)` or `None`.
pub fn wizard_hit(w: usize, h: usize, col: usize, row: usize) -> Option<WizardHit> {
    let (x0, y0, minus_x, plus_x, fields) = wizard_rows(w, h)?;
    let (_x0b, _y0b, pw, ph) = wizard_box_geom(w, h)?;
    // Field rows: y0+2 .. y0+2+5.
    for (i, f) in fields.iter().enumerate() {
        let ry = y0 + 2 + i;
        if row == ry {
            // [-] / [+] take precedence over the row-focus on those columns.
            if col >= minus_x && col < minus_x + 3 {
                return Some(WizardHit::Minus);
            }
            if col >= plus_x && col < plus_x + 3 {
                return Some(WizardHit::Plus);
            }
            if col > x0 && col < x0 + pw - 1 {
                return Some(WizardHit::Focus(*f));
            }
        }
    }
    // Footer button row (y0 + ph - 2): [Clr] [Make] [Cancel].
    let frow = y0 + ph - 2;
    if row == frow {
        let clr_s = x0 + 2;
        let clr_e = clr_s + "[Clr]".len();
        let make_s = clr_e + 1;
        let make_e = make_s + "[Make]".len();
        let cancel_s = make_e + 1;
        let cancel_e = cancel_s + "[Cancel]".len();
        if col >= clr_s && col < clr_e {
            return Some(WizardHit::Clear);
        }
        if col >= make_s && col < make_e {
            return Some(WizardHit::Make);
        }
        if col >= cancel_s && col < cancel_e {
            return Some(WizardHit::Cancel);
        }
    }
    None
}

/// The geometry of the SAVE-AS box: `(x0, y0, pw, ph)`. Mirrors `draw_saveas`.
/// A small centered box wide enough for a filename + the footer buttons.
pub fn saveas_box_geom(w: usize, h: usize) -> Option<(usize, usize, usize, usize)> {
    let pw = 40usize.min(w);
    let ph = 7usize.min(h);
    if pw < 12 || ph < 6 {
        return None;
    }
    Some(((w - pw) / 2, (h - ph) / 2, pw, ph))
}

/// A clickable control inside the SAVE-AS prompt, returned by the hit-test.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SaveAsHit {
    /// `[Clr]` → clear the filename buffer.
    Clear,
    /// `[Save]` → confirm + write (same as Enter).
    Save,
    /// `[Cancel]` → close the prompt.
    Cancel,
}

/// The SAVE-AS footer button spans: `((clr_s, clr_e), (save_s, save_e),
/// (cancel_s, cancel_e), row)` as absolute columns + the absolute footer row.
/// THE single source of truth shared by the painter and the hit-test. `None` if
/// the box is too small.
#[allow(clippy::type_complexity)]
pub fn saveas_footer_spans(
    w: usize,
    h: usize,
) -> Option<((usize, usize), (usize, usize), (usize, usize), usize)> {
    let (x0, y0, _pw, ph) = saveas_box_geom(w, h)?;
    let row = y0 + ph - 2;
    let clr_s = x0 + 2;
    let clr_e = clr_s + "[Clr]".len();
    let save_s = clr_e + 1;
    let save_e = save_s + "[Save]".len();
    let cancel_s = save_e + 1;
    let cancel_e = cancel_s + "[Cancel]".len();
    Some(((clr_s, clr_e), (save_s, save_e), (cancel_s, cancel_e), row))
}

/// Hit-test a click inside the SAVE-AS box footer. Returns the [`SaveAsHit`]
/// under `(col, row)` or `None`. The `[X]` close on the top border is handled
/// separately in `mouse.rs` via [`modal_close_button`] (mirroring the wizard).
pub fn saveas_hit(w: usize, h: usize, col: usize, row: usize) -> Option<SaveAsHit> {
    let ((cls, cle), (svs, sve), (cas, cae), frow) = saveas_footer_spans(w, h)?;
    if row != frow {
        return None;
    }
    if col >= cls && col < cle {
        return Some(SaveAsHit::Clear);
    }
    if col >= svs && col < sve {
        return Some(SaveAsHit::Save);
    }
    if col >= cas && col < cae {
        return Some(SaveAsHit::Cancel);
    }
    None
}

/// The geometry of the PALETTE box: `(x0, y0, pw, ph)`. Mirrors `draw_palette`.
pub fn palette_box_geom(w: usize, h: usize) -> Option<(usize, usize, usize, usize)> {
    let items = tree::PALETTE_ITEMS;
    let pw = 24usize.min(w);
    let ph = (items.len() + 4).min(h);
    if pw < 10 || ph < 6 {
        return None;
    }
    Some(((w - pw) / 2, (h - ph) / 2, pw, ph))
}

// ─── PANEL ACTION BUTTONS (P2 inspector / P3 view / P4 fill / P6 tree) ───────
//
// A grid of small clickable `[Label]` controls drawn near the bottom of the side
// panel. Single source of truth: `panel_buttons` returns each button's absolute
// `(start, end, row)` span, used by both `draw_panel_buttons` (painter) and the
// `mouse.rs` hit-test. Each button maps to a key (synthesized via `apply_key`) or
// a direct backend call — see `mouse::apply_panel_button`.

/// A clickable control in the side-panel action area.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PanelButton {
    // P2 — inspector (focused-field editing); only shown when a field context is on.
    Field,  // `;` focus next field
    Minus,  // `-` adjust down
    Plus,   // `+` adjust up
    Cycle,  // `c` cycle enum
    Delete, // `x` delete
    // Inspector extras — context-specific.
    NewMat,     // `e` create a new material (material mode only)
    PipeSwitch, // `g` toggle the selected pipe's switch cell (pipe placement only)
    TagNode,    // `n` tag the cursor cell as a node (node placement only)
    // P3 — view.
    CreateObj, // `i` toggle the create-object palette (always available)
    ZoomIn,
    ZoomOut,
    FloorUp,
    FloorDown,
    Noclip,
    Tree,
    GoToCursor,
    // P4 — fill.
    SetA,
    Rect,
    Line,
    Bucket,
    // P6 — tree actions (operate on the current tree cursor; only when tree open).
    TreeDup,
    TreeDel,
    TreeParent,
    TreeUnparent,
}

impl PanelButton {
    // Full, readable labels — never abbreviated (we are not saving pixels). The
    // label is THE single source of truth for the clickable span: `panel_buttons`
    // sizes each button from `label().len()`, so widening here automatically
    // widens both the drawn glyphs and the hit-test span.
    fn label(self) -> &'static str {
        match self {
            PanelButton::Field => "Next field",
            PanelButton::Minus => "Decrease",
            PanelButton::Plus => "Increase",
            PanelButton::Cycle => "Cycle",
            PanelButton::Delete => "Delete",
            PanelButton::NewMat => "New material",
            PanelButton::PipeSwitch => "Pipe switch",
            PanelButton::TagNode => "Tag node",
            PanelButton::CreateObj => "Create object",
            PanelButton::ZoomIn => "Zoom +",
            PanelButton::ZoomOut => "Zoom -",
            PanelButton::FloorUp => "Floor up",
            PanelButton::FloorDown => "Floor down",
            PanelButton::Noclip => "Noclip",
            PanelButton::Tree => "Tree",
            PanelButton::GoToCursor => "Go to cursor",
            PanelButton::SetA => "Set corner A",
            PanelButton::Rect => "Rect fill",
            PanelButton::Line => "Line fill",
            PanelButton::Bucket => "Bucket fill",
            PanelButton::TreeDup => "Duplicate",
            PanelButton::TreeDel => "Delete",
            PanelButton::TreeParent => "Reparent",
            PanelButton::TreeUnparent => "Unparent",
        }
    }
}

/// True when an INSPECTOR focused-field context is active (so the P2 buttons are
/// meaningful): object placement, material, or vox mode.
fn inspector_context(state: &EditorState) -> bool {
    state.place.is_some() || state.material_mode || state.vox_mode
}

/// True when the side panel should favour the INSPECTOR in the SINGLE-ORGAN
/// fallback (the panel was too short to host both): something is selected/active.
/// In the no-split fallback the OUTLINER and INSPECTOR are mutually exclusive
/// (like the old elevation-vs-tree carve), so this decides which one to show.
fn inspector_active(state: &EditorState) -> bool {
    inspector_context(state) || state.stair_mode || state.shape_mode
}

/// The OUTLINER (object list) sub-rect of the side panel, applying the fallback
/// rule: when the panel splits, the top organ; when it can't, the WHOLE panel
/// UNLESS the inspector is the active organ (then empty). THE single source of
/// truth shared by [`draw_tree`] (via `draw_system`) and [`tree_row_at`].
pub fn outliner_subrect(state: &EditorState, panel: crate::layout::Rect) -> crate::layout::Rect {
    let split = crate::layout::panel_subrects(panel);
    if split.split {
        return split.outliner;
    }
    // No-split fallback: show the outliner only when the inspector is NOT the
    // active organ (mutually exclusive).
    if inspector_active(state) {
        crate::layout::Rect {
            x: panel.x,
            y: panel.y,
            w: panel.w,
            h: 0,
        }
    } else {
        split.outliner // == whole panel
    }
}

/// The INSPECTOR (properties) sub-rect of the side panel, applying the fallback
/// rule: when the panel splits, the bottom organ; when it can't, the WHOLE panel
/// only when the inspector is the active organ, else empty. THE single source of
/// truth shared by [`draw_panel`]/[`draw_panel_buttons`] and [`panel_button_hit`]
/// + the z-slider hit-test.
pub fn inspector_subrect(state: &EditorState, panel: crate::layout::Rect) -> crate::layout::Rect {
    let split = crate::layout::panel_subrects(panel);
    if split.split {
        return split.inspector;
    }
    // No-split fallback: show the inspector only when it is the active organ.
    if inspector_active(state) {
        panel
    } else {
        crate::layout::Rect {
            x: panel.x,
            y: panel.y,
            w: panel.w,
            h: 0,
        }
    }
}

/// EFF-AWARE outliner sub-rect: [`outliner_subrect`] extended with
/// the effective panel states — a railed/hidden OUTLINER yields an empty rect
/// (its icon sits in the rail instead); a railed/hidden INSPECTOR hands the
/// outliner the whole panel. With both organs expanded this is exactly the
/// legacy split, so callers/tests that delegate with the default all-Expanded
/// eff are untouched.
pub fn outliner_subrect_eff(
    state: &EditorState,
    panel: crate::layout::Rect,
    eff: crate::layout::EffPanels,
) -> crate::layout::Rect {
    use crate::layout::PanelState;
    if eff.outliner != PanelState::Expanded {
        return crate::layout::Rect {
            x: panel.x,
            y: panel.y,
            w: panel.w,
            h: 0,
        };
    }
    if eff.inspector != PanelState::Expanded {
        return panel; // the outliner owns the whole column
    }
    outliner_subrect(state, panel)
}

/// EFF-AWARE inspector sub-rect — the mirror of [`outliner_subrect_eff`].
pub fn inspector_subrect_eff(
    state: &EditorState,
    panel: crate::layout::Rect,
    eff: crate::layout::EffPanels,
) -> crate::layout::Rect {
    use crate::layout::PanelState;
    if eff.inspector != PanelState::Expanded {
        return crate::layout::Rect {
            x: panel.x,
            y: panel.y,
            w: panel.w,
            h: 0,
        };
    }
    if eff.outliner != PanelState::Expanded {
        return panel; // the inspector owns the whole column
    }
    inspector_subrect(state, panel)
}

// ─── ICON RAIL ────────────────────────────────────────────────────────────────

/// The rail's icon rows: `(panel, icon, absolute row)` for every panel whose
/// effective state is [`crate::layout::PanelState::Rail`], stacked from the
/// rail top with one blank row between icons. THE single source the painter
/// ([`draw_rail`]) and the hit-test ([`rail_hit`]) share.
pub fn rail_icon_rows(
    rail: crate::layout::Rect,
    eff: crate::layout::EffPanels,
) -> Vec<(crate::layout::PanelId, char, usize)> {
    use crate::layout::{PanelId, PanelState};
    if rail.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut row = rail.y + 1;
    for (id, icon, st) in [
        (PanelId::Toolbox, 'T', eff.toolbox),
        (PanelId::Outliner, 'O', eff.outliner),
        (PanelId::Inspector, 'I', eff.inspector),
    ] {
        if st == PanelState::Rail && row < rail.y + rail.h {
            out.push((id, icon, row));
            row += 2;
        }
    }
    out
}

/// Paint the 1-col icon rail: one letter icon per collapsed panel on a subtle
/// column backdrop. Clicking an icon reopens its panel (`mouse::apply_mouse`).
pub fn draw_rail(fb: &mut Framebuffer, rail: crate::layout::Rect, eff: crate::layout::EffPanels) {
    if rail.is_empty() {
        return;
    }
    let bg = 0x00_141A26;
    for ry in 0..rail.h {
        fb.set(
            rail.x,
            rail.y + ry,
            Cell {
                glyph: ' ' as u32,
                fg: TOOLBOX_HEADER_FG,
                bg,
                ..Cell::default()
            },
        );
    }
    for (_, icon, row) in rail_icon_rows(rail, eff) {
        fb.set(
            rail.x,
            row,
            Cell {
                glyph: icon as u32,
                fg: 0x00_FFD000,
                bg,
                ..Cell::default()
            },
        );
    }
}

/// Hit-test the rail: the collapsed panel whose icon sits under `(col, row)`.
pub fn rail_hit(
    rail: crate::layout::Rect,
    eff: crate::layout::EffPanels,
    col: usize,
    row: usize,
) -> Option<crate::layout::PanelId> {
    if rail.is_empty() || col != rail.x {
        return None;
    }
    rail_icon_rows(rail, eff)
        .into_iter()
        .find(|&(_, _, r)| r == row)
        .map(|(id, _, _)| id)
}

// ─── COMMAND PALETTE OVERLAY ──────────────────────────────────────────────────

/// Geometry of the `:` command palette box: `(x0, y0, pw, ph)` — a centered
/// query line + match rows. `None` when the framebuffer can't host it. Shared
/// by the painter and the mouse hit-test.
pub fn cmd_palette_box_geom(w: usize, h: usize) -> Option<(usize, usize, usize, usize)> {
    if w < 30 || h < 8 {
        return None;
    }
    let pw = (w * 2 / 3).clamp(28, 70).min(w);
    let ph = (h * 2 / 3).clamp(8, 20).min(h);
    Some(((w - pw) / 2, (h - ph) / 4 + 1, pw, ph))
}

/// How many match rows the palette box shows.
const fn cmd_palette_visible(ph: usize) -> usize {
    ph.saturating_sub(4) // border ×2 + query row + hint row
}

/// Map an absolute `(col, row)` click inside the palette box to the MATCH ROW
/// index drawn there (into [`crate::keymap::palette_matches`], after the
/// selection-following scroll), or `None`. Same windowing as [`draw_cmd_palette`].
pub fn cmd_palette_row_at(
    state: &EditorState,
    w: usize,
    h: usize,
    col: usize,
    row: usize,
) -> Option<usize> {
    let (x0, y0, pw, ph) = cmd_palette_box_geom(w, h)?;
    if col < x0 + 1 || col >= x0 + pw - 1 {
        return None;
    }
    let visible = cmd_palette_visible(ph);
    let top = y0 + 2; // border + query line
    if visible == 0 || row < top || row >= top + visible {
        return None;
    }
    let matches = crate::keymap::palette_matches(state);
    let sel = state.cmd_palette.as_ref().map_or(0, |p| p.sel);
    let scroll = settings_scroll(
        sel.min(matches.len().saturating_sub(1)),
        matches.len(),
        visible,
    );
    let idx = scroll + (row - top);
    if idx < matches.len() {
        Some(idx)
    } else {
        None
    }
}

/// Paint the `:` command palette: bordered box, the live query with a caret,
/// then the fuzzy matches (selected row highlighted, chord column right-ish).
/// Rows/scroll come from the same [`crate::keymap::palette_matches`] +
/// [`settings_scroll`] the key handler and the click hit-test use.
pub fn draw_cmd_palette(state: &EditorState, fb: &mut Framebuffer, w: usize, h: usize) {
    let Some(p) = state.cmd_palette.as_ref() else {
        return;
    };
    let Some((x0, y0, pw, ph)) = cmd_palette_box_geom(w, h) else {
        return;
    };
    let bg = 0x00_0C1018;
    let border = 0x00_77AADD;
    let fg = 0x00_C8D4E0;
    for ry in 0..ph {
        for rx in 0..pw {
            let edge = rx == 0 || ry == 0 || rx == pw - 1 || ry == ph - 1;
            fb.set(
                x0 + rx,
                y0 + ry,
                Cell {
                    glyph: if edge { '#' as u32 } else { ' ' as u32 },
                    fg: if edge { border } else { fg },
                    bg,
                    ..Cell::default()
                },
            );
        }
    }
    let inner_w = pw.saturating_sub(2);
    let ix = x0 + 1;
    // Query line with a visible caret.
    let q: String = format!(": {}\u{2038}", p.query)
        .chars()
        .take(inner_w)
        .collect();
    draw_str(fb, ix, y0 + 1, &q, STATUS_FG, bg);
    // Match rows.
    let matches = crate::keymap::palette_matches(state);
    let visible = cmd_palette_visible(ph);
    let sel = p.sel.min(matches.len().saturating_sub(1));
    let scroll = settings_scroll(sel, matches.len(), visible);
    for (i, m) in matches.iter().skip(scroll).take(visible).enumerate() {
        let idx = scroll + i;
        let row_y = y0 + 2 + i;
        let selected = idx == sel && !matches.is_empty();
        let (rfg, rbg) = if selected {
            (0x00_101010, 0x00_FFD000)
        } else {
            (fg, bg)
        };
        if selected {
            for rx in 1..pw - 1 {
                fb.set(
                    x0 + rx,
                    row_y,
                    Cell {
                        glyph: ' ' as u32,
                        fg: rfg,
                        bg: rbg,
                        ..Cell::default()
                    },
                );
            }
        }
        let chord = if m.chord.is_empty() {
            String::new()
        } else {
            format!("  [{}]", m.chord)
        };
        let line: String = format!("{}{}", m.text, chord)
            .chars()
            .take(inner_w)
            .collect();
        draw_str(fb, ix, row_y, &line, rfg, rbg);
    }
    if matches.is_empty() {
        draw_str(fb, ix, y0 + 2, "(no match)", 0x00_8899AA, bg);
    }
    let hint: String = "Enter run  Esc close".chars().take(inner_w).collect();
    draw_str(fb, ix, y0 + ph.saturating_sub(2), &hint, 0x00_8899AA, bg);
}

// ─── INSPECTOR ROW MODEL ──────────────────────────────────────────────────────
//
// The inspector's property sheet is a TYPED, SCROLLABLE row list: one row per
// field of the selected object (every field, not just the focused one), plus a
// summary header. `inspector_rows` is THE single source of truth consumed by the
// painter (`draw_panel`), the click hit-test (`inspector_row_hit` in this module,
// routed by `mouse.rs`), and the geometry (`inspector_viewport`), so what is
// drawn, what is clickable, and what scrolls can never disagree. Scrolling
// reuses the proven `settings_scroll` keep-focus-visible math.

/// What KIND of row an inspector row is (affects colour + click behaviour).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RowKind {
    /// A per-object summary / section banner row (not clickable).
    Header,
    /// An editable field row: `label = value`, clickable to focus.
    Field,
    /// A passive hint/readout row (not clickable).
    Info,
}

/// What a CLICK on an inspector row does. Field rows focus their field (the same
/// focus `;`/`'` page through); header/info rows do nothing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RowAction {
    /// Not clickable.
    None,
    /// Focus a non-emitter entity field (`state.entity_field`).
    FocusEntity(crate::state::EntityField),
    /// Focus an emitter field (`state.emitter_field`).
    FocusEmitter(crate::state::EmitterField),
    /// Focus a material field (`state.material_field`).
    FocusMaterial(crate::material::MaterialField),
}

/// One typed row of the inspector property sheet.
#[derive(Clone, Debug)]
pub struct InspectorRow {
    /// Row class (header / field / info).
    pub kind: RowKind,
    /// Left-hand label (field name, or the whole text for header/info rows).
    pub label: String,
    /// Right-hand value. TYPED: the row carries the datum plus the
    /// metadata a widget needs, not a pre-formatted `String`, so a click can
    /// toggle / step / open an editor instead of only focusing the field.
    /// Header and info rows carry an empty [`RowValue::Text`].
    pub value: RowValue,
    /// True when this row's field is the focused one (`;`/`'` cursor).
    pub focused: bool,
    /// What clicking the row does.
    pub action: RowAction,
}

impl InspectorRow {
    /// A non-clickable header/info row.
    fn banner(kind: RowKind, text: impl Into<String>) -> Self {
        Self {
            kind,
            label: text.into(),
            value: RowValue::Text(String::new()),
            focused: false,
            action: RowAction::None,
        }
    }

    /// A clickable field row.
    fn field(label: impl Into<String>, value: RowValue, focused: bool, action: RowAction) -> Self {
        Self {
            kind: RowKind::Field,
            label: label.into(),
            value,
            focused,
            action,
        }
    }

    /// The display text of the row at UNBOUNDED width. Field rows render
    /// through [`crate::rowvalue::row_text_in`] — THE layout SSOT the hit-test
    /// also consumes — so a plain text field reads `label = value` and a typed
    /// one reads `label = <widget>`. Header/info rows are their bare label.
    ///
    /// TEST-ONLY. The real painter calls `row_text_in` directly with the
    /// panel's measured width (see `draw_inspector`); this unbounded form
    /// exists so assertions can talk about a row without a panel.
    #[cfg(test)]
    pub fn text(&self) -> String {
        match self.kind {
            RowKind::Field => crate::rowvalue::row_text_in(&self.label, &self.value, usize::MAX),
            _ => self.label.clone(),
        }
    }

    /// Resolve an absolute column inside this row's drawn line to its
    /// [`RowZone`]. `text_x` is the column the row's TEXT starts at (one past
    /// the focus marker the painter prefixes). Non-field rows are all label.
    fn zone_at(&self, text_x: usize, width: usize, col: usize) -> RowZone {
        if self.kind != RowKind::Field || col < text_x {
            return RowZone::Label;
        }
        crate::rowvalue::zone_at_in(&self.label, &self.value, width, col - text_x)
    }
}

/// Build the typed row list for the ACTIVE inspector context: the
/// selected object's summary + one row per editable field, or the vox/material
/// equivalents, or a tips banner when nothing is active. Pure; the SSOT shared
/// by painter, hit-test, and scroll geometry.
pub fn inspector_rows(state: &EditorState) -> Vec<InspectorRow> {
    use crate::state::EMITTER_FIELDS;
    let mut rows = Vec::new();
    if state.vox_mode {
        // Vox mode has no field cycle: a summary + the live cursor offset.
        if crate::vox::selected_file(state).is_none() {
            rows.push(InspectorRow::banner(RowKind::Info, "no .vox in levels/"));
            return rows;
        }
        rows.push(InspectorRow::banner(
            RowKind::Header,
            crate::vox::vox_summary(state),
        ));
        let (cx, cy) = state.cursor;
        rows.push(InspectorRow::banner(
            RowKind::Info,
            format!("off {cx},{cy},{} (cursor)", state.z),
        ));
        return rows;
    }
    if let Some(kind) = state.place {
        if kind == EntityKind::Emitter {
            match state
                .selected_emitter
                .filter(|&i| i < state.level.emitters.len())
            {
                Some(i) => {
                    let e = &state.level.emitters[i];
                    rows.push(InspectorRow::banner(
                        RowKind::Header,
                        format!(
                            "#{i} {} r{:.0} sp{:.1} cone{:.0}",
                            e.family, e.rate, e.speed, e.spread_deg
                        ),
                    ));
                    for f in EMITTER_FIELDS {
                        rows.push(InspectorRow::field(
                            format!("[{}] {}", f.group().label(), f.label()),
                            crate::rowtype::emitter_value(state, f),
                            state.emitter_field == f,
                            RowAction::FocusEmitter(f),
                        ));
                    }
                }
                None => rows.push(InspectorRow::banner(
                    RowKind::Info,
                    "place/select an emitter",
                )),
            }
            return rows;
        }
        rows.push(InspectorRow::banner(
            RowKind::Header,
            entity_summary(state, kind),
        ));
        // While the PART MASK modal is open the mask readout IS the sheet; the
        // field rows would fight the modal for attention (and its keys).
        if state.mask_mode {
            return rows;
        }
        for &f in crate::state::entity_fields_for(kind) {
            // The two door-target fields and NodeEmits render a shared cursor
            // readout; every field still gets its own row so all are clickable.
            rows.push(InspectorRow::field(
                f.label(),
                crate::rowtype::entity_value(state, f),
                state.entity_field == f,
                RowAction::FocusEntity(f),
            ));
        }
        return rows;
    }
    if state.material_mode {
        if crate::material::sel_index(state).is_none() {
            rows.push(InspectorRow::banner(RowKind::Info, "no materials (e: new)"));
            return rows;
        }
        rows.push(InspectorRow::banner(
            RowKind::Header,
            crate::material::material_summary(state),
        ));
        // The trailing MATERIAL_FIELDS entry is the wrap sentinel (duplicates the
        // first) — list each field once.
        let unique =
            &crate::material::MATERIAL_FIELDS[..crate::material::MATERIAL_FIELDS.len() - 1];
        for &f in unique {
            rows.push(InspectorRow::field(
                f.label(),
                crate::rowtype::material_value(state, f),
                state.material_field == f,
                RowAction::FocusMaterial(f),
            ));
        }
        return rows;
    }
    if state.stair_mode || state.shape_mode {
        // The brush line (drawn in the fixed furniture) already summarizes these
        // tools; no property sheet.
        return rows;
    }
    // Nothing active: a short Tips banner instead of blank space.
    rows.push(InspectorRow::banner(
        RowKind::Info,
        "Pick a tool \u{00B7} click grid to",
    ));
    rows.push(InspectorRow::banner(
        RowKind::Info,
        "paint \u{00B7} o = place object",
    ));
    rows
}

/// The [`RowAction`] of the CURRENTLY FOCUSED property-sheet row — what `Enter`
/// acts on. Resolved from the same [`inspector_rows`] list the painter and the
/// hit-test use, so the keyboard and the mouse always name the same field.
/// [`RowAction::None`] when nothing is focused (no active field context).
#[must_use]
pub fn focused_row_action(state: &EditorState) -> RowAction {
    inspector_rows(state)
        .into_iter()
        .find(|r| r.focused)
        .map_or(RowAction::None, |r| r.action)
}

/// First inspector row (relative to the organ top) of the scrollable property
/// sheet viewport. Rows 0..=4 are fixed furniture: header, file, brush/mode,
/// fill-anchor, z-slider ([`Z_SLIDER_ROW`] = 4).
pub const INSPECTOR_ROWS_TOP: usize = 5;

/// How many wrapped rows the current button set needs at panel width `w`.
/// Mirrors the wrap loop in [`panel_buttons`] exactly (same SSOT labels).
fn button_rows_needed(state: &EditorState, w: usize) -> usize {
    if w == 0 {
        return 0;
    }
    let mut rows = 1usize;
    let mut col = 0usize;
    for b in panel_button_set(state) {
        let bw = b.label().len() + 2;
        if col + bw > w {
            rows += 1;
            col = 0;
        }
        col += bw + 1;
    }
    rows
}

/// The inspector VIEWPORT geometry: `(visible_rows, button_anchor_row)` where
/// `visible_rows` is how many property-sheet rows fit and `button_anchor_row` is
/// the ABSOLUTE row the action-button block starts on (one blank separator below
/// the viewport). THE geometry SSOT shared by the painter ([`draw_panel`]), the
/// row hit-test ([`inspector_row_hit`]), and the button layout
/// ([`panel_buttons`]) — replacing the old fixed `PANEL_BTN_ROW0 = 12` anchor.
pub fn inspector_viewport(state: &EditorState, panel: crate::layout::Rect) -> (usize, usize) {
    if panel.is_empty() {
        return (0, panel.y + INSPECTOR_ROWS_TOP + 1);
    }
    let total = inspector_rows(state).len();
    let btn_rows = button_rows_needed(state, panel.w);
    // Rows available below the fixed furniture.
    let avail_total = panel.h.saturating_sub(INSPECTOR_ROWS_TOP);
    // Ideal split: sheet + separator + the FULL button block.
    let ideal = avail_total.saturating_sub(btn_rows + 1);
    let visible = if ideal >= total {
        total
    } else {
        // Everything does not fit. The property sheet has PRIORITY on short
        // panels (an unreachable field is a data-entry hole; a shed button has a
        // keyboard equivalent): keep at least [`MIN_SHEET_ROWS`] sheet rows, and
        // let the button block clip at the panel bottom (`panel_buttons` already
        // stops at the rect edge). Capped one short of the space so the button
        // anchor row stays inside the panel when any space remains.
        total
            .min(ideal.max(MIN_SHEET_ROWS))
            .min(avail_total.saturating_sub(1))
    };
    (visible, panel.y + INSPECTOR_ROWS_TOP + visible + 1)
}

/// Minimum property-sheet rows kept visible (with ▲/▼ scrolling) when the panel
/// is too short for the sheet plus the full button block — the sheet wins,
/// buttons shed (they all have keyboard equivalents).
const MIN_SHEET_ROWS: usize = 4;

/// The column an inspector row's TEXT starts at: one past the panel edge,
/// because the painter prefixes a focus-marker cell. Shared by the painter and
/// the hit-test so the marker is only accounted for once.
fn inspector_text_x(panel: crate::layout::Rect) -> usize {
    panel.x + 1
}

/// Cells available for an inspector row's TEXT: the panel width less the focus
/// marker column. Rows are laid out to this width (`rowvalue::zones_in`), so a
/// narrow panel compacts the affordances instead of clipping them off the row —
/// the painter and the hit-test read the same number.
fn inspector_text_w(panel: crate::layout::Rect) -> usize {
    panel.w.saturating_sub(1)
}

/// Map an absolute `(col, row)` click inside the INSPECTOR organ to the
/// [`RowAction`] of the property-sheet row drawn there AND the [`RowZone`] the
/// column fell in, or `None` (outside the viewport / a non-clickable row).
///
/// Returning the ZONE is what makes typed rows directly manipulable: a click on
/// the LABEL focuses the field (the pre-existing behaviour, unchanged), while a
/// click on a widget part toggles / steps / opens that field's editor. Uses the
/// SAME rows + scroll math as the painter and the SAME per-row span layout
/// (`rowvalue::zones_in`), so a click lands on exactly what is drawn.
pub fn inspector_row_hit(
    state: &EditorState,
    panel: crate::layout::Rect,
    col: usize,
    row: usize,
) -> Option<(RowAction, RowZone)> {
    if panel.is_empty() || col < panel.x || col >= panel.x + panel.w {
        return None;
    }
    let (visible, _) = inspector_viewport(state, panel);
    let top = panel.y + INSPECTOR_ROWS_TOP;
    if visible == 0 || row < top || row >= top + visible {
        return None;
    }
    let rows = inspector_rows(state);
    let focused_idx = rows.iter().position(|r| r.focused).unwrap_or(0);
    let scroll = settings_scroll(focused_idx, rows.len(), visible);
    let idx = scroll + (row - top);
    let r = rows.get(idx)?;
    if r.action == RowAction::None {
        return None;
    }
    Some((
        r.action,
        r.zone_at(inspector_text_x(panel), inspector_text_w(panel), col),
    ))
}

/// The `(x0, x1, row)` absolute span of ONE panel action button, or `None` when
/// it is not currently shown. The compatibility accessor the typed inspector
/// row model preserves: `mouse.rs` span discovery and the elevation-strip
/// anchor both resolve button geometry through the same [`panel_buttons`]
/// layout.
// Consumed today by the span-discovery TESTS (mouse.rs `panel_btn_mid` + the
// chrome parity tests); production callers arrive through the adaptive panel
// work, so the non-test build carries it knowingly.
#[cfg_attr(not(test), allow(dead_code))]
pub fn button_span(
    state: &EditorState,
    panel: crate::layout::Rect,
    button: PanelButton,
) -> Option<(usize, usize, usize)> {
    panel_buttons(state, panel)
        .into_iter()
        .find(|&(b, _, _, _)| b == button)
        .map(|(_, s, e, r)| (s, e, r))
}

/// The action buttons to show for the current state, in row-major order. P2
/// (inspector) only appears when a field context is active; P6 (tree actions) only
/// when the tree is open. Pure; THE single source of truth for layout.
fn panel_button_set(state: &EditorState) -> Vec<PanelButton> {
    let mut v = Vec::new();
    if inspector_context(state) {
        // `[Fld]` cycles a field — but vox mode has NO field cycle (it uses +/-
        // scale, `c` material, `x` delete), so omit it there.
        if !state.vox_mode {
            v.push(PanelButton::Field);
        }
        v.extend([
            PanelButton::Minus,
            PanelButton::Plus,
            PanelButton::Cycle,
            PanelButton::Delete,
        ]);
        // `[+Mat]` (new material) belongs with the material context.
        if state.material_mode {
            v.push(PanelButton::NewMat);
        }
        // `[Sw]` (pipe switch toggle) only while placing a Pipe.
        if state.place == Some(EntityKind::Pipe) {
            v.push(PanelButton::PipeSwitch);
        }
        // `[Tag]` (tag node at cursor) only while placing a Node.
        if state.place == Some(EntityKind::Node) {
            v.push(PanelButton::TagNode);
        }
    }
    v.extend([
        PanelButton::ZoomIn,
        PanelButton::ZoomOut,
        PanelButton::FloorUp,
        PanelButton::FloorDown,
        PanelButton::Noclip,
        PanelButton::Tree,
        PanelButton::GoToCursor,
        // `[Create]` (open the create-object palette) is a general affordance.
        PanelButton::CreateObj,
    ]);
    v.extend([
        PanelButton::SetA,
        PanelButton::Rect,
        PanelButton::Line,
        PanelButton::Bucket,
    ]);
    if state.tree.open {
        v.extend([
            PanelButton::TreeDup,
            PanelButton::TreeDel,
            PanelButton::TreeParent,
            PanelButton::TreeUnparent,
        ]);
    }
    v
}

/// Layout the panel action buttons into `(button, start, end, row)` absolute
/// spans, wrapping to new rows within the panel width. Shared by painter +
/// hit-test. `panel` is the panel rect; buttons stay within it.
///
/// The block is ANCHORED DYNAMICALLY one row below the scrollable
/// property-sheet viewport ([`inspector_viewport`]) rather than a fixed row,
/// so the buttons sit right under however many field rows the selected
/// object shows.
pub fn panel_buttons(
    state: &EditorState,
    panel: crate::layout::Rect,
) -> Vec<(PanelButton, usize, usize, usize)> {
    let mut out = Vec::new();
    let (_, btn_row0) = inspector_viewport(state, panel);
    if panel.is_empty() || btn_row0 >= panel.y + panel.h {
        return out;
    }
    let x0 = panel.x;
    let max_w = panel.w;
    let mut row = btn_row0;
    let last_row = panel.y + panel.h; // exclusive
    let mut col = 0usize;
    for b in panel_button_set(state) {
        let w = b.label().len() + 2; // [label]
        if col + w > max_w {
            // wrap
            col = 0;
            row += 1;
        }
        if row >= last_row {
            break; // out of panel height
        }
        out.push((b, x0 + col, x0 + col + w, row));
        col += w + 1; // single-space gap
    }
    out
}

/// Paint the INSPECTOR action buttons (P2/P3/P4/P6) into the inspector sub-rect.
/// No-op if none fit. The per-type OBJECTS legend that used to follow the buttons
/// has moved into the OUTLINER organ (live counts now ride on the tree's group
/// headers), so this paints buttons only.
pub fn draw_panel_buttons(state: &EditorState, fb: &mut Framebuffer, panel: crate::layout::Rect) {
    let bg = 0x00_141A26;
    let fg = 0x00_AEC4DA;
    for (b, s, _e, row) in panel_buttons(state, panel) {
        draw_str(fb, s, row, &format!("[{}]", b.label()), fg, bg);
    }
}

/// Hit-test a panel action button under absolute `(col, row)`.
pub fn panel_button_hit(
    state: &EditorState,
    panel: crate::layout::Rect,
    col: usize,
    row: usize,
) -> Option<PanelButton> {
    panel_buttons(state, panel)
        .into_iter()
        .find(|&(_, s, e, r)| r == row && col >= s && col < e)
        .map(|(b, _, _, _)| b)
}

/// Map an absolute `(col, row)` click in the OUTLINER pane to the tree ROW INDEX
/// under it, or `None`. Uses the SAME body offset ([`OUTLINER_HEADER_ROWS`]) +
/// `window_start` windowing as `draw_tree`, so a click selects the row drawn
/// under the pointer.
pub fn tree_row_at(
    state: &EditorState,
    tree_rect: crate::layout::Rect,
    col: usize,
    row: usize,
) -> Option<usize> {
    if tree_rect.is_empty() {
        return None;
    }
    if col < tree_rect.x || col >= tree_rect.x + tree_rect.w {
        return None;
    }
    let rows = tree::build_tree(state);
    if rows.is_empty() {
        return None;
    }
    let body_y0 = tree_rect.y + OUTLINER_HEADER_ROWS;
    let body_h = tree_rect.h.saturating_sub(OUTLINER_HEADER_ROWS);
    if body_h == 0 || row < body_y0 || row >= body_y0 + body_h {
        return None;
    }
    let cursor = state.tree.cursor.min(rows.len() - 1);
    let start = window_start(cursor, rows.len(), body_h);
    let ri = start + (row - body_y0);
    if ri < rows.len() {
        Some(ri)
    } else {
        None
    }
}

/// Draw the PREVIEW-CAPTURE indicator banner: a bright one-row strip across the
/// top of the framebuffer announcing free-look is active and how to release it.
/// Called from `draw_system` while `state.preview_captured`. Clipped to `w`.
pub fn draw_capture_banner(fb: &mut Framebuffer, w: usize, _h: usize) {
    if w == 0 {
        return;
    }
    let bg = 0x00_802000; // warm alert background
    let fg = 0x00_FFE060; // bright text
    for rx in 0..w {
        fb.set(
            rx,
            0,
            Cell {
                glyph: ' ' as u32,
                fg,
                bg,
                ..Cell::default()
            },
        );
    }
    let msg = "LOOK — release right mouse to stop";
    let shown: String = msg.chars().take(w).collect();
    // Center the message if there is room.
    let x = (w.saturating_sub(shown.chars().count())) / 2;
    draw_str(fb, x, 0, &shown, fg, bg);
}

/// Draw a one-line CONTROLS HINT along the bottom row of the preview pane so the
/// look/move controls are discoverable. A no-op while free-look is captured (the
/// capture banner already explains how to release it) or if the pane is too thin
/// to spare a row. Overwrites at most ONE row of the rendered scene. Intended to
/// be called by `main.rs` right after `preview::render_preview`.
pub fn draw_preview_hint(state: &EditorState, fb: &mut Framebuffer, rect: crate::layout::Rect) {
    if rect.is_empty() || rect.h < 3 || rect.w < 8 || state.preview_captured {
        return;
    }
    let bg = 0x00_0A0E16;
    let fg = 0x00_88A8C8;
    let row = rect.y + rect.h - 1;
    // Paint the hint row background so the text reads over the scene.
    for cx in 0..rect.w {
        fb.set(
            rect.x + cx,
            row,
            Cell {
                glyph: ' ' as u32,
                fg,
                bg,
                ..Cell::default()
            },
        );
    }
    // ONE clean single-width hint, clipped to the pane width. Kept short so it reads
    // cleanly inside a narrow preview and never runs into the side panel beside it.
    let full = "hold right mouse = look \u{00B7} WASD = fly \u{00B7} wheel = dolly";
    let shown: String = full.chars().take(rect.w).collect();
    draw_str(fb, rect.x, row, &shown, fg, bg);
}

/// Minimum terminal size (cols × rows) the editor needs to draw a usable chrome:
/// rulers + a grid + the side panel + the toolbar don't fit below this. Exposed so
/// `main.rs` can show a friendly notice instead of a corrupt, cramped layout.
pub const MIN_COLS: usize = 40;
pub const MIN_ROWS: usize = 12;

/// True when the terminal is too small for a usable editor (below [`MIN_COLS`] ×
/// [`MIN_ROWS`]). `main.rs` calls this each frame and, when true, paints
/// [`draw_too_small`] instead of the normal chrome.
pub fn too_small(cols: usize, rows: usize) -> bool {
    cols < MIN_COLS || rows < MIN_ROWS
}

/// Clear the framebuffer and centre a readable "terminal too small" notice telling
/// the user the minimum size to enlarge to. Used when [`too_small`] is true.
pub fn draw_too_small(fb: &mut re_core::Framebuffer, cols: usize, rows: usize) {
    let bg = 0x00_000000;
    let fg = 0x00_FFD24A;
    // Clear the whole pane.
    for y in 0..rows {
        for x in 0..cols {
            fb.set(
                x,
                y,
                Cell {
                    glyph: ' ' as u32,
                    fg,
                    bg,
                    ..Cell::default()
                },
            );
        }
    }
    let line1 = "Terminal too small";
    let line2 = format!("please enlarge (min {MIN_COLS}x{MIN_ROWS})");
    let mid = rows / 2;
    let put = |fb: &mut re_core::Framebuffer, text: &str, y: usize| {
        let shown: String = text.chars().take(cols).collect();
        let x = cols.saturating_sub(shown.chars().count()) / 2;
        draw_str(fb, x, y, &shown, fg, bg);
    };
    put(fb, line1, mid.saturating_sub(1));
    put(fb, &line2, mid);
}

/// The `[X]` close-button span on a modal box's top border: `(x, y, start, end)`
/// columns (absolute), placed at the top-right inside the border. Shared by the
/// help painter and the hit-test. Returns the absolute row `y` and `[start,end)`
/// absolute column range of the 3-cell `[X]`.
pub fn modal_close_button(x0: usize, y0: usize, pw: usize) -> (usize, usize, usize) {
    // 3 cells "[X]" ending one cell in from the right border.
    let end = x0 + pw.saturating_sub(1);
    let start = end.saturating_sub(3);
    (y0, start, end)
}

/// Map an absolute `(col, row)` click to the PALETTE item index under it, or
/// `None`. Items are drawn at `y0 + 2 + i` (mirrors `draw_palette`).
pub fn palette_item_at(w: usize, h: usize, col: usize, row: usize) -> Option<usize> {
    let (x0, y0, pw, _ph) = palette_box_geom(w, h)?;
    let items = tree::PALETTE_ITEMS;
    if col <= x0 || col >= x0 + pw - 1 {
        return None;
    }
    let first = y0 + 2;
    if row < first {
        return None;
    }
    let i = row - first;
    if i < items.len() {
        Some(i)
    } else {
        None
    }
}

/// Distinct top-down marker glyph + foreground colour for an ENTITY at a cell.
/// Door `D` (red), Spawn `@` (green), Pedestal `X` (yellow), Node `N` (cyan,
/// tinted by io role). Drawn ON TOP of the geometry glyph so placed entities are
/// always visible.
struct EntityMarker {
    glyph: char,
    fg: u32,
}

// ── OBJECT MARKER PALETTE (single source of truth) ───────────────────────────
//
// The grid markers AND the side-panel OBJECTS legend both read from these so the
// legend always matches what is painted on the grid. Hues are deliberately spread
// so no two object TYPES are easily confused at a glance:
//   Door     red       Spawn    green     Pedestal gold
//   Node     cyan      input    teal      output   violet
//   Emitter  orange    Trigger  spring-green   Vox  magenta   Pipe  steel-blue
const DOOR_FG: u32 = 0x00_FF4040; // pure red
const SPAWN_FG: u32 = 0x00_40FF60; // pure green
const PEDESTAL_FG: u32 = 0x00_FFC020; // gold (warmer/darker than the orange emitter)
const NODE_FG: u32 = 0x00_40D0FF; // sky cyan
const NODE_INPUT_FG: u32 = 0x00_40FFD0; // teal-shifted input
const NODE_OUTPUT_FG: u32 = 0x00_C080FF; // violet-shifted output
const EMITTER_FG: u32 = 0x00_FF8000; // warm orange emitter spark (no green channel — distinct from gold)
const EMITTER_SEL_FG: u32 = 0x00_FFE060; // brighter when selected
const VOX_FG: u32 = 0x00_FF40FF; // magenta vox/box-model anchor
const VOX_SEL_FG: u32 = 0x00_FF90FF; // brighter when selected
const PIPE_FG: u32 = 0x00_5080C0; // steel-blue pipe-path trace (distinct from node teal/cyan)
const PIPE_SEL_FG: u32 = 0x00_80B0FF; // brighter steel-blue when the pipe is selected
const PIPE_SWITCH_FG: u32 = 0x00_FFD040; // amber switch-cell marker
const PART_FG: u32 = 0x00_B0FF40; // chartreuse part-volume anchor (distinct from gold/orange/green)
const PART_SEL_FG: u32 = 0x00_D0FF90; // brighter chartreuse when the part is selected

// ─── ORGAN THEME (region accent colours) ─────────────────────────────────────
//
// Real 3D editors (Blender/Godot/Unity) split their chrome into recognisable
// ORGANS — a command bar, a toolbox, an outliner, an inspector, a status line —
// each with its own visual identity so the eye finds "where do I do X" without
// reading every label. These accent colours give each editor region one HEADER
// hue so the regions read as distinct places, not "buttons thrown in a row".
//
// RESERVED-HUE RULE: the ENTITY marker hues above (DOOR_FG / SPAWN_FG /
// PEDESTAL_FG / NODE_FG / EMITTER_FG / TRIGGER_FG / VOX_FG / PIPE_FG and their
// `*_SEL` variants) are reserved for OBJECT IDENTITY — they say *what an object
// is* on the grid + in the legend. They MUST NOT be reused as organ chrome: a
// command-bar header tinted the same red as a Door would make the user read the
// toolbar as an object. The organ hues below are deliberately chosen clear of
// those object hues.
/// COMMAND BAR (top toolbar: File/Edit/View/Help clusters) — soft blue. Used to
/// tint the command bar's cluster dividers in `toolbar.rs`.
pub const COMMAND_HEADER_FG: u32 = 0x00_77AADD;
/// TOOLBOX (the vertical left tool column) — cool neutral grey (a tool tray, not
/// an object), so it never competes with the saturated object hues. Used by
/// `palette_bar::draw_toolbox` to tint the group HEADER rows (GEOMETRY / SHAPES /
/// OBJECTS / PAINT / ERASE).
pub const TOOLBOX_HEADER_FG: u32 = 0x00_8899AA;
/// OUTLINER (object tree / scene list) — bright sky blue. Used for the tree
/// pane's header in [`draw_tree`].
pub const OUTLINER_HEADER_FG: u32 = 0x00_88CCFF;
/// INSPECTOR (the side panel where properties are changed) — warm amber, the
/// deliberate odd-one-out so "where I change properties" is obvious at a glance.
/// Used for the side panel's header in [`draw_panel`].
pub const INSPECTOR_HEADER_FG: u32 = 0x00_FF9C40;
/// STATUS BAR (the bottom in-the-moment readout row) — golden, high-contrast.
/// Used for the readouts in [`draw_status_bar`].
pub const STATUS_FG: u32 = 0x00_FFD000;
/// ORGAN DIVIDER — a subtle slate used for the single-width box-drawing rules
/// (`│` / `─`) painted where two organs meet (or where segments meet inside one
/// organ), so the regions read as separated. Used for the `│` group separators
/// in [`draw_status_bar`].
pub const ORGAN_DIVIDER_FG: u32 = 0x00_2A3340;

/// VIEWPORT HEADER — a dim slate-blue accent for the 1-row label drawn atop each
/// CENTER viewport (the 2D grid and the 3D preview). Deliberately dim so it reads
/// as a quiet caption, not a clickable organ header (those use the saturated organ
/// hues above). Single-width glyphs only.
pub const VIEWPORT_HEADER_FG: u32 = 0x00_6F8398;
/// VIEWPORT HEADER (FOCUSED) — a bright cyan accent for the caption of the CENTER
/// viewport that currently holds focus (set by click-to-focus). The unfocused
/// viewport keeps the dim [`VIEWPORT_HEADER_FG`]. A `*` marker is also prefixed on
/// the focused caption so the cue reads even without colour. Focus is a VISUAL CUE
/// (and the swap target) only — it does NOT change key routing.
pub const VIEWPORT_FOCUS_FG: u32 = 0x00_60E0FF;
/// AUDITION ON accent — a hot amber-red for the 3D header label while the
/// live sound-field audition (`Shift+U`, `crate::field::Audition`) is armed.
/// Saturated on purpose: this is a "the app is doing something with real
/// audio" state, so it must read even without colour precision, unlike the
/// deliberately-quiet viewport captions above it.
pub const AUDITION_ON_FG: u32 = 0x00_FF6A3C;
/// SWAP CONTROL accent — the clickable `[<>]` glyph drawn at the right end of the
/// LEFT center viewport's header; clicking it flips the 2D/3D sides in Split. A
/// warm amber so it reads as an actionable control, distinct from the dim caption.
pub const VIEWPORT_SWAP_FG: u32 = 0x00_FFB050;
/// Background painted behind a viewport header row so the caption reads over any
/// scene content that would otherwise sit on that row.
const VIEWPORT_HEADER_BG: u32 = 0x00_0A0E16;
/// Rows reserved at the TOP of a CENTER viewport for its 1-row caption header. The
/// scene is drawn BELOW this so the header never overwrites scene content.
pub const VIEWPORT_HEADER_H: usize = 1;

/// The single-width clickable SWAP control label drawn at the right end of the
/// LEFT center viewport's header. Clicking it toggles [`EditorState::swap_viewports`].
pub const VIEWPORT_SWAP_LABEL: &str = "[<>]";

/// The absolute column span `(start, end)` of the `[<>]` swap control on the header
/// row of the viewport at `rect`, or `None` when the header is too thin to host it
/// without colliding with the caption. The control hugs the right edge of the
/// header row at `rect.y`. THE single source of truth shared by the header painters
/// and the `mouse.rs` hit-test, so a click toggles exactly what is drawn. The
/// caller only draws it on the LEFT viewport in Split (see `draw_system`).
pub fn viewport_swap_span(rect: crate::layout::Rect) -> Option<(usize, usize)> {
    let lw = VIEWPORT_SWAP_LABEL.chars().count();
    // Need room for a short caption + a gap + the control; below this the control is
    // omitted so it never overwrites the `2D …` / `3D …` text.
    if rect.is_empty() || rect.w < lw + 6 {
        return None;
    }
    let start = rect.x + rect.w - lw;
    Some((start, start + lw))
}

/// Paint the 1-row header at the top of the 2D GRID viewport: `2D · z {z}/{sz} ·
/// ×{zoom}` in the dim viewport accent. When `focused` the caption uses the bright
/// [`VIEWPORT_FOCUS_FG`] and is prefixed with a `*` so the focus reads even without
/// colour; otherwise the dim [`VIEWPORT_HEADER_FG`]. When `show_swap` the clickable
/// `[<>]` control is drawn at the right edge (the LEFT viewport in Split). Fills the
/// header row background first so it reads over anything beneath. No-op for an
/// empty/too-thin rect. The caller draws the grid scene starting one row LOWER
/// ([`VIEWPORT_HEADER_H`]) so this never overwrites cells. Single-width glyphs.
pub fn draw_grid_header(
    state: &EditorState,
    fb: &mut Framebuffer,
    rect: crate::layout::Rect,
    focused: bool,
    show_swap: bool,
) {
    if rect.is_empty() || rect.w < 4 {
        return;
    }
    let (_sx, _sy, sz) = state.map.size();
    let bg = VIEWPORT_HEADER_BG;
    let fg = if focused {
        VIEWPORT_FOCUS_FG
    } else {
        VIEWPORT_HEADER_FG
    };
    for cx in 0..rect.w {
        fb.set(rect.x + cx, rect.y, glyph_cell(' ', fg, bg));
    }
    // `·` is U+00B7 MIDDLE DOT, `×` is U+00D7 — both single-width. A `*` prefix
    // marks the focused viewport (reads without colour).
    let mark = if focused { "*" } else { "" };
    // The ONION SKIN badge is appended only while the aid is ON — an off editor
    // paints the exact caption it painted before the aid existed.
    let onion = if state.onion.is_on() {
        format!(" \u{00B7} onion {}", state.onion.label())
    } else {
        String::new()
    };
    let label = format!(
        "{mark}2D \u{00B7} z {}/{} \u{00B7} \u{00D7}{}{onion}",
        state.z,
        sz.saturating_sub(1),
        state.zoom
    );
    let shown: String = label.chars().take(rect.w).collect();
    draw_str(fb, rect.x, rect.y, &shown, fg, bg);
    if show_swap {
        draw_viewport_swap(fb, rect, bg);
    }
}

/// Paint the 1-row header at the top of the 3D PREVIEW viewport: `3D · noclip` /
/// `3D · collide` in the dim viewport accent (bright + `*`-prefixed when `focused`,
/// with the optional `[<>]` control). Same row-bg-then-caption discipline as
/// [`draw_grid_header`]; the caller renders the scene one row lower. No-op for an
/// empty/too-thin rect. Single-width glyphs.
pub fn draw_preview_header(
    state: &EditorState,
    fb: &mut Framebuffer,
    rect: crate::layout::Rect,
    focused: bool,
    show_swap: bool,
) {
    if rect.is_empty() || rect.w < 4 {
        return;
    }
    let bg = VIEWPORT_HEADER_BG;
    let fg = if focused {
        VIEWPORT_FOCUS_FG
    } else {
        VIEWPORT_HEADER_FG
    };
    for cx in 0..rect.w {
        fb.set(rect.x + cx, rect.y, glyph_cell(' ', fg, bg));
    }
    let mark = if focused { "*" } else { "" };
    let mode = if state.noclip { "noclip" } else { "collide" };
    // AUDITION ON indicator: appended to
    // the header label only while `Shift+U` has armed `state.audition` — off
    // by default, so `HEADER_DIGEST_BEFORE` (captured with no audition
    // running) is unaffected.
    let audition = if state.audition.is_some() {
        " \u{00B7} AUDITION ON"
    } else {
        ""
    };
    let label = format!("{mark}3D \u{00B7} {mode}{audition}");
    let shown: String = label.chars().take(rect.w).collect();
    let label_fg = if state.audition.is_some() {
        AUDITION_ON_FG
    } else {
        fg
    };
    draw_str(fb, rect.x, rect.y, &shown, label_fg, bg);
    if show_swap {
        draw_viewport_swap(fb, rect, bg);
    }
}

/// Paint the `[<>]` swap control at the right edge of `rect`'s header row in the
/// warm [`VIEWPORT_SWAP_FG`] accent. Geometry from [`viewport_swap_span`] (the SAME
/// the hit-test uses). No-op when the header is too thin.
fn draw_viewport_swap(fb: &mut Framebuffer, rect: crate::layout::Rect, bg: u32) {
    if let Some((start, _end)) = viewport_swap_span(rect) {
        draw_str(fb, start, rect.y, VIEWPORT_SWAP_LABEL, VIEWPORT_SWAP_FG, bg);
    }
}

/// Paint the 1-row ORGAN DIVIDER between the OUTLINER and the INSPECTOR: a
/// single-width box-drawing rule (`─`) in [`ORGAN_DIVIDER_FG`] with a centred
/// `─ INSPECTOR ─` label in [`INSPECTOR_HEADER_FG`], so the two stacked organs
/// read as separate places. `x0`/`y0` are the divider row origin; `w` its width.
/// Single-width glyphs only. Pure painting.
pub fn draw_organ_divider(fb: &mut Framebuffer, x0: usize, y0: usize, w: usize) {
    if w == 0 {
        return;
    }
    let bg = 0x00_080C12;
    for rx in 0..w {
        fb.set(x0 + rx, y0, glyph_cell('\u{2500}', ORGAN_DIVIDER_FG, bg)); // ─
    }
    let label = "\u{2500} INSPECTOR \u{2500}"; // ─ INSPECTOR ─
    let lw = label.chars().count();
    if lw <= w {
        let lx = x0 + (w - lw) / 2;
        draw_str(fb, lx, y0, label, INSPECTOR_HEADER_FG, bg);
    }
}

/// A pipe-path cell render: a thin line glyph (`+` for a switch cell, `·`
/// otherwise) in a pipe colour. The SELECTED pipe is drawn brighter so the wire
/// being edited stands out.
struct PipeMarker {
    glyph: char,
    fg: u32,
}

/// Resolve the topmost PIPE-PATH marker at world cell `(wx, wy)` on layer `z`, if
/// any. Pipes are WIRING and render BELOW the entity markers (a node/door standing
/// on a pipe cell still reads on top). The selected pipe is preferred when several
/// path cells overlap, and brighter. A `switch_at` cell shows a `+` in amber.
fn pipe_marker_at(state: &EditorState, wx: i32, wy: i32, z: i32) -> Option<PipeMarker> {
    let idx = crate::pipe::pipe_at(state, wx, wy, z)?;
    let p = &state.level.pipes[idx];
    let is_switch = p
        .switch_cell
        .and_then(|c| p.path.get(c))
        .is_some_and(|&c| c == (wx, wy, z));
    let selected = state.selected_pipe == Some(idx);
    if is_switch {
        return Some(PipeMarker {
            glyph: '+',
            fg: PIPE_SWITCH_FG,
        });
    }
    let fg = if selected { PIPE_SEL_FG } else { PIPE_FG };
    Some(PipeMarker {
        glyph: '\u{00B7}',
        fg,
    }) // · middle dot
}

/// Resolve the topmost entity marker at world cell `(wx, wy)` on layer `z`, if
/// any. Node wins over the singletons when stacked (it carries the most info),
/// then Door, Spawn, Pedestal. Returns `None` when no entity sits on this cell.
fn entity_marker_at(state: &EditorState, wx: i32, wy: i32, z: i32) -> Option<EntityMarker> {
    let lvl = &state.level;
    // VOX (box-model) anchors sit at the TOP of the priority order so a placed
    // model's anchor is always visible (a `▦` magenta marker, brighter when the
    // stamp is selected). The stamped voxels themselves are real Map cells and
    // render via `resolve_cell`; this just flags the anchor.
    if let Some(idx) = crate::vox::vox_at(state, wx, wy, z) {
        let fg = if state.selected_vox == Some(idx) {
            VOX_SEL_FG
        } else {
            VOX_FG
        };
        return Some(EntityMarker {
            glyph: '\u{25A6}',
            fg,
        }); // ▦
    }
    // Emitters sit ABOVE nodes/singletons in priority so a placed emitter is
    // always visible at its origin cell; the selected one is highlighted.
    if let Some(idx) = crate::entity::emitter_at(state, wx, wy, z) {
        let fg = if state.selected_emitter == Some(idx) {
            EMITTER_SEL_FG
        } else {
            EMITTER_FG
        };
        return Some(EntityMarker { glyph: '*', fg });
    }
    if let Some(idx) = lvl.node_at(wx, wy, z) {
        let n = &lvl.nodes[idx];
        let fg = if n.is_input {
            NODE_INPUT_FG
        } else if n.is_output {
            NODE_OUTPUT_FG
        } else {
            NODE_FG
        };
        // Use the node's authored glyph if any, else 'N'.
        let glyph = if n.glyph != '\0' && n.glyph != ' ' {
            n.glyph
        } else {
            'N'
        };
        return Some(EntityMarker { glyph, fg });
    }
    if (lvl.door.x, lvl.door.y, lvl.door.z) == (wx, wy, z)
        && (lvl.door.x, lvl.door.y, lvl.door.z) != (0, 0, 0)
    {
        return Some(EntityMarker {
            glyph: 'D',
            fg: DOOR_FG,
        });
    }
    if (lvl.player.x, lvl.player.y, lvl.player.z) == (wx, wy, z)
        && (lvl.player.x, lvl.player.y, lvl.player.z) != (0, 0, 0)
    {
        return Some(EntityMarker {
            glyph: '@',
            fg: SPAWN_FG,
        });
    }
    if (lvl.pedestal.cx, lvl.pedestal.cy, lvl.pedestal.cz) == (wx, wy, z)
        && (lvl.pedestal.w != 0 || lvl.pedestal.h != 0 || lvl.pedestal.d != 0)
    {
        return Some(EntityMarker {
            glyph: 'X',
            fg: PEDESTAL_FG,
        });
    }
    // NOTES are point placards (like Spawn). Above the region markers so a note
    // standing inside a trigger/zone box still reads on top; the selected note is
    // highlighted.
    if let Some(nidx) = crate::entity::note_at(state, wx, wy, z) {
        let fg = if state.selected_note == Some(nidx) {
            NOTE_SEL_FG
        } else {
            NOTE_FG
        };
        return Some(EntityMarker {
            glyph: '\u{266A}',
            fg,
        }); // ♪ placard
    }
    // PARTS are voxel volumes; their ANCHOR cell shows a `▣` chartreuse marker so a
    // placed part is visible. Above the region markers (a point anchor on a trigger/
    // zone box still reads on top). The unselected marker uses the part's own
    // `color_rgb` so author-set colours read on the grid; the selected one is the
    // bright chartreuse highlight.
    if let Some(pidx) = crate::entity::part_at(state, wx, wy, z) {
        let fg = if state.selected_part == Some(pidx) {
            PART_SEL_FG
        } else {
            let c = state.level.parts[pidx].color_rgb & 0xFF_FFFF;
            // A near-black authored colour would be invisible; fall back to the
            // chartreuse identity hue so the anchor always reads.
            if c < 0x10_1010 {
                PART_FG
            } else {
                c
            }
        };
        return Some(EntityMarker {
            glyph: '\u{25A3}',
            fg,
        }); // ▣
    }
    // ZONE INPUT FIELDS are box regions (like triggers); a covered cell shows `Z`
    // in indigo (brighter when selected). Lowest priority alongside triggers so a
    // point entity on the region still reads on top.
    if let Some(zidx) = zone_cover_at(state, wx, wy, z) {
        let fg = if state.selected_zone_input == Some(zidx) {
            ZONE_SEL_FG
        } else {
            ZONE_FG
        };
        return Some(EntityMarker { glyph: 'Z', fg });
    }
    // Trigger FIELDS are box regions (not point entities); a covered cell on this
    // layer shows a `T` in violet (brighter when the trigger is selected). Lowest
    // priority so a node/door standing on the region still reads on top.
    if let Some(tidx) = trigger_cover_at(state, wx, wy, z) {
        let fg = if state.selected_trigger == Some(tidx) {
            TRIGGER_SEL_FG
        } else {
            TRIGGER_FG
        };
        return Some(EntityMarker { glyph: 'T', fg });
    }
    // WALL DECAL regions are box surfaces (like triggers/zones); a covered cell on
    // this layer shows `%` in AMBER (brighter when selected). Lowest priority — a
    // node/door/trigger standing on the region still reads on top.
    if let Some(didx) = decal_cover_at(state, wx, wy, z) {
        let fg = if state.selected_decal == Some(didx) {
            DECAL_SEL_FG
        } else {
            DECAL_FG
        };
        return Some(EntityMarker { glyph: '%', fg });
    }
    // HOLOGRAM objects are box regions (like decals/zones); a covered cell on this
    // layer shows `H` in CYAN (brighter when selected). Lowest priority — a
    // node/door/trigger standing on the region still reads on top.
    if let Some(hidx) = holo_cover_at(state, wx, wy, z) {
        let fg = if state.selected_holo == Some(hidx) {
            HOLO_SEL_FG
        } else {
            HOLO_FG
        };
        return Some(EntityMarker { glyph: 'H', fg });
    }
    // WARP PORTAL objects are PAIRS of box regions; a cell covered by EITHER region
    // on this layer shows `@` in MAGENTA-ROSE (brighter when selected), matching the
    // portal cell glyph hue. Lowest priority — any point entity standing on a region
    // still reads on top.
    if let Some(pidx) = portal_cover_at(state, wx, wy, z) {
        let fg = if state.selected_portal == Some(pidx) {
            PORTAL_SEL_FG
        } else {
            PORTAL_FG
        };
        return Some(EntityMarker { glyph: '@', fg });
    }
    // OVERLAY ZONES are box regions; a covered cell shows `S` (Scalable material) in
    // gold-olive (brighter when selected). Lowest priority alongside the other regions.
    if let Some(zidx) = crate::entity::zone_overlay_at(state, wx, wy, z) {
        let fg = if state.selected_zone == Some(zidx) {
            OVZONE_SEL_FG
        } else {
            OVZONE_FG
        };
        return Some(EntityMarker { glyph: 'S', fg });
    }
    // OVERLAY LINES are polylines; a cell on the path shows `~` in lime (brighter
    // when selected). Mirrors the pipe-path render but for the overlay line.
    if let Some(lidx) = crate::entity::line_at(state, wx, wy, z) {
        let fg = if state.selected_line == Some(lidx) {
            OVLINE_SEL_FG
        } else {
            OVLINE_FG
        };
        return Some(EntityMarker { glyph: '~', fg });
    }
    // OVERLAY INSETS are box regions; a covered cell shows `I` (inset) in coral
    // (brighter when selected). Lowest priority.
    if let Some(iidx) = crate::entity::inset_at(state, wx, wy, z) {
        let fg = if state.selected_inset == Some(iidx) {
            OVINSET_SEL_FG
        } else {
            OVINSET_FG
        };
        return Some(EntityMarker { glyph: 'I', fg });
    }
    // SOUND FIELDS: a `⊙` at the field's own placement cell, tinted by
    // archetype family (brighter when selected). Gated behind the
    // FIELDS view aid — with `fields_aid` off this arm never runs at all, so
    // `GRID_DIGEST_BEFORE` is untouched by construction (not merely by luck).
    // Only a STANDALONE Sphere field has a placement cell to mark (`field_at`'s
    // own contract — see `crate::field`'s module doc); Room/inline fields paint
    // no point marker, matching `footprint_at`'s "no footprint" rule for Room.
    if state.fields_aid {
        if let Some(fidx) = crate::field::field_at(state, wx, wy, z) {
            let base = crate::field::family_color(&state.level.resolved_fields[fidx].archetype);
            let fg = if state.selected_field == Some(fidx) {
                base
            } else {
                dim(base, 2, 3)
            };
            return Some(EntityMarker {
                glyph: '\u{2299}',
                fg,
            }); // ⊙
        }
    }
    None
}

/// Distinct top-down glyph for a non-`Full` 45° [`Shape`], conveying
/// orientation: box-drawing diagonals for the four [`Shape::WedgeXY`] walls and
/// arrow markers for the four [`Shape::RampZ`] rise directions. Returns `None`
/// for [`Shape::Full`] (use the plain cell-kind glyph). The wedge glyph points
/// its filled corner at the SOLID triangle (matching `re_core`'s solid-half
/// inequalities); the ramp arrow points UPHILL (the rise direction).
fn shape_glyph(shape: Shape) -> Option<char> {
    Some(match shape {
        Shape::Full => return None,
        // Box-drawing solid triangles: filled corner = solid triangle quadrant.
        Shape::WedgeXY(DiagXY::NE) => '\u{25E5}', // ◥ solid NE corner
        Shape::WedgeXY(DiagXY::NW) => '\u{25E4}', // ◤ solid NW corner
        Shape::WedgeXY(DiagXY::SE) => '\u{25E2}', // ◢ solid SE corner
        Shape::WedgeXY(DiagXY::SW) => '\u{25E3}', // ◣ solid SW corner
        // Ramp arrows point uphill (the rise direction). Screen convention:
        // +X = east (▶), -X = west (◀), +Y = south (▼), -Y = north (▲).
        Shape::RampZ(RampDir::PosX) => '\u{25B6}', // ▶
        Shape::RampZ(RampDir::NegX) => '\u{25C0}', // ◀
        Shape::RampZ(RampDir::PosY) => '\u{25BC}', // ▼
        Shape::RampZ(RampDir::NegY) => '\u{25B2}', // ▲
    })
}

/// Foreground colour for a 45° shape glyph — a saturated accent that stands out
/// from the plain wall greys so shaped cells are obvious in the top-down.
fn shape_fg(shape: Shape) -> u32 {
    match shape {
        Shape::Full => fg_for(CellKind::WallSolid),
        // Wedges in warm amber, ramps in cyan — two families, two hues.
        Shape::WedgeXY(_) => 0x00_FFB030,
        Shape::RampZ(_) => 0x00_30D0FF,
    }
}

// Trigger fields read in SPRING-GREEN, not violet — violet is the node-OUTPUT
// hue, and the two were easily confused. Spring-green is well clear of the green
// SPAWN marker (Spawn uses `@`, Trigger uses `T`, and the hue is yellower).
const TRIGGER_FG: u32 = 0x00_60FFA0; // spring-green trigger-field outline
const TRIGGER_SEL_FG: u32 = 0x00_B0FFD0; // brighter when selected

// ZoneInput regions read in INDIGO — a blue-violet well clear of the spring-green
// trigger, the node teal/cyan/violet, the steel-blue pipe, and the spawn green.
// (Node-output violet is pinker/redder; this is a cooler, bluer indigo.)
const ZONE_FG: u32 = 0x00_6070FF; // indigo zone-input region outline
const ZONE_SEL_FG: u32 = 0x00_A0B0FF; // brighter indigo when selected
                                      // Notes read in ROSE — a warm pink point marker distinct from the red door, the
                                      // gold pedestal, the orange emitter, and the magenta vox (rose is softer / pinker
                                      // than magenta and has a red bias the cool object hues lack).
const NOTE_FG: u32 = 0x00_FF80B0; // rose note placard marker
const NOTE_SEL_FG: u32 = 0x00_FFB0D0; // brighter rose when selected
                                      // Wall decals read in AMBER — a warm orange distinct from every other entity hue
                                      // (the indigo zone, spring-green trigger, rose note, gold pedestal, red door, and
                                      // the cooler node/pipe blues). Decals are surface textures, so a warm "painted on
                                      // the wall" hue suits them.
const DECAL_FG: u32 = 0x00_FF_A0_00; // amber wall-decal region marker
const DECAL_SEL_FG: u32 = 0x00_FF_C8_60; // brighter amber when selected
                                         // Hologram objects read in CYAN — matching the see-through `H` cell hue
                                         // (`fg_for(Hologram)`), a cool blue-cyan well clear of the warm amber decal, the
                                         // indigo zone, the spring-green trigger, and the rose note.
const HOLO_FG: u32 = 0x00_44_CC_FF; // cyan hologram region marker
const HOLO_SEL_FG: u32 = 0x00_A0_E8_FF; // brighter cyan when selected
                                        // Warp portals read in MAGENTA-ROSE — matching the see-through `@` portal cell hue
                                        // (`CellKind::Portal => 0x00_FF44AA` in the grid-glyph colour map), a hot pink well
                                        // clear of the cyan hologram, the amber decal, the indigo zone, the spring-green
                                        // trigger, and the rose note (which is softer / pinker and uses a different glyph).
const PORTAL_FG: u32 = 0x00_FF_44_AA; // magenta-rose portal region marker
const PORTAL_SEL_FG: u32 = 0x00_FF_A0_D0; // brighter magenta-rose when selected
                                          // Overlay ZONES read in DEEP GOLD-OLIVE — a warm material-zone hue distinct from
                                          // the cyan hologram, magenta portal, amber decal, and indigo zone-input. Glyph `S`
                                          // (Scalable material).
const OVZONE_FG: u32 = 0x00_C0_A0_30; // gold-olive material-zone region
const OVZONE_SEL_FG: u32 = 0x00_FF_E0_70; // brighter when selected
                                          // Overlay LINES read in BRIGHT LIME — a crisp scalable-line hue, glyph `~`.
const OVLINE_FG: u32 = 0x00_80_FF_30; // lime scalable-line path
const OVLINE_SEL_FG: u32 = 0x00_C0_FF_80; // brighter when selected
                                          // Overlay INSETS read in HOT CORAL — a "врезка" override hue distinct from the
                                          // others, glyph `I`.
const OVINSET_FG: u32 = 0x00_FF_60_50; // coral inset region
const OVINSET_SEL_FG: u32 = 0x00_FF_A0_90; // brighter when selected

/// Read a trigger's box region corners `((minx,miny,minz),(maxx,maxy,maxz))`. A
/// non-Box region (future analytic shape) reports an origin point so the painter
/// never panics.
fn trigger_box_corners(t: &Trigger) -> ((i32, i32, i32), (i32, i32, i32)) {
    match t.region {
        Region::Box { min, max } => min_max(min, max),
        _ => ((0, 0, 0), (0, 0, 0)),
    }
}

/// Normalise two corners to `(min, max)` per axis.
fn min_max(a: (i32, i32, i32), b: (i32, i32, i32)) -> ((i32, i32, i32), (i32, i32, i32)) {
    (
        (a.0.min(b.0), a.1.min(b.1), a.2.min(b.2)),
        (a.0.max(b.0), a.1.max(b.1), a.2.max(b.2)),
    )
}

/// Index of the trigger whose region covers cell `(x, y, z)`, if any, with the
/// SELECTED trigger preferred when several overlap (so the highlighted box stays
/// the one being edited).
fn trigger_cover_at(state: &EditorState, x: i32, y: i32, z: i32) -> Option<usize> {
    if let Some(sel) = state.selected_trigger {
        if state
            .level
            .triggers
            .get(sel)
            .is_some_and(|t| t.contains(x, y, z))
        {
            return Some(sel);
        }
    }
    state
        .level
        .triggers
        .iter()
        .position(|t| t.contains(x, y, z))
}

/// Index of the zone input whose box covers cell `(x, y, z)`, if any, with the
/// SELECTED zone preferred when several overlap. Mirrors [`trigger_cover_at`].
fn zone_cover_at(state: &EditorState, x: i32, y: i32, z: i32) -> Option<usize> {
    if let Some(sel) = state.selected_zone_input {
        if state
            .level
            .zone_inputs
            .get(sel)
            .is_some_and(|zi| zi.contains(x, y, z))
        {
            return Some(sel);
        }
    }
    state
        .level
        .zone_inputs
        .iter()
        .position(|zi| zi.contains(x, y, z))
}

/// Index of the wall decal whose box covers cell `(x, y, z)`, if any, with the
/// SELECTED decal preferred when several overlap. Mirrors [`zone_cover_at`].
fn decal_cover_at(state: &EditorState, x: i32, y: i32, z: i32) -> Option<usize> {
    let covers = |d: &re_level::WallDecal| {
        x >= d.x0.min(d.x1)
            && x <= d.x0.max(d.x1)
            && y >= d.y0.min(d.y1)
            && y <= d.y0.max(d.y1)
            && z >= d.z0.min(d.z1)
            && z <= d.z0.max(d.z1)
    };
    if let Some(sel) = state.selected_decal {
        if state.level.decals.get(sel).is_some_and(covers) {
            return Some(sel);
        }
    }
    state.level.decals.iter().position(covers)
}

/// Index of the hologram whose box covers cell `(x, y, z)`, if any, with the
/// SELECTED hologram preferred when several overlap. Mirrors [`decal_cover_at`].
fn holo_cover_at(state: &EditorState, x: i32, y: i32, z: i32) -> Option<usize> {
    let covers = |h: &re_level::LevelHolo| {
        x >= h.x0.min(h.x1)
            && x <= h.x0.max(h.x1)
            && y >= h.y0.min(h.y1)
            && y <= h.y0.max(h.y1)
            && z >= h.z0.min(h.z1)
            && z <= h.z0.max(h.z1)
    };
    if let Some(sel) = state.selected_holo {
        if state.level.holos.get(sel).is_some_and(covers) {
            return Some(sel);
        }
    }
    state.level.holos.iter().position(covers)
}

/// Index of the portal whose EITHER region covers cell `(x, y, z)`, if any, with
/// the SELECTED portal preferred when several overlap. So a placed portal shows its
/// `@` marker over BOTH region A and region B in the top-down grid. Mirrors
/// [`holo_cover_at`] but tests two boxes per entity.
fn portal_cover_at(state: &EditorState, x: i32, y: i32, z: i32) -> Option<usize> {
    let covers = |p: &re_level::LevelPortal| {
        let in_box = |lo: (i32, i32, i32), hi: (i32, i32, i32)| {
            x >= lo.0.min(hi.0)
                && x <= lo.0.max(hi.0)
                && y >= lo.1.min(hi.1)
                && y <= lo.1.max(hi.1)
                && z >= lo.2.min(hi.2)
                && z <= lo.2.max(hi.2)
        };
        in_box((p.ax0, p.ay0, p.az0), (p.ax1, p.ay1, p.az1))
            || in_box((p.bx0, p.by0, p.bz0), (p.bx1, p.by1, p.bz1))
    };
    if let Some(sel) = state.selected_portal {
        if state.level.portals.get(sel).is_some_and(covers) {
            return Some(sel);
        }
    }
    state.level.portals.iter().position(covers)
}

/// Bright accent for the PREVIEW-CAMERA arrow on the top-down grid — a saturated
/// gold that stands out from geometry, entity markers, and the cursor highlight.
const CAM_FG: u32 = 0x00_FFE040;
/// Off-layer preview-camera tint (camera is on a different z than the one viewed):
/// a dimmer cyan with a diamond glyph so the user reads it as "above/below here".
const CAM_OFFLAYER_FG: u32 = 0x00_50A0C0;

/// Map a camera `yaw` (radians) to one of 8 top-down arrow glyphs pointing the way
/// the camera looks in SCREEN space. Facing direction is `(sin yaw, cos yaw)`
/// (`re_world::CameraPose::dir`); the grid draws row 0 at top with +Y increasing
/// DOWNWARD, so the screen convention is `+x = → (east)`, `+y = ↓ (south)`. Thus
/// yaw 0 → dir (0, +1) → ↓. Pure; unit-tested.
fn cam_arrow_glyph(yaw: f32) -> char {
    use std::f32::consts::TAU;
    // Screen-space facing vector: x to the right, y downward.
    let sx = yaw.sin();
    let sy = yaw.cos();
    // Angle of the screen vector measured CCW from +x (east). atan2 takes (y, x);
    // we negate y because screen-y points down, so the angle is a normal math angle
    // where 0 = east, π/2 = north (up). Quantize to 8 sectors of 45°.
    let ang = (-sy).atan2(sx).rem_euclid(TAU); // 0 = →, increasing CCW
    let sector = ((ang / (TAU / 8.0)).round() as i32).rem_euclid(8);
    // 0:E 1:NE 2:N 3:NW 4:W 5:SW 6:S 7:SE  (CCW from east)
    match sector {
        0 => '\u{2192}', // →
        1 => '\u{2197}', // ↗
        2 => '\u{2191}', // ↑
        3 => '\u{2196}', // ↖
        4 => '\u{2190}', // ←
        5 => '\u{2199}', // ↙
        6 => '\u{2193}', // ↓
        _ => '\u{2198}', // ↘ (sector 7)
    }
}

const CURSOR_BG: u32 = 0x00_4444AA;
/// High-contrast foreground forced onto the cursor cell + its `[ ]` brackets so
/// the edit cursor reads on any underlying glyph and in 256-colour terminals.
const CURSOR_HI_FG: u32 = 0x00_FFFFFF;
/// Faint colour for rulers, chunk lines, and below-layer ghosts.
const DIM_FG: u32 = 0x00_445566;
const RULER_FG: u32 = 0x00_667788;
const CHUNK_BG: u32 = 0x00_101820;
/// Width reserved on the LEFT of the grid pane for the row-number ruler.
pub const RULER_W: usize = 3;
/// Height reserved on the TOP of the grid pane for the column-number ruler.
pub const RULER_H: usize = 1;

/// Row offset of the Z-slider within the side panel (from the panel's `y0`).
/// Shared by the painter ([`draw_z_slider`]) and the mouse hit-test
/// (`mouse::apply_mouse`) so a click on the bar maps to the cell it drew.
pub const Z_SLIDER_ROW: usize = 4;
/// Total width (cells) of the Z-slider, including the `[` and `]` end caps.
pub const Z_SLIDER_W: usize = 16;
/// Faint crosshair-tint bg painted through the cursor's row/column in the grid.
const CROSSHAIR_BG: u32 = 0x00_1A2238;
/// Highlight fg for the cursor-coordinate tick on the rulers.
const CURSOR_TICK_FG: u32 = 0x00_FFD24A;
/// Very faint fg for the per-cell sub-grid drawn at high zoom.
const SUBGRID_FG: u32 = 0x00_2A3340;

fn glyph_for(kind: CellKind) -> char {
    match kind {
        CellKind::Empty => '.',
        CellKind::WallSolid => '#',
        CellKind::WallTrace => '/',
        CellKind::WallMatrix => '%',
        CellKind::Floor => '_',
        CellKind::Ceiling => '^',
        CellKind::Object => 'O',
        CellKind::Hologram => 'H',
        // `Portal` is a non-euclidean warp-portal face (present in re_world but
        // not in the plan's CellKind list); shown as `@` so the match is total.
        CellKind::Portal => '@',
    }
}

/// Per-cell base foreground so kinds read apart at a glance.
fn fg_for(kind: CellKind) -> u32 {
    match kind {
        CellKind::Empty => 0x00_303840,
        CellKind::WallSolid => 0x00_C8C8C8,
        CellKind::WallTrace => 0x00_55CC99,
        CellKind::WallMatrix => 0x00_33CC55,
        CellKind::Floor => 0x00_6688AA,
        CellKind::Ceiling => 0x00_8866AA,
        CellKind::Object => 0x00_FFCC44,
        CellKind::Hologram => 0x00_44CCFF,
        CellKind::Portal => 0x00_FF44AA,
    }
}

/// Scale an `0x00RRGGBB` colour by `num/den` (per channel) for ghost-dimming.
fn dim(color: u32, num: u32, den: u32) -> u32 {
    let r = ((color >> 16) & 0xFF) * num / den;
    let g = ((color >> 8) & 0xFF) * num / den;
    let b = (color & 0xFF) * num / den;
    (r << 16) | (g << 8) | b
}

/// Resolve the glyph + fg for a world cell at the active z, applying the height
/// overlay: if the active layer is empty, fall through to the nearest non-empty
/// layer BELOW it, dimmed proportionally to how far down it sits (so stacked
/// structure reads from the top-down). Returns `(glyph, fg, is_ghost)`.
fn resolve_cell(state: &EditorState, wx: usize, wy: usize) -> (char, u32, bool) {
    let z = state.z as i32;
    let active = state.map.cell_at(wx as i32, wy as i32, z);
    if active != CellKind::Empty {
        // A non-Full 45° shape overrides the kind glyph with an orientation
        // marker so wedges/ramps read distinctly from plain solids.
        let shape = state.map.shape_at(wx as i32, wy as i32, z);
        if let Some(g) = shape_glyph(shape) {
            return (g, shape_fg(shape), false);
        }
        // A painted MATERIAL id tints the cell with the material's fg + glyph so
        // material painting is visible in the top-down view.
        let id = state.map.side_data_at(wx as i32, wy as i32, z);
        if let Some((g, fg)) = crate::material::material_tint(state, id) {
            return (g, fg, false);
        }
        return (glyph_for(active), fg_for(active), false);
    }
    // Active layer is empty, so a ghost may show through. Nothing below this
    // point can ever paint over a live cell.
    //
    // ONION SKIN (view aid) takes first refusal: when it is ON it claims the
    // NEAREST occupied layer within +/- depth, above as well as below, and paints
    // it in `viewaids::ghost_fg` — dimmed by distance and tinted warm (below) or
    // cool (above) so the direction reads off the colour. When it is OFF
    // `onion_ghost` returns `None` on its first line and the original height
    // overlay below runs completely unchanged.
    if let Some(g) = crate::viewaids::onion_ghost(&state.map, wx as i32, wy as i32, z, state.onion)
    {
        return (
            glyph_for(g.kind),
            crate::viewaids::ghost_fg(fg_for(g.kind), g.dz),
            true,
        );
    }
    // HEIGHT OVERLAY (pre-existing): search downward, unbounded, for the nearest
    // cell below. Onion skin deliberately does NOT replace this — with the aid on
    // at depth 1 a cell four layers down still ghosts exactly as it always did, so
    // enabling the aid never REMOVES information from the pane.
    let mut depth = 1;
    let mut zz = z - 1;
    while zz >= 0 {
        let k = state.map.cell_at(wx as i32, wy as i32, zz);
        if k != CellKind::Empty {
            // Dim more steeply the deeper the cell: 1 below → 1/2, etc.
            let fg = dim(fg_for(k), 1, (depth + 1) as u32);
            return (glyph_for(k), fg, true);
        }
        depth += 1;
        zz -= 1;
    }
    (glyph_for(CellKind::Empty), fg_for(CellKind::Empty), false)
}

/// Draw the current z-slice as a grid filling the rect (x0,y0,w,h) with integer
/// zoom, pan, world-coordinate rulers, chunk guide lines, height overlay, and a
/// block-scaled cursor highlight. Cells beyond the pane are clipped.
pub fn draw_grid(
    state: &EditorState,
    fb: &mut Framebuffer,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
) {
    let (sx, sy, _sz) = state.map.size();
    let zoom = state.zoom.max(1) as usize;

    // Reserve rulers on the top/left; the remaining area holds the cell blocks.
    let cells_x0 = x0 + RULER_W;
    let cells_y0 = y0 + RULER_H;
    let cells_w = w.saturating_sub(RULER_W);
    let cells_h = h.saturating_sub(RULER_H);

    let (pan_x, pan_y) = state.pan;
    let vis_x = view::visible_cells(cells_w, state.zoom).min(sx.saturating_sub(pan_x));
    let vis_y = view::visible_cells(cells_h, state.zoom).min(sy.saturating_sub(pan_y));

    // --- cell blocks ---
    for vy in 0..vis_y {
        let wy = pan_y + vy;
        for vx in 0..vis_x {
            let wx = pan_x + vx;
            let (mut glyph, mut fg, _ghost) = resolve_cell(state, wx, wy);
            // ENTITY MARKERS: draw on the active layer in full colour, and on a
            // layer below faintly (so off-layer entities still read). An entity
            // overrides the geometry glyph at this cell's top-left. Gated by
            // `show_objects` so the author can toggle them off to read the raw
            // geometry; the camera arrow + cursor below are NOT gated.
            let z = state.z as i32;
            if state.show_objects {
                if let Some(m) = entity_marker_at(state, wx as i32, wy as i32, z) {
                    glyph = m.glyph;
                    fg = m.fg;
                } else if let Some(pm) = pipe_marker_at(state, wx as i32, wy as i32, z) {
                    // PIPE PATHS render below entities (wiring) but above geometry, so
                    // a routed pipe is visible on its active layer.
                    glyph = pm.glyph;
                    fg = pm.fg;
                } else {
                    // Look one layer below for a faint off-layer marker (entity, then
                    // pipe-path).
                    for zz in (0..z).rev() {
                        if let Some(m) = entity_marker_at(state, wx as i32, wy as i32, zz) {
                            let depth = (z - zz) as u32;
                            glyph = m.glyph;
                            fg = dim(m.fg, 1, depth + 1);
                            break;
                        }
                        if let Some(pm) = pipe_marker_at(state, wx as i32, wy as i32, zz) {
                            let depth = (z - zz) as u32;
                            glyph = pm.glyph;
                            fg = dim(pm.fg, 1, depth + 1);
                            break;
                        }
                    }
                }
            }
            let is_cursor = (wx, wy) == state.cursor;
            // Faint chunk-line backdrop on chunk-aligned world cells.
            let on_chunk = view::is_chunk_line(wx) || view::is_chunk_line(wy);
            // CROSSHAIR: faint tint across the cursor's row + column (but never on
            // the cursor cell itself — its CURSOR_BG wins — and chunk lines keep
            // their own backdrop so the orientation guides stay readable).
            let on_crosshair = !is_cursor && (wx == state.cursor.0 || wy == state.cursor.1);
            // SOUND FIELDS view aid: a radius/inner/outer-ring wash, colour
            // by archetype family, halved when occluded. `cell_tint` itself
            // returns `None` unconditionally while `state.fields_aid` is off (its
            // own first line), so this call — and therefore `GRID_DIGEST_BEFORE`
            // — is unchanged with the aid at its default OFF.
            let field_tint = crate::field::cell_tint(state, wx as i32, wy as i32, z);
            for by in 0..zoom {
                for bx in 0..zoom {
                    let px = cells_x0 + vx * zoom + bx;
                    let py = cells_y0 + vy * zoom + by;
                    if px >= x0 + w || py >= y0 + h {
                        continue;
                    }
                    let top_left = bx == 0 && by == 0;
                    // Glyph at the block's top-left; the rest of the block is fill,
                    // upgraded to a faint sub-cell grid at high zoom so the extra
                    // block space reads as increased drawing precision.
                    let (glyph_here, glyph_fg) = if top_left {
                        (glyph, fg)
                    } else if zoom >= 2 {
                        match subcell_glyph(bx, by, zoom) {
                            Some(g) => (g, SUBGRID_FG),
                            None => (' ', fg),
                        }
                    } else {
                        (' ', fg)
                    };
                    let bg = if is_cursor {
                        CURSOR_BG
                    } else if let Some(t) = field_tint {
                        t
                    } else if on_chunk {
                        CHUNK_BG
                    } else if on_crosshair {
                        CROSSHAIR_BG
                    } else {
                        0
                    };
                    // CURSOR CELL: force a high-contrast foreground so the glyph
                    // reads on ANY underlying cell (empty/floor/bright wall) and in
                    // 256-colour terminals where the muted CURSOR_BG alone vanishes.
                    let glyph_fg = if is_cursor { CURSOR_HI_FG } else { glyph_fg };
                    fb.set(
                        px,
                        py,
                        Cell {
                            glyph: glyph_here as u32,
                            fg: glyph_fg,
                            bg,
                            ..Cell::default()
                        },
                    );
                }
            }
        }
    }

    // CURSOR BRACKET: draw a bright `[` `]` hugging the cursor block's left/right
    // edges (and at zoom>=2 a box outline), the way the direction-picker brackets
    // its cube cursor. This makes the edit cursor unmistakable even when its cell
    // is a bright painted glyph or the terminal can't show the CURSOR_BG tint.
    let (curx, cury) = state.cursor;
    if curx >= pan_x && curx < pan_x + vis_x && cury >= pan_y && cury < pan_y + vis_y {
        let bx0 = cells_x0 + (curx - pan_x) * zoom;
        let by0 = cells_y0 + (cury - pan_y) * zoom;
        let bx1 = bx0 + zoom - 1; // inclusive right edge of the block
        let in_pane = |px: usize, py: usize| px < x0 + w && py < y0 + h;
        // Left + right edge brackets on the cursor's centre row.
        let mid_y = by0 + (zoom - 1) / 2;
        if bx0 > cells_x0 && in_pane(bx0 - 1, mid_y) {
            fb.set(bx0 - 1, mid_y, glyph_cell('[', CURSOR_HI_FG, CURSOR_BG));
        }
        if in_pane(bx1 + 1, mid_y) {
            fb.set(bx1 + 1, mid_y, glyph_cell(']', CURSOR_HI_FG, CURSOR_BG));
        }
    }

    // PREVIEW-CAMERA ARROW: show WHERE the fly-through camera sits and WHICH WAY
    // it faces, as a bright directional glyph at its grid cell. Drawn AFTER the
    // cursor bracket so it overlays (the camera marker wins even on the cursor
    // cell — it forces its own bright fg). When the camera is on a different
    // z-layer than the one being viewed it is dimmed and drawn as a `◇` diamond so
    // the user can tell it is above/below the current slice.
    {
        let cam = &state.cam;
        // Floor the continuous camera position to its world cell. Negative or
        // out-of-grid positions are simply skipped (no marker).
        if cam.pos.x >= 0.0 && cam.pos.y >= 0.0 {
            let camx = cam.pos.x.floor() as usize;
            let camy = cam.pos.y.floor() as usize;
            let camz = cam.pos.z.floor();
            let on_layer = camz >= 0.0 && camz as usize == state.z;
            if camx >= pan_x && camx < pan_x + vis_x && camy >= pan_y && camy < pan_y + vis_y {
                let (glyph, fg) = if on_layer {
                    (cam_arrow_glyph(cam.yaw), CAM_FG)
                } else {
                    ('\u{25C7}', CAM_OFFLAYER_FG) // ◇ off-layer diamond
                };
                let px = cells_x0 + (camx - pan_x) * zoom;
                let py = cells_y0 + (camy - pan_y) * zoom;
                if px < x0 + w && py < y0 + h {
                    // Preserve whatever bg already sits under the camera cell (the
                    // cursor highlight, a chunk line, etc.) so the marker overlays
                    // without erasing the cursor's box.
                    let bg = fb.get(px, py).map(|c| c.bg).unwrap_or(0);
                    fb.set(px, py, glyph_cell(glyph, fg, bg));
                }
            }
        }
    }

    // --- column ruler (top): world-x labels at zoom-aware ticks ---
    for (world, off) in view::ruler_ticks(pan_x, cells_w, state.zoom, sx) {
        let px = cells_x0 + off * zoom;
        if px < x0 + w {
            draw_str(fb, px, y0, &ruler_label(world), RULER_FG, 0);
        }
    }
    // --- row ruler (left): world-y labels at zoom-aware ticks ---
    for (world, off) in view::ruler_ticks(pan_y, cells_h, state.zoom, sy) {
        let py = cells_y0 + off * zoom;
        if py < y0 + h {
            draw_str(fb, x0, py, &ruler_label(world), RULER_FG, 0);
        }
    }
    // AXIS CURSOR MARKERS: highlight the cursor's exact x on the top ruler and
    // its exact y on the left ruler, so the cursor coordinate is readable off the
    // axes even between chunk ticks. Only when the cursor is in the visible window.
    let (cx, cy) = state.cursor;
    if cx >= pan_x && cx < pan_x + vis_x {
        let px = cells_x0 + (cx - pan_x) * zoom;
        if px < x0 + w {
            draw_str(fb, px, y0, &ruler_label(cx), CURSOR_TICK_FG, CHUNK_BG);
        }
    }
    if cy >= pan_y && cy < pan_y + vis_y {
        let py = cells_y0 + (cy - pan_y) * zoom;
        if py < y0 + h {
            draw_str(fb, x0, py, &ruler_label(cy), CURSOR_TICK_FG, CHUNK_BG);
        }
    }
    // Origin corner marker keeps the two axes legible.
    draw_str(fb, x0, y0, "y\\x", DIM_FG, 0);
}

/// Sub-cell guide glyph for a non-top-left cell of a zoomed block, or `None` for
/// a plain blank fill. At zoom 2 the block's interior gets a faint dot at the
/// bottom-right; at zoom >= 3 every interior cell gets a faint `·`, with the
/// block's centre marked `+`, so a bigger block reads as finer drawing
/// precision. Pure; `bx`/`by` are 1..zoom-1 here (top-left handled by caller).
fn subcell_glyph(bx: usize, by: usize, zoom: usize) -> Option<char> {
    if zoom < 2 {
        return None;
    }
    if zoom == 2 {
        // One faint marker in the otherwise-blank 2x2 block.
        return if bx == 1 && by == 1 {
            Some('\u{00B7}')
        } else {
            None
        };
    }
    // zoom >= 3: a faint sub-grid; centre gets a cross.
    let mid = zoom / 2;
    if bx == mid && by == mid {
        Some('+')
    } else {
        Some('\u{00B7}') // middle dot
    }
}

/// Ruler tick label — the FULL number, never truncated. Earlier code trimmed to
/// the last 3 digits (1024 → "024"), which is silently wrong; we show the whole
/// coordinate and let it use the space it needs (ticks are already spaced by
/// `view::ruler_ticks`, and `draw_str` clips at the pane edge rather than
/// corrupting the digits).
fn ruler_label(n: usize) -> String {
    n.to_string()
}

/// Draw the full-width STATUS BAR (the "in-the-moment" organ) across `rect`'s
/// single bottom row. Left-to-right it surfaces the live readouts that used to be
/// scattered through the side panel — view mode, `z {z}/{sz} ×{zoom}`, the cursor
/// cell, the collisions checkbox, undo/redo depth, the saved/unsaved flag — then
/// the transient `state.status` message and a short key hint. The READOUTS paint
/// in [`STATUS_FG`] (golden) and the single-width `│` GROUP SEPARATORS in the
/// neutral [`TOOLBOX_HEADER_FG`] grey, so the segments read apart without an
/// emoji or a double-width glyph anywhere. Works in EVERY view mode (including
/// Full3D, which has no side panel), so the readouts are always visible. Clipped
/// to the row width; no-op for an empty rect. Pure painting.
pub fn draw_status_bar(state: &EditorState, fb: &mut Framebuffer, rect: crate::layout::Rect) {
    if rect.is_empty() {
        return;
    }
    let bg = 0x00_141A26; // a faint bar backdrop so the row reads as one strip
    let row = rect.y;
    // Paint the bar backdrop first so the text reads as a contiguous strip.
    for cx in 0..rect.w {
        fb.set(
            rect.x + cx,
            row,
            Cell {
                glyph: ' ' as u32,
                fg: STATUS_FG,
                bg,
                ..Cell::default()
            },
        );
    }

    let (_sx, _sy, sz) = state.map.size();
    let cursor_read = cursor_coord_readout(state);
    // SOUND FIELDS cursor readout: the top-3 audible fields + the
    // resulting AMB LP/WD/VOL at the cursor's cell, appended to the `@`
    // segment ONLY while the fields view aid is on — off by default, so the
    // status bar text (and any digest pinned on it) is unaffected.
    let field_note = if state.fields_aid {
        let listener = [
            state.cursor.0 as f32 + 0.5,
            state.cursor.1 as f32 + 0.5,
            state.z as f32 + 0.5,
        ];
        format!("  {}", crate::field::cursor_readout(&state.level, listener))
    } else {
        String::new()
    };
    let undo_n = state.history.undo_len();
    let redo_n = state.history.redo_len();
    // SAVED / UNSAVED flag, derived from the same fields the panel used: a dirty
    // doc is "unsaved", a never-saved doc with no edits is "unsaved" too (no file
    // backs it), and a clean saved doc is "saved".
    let saved = if state.dirty || state.path.is_none() {
        "unsaved"
    } else {
        "saved"
    };
    // The collisions checkbox is reused verbatim from the panel (single source of
    // truth for the label + checked glyph).
    let collisions = collisions_checkbox(state);

    // The readout SEGMENTS, drawn left-to-right with a `│` rule between each so the
    // groups read apart. The trailing key HINT is a secondary affordance, so it
    // gets the muted toolbox grey; the live readouts get the golden STATUS_FG.
    let segments: [(String, u32); 8] = [
        (format!("VIEW {:?}", state.view_mode), STATUS_FG),
        (
            format!(
                "z {}/{} \u{00D7}{}",
                state.z,
                sz.saturating_sub(1),
                state.zoom
            ),
            STATUS_FG,
        ), // ×
        (format!("@ {cursor_read}{field_note}"), STATUS_FG),
        (collisions, STATUS_FG),
        (format!("undo {undo_n} redo {redo_n}"), STATUS_FG),
        (saved.to_string(), STATUS_FG),
        (state.status.clone(), STATUS_FG),
        // MODE-AWARE key hint: generated from the keymap table for
        // the ACTIVE input context, so the bar always shows keys that work
        // right now — and can never drift from the real bindings.
        (crate::keymap::hint_line(state), TOOLBOX_HEADER_FG),
    ];
    // Walk the row, clipping to its width; the ` │ ` group separators paint in the
    // subtle organ-divider slate so the segments read apart without an emoji.
    let mut cx = rect.x;
    let row_end = rect.x + rect.w;
    let mut put = |cx: &mut usize, text: &str, fg: u32| {
        if *cx >= row_end {
            return;
        }
        let room = row_end - *cx;
        let shown: String = text.chars().take(room).collect();
        let drawn = shown.chars().count();
        draw_str(fb, *cx, row, &shown, fg, bg);
        *cx += drawn;
    };
    for (i, (seg, fg)) in segments.iter().enumerate() {
        if i > 0 {
            put(&mut cx, " \u{2502} ", ORGAN_DIVIDER_FG); // ` │ ` organ-divider rule
        }
        put(&mut cx, seg, *fg);
    }
}

/// Draw the side panel (mode, z, zoom, brush, z-slider) into the rect starting at
/// (x0,y0). `_h` (the inspector height) is retained for call-site/signature
/// symmetry; the panel draws at fixed offsets from `y0`. The "in-the-moment"
/// readouts (view / z / cursor / collisions / undo-redo / saved / status) live in
/// the full-width [`draw_status_bar`] at the bottom of the screen, so this panel is
/// the INSPECTOR organ: brush + entity/material/vox properties + the z-slider. (The
/// thin Full3D one-line strip was retired — that mode now relies on the shared
/// bottom status bar, and its `panel` rect is empty.)
pub fn draw_panel(state: &EditorState, fb: &mut Framebuffer, x0: usize, y0: usize, _h: usize) {
    let fg = 0x00_FFFFFF;
    let bg = 0x00_000000;
    // INSPECTOR organ header — the warm-amber accent marks "where I change
    // properties". The current file basename rides on the header so the author
    // always knows what they're editing (this is identity, not an in-the-moment
    // readout, so it stays in the inspector). The Full2D exit hint lives here too.
    let file = state
        .path
        .as_deref()
        .map(|p| p.rsplit(['/', '\\']).next().unwrap_or(p))
        .unwrap_or("untitled");
    let mode_hint = if state.view_mode == crate::state::ViewMode::Full2D {
        "  (Tab=exit)"
    } else {
        ""
    };
    // Row 0 of the inspector body: the file identity (the `─ INSPECTOR ─` organ
    // label is painted on the divider row by `draw_system`). When NOTHING is
    // selected/active, a short Tips line takes the brush row instead of leaving
    // the inspector blank.
    draw_str(
        fb,
        x0,
        y0,
        &format!("INSPECTOR{mode_hint}"),
        INSPECTOR_HEADER_FG,
        bg,
    );
    // FILE identity on row 1 — rows 5+ belong to the scrollable property
    // sheet.
    draw_str(fb, x0, y0 + 1, &format!("FILE {file}"), 0x00_9FB6C9, bg);
    // A pending fill-anchor flag (an inspector-relevant "armed corner" cue; the
    // undo/redo depth itself now lives in the status bar). Row 3 (fixed furniture).
    if let Some((ax, ay, az)) = state.fill_anchor {
        draw_str(
            fb,
            x0,
            y0 + 3,
            &format!("fill A:{ax},{ay},{az}"),
            0x00_99BBDD,
            bg,
        );
    }
    // The brush line reflects: vox mode (if active), else material mode, else the
    // object selector, else the 45° shape brush, else the plain geometry brush.
    if state.vox_mode {
        draw_str(
            fb,
            x0,
            y0 + 2,
            &format!("VOX {}", crate::vox::vox_summary(state)),
            VOX_FG,
            bg,
        );
    } else if state.material_mode {
        draw_str(
            fb,
            x0,
            y0 + 2,
            &format!("MATERIAL {}", crate::material::material_summary(state)),
            0x00_60FFC0,
            bg,
        );
    } else if let Some(kind) = state.place {
        draw_str(
            fb,
            x0,
            y0 + 2,
            &format!("PLACE {}", kind.label()),
            0x00_FF80C0,
            bg,
        );
    } else if state.stair_mode {
        draw_str(
            fb,
            x0,
            y0 + 2,
            &format!("STAIR {}", crate::staircase::summary(state)),
            0x00_FFB030,
            bg,
        );
    } else if state.shape_mode {
        draw_str(
            fb,
            x0,
            y0 + 2,
            &format!("BRUSH 45* {}", shape_label(state.shape_sel)),
            0x00_FFB030,
            bg,
        );
    } else {
        draw_str(fb, x0, y0 + 2, &format!("BRUSH {:?}", state.brush), fg, bg);
    }
    // (The COLLISIONS checkbox + the transient status message both moved to the
    // full-width bottom STATUS BAR — see `draw_status_bar` — so they are visible
    // in every view mode, not just when the side panel is drawn.)

    // Z height slider gauge (horizontal): floor .. top with a knob at the
    // active layer. Width-bounded to the panel. (Clickable — see mouse.rs.)
    draw_z_slider(state, fb, x0, y0 + Z_SLIDER_ROW, Z_SLIDER_W);

    // PROPERTY SHEET: the typed, scrollable row list — every field of the
    // selected object/material, the focused row marked + kept visible by the
    // shared scroll math, ▲/▼ overflow indicators at the right edge. The rows,
    // the scroll, and the click hit-test all come from the same SSOT
    // (`inspector_rows` / `inspector_viewport` / `settings_scroll`). The organ
    // rect is re-derived from the live layout — the exact rect `draw_system`
    // passes (`inspector_subrect` of the layout panel), so painter and hit-test
    // stay aligned.
    let l = state.layout_for(fb.cols(), fb.rows());
    let panel = inspector_subrect_eff(state, l.panel, state.panels_eff(fb.cols()));
    draw_inspector_rows(state, fb, panel);

    // (The old `? help  O open  s save` discoverability line was REMOVED: it sits in
    // the inspector's columns on the same absolute row as the preview pane's bottom
    // hint at common terminal heights, so the two read as one garbled run. The same
    // affordances are already covered by the top toolbar's Open/Save/Help buttons
    // and the bottom status bar's `?=help`, so the inspector line was pure
    // duplication — dropping it leaves the preview hint as the only hint on that row.)
}

/// The bottom sub-strip of the INSPECTOR organ reserved for the optional
/// ELEVATION cross-section, or `None` when the organ is too short to spare it
/// after the action buttons. Shared single source of truth so `draw_system`
/// paints the strip exactly where it fits without overlapping the buttons. The
/// elevation needs at least [`ELEV_MIN_H`] rows BELOW the last button row.
pub fn inspector_elevation_rect(
    state: &EditorState,
    inspector: crate::layout::Rect,
) -> Option<crate::layout::Rect> {
    if inspector.is_empty() {
        return None;
    }
    // Lowest row used by the inspector content/buttons (exclusive top of any free
    // tail). Buttons sit below the scrollable property-sheet
    // viewport ([`inspector_viewport`]); when none are drawn (panel too short),
    // fall back to the viewport's own button-anchor row so the strip still
    // lands below the sheet.
    let last_used = panel_buttons(state, inspector)
        .iter()
        .map(|&(_, _, _, r)| r)
        .max()
        .unwrap_or_else(|| inspector_viewport(state, inspector).1.saturating_sub(1));
    let strip_y = last_used + 2; // one blank separator row
    let bottom = inspector.y + inspector.h; // exclusive
    if strip_y + ELEV_MIN_H > bottom {
        return None;
    }
    Some(crate::layout::Rect {
        x: inspector.x,
        y: strip_y,
        w: inspector.w,
        h: bottom - strip_y,
    })
}

/// Minimum rows the optional inspector ELEVATION strip needs (header + readout +
/// a couple of layer rows).
const ELEV_MIN_H: usize = 5;

/// The COLLISIONS checkbox label reflecting the live collision state: `[x]
/// Collisions` when collisions are ON (i.e. `noclip == false`), `[ ] Collisions`
/// when OFF. Single source of truth for the panel readout and the Full3D strip.
/// Pure; unit-tested.
fn collisions_checkbox(state: &EditorState) -> String {
    let mark = if state.noclip { ' ' } else { 'x' };
    format!("[{mark}] Collisions")
}

/// The cursor-coordinate readout: the HONEST integer cell coordinate `"x,y z=Z"`.
/// The cursor is cell-quantized, so there is no sub-cell precision to report — the
/// old `x.0/q` form invented fractional digits that never changed. Pure; used by
/// the panel and unit-tested.
fn cursor_coord_readout(state: &EditorState) -> String {
    let (cx, cy) = state.cursor;
    format!("{cx},{cy} z={}", state.z)
}

/// Full cheat-sheet text — the toggleable `?` popup. GENERATED from the single
/// keymap table (`crate::keymap`): the help body, the status-bar
/// hint segment, and the `:` command palette all render from [`crate::keymap::KEYMAP`],
/// so a binding added to the table appears everywhere with no other edit.
fn help_lines() -> Vec<String> {
    crate::keymap::help_lines()
}

/// Draw the toggleable HELP popup: a centered bordered box over the SCROLLED,
/// FILTERED cheat-sheet ([`crate::keymap::help_lines_filtered`]), with a filter
/// band, a position footer, and a scrollbar column on the right border. Topmost
/// overlay (drawn last by `draw_system`). No-op if the framebuffer is too small.
///
/// Every row painted here comes from [`help_layout`] — the same struct the
/// `[X]` hit-test reads — so nothing in the box is positioned twice.
pub fn draw_help(state: &EditorState, fb: &mut Framebuffer, w: usize, h: usize) {
    let Some(lay) = help_layout(state, w, h) else {
        return;
    };
    let lines = crate::keymap::help_lines_filtered(&state.help_filter);
    let bg = 0x00_0C1018;
    let border = 0x00_77AADD;
    let fg = 0x00_C8D4E0;
    let dim = 0x00_7788AA;
    let (x0, y0, pw, ph) = (lay.x0, lay.y0, lay.pw, lay.ph);
    for ry in 0..ph {
        for rx in 0..pw {
            let edge = rx == 0 || ry == 0 || rx == pw - 1 || ry == ph - 1;
            fb.set(
                x0 + rx,
                y0 + ry,
                Cell {
                    glyph: if edge { '#' as u32 } else { ' ' as u32 },
                    fg: if edge { border } else { fg },
                    bg,
                    ..Cell::default()
                },
            );
        }
    }
    draw_str(fb, lay.text_x, lay.title_y, "HELP — controls", border, bg);
    // Clickable [X] close button on the top border (shared geom).
    let (bx_y, bx_s, _bx_e) = modal_close_button(x0, y0, pw);
    draw_str(fb, bx_s, bx_y, "[X]", 0x00_FFC0C0, bg);
    // FILTER band: the live query + how many rows survive it.
    let filt: String = help_filter_text(state).chars().take(lay.text_w).collect();
    let filt_fg = if state.help_filter.is_empty() {
        dim
    } else {
        0x00_FFE060
    };
    draw_str(fb, lay.text_x, lay.filter_y, &filt, filt_fg, bg);
    // BODY: the window `[scroll, scroll + body_h)` of the filtered sheet.
    for i in 0..lay.body_h {
        let Some(line) = lines.get(lay.scroll + i) else {
            break;
        };
        let shown: String = line.chars().take(lay.text_w).collect();
        draw_str(fb, lay.text_x, lay.body_y + i, &shown, fg, bg);
    }
    // SCROLLBAR: a thumb on the right border spanning the visible fraction, so
    // the "there is more below" cue is on screen even mid-read. Drawn on the
    // border column, which owns no content.
    draw_help_scrollbar(fb, &lay, border);
    // POSITION FOOTER on its own reserved row.
    let foot: String = help_footer_text(
        lay.scroll,
        lay.body_h,
        lay.total,
        !state.help_filter.is_empty(),
    )
    .chars()
    .take(lay.text_w)
    .collect();
    draw_str(fb, lay.text_x, lay.footer_y, &foot, dim, bg);
}

/// Paint the help sheet's scrollbar THUMB on the box's right border column,
/// between the title band and the footer. Nothing is drawn when the whole sheet
/// fits (there is nothing to scroll, and a full-height thumb would imply there
/// is).
fn draw_help_scrollbar(fb: &mut Framebuffer, lay: &HelpLayout, border: u32) {
    if lay.total <= lay.body_h {
        return;
    }
    let col = lay.x0 + lay.pw - 1;
    let track = lay.body_h;
    // Thumb size + position in track rows, both at least one row.
    let thumb = (track * track / lay.total).max(1);
    let max_scroll = lay.total - lay.body_h;
    let span = track - thumb;
    let top = (span * lay.scroll).checked_div(max_scroll).unwrap_or(0);
    let bg = 0x00_0C1018;
    for i in 0..track {
        let on = i >= top && i < top + thumb;
        fb.set(
            col,
            lay.body_y + i,
            Cell {
                glyph: if on { '|' as u32 } else { ':' as u32 },
                fg: if on { 0x00_FFE060 } else { border },
                bg,
                ..Cell::default()
            },
        );
    }
}

/// Geometry of the DIRECTION-PICKER box: `(x0, y0, pw, ph)`. Fixed-size for the
/// 3×3 layer (a small, stable overlay). Mirrors `draw_dir_picker`. `None` if the
/// framebuffer can't hold it.
pub fn dir_picker_box_geom(w: usize, h: usize) -> Option<(usize, usize, usize, usize)> {
    // Inner content: a 3×3 cell grid (each cell 3 wide so an arrow + spacing fits)
    // plus title + layer label + a footer legend. 26 wide × 11 tall is comfortable.
    let pw = 28.min(w);
    let ph = 11.min(h);
    if pw < 20 || ph < 9 {
        return None;
    }
    Some(((w - pw) / 2, (h - ph) / 2, pw, ph))
}

/// Draw the 3×3×3 DIRECTION-PICKER overlay: the currently-displayed Z layer as a
/// 3×3 grid of arrow glyphs (one per cube cell), the CENTRE cell showing the
/// emitter's CURRENT aim arrow, the CURSOR cell highlighted, and the cell that
/// matches the current direction marked. Topmost-but-one overlay. No-op if the
/// picker is closed or the framebuffer is too small.
///
/// Layout per cell `(dx, dy)` in the layer: `dx` across (col, +X right), `dy`
/// down (row, +Y south) — the editor's top-down convention (see `crate::picker`).
pub fn draw_dir_picker(state: &EditorState, fb: &mut Framebuffer, w: usize, h: usize) {
    let Some(p) = state.dir_picker else { return };
    let Some((x0, y0, pw, ph)) = dir_picker_box_geom(w, h) else {
        return;
    };
    let bg = 0x00_0C1018;
    let border = 0x00_77AADD;
    let fg = 0x00_C8D4E0;
    let cursor_fg = 0x00_FFE060; // highlight under the cube cursor
    let current_fg = 0x00_60FFA0; // the cell matching the live direction
    for ry in 0..ph {
        for rx in 0..pw {
            let edge = rx == 0 || ry == 0 || rx == pw - 1 || ry == ph - 1;
            fb.set(
                x0 + rx,
                y0 + ry,
                Cell {
                    glyph: if edge { '#' as u32 } else { ' ' as u32 },
                    fg: if edge { border } else { fg },
                    bg,
                    ..Cell::default()
                },
            );
        }
    }
    let ix = x0 + 2;
    draw_str(fb, ix, y0 + 1, "AIM — 3x3x3 picker", border, bg);
    draw_str(
        fb,
        ix,
        y0 + 2,
        &format!("layer: {}", p.layer_label()),
        fg,
        bg,
    );
    // The cell of the cube that matches the emitter's live direction (so the user
    // sees the current selection across layers).
    let live_cell = state
        .selected_emitter
        .filter(|&i| i < state.level.emitters.len())
        .map(|i| crate::picker::direction_to_cell(state.level.emitters[i].direction));
    // 3×3 grid: dy = -1,0,+1 rows; dx = -1,0,+1 cols. Each cell 3 cols wide.
    let grid_y0 = y0 + 4;
    let grid_x0 = ix + 4;
    for (ri, dy) in [-1, 0, 1].iter().enumerate() {
        for (ci, dx) in [-1, 0, 1].iter().enumerate() {
            let cell = crate::picker::CubeCell {
                dx: *dx,
                dy: *dy,
                dz: p.cz,
            };
            // Centre cell shows the emitter's CURRENT aim arrow; the 8 around it
            // show the direction THEY would set.
            let glyph = if cell.dx == 0 && cell.dy == 0 {
                state
                    .selected_emitter
                    .filter(|&i| i < state.level.emitters.len())
                    .map(|i| crate::picker::direction_to_arrow(state.level.emitters[i].direction))
                    .unwrap_or('·')
            } else {
                cell.direction()
                    .map(crate::picker::direction_to_arrow)
                    .unwrap_or('·')
            };
            let is_cursor = *dx == p.cx && *dy == p.cy;
            let is_live = live_cell == Some(cell) && !(cell.dx == 0 && cell.dy == 0);
            let cell_fg = if is_cursor {
                cursor_fg
            } else if is_live {
                current_fg
            } else {
                fg
            };
            let cx = grid_x0 + ci * 4;
            let cy = grid_y0 + ri;
            // Bracket the cursor cell so the highlight reads even in 256-colour.
            if is_cursor {
                fb.set(cx.saturating_sub(1), cy, glyph_cell('[', cursor_fg, bg));
                fb.set(cx + 1, cy, glyph_cell(']', cursor_fg, bg));
            }
            fb.set(cx, cy, glyph_cell(glyph, cell_fg, bg));
        }
    }
    draw_str(fb, ix, y0 + ph - 2, "arrows [/] Enter Esc", border, bg);
}

/// A single styled `Cell` carrying `glyph` (helper for the picker grid).
fn glyph_cell(glyph: char, fg: u32, bg: u32) -> Cell {
    Cell {
        glyph: glyph as u32,
        fg,
        bg,
        ..Cell::default()
    }
}

/// Draw the LOAD/OPEN file-picker overlay: a centered bordered list of the
/// scanned `levels/*.lvl` files with the highlighted one marked `>`, plus a key
/// legend footer. No-op if too small or the overlay is closed.
pub fn draw_open(state: &EditorState, fb: &mut Framebuffer, w: usize, h: usize) {
    let Some(o) = state.open.as_ref() else { return };
    let title = "OPEN LEVEL";
    // Geometry shared with the hit-test (`open_box_geom`).
    let Some((x0, y0, pw, ph)) = open_box_geom(state, w, h) else {
        return;
    };
    let bg = 0x00_10182A;
    let border = 0x00_77AADD;
    let fg = 0x00_D0D8E8;
    let focus_fg = 0x00_FFE060;
    for ry in 0..ph {
        for rx in 0..pw {
            let edge = rx == 0 || ry == 0 || rx == pw - 1 || ry == ph - 1;
            fb.set(
                x0 + rx,
                y0 + ry,
                Cell {
                    glyph: if edge { '#' as u32 } else { ' ' as u32 },
                    fg: if edge { border } else { fg },
                    bg,
                    ..Cell::default()
                },
            );
        }
    }
    let ix = x0 + 2;
    draw_str(fb, ix, y0 + 1, title, border, bg);
    let body_y0 = y0 + 2;
    let body_h = ph.saturating_sub(4);
    if o.files.is_empty() {
        draw_str(fb, ix, body_y0, "(no .lvl in levels/)", fg, bg);
    } else {
        // Window the list around the cursor so a long levels/ dir stays navigable.
        let start = window_start(o.cursor.min(o.files.len() - 1), o.files.len(), body_h);
        for line in 0..body_h {
            let fi = start + line;
            if fi >= o.files.len() {
                break;
            }
            let focused = fi == o.cursor;
            let marker = if focused { '>' } else { ' ' };
            let color = if focused { focus_fg } else { fg };
            let name = o.files[fi]
                .rsplit(['/', '\\'])
                .next()
                .unwrap_or(&o.files[fi]);
            draw_str(
                fb,
                ix,
                body_y0 + line,
                &format!("{marker}{name}"),
                color,
                bg,
            );
        }
    }
    // Footer: clickable [Load][Cancel] buttons (shared geom).
    if let Some(((ls, _le), (cs, _ce), frow)) = open_footer_buttons(state, w, h) {
        draw_str(fb, ls, frow, "[Load]", 0x00_B0FFC0, bg);
        draw_str(fb, cs, frow, "[Cancel]", 0x00_FFC0C0, bg);
    }
    // Also a [X] close button on the title border.
    let (bx_y, bx_s, _bx_e) = modal_close_button(x0, y0, pw);
    draw_str(fb, bx_s, bx_y, "[X]", 0x00_FFC0C0, bg);
}

/// Draw the NEW-LEVEL wizard as a small CENTERED panel over the whole
/// framebuffer (`w` x `h`). Lists the size X/Y/Z, name, and shell fields with the
/// focused one marked `>` and highlighted, plus the key legend. Drawn last so it
/// overlays the editor. No-op if the framebuffer is too small.
pub fn draw_wizard(state: &EditorState, fb: &mut Framebuffer, w: usize, h: usize) {
    let Some(wiz) = state.wizard.as_ref() else {
        return;
    };
    let Some((x0, y0, pw, ph)) = wizard_box_geom(w, h) else {
        return;
    };
    let bg = 0x00_101826;
    let border = 0x00_77AADD;
    let fg = 0x00_E8E8FF;
    let focus_fg = 0x00_FFE060;

    // Fill the panel background + a simple border.
    for ry in 0..ph {
        for rx in 0..pw {
            let edge = rx == 0 || ry == 0 || rx == pw - 1 || ry == ph - 1;
            fb.set(
                x0 + rx,
                y0 + ry,
                Cell {
                    glyph: if edge { '#' as u32 } else { ' ' as u32 },
                    fg: if edge { border } else { fg },
                    bg,
                    ..Cell::default()
                },
            );
        }
    }

    let ix = x0 + 2;
    draw_str(fb, ix, y0 + 1, "NEW LEVEL", border, bg);
    // Clickable [X] close button on the top border (shared geom with the hit-test).
    let (bx_y, bx_s, _bx_e) = modal_close_button(x0, y0, pw);
    draw_str(fb, bx_s, bx_y, "[X]", 0x00_FFC0C0, bg);

    let rows = [
        (
            crate::wizard::WizardField::SizeX,
            format!("size X : {}", wiz.sx),
        ),
        (
            crate::wizard::WizardField::SizeY,
            format!("size Y : {}", wiz.sy),
        ),
        (
            crate::wizard::WizardField::SizeZ,
            format!("size Z : {}", wiz.sz),
        ),
        (
            crate::wizard::WizardField::Name,
            // Show a trailing `_` text cursor while the Name field is focused so the
            // author sees the free-text caret.
            if wiz.focus == crate::wizard::WizardField::Name {
                format!("name   : {}_", wiz.name())
            } else {
                format!("name   : {}", wiz.name())
            },
        ),
        (
            crate::wizard::WizardField::Shell,
            format!("shell  : {}", if wiz.shell { "yes" } else { "no" }),
        ),
    ];
    // The `[-]`/`[+]` per-row button columns (shared geom with the hit-test).
    let (minus_x, plus_x) = wizard_rows(w, h)
        .map(|(_x0, _y0, mx, px, _f)| (mx, px))
        .unwrap_or((x0 + pw.saturating_sub(7), x0 + pw.saturating_sub(4)));
    for (i, (field, text)) in rows.iter().enumerate() {
        let focused = *field == wiz.focus;
        let marker = if focused { '>' } else { ' ' };
        let line = format!("{marker}{text}");
        let color = if focused { focus_fg } else { fg };
        let ry = y0 + 2 + i;
        draw_str(fb, ix, ry, &line, color, bg);
        // Clickable [-]/[+] adjust controls on every field row.
        draw_str(fb, minus_x, ry, "[-]", 0x00_C0D0E0, bg);
        draw_str(fb, plus_x, ry, "[+]", 0x00_C0D0E0, bg);
    }
    // Footer buttons (shared geom in `wizard_hit`): [Clr] [Make] [Cancel].
    let frow = y0 + ph - 2;
    let clr_s = x0 + 2;
    let make_s = clr_s + "[Clr]".len() + 1;
    let cancel_s = make_s + "[Make]".len() + 1;
    draw_str(fb, clr_s, frow, "[Clr]", 0x00_FFD0A0, bg);
    draw_str(fb, make_s, frow, "[Make]", 0x00_B0FFC0, bg);
    draw_str(fb, cancel_s, frow, "[Cancel]", 0x00_FFC0C0, bg);
}

/// Build the LEVEL SETTINGS row list as `(focused?, text)` in DRAW ORDER. This is
/// the single source of truth shared by [`draw_settings`] (the painter) and the
/// scroll/focus mapping: the focused row index is derived as the first row whose
/// `focused` flag is set, so the keep-focus-visible scroll can never disagree with
/// what is painted. Section foci (Inputs/Observe/Steps) flag the cursor row (or the
/// `(none)` placeholder when the section is empty). Pure — no terminal, testable.
#[must_use]
pub fn settings_rows(s: &crate::settings::SettingsState) -> Vec<(bool, String)> {
    use crate::settings::{ambient_label, SettingsField as F};
    let mut rows: Vec<(bool, String)> = Vec::new();

    // META.
    rows.push((false, "-- Meta --".to_string()));
    let name_caret = if s.focus == F::Name { "_" } else { "" };
    rows.push((
        s.focus == F::Name,
        format!("name     : {}{}", s.name, name_caret),
    ));
    let title_caret = if s.focus == F::Title { "_" } else { "" };
    rows.push((
        s.focus == F::Title,
        format!("title    : {}{}", s.title, title_caret),
    ));
    // Free-text metadata rows share ONE renderer so `(none)` / caret behaviour is
    // identical across every one of them (and matches the `[sky]` path rows).
    let text_row = |label: &str, field: F| -> (bool, String) {
        let focused = s.focus == field;
        let buf = s.text_of(field);
        let caret = if focused { "_" } else { "" };
        let shown = if buf.is_empty() && !focused {
            "(none)".to_string()
        } else {
            format!("{buf}{caret}")
        };
        (focused, format!("{label} : {shown}"))
    };
    rows.push(text_row("version ", F::Version));
    rows.push(text_row("spec_rev", F::SpecRev));

    // ROOM.
    rows.push((false, "-- Room --".to_string()));
    rows.push((
        s.focus == F::Ambient,
        format!("ambient : {}", ambient_label(s.ambient())),
    ));
    rows.push((
        s.focus == F::StoreyHeight,
        format!("storey  : {}", s.storey_height),
    ));
    // Live grid dimensions — editable; [Resize] (R) applies them to the map.
    rows.push((false, "  size (W H D) — R to resize:".to_string()));
    rows.push((s.focus == F::SizeW, format!("  width  (sx) : {}", s.size_w)));
    rows.push((s.focus == F::SizeH, format!("  height (sy) : {}", s.size_h)));
    rows.push((s.focus == F::SizeD, format!("  depth  (sz) : {}", s.size_d)));
    // `power_budget` — a TYPED digit buffer (0-9 type, -/+ step). An empty buffer
    // reads `(none)`, which is exactly what the writer emits for a zero budget.
    let pb_caret = if s.focus == F::PowerBudget { "_" } else { "" };
    let pb_shown = if s.power_budget.is_empty() && s.focus != F::PowerBudget {
        "(none)".to_string()
    } else {
        format!("{}{pb_caret}", s.power_budget)
    };
    rows.push((
        s.focus == F::PowerBudget,
        format!("power_budget : {pb_shown}   (type 0-9, -/+ step)"),
    ));
    // `[room] floor` / `ceiling` are INERT STYLE TAGS, not material names — the
    // surface's look comes from a `[material] surface = floor|ceiling` binding,
    // authored on the material. Say so on the row so nobody expects otherwise.
    rows.push(text_row("floor       ", F::RoomFloor));
    rows.push(text_row("ceiling     ", F::RoomCeiling));
    rows.push((
        false,
        "  (floor/ceiling = inert style tags; look = [material] surface=)".to_string(),
    ));
    rows.push((
        s.focus == F::CycleKeep,
        format!("cycle_keep   : {}", if s.cycle_keep { "yes" } else { "no" }),
    ));
    rows.push((
        s.focus == F::RenderPipes,
        format!(
            "render_pipes : {}",
            if s.render_pipes { "yes" } else { "no" }
        ),
    ));

    // EFFECTS ([effects]) — snow/matrix/glitch shown to 1 decimal, steam a bool.
    rows.push((false, "-- Effects --".to_string()));
    rows.push((s.focus == F::Snow, format!("snow   : {:.1}", s.snow)));
    rows.push((
        s.focus == F::Steam,
        format!("steam  : {}", if s.steam { "yes" } else { "no" }),
    ));
    rows.push((s.focus == F::Matrix, format!("matrix : {:.1}", s.matrix)));
    rows.push((s.focus == F::Glitch, format!("glitch : {:.1}", s.glitch)));

    // FOG ([palette]) — hex colour text buffer + near/far distances. An empty colour
    // reads `(none)` so the author sees fog is OFF; near/far show `(off)` until set.
    rows.push((false, "-- Fog (palette) --".to_string()));
    let fog_caret = if s.focus == F::FogColor { "_" } else { "" };
    let fog_shown = if s.fog_color.is_empty() && s.focus != F::FogColor {
        "(none)".to_string()
    } else {
        format!("{}{}", s.fog_color, fog_caret)
    };
    rows.push((s.focus == F::FogColor, format!("fog_color : {fog_shown}")));
    let dist_note = if s.has_fog_dist {
        String::new()
    } else {
        "  (off)".to_string()
    };
    rows.push((
        s.focus == F::FogNear,
        format!("fog_near  : {:.1}{dist_note}", s.fog_near),
    ));
    rows.push((
        s.focus == F::FogFar,
        format!("fog_far   : {:.1}{dist_note}", s.fog_far),
    ));

    // ATMOSPHERE ([atmosphere]) — the per-level LIGHTING mood. Every value is
    // OPTIONAL: unset reads `(default)`/`(none)`/`(off)` and emits no line, so an
    // atmosphere-less level stays byte-identical. The ambient row is labelled
    // `(0-1)` to distinguish it from the `[room] ambient` wall-KIND tag above.
    rows.push((false, "-- Atmosphere (lighting) --".to_string()));
    let amb_shown = s
        .atmos_ambient
        .map_or_else(|| ambient_label("").to_string(), |a| format!("{a:.2}"));
    rows.push((
        s.focus == F::AtmosAmbient,
        format!("ambient (0-1) : {amb_shown}"),
    ));
    let key_caret = if s.focus == F::KeyColor { "_" } else { "" };
    let key_shown = if s.key_color.is_empty() && s.focus != F::KeyColor {
        "(none)".to_string()
    } else {
        format!("{}{}", s.key_color, key_caret)
    };
    rows.push((
        s.focus == F::KeyColor,
        format!("key_color     : {key_shown}"),
    ));
    let hf_note = if s.has_fog_height {
        String::new()
    } else {
        "  (off)".to_string()
    };
    rows.push((
        s.focus == F::FogHeightBase,
        format!("fog_h base    : {:.1}{hf_note}", s.fog_height_base),
    ));
    rows.push((
        s.focus == F::FogHeightFalloff,
        format!("fog_h falloff : {:.1}{hf_note}", s.fog_height_falloff),
    ));
    rows.push((
        s.focus == F::Postfx,
        format!("postfx        : {}", ambient_label(s.postfx())),
    ));

    // SKY ([sky]) — three optional image paths (empty ⇒ no `[sky]` key emitted).
    rows.push((false, "-- Sky --".to_string()));
    let sky_row = |label: &str, focused: bool, buf: &str| -> (bool, String) {
        let caret = if focused { "_" } else { "" };
        let shown = if buf.is_empty() && !focused {
            "(none)".to_string()
        } else {
            format!("{buf}{caret}")
        };
        (focused, format!("{label} : {shown}"))
    };
    rows.push(sky_row(
        "cylinder  ",
        s.focus == F::SkyCylinder,
        &s.sky_cylinder,
    ));
    rows.push(sky_row(
        "cap_top   ",
        s.focus == F::SkyCapTop,
        &s.sky_cap_top,
    ));
    rows.push(sky_row(
        "cap_bottom",
        s.focus == F::SkyCapBottom,
        &s.sky_cap_bottom,
    ));

    // TEXTURES ([texture NAME]) — name = file rows. The selected row's edited COLUMN
    // (name/file) is marked with a caret; a:add  x:remove  Left/Right pick column.
    rows.push((false, "-- Textures (name = file) --".to_string()));
    if s.textures.is_empty() {
        rows.push((s.focus == F::Textures, "(none)  a:add".to_string()));
    } else {
        for (i, (name, file)) in s.textures.iter().enumerate() {
            let sel = s.focus == F::Textures && i == s.texture_cursor;
            let cur = if sel { '>' } else { ' ' };
            // Caret marks which column typing edits on the selected row.
            let name_caret = if sel && s.texture_col == 0 { "_" } else { "" };
            let file_caret = if sel && s.texture_col == 1 { "_" } else { "" };
            let name_shown = if name.is_empty() && !(sel && s.texture_col == 0) {
                "(name)".to_string()
            } else {
                format!("{name}{name_caret}")
            };
            let file_shown = if file.is_empty() && !(sel && s.texture_col == 1) {
                "(file)".to_string()
            } else {
                format!("{file}{file_caret}")
            };
            rows.push((sel, format!("{cur}{name_shown} = {file_shown}")));
        }
    }

    // DOOR ([door]) — one door per level (`Level::door` is a single struct, not a
    // list), so its inert `target_model` tag is a per-level field like the rest.
    rows.push((false, "-- Door --".to_string()));
    rows.push(text_row("target_model", F::DoorTargetModel));

    // INPUTS.
    rows.push((false, "-- Inputs --".to_string()));
    if s.inputs.is_empty() {
        rows.push((s.focus == F::Inputs, "(no input nodes)".to_string()));
    } else {
        for (i, (name, on)) in s.inputs.iter().enumerate() {
            let sel = s.focus == F::Inputs && i == s.input_cursor;
            let cur = if sel { '>' } else { ' ' };
            rows.push((
                sel,
                format!("{cur}{name} = {}", if *on { "on" } else { "off" }),
            ));
        }
    }

    // SEQUENCE: observe window.
    rows.push((false, "-- Observe --".to_string()));
    if s.observe.is_empty() {
        rows.push((s.focus == F::Observe, "(none)  a:add".to_string()));
    } else {
        for (i, name) in s.observe.iter().enumerate() {
            let sel = s.focus == F::Observe && i == s.observe_cursor;
            let cur = if sel { '>' } else { ' ' };
            rows.push((sel, format!("{cur}{name}")));
        }
    }

    // SEQUENCE: steps.
    rows.push((false, "-- Steps --".to_string()));
    if s.steps.is_empty() {
        rows.push((s.focus == F::Steps, "(none)  a:add".to_string()));
    } else {
        for (i, step) in s.steps.iter().enumerate() {
            let sel = s.focus == F::Steps && i == s.step_cursor;
            let cur = if sel { '>' } else { ' ' };
            // Mark the in-step name cursor with brackets on the selected step.
            let body = if step.is_empty() {
                "(all off)".to_string()
            } else {
                step.iter()
                    .enumerate()
                    .map(|(j, n)| {
                        if sel && j == s.name_cursor {
                            format!("[{n}]")
                        } else {
                            n.clone()
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            };
            rows.push((sel, format!("{cur}{}: {}", i + 1, body)));
        }
    }

    rows
}

/// Draw the LEVEL SETTINGS form as a generous CENTERED panel over the whole
/// framebuffer (`w` x `h`). Lists the Meta / Room / Inputs / Sequence sections;
/// the focused field/row is marked with `\u{25B6}` (▶) and highlighted. The row
/// list can exceed the box height, so it SCROLLS to keep the focused row visible
/// (off-screen rows flagged with ▲/▼); the footer stays pinned + clickable. Drawn
/// after the wizard so it overlays the editor. No-op if the form is closed or the
/// box does not fit (mirrors the geom early-out).
pub fn draw_settings(state: &EditorState, fb: &mut Framebuffer, w: usize, h: usize) {
    let Some(s) = state.settings.as_ref() else {
        return;
    };
    let Some((x0, y0, pw, ph)) = settings_box_geom(w, h) else {
        return;
    };
    let bg = 0x00_101826;
    let border = 0x00_77AADD;
    let fg = 0x00_E8E8FF;
    let focus_fg = 0x00_FFE060;
    let header_fg = 0x00_AEC4DA;

    // Panel background + border (same recipe as draw_wizard).
    for ry in 0..ph {
        for rx in 0..pw {
            let edge = rx == 0 || ry == 0 || rx == pw - 1 || ry == ph - 1;
            fb.set(
                x0 + rx,
                y0 + ry,
                Cell {
                    glyph: if edge { '#' as u32 } else { ' ' as u32 },
                    fg: if edge { border } else { fg },
                    bg,
                    ..Cell::default()
                },
            );
        }
    }

    let ix = x0 + 2;
    draw_str(fb, ix, y0 + 1, "LEVEL SETTINGS", border, bg);

    // Build the row list (the SSOT shared with the scroll/focus mapping). The
    // focused row gets the ▶ marker + highlight.
    let marker = '\u{25B6}';
    let rows = settings_rows(s);

    // Paint rows, clipping to the box interior (leave the bottom 2 rows for the
    // footer). The content starts at y0+2. The list can be taller than the window,
    // so we SCROLL it to keep the focused row visible — without a scroll, Tab could
    // move focus to a row clipped off the bottom and the user would edit it blind.
    let content_top = y0 + 2;
    let content_bottom = y0 + ph - 2; // exclusive draw limit (footer below)
    let visible_rows = content_bottom.saturating_sub(content_top);
    // The focused-row index is derived from the SAME `rows` list the painter built
    // (single source of truth): the first row flagged focused. Section foci
    // (Inputs/Observe/Steps) flag the cursor row (or the `(none)` placeholder), so
    // this resolves to exactly the row the user is editing. Defaults to 0.
    let focused_idx = rows.iter().position(|(f, _)| *f).unwrap_or(0);
    let scroll = settings_scroll(focused_idx, rows.len(), visible_rows);
    let max_inner_w = pw.saturating_sub(3);
    for (vi, (focused, text)) in rows.iter().skip(scroll).take(visible_rows).enumerate() {
        let ry = content_top + vi;
        let is_header = text.starts_with("--");
        let mk = if *focused { marker } else { ' ' };
        let mut line = format!("{mk}{text}");
        if line.chars().count() > max_inner_w {
            line = line.chars().take(max_inner_w).collect();
        }
        let color = if *focused {
            focus_fg
        } else if is_header {
            header_fg
        } else {
            fg
        };
        draw_str(fb, ix, ry, &line, color, bg);
    }

    // Off-screen indicators: a single-width ▲ / ▼ pinned just inside the right
    // border on the top / bottom content rows when rows are hidden above / below.
    // Drawn last so they sit atop the (clamped) row text at the far-right column.
    if visible_rows > 0 {
        let arrow_x = x0 + pw.saturating_sub(2);
        if scroll > 0 {
            draw_str(fb, arrow_x, content_top, "\u{25B2}", header_fg, bg);
        }
        if scroll + visible_rows < rows.len() {
            draw_str(fb, arrow_x, content_bottom - 1, "\u{25BC}", header_fg, bg);
        }
    }

    // Footer: [Apply] [Resize] [Cancel] — mouse-clickable (mouse::apply_settings_hit)
    // AND keyboard-driven. Geometry is the shared SSOT `settings_footer_spans` so a
    // click lands exactly on what's drawn here.
    if let Some(((aps, _), (rss, _), (cas, _), frow)) = settings_footer_spans(w, h) {
        draw_str(fb, aps, frow, "[Apply]", 0x00_B0FFC0, bg);
        draw_str(fb, rss, frow, "[Resize]", 0x00_B0D0FF, bg);
        draw_str(fb, cas, frow, "[Cancel]", 0x00_FFC0C0, bg);
    }
}

/// Draw the SAVE-AS filename prompt overlay, centered. Shows the FINAL path it
/// will write (`levels/<buf>_.lvl`, with a trailing `_` caret) plus a footer of
/// `[Clr] [Save] [Cancel]` and the shared `[X]` close. Early-returns if the
/// prompt is closed or the box does not fit (mirrors the geom early-out). The
/// footer button columns come from [`saveas_footer_spans`] (shared with the
/// hit-test) so a click lands on what is drawn.
pub fn draw_saveas(state: &EditorState, fb: &mut Framebuffer, w: usize, h: usize) {
    let Some(sa) = state.saveas.as_ref() else {
        return;
    };
    let Some((x0, y0, pw, ph)) = saveas_box_geom(w, h) else {
        return;
    };
    let bg = 0x00_101826;
    let border = 0x00_77AADD;
    let fg = 0x00_E8E8FF;

    // Panel background + border.
    for ry in 0..ph {
        for rx in 0..pw {
            let edge = rx == 0 || ry == 0 || rx == pw - 1 || ry == ph - 1;
            fb.set(
                x0 + rx,
                y0 + ry,
                Cell {
                    glyph: if edge { '#' as u32 } else { ' ' as u32 },
                    fg: if edge { border } else { fg },
                    bg,
                    ..Cell::default()
                },
            );
        }
    }

    let ix = x0 + 2;
    draw_str(fb, ix, y0 + 1, "SAVE AS", border, bg);
    // Clickable [X] close button (shared geom with the hit-test in mouse.rs).
    let (bx_y, bx_s, _bx_e) = modal_close_button(x0, y0, pw);
    draw_str(fb, bx_s, bx_y, "[X]", 0x00_FFC0C0, bg);

    // The final path the save will write, with a trailing `_` caret. Clamp to the
    // inner width so a long name can't overrun the right border.
    let path_line = format!("levels/{}_.lvl", sa.buf);
    let inner = pw.saturating_sub(4);
    let shown: String = path_line.chars().take(inner).collect();
    draw_str(fb, ix, y0 + 2, &shown, 0x00_FFE060, bg);

    // Footer: [Clr] [Save] [Cancel] (shared spans with saveas_hit).
    if let Some(((cls, _cle), (svs, _sve), (cas, _cae), frow)) = saveas_footer_spans(w, h) {
        draw_str(fb, cls, frow, "[Clr]", 0x00_FFD0A0, bg);
        draw_str(fb, svs, frow, "[Save]", 0x00_B0FFC0, bg);
        draw_str(fb, cas, frow, "[Cancel]", 0x00_FFC0C0, bg);
    }
}

/// Draw the F2 RENAME prompt, centered — the same modal geometry as the
/// SAVE-AS prompt (`saveas_box_geom` is reused so both prompts share one
/// size/degradation contract). Shows the target's kind-agnostic name buffer
/// with a trailing `_` caret and the key hints.
pub fn draw_rename(state: &EditorState, fb: &mut Framebuffer, w: usize, h: usize) {
    let Some(rn) = state.rename.as_ref() else {
        return;
    };
    let Some((x0, y0, pw, ph)) = saveas_box_geom(w, h) else {
        return;
    };
    let bg = 0x00_101826;
    let border = 0x00_77AADD;
    let fg = 0x00_E8E8FF;
    for ry in 0..ph {
        for rx in 0..pw {
            let edge = rx == 0 || ry == 0 || rx == pw - 1 || ry == ph - 1;
            fb.set(
                x0 + rx,
                y0 + ry,
                Cell {
                    glyph: if edge { '#' as u32 } else { ' ' as u32 },
                    fg: if edge { border } else { fg },
                    bg,
                    ..Cell::default()
                },
            );
        }
    }
    let ix = x0 + 2;
    draw_str(fb, ix, y0 + 1, "RENAME", border, bg);
    // The editable name with a `_` caret, clamped to the inner width.
    let line = format!("{}_", rn.buf);
    let inner = pw.saturating_sub(4);
    let shown: String = line.chars().take(inner).collect();
    draw_str(fb, ix, y0 + 2, &shown, 0x00_FFE060, bg);
    let hint: String = "Enter rename  Esc cancel".chars().take(inner).collect();
    draw_str(
        fb,
        ix,
        y0 + 4.min(ph.saturating_sub(2)),
        &hint,
        0x00_8899AA,
        bg,
    );
}

// ─── CRASH-RECOVERY OFFER OVERLAY ─────────────────────────────────────────────
//
// `open::load_path` raises `state.autosave_offer` when this document's autosave
// sidecar is strictly newer than the file just opened, and `edit::apply_key`
// dispatches it at the very FRONT of the key chain, swallowing every key. A
// modal that owns the keyboard but paints nothing looks like a frozen editor,
// so it is drawn here, and its
// two buttons are hit-tested from the SAME geometry that paints them
// ([`autosave_layout`]), so a click can never land on a control that is not
// there.

/// The `[ Restore ]` control label in the crash-recovery overlay.
pub const AUTOSAVE_RESTORE_LABEL: &str = "[ Restore ]";
/// The `[ Discard ]` control label in the crash-recovery overlay.
pub const AUTOSAVE_DISCARD_LABEL: &str = "[ Discard ]";

/// A clickable control inside the crash-recovery overlay, returned by
/// [`autosave_hit`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AutosaveHit {
    /// `[ Restore ]` → load the sidecar into the editor (same as `y`).
    Restore,
    /// `[ Discard ]` → drop the offer, keep the sidecar on disk (same as `n`).
    Discard,
}

/// The fully-resolved geometry of the crash-recovery overlay: the box, its inner
/// text column, the absolute row of every line it paints, and the absolute column
/// spans of its two buttons.
///
/// THE single source of truth — [`draw_autosave_offer`] paints from it and
/// [`autosave_hit`] hit-tests against it, so the painted rect and the click
/// target cannot drift apart (the B1 colour-overlay lesson: geometry computed in
/// two places overpaints a still-clickable control).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct AutosaveLayout {
    /// Box origin + size, `(x0, y0, pw, ph)`.
    pub x0: usize,
    /// See [`AutosaveLayout::x0`].
    pub y0: usize,
    /// See [`AutosaveLayout::x0`].
    pub pw: usize,
    /// See [`AutosaveLayout::x0`].
    pub ph: usize,
    /// Inner text column (the box's left padding already applied).
    pub ix: usize,
    /// Width available for a text line before it would hit the right border.
    pub inner_w: usize,
    /// Row of the "a newer autosave was found" lead line.
    pub lead_row: usize,
    /// Row showing the sidecar path.
    pub path_row: usize,
    /// Row carrying `[ Restore ]` and `[ Discard ]`.
    pub button_row: usize,
    /// Row of the key-equivalent hint line.
    pub keys_row: usize,
    /// Row of the "discarding keeps the sidecar on disk" note.
    pub note_row: usize,
    /// Absolute `(start, end)` columns of `[ Restore ]` (end exclusive).
    pub restore: (usize, usize),
    /// Absolute `(start, end)` columns of `[ Discard ]` (end exclusive).
    pub discard: (usize, usize),
}

/// The crash-recovery overlay box: `(x0, y0, pw, ph)`, centered. `None` when the
/// terminal is too small to host it — in which case the modal simply does not
/// paint (it still owns the keyboard and swallows clicks), mirroring every other
/// modal's degradation contract rather than clipping into garbage.
#[must_use]
pub fn autosave_box_geom(w: usize, h: usize) -> Option<(usize, usize, usize, usize)> {
    let pw = 62usize.min(w);
    let ph = 10usize.min(h);
    if pw < 34 || ph < 10 {
        return None;
    }
    Some(((w - pw) / 2, (h - ph) / 2, pw, ph))
}

/// Resolve [`AutosaveLayout`] for a `w`×`h` framebuffer. Every row comes from ONE
/// running cursor, and the fn declines (returns `None`) if the last row or the
/// widest button would reach the border — so no line can ever overpaint the frame
/// it is drawn inside.
#[must_use]
pub fn autosave_layout(w: usize, h: usize) -> Option<AutosaveLayout> {
    let (x0, y0, pw, ph) = autosave_box_geom(w, h)?;
    let ix = x0 + 2;
    let inner_w = pw.saturating_sub(4);
    // One running cursor. `draw_overlay_box` returns `y0 + 2` as its first inner
    // text row (row `y0 + 1` is the title), so the cursor starts there.
    let mut y = y0 + 2;
    let lead_row = y;
    y += 1;
    let path_row = y;
    y += 2; // one blank separator row
    let button_row = y;
    y += 1;
    let keys_row = y;
    y += 1;
    let note_row = y;
    // The bottom border sits on `y0 + ph - 1`; nothing may reach it.
    if note_row >= y0 + ph - 1 {
        return None;
    }
    let rs = ix;
    let re = rs + AUTOSAVE_RESTORE_LABEL.chars().count();
    let ds = re + 2;
    let de = ds + AUTOSAVE_DISCARD_LABEL.chars().count();
    if de > x0 + pw - 2 {
        return None;
    }
    Some(AutosaveLayout {
        x0,
        y0,
        pw,
        ph,
        ix,
        inner_w,
        lead_row,
        path_row,
        button_row,
        keys_row,
        note_row,
        restore: (rs, re),
        discard: (ds, de),
    })
}

/// Hit-test a click inside the crash-recovery overlay. Returns the
/// [`AutosaveHit`] under `(col, row)`, or `None` for anywhere else (which
/// `mouse.rs` consumes without effect — a modal never leaks a click).
#[must_use]
pub fn autosave_hit(w: usize, h: usize, col: usize, row: usize) -> Option<AutosaveHit> {
    let l = autosave_layout(w, h)?;
    if row != l.button_row {
        return None;
    }
    if col >= l.restore.0 && col < l.restore.1 {
        return Some(AutosaveHit::Restore);
    }
    if col >= l.discard.0 && col < l.discard.1 {
        return Some(AutosaveHit::Discard);
    }
    None
}

/// Shorten `path` to at most `max` columns, keeping the TAIL. A sidecar path's
/// informative half is its file name, so a head-truncation (`levels/very-lon…`)
/// would hide exactly what the user needs to recognise. ASCII `...` keeps every
/// glyph single-width.
fn tail_clamp(path: &str, max: usize) -> String {
    let n = path.chars().count();
    if n <= max {
        return path.to_string();
    }
    if max <= 3 {
        return path.chars().skip(n - max).collect();
    }
    let keep = max - 3;
    let tail: String = path.chars().skip(n - keep).collect();
    format!("...{tail}")
}

/// Draw the CRASH-RECOVERY offer, centered. States that a NEWER autosave sidecar
/// was found after this file was opened, shows the sidecar's path, offers the two
/// choices as clickable controls, and says that discarding leaves the sidecar file
/// on disk. Early-returns when no offer is pending or the box does not fit.
pub fn draw_autosave_offer(state: &EditorState, fb: &mut Framebuffer, w: usize, h: usize) {
    let Some(sidecar) = state.autosave_offer.as_deref() else {
        return;
    };
    let Some(l) = autosave_layout(w, h) else {
        return;
    };
    let bg = 0x00_0C1018;
    // Amber border: this prompt guards unsaved work, so it does not wear the
    // ordinary blue modal chrome.
    let border = 0x00_FFB040;
    draw_overlay_box(fb, (l.x0, l.y0, l.pw, l.ph), "CRASH RECOVERY", border);

    let lead: String = "A NEWER autosave was found after this file opened:"
        .chars()
        .take(l.inner_w)
        .collect();
    draw_str(fb, l.ix, l.lead_row, &lead, 0x00_E8E8FF, bg);
    draw_str(
        fb,
        l.ix,
        l.path_row,
        &tail_clamp(sidecar, l.inner_w),
        0x00_FFE060,
        bg,
    );
    draw_str(
        fb,
        l.restore.0,
        l.button_row,
        AUTOSAVE_RESTORE_LABEL,
        0x00_B0FFC0,
        bg,
    );
    draw_str(
        fb,
        l.discard.0,
        l.button_row,
        AUTOSAVE_DISCARD_LABEL,
        0x00_FFC0C0,
        bg,
    );
    let keys: String = "Y restore   N discard   Esc dismiss"
        .chars()
        .take(l.inner_w)
        .collect();
    draw_str(fb, l.ix, l.keys_row, &keys, 0x00_8899AA, bg);
    let note: String = "Discard keeps the sidecar file on disk."
        .chars()
        .take(l.inner_w)
        .collect();
    draw_str(fb, l.ix, l.note_row, &note, 0x00_8899AA, bg);
}

const TREE_HEADER_FG: u32 = 0x00_88CCFF;
const TREE_GROUP_FG: u32 = 0x00_C0A060;
const TREE_OBJ_FG: u32 = 0x00_B0C0D0;
const TREE_SEL_BG: u32 = 0x00_303A66;
const TREE_SEL_FG: u32 = 0x00_FFE060;

/// Rows the OUTLINER reserves at the TOP of its rect for the organ header before
/// the foldable tree body begins: row 0 = `OUTLINER` title, row 1 = the
/// `[x] Show objects` toggle, row 2 = the `[+ Add object]` affordance. The two
/// controls sit on SEPARATE rows so neither clips at the 30-col panel's right edge
/// (full-word labels are an explicit project rule — never abbreviate). Shared by
/// the painter ([`draw_tree`]) and the hit-tests ([`tree_row_at`] /
/// [`outliner_controls`]).
pub const OUTLINER_HEADER_ROWS: usize = 3;

/// The clickable control spans on the OUTLINER's two control rows:
/// `((check_s, check_e, check_row), (add_s, add_e, add_row))` as absolute columns +
/// the absolute row of each. The Show-objects toggle is on `y0 + 1`, the
/// Add-object affordance on `y0 + 2` (each starting at the pane's left edge so it
/// never clips). THE single source of truth shared by [`draw_tree`] (painter) and
/// the `mouse.rs` hit-test, so a click toggles/opens exactly what is drawn.
#[allow(clippy::type_complexity)]
pub fn outliner_controls(
    rect: crate::layout::Rect,
) -> Option<((usize, usize, usize), (usize, usize, usize))> {
    if rect.is_empty() || rect.h < OUTLINER_HEADER_ROWS {
        return None;
    }
    let check_row = rect.y + 1;
    let add_row = rect.y + 2;
    let check_s = rect.x;
    let check_e = check_s + "[x] Show objects".len();
    let add_s = rect.x;
    let add_e = add_s + "[+ Add object]".len();
    Some(((check_s, check_e, check_row), (add_s, add_e, add_row)))
}

/// Draw the OBJECT TREE / OUTLINER into the rect `(x0,y0,w,h)`: the organ header
/// (title + `[x] Show objects` toggle + `[+ Add object]`), then a count summary,
/// per-type group headers (`v`/`>` fold marker + label + LIVE count) with each
/// group's object rows indented beneath. The row under the tree cursor is
/// highlighted. The list is windowed around the cursor so a long level stays
/// navigable in a short pane. Always visible (no longer gated behind `tree.open`).
/// Pure read view over `crate::tree::build_tree` — never mutates the level.
pub fn draw_tree(
    state: &EditorState,
    fb: &mut Framebuffer,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
) {
    if w == 0 || h == 0 {
        return;
    }
    let bg = 0x00_080C12;
    // Paint the pane background so the organ reads as its own window.
    for ry in 0..h {
        for rx in 0..w {
            fb.set(
                x0 + rx,
                y0 + ry,
                Cell {
                    glyph: ' ' as u32,
                    fg: TREE_OBJ_FG,
                    bg,
                    ..Cell::default()
                },
            );
        }
    }
    // OUTLINER organ header accent (sky blue) marks "where I select objects".
    // While the `/` filter is live the query rides on the header row
    // with a caret, so the filter is always visible where it acts.
    if let Some(f) = state.tree.filter.as_deref() {
        let head: String = format!("OUTLINER /{f}\u{2038}").chars().take(w).collect();
        draw_str(fb, x0, y0, &head, STATUS_FG, bg);
    } else {
        draw_str(fb, x0, y0, "OUTLINER", OUTLINER_HEADER_FG, bg);
    }
    // Control rows: the Show-objects toggle (H) on row 1, the create affordance on
    // row 2 — two rows so `[+ Add object]` is fully visible (no right-edge clip).
    let rect = crate::layout::Rect { x: x0, y: y0, w, h };
    if let Some(((cs, _ce, crow), (as_, _ae, arow))) = outliner_controls(rect) {
        let check = if state.show_objects { 'x' } else { ' ' };
        let check_fg = if state.show_objects {
            0x00_60FF90
        } else {
            0x00_FF8060
        };
        draw_str(
            fb,
            cs,
            crow,
            &format!("[{check}] Show objects"),
            check_fg,
            bg,
        );
        draw_str(fb, as_, arow, "[+ Add object]", 0x00_88CCAA, bg);
    }

    let rows = tree::build_tree(state);
    if rows.is_empty() {
        return;
    }
    let body_y0 = y0 + OUTLINER_HEADER_ROWS;
    let body_h = h.saturating_sub(OUTLINER_HEADER_ROWS);
    if body_h == 0 {
        return;
    }
    let cursor = state.tree.cursor.min(rows.len() - 1);
    // Window the rows so the cursor stays in view (scrolls a long list).
    let start = window_start(cursor, rows.len(), body_h);
    for line in 0..body_h {
        let ri = start + line;
        if ri >= rows.len() {
            break;
        }
        let py = body_y0 + line;
        let is_cursor = ri == cursor;
        let (text, base_fg, indent) = match &rows[ri] {
            TreeRow::Summary { total } => (format!("Objects: {total}"), TREE_HEADER_FG, 0),
            TreeRow::Group {
                group,
                count,
                collapsed,
            } => {
                let mark = if *collapsed { '\u{25B8}' } else { '\u{25BE}' }; // ▸ / ▾
                (
                    format!("{mark} {} ({count})", group.label()),
                    TREE_GROUP_FG,
                    0,
                )
            }
            TreeRow::Object { text, depth, .. } => (text.clone(), TREE_OBJ_FG, 2 + *depth as usize),
        };
        let fg = if is_cursor { TREE_SEL_FG } else { base_fg };
        let line_bg = if is_cursor { TREE_SEL_BG } else { bg };
        // Highlight the whole row width when selected.
        if is_cursor {
            for rx in 0..w {
                fb.set(
                    x0 + rx,
                    py,
                    Cell {
                        glyph: ' ' as u32,
                        fg,
                        bg: line_bg,
                        ..Cell::default()
                    },
                );
            }
        }
        let avail = w.saturating_sub(indent);
        let shown: String = text.chars().take(avail).collect();
        draw_str(fb, x0 + indent, py, &shown, fg, line_bg);
    }
}

/// First row to show in a `body_h`-tall tree window so the `cursor` row stays
/// visible: keep the cursor centred, clamped to the list ends.
fn window_start(cursor: usize, len: usize, body_h: usize) -> usize {
    if len <= body_h {
        return 0;
    }
    let half = body_h / 2;
    cursor.saturating_sub(half).min(len - body_h)
}

/// Draw the object-TYPE SELECTOR palette as a small centered menu: the create
/// types in display order, the highlighted one marked `>` and tinted. The
/// discoverable alternative to the `o`-cycle. No-op if too small.
pub fn draw_palette(state: &EditorState, fb: &mut Framebuffer, w: usize, h: usize) {
    let Some(sel) = state.palette else { return };
    let items = tree::PALETTE_ITEMS;
    let Some((x0, y0, pw, ph)) = palette_box_geom(w, h) else {
        return;
    };
    let bg = 0x00_10182A;
    let border = 0x00_77AADD;
    let fg = 0x00_E0E8FF;
    let focus_fg = 0x00_FFE060;
    for ry in 0..ph {
        for rx in 0..pw {
            let edge = rx == 0 || ry == 0 || rx == pw - 1 || ry == ph - 1;
            fb.set(
                x0 + rx,
                y0 + ry,
                Cell {
                    glyph: if edge { '#' as u32 } else { ' ' as u32 },
                    fg: if edge { border } else { fg },
                    bg,
                    ..Cell::default()
                },
            );
        }
    }
    let ix = x0 + 2;
    draw_str(fb, ix, y0 + 1, "CREATE OBJECT", border, bg);
    for (i, item) in items.iter().enumerate() {
        let focused = i == sel;
        let marker = if focused { '>' } else { ' ' };
        let color = if focused { focus_fg } else { fg };
        draw_str(
            fb,
            ix,
            y0 + 2 + i,
            &format!("{marker}{}", item.label()),
            color,
            bg,
        );
    }
    // Clickable [X] close on the title border.
    let (bx_y, bx_s, _bx_e) = modal_close_button(x0, y0, pw);
    draw_str(fb, bx_s, bx_y, "[X]", 0x00_FFC0C0, bg);
}

/// A compact `#`/`.` readout of the part's ACTIVE mask h-layer (`mask_layer`): one
/// `lz` row per `/`-separated segment, `lx` across, with the cursor voxel shown in
/// `[ ]`. Reads each voxel via the `part.rs` SSOT [`crate::part::mask_solid`] so the
/// panel and the on-disk mask never disagree. Used by the Part inspector while the
/// mask-edit modal is open.
fn mask_layer_readout(state: &EditorState, p: &re_level::Part) -> String {
    let ly = state.mask_layer as i32;
    let (cur_x, cur_z) = state.mask_cursor;
    let mut rows: Vec<String> = Vec::new();
    for lz in 0..p.d.max(1) {
        let mut row = String::new();
        for lx in 0..p.w.max(1) {
            let ch = if crate::part::mask_solid(p, lx, ly, lz) {
                '#'
            } else {
                '.'
            };
            if lx as usize == cur_x && lz as usize == cur_z {
                row.push('[');
                row.push(ch);
                row.push(']');
            } else {
                row.push(' ');
                row.push(ch);
                row.push(' ');
            }
        }
        rows.push(row.trim().to_string());
    }
    format!("L{} {}", state.mask_layer, rows.join(" / "))
}

// ─── ROW-EDITOR OVERLAYS ──────────────────────────────────────────────────────
//
// The painters for the three typed-row editors. Every one of them takes its box
// from `rowedit`'s geometry fns — the SAME fns `mouse.rs` hit-tests against —
// so a click lands on exactly the swatch or row drawn under it, and every one
// declines to draw (rather than clipping into garbage) when the terminal is too
// small, matching the pre-existing modal painters.

/// Paint a filled, bordered overlay box and return its inner text origin.
fn draw_overlay_box(
    fb: &mut Framebuffer,
    (x0, y0, pw, ph): (usize, usize, usize, usize),
    title: &str,
    border: u32,
) -> (usize, usize) {
    let bg = 0x00_0C1018;
    let fg = 0x00_C8D4E0;
    for ry in 0..ph {
        for rx in 0..pw {
            let edge = rx == 0 || ry == 0 || rx == pw - 1 || ry == ph - 1;
            fb.set(
                x0 + rx,
                y0 + ry,
                Cell {
                    glyph: if edge { '#' as u32 } else { ' ' as u32 },
                    fg: if edge { border } else { fg },
                    bg,
                    ..Cell::default()
                },
            );
        }
    }
    let inner = x0 + 2;
    let t: String = title.chars().take(pw.saturating_sub(4)).collect();
    draw_str(fb, inner, y0 + 1, &t, border, bg);
    (inner, y0 + 2)
}

/// Paint one grid of colour swatches, highlighting `sel` when `active`. Uses
/// [`crate::rowedit::grid_cell`] so the painted position and the click target
/// are the same geometry.
fn draw_swatch_grid(
    fb: &mut Framebuffer,
    x: usize,
    y: usize,
    colors: &[u32],
    sel: usize,
    active: bool,
    max_rows: usize,
) {
    let bg = 0x00_0C1018;
    for (i, &c) in colors.iter().enumerate() {
        let (dx, dy) = crate::rowedit::grid_cell(i);
        if dy >= max_rows {
            break;
        }
        draw_str(
            fb,
            x + dx,
            y + dy,
            crate::rowvalue::SWATCH,
            c & 0xFF_FFFF,
            bg,
        );
        // The highlight goes in the swatch's own trailing GAP cell — the one
        // column `grid_hit` already excludes from every swatch. It cannot be
        // drawn UNDER the swatch (the earlier attempt): `grid_cell` allocates
        // exactly one row per grid row, so `dy + 1` always landed on the next
        // row of swatches, on the following caption, or on the box border.
        // Reverse-video on the swatch itself is out too — it would hide the
        // very colour the swatch exists to show.
        if active && i == sel {
            draw_str(
                fb,
                x + dx + crate::rowedit::SWATCH_W - 1,
                y + dy,
                "<",
                0x00_FFE060,
                bg,
            );
        }
    }
}

/// The INLINE NUMERIC editor overlay: the field's name, its clamp range, and
/// the caret buffer being typed.
pub fn draw_num_edit(state: &EditorState, fb: &mut Framebuffer, w: usize, h: usize) {
    let (Some(ed), Some(rect)) = (
        state.num_edit.as_ref(),
        crate::rowedit::num_box(state, w, h),
    ) else {
        return;
    };
    let bg = 0x00_0C1018;
    let (ix, mut iy) = draw_overlay_box(fb, rect, "EDIT NUMBER", 0x00_77AADD);
    let inner_w = rect.2.saturating_sub(4);
    let put = |fb: &mut Framebuffer, y: usize, t: &str, c: u32| {
        let t: String = t.chars().take(inner_w).collect();
        draw_str(fb, ix, y, &t, c, bg);
    };
    put(fb, iy, &ed.label, 0x00_9FB8D8);
    iy += 1;
    put(
        fb,
        iy,
        &crate::widget::caret_line_readout(&ed.buf, ed.caret),
        0x00_FFE060,
    );
    iy += 1;
    put(fb, iy, &ed.bounds.caption(), 0x00_7A8CA0);
    put(
        fb,
        rect.1 + rect.3.saturating_sub(2),
        "Enter commit  Esc cancel",
        0x00_8899AA,
    );
}

/// The ENUM OPTION popup: a scrollable list of the row's options with the
/// current one marked, windowed by the same `settings_scroll` keep-selection-
/// visible math every other list in the editor uses.
pub fn draw_enum_pick(state: &EditorState, fb: &mut Framebuffer, w: usize, h: usize) {
    let (Some(p), Some(rect)) = (
        state.enum_pick.as_ref(),
        crate::rowedit::enum_box(state, w, h),
    ) else {
        return;
    };
    let bg = 0x00_0C1018;
    let (ix, iy) = draw_overlay_box(fb, rect, &format!("SET {}", p.label), 0x00_60FFC0);
    let inner_w = rect.2.saturating_sub(4);
    let visible = crate::rowedit::enum_visible(rect.3);
    let scroll = settings_scroll(p.sel, p.options.len(), visible);
    for (vi, opt) in p.options.iter().skip(scroll).take(visible).enumerate() {
        let marked = scroll + vi == p.sel;
        let line: String = format!("{} {opt}", if marked { '\u{25B6}' } else { ' ' })
            .chars()
            .take(inner_w)
            .collect();
        draw_str(
            fb,
            ix,
            iy + vi,
            &line,
            if marked { 0x00_FFE060 } else { 0x00_C8D4E0 },
            bg,
        );
    }
    if scroll > 0 {
        draw_str(fb, ix + inner_w, iy, "\u{25B2}", 0x00_AEC4DA, bg);
    }
    if scroll + visible < p.options.len() {
        draw_str(
            fb,
            ix + inner_w,
            iy + visible - 1,
            "\u{25BC}",
            0x00_AEC4DA,
            bg,
        );
    }
    let hint: String = "Up/Dn move  Enter pick  Esc cancel"
        .chars()
        .take(inner_w)
        .collect();
    draw_str(
        fb,
        ix,
        rect.1 + rect.3.saturating_sub(2),
        &hint,
        0x00_8899AA,
        bg,
    );
}

/// The COLOUR overlay: a hex entry field, the 12-swatch palette grid, and the
/// level's own used colours — the three ways to reach a colour the 12-swatch
/// stepper alone could not.
pub fn draw_color_edit(state: &EditorState, fb: &mut Framebuffer, w: usize, h: usize) {
    let (Some(e), Some(l)) = (
        state.color_edit.as_ref(),
        crate::rowedit::color_layout(state, w, h),
    ) else {
        return;
    };
    use crate::rowedit::ColorFocus as CF;
    let bg = 0x00_0C1018;
    draw_overlay_box(fb, l.rect, &format!("COLOUR {}", e.label), 0x00_FF80C0);
    let ix = l.x;
    let inner_w = l.rect.2.saturating_sub(4);
    let sel_fg = |on: bool| if on { 0x00_FFE060 } else { 0x00_7A8CA0 };
    // Hex field + a live preview swatch of whatever it currently parses to.
    let hex: String = format!("#{}", crate::widget::caret_line_readout(&e.buf, e.caret))
        .chars()
        .take(inner_w.saturating_sub(3))
        .collect();
    draw_str(fb, ix, l.hex_y, &hex, sel_fg(e.focus == CF::Hex), bg);
    let preview = crate::rowvalue::parse_hex(&e.buf).unwrap_or(e.original);
    draw_str(
        fb,
        ix + inner_w.saturating_sub(2),
        l.hex_y,
        crate::rowvalue::SWATCH,
        preview,
        bg,
    );
    draw_str(
        fb,
        ix,
        l.pal_label_y,
        "palette",
        sel_fg(e.focus == CF::Palette),
        bg,
    );
    draw_swatch_grid(
        fb,
        ix,
        l.pal_y,
        &crate::material::PALETTE,
        e.pal_sel,
        e.focus == CF::Palette,
        l.pal_rows,
    );
    if let (Some(ly), Some(gy)) = (l.used_label_y, l.used_y) {
        draw_str(
            fb,
            ix,
            ly,
            "used in this level",
            sel_fg(e.focus == CF::Used),
            bg,
        );
        draw_swatch_grid(
            fb,
            ix,
            gy,
            &e.used,
            e.used_sel,
            e.focus == CF::Used,
            l.used_rows,
        );
    }
    let hint: String = "Tab section  click a swatch  Esc cancel"
        .chars()
        .take(inner_w)
        .collect();
    draw_str(fb, ix, l.hint_y, &hint, 0x00_8899AA, bg);
}

/// Paint the PROPERTY SHEET rows: the scrollable typed row list built by
/// [`inspector_rows`], windowed by [`inspector_viewport`] + [`settings_scroll`]
/// (focused row always visible), with `▲`/`▼` overflow indicators pinned at the
/// right edge of the top/bottom viewport rows when rows are hidden above/below.
fn draw_inspector_rows(state: &EditorState, fb: &mut Framebuffer, panel: crate::layout::Rect) {
    if panel.is_empty() {
        return;
    }
    let rows = inspector_rows(state);
    let (visible, _) = inspector_viewport(state, panel);
    if visible == 0 || rows.is_empty() {
        return;
    }
    let bg = 0x00_000000;
    let top = panel.y + INSPECTOR_ROWS_TOP;
    let focused_idx = rows.iter().position(|r| r.focused).unwrap_or(0);
    let scroll = settings_scroll(focused_idx, rows.len(), visible);
    let text_x = inspector_text_x(panel);
    // Per-context header accent (keeps the old prop-panel colour language).
    let header_fg = if state.material_mode {
        0x00_60FFC0
    } else if state.vox_mode {
        VOX_FG
    } else {
        0x00_FF80C0
    };
    for (vi, r) in rows.iter().skip(scroll).take(visible).enumerate() {
        let ry = top + vi;
        let mk = if r.focused { '\u{25B6}' } else { ' ' };
        let body = match r.kind {
            RowKind::Field => {
                crate::rowvalue::row_text_in(&r.label, &r.value, inspector_text_w(panel))
            }
            _ => r.label.clone(),
        };
        let mut line = format!("{mk}{body}");
        if line.chars().count() > panel.w {
            line = line.chars().take(panel.w).collect();
        }
        let color = if r.focused {
            0x00_FFE060
        } else {
            match r.kind {
                RowKind::Header => header_fg,
                RowKind::Info => 0x00_7A8CA0,
                RowKind::Field => 0x00_E0E8FF,
            }
        };
        draw_str(fb, panel.x, ry, &line, color, bg);
        // A COLOUR row's block is repainted in the colour it names — the whole
        // point of a swatch. The span comes from `rowvalue::swatch_span_in`, the
        // same layout the text above was built from, so the tint can never
        // drift off the block. Cells clipped by a narrow panel are skipped.
        if let (Some((off, len)), RowValue::Color(rgb)) = (
            crate::rowvalue::swatch_span_in(&r.label, &r.value, inspector_text_w(panel)),
            &r.value,
        ) {
            for i in 0..len {
                let cx = text_x + off + i;
                if cx < panel.x + panel.w {
                    draw_str(
                        fb,
                        cx,
                        ry,
                        crate::rowvalue::SWATCH_CELL,
                        rgb & 0xFF_FFFF,
                        bg,
                    );
                }
            }
        }
    }
    // Off-screen indicators (same recipe as the LEVEL SETTINGS form).
    let arrow_x = panel.x + panel.w.saturating_sub(1);
    if scroll > 0 {
        draw_str(fb, arrow_x, top, "\u{25B2}", 0x00_AEC4DA, bg);
    }
    if scroll + visible < rows.len() {
        draw_str(fb, arrow_x, top + visible - 1, "\u{25BC}", 0x00_AEC4DA, bg);
    }
}

/// The selected entity's compact one-line SUMMARY — the property sheet's
/// header row. Pure. Emitter rows are built directly in
/// [`inspector_rows`] (they need the emitter list), so Emitter is unreachable.
fn entity_summary(state: &EditorState, kind: EntityKind) -> String {
    let lvl = &state.level;
    match kind {
        EntityKind::Door => format!(
            "{},{},{} on:{} off:{}",
            lvl.door.x,
            lvl.door.y,
            lvl.door.z,
            lvl.door.target_parts.len(),
            lvl.door.target_off.len()
        ),
        EntityKind::Spawn => format!(
            "{},{},{} face:{}",
            lvl.player.x,
            lvl.player.y,
            lvl.player.z,
            facing_label(lvl.player.facing_yaw_col)
        ),
        EntityKind::Pedestal => {
            let p = &lvl.pedestal;
            format!(
                "{},{},{} vol:{}x{}x{}",
                p.cx,
                p.cy,
                p.cz,
                p.w.max(1),
                p.h.max(1),
                p.d.max(1)
            )
        }
        EntityKind::Node => match state.selected_node.filter(|&i| i < lvl.nodes.len()) {
            Some(i) => {
                let n = &lvl.nodes[i];
                format!(
                    "{} {} thr:{}",
                    n.name,
                    io_role_label(n.is_input, n.is_output),
                    n.threshold
                )
            }
            None => "place/select a node".to_string(),
        },
        EntityKind::Trigger => match state.selected_trigger.filter(|&i| i < lvl.triggers.len()) {
            Some(i) => {
                let t = &lvl.triggers[i];
                let ((x0, y0, z0), (x1, y1, z1)) = trigger_box_corners(t);
                format!(
                    "#{i} ({x0},{y0},{z0})-({x1},{y1},{z1}) {} once:{}",
                    crate::entity::trigger_action_label(state, t),
                    t.once
                )
            }
            None => "place/select a trigger".to_string(),
        },
        EntityKind::Pipe => match crate::pipe::selected_summary(state) {
            Some(s) => s,
            None => {
                if state.pipe_pending.is_some() {
                    "pipe: pick TARGET node (space/Enter)".to_string()
                } else {
                    "pipe: P on a node to wire".to_string()
                }
            }
        },
        EntityKind::ZoneInput => {
            match state
                .selected_zone_input
                .filter(|&i| i < lvl.zone_inputs.len())
            {
                Some(i) => {
                    let zi = &lvl.zone_inputs[i];
                    let node = if zi.node.is_empty() {
                        "(unlinked!)"
                    } else {
                        zi.node.as_str()
                    };
                    format!(
                        "#{i} ({},{},{})-({},{},{}) -> {node}",
                        zi.x0, zi.y0, zi.z0, zi.x1, zi.y1, zi.z1
                    )
                }
                None => "place/select a zone input".to_string(),
            }
        }
        EntityKind::Note => match state.selected_note.filter(|&i| i < lvl.notes.len()) {
            Some(i) => {
                let n = &lvl.notes[i];
                format!(
                    "#{i} {} ({},{},{}) face:{}",
                    crate::state::note_kind_label(n.kind),
                    n.x,
                    n.y,
                    n.z,
                    facing_label(n.facing)
                )
            }
            None => "place/select a note".to_string(),
        },
        EntityKind::WallDecal => match state.selected_decal.filter(|&i| i < lvl.decals.len()) {
            Some(i) => {
                let d = &lvl.decals[i];
                format!(
                    "#{i} {} ({},{},{})-({},{},{}) den:{:.1} head:#{:06X} tail:#{:06X}",
                    crate::state::decal_surface_label(d.surface_kind),
                    d.x0,
                    d.y0,
                    d.z0,
                    d.x1,
                    d.y1,
                    d.z1,
                    d.density,
                    d.color_head & 0xFF_FFFF,
                    d.color_tail & 0xFF_FFFF
                )
            }
            None => "place/select a wall decal".to_string(),
        },
        EntityKind::Holo => match state.selected_holo.filter(|&i| i < lvl.holos.len()) {
            Some(i) => {
                let h = &lvl.holos[i];
                format!(
                    "#{i} {} ({},{},{})-({},{},{}) style:{}",
                    crate::state::holo_shape_label(h.shape),
                    h.x0,
                    h.y0,
                    h.z0,
                    h.x1,
                    h.y1,
                    h.z1,
                    h.style
                )
            }
            None => "place/select a hologram".to_string(),
        },
        EntityKind::Portal => match state.selected_portal.filter(|&i| i < lvl.portals.len()) {
            Some(i) => {
                let p = &lvl.portals[i];
                let skin = if p.skin.is_empty() {
                    "(none)"
                } else {
                    p.skin.as_str()
                };
                format!(
                    "#{i} Region A ({},{},{})-({},{},{}) Region B ({},{},{})-({},{},{}) rotate:{} skin:{}",
                    p.ax0, p.ay0, p.az0, p.ax1, p.ay1, p.az1,
                    p.bx0, p.by0, p.bz0, p.bx1, p.by1, p.bz1,
                    p.rotate, skin
                )
            }
            None => "place/select a portal".to_string(),
        },
        EntityKind::Zone => match state.selected_zone.filter(|&i| i < lvl.overlay.zones.len()) {
            Some(i) => {
                let z = &lvl.overlay.zones[i];
                let ((x0, y0, z0), (x1, y1, z1)) = match z.region {
                    re_core::Region::Box { min, max } => (min, max),
                    _ => ((0, 0, 0), (0, 0, 0)),
                };
                let mat = if z.material.0.is_empty() {
                    "(none)"
                } else {
                    z.material.0.as_str()
                };
                format!(
                    "#{i} \"{}\" ({x0},{y0},{z0})-({x1},{y1},{z1}) mat:{mat} scale:{} insets:{}",
                    z.name,
                    crate::state::scale_rule_label(&z.scale),
                    z.insets.len()
                )
            }
            None => "place/select a zone".to_string(),
        },
        EntityKind::Line => match state.selected_line.filter(|&i| i < lvl.overlay.lines.len()) {
            Some(i) => {
                let l = &lvl.overlay.lines[i];
                let mat = if l.material.0.is_empty() {
                    "(none)"
                } else {
                    l.material.0.as_str()
                };
                format!(
                    "#{i} \"{}\" pts:{} mat:{mat} scale:{}",
                    l.name,
                    l.path.points.len(),
                    crate::state::scale_rule_label(&l.scale)
                )
            }
            None => "place/select a line (space adds waypoints)".to_string(),
        },
        EntityKind::Inset => match state
            .selected_inset
            .filter(|&i| i < lvl.overlay.insets.len())
        {
            Some(i) => {
                let ins = &lvl.overlay.insets[i];
                let ((x0, y0, z0), (x1, y1, z1)) = match ins.region {
                    re_core::Region::Box { min, max } => (min, max),
                    _ => ((0, 0, 0), (0, 0, 0)),
                };
                let re_level::OverlayInsetContent::ChildScene(child) = &ins.content;
                let child = if child.is_empty() {
                    "(empty)"
                } else {
                    child.as_str()
                };
                format!(
                    "#{i} \"{}\" oct:{}-{} ({x0},{y0},{z0})-({x1},{y1},{z1}) child:{child}",
                    ins.name, ins.at_octave.0, ins.at_octave.1
                )
            }
            None => "place/select an inset".to_string(),
        },
        EntityKind::Part => match state.selected_part.filter(|&i| i < lvl.parts.len()) {
            Some(i) => {
                let p = &lvl.parts[i];
                if state.mask_mode {
                    // While editing the mask, show the ACTIVE h-layer footprint as a
                    // `#`/`.` row strip (lx across, lz down, joined by `/`) with the
                    // cursor cell bracketed, so the modal has a readable readout.
                    mask_layer_readout(state, p)
                } else {
                    let solid = p.mask.iter().filter(|&&b| b != 0).count();
                    format!(
                        "\"{}\" {}x{}x{} @({},{},{}) solid:{} (q mask)",
                        p.name, p.w, p.h, p.d, p.off_x, p.off_y, p.off_z, solid
                    )
                }
            }
            None => "place/select a part".to_string(),
        },
        EntityKind::Field => match state.selected_field.filter(|&i| i < lvl.field_defs.len()) {
            Some(i) => {
                let name = &lvl.field_defs[i].name;
                use crate::state::EntityField as F;
                let archetype = crate::field::value_of(state, F::FieldArchetype);
                let shape = crate::field::value_of(state, F::FieldShapeKind);
                let radius = crate::field::value_of(state, F::FieldRadius);
                format!("\"{name}\" {archetype} {shape} r:{radius}")
            }
            None => "place/select a sound field".to_string(),
        },
        EntityKind::Emitter => unreachable!("emitter rows are built in inspector_rows"),
    }
}

/// Horizontal Z gauge: `[___|___]` style bar of `width` cells with a knob whose
/// position tracks `z` within `0..sz`. This is the height "ползунок".
fn draw_z_slider(state: &EditorState, fb: &mut Framebuffer, x0: usize, y0: usize, width: usize) {
    let (_sx, _sy, sz) = state.map.size();
    let inner = width.saturating_sub(2).max(1);
    let knob = view::z_slider_knob(state.z, sz, inner);
    let track_fg = 0x00_556677;
    let knob_fg = 0x00_FFD000;
    draw_str(fb, x0, y0, "[", track_fg, 0);
    for i in 0..inner {
        let (g, c) = if i == knob {
            ('#', knob_fg)
        } else {
            ('-', track_fg)
        };
        draw_str(fb, x0 + 1 + i, y0, &g.to_string(), c, 0);
    }
    draw_str(fb, x0 + 1 + inner, y0, "]", track_fg, 0);
}

/// Draw the elevation side-window: a vertical cross-section (Z-stack) at the
/// cursor column, top layer at the top, with the Z axis labelled and a numeric
/// readout of the cursor's world (x,y,z) + the cell kind there. Structured so a
/// 45° ramp would read as a slope: each layer is one row, the active layer
/// is marked, and a left-side `>` marks where geometry stands.
pub fn draw_elevation(
    state: &EditorState,
    fb: &mut Framebuffer,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
) {
    if w == 0 || h == 0 {
        return;
    }
    let label_fg = 0x00_88AACC;
    let bg = 0x00_000000;
    draw_str(fb, x0, y0, "ELEVATION", label_fg, bg);

    let (cx, cy) = state.cursor;
    let here = state.map.cell_at(cx as i32, cy as i32, state.z as i32);
    draw_str(
        fb,
        x0,
        y0 + 1,
        &format!("@{cx},{cy},{} {:?}", state.z, here),
        0x00_88FFAA,
        bg,
    );

    // The cross-section occupies the rows below the two header lines.
    let col = view::elevation_column(&state.map, cx, cy);
    let body_y0 = y0 + 2;
    let body_h = h.saturating_sub(2);
    if body_h == 0 || col.is_empty() {
        return;
    }
    // Render high-z first (top of the window = top of the world). Show at most
    // `body_h` layers, centred on the active z so tall stacks stay in view.
    let sz = col.len();
    let start = top_layer_for_window(state.z, sz, body_h);
    for row in 0..body_h {
        // world z for this row, counting down from `start`.
        if row > start {
            break;
        }
        let zz = start - row;
        let sample = col[zz];
        let py = body_y0 + row;
        let is_active = zz == state.z;
        let solid = sample.kind != CellKind::Empty;
        let marker = if is_active { '>' } else { ' ' };
        let zfg = if is_active { 0x00_FFD000 } else { RULER_FG };
        draw_str(fb, x0, py, &format!("{marker}{zz:>2} "), zfg, bg);
        let line_bg = if is_active { CHUNK_BG } else { bg };
        let strip_w = w.saturating_sub(4);
        // Shape-aware profile: a RampZ reads as a slanted slope (`/`/`\` filling
        // a triangle across the strip — "наклон по z"); a WedgeXY shows its
        // diagonal-wall marker; a Full solid is a flat glyph + fill.
        let (glyph, cell_fg) = if solid {
            match shape_glyph(sample.shape) {
                Some(g) => (g, shape_fg(sample.shape)),
                None => (glyph_for(sample.kind), fg_for(sample.kind)),
            }
        } else {
            (glyph_for(sample.kind), DIM_FG)
        };
        for i in 0..strip_w {
            let g = slope_profile_glyph(sample.shape, glyph, i, strip_w);
            fb.set(
                x0 + 4 + i,
                py,
                Cell {
                    glyph: g as u32,
                    fg: cell_fg,
                    bg: line_bg,
                    ..Cell::default()
                },
            );
        }
    }
}

/// Pick the glyph for column `i` (of `strip_w`) of an elevation profile row.
///
/// For a [`Shape::RampZ`] the strip reads as a 45° SLOPE: a rising ramp draws
/// `/` characters that march up-and-across the strip; a falling ramp draws `\`.
/// `+X`/`+Y` rise toward the east/south end of the strip (rightward `/`);
/// `-X`/`-Y` rise toward the west/north (leftward `\`). For every other shape
/// (Full cube, WedgeXY) the profile is the leading glyph then blank fill (a flat
/// block), preserving the prior look.
fn slope_profile_glyph(shape: Shape, lead: char, i: usize, strip_w: usize) -> char {
    let _ = (i, strip_w);
    match shape {
        // A ramp row is filled with the slope stroke so the tilt is unmistakable:
        // `/` when it rises toward the +X/+Y (east/south) end, `\` toward -X/-Y.
        Shape::RampZ(dir) => {
            if matches!(dir, RampDir::PosX | RampDir::PosY) {
                '/'
            } else {
                '\\'
            }
        }
        // Full cube / WedgeXY: leading glyph then blank fill (flat block).
        _ => {
            if i == 0 {
                lead
            } else {
                ' '
            }
        }
    }
}

/// Top world-z to show in an elevation window of `body_h` rows, centred on the
/// active `z` and clamped so the window stays within `0..sz`.
fn top_layer_for_window(z: usize, sz: usize, body_h: usize) -> usize {
    if sz <= body_h {
        return sz.saturating_sub(1); // whole stack fits; top is the top layer
    }
    // Centre the active layer; bias the top so `z` sits mid-window.
    let half = body_h / 2;
    (z + half).min(sz - 1).max(body_h - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1.2: the help cheat-sheet lists EVERY toolbar button — including the
    /// previously-omitted `[Undo]`, `[Redo]`, and `[Level]` — plus the new `E`
    /// (Level Settings) chord. Labels mirror `toolbar::button_label`.
    #[test]
    fn help_lists_every_toolbar_button_and_settings_chord() {
        let joined = help_lines().join("\n");
        for needle in [
            "[New level]",
            "[Open]",
            "[Save]",
            "[Save As]",
            "[Undo]",
            "[Redo]",
            "[2D view]",
            "[3D view]",
            "[Split view]",
            "[Level]",
            "[Help]",
        ] {
            assert!(joined.contains(needle), "help must list toolbar {needle}");
        }
        assert!(
            joined.contains("E level settings"),
            "help must list the `E` Level-Settings chord"
        );
    }

    /// 1.2: the status bar's trailing key hint advertises the settings chord
    /// alongside help (secondary affordance, muted color).
    #[test]
    fn status_bar_hint_advertises_settings_chord() {
        let state = EditorState::new_empty(4, 4, 2);
        let mut fb = Framebuffer::new(200, 30);
        let rect = crate::layout::Rect {
            x: 0,
            y: 29,
            w: 200,
            h: 1,
        };
        draw_status_bar(&state, &mut fb, rect);
        let row: String = (0..200)
            .filter_map(|x| fb.get(x, 29).and_then(|c| char::from_u32(c.glyph)))
            .collect();
        assert!(row.contains("?=help"), "hint keeps ?=help");
        assert!(row.contains("E=settings"), "hint advertises E=settings");
    }

    #[test]
    fn cursor_cell_is_highlighted() {
        let mut state = EditorState::new_empty(4, 4, 2);
        state.cursor = (2, 1);
        let mut fb = Framebuffer::new(40, 20);
        draw_grid(&state, &mut fb, 0, 0, 40, 20);
        // With rulers reserved, world cell (2,1) maps to a framebuffer block.
        let px = RULER_W + 2 * state.zoom as usize;
        let py = RULER_H + state.zoom as usize;
        assert_eq!(fb.get(px, py).unwrap().bg, CURSOR_BG);
        // The origin world cell is not the cursor.
        let ox = RULER_W;
        let oy = RULER_H;
        assert_ne!(fb.get(ox, oy).unwrap().bg, CURSOR_BG);
    }

    #[test]
    fn pipe_path_cells_render_in_top_down() {
        let mut state = EditorState::new_empty(12, 12, 2);
        state.level.push_node(1, 1, 0);
        state.level.push_node(4, 1, 0);
        state.level.push_pipe_between(0, 1).expect("pipe wired");
        state.z = 0;
        // Make sure the cursor isn't on the path cell we test (so its bg is plain).
        state.cursor = (10, 10);
        let mut fb = Framebuffer::new(40, 20);
        draw_grid(&state, &mut fb, 0, 0, 40, 20);
        // A mid-path cell (2,1,0) along the straight x-leg shows the pipe dot glyph.
        let px = RULER_W + 2 * state.zoom as usize;
        let py = RULER_H + state.zoom as usize;
        assert_eq!(
            fb.get(px, py).unwrap().glyph,
            '\u{00B7}' as u32,
            "pipe path cell draws the middle-dot trace"
        );
        assert_eq!(
            fb.get(px, py).unwrap().fg,
            PIPE_FG,
            "unselected pipe is dim"
        );
    }

    #[test]
    fn show_objects_false_hides_marker_but_keeps_geometry() {
        // A node placed on a solid wall cell. With show_objects ON the node glyph
        // 'N' wins; with it OFF the geometry '#' shows through and no marker paints.
        let mut state = EditorState::new_empty(8, 8, 2);
        state.map.set(2, 1, 0, CellKind::WallSolid);
        state.level.push_node(2, 1, 0);
        state.z = 0;
        state.cursor = (7, 7); // keep the cursor off the tested cell
        let px = RULER_W + 2 * state.zoom as usize;
        let py = RULER_H + state.zoom as usize;

        // show_objects ON → the entity marker overrides the geometry glyph.
        assert!(state.show_objects);
        let mut fb = Framebuffer::new(40, 20);
        draw_grid(&state, &mut fb, 0, 0, 40, 20);
        assert_eq!(
            fb.get(px, py).unwrap().glyph,
            'N' as u32,
            "marker shown when on"
        );

        // show_objects OFF → only the geometry glyph remains, no marker.
        state.show_objects = false;
        let mut fb2 = Framebuffer::new(40, 20);
        draw_grid(&state, &mut fb2, 0, 0, 40, 20);
        assert_eq!(
            fb2.get(px, py).unwrap().glyph,
            glyph_for(CellKind::WallSolid) as u32,
            "geometry still painted when objects hidden"
        );
        assert_eq!(
            fb2.get(px, py).unwrap().fg,
            fg_for(CellKind::WallSolid),
            "geometry colour, not the marker colour, when objects hidden"
        );
    }

    /// Helper: render the OUTLINER into a fresh fb and return its rows as text.
    fn outliner_text(state: &EditorState, w: usize, h: usize) -> String {
        let mut fb = Framebuffer::new(w, h);
        draw_tree(state, &mut fb, 0, 0, w, h);
        let mut out = String::new();
        for r in 0..h {
            for c in 0..w {
                if let Some(cell) = fb.get(c, r) {
                    out.push(char::from_u32(cell.glyph).unwrap_or(' '));
                }
            }
            out.push('\n');
        }
        out
    }

    #[test]
    fn outliner_group_headers_carry_live_counts() {
        let mut state = EditorState::new_empty(16, 16, 2);
        state.level.push_node(1, 1, 0);
        state.level.push_node(4, 1, 0);
        state.level.push_pipe_between(0, 1).expect("pipe wired");
        state.level.door.x = 5;
        state.level.player.y = 6;
        let all = outliner_text(&state, 30, 30);
        assert!(all.contains("OUTLINER"), "organ header present:\n{all}");
        assert!(all.contains("Show objects"), "checkbox present");
        assert!(all.contains("Add object"), "create affordance present");
        // Live counts ride on the group headers (`▾ Nodes (2)` etc.).
        assert!(all.contains("Nodes (2)"), "node count = 2:\n{all}");
        assert!(all.contains("Pipes (1)"), "pipe count = 1");
        assert!(all.contains("Doors (1)"), "door singleton counts 1");
        assert!(all.contains("Spawn (1)"), "spawn singleton counts 1");
        assert!(all.contains("Pedestal (0)"), "no pedestal → 0");
        assert!(all.contains("Emitters (0)"), "no emitters → 0");
    }

    #[test]
    fn outliner_checkbox_reflects_show_objects_state() {
        let mut state = EditorState::new_empty(8, 8, 2);
        assert!(state.show_objects);
        assert!(
            outliner_text(&state, 30, 20).contains("[x] Show objects"),
            "checked when shown"
        );
        state.show_objects = false;
        assert!(
            outliner_text(&state, 30, 20).contains("[ ] Show objects"),
            "unchecked when hidden"
        );
    }

    #[test]
    fn selected_pipe_switch_cell_marks_brighter() {
        let mut state = EditorState::new_empty(12, 12, 2);
        state.level.push_node(1, 1, 0);
        state.level.push_node(4, 1, 0);
        state.level.push_pipe_between(0, 1).expect("pipe wired");
        // Mark the switch at path index 2 = cell (3,1,0), and select the pipe.
        state.level.pipes[0].switch_cell = Some(2);
        state.selected_pipe = Some(0);
        state.z = 0;
        state.cursor = (10, 10);
        let mut fb = Framebuffer::new(40, 20);
        draw_grid(&state, &mut fb, 0, 0, 40, 20);
        // The switch cell (3,1,0) shows a `+` in the switch colour.
        let sx = RULER_W + 3 * state.zoom as usize;
        let sy = RULER_H + state.zoom as usize;
        assert_eq!(
            fb.get(sx, sy).unwrap().glyph,
            '+' as u32,
            "switch cell marked +"
        );
        assert_eq!(fb.get(sx, sy).unwrap().fg, PIPE_SWITCH_FG);
        // A non-switch path cell (2,1,0) of the SELECTED pipe is brighter.
        let px = RULER_W + 2 * state.zoom as usize;
        assert_eq!(
            fb.get(px, sy).unwrap().fg,
            PIPE_SEL_FG,
            "selected pipe is bright"
        );
    }

    #[test]
    fn zoom_makes_blocks_bigger() {
        let mut state = EditorState::new_empty(8, 8, 2);
        state.zoom = 2;
        state.cursor = (1, 0);
        let mut fb = Framebuffer::new(40, 20);
        draw_grid(&state, &mut fb, 0, 0, 40, 20);
        // World cell (1,0) at zoom 2 covers a 2x2 block; all four cursor cells
        // carry the highlight bg.
        let bx = RULER_W + 2;
        let by = RULER_H;
        assert_eq!(fb.get(bx, by).unwrap().bg, CURSOR_BG);
        assert_eq!(fb.get(bx + 1, by + 1).unwrap().bg, CURSOR_BG);
    }

    #[test]
    fn height_overlay_ghosts_lower_layers() {
        let mut state = EditorState::new_empty(4, 4, 3);
        // Solid below, active layer empty → resolves to a dimmed ghost glyph.
        state.map.set(1, 1, 0, CellKind::WallSolid);
        state.z = 2;
        let (glyph, fg, ghost) = resolve_cell(&state, 1, 1);
        assert_eq!(glyph, '#');
        assert!(ghost, "lower layer shows as a ghost");
        assert!(fg < fg_for(CellKind::WallSolid), "ghost is dimmed");
    }

    #[test]
    fn active_layer_takes_precedence_over_ghost() {
        let mut state = EditorState::new_empty(4, 4, 3);
        state.map.set(1, 1, 0, CellKind::WallSolid); // below
        state.map.set(1, 1, 2, CellKind::Floor); // active
        state.z = 2;
        let (glyph, _fg, ghost) = resolve_cell(&state, 1, 1);
        assert_eq!(glyph, '_');
        assert!(!ghost);
    }

    #[test]
    fn elevation_window_renders_cursor_readout() {
        let mut state = EditorState::new_empty(6, 6, 4);
        state.cursor = (3, 2);
        state.z = 1;
        state.map.set(3, 2, 0, CellKind::Floor);
        let mut fb = Framebuffer::new(20, 14);
        draw_elevation(&state, &mut fb, 0, 0, 20, 14);
        // Header line 0 says ELEVATION; line 1 carries the world coords.
        assert_eq!(fb.get(0, 0).unwrap().glyph, 'E' as u32);
        assert_eq!(fb.get(0, 1).unwrap().glyph, '@' as u32);
    }

    #[test]
    fn shaped_cell_uses_orientation_glyph_top_down() {
        let mut state = EditorState::new_empty(4, 4, 2);
        state.map.set(2, 1, 0, CellKind::WallSolid);
        state.map.set_shape(2, 1, 0, Shape::WedgeXY(DiagXY::SE));
        // resolve_cell at the active layer returns the wedge marker, not '#'.
        let (glyph, fg, ghost) = resolve_cell(&state, 2, 1);
        assert_eq!(glyph, shape_glyph(Shape::WedgeXY(DiagXY::SE)).unwrap());
        assert_ne!(glyph, '#', "shaped cell overrides the plain solid glyph");
        assert_eq!(fg, shape_fg(Shape::WedgeXY(DiagXY::SE)));
        assert!(!ghost);
    }

    #[test]
    fn each_orientation_has_a_distinct_glyph() {
        use crate::state::SHAPE_CYCLE;
        let mut seen = std::collections::HashSet::new();
        for &s in &SHAPE_CYCLE {
            let g = shape_glyph(s).expect("non-Full has a glyph");
            assert!(
                seen.insert(g),
                "glyph {g:?} is not unique across orientations"
            );
        }
        assert_eq!(seen.len(), 8);
        assert!(shape_glyph(Shape::Full).is_none());
    }

    #[test]
    fn ramp_reads_as_a_slope_in_elevation() {
        // A RampZ at the cursor fills its elevation row with a slope stroke.
        let mut state = EditorState::new_empty(4, 4, 3);
        state.cursor = (1, 1);
        state.z = 0;
        state.map.set(1, 1, 0, CellKind::WallSolid);
        state.map.set_shape(1, 1, 0, Shape::RampZ(RampDir::PosX));
        // PosX rises east → forward slope '/'.
        assert_eq!(
            slope_profile_glyph(Shape::RampZ(RampDir::PosX), '#', 3, 8),
            '/'
        );
        // NegX rises west → back slope '\'.
        assert_eq!(
            slope_profile_glyph(Shape::RampZ(RampDir::NegX), '#', 3, 8),
            '\\'
        );
        // A Full cube keeps the flat profile (glyph then blanks).
        assert_eq!(slope_profile_glyph(Shape::Full, '#', 0, 8), '#');
        assert_eq!(slope_profile_glyph(Shape::Full, '#', 3, 8), ' ');

        // Render and confirm a slope glyph lands in the active row's strip.
        let mut fb = Framebuffer::new(20, 14);
        draw_elevation(&state, &mut fb, 0, 0, 20, 14);
        // Active layer z=0 is the bottom row of the window; scan for a '/'.
        let mut found = false;
        for py in 0..14 {
            for px in 0..20 {
                if fb.get(px, py).map(|c| c.glyph) == Some('/' as u32) {
                    found = true;
                }
            }
        }
        assert!(found, "elevation window shows the ramp slope stroke");
    }

    #[test]
    fn entity_markers_render_on_active_layer() {
        let mut state = EditorState::new_empty(8, 8, 4);
        state.z = 1;
        // Place a door, a spawn, a pedestal, and a node on the active layer.
        state.level.door.x = 2;
        state.level.door.y = 2;
        state.level.door.z = 1;
        state.level.player.x = 3;
        state.level.player.y = 2;
        state.level.player.z = 1;
        state.level.pedestal.cx = 4;
        state.level.pedestal.cy = 2;
        state.level.pedestal.cz = 1;
        state.level.pedestal.w = 1;
        let _ = state.level.push_node(5, 2, 1);

        // Marker resolution returns the distinct glyphs.
        assert_eq!(entity_marker_at(&state, 2, 2, 1).unwrap().glyph, 'D');
        assert_eq!(entity_marker_at(&state, 3, 2, 1).unwrap().glyph, '@');
        assert_eq!(entity_marker_at(&state, 4, 2, 1).unwrap().glyph, 'X');
        assert_eq!(entity_marker_at(&state, 5, 2, 1).unwrap().glyph, 'N');
        // An empty cell has no marker.
        assert!(entity_marker_at(&state, 0, 0, 1).is_none());
        // The door at the level origin (0,0,0) is "unset" and must NOT render.
        let mut s2 = EditorState::new_empty(8, 8, 4);
        s2.z = 0;
        assert!(
            entity_marker_at(&s2, 0, 0, 0).is_none(),
            "unset door at origin shows no marker"
        );

        // The grid actually paints the door glyph somewhere.
        let mut fb = Framebuffer::new(60, 30);
        draw_grid(&state, &mut fb, 0, 0, 60, 30);
        let mut saw_door = false;
        for py in 0..30 {
            for px in 0..60 {
                if fb.get(px, py).map(|c| c.glyph) == Some('D' as u32) {
                    saw_door = true;
                }
            }
        }
        assert!(saw_door, "door marker painted into the top-down grid");
    }

    #[test]
    fn off_layer_entity_marker_is_dimmed() {
        let mut state = EditorState::new_empty(8, 8, 4);
        // Node one layer below the active z.
        let _ = state.level.push_node(3, 3, 0);
        state.z = 1;
        // On the active layer the cell has no entity, but the dim off-layer pass
        // (inside draw_grid) should still surface a node glyph below.
        assert!(entity_marker_at(&state, 3, 3, 1).is_none());
        assert!(entity_marker_at(&state, 3, 3, 0).is_some());
    }

    #[test]
    fn cam_arrow_maps_yaw_to_eight_screen_directions() {
        use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};
        // Convention: dir = (sin yaw, cos yaw); screen +y is DOWN, so yaw 0 = ↓.
        // Cardinals, then diagonals, sweeping yaw which turns the facing vector.
        assert_eq!(cam_arrow_glyph(0.0), '\u{2193}', "yaw 0 → +Y → ↓ (down)");
        assert_eq!(
            cam_arrow_glyph(FRAC_PI_2),
            '\u{2192}',
            "yaw π/2 → +X → → (right)"
        );
        assert_eq!(cam_arrow_glyph(PI), '\u{2191}', "yaw π → -Y → ↑ (up)");
        assert_eq!(
            cam_arrow_glyph(-FRAC_PI_2),
            '\u{2190}',
            "yaw -π/2 → -X → ← (left)"
        );
        // Diagonals at the 45° midpoints.
        assert_eq!(cam_arrow_glyph(FRAC_PI_4), '\u{2198}', "yaw π/4 → +X+Y → ↘");
        assert_eq!(
            cam_arrow_glyph(3.0 * FRAC_PI_4),
            '\u{2197}',
            "yaw 3π/4 → +X-Y → ↗"
        );
        assert_eq!(
            cam_arrow_glyph(-FRAC_PI_4),
            '\u{2199}',
            "yaw -π/4 → -X+Y → ↙"
        );
        assert_eq!(
            cam_arrow_glyph(-3.0 * FRAC_PI_4),
            '\u{2196}',
            "yaw -3π/4 → -X-Y → ↖"
        );
    }

    #[test]
    fn camera_arrow_drawn_at_camera_cell_on_active_layer() {
        let mut state = EditorState::new_empty(8, 8, 4);
        state.z = 1;
        // Camera centred in cell (2, 3) on the active layer, facing +Y (yaw 0 → ↓).
        state.cam.pos = re_world::CameraPose::spawn_at(2, 3, 1, 0).pos;
        state.cam.yaw = 0.0;
        // Keep the cursor elsewhere so it doesn't sit on the camera cell.
        state.cursor = (7, 7);
        let mut fb = Framebuffer::new(60, 30);
        draw_grid(&state, &mut fb, 0, 0, 60, 30);
        // Camera cell (2,3) at zoom 1 maps to (RULER_W + 2, RULER_H + 3).
        let px = RULER_W + 2 * state.zoom as usize;
        let py = RULER_H + 3 * state.zoom as usize;
        assert_eq!(
            fb.get(px, py).unwrap().glyph,
            '\u{2193}' as u32,
            "camera arrow (down) painted at its grid cell"
        );
        assert_eq!(
            fb.get(px, py).unwrap().fg,
            CAM_FG,
            "on-layer camera arrow is bright"
        );
    }

    #[test]
    fn off_layer_camera_shows_dim_diamond() {
        let mut state = EditorState::new_empty(8, 8, 4);
        state.z = 2;
        // Camera on layer 0 while viewing layer 2 → off-layer ◇ marker.
        state.cam.pos = re_world::CameraPose::spawn_at(1, 1, 0, 0).pos;
        state.cursor = (7, 7);
        let mut fb = Framebuffer::new(60, 30);
        draw_grid(&state, &mut fb, 0, 0, 60, 30);
        let px = RULER_W + state.zoom as usize;
        let py = RULER_H + state.zoom as usize;
        assert_eq!(
            fb.get(px, py).unwrap().glyph,
            '\u{25C7}' as u32,
            "off-layer ◇"
        );
        assert_eq!(
            fb.get(px, py).unwrap().fg,
            CAM_OFFLAYER_FG,
            "off-layer is dim"
        );
    }

    #[test]
    fn collisions_checkbox_reflects_noclip() {
        let mut state = EditorState::new_empty(4, 4, 2);
        state.noclip = false; // collisions ON
        assert_eq!(collisions_checkbox(&state), "[x] Collisions");
        state.noclip = true; // collisions OFF
        assert_eq!(collisions_checkbox(&state), "[ ] Collisions");
    }

    #[test]
    fn ruler_label_shows_full_number() {
        // The full coordinate, never truncated (1024 must NOT become "024").
        assert_eq!(ruler_label(7), "7");
        assert_eq!(ruler_label(128), "128");
        assert_eq!(ruler_label(1234), "1234");
        assert_eq!(ruler_label(1024), "1024");
    }

    /// Paint the HELP sheet into a fresh framebuffer and read it back as text
    /// rows — the sheet exactly as a terminal of that size would show it.
    fn render_help(state: &EditorState, w: usize, h: usize) -> Vec<String> {
        let mut fb = Framebuffer::new(w, h);
        draw_help(state, &mut fb, w, h);
        (0..h)
            .map(|y| {
                let row: String = (0..w)
                    .map(|x| {
                        fb.get(x, y)
                            .and_then(|c| char::from_u32(c.glyph))
                            .unwrap_or(' ')
                    })
                    .collect();
                row.trim_end().to_string()
            })
            .collect()
    }

    /// The BODY rows of the painted sheet (between the filter band and the
    /// footer), with the box border and left padding stripped.
    fn help_body(state: &EditorState, w: usize, h: usize) -> Vec<String> {
        let lay = help_layout(state, w, h).expect("the sheet fits at this size");
        let rows = render_help(state, w, h);
        (0..lay.body_h)
            .map(|i| {
                let row = &rows[lay.body_y + i];
                let start = lay.text_x.min(row.len());
                let end = (lay.text_x + lay.text_w).min(row.len());
                row[start..end].trim_end().to_string()
            })
            .collect()
    }

    /// THE DEFECT, as an invariant: at a realistic terminal size EVERY generated
    /// help line can be brought on screen by scrolling. The old painter clipped
    /// at `inner_h` with no offset, so ~140 of ~178 lines — the whole `emitter`
    /// and `3D fly` tail — were unreachable from the surface whose entire job is
    /// to list them.
    #[test]
    fn every_help_line_is_reachable_by_scrolling() {
        let (w, h) = (120usize, 40usize);
        let mut s = EditorState::new_empty(8, 8, 2);
        s.open_help();
        let lines = crate::keymap::help_lines();
        let body_h = help_layout(&s, w, h).expect("fits").body_h;
        s.help_page = body_h;
        let mut seen: Vec<bool> = vec![false; lines.len()];
        let mut scroll = 0usize;
        while scroll < lines.len() {
            s.help_scroll = scroll;
            let body = help_body(&s, w, h);
            for (i, want) in lines.iter().enumerate() {
                if body.iter().any(|painted| painted == want.trim_end()) {
                    seen[i] = true;
                }
            }
            scroll += body_h;
        }
        let missing: Vec<&String> = lines
            .iter()
            .zip(&seen)
            .filter(|(_, ok)| !**ok)
            .map(|(l, _)| l)
            .collect();
        assert!(
            missing.is_empty(),
            "{} of {} help lines can never be painted at {w}x{h}: {:?}",
            missing.len(),
            lines.len(),
            &missing[..missing.len().min(8)]
        );
        // And the tail specifically — the groups the old clip ate.
        assert!(lines.iter().any(|l| l.contains("-- emitter --")), "premise");
    }

    /// SCROLL CLAMP: the last page IS the last page. Scrolling far past the end
    /// must leave the final line on screen, not an empty box.
    #[test]
    fn help_scroll_clamps_to_the_last_page() {
        assert_eq!(clamp_help_scroll(0, 10, 4), 0);
        assert_eq!(clamp_help_scroll(3, 10, 4), 3);
        assert_eq!(clamp_help_scroll(6, 10, 4), 6, "last legal page");
        assert_eq!(clamp_help_scroll(7, 10, 4), 6, "one past the end clamps");
        assert_eq!(clamp_help_scroll(usize::MAX, 10, 4), 6, "End clamps");
        assert_eq!(clamp_help_scroll(9, 3, 4), 0, "everything fits: no scroll");
        assert_eq!(clamp_help_scroll(9, 10, 0), 0, "no window: no scroll");

        let (w, h) = (120usize, 40usize);
        let mut s = EditorState::new_empty(8, 8, 2);
        s.open_help();
        s.help_scroll = 100_000;
        let lines = crate::keymap::help_lines();
        let body = help_body(&s, w, h);
        assert_eq!(
            body.last().map(String::as_str),
            lines.last().map(|l| l.trim_end()),
            "the last body row shows the last help line"
        );
        assert!(
            body.iter().filter(|r| r.is_empty()).count() <= 1,
            "no run of blank rows past the end: {body:?}"
        );
    }

    /// BAND SEPARATION (the B1 shape): title, filter, body and footer each own
    /// their own row, and the box height is exactly the bands plus the chrome.
    /// A footer that overpaints the last content row costs a line of the very
    /// document it is measuring.
    #[test]
    fn help_bands_never_share_a_row() {
        let mut s = EditorState::new_empty(8, 8, 2);
        s.open_help();
        for (w, h) in [(120usize, 40usize), (80, 30), (60, 12), (40, 8)] {
            let Some(lay) = help_layout(&s, w, h) else {
                continue;
            };
            assert!(lay.y0 < lay.title_y, "title is below the top border");
            assert!(lay.title_y < lay.filter_y, "filter band below the title");
            assert!(lay.filter_y < lay.body_y, "body below the filter band");
            assert_eq!(
                lay.body_y + lay.body_h,
                lay.footer_y,
                "the footer starts exactly where the body ends ({w}x{h})"
            );
            assert_eq!(
                lay.footer_y + 1,
                lay.y0 + lay.ph - 1,
                "footer sits above the bottom border"
            );
            assert_eq!(
                lay.ph,
                lay.body_h + HELP_CHROME_ROWS,
                "box height = body + chrome, one arithmetic ({w}x{h})"
            );
            // And the painted proof: the footer row really carries the readout.
            let rows = render_help(&s, w, h);
            assert!(
                rows[lay.footer_y].contains(" of "),
                "the footer row shows the position readout at {w}x{h}: {:?}",
                rows[lay.footer_y]
            );
        }
    }

    /// The position footer names the window the user is looking at.
    #[test]
    fn help_footer_reports_the_visible_window() {
        let mut s = EditorState::new_empty(8, 8, 2);
        s.open_help();
        assert_eq!(help_position_text(0, 35, 178), "-- 1-35 of 178 lines --");
        assert_eq!(help_position_text(40, 35, 178), "-- 41-75 of 178 lines --");
        assert_eq!(
            help_position_text(170, 35, 178),
            "-- 171-178 of 178 lines --"
        );
        let (w, h) = (120usize, 40usize);
        let lay = help_layout(&s, w, h).expect("fits");
        let rows = render_help(&s, w, h);
        assert!(
            rows[lay.footer_y].contains(&format!("of {} lines", lay.total)),
            "the painted footer counts the whole sheet: {:?}",
            rows[lay.footer_y]
        );
        assert!(
            rows[lay.footer_y].contains("?/Esc close"),
            "and teaches the keys without truncating them away: {:?}",
            rows[lay.footer_y]
        );
        // ...at BOTH ends of the sheet: the position readout is widest on the
        // last page, and a footer whose tail falls off the border there teaches
        // the keys only to a reader who has not scrolled yet.
        s.help_scroll = usize::MAX;
        let bottom = render_help(&s, w, h);
        let lay = help_layout(&s, w, h).expect("fits");
        assert!(
            bottom[lay.footer_y].contains("?/Esc close"),
            "the footer survives on the LAST page too: {:?}",
            bottom[lay.footer_y]
        );
        assert!(
            bottom[lay.footer_y].contains(&format!(
                "{}-{} of",
                lay.total - lay.body_h + 1,
                lay.total
            )),
            "and reads the last window: {:?}",
            bottom[lay.footer_y]
        );
    }

    /// FILTER: the band shows the query and the match count, and the body really
    /// shrinks to the matches. A group banner survives only with a row under it.
    #[test]
    fn help_filter_narrows_the_sheet_and_shows_its_count() {
        let (w, h) = (120usize, 40usize);
        let mut s = EditorState::new_empty(8, 8, 2);
        s.open_help();
        let full = help_layout(&s, w, h).expect("fits").total;
        s.help_filter = "mask".into();
        let lay = help_layout(&s, w, h).expect("fits");
        assert!(
            lay.total < full,
            "the filter removes lines: {} of {full}",
            lay.total
        );
        let rows = render_help(&s, w, h);
        assert!(
            rows[lay.filter_y].contains("filter: mask"),
            "the band echoes the query: {:?}",
            rows[lay.filter_y]
        );
        assert!(
            rows[lay.filter_y]
                .contains(&format!("{} of ", crate::keymap::help_match_count("mask"))),
            "and the match count: {:?}",
            rows[lay.filter_y]
        );
        let body = help_body(&s, w, h);
        for line in body.iter().filter(|l| !l.is_empty()) {
            assert!(
                line.to_lowercase().contains("mask") || line.starts_with("--"),
                "every surviving line matches or is a banner: {line:?}"
            );
        }
        // A banner with nothing under it is worse than no banner.
        for (i, line) in body.iter().enumerate() {
            if line.starts_with("--") {
                assert!(
                    body.get(i + 1)
                        .is_some_and(|n| !n.starts_with("--") && !n.is_empty()),
                    "banner {line:?} has a row under it"
                );
            }
        }
    }

    /// The box does not resize horizontally while a filter is typed — a box that
    /// jumps under the pointer on every keystroke is unusable with the mouse,
    /// and the `[X]` hit-test rides on this width.
    #[test]
    fn help_box_width_is_stable_across_filters() {
        let (w, h) = (120usize, 40usize);
        let mut s = EditorState::new_empty(8, 8, 2);
        s.open_help();
        let wide = help_layout(&s, w, h).expect("fits").pw;
        for q in ["m", "mask", "zzzzz"] {
            s.help_filter = q.into();
            assert_eq!(
                help_layout(&s, w, h).expect("fits").pw,
                wide,
                "width stable for {q:?}"
            );
        }
    }

    /// The `[X]` hit-test and the painter read ONE geometry function.
    #[test]
    fn help_close_button_sits_on_the_painted_box() {
        let (w, h) = (120usize, 40usize);
        let mut s = EditorState::new_empty(8, 8, 2);
        s.open_help();
        let (x0, y0, pw, _ph) = help_box_geom(&s, w, h).expect("fits");
        let lay = help_layout(&s, w, h).expect("fits");
        assert_eq!(
            (x0, y0, pw),
            (lay.x0, lay.y0, lay.pw),
            "one geometry, two callers"
        );
        let (bx_y, bx_s, _bx_e) = modal_close_button(x0, y0, pw);
        let rows = render_help(&s, w, h);
        assert!(
            rows[bx_y][bx_s..].starts_with("[X]"),
            "the [X] is painted where it is hit-tested: {:?}",
            rows[bx_y]
        );
    }

    /// The SCROLLBAR thumb tracks the scroll and vanishes when nothing scrolls.
    #[test]
    fn help_scrollbar_tracks_the_scroll() {
        let (w, h) = (120usize, 40usize);
        let mut s = EditorState::new_empty(8, 8, 2);
        s.open_help();
        let lay = help_layout(&s, w, h).expect("fits");
        let col = lay.x0 + lay.pw - 1;
        let thumb_rows = |st: &EditorState| -> Vec<usize> {
            let rows = render_help(st, w, h);
            (0..lay.body_h)
                .filter(|i| rows[lay.body_y + i].chars().nth(col) == Some('|'))
                .collect()
        };
        let top = thumb_rows(&s);
        assert!(
            !top.is_empty(),
            "a thumb is painted while the sheet overflows"
        );
        assert_eq!(
            top.first(),
            Some(&0),
            "at the top the thumb starts at the top"
        );
        s.help_scroll = lay.total; // clamped to the last page by the painter
        let bottom = thumb_rows(&s);
        assert_eq!(
            bottom.last(),
            Some(&(lay.body_h - 1)),
            "at the end the thumb reaches the bottom"
        );
        // A filter short enough to fit leaves no thumb (nothing to scroll).
        s.help_scroll = 0;
        s.help_filter = "staircase".into();
        let lay2 = help_layout(&s, w, h).expect("fits");
        assert!(lay2.total <= lay2.body_h, "premise: this filter fits");
        let rows = render_help(&s, w, h);
        let c2 = lay2.x0 + lay2.pw - 1;
        assert!(
            (0..lay2.body_h).all(|i| rows[lay2.body_y + i].chars().nth(c2) != Some('|')),
            "no thumb when there is nothing to scroll"
        );
    }

    /// The `?` sheet as a 120x40 terminal paints it — top, bottom, and filtered.
    /// A generator for eyeballing the surface, not an assertion (the assertions
    /// are the seven tests above it).
    #[test]
    #[ignore = "visual aid, not an assertion"]
    fn dump_help_sheet() {
        let (w, h) = (120usize, 40usize);
        let mut s = EditorState::new_empty(8, 8, 2);
        s.open_help();
        s.help_page = help_layout(&s, w, h).expect("fits").body_h;
        let mut show = |title: &str, st: &EditorState| {
            println!("\n===== {title} =====");
            for r in render_help(st, w, h) {
                println!("{r}");
            }
        };
        show("TOP (just opened)", &s);
        s.help_scroll = usize::MAX;
        crate::edit::help_set_scroll(&mut s, usize::MAX);
        show("BOTTOM (End)", &s);
        s.help_scroll = 0;
        s.help_filter = "colour".into();
        show("FILTER: colour", &s);
        s.help_filter = "help".into();
        show("FILTER: help", &s);
    }

    #[test]
    fn help_popup_draws_border_and_title() {
        let state = EditorState::new_empty(8, 8, 2);
        let mut fb = Framebuffer::new(80, 30);
        draw_help(&state, &mut fb, 80, 30);
        // Some '#' border glyph and the 'H' of "HELP" must appear somewhere.
        let mut saw_border = false;
        let mut saw_h = false;
        for py in 0..30 {
            for px in 0..80 {
                match fb.get(px, py).map(|c| c.glyph) {
                    Some(g) if g == '#' as u32 => saw_border = true,
                    Some(g) if g == 'H' as u32 => saw_h = true,
                    _ => {}
                }
            }
        }
        assert!(saw_border, "help popup has a border");
        assert!(saw_h, "help popup shows its title");
    }

    #[test]
    fn open_overlay_lists_files_and_marks_cursor() {
        let mut state = EditorState::new_empty(8, 8, 2);
        state.open = Some(crate::state::OpenState {
            files: vec!["levels/a.lvl".into(), "levels/b.lvl".into()],
            cursor: 1,
        });
        let mut fb = Framebuffer::new(60, 20);
        draw_open(&state, &mut fb, 60, 20);
        // The '>' cursor marker for the selected (second) file appears.
        let mut saw_marker = false;
        for py in 0..20 {
            for px in 0..60 {
                if fb.get(px, py).map(|c| c.glyph) == Some('>' as u32) {
                    saw_marker = true;
                }
            }
        }
        assert!(saw_marker, "open overlay marks the highlighted file");
    }

    #[test]
    fn crosshair_tints_cursor_row_and_column_not_the_cell() {
        let mut state = EditorState::new_empty(8, 8, 2);
        state.cursor = (3, 2);
        let mut fb = Framebuffer::new(40, 20);
        draw_grid(&state, &mut fb, 0, 0, 40, 20);
        let zoom = state.zoom as usize;
        // A cell on the cursor's column (x=3) at a non-chunk, non-cursor row
        // (row 1) → crosshair tint.
        let px = RULER_W + 3 * zoom;
        let py_row1 = RULER_H + zoom; // world row 1
        assert_eq!(
            fb.get(px, py_row1).unwrap().bg,
            CROSSHAIR_BG,
            "cursor column is crosshair-tinted"
        );
        // The cursor cell itself keeps CURSOR_BG (crosshair must not overwrite it).
        let cpx = RULER_W + 3 * zoom;
        let cpy = RULER_H + 2 * zoom;
        assert_eq!(
            fb.get(cpx, cpy).unwrap().bg,
            CURSOR_BG,
            "cursor cell keeps CURSOR_BG"
        );
    }

    #[test]
    fn coord_readout_is_honest_integer_cell() {
        let mut state = EditorState::new_empty(8, 8, 4);
        state.cursor = (3, 4);
        state.z = 2;
        // Honest integer cell coordinate + z — no invented sub-cell fraction.
        let r = cursor_coord_readout(&state);
        assert_eq!(r, "3,4 z=2");
        // Zoom must NOT inject fake `/q` precision into the readout.
        state.zoom = 3;
        assert!(
            !cursor_coord_readout(&state).contains('/'),
            "no fake sub-cell ticks"
        );
    }

    #[test]
    fn panel_button_hit_lands_on_every_drawn_span() {
        // Cover several states so the full button set (inspector + tree + view +
        // fill) is exercised, with a panel wide/tall enough to lay them all out.
        let mut state = EditorState::new_empty(8, 8, 4);
        state.material_mode = true; // turns on the inspector context (+ NewMat)
        state.tree.open = true; // turns on the P6 tree action buttons
        let panel = crate::layout::Rect {
            x: 2,
            y: 0,
            w: 30,
            h: 40,
        };
        let spans = panel_buttons(&state, panel);
        assert!(!spans.is_empty(), "buttons must lay out in a roomy panel");
        for (b, s, e, r) in spans {
            // The hit-test on the FIRST glyph and the LAST glyph of the drawn span
            // must both resolve to this same button.
            assert_eq!(
                panel_button_hit(&state, panel, s, r),
                Some(b),
                "click on start of {b:?} span hits it"
            );
            assert_eq!(
                panel_button_hit(&state, panel, e - 1, r),
                Some(b),
                "click on end of {b:?} span hits it"
            );
            // And the drawn glyph width equals the bracketed label width.
            assert_eq!(e - s, b.label().len() + 2, "span width matches [label]");
        }
    }

    // ── R-1 PARITY HARNESS ───────────────────────────────────────────────────
    //
    // `chrome.rs` is the painter for everything, so the row model is refactored
    // under a parity guard. `parity_snapshot` walks one representative object of
    // EVERY `EntityKind` (plus the material and empty contexts) and joins each
    // context's `InspectorRow::text()` lines.
    //
    // NOTE: there is deliberately NO frozen `PARITY_SNAPSHOT` blob to diff
    // against — `parity_snapshot()` exists only to feed the `#[ignore]`d
    // `parity_snapshot_dump` generator, for eyeballing a refactor by hand.
    // Parity is enforced by INVARIANTS instead, which is stronger than a blob
    // because it keeps holding as fields are added:
    //   - `rowtype`'s three `*_matches_existing_display_for_every_field` tests
    //     prove each typed value reproduces the ORIGINAL formatter exactly, for
    //     every field of every kind;
    //   - `every_typed_row_renders_the_same_value_it_always_did` proves the
    //     widget affordance only decorates that value, never changes it.
    // If a frozen blob is ever wanted, `parity_snapshot_dump` prints one.

    /// Every `EntityKind`, in a fixed order, so the snapshot is stable.
    const PARITY_KINDS: [EntityKind; 16] = [
        EntityKind::Door,
        EntityKind::Spawn,
        EntityKind::Pedestal,
        EntityKind::Part,
        EntityKind::Node,
        EntityKind::Emitter,
        EntityKind::Trigger,
        EntityKind::Pipe,
        EntityKind::ZoneInput,
        EntityKind::Note,
        EntityKind::WallDecal,
        EntityKind::Holo,
        EntityKind::Portal,
        EntityKind::Zone,
        EntityKind::Line,
        EntityKind::Inset,
    ];

    /// A state with ONE object of `kind` placed at the cursor and selected, so
    /// the property sheet has a full field list to render.
    fn parity_state(kind: EntityKind) -> EditorState {
        let mut s = EditorState::new_empty(16, 16, 4);
        s.panels.adaptive = false;
        s.cursor = (2, 2);
        s.z = 1;
        s.place = Some(kind);
        crate::entity::place_at_cursor(&mut s);
        s
    }

    /// The joined `text()` of every inspector row, for every context.
    fn parity_snapshot() -> String {
        let mut out = String::new();
        for kind in PARITY_KINDS {
            let s = parity_state(kind);
            out.push_str(&format!(
                "== {kind:?}
"
            ));
            for r in inspector_rows(&s) {
                out.push_str(&format!(
                    "  {}
",
                    r.text()
                ));
            }
        }
        // The MATERIAL context (22 fields) and the empty/tips context.
        let mut m = EditorState::new_empty(16, 16, 4);
        m.panels.adaptive = false;
        crate::material::new_material(&mut m);
        m.material_mode = true;
        out.push_str(
            "== Material
",
        );
        for r in inspector_rows(&m) {
            out.push_str(&format!(
                "  {}
",
                r.text()
            ));
        }
        let empty = EditorState::new_empty(16, 16, 4);
        out.push_str(
            "== Empty
",
        );
        for r in inspector_rows(&empty) {
            out.push_str(&format!(
                "  {}
",
                r.text()
            ));
        }
        out
    }

    /// Print the live snapshot so it can be pasted into `PARITY_SNAPSHOT`.
    /// `cargo test -p ts_editor parity_snapshot_dump -- --nocapture --ignored`
    #[test]
    #[ignore = "generator for PARITY_SNAPSHOT, not an assertion"]
    fn parity_snapshot_dump() {
        println!(
            "<<<SNAP
{}SNAP>>>",
            parity_snapshot()
        );
    }

    /// Strip a typed row's WIDGET DECORATION back to the bare value text the
    /// pre-typed row model rendered. The inverse of `rowvalue::zones_in`' value
    /// half, used only by the parity test below.
    fn undecorate(v: &str) -> String {
        let v = v.trim_end();
        if let Some(inner) = v.strip_prefix("[-] ").and_then(|t| t.strip_suffix(" [+]")) {
            return inner.to_string();
        }
        if let Some(inner) = v.strip_suffix(" [< >]") {
            return inner.to_string();
        }
        if let Some(inner) = v.strip_suffix(" \u{2588}\u{2588}") {
            return inner.to_string();
        }
        match v {
            "[x]" => "true".to_string(),
            "[ ]" => "false".to_string(),
            _ => v.to_string(),
        }
    }

    /// R-1 PARITY, as an invariant rather than a literal snapshot.
    ///
    /// The typed row model was allowed to change exactly ONE thing about the
    /// property sheet: it may ADD a widget affordance to a row. It may never
    /// change the VALUE a row shows. This walks one representative object of
    /// every `EntityKind` plus the material context — 140 rows across 17
    /// contexts at the time of writing — and proves, for every single field
    /// row, that stripping the affordance recovers byte-for-byte the string
    /// the pre-refactor `label = value` rendering produced.
    ///
    /// Stated this way the test keeps working as fields are added, which a
    /// pasted snapshot blob would not.
    #[test]
    fn every_typed_row_renders_the_same_value_it_always_did() {
        let mut checked = 0usize;
        let mut decorated = 0usize;
        for kind in PARITY_KINDS {
            let s = parity_state(kind);
            for r in inspector_rows(&s) {
                if r.kind != RowKind::Field {
                    continue;
                }
                let text = r.text();
                let (label, value) = text
                    .split_once(" = ")
                    .unwrap_or_else(|| panic!("{kind:?} row is not `label = value`: {text}"));
                assert_eq!(label, r.label, "{kind:?}: label half is the row's label");
                assert_eq!(
                    undecorate(value),
                    r.value.display(),
                    "{kind:?} row '{label}': the affordance changed the VALUE, not just the chrome"
                );
                if undecorate(value) != value {
                    decorated += 1;
                }
                checked += 1;
                // The rendered line and the hit-test geometry are the same layout.
                assert_eq!(
                    text,
                    crate::rowvalue::row_text_in(&r.label, &r.value, usize::MAX),
                    "{kind:?} row '{label}': text() must BE the span layout"
                );
            }
        }
        // The material context too (22 fields).
        let mut m = EditorState::new_empty(16, 16, 4);
        m.panels.adaptive = false;
        crate::material::new_material(&mut m);
        m.material_mode = true;
        for r in inspector_rows(&m) {
            if r.kind != RowKind::Field {
                continue;
            }
            let text = r.text();
            let (label, value) = text
                .split_once(" = ")
                .expect("material row is `label = value`");
            assert_eq!(
                undecorate(value),
                r.value.display(),
                "material row '{label}'"
            );
            checked += 1;
        }
        assert!(
            checked > 150,
            "the sweep must actually cover the sheet, got {checked}"
        );
        assert!(
            decorated > 100,
            "typing the rows must have produced widgets, got {decorated}"
        );
    }

    /// A HEADER/INFO row still renders as its bare label — no `=`, no widget.
    #[test]
    fn banner_rows_are_unchanged_by_the_typed_model() {
        let s = parity_state(EntityKind::Node);
        let rows = inspector_rows(&s);
        assert_eq!(rows[0].kind, RowKind::Header);
        assert_eq!(
            rows[0].text(),
            rows[0].label,
            "header renders its bare label"
        );
        assert!(!rows[0].text().contains(" = "));
        let empty = EditorState::new_empty(16, 16, 4);
        for r in inspector_rows(&empty) {
            assert_eq!(r.text(), r.label);
            assert_eq!(r.action, RowAction::None, "tips rows are inert");
        }
    }

    /// REGRESSION — inspector affordances must not run off the DEFAULT panel.
    ///
    /// `PANEL_W` is 30. Naturally, `[Rates] max_particles = [-] 256 [+]` is 35
    /// cells plus the focus marker: end-clipping cut the number in half and
    /// dropped `[+]` off the row entirely, so the increment control could not be
    /// clicked at the width most users actually run. Rows now COMPACT (and, at
    /// the extreme, truncate the label) instead of clipping, so every widget
    /// zone stays on the row.
    #[test]
    fn every_widget_zone_fits_inside_the_default_panel_width() {
        use crate::rowvalue::{RowZone, WidgetPart};
        let width = crate::layout::PANEL_W - 1; // less the focus-marker column
        let mut checked = 0usize;
        for kind in PARITY_KINDS {
            let s = parity_state(kind);
            for r in inspector_rows(&s) {
                if r.kind != RowKind::Field {
                    continue;
                }
                let line = crate::rowvalue::row_text_in(&r.label, &r.value, width);
                let n = line.chars().count();
                assert!(
                    n <= width,
                    "{kind:?} row {:?} is {n} cells at PANEL_W: {line}",
                    r.label
                );
                // Every widget part the row's value defines is still present.
                let zones: Vec<RowZone> = (0..n)
                    .map(|i| crate::rowvalue::zone_at_in(&r.label, &r.value, width, i))
                    .collect();
                let want: &[WidgetPart] = match &r.value {
                    crate::rowvalue::RowValue::Int { .. }
                    | crate::rowvalue::RowValue::Float { .. } => {
                        &[WidgetPart::Dec, WidgetPart::Value, WidgetPart::Inc]
                    }
                    crate::rowvalue::RowValue::Bool(_) => &[WidgetPart::Toggle],
                    crate::rowvalue::RowValue::Enum { .. } => &[WidgetPart::Options],
                    crate::rowvalue::RowValue::Color(_) => &[WidgetPart::Swatch],
                    crate::rowvalue::RowValue::Text(_) => &[],
                };
                for part in want {
                    assert!(
                        zones.contains(&RowZone::Widget(*part)),
                        "{kind:?} row {:?}: {part:?} fell off the row at PANEL_W ({line})",
                        r.label
                    );
                }
                checked += 1;
            }
        }
        // 148 field rows across the 16 entity kinds (the material context has
        // its own 22 and is swept by the parity test).
        assert!(
            checked > 140,
            "the sweep must cover the sheet, got {checked}"
        );
    }

    /// The specific row the review named: at `PANEL_W` the emitter's
    /// `max_particles` row keeps its whole VALUE and both steppers.
    #[test]
    fn the_widest_emitter_row_keeps_its_value_and_both_steppers() {
        use crate::rowvalue::{RowZone, WidgetPart};
        let s = parity_state(EntityKind::Emitter);
        let width = crate::layout::PANEL_W - 1;
        let row = inspector_rows(&s)
            .into_iter()
            .find(|r| r.label.contains("max_particles"))
            .expect("the emitter sheet has a max_particles row");
        let line = crate::rowvalue::row_text_in(&row.label, &row.value, width);
        assert!(line.chars().count() <= width, "fits at PANEL_W: {line}");
        // The value is never truncated — a half-printed number is worse than a
        // shortened label.
        assert!(
            line.contains(&row.value.display()),
            "the value survives whole: {line} / {}",
            row.value.display()
        );
        let n = line.chars().count();
        for part in [WidgetPart::Dec, WidgetPart::Value, WidgetPart::Inc] {
            assert!(
                (0..n).any(
                    |i| crate::rowvalue::zone_at_in(&row.label, &row.value, width, i)
                        == RowZone::Widget(part)
                ),
                "{part:?} is reachable: {line}"
            );
        }
    }

    /// Show what the narrow-panel forms actually look like.
    #[test]
    #[ignore = "visual aid, not an assertion"]
    fn dump_narrow_rows() {
        let w = crate::layout::PANEL_W - 1;
        for kind in [EntityKind::Emitter, EntityKind::Node] {
            let s = parity_state(kind);
            for r in inspector_rows(&s)
                .iter()
                .filter(|r| r.kind == RowKind::Field)
                .take(8)
            {
                println!("|{}|", crate::rowvalue::row_text_in(&r.label, &r.value, w));
            }
        }
    }

    // ── ZONE-AWARE HIT-TESTING ───────────────────────────────────────────────

    /// A state + panel rect with a NODE selected, for the row hit-tests.
    fn hit_fixture() -> (EditorState, crate::layout::Rect) {
        let mut s = parity_state(EntityKind::Node);
        s.view_mode = crate::state::ViewMode::Full2D;
        let panel = crate::layout::Rect {
            x: 40,
            y: 0,
            w: 34,
            h: 40,
        };
        (s, panel)
    }

    /// The absolute `(col, row)` of one zone of the row for `field`, resolved
    /// through the SAME viewport + scroll math the painter uses.
    fn zone_col(
        s: &EditorState,
        panel: crate::layout::Rect,
        field: crate::state::EntityField,
        want: crate::rowvalue::RowZone,
    ) -> Option<(usize, usize)> {
        let rows = inspector_rows(s);
        let idx = rows
            .iter()
            .position(|r| r.action == RowAction::FocusEntity(field))?;
        let (visible, _) = inspector_viewport(s, panel);
        let focused_idx = rows.iter().position(|r| r.focused).unwrap_or(0);
        let scroll = settings_scroll(focused_idx, rows.len(), visible);
        if idx < scroll || idx >= scroll + visible {
            return None;
        }
        let ry = panel.y + INSPECTOR_ROWS_TOP + (idx - scroll);
        let text_x = inspector_text_x(panel);
        // Lay out at the SAME width the painter and hit-test use, or the probe
        // column would be computed from a different rendering than the one drawn.
        let width = inspector_text_w(panel);
        let n = crate::rowvalue::row_text_in(&rows[idx].label, &rows[idx].value, width)
            .chars()
            .count();
        (0..n)
            .find(|&i| {
                crate::rowvalue::zone_at_in(&rows[idx].label, &rows[idx].value, width, i) == want
            })
            .map(|i| (text_x + i, ry))
    }

    /// A click on a row's LABEL still focuses that field — unchanged by the
    /// typed model.
    #[test]
    fn clicking_a_row_label_still_focuses_the_field() {
        use crate::rowvalue::RowZone;
        use crate::state::EntityField as F;
        let (s, panel) = hit_fixture();
        let (col, row) = zone_col(&s, panel, F::NodeThreshold, RowZone::Label)
            .expect("threshold row is visible");
        let (action, zone) =
            inspector_row_hit(&s, panel, col, row).expect("label is a live hit target");
        assert_eq!(action, RowAction::FocusEntity(F::NodeThreshold));
        assert_eq!(
            zone,
            RowZone::Label,
            "the label zone focuses, it does not act"
        );
    }

    /// A click on a NUMERIC row's `[-]` / `[+]` / value resolves to the right
    /// widget part — three distinct actions inside one row.
    #[test]
    fn numeric_row_zones_resolve_separately() {
        use crate::rowvalue::{RowZone, WidgetPart as W};
        use crate::state::EntityField as F;
        let (s, panel) = hit_fixture();
        for part in [W::Dec, W::Value, W::Inc] {
            let (col, row) = zone_col(&s, panel, F::NodeThreshold, RowZone::Widget(part))
                .unwrap_or_else(|| panic!("{part:?} is drawn"));
            let (action, zone) = inspector_row_hit(&s, panel, col, row).expect("live target");
            assert_eq!(action, RowAction::FocusEntity(F::NodeThreshold));
            assert_eq!(zone, RowZone::Widget(part));
        }
    }

    /// A COLOUR row's swatch is its own zone, distinct from the label.
    #[test]
    fn colour_row_swatch_is_its_own_zone() {
        use crate::rowvalue::{RowZone, WidgetPart as W};
        use crate::state::EntityField as F;
        let (s, panel) = hit_fixture();
        let (col, row) =
            zone_col(&s, panel, F::NodeColor, RowZone::Widget(W::Swatch)).expect("swatch is drawn");
        let (action, zone) = inspector_row_hit(&s, panel, col, row).expect("live target");
        assert_eq!(action, RowAction::FocusEntity(F::NodeColor));
        assert_eq!(zone, RowZone::Widget(W::Swatch));
        // And the label half of the SAME row still focuses.
        let (lc, lr) = zone_col(&s, panel, F::NodeColor, RowZone::Label).expect("label");
        assert_eq!(
            inspector_row_hit(&s, panel, lc, lr).unwrap().1,
            RowZone::Label
        );
    }

    /// An ENUM row's `[< >]` is its own zone.
    #[test]
    fn enum_row_options_affordance_is_its_own_zone() {
        use crate::rowvalue::{RowZone, WidgetPart as W};
        use crate::state::EntityField as F;
        let (s, panel) = hit_fixture();
        let (col, row) =
            zone_col(&s, panel, F::NodeIo, RowZone::Widget(W::Options)).expect("[< >] is drawn");
        assert_eq!(
            inspector_row_hit(&s, panel, col, row).unwrap(),
            (
                RowAction::FocusEntity(F::NodeIo),
                RowZone::Widget(W::Options)
            )
        );
    }

    /// A BOOL row's checkbox is its own zone.
    #[test]
    fn bool_row_checkbox_is_its_own_zone() {
        use crate::rowvalue::{RowZone, WidgetPart as W};
        use crate::state::EntityField as F;
        let mut s = parity_state(EntityKind::Trigger);
        s.view_mode = crate::state::ViewMode::Full2D;
        let panel = crate::layout::Rect {
            x: 40,
            y: 0,
            w: 34,
            h: 40,
        };
        let (col, row) = zone_col(&s, panel, F::TrigOnce, RowZone::Widget(W::Toggle))
            .expect("checkbox is drawn");
        assert_eq!(
            inspector_row_hit(&s, panel, col, row).unwrap(),
            (
                RowAction::FocusEntity(F::TrigOnce),
                RowZone::Widget(W::Toggle)
            )
        );
    }

    /// The focus-marker column (the very first cell of the row) is LABEL, not a
    /// widget — the marker must not swallow a click into the wrong zone.
    #[test]
    fn the_focus_marker_column_is_never_a_widget() {
        use crate::rowvalue::RowZone;
        let (s, panel) = hit_fixture();
        let (visible, _) = inspector_viewport(&s, panel);
        for i in 0..visible {
            let row = panel.y + INSPECTOR_ROWS_TOP + i;
            if let Some((_, zone)) = inspector_row_hit(&s, panel, panel.x, row) {
                assert_eq!(zone, RowZone::Label, "marker column of row {i}");
            }
        }
    }

    /// `focused_row_action` names the same field the sheet marks as focused —
    /// the invariant `Enter` depends on.
    #[test]
    fn focused_row_action_tracks_the_focused_field() {
        use crate::state::EntityField as F;
        let mut s = parity_state(EntityKind::Node);
        for f in [F::NodeThreshold, F::NodeColor, F::NodeIo, F::NodePosX] {
            s.entity_field = f;
            assert_eq!(focused_row_action(&s), RowAction::FocusEntity(f));
        }
        // Nothing active ⇒ no action for Enter to run.
        let empty = EditorState::new_empty(16, 16, 4);
        assert_eq!(focused_row_action(&empty), RowAction::None);
    }

    // ── the row-editor OVERLAYS at the minimum terminal size ────────────────

    /// Every row-editor overlay paints inside the framebuffer at 80×24 and does
    /// not panic at sizes far below it. The editor must degrade, never crash.
    #[test]
    fn row_editor_overlays_paint_within_bounds_at_80x24() {
        use crate::state::EntityField as F;
        for field in [F::NodeThreshold, F::NodeIo, F::NodeColor] {
            let mut s = parity_state(EntityKind::Node);
            s.entity_field = field;
            assert!(
                crate::rowedit::open_for(&mut s, RowAction::FocusEntity(field)),
                "{field:?} opens an editor"
            );
            for (w, h) in [
                (80usize, 24usize),
                (60, 20),
                (40, 12),
                (20, 8),
                (12, 5),
                (4, 2),
            ] {
                let mut fb = Framebuffer::new(w, h);
                draw_num_edit(&s, &mut fb, w, h);
                draw_enum_pick(&s, &mut fb, w, h);
                draw_color_edit(&s, &mut fb, w, h);
            }
            // At 80×24 the overlay actually drew something.
            let mut fb = Framebuffer::new(80, 24);
            draw_num_edit(&s, &mut fb, 80, 24);
            draw_enum_pick(&s, &mut fb, 80, 24);
            draw_color_edit(&s, &mut fb, 80, 24);
            let painted = (0..24)
                .flat_map(|y| (0..80).map(move |x| (x, y)))
                .any(|(x, y)| fb.get(x, y).is_some_and(|c| c.glyph == '#' as u32));
            assert!(painted, "{field:?} overlay drew its box at 80x24");
        }
    }

    /// Helper: a state with a placed + selected PORTAL (a 14-field object,
    /// the largest inspector sheet).
    fn portal_state() -> EditorState {
        let mut s = EditorState::new_empty(16, 16, 4);
        // These parity tests build their rects from the LEGACY `compute`, so
        // the state must not adaptively shed panels at narrow sizes (80×24) —
        // the adaptive panel behaviour has its own tests in `layout`/`main`.
        s.panels.adaptive = false;
        s.view_mode = crate::state::ViewMode::Full2D;
        s.level.portals.push(re_level::LevelPortal {
            name: "p0".into(),
            ..Default::default()
        });
        s.place = Some(EntityKind::Portal);
        s.selected_portal = Some(0);
        s.entity_field = crate::state::EntityField::PortalAx0;
        s
    }

    /// The row model lists the portal summary header + ALL 14 fields, each
    /// carrying the focus action + live value for its own field.
    #[test]
    fn inspector_rows_portal_lists_header_plus_all_14_fields() {
        let s = portal_state();
        let rows = inspector_rows(&s);
        assert_eq!(rows[0].kind, RowKind::Header, "summary header first");
        let fields: Vec<&InspectorRow> = rows.iter().filter(|r| r.kind == RowKind::Field).collect();
        let expected = crate::state::entity_fields_for(EntityKind::Portal);
        assert_eq!(fields.len(), expected.len(), "one row per portal field");
        assert_eq!(fields.len(), 14, "portal has 14 fields");
        for (row, &f) in fields.iter().zip(expected) {
            assert_eq!(
                row.action,
                RowAction::FocusEntity(f),
                "row action focuses its field"
            );
            // The row value is TYPED now, so the parity assertion compares what
            // it RENDERS: a typed row must still display exactly what the
            // pre-existing value function produced for its own field.
            assert_eq!(
                row.value.display(),
                crate::entity::entity_field_value_of(&s, f),
                "row value renders ITS field, not the focused one"
            );
            assert_eq!(row.focused, f == s.entity_field);
        }
    }

    /// Acceptance: at 80×24 the portal sheet SCROLLS — every one of the 14
    /// fields is reachable (the focused row is always inside the viewport window)
    /// and the sheet keeps its minimum row count.
    #[test]
    fn portal_every_field_reachable_by_scroll_at_80x24() {
        let mut s = portal_state();
        let l = crate::layout::compute(s.view_mode, 80, 24, s.tree.open);
        let inspector = inspector_subrect(&s, l.panel);
        assert!(!inspector.is_empty(), "inspector organ present at 80x24");
        let (visible, _) = inspector_viewport(&s, inspector);
        let total = inspector_rows(&s).len();
        assert!(
            visible >= MIN_SHEET_ROWS.min(total),
            "sheet keeps its minimum rows"
        );
        assert!(
            visible < total,
            "the portal sheet must actually scroll at 80x24"
        );
        for &f in crate::state::entity_fields_for(EntityKind::Portal) {
            s.entity_field = f;
            let rows = inspector_rows(&s);
            let idx = rows
                .iter()
                .position(|r| r.focused)
                .expect("focused row exists");
            let scroll = settings_scroll(idx, rows.len(), visible);
            assert!(
                idx >= scroll && idx < scroll + visible,
                "focused {f:?} row stays inside the viewport"
            );
        }
    }

    /// The ▼ (rows below) and ▲ (rows above) overflow indicators are
    /// painted at the viewport's right edge at 80×24.
    #[test]
    fn portal_sheet_draws_scroll_indicators_at_80x24() {
        let s = portal_state(); // first field focused → rows hidden BELOW → ▼
        let l = crate::layout::compute(s.view_mode, 80, 24, s.tree.open);
        let inspector = inspector_subrect(&s, l.panel);
        let (visible, _) = inspector_viewport(&s, inspector);
        assert!(visible > 0);
        let top = inspector.y + INSPECTOR_ROWS_TOP;
        let arrow_x = inspector.x + inspector.w - 1;
        let mut fb = Framebuffer::new(80, 24);
        draw_panel(&s, &mut fb, inspector.x, inspector.y, inspector.h);
        assert_eq!(
            fb.get(arrow_x, top + visible - 1).unwrap().glyph,
            '\u{25BC}' as u32,
            "▼ marks rows scrolled off below"
        );
        // Focus the LAST field → rows hidden ABOVE → ▲ on the top viewport row.
        let mut s2 = portal_state();
        s2.entity_field = crate::state::EntityField::PortalSkin;
        let mut fb2 = Framebuffer::new(80, 24);
        draw_panel(&s2, &mut fb2, inspector.x, inspector.y, inspector.h);
        assert_eq!(
            fb2.get(arrow_x, top).unwrap().glyph,
            '\u{25B2}' as u32,
            "▲ marks rows scrolled off above"
        );
    }

    /// The click hit-test resolves through the SAME rows + scroll the
    /// painter draws — every visible row maps back to its own action.
    #[test]
    fn inspector_row_hit_maps_click_to_drawn_row() {
        let s = portal_state();
        let l = crate::layout::compute(s.view_mode, 120, 40, s.tree.open);
        let inspector = inspector_subrect(&s, l.panel);
        let (visible, _) = inspector_viewport(&s, inspector);
        assert!(visible > 1, "several sheet rows visible at 120x40");
        let rows = inspector_rows(&s);
        let focused_idx = rows.iter().position(|r| r.focused).unwrap_or(0);
        let scroll = settings_scroll(focused_idx, rows.len(), visible);
        let top = inspector.y + INSPECTOR_ROWS_TOP;
        for vi in 0..visible {
            let got = inspector_row_hit(&s, inspector, inspector.x + 1, top + vi);
            match rows[scroll + vi].action {
                RowAction::None => assert_eq!(got, None, "banner rows are inert"),
                // The probe column is the row's FIRST text cell — the label —
                // so every hit resolves to that row's action in the LABEL zone
                // (click-to-focus, the behaviour typed rows preserve).
                a => assert_eq!(
                    got,
                    Some((a, RowZone::Label)),
                    "row {vi} resolves to its own action"
                ),
            }
        }
        // One row below the viewport → not a sheet hit.
        assert_eq!(
            inspector_row_hit(&s, inspector, inspector.x, top + visible),
            None
        );
        // Outside the panel columns → None.
        assert_eq!(
            inspector_row_hit(&s, inspector, inspector.x + inspector.w, top),
            None
        );
    }

    /// Compatibility contract: `button_span` resolves exactly the spans
    /// `panel_buttons` lays out (the accessor the mouse tests + elevation anchor
    /// rely on), and `None` off-panel.
    #[test]
    fn button_span_accessor_matches_panel_buttons() {
        let mut state = EditorState::new_empty(8, 8, 4);
        state.material_mode = true;
        state.tree.open = true;
        let panel = crate::layout::Rect {
            x: 2,
            y: 0,
            w: 30,
            h: 40,
        };
        let spans = panel_buttons(&state, panel);
        assert!(!spans.is_empty());
        for (b, s, e, r) in spans {
            assert_eq!(button_span(&state, panel, b), Some((s, e, r)));
        }
        let empty = crate::layout::Rect {
            x: 0,
            y: 0,
            w: 0,
            h: 0,
        };
        assert_eq!(button_span(&state, empty, PanelButton::ZoomIn), None);
    }

    /// With a full portal sheet pushing the (dynamic) button
    /// block down, the elevation strip still lands BELOW the last button row and
    /// inside the panel.
    #[test]
    fn elevation_rect_stays_below_dynamic_buttons() {
        let s = portal_state();
        let panel = crate::layout::Rect {
            x: 0,
            y: 0,
            w: 30,
            h: 60,
        };
        let spans = panel_buttons(&s, panel);
        assert!(!spans.is_empty(), "buttons lay out in a 60-row panel");
        let last = spans.iter().map(|&(_, _, _, r)| r).max().unwrap();
        let elev = inspector_elevation_rect(&s, panel).expect("strip fits");
        assert!(elev.y > last, "elevation sits below the last button row");
        assert!(
            elev.y + elev.h <= panel.y + panel.h,
            "strip stays inside the panel"
        );
    }

    #[test]
    fn subcell_glyph_appears_only_at_higher_zoom() {
        assert_eq!(subcell_glyph(1, 1, 1), None);
        assert_eq!(subcell_glyph(1, 1, 2), Some('\u{00B7}'));
        // zoom 3 centre is a cross.
        assert_eq!(subcell_glyph(1, 1, 3), Some('+'));
    }

    #[test]
    fn top_layer_window_centres_active() {
        // Whole stack fits: top is the top layer.
        assert_eq!(top_layer_for_window(0, 4, 8), 3);
        // Tall stack: window centred on z.
        let sz = 20;
        let h = 6;
        let top = top_layer_for_window(10, sz, h);
        assert!(top >= 10, "active layer is within the window");
        assert!(top < sz);
    }

    // ── LEVEL SETTINGS scroll / keep-focus-visible ──────────────────────────────

    /// A `SettingsState` with a fixed `focus`, exercising the full (tall) row list.
    fn settings_fixture(focus: crate::settings::SettingsField) -> crate::settings::SettingsState {
        crate::settings::SettingsState {
            name: "vault".into(),
            title: "The Vault".into(),
            version: String::new(),
            spec_rev: String::new(),
            power_budget: String::new(),
            room_floor: String::new(),
            room_ceiling: String::new(),
            door_target_model: String::new(),
            ambient_idx: 0,
            storey_height: 4,
            size_w: 8,
            size_h: 8,
            size_d: 4,
            cycle_keep: false,
            render_pipes: false,
            snow: 0.0,
            steam: false,
            matrix: 0.0,
            glitch: 0.0,
            fog_color: String::new(),
            fog_near: 0.0,
            fog_far: 0.0,
            has_fog_dist: false,
            atmos_ambient: None,
            key_color: String::new(),
            fog_height_base: 0.0,
            fog_height_falloff: 0.0,
            has_fog_height: false,
            postfx_idx: 0,
            sky_cylinder: String::new(),
            sky_cap_top: String::new(),
            sky_cap_bottom: String::new(),
            textures: Vec::new(),
            texture_cursor: 0,
            texture_col: 0,
            inputs: vec![("in0".into(), false), ("in1".into(), true)],
            observe: vec!["outA".into(), "outB".into()],
            steps: vec![vec!["outA".into()], vec!["outB".into()]],
            input_cursor: 0,
            observe_cursor: 0,
            step_cursor: 0,
            name_cursor: 0,
            node_names: vec!["in0".into(), "in1".into(), "outA".into(), "outB".into()],
            output_names: vec!["outA".into(), "outB".into()],
            focus,
        }
    }

    #[test]
    fn settings_scroll_keeps_focus_visible_and_clamps() {
        // No scroll when everything fits.
        assert_eq!(settings_scroll(0, 5, 10), 0);
        assert_eq!(settings_scroll(4, 5, 5), 0);
        // visible == 0 → degenerate, never panics.
        assert_eq!(settings_scroll(3, 10, 0), 0);
        // Focus inside the first window → no scroll.
        assert_eq!(settings_scroll(2, 30, 8), 0);
        // Focus below the window → scroll so focus is the LAST visible row.
        assert_eq!(settings_scroll(8, 30, 8), 1);
        // Last row focused → window is pinned at the end (no blank tail).
        assert_eq!(settings_scroll(29, 30, 8), 22);
        assert_eq!(settings_scroll(29, 30, 8) + 8, 30);
    }

    #[test]
    fn settings_scroll_invariant_holds_for_every_focus_and_height() {
        // The full focus order, taken from the SSOT table rather than re-listed
        // here — a field added to `SettingsField::ORDER` is covered automatically.
        // The last entries (Door, Steps) are the danger zone: they sit at the
        // bottom of a ~45-row list and clip on a short terminal.
        let focuses = crate::settings::SettingsField::ORDER;
        // Modal heights from the smallest that passes the `ph < 12` guard up past the
        // full list — the short ones MUST clip and therefore must scroll.
        for total_h in 12..=40usize {
            // content window = box height minus the title (2 top rows) and footer (2).
            let visible = total_h.saturating_sub(4);
            for &focus in focuses {
                let s = settings_fixture(focus);
                let rows = settings_rows(&s);
                let total = rows.len();
                // SSOT: the focused index is the first flagged row (as the painter derives it).
                let focused_idx = rows.iter().position(|(f, _)| *f).unwrap_or(0);
                let scroll = settings_scroll(focused_idx, total, visible);
                if visible == 0 {
                    continue;
                }
                let lo = scroll;
                let hi = scroll + visible; // exclusive
                assert!(
                    focused_idx >= lo && focused_idx < hi,
                    "focus {focus:?} (row {focused_idx}/{total}) out of window [{lo},{hi}) at height {total_h} (visible {visible})"
                );
                // The window never runs past the list end while content remains.
                assert!(scroll <= total.saturating_sub(visible));
            }
        }
    }

    #[test]
    fn settings_last_field_is_visible_on_short_modal() {
        // The single most-clipped case: focus the LAST field on a short modal that
        // can host only a handful of content rows. The focused row must be visible.
        let s = settings_fixture(crate::settings::SettingsField::SkyCapBottom);
        let rows = settings_rows(&s);
        let focused_idx = rows.iter().position(|(f, _)| *f).unwrap();
        let visible = 6; // a deliberately short window
        let scroll = settings_scroll(focused_idx, rows.len(), visible);
        assert!(
            focused_idx >= scroll && focused_idx < scroll + visible,
            "SkyCapBottom row {focused_idx} hidden: scroll {scroll}, window {visible}"
        );
    }

    #[test]
    fn settings_rows_flags_exactly_one_focus_for_scalar_fields() {
        // Each scalar field maps to exactly one flagged row, so the focus→index
        // derivation is unambiguous. Driven off the SSOT order so a newly added
        // field that the painter forgot to draw fails HERE (0 flagged rows).
        use crate::settings::SettingsField as F;
        // The four SECTION foci flag a row only when their list is non-empty, and
        // the fixture leaves `textures` empty — they are covered by the scroll
        // invariant test instead.
        let sections = [F::Textures, F::Inputs, F::Observe, F::Steps];
        for &focus in F::ORDER.iter().filter(|f| !sections.contains(f)) {
            let s = settings_fixture(focus);
            let rows = settings_rows(&s);
            let flagged = rows.iter().filter(|(f, _)| *f).count();
            assert_eq!(flagged, 1, "exactly one flagged row for {focus:?}");
        }
    }

    /// The five newly-authorable `.lvl` keys each have a VISIBLE row
    /// in the settings form, and `[room] floor`/`ceiling` say on-screen that they
    /// are inert style tags (they are NOT material names — the surface's look
    /// comes from a `[material] surface = floor|ceiling` binding).
    #[test]
    fn settings_rows_show_the_new_metadata_fields() {
        use crate::settings::SettingsField as F;
        let mut s = settings_fixture(F::Name);
        s.version = "1".into();
        s.spec_rev = "2026-06-15".into();
        s.power_budget = "12".into();
        s.room_floor = "grid".into();
        s.room_ceiling = "dim".into();
        s.door_target_model = "atrium".into();
        let body = settings_rows(&s)
            .iter()
            .map(|(_, t)| t.clone())
            .collect::<Vec<_>>()
            .join("\n");
        for want in [
            "version  : 1",
            "spec_rev : 2026-06-15",
            "power_budget : 12",
            "floor        : grid",
            "ceiling      : dim",
            "-- Door --",
            "target_model : atrium",
        ] {
            assert!(
                body.contains(want),
                "settings form must show '{want}':\n{body}"
            );
        }
        assert!(
            body.contains("inert style tags"),
            "floor/ceiling are labelled inert so nobody types a material name there:\n{body}"
        );
        // An empty buffer reads `(none)`, matching what the writer emits (nothing).
        let empty = settings_rows(&settings_fixture(F::Name))
            .iter()
            .map(|(_, t)| t.clone())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            empty.contains("power_budget : (none)"),
            "empty budget reads (none)"
        );
        assert!(
            empty.contains("version  : (none)"),
            "empty version reads (none)"
        );
    }

    /// The five `[atmosphere]` rows are VISIBLE, show `(default)`/`(none)`/`(off)`
    /// while unset, and show the staged value once authored. The atmosphere
    /// `ambient` row is labelled `(0-1)` so it cannot be confused with the
    /// `[room] ambient` wall-KIND row above it.
    #[test]
    fn settings_rows_show_the_atmosphere_fields() {
        use crate::settings::SettingsField as F;
        let unset = settings_rows(&settings_fixture(F::Name))
            .iter()
            .map(|(_, t)| t.clone())
            .collect::<Vec<_>>()
            .join("\n");
        for want in [
            "-- Atmosphere (lighting) --",
            "ambient (0-1) : (default)",
            "key_color     : (none)",
            "fog_h base    : 0.0  (off)",
            "fog_h falloff : 0.0  (off)",
            "postfx        : (default)",
        ] {
            assert!(
                unset.contains(want),
                "settings form must show '{want}':\n{unset}"
            );
        }
        // The `[room] ambient` row is still there and still distinct.
        assert!(
            unset.contains("ambient : (default)"),
            "the [room] ambient row survives"
        );

        let mut s = settings_fixture(F::Name);
        s.atmos_ambient = Some(0.25);
        s.key_color = "#0AF0C1".into();
        s.fog_height_base = 2.5;
        s.fog_height_falloff = 0.3;
        s.has_fog_height = true;
        s.postfx_idx = 1;
        let set = settings_rows(&s)
            .iter()
            .map(|(_, t)| t.clone())
            .collect::<Vec<_>>()
            .join("\n");
        for want in [
            "ambient (0-1) : 0.25",
            "key_color     : #0AF0C1",
            "fog_h base    : 2.5",
            "fog_h falloff : 0.3",
        ] {
            assert!(
                set.contains(want),
                "settings form must show '{want}':\n{set}"
            );
        }
        assert!(
            set.contains(&format!("postfx        : {}", re_world::POSTFX_PRESETS[0])),
            "postfx shows the selected preset:\n{set}"
        );
        // An authored `fog_height` drops the `(off)` note from ITS two rows (the
        // still-unset `[palette]` fog rows above keep theirs).
        for line in set.lines().filter(|l| l.starts_with("fog_h ")) {
            assert!(
                !line.contains("(off)"),
                "an authored fog_height drops the (off) note: {line}"
            );
        }
    }

    // ─── CRASH-RECOVERY OFFER OVERLAY ────────────────────────────────────────

    /// Read the whole framebuffer back as text, one row per line — the same
    /// glyph read-back `outliner_text` does, generalised so the offer painter can
    /// be asserted against what actually lands in cells.
    fn fb_text(fb: &Framebuffer, w: usize, h: usize) -> String {
        let mut out = String::new();
        for r in 0..h {
            for c in 0..w {
                if let Some(cell) = fb.get(c, r) {
                    out.push(char::from_u32(cell.glyph).unwrap_or(' '));
                }
            }
            out.push('\n');
        }
        out
    }

    fn offer_text(sidecar: &str, w: usize, h: usize) -> String {
        let mut s = EditorState::new_empty(16, 16, 4);
        s.autosave_offer = Some(sidecar.to_string());
        let mut fb = Framebuffer::new(w, h);
        draw_autosave_offer(&s, &mut fb, w, h);
        fb_text(&fb, w, h)
    }

    /// A modal that owns the keyboard but paints nothing looks like a frozen
    /// editor, so the offer must state what happened, name the sidecar, offer
    /// both choices as visible controls, and say that discarding keeps the file.
    #[test]
    fn the_crash_recovery_offer_actually_paints_its_prompt() {
        let (w, h) = (100usize, 30usize);
        let text = offer_text("levels/09-code-wall.lvl.autosave", w, h);
        for want in [
            "CRASH RECOVERY",
            "A NEWER autosave was found after this file opened:",
            "levels/09-code-wall.lvl.autosave",
            AUTOSAVE_RESTORE_LABEL,
            AUTOSAVE_DISCARD_LABEL,
            "Y restore",
            "N discard",
            "Discard keeps the sidecar file on disk.",
        ] {
            assert!(
                text.contains(want),
                "offer overlay must show {want:?}:\n{text}"
            );
        }
    }

    /// No offer → no paint at all. The painter is gated on the state, not on the
    /// caller remembering to guard it.
    #[test]
    fn no_offer_paints_nothing() {
        let (w, h) = (100usize, 30usize);
        let s = EditorState::new_empty(16, 16, 4);
        let mut fb = Framebuffer::new(w, h);
        draw_autosave_offer(&s, &mut fb, w, h);
        let text = fb_text(&fb, w, h);
        assert!(
            text.chars().all(|c| c == ' ' || c == '\n' || c == '\0'),
            "an unset offer paints nothing:\n{text}"
        );
    }

    /// SSOT: every clickable label is painted at EXACTLY the columns
    /// `autosave_hit` resolves, and the two spans do not overlap. This is the
    /// property that makes "one geometry fn" mean something — a click on the
    /// middle of each drawn label must return that label's hit.
    #[test]
    fn the_painted_buttons_are_where_the_hit_test_says_they_are() {
        let (w, h) = (100usize, 30usize);
        let lay = autosave_layout(w, h).expect("fits at 100x30");
        let mut s = EditorState::new_empty(16, 16, 4);
        s.autosave_offer = Some("levels/x.lvl.autosave".into());
        let mut fb = Framebuffer::new(w, h);
        draw_autosave_offer(&s, &mut fb, w, h);

        let read = |x0: usize, x1: usize, y: usize| -> String {
            (x0..x1)
                .map(|c| {
                    fb.get(c, y)
                        .map_or(' ', |cell| char::from_u32(cell.glyph).unwrap_or(' '))
                })
                .collect()
        };
        assert_eq!(
            read(lay.restore.0, lay.restore.1, lay.button_row),
            AUTOSAVE_RESTORE_LABEL,
            "the restore span holds exactly the restore label"
        );
        assert_eq!(
            read(lay.discard.0, lay.discard.1, lay.button_row),
            AUTOSAVE_DISCARD_LABEL,
            "the discard span holds exactly the discard label"
        );
        assert!(lay.restore.1 <= lay.discard.0, "spans are disjoint");

        // The middle of each drawn label hit-tests to that label.
        let mid = |(start, end): (usize, usize)| usize::midpoint(start, end);
        assert_eq!(
            autosave_hit(w, h, mid(lay.restore), lay.button_row),
            Some(AutosaveHit::Restore)
        );
        assert_eq!(
            autosave_hit(w, h, mid(lay.discard), lay.button_row),
            Some(AutosaveHit::Discard)
        );
        // Off the button row, and in the gap between them, nothing is hit.
        assert_eq!(
            autosave_hit(w, h, mid(lay.restore), lay.button_row + 1),
            None
        );
        assert_eq!(
            autosave_hit(w, h, lay.restore.1, lay.button_row),
            None,
            "the gap is not a button"
        );
    }

    /// A long sidecar path is clamped to the box's inner width — never past the
    /// right border — and the clamp keeps the TAIL, so the file name survives.
    #[test]
    fn a_long_sidecar_path_is_tail_clamped_inside_the_border() {
        let (w, h) = (100usize, 30usize);
        let lay = autosave_layout(w, h).expect("fits");
        let long = format!("levels/{}/deep-nested-name.lvl.autosave", "sub".repeat(60));
        let text = offer_text(&long, w, h);
        assert!(
            text.contains("deep-nested-name.lvl.autosave"),
            "the tail (the file name) survives the clamp:\n{text}"
        );
        // Nothing painted on the path row may reach the border column.
        let mut fb = Framebuffer::new(w, h);
        let mut s = EditorState::new_empty(16, 16, 4);
        s.autosave_offer = Some(long);
        draw_autosave_offer(&s, &mut fb, w, h);
        let border_col = lay.x0 + lay.pw - 1;
        let g = fb.get(border_col, lay.path_row).map(|c| c.glyph);
        assert_eq!(g, Some('#' as u32), "the right border survives a long path");
    }

    /// Degradation: too small a terminal declines to paint rather than clipping
    /// garbage — the same contract every other modal painter keeps.
    #[test]
    fn a_tiny_terminal_declines_to_paint_the_offer() {
        assert_eq!(autosave_box_geom(20, 30), None, "too narrow");
        assert_eq!(autosave_box_geom(100, 6), None, "too short");
        assert!(autosave_layout(20, 30).is_none(), "no layout without a box");
        assert_eq!(autosave_hit(20, 30, 5, 5), None, "and nothing is clickable");
        let (w, h) = (20usize, 30usize);
        let text = offer_text("levels/x.lvl.autosave", w, h);
        assert!(
            text.chars().all(|c| c == ' ' || c == '\n' || c == '\0'),
            "no partial box is painted:\n{text}"
        );
    }

    /// Sound-field authoring VISUAL EVIDENCE (task requirement: "editor frames
    /// with a field selected and the radius aid on, to PNG"). Renders the REAL
    /// production painters — `draw_grid` (2D plan, radius/cone outline
    /// color-by-archetype-family) and `preview::render_preview` (3D marker,
    /// via `crate::preview::overlay_field_markers`) — into `Framebuffer`s with
    /// `fields_aid` ON and a field selected, then encodes each cell as a block
    /// in a real PNG so the aid can be inspected outside the terminal. Also
    /// proves the aid produces visibly different pixels than the aid-off state
    /// (the actual assertion; the PNG write is evidence, not the check).
    #[test]
    fn field_evidence_frames_render_to_png() {
        let mut state = EditorState::new_empty(6, 6, 4);
        state.map = re_world::test_room();
        state.cam = re_world::CameraPose::spawn_in_test_room();
        state.z = 1;
        state.cursor = (3, 3);
        state.level.field_defs.push(re_level::fields::FieldDef {
            name: "river".into(),
            line: 0,
            host: re_level::fields::FieldHost::Standalone,
            keys: vec![
                re_level::fields::FieldKey {
                    key: "kind".into(),
                    value: "lethe_wash".into(),
                    line: 0,
                },
                re_level::fields::FieldKey {
                    key: "at".into(),
                    value: "3, 3, 1".into(),
                    line: 0,
                },
                re_level::fields::FieldKey {
                    key: "radius".into(),
                    value: "2.5".into(),
                    line: 0,
                },
            ],
        });
        state.level.field_defs.push(re_level::fields::FieldDef {
            name: "signal".into(),
            line: 0,
            host: re_level::fields::FieldHost::Standalone,
            keys: vec![
                re_level::fields::FieldKey {
                    key: "kind".into(),
                    value: "recognition_narrow".into(),
                    line: 0,
                },
                re_level::fields::FieldKey {
                    key: "at".into(),
                    value: "4, 4, 1".into(),
                    line: 0,
                },
                re_level::fields::FieldKey {
                    key: "radius".into(),
                    value: "1.5".into(),
                    line: 0,
                },
            ],
        });
        crate::field::re_resolve(&mut state);
        assert_eq!(
            state.level.resolved_fields.len(),
            2,
            "both fields resolve cleanly"
        );
        state.selected_field = Some(0);

        let (gw, gh) = (60usize, 24usize);
        let (pw, ph) = (60usize, 24usize);
        let mut grid_off = Framebuffer::new(gw, gh);
        draw_grid(&state, &mut grid_off, 0, 0, gw, gh);
        let mut preview_off = Framebuffer::new(pw, ph);
        crate::preview::render_preview(&state, &mut preview_off, 0, 0, pw, ph);

        state.fields_aid = true;
        let mut grid_on = Framebuffer::new(gw, gh);
        draw_grid(&state, &mut grid_on, 0, 0, gw, gh);
        let mut preview_on = Framebuffer::new(pw, ph);
        crate::preview::render_preview(&state, &mut preview_on, 0, 0, pw, ph);

        let differs = |a: &Framebuffer, b: &Framebuffer, w: usize, h: usize| {
            (0..h).any(|y| (0..w).any(|x| a.get(x, y) != b.get(x, y)))
        };
        assert!(
            differs(&grid_off, &grid_on, gw, gh),
            "the fields aid changes the 2D plan view"
        );
        assert!(
            differs(&preview_off, &preview_on, pw, ph),
            "the fields aid changes the 3D preview"
        );

        if let Ok(dir) = std::env::var("TS_FIELD_EVIDENCE_DIR") {
            std::fs::create_dir_all(&dir).ok();
            dump_framebuffer_png(&grid_on, gw, gh, &format!("{dir}/field_grid_aid_on.png"));
            dump_framebuffer_png(
                &preview_on,
                pw,
                ph,
                &format!("{dir}/field_preview_aid_on.png"),
            );
        }
    }

    /// Encode a `Framebuffer` as a real PNG: each cell becomes a `cell_px`
    /// pixel block, the block's background filled with the cell's `bg`, and a
    /// smaller centered square filled with its `fg` — a legible, dependency-
    /// light stand-in for glyph rendering (no font rasterizer in this crate).
    fn dump_framebuffer_png(fb: &Framebuffer, w: usize, h: usize, path: &str) {
        const CELL_PX: u32 = 14;
        let (iw, ih) = (w as u32 * CELL_PX, h as u32 * CELL_PX);
        let mut img = image::RgbImage::new(iw, ih);
        let to_rgb =
            |packed: u32| image::Rgb([(packed >> 16) as u8, (packed >> 8) as u8, packed as u8]);
        for y in 0..h {
            for x in 0..w {
                let Some(cell) = fb.get(x, y) else { continue };
                let (bg, fg) = (to_rgb(cell.bg), to_rgb(cell.fg));
                let (px0, py0) = (x as u32 * CELL_PX, y as u32 * CELL_PX);
                for dy in 0..CELL_PX {
                    for dx in 0..CELL_PX {
                        let inner =
                            (3..CELL_PX - 3).contains(&dx) && (3..CELL_PX - 3).contains(&dy);
                        let color = if inner && cell.glyph != ' ' as u32 {
                            fg
                        } else {
                            bg
                        };
                        img.put_pixel(px0 + dx, py0 + dy, color);
                    }
                }
            }
        }
        img.save(path).expect("write evidence PNG");
    }
}
