//! View-mode rect math: derive pane rectangles from the framebuffer size so the
//! layout adapts to any terminal (never hardcoded 120x40). Pure + tested.
//!
//! No key or mouse handling lives here (`edit.rs`/`mouse.rs` own that) and no
//! `re_level`/`re_render` types appear in this module's signatures — it only
//! knows terminal cell counts in and pane [`Rect`]s out, which is what lets
//! `chrome` (painting) and `mouse` (hit-testing) share one geometry source
//! without either depending on the other.
use crate::state::ViewMode;

/// Minimum terminal width below which the editor body cannot host a usable
/// pane (grid/preview would collapse to slivers). `chrome`/`main` reference this
/// to decide whether to draw a "terminal too small" message instead of panes.
pub const MIN_COLS: usize = 40;
/// Minimum terminal height below which the editor body is unusable. Paired with
/// [`MIN_COLS`]; `main` draws the too-small message when either floor is missed.
pub const MIN_ROWS: usize = 12;

/// `true` when the terminal is below the usable floor in either dimension. When
/// this holds, `compute` returns empty body rects (grid/panel/preview/tree) and
/// callers should draw a too-small message rather than try to render panes.
pub fn too_small(w: usize, h: usize) -> bool {
    w < MIN_COLS || h < MIN_ROWS
}

/// A rectangle in framebuffer cells.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Rect {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
}

impl Rect {
    /// `true` when the rect has no drawable area.
    pub fn is_empty(&self) -> bool {
        self.w == 0 || self.h == 0
    }
}

/// Minimum rows the OUTLINER organ needs to be worth drawing as its own region
/// (header + a few object rows). Below the SPLIT threshold the panel shows ONE
/// organ at a time instead.
const OUTLINER_MIN_H: usize = 8;
/// Minimum rows the INSPECTOR organ needs (header + brush + z-slider + a couple
/// of property rows). Paired with [`OUTLINER_MIN_H`]; the panel splits into two
/// organs only when it can give BOTH their minimum.
const INSPECTOR_MIN_H: usize = 11;
/// The 1-row labeled divider painted between the two organs.
const ORGAN_DIVIDER_H: usize = 1;
/// Below this panel height we cannot host OUTLINER + divider + INSPECTOR; the
/// panel then shows a SINGLE organ (caller picks which by context).
const PANEL_SPLIT_MIN_H: usize = OUTLINER_MIN_H + ORGAN_DIVIDER_H + INSPECTOR_MIN_H;

/// Split the side `panel` rect VERTICALLY into the two stacked editor organs:
/// `(outliner, inspector)`. The OUTLINER (the always-visible object list) takes
/// the TOP ~55% of the panel (at least [`OUTLINER_MIN_H`] rows); the INSPECTOR
/// (context-sensitive properties of the current selection/tool) takes the
/// remainder BELOW it. Between them sits a 1-row labeled divider, which is NOT
/// part of either returned rect (the painter draws `─ INSPECTOR ─` on that row).
///
/// THE single source of truth shared by the painters (`chrome::draw_*`) and the
/// mouse hit-tests (`chrome::tree_row_at` / `panel_button_hit` / the z-slider),
/// so a click always lands on the organ drawn under it.
///
/// FALLBACK: when the panel is too short to host both organs plus the divider
/// (below [`PANEL_SPLIT_MIN_H`]), the two are MUTUALLY EXCLUSIVE — the whole
/// panel becomes ONE organ. `split` is then `false` and the OUTLINER rect is the
/// whole panel while the INSPECTOR is empty (or vice-versa); the caller chooses
/// which to show by passing the panel through `chrome::outliner_subrect` /
/// `chrome::inspector_subrect`, which apply the context rule. An empty panel
/// returns two empty rects.
pub fn panel_subrects(panel: Rect) -> PanelSplit {
    if panel.is_empty() {
        let empty = Rect {
            x: 0,
            y: 0,
            w: 0,
            h: 0,
        };
        return PanelSplit {
            split: false,
            outliner: empty,
            inspector: empty,
            divider_y: None,
        };
    }
    if panel.h < PANEL_SPLIT_MIN_H {
        // Too short for two organs — report no split; the caller resolves which
        // single organ to show (the whole panel) by context.
        return PanelSplit {
            split: false,
            outliner: panel,
            inspector: Rect {
                x: panel.x,
                y: panel.y,
                w: panel.w,
                h: 0,
            },
            divider_y: None,
        };
    }
    // OUTLINER ~55% of the panel, clamped so BOTH organs keep their minimum and
    // the divider row fits between them.
    let avail = panel.h - ORGAN_DIVIDER_H;
    let mut out_h = (panel.h * 55) / 100;
    out_h = out_h
        .max(OUTLINER_MIN_H)
        .min(avail.saturating_sub(INSPECTOR_MIN_H));
    let outliner = Rect {
        x: panel.x,
        y: panel.y,
        w: panel.w,
        h: out_h,
    };
    let divider_y = panel.y + out_h;
    let insp_y = divider_y + ORGAN_DIVIDER_H;
    let insp_h = (panel.y + panel.h).saturating_sub(insp_y);
    let inspector = Rect {
        x: panel.x,
        y: insp_y,
        w: panel.w,
        h: insp_h,
    };
    PanelSplit {
        split: true,
        outliner,
        inspector,
        divider_y: Some(divider_y),
    }
}

/// The result of [`panel_subrects`]: the two organ rects + whether the panel was
/// tall enough to split and (if so) the divider row.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PanelSplit {
    /// `true` when the panel was tall enough to host BOTH organs (with a divider);
    /// `false` when it falls back to one organ at a time.
    pub split: bool,
    /// The OUTLINER (object list) rect — the whole panel in the no-split fallback.
    pub outliner: Rect,
    /// The INSPECTOR (properties) rect — empty in the no-split fallback.
    pub inspector: Rect,
    /// The divider row (absolute y) when split, else `None`.
    pub divider_y: Option<usize>,
}

// ─── ADAPTIVE PANEL MODEL ────────────────────────────────────────────────
//
// The editor's side panels are not fixed furniture: each has a user-settable
// [`PanelState`] and, on narrow terminals, PRIORITY SHEDDING demotes them to a
// one-column ICON RAIL instead of proportionally starving every pane. The
// shedding priority is Grid > Status > Inspector > Preview > Outliner >
// Toolbox — the grid never sheds; the toolbox sheds first.
//
// [`compute_with_panels`] is the panel-aware entry point; [`compute_swapped`]
// keeps its older signature and DELEGATES to it with [`PanelPrefs::default()`]
// (adaptive off), so every layout/mouse test written against the legacy
// signature still sees byte-identical rects.

/// Identity of one editor pane, for zoom / rail / focus bookkeeping.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PanelId {
    /// The top-down edit grid (never sheds).
    Grid,
    /// The 3D preview viewport.
    Preview,
    /// The outliner (object tree) organ of the right panel.
    Outliner,
    /// The inspector (property sheet) organ of the right panel.
    Inspector,
    /// The vertical left toolbox column.
    Toolbox,
}

/// Display state of one collapsible panel. `Expanded` is the full pane; `Rail`
/// collapses it to a one-cell icon in the right-edge rail (click/key reopens);
/// `Hidden` removes it entirely (no icon).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum PanelState {
    /// Fully visible (the default).
    #[default]
    Expanded,
    /// Collapsed to a 1-col rail icon; reopened by clicking the icon.
    Rail,
    /// Not shown at all (no rail icon either).
    Hidden,
}

/// Per-panel user preferences + the adaptive (width-shedding) switch. Session
/// state only — deliberately NOT persisted to a config file.
///
/// `Default` is all-Expanded with `adaptive = false`, which reproduces the
/// legacy layout exactly — that is what [`compute_swapped`] delegates with, so
/// tests written against the legacy layout need no edits. The live editor sets
/// `adaptive = true`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct PanelPrefs {
    /// The left toolbox column.
    pub toolbox: PanelState,
    /// The outliner organ (top of the right panel).
    pub outliner: PanelState,
    /// The inspector organ (bottom of the right panel).
    pub inspector: PanelState,
    /// `true` enables the width-threshold priority shedding below. Off by
    /// default so the legacy entry points stay byte-identical.
    pub adaptive: bool,
}

/// The EFFECTIVE per-panel states after width shedding — what actually renders.
/// Pure function of `(mode, w, prefs)` via [`effective_panels`], so the painter
/// (`chrome`/`main`) and the hit-tests (`mouse`) can never disagree.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct EffPanels {
    /// Effective toolbox state.
    pub toolbox: PanelState,
    /// Effective outliner state.
    pub outliner: PanelState,
    /// Effective inspector state.
    pub inspector: PanelState,
}

impl EffPanels {
    /// `true` when any panel is collapsed to the rail (the rail column shows).
    pub fn any_rail(self) -> bool {
        [self.toolbox, self.outliner, self.inspector].contains(&PanelState::Rail)
    }
}

/// Below this terminal width (cols) the adaptive ladder sheds the TOOLBOX and
/// the OUTLINER to the rail (the inspector — higher priority — survives).
/// Chosen ABOVE the acceptance size 100×30 and BELOW the canonical 120×40 test
/// size, so the legacy layout's 120-col expectations are untouched.
pub const SHED_RAIL_MAX_W: usize = 120;
/// Below this width the ladder goes SINGLE-PANE: every side panel (toolbox,
/// outliner, inspector) rails and the preview collapses — the grid gets
/// everything (plus toolbar/status/rail). Covers the 80×24 floor.
pub const SHED_SINGLE_MAX_W: usize = 90;
/// Width of the icon rail column carved off the RIGHT edge when any panel is
/// collapsed to [`PanelState::Rail`].
pub const RAIL_W: usize = 1;

/// The EFFECTIVE panel states for terminal width `w`: user prefs demoted by the
/// priority-shedding ladder when `prefs.adaptive` holds. `Full3D` hides every
/// side panel outright (a walk-through has no chrome). Pure — THE single source
/// of truth consumed by [`compute_with_panels`], the organ sub-rects
/// (`chrome::{outliner,inspector}_subrect_eff`), and the rail hit-test.
pub fn effective_panels(mode: ViewMode, w: usize, prefs: PanelPrefs) -> EffPanels {
    if matches!(mode, ViewMode::Full3D) {
        return EffPanels {
            toolbox: PanelState::Hidden,
            outliner: PanelState::Hidden,
            inspector: PanelState::Hidden,
        };
    }
    let mut e = EffPanels {
        toolbox: prefs.toolbox,
        outliner: prefs.outliner,
        inspector: prefs.inspector,
    };
    if !prefs.adaptive {
        return e;
    }
    let demote = |s: &mut PanelState| {
        if *s == PanelState::Expanded {
            *s = PanelState::Rail;
        }
    };
    if w < SHED_SINGLE_MAX_W {
        // Single-pane band: everything sheds; the grid wins.
        demote(&mut e.toolbox);
        demote(&mut e.outliner);
        demote(&mut e.inspector);
    } else if w < SHED_RAIL_MAX_W {
        // Rail band: toolbox first, then the outliner — the INSPECTOR (highest
        // side-panel priority) keeps the right panel. When the user explicitly
        // railed/hid the inspector, the outliner may keep the panel instead
        // (that is the rail-icon organ swap).
        demote(&mut e.toolbox);
        if e.inspector == PanelState::Expanded {
            demote(&mut e.outliner);
        }
    }
    e
}

/// Internal layout options — the parameterization [`compute_with_panels`]
/// resolves panel states into. Defaults reproduce the legacy [`compute`].
#[derive(Clone, Copy)]
struct LayoutOpts {
    /// Carve the left toolbox column (subject to the width budget).
    toolbox_on: bool,
    /// Show the 3D preview in Split (off in the single-pane band).
    preview_on: bool,
    /// Natural width cap of the right panel column: `PANEL_W` normally,
    /// `PANEL_WIDE_W` under the accordion boost, `0` = panel collapsed.
    panel_cap: usize,
}

impl Default for LayoutOpts {
    fn default() -> Self {
        Self {
            toolbox_on: true,
            preview_on: true,
            panel_cap: PANEL_W,
        }
    }
}

/// ACCORDION width cap for the focused right panel (lazygit
/// `expandFocusedSidePanel` model): the outliner-focused panel may grow to this
/// instead of [`PANEL_W`], budget permitting.
const PANEL_WIDE_W: usize = 38;

/// The three panes for the current view mode. Any pane may be empty (zero w/h),
/// meaning "not shown in this mode".
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Layout {
    /// Top toolbar strip (full width, height 1 when the fb is tall enough, else
    /// empty). Clickable mode/Open/Save/Help buttons; all other panes sit below.
    pub toolbar: Rect,
    /// RETIRED row-1 strip. The old horizontal TOOL PALETTE row was replaced by the
    /// vertical [`Layout::toolbox`] left column (this commit). The field is kept
    /// ALWAYS-EMPTY so the `Layout` shape and downstream consumers/tests stay
    /// stable; nothing carves it any more.
    pub tool_bar: Rect,
    /// Vertical LEFT TOOLBOX column (Geometry / Shapes / Objects / Paint / Erase),
    /// [`TOOLBOX_W`] cells wide, spanning the body height between the top command
    /// band and the bottom status row. The center (grid + preview) and the right
    /// panel are shifted right by its width. Empty (zero-size) when the terminal is
    /// too narrow to afford the column AND keep the center grid at its usable
    /// minimum, below the [`too_small`] floor, OR in `Full3D` (a walk-through hides
    /// the toolbox).
    pub toolbox: Rect,
    /// Top-down z-slice edit grid.
    pub grid: Rect,
    /// Side panel (chrome / status / help).
    pub panel: Rect,
    /// Live 3D preview.
    pub preview: Rect,
    /// Elevation side-window: vertical cross-section at the cursor. Empty in
    /// `Full3D`; a bottom strip of the panel column in Split / `Full2D`.
    pub elevation: Rect,
    /// OBJECT TREE / OUTLINER side-window. Empty unless the tree is open AND the
    /// mode shows a panel column (Split / `Full2D`) tall enough to carve it; it
    /// then takes the bottom of the panel column (below the chrome, replacing the
    /// elevation strip while open so the layout stays within bounds).
    pub tree: Rect,
    /// Full-width STATUS BAR carved off the BOTTOM row of the framebuffer. It
    /// holds the "in-the-moment" readout (view, z, cursor, collisions, undo/redo,
    /// saved flag, status message, key hint). Height 1 when the fb is tall enough
    /// to spare it, otherwise empty. Every body pane sits ABOVE it (their height
    /// is reduced by one row).
    pub status: Rect,
    /// The 1-col ICON RAIL down the RIGHT edge of the body: one
    /// icon per panel collapsed to [`PanelState::Rail`]; clicking an icon
    /// reopens its panel. Empty unless [`compute_with_panels`] was given a
    /// pref set whose effective states rail something — the legacy
    /// [`compute`]/[`compute_swapped`] paths always leave it empty.
    pub rail: Rect,
}

/// Height of the top TOOLBAR strip (the clickable mode/Open/Save/Help row). One
/// row when the fb is tall enough to spare it; otherwise the toolbar is empty and
/// the panes use the full height (a tiny terminal keeps content over chrome).
const TOOLBAR_H: usize = 1;
/// Below this fb height we drop the toolbar entirely so a very short terminal
/// still shows the editor panes.
const TOOLBAR_MIN_FB_H: usize = 6;
/// FULL width of the vertical LEFT TOOLBOX column ([`Layout::toolbox`]) — the
/// roomy mode. Wide enough for the longest tool word ("Material"/"Ceiling" = 8)
/// plus a leading `▶` marker and a pad. Carved off the left of the body when the
/// leftmost pane can still keep its usable minimum after reserving it
/// (anti-sliver). On a narrower terminal the toolbox DEGRADES to
/// [`TOOLBOX_ICON_W`] (icon-only column) and then to `0` (hidden) — see
/// [`toolbox_w_for`].
pub const TOOLBOX_W: usize = 12;

/// COMPACT (icon-only) width of the toolbox column: one single-width glyph per
/// tool. Used at narrow widths where the full 12-wide word column would starve the
/// center grid but a 3-wide icon strip still fits. The painter
/// (`palette_bar::draw_toolbox`) renders icons when the rect is this narrow; the
/// hit-test is identical (both drive off `tool_button_rows`).
pub const TOOLBOX_ICON_W: usize = 3;

/// FULL width reserved for the side panel (the right column in both Split and
/// `Full2D` — the canonical TOOLBOX | center | PANEL arrangement). 30 cells so the
/// panel can host TWO stacked organs (OUTLINER over INSPECTOR) with full-word
/// labels. Shrinks toward [`PANEL_MIN_W`] before collapsing — see [`panel_w_for`].
pub(crate) const PANEL_W: usize = 30;
/// MINIMUM useful panel width: below this the OUTLINER/INSPECTOR labels and the
/// Z-slider ([`crate::chrome::Z_SLIDER_W`] = 16) no longer fit, so the panel
/// collapses to zero rather than shrink further. The panel shrinks from
/// [`PANEL_W`] down to this floor before collapsing, so a mid-width terminal keeps
/// a (narrower) panel instead of losing it outright.
const PANEL_MIN_W: usize = 24;

/// The clamped panel width for `mode` at body width `w` — the SAME clamp the
/// per-mode splitters apply, factored out so `compute` can budget the toolbox
/// carve against it before laying out the center. `Full3D` has no panel.
///
/// DEGRADATION: the panel keeps its full [`PANEL_W`] when the body can spare it
/// (the per-mode `w/4` / `w/3` guard); as the terminal narrows it SHRINKS toward
/// [`PANEL_MIN_W`] rather than collapsing immediately, and only drops to zero when
/// even the minimum would leave the center below [`MIN_PANE_W`].
fn panel_w_for(mode: ViewMode, w: usize) -> usize {
    panel_w_for_cap(mode, w, PANEL_W)
}

/// [`panel_w_for`] with an explicit natural-width cap — the ACCORDION variant:
/// the focused right panel may use [`PANEL_WIDE_W`] instead of [`PANEL_W`].
/// Identical clamp/degradation rules otherwise.
fn panel_w_for_cap(mode: ViewMode, w: usize, natural: usize) -> usize {
    let cap = match mode {
        // The accordion (wide) cap may take up to a third of the width; the
        // natural cap keeps the legacy quarter clamp exactly.
        ViewMode::Split if natural > PANEL_W => natural.min(w / 3),
        ViewMode::Split => natural.min(w / 4),
        ViewMode::Full2D => natural.min(w / 3),
        ViewMode::Full3D => return 0,
    };
    if cap >= PANEL_MIN_W {
        // The clamp still affords a full-or-near-full panel.
        return cap;
    }
    // The `w/4` / `w/3` clamp dropped below the useful minimum. Keep a
    // PANEL_MIN_W-wide panel as long as the center can still stay usable WITHOUT a
    // toolbox (the toolbox yields first); otherwise collapse the panel to zero.
    if w.saturating_sub(PANEL_MIN_W) >= MIN_PANE_W {
        PANEL_MIN_W
    } else {
        0
    }
}

/// The toolbox column width to carve for `mode` at body width `w`, given the
/// already-budgeted `panel_w`. Degrades `TOOLBOX_W → TOOLBOX_ICON_W → 0` by the
/// width budget: the FULL word column when the center keeps [`MIN_PANE_W`] after
/// reserving it plus the panel, else the COMPACT icon column when the narrower
/// strip still leaves a usable center, else hidden (0) so the grid is never
/// starved below its minimum. `Full3D` never carves a toolbox (a walk-through hides
/// it). THE single source of truth `compute` uses to size + place the toolbox.
pub const fn toolbox_w_for(mode: ViewMode, w: usize, panel_w: usize) -> usize {
    if matches!(mode, ViewMode::Full3D) {
        return 0;
    }
    if w.saturating_sub(TOOLBOX_W + panel_w) >= MIN_PANE_W {
        TOOLBOX_W
    } else if w.saturating_sub(TOOLBOX_ICON_W + panel_w) >= MIN_PANE_W {
        TOOLBOX_ICON_W
    } else {
        0
    }
}
/// Minimum cell-area width a body pane (grid / preview) needs to be worth
/// drawing into. Below this a pane is an unusable sliver; in Split we collapse
/// the preview (return it empty) so the grid keeps usable width rather than
/// splitting the row into two slivers.
const MIN_PANE_W: usize = 12;
/// Height of the full-width STATUS BAR carved off the BOTTOM row (all view
/// modes). One row when the fb is tall enough to spare it above the body floor;
/// otherwise the status bar is empty and the body uses the full height.
const STATUS_H: usize = 1;
/// Below this fb height we drop the bottom status bar so a very short terminal
/// still shows the editor body.
const STATUS_MIN_FB_H: usize = 6;
/// Side-window carve for the panel column. The side panel now hosts BOTH the
/// OUTLINER and the INSPECTOR as TWO STACKED ORGANS that split the FULL panel
/// height internally (see `chrome::panel_subrects`), so the panel is no longer
/// shortened to make room for a separate bottom strip — it keeps its full height.
///
/// The `elevation` cross-section and the old carved `tree` strip are therefore
/// retired from the right column (the outliner is always visible inside the panel
/// instead). The two rects are kept (always empty) so downstream consumers and
/// the `Layout` shape are unchanged. `tree_open` no longer affects the carve; it
/// is retained only to keep the `compute` signature stable for callers/tests.
/// Returns `(panel, elevation=empty, tree=empty)`.
const fn carve_side_windows(panel: Rect, _tree_open: bool) -> (Rect, Rect, Rect) {
    let empty = Rect {
        x: 0,
        y: 0,
        w: 0,
        h: 0,
    };
    (panel, empty, empty)
}

/// SWAP-AWARE layout: identical to the un-swapped [`compute`], but when `swap` is
/// `true` AND `mode` is [`ViewMode::Split`], the two CENTER viewports trade
/// left/right positions so the 3D PREVIEW sits LEFT and the 2D GRID sits RIGHT.
/// Full2D/Full3D ignore `swap` (they show at most one center viewport). The returned
/// [`Layout`] still holds each viewport in its own field (`grid`/`preview`) at its
/// correct PHYSICAL position, so `pane_at` and `draw_system` work UNCHANGED — they
/// read `l.grid` / `l.preview` and never assume an x-order. THE entry point the
/// editor (`draw_system` / `apply_mouse`) uses with the live `swap_viewports` flag.
// The legacy entry points below are a PRESERVED compatibility seam: the live
// editor routes through `EditorState::layout_for` → `compute_with_panels`,
// while non-adaptive layout/mouse test suites (and any future non-adaptive
// caller) keep `compute_swapped`/`compute` byte-identical via delegation. They
// are exercised by the test builds only, hence the knowing dead-code carve.
#[cfg_attr(not(test), allow(dead_code))]
pub fn compute_swapped(mode: ViewMode, w: usize, h: usize, tree_open: bool, swap: bool) -> Layout {
    // Delegates to the panel-aware entry point with default prefs (adaptive
    // OFF) and no zoom — which resolves to the identical rects the
    // non-adaptive code produced.
    compute_with_panels(mode, w, h, tree_open, swap, PanelPrefs::default(), None)
}

/// Exchange the two CENTER viewports' positions when `swap` holds (Split only —
/// the only mode with two center viewports). Factored out of the old
/// `compute_swapped` so the panel-aware entry point applies the same rule.
fn apply_swap(mut l: Layout, mode: ViewMode, swap: bool) -> Layout {
    // Only Split has two side-by-side (or stacked) center viewports to swap, and a
    // swap is meaningful only when BOTH are present. We exchange the rects' POSITIONS
    // (x for side-by-side, y for the narrow stacked case) while keeping `grid` in
    // `l.grid` and `preview` in `l.preview`, so downstream code stays order-agnostic.
    if swap && matches!(mode, ViewMode::Split) && !l.grid.is_empty() && !l.preview.is_empty() {
        let (g, p) = (l.grid, l.preview);
        // The preview takes the grid's slot in both arrangements.
        l.preview = g;
        if g.y == p.y {
            // Side-by-side: the grid takes the preview's right slot. Widths
            // follow their new horizontal position.
            l.grid = Rect {
                x: g.x + g.w,
                y: p.y,
                w: p.w,
                h: p.h,
            };
        } else {
            // Stacked (narrow center): preview on top, grid below.
            l.grid = Rect {
                x: p.x,
                y: g.y + g.h,
                w: p.w,
                h: p.h,
            };
        }
    }
    l
}

/// The PANEL-AWARE layout entry point: [`compute_swapped`]'s
/// behaviour extended with per-panel collapse states, priority shedding, the
/// right-edge icon rail, the accordion width boost, and the `z` zoom-to-pane
/// short-circuit. THE entry point the live editor uses (via
/// `EditorState::layout_for`); the legacy entry points delegate here with
/// defaults, so their behaviour is unchanged.
pub fn compute_with_panels(
    mode: ViewMode,
    w: usize,
    h: usize,
    tree_open: bool,
    swap: bool,
    prefs: PanelPrefs,
    zoomed: Option<PanelId>,
) -> Layout {
    // ── ZOOM-TO-PANE ── one pane fullscreens the body; toolbar + status stay.
    // Only the two center viewports are zoomable (v1); the same key restores.
    if let Some(zp) = zoomed {
        if !too_small(w, h) && matches!(zp, PanelId::Grid | PanelId::Preview) {
            let toolbar_h = if h >= TOOLBAR_MIN_FB_H { TOOLBAR_H } else { 0 };
            let status_h = if h >= STATUS_MIN_FB_H { STATUS_H } else { 0 };
            let empty = Rect {
                x: 0,
                y: 0,
                w: 0,
                h: 0,
            };
            let body = Rect {
                x: 0,
                y: toolbar_h,
                w,
                h: h.saturating_sub(toolbar_h + status_h),
            };
            let mut l = Layout {
                toolbar: if toolbar_h > 0 {
                    Rect {
                        x: 0,
                        y: 0,
                        w,
                        h: toolbar_h,
                    }
                } else {
                    empty
                },
                tool_bar: empty,
                toolbox: empty,
                grid: empty,
                panel: empty,
                preview: empty,
                elevation: empty,
                tree: empty,
                status: if status_h > 0 {
                    Rect {
                        x: 0,
                        y: h - status_h,
                        w,
                        h: status_h,
                    }
                } else {
                    empty
                },
                rail: empty,
            };
            match zp {
                PanelId::Grid => l.grid = body,
                PanelId::Preview => l.preview = body,
                _ => {}
            }
            return l;
        }
    }
    let eff = effective_panels(mode, w, prefs);
    // The rail column shows only when something is actually collapsed to it,
    // the body exists at all, and carving it would not push the body below
    // the usable floor (the grid always outranks the rail).
    let rail_w = if eff.any_rail() && !too_small(w, h) && !too_small(w - RAIL_W, h) {
        RAIL_W
    } else {
        0
    };
    let panel_shown = eff.outliner == PanelState::Expanded || eff.inspector == PanelState::Expanded;
    // Accordion: the focused (tree-open) outliner widens the right panel.
    let accordion = prefs.adaptive && tree_open && eff.outliner == PanelState::Expanded;
    let opts = LayoutOpts {
        toolbox_on: eff.toolbox == PanelState::Expanded,
        // Single-pane band: the preview sheds too — the grid gets the center.
        preview_on: !(prefs.adaptive && w < SHED_SINGLE_MAX_W),
        panel_cap: match (panel_shown, accordion) {
            (false, _) => 0,
            (true, false) => PANEL_W,
            (true, true) => PANEL_WIDE_W,
        },
    };
    let mut l = compute_opts(mode, w - rail_w, h, tree_open, &opts);
    // The chrome rows span the FULL width (the rail is a body-only column).
    if !l.toolbar.is_empty() {
        l.toolbar.w = w;
    }
    if !l.status.is_empty() {
        l.status.w = w;
    }
    if rail_w > 0 {
        let rail_y = l.toolbar.h; // 0 when the toolbar collapsed
        let rail_bottom = if l.status.is_empty() { h } else { l.status.y };
        l.rail = Rect {
            x: w - rail_w,
            y: rail_y,
            w: rail_w,
            h: rail_bottom.saturating_sub(rail_y),
        };
    }
    apply_swap(l, mode, swap)
}

/// Compute pane rects for `mode` given a framebuffer `w` x `h`. `tree_open` is
/// retained for signature stability (the outliner is now always an in-panel organ
/// — see [`carve_side_windows`]). This is the UN-SWAPPED layout; the editor calls
/// [`compute_swapped`] with the live `swap_viewports` flag (swap only flips the two
/// CENTER viewports in Split).
///
/// Canonical body arrangement, left → right: `TOOLBOX | center (grid + preview) |
/// PANEL`. The PANEL is the FAR-RIGHT column; the grid + preview share the
/// flexible center between the toolbox and the panel. `Full3D` is the exception: it
/// hides BOTH the toolbox and the panel so the preview fills the whole body.
///
/// Reserves a one-row TOOLBAR at the very top (row 0) when the fb is tall enough;
/// every other pane is then laid out in the remaining area below it and shifted
/// down by the toolbar height. On a tiny terminal the toolbar collapses to empty
/// and the panes use the full height.
// See the compatibility-seam note on [`compute_swapped`].
#[cfg_attr(not(test), allow(dead_code))]
pub fn compute(mode: ViewMode, w: usize, h: usize, tree_open: bool) -> Layout {
    compute_opts(mode, w, h, tree_open, &LayoutOpts::default())
}

/// The parameterized layout core — the old `compute` body with the panel
/// switches [`compute_with_panels`] resolves. Defaults reproduce [`compute`]
/// exactly.
fn compute_opts(mode: ViewMode, w: usize, h: usize, tree_open: bool, opts: &LayoutOpts) -> Layout {
    // Carve the toolbar (row 0) off the top, then lay the body out below it. The
    // old row-1 tool-palette strip is RETIRED — tools now live in the vertical
    // left [`Layout::toolbox`] column carved off the body below.
    let toolbar_h = if h >= TOOLBAR_MIN_FB_H { TOOLBAR_H } else { 0 };
    let chrome_h = toolbar_h;
    // Carve a one-row STATUS BAR off the very BOTTOM when the fb is tall enough.
    // It is full width and every body pane sits above it.
    let status_h = if h >= STATUS_MIN_FB_H { STATUS_H } else { 0 };
    let toolbar = if toolbar_h > 0 {
        Rect {
            x: 0,
            y: 0,
            w,
            h: toolbar_h,
        }
    } else {
        Rect {
            x: 0,
            y: 0,
            w: 0,
            h: 0,
        }
    };
    // The old row-1 tool-palette strip is retired; the field stays always-empty.
    let tool_bar = Rect {
        x: 0,
        y: 0,
        w: 0,
        h: 0,
    };
    // Body occupies the rows between the top chrome and the bottom status bar.
    let body_h = h.saturating_sub(chrome_h + status_h);
    // The status bar is the bottom row of the fb (it never overlaps the body).
    let status = if status_h > 0 {
        Rect {
            x: 0,
            y: h - status_h,
            w,
            h: status_h,
        }
    } else {
        Rect {
            x: 0,
            y: 0,
            w: 0,
            h: 0,
        }
    };
    // The carved LEFT TOOLBOX column width (12 / 3 / 0 by width budget; set while
    // carving the leftmost body pane below); drives the `toolbox` rect width.
    let mut toolbox_carved_w = 0usize;
    // Below the usable floor the body cannot host a real pane (it would collapse
    // to 1-wide slivers). Return empty body rects — chrome stays full-width when
    // it fits — and let `main` draw a "terminal too small" message instead of
    // trying to render into unusable panes.
    let mut body = if too_small(w, h) {
        let empty = Rect {
            x: 0,
            y: 0,
            w: 0,
            h: 0,
        };
        Layout {
            toolbar: empty,
            tool_bar: empty,
            toolbox: empty,
            grid: empty,
            panel: empty,
            preview: empty,
            elevation: empty,
            tree: empty,
            status: empty,
            rail: empty,
        }
    } else {
        // Canonical arrangement: TOOLBOX(left) | CENTER viewports (grid+preview) |
        // PANEL(right). We first decide whether the left TOOLBOX column is afforded
        // (it needs the body to keep a usable CENTER after reserving it plus the
        // panel), then hand the per-mode splitter the body width LEFT of the
        // toolbox so grid/preview/panel are laid out in that region and shifted
        // right by the toolbox width. Full3D is special: it hides BOTH the toolbox
        // and the panel — the preview fills the whole body for a clean walk-through.
        // Budget the right PANEL from the FULL width (so it keeps its natural width
        // regardless of the toolbox carve), then afford the toolbox only when the
        // remaining body still leaves the CENTER a usable grid:
        // body - toolbox - panel >= MIN_PANE_W. Full3D never carves a toolbox or
        // a panel (you walk, not edit).
        let panel_w = if opts.panel_cap == 0 {
            // Panel collapsed to the rail / hidden: the center takes its width.
            0
        } else if opts.panel_cap > PANEL_W {
            // Accordion boost, never narrower than the legacy width.
            panel_w_for_cap(mode, w, opts.panel_cap).max(panel_w_for(mode, w))
        } else {
            panel_w_for(mode, w)
        };
        // Degrade the toolbox 12 → 3 → 0 by the width budget (Full3D = 0), or
        // drop it outright when collapsed to the rail / hidden.
        let toolbox_w = if opts.toolbox_on {
            toolbox_w_for(mode, w, panel_w)
        } else {
            0
        };
        toolbox_carved_w = toolbox_w;
        // Width available to the grid/preview/panel region, right of the toolbox.
        let inner_w = w - toolbox_w;
        let mut b = match mode {
            ViewMode::Split => split(inner_w, body_h, panel_w, tree_open, opts.preview_on),
            ViewMode::Full2D => full_2d(inner_w, body_h, panel_w, tree_open),
            ViewMode::Full3D => full_3d(inner_w, body_h),
        };
        // Shift the body panes right past the toolbox column.
        if toolbox_w > 0 {
            for r in [&mut b.grid, &mut b.panel, &mut b.preview] {
                if !r.is_empty() {
                    r.x += toolbox_w;
                }
            }
        }
        b
    };
    // Shift every body pane down past the chrome rows.
    for r in [
        &mut body.grid,
        &mut body.panel,
        &mut body.preview,
        &mut body.elevation,
        &mut body.tree,
    ] {
        if !r.is_empty() {
            r.y += chrome_h;
        }
    }
    // The TOOLBOX is the left column of the body: it spans the body height between
    // the top chrome and the bottom status row. Width is the carved budget (12
    // full / 3 icon-only / 0 hidden); present only when it was afforded (carved out
    // of the leftmost body pane without starving the center).
    let toolbox = if toolbox_carved_w > 0 && body_h > 0 {
        Rect {
            x: 0,
            y: chrome_h,
            w: toolbox_carved_w,
            h: body_h,
        }
    } else {
        Rect {
            x: 0,
            y: 0,
            w: 0,
            h: 0,
        }
    };
    body.toolbar = toolbar;
    body.tool_bar = tool_bar;
    body.toolbox = toolbox;
    body.status = status;
    body
}

/// Below this CENTER width we STACK the two viewports vertically (grid on top,
/// preview below) rather than side-by-side, so neither becomes an unusable
/// sliver. Sized so a side-by-side split would give each less than [`MIN_PANE_W`]
/// plus a little breathing room; below it, a vertical stack reads better.
const CENTER_STACK_MIN_W: usize = 2 * MIN_PANE_W + 6;
/// Minimum rows a stacked viewport (grid OR preview) needs to be worth drawing.
/// Below `2 * this` we cannot stack two usable viewports, so we fall back to the
/// grid-only collapse (the grid wins).
const STACK_MIN_H: usize = 4;

/// Split: the canonical `[ grid | preview | panel ]` — the PANEL is the FAR-RIGHT
/// column; the grid (left of center) and preview (right of center) share the
/// flexible CENTER region between the (caller-carved) toolbox and the panel.
///
/// Anti-sliver rule (called only above the [`too_small`] floor): both the grid
/// and the preview must be at least [`MIN_PANE_W`] cells of drawable area, or we
/// don't draw two of them. If, after reserving the panel, the CENTER can't give
/// BOTH the grid and the preview a usable width SIDE-BY-SIDE, we either (a) STACK
/// them vertically when the center is narrow but tall enough, or (b) collapse the
/// preview and hand the whole center to the grid — one usable pane beats two
/// unusable slivers. The grid always stays usable (the floor guarantees it).
fn split(w: usize, h: usize, panel_w: usize, tree_open: bool, preview_on: bool) -> Layout {
    // `panel_w` is budgeted by `compute` from the FULL width so the right column
    // keeps its natural width; clamp to the inner width here so it never exceeds it.
    let panel_w = panel_w.min(w);
    // The panel is the FAR-RIGHT column; the center is everything left of it.
    let center_w = w.saturating_sub(panel_w);
    let panel_x = center_w;
    let (panel, elevation, tree) = carve_side_windows(
        Rect {
            x: panel_x,
            y: 0,
            w: panel_w,
            h,
        },
        tree_open,
    );

    // Decide how the center hosts grid + preview. When the preview is shed
    // (single-pane band), the grid takes the whole center outright.
    let side_by_side = preview_on && center_w >= 2 * MIN_PANE_W;
    let (grid, preview) = if !preview_on {
        (
            Rect {
                x: 0,
                y: 0,
                w: center_w,
                h,
            },
            Rect {
                x: 0,
                y: 0,
                w: 0,
                h: 0,
            },
        )
    } else if side_by_side && center_w >= CENTER_STACK_MIN_W {
        // Side-by-side: grid on the left of center, preview on the right.
        let grid_w = (center_w / 2).max(MIN_PANE_W).min(center_w - MIN_PANE_W);
        let preview_w = center_w - grid_w;
        (
            Rect {
                x: 0,
                y: 0,
                w: grid_w,
                h,
            },
            Rect {
                x: grid_w,
                y: 0,
                w: preview_w,
                h,
            },
        )
    } else if center_w >= MIN_PANE_W && h >= 2 * STACK_MIN_H {
        // Narrow but tall center: STACK the viewports vertically (grid on top).
        let grid_h = h / 2;
        let preview_h = h - grid_h;
        (
            Rect {
                x: 0,
                y: 0,
                w: center_w,
                h: grid_h,
            },
            Rect {
                x: 0,
                y: grid_h,
                w: center_w,
                h: preview_h,
            },
        )
    } else {
        // Too small for two usable viewports: collapse the preview, give the whole
        // center to the grid so it stays usable.
        (
            Rect {
                x: 0,
                y: 0,
                w: center_w,
                h,
            },
            Rect {
                x: 0,
                y: 0,
                w: 0,
                h: 0,
            },
        )
    };
    Layout {
        toolbar: Rect {
            x: 0,
            y: 0,
            w: 0,
            h: 0,
        },
        tool_bar: Rect {
            x: 0,
            y: 0,
            w: 0,
            h: 0,
        },
        toolbox: Rect {
            x: 0,
            y: 0,
            w: 0,
            h: 0,
        },
        grid,
        panel,
        preview,
        elevation,
        tree,
        status: Rect {
            x: 0,
            y: 0,
            w: 0,
            h: 0,
        },
        rail: Rect {
            x: 0,
            y: 0,
            w: 0,
            h: 0,
        },
    }
}

/// Full2D: `[ grid(center) | panel(right) ]` — the grid fills the body width
/// (right of the caller-carved toolbox) minus the PANEL on the far right; no
/// preview.
fn full_2d(w: usize, h: usize, panel_w: usize, tree_open: bool) -> Layout {
    // `panel_w` is budgeted by `compute` from the FULL width; clamp to the inner
    // width so it never exceeds the region left of the toolbox.
    let panel_w = panel_w.min(w);
    let grid_w = w.saturating_sub(panel_w);
    let (panel, elevation, tree) = carve_side_windows(
        Rect {
            x: grid_w,
            y: 0,
            w: panel_w,
            h,
        },
        tree_open,
    );
    Layout {
        toolbar: Rect {
            x: 0,
            y: 0,
            w: 0,
            h: 0,
        },
        tool_bar: Rect {
            x: 0,
            y: 0,
            w: 0,
            h: 0,
        },
        toolbox: Rect {
            x: 0,
            y: 0,
            w: 0,
            h: 0,
        },
        grid: Rect {
            x: 0,
            y: 0,
            w: grid_w,
            h,
        },
        panel,
        preview: Rect {
            x: 0,
            y: 0,
            w: 0,
            h: 0,
        },
        elevation,
        tree,
        status: Rect {
            x: 0,
            y: 0,
            w: 0,
            h: 0,
        },
        rail: Rect {
            x: 0,
            y: 0,
            w: 0,
            h: 0,
        },
    }
}

/// Full3D: a clean walk-through — the preview FILLS THE WHOLE BODY. The left
/// TOOLBOX and the right PANEL are BOTH hidden in this mode (you walk, not edit),
/// so `compute` passes the full body width here and the preview spans it. The
/// readouts live in the shared full-width STATUS BAR carved off the bottom of the
/// framebuffer by [`compute`] (so Full3D uses the same bar as the other modes).
fn full_3d(w: usize, h: usize) -> Layout {
    Layout {
        toolbar: Rect {
            x: 0,
            y: 0,
            w: 0,
            h: 0,
        },
        tool_bar: Rect {
            x: 0,
            y: 0,
            w: 0,
            h: 0,
        },
        toolbox: Rect {
            x: 0,
            y: 0,
            w: 0,
            h: 0,
        },
        grid: Rect {
            x: 0,
            y: 0,
            w: 0,
            h: 0,
        },
        panel: Rect {
            x: 0,
            y: 0,
            w: 0,
            h: 0,
        },
        preview: Rect { x: 0, y: 0, w, h },
        elevation: Rect {
            x: 0,
            y: 0,
            w: 0,
            h: 0,
        },
        tree: Rect {
            x: 0,
            y: 0,
            w: 0,
            h: 0,
        },
        status: Rect {
            x: 0,
            y: 0,
            w: 0,
            h: 0,
        },
        rail: Rect {
            x: 0,
            y: 0,
            w: 0,
            h: 0,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toolbar_reserves_top_row_and_shifts_panes_down() {
        let l = compute(ViewMode::Split, 120, 40, false);
        // The toolbar is the full-width top row; the old row-1 tool-palette strip is
        // retired (always empty) — tools now live in the vertical left toolbox.
        assert_eq!(
            l.toolbar,
            Rect {
                x: 0,
                y: 0,
                w: 120,
                h: 1
            }
        );
        assert!(l.tool_bar.is_empty(), "row-1 tool strip retired");
        // The status bar is the full-width BOTTOM row.
        assert_eq!(
            l.status,
            Rect {
                x: 0,
                y: 39,
                w: 120,
                h: 1
            }
        );
        // Body panes start at y=1 (below the single chrome row).
        assert_eq!(l.grid.y, 1, "grid sits below the one chrome row");
        assert_eq!(l.preview.y, 1);
        // Body height is fb height minus the one chrome row AND the status row.
        assert_eq!(l.grid.h, 38);
        // No body pane overlaps the bottom status row.
        assert!(l.grid.y + l.grid.h <= l.status.y);
        // The TOOLBOX is the left column spanning the body height, below the chrome.
        assert_eq!(
            l.toolbox,
            Rect {
                x: 0,
                y: 1,
                w: TOOLBOX_W,
                h: 38
            }
        );
        // The grid is shifted right past the toolbox column.
        assert_eq!(l.grid.x, TOOLBOX_W, "grid shifted right past the toolbox");
    }

    #[test]
    fn status_bar_is_bottom_row_and_no_pane_overlaps_it() {
        // In every view mode, when a status bar is carved, no body pane intrudes on
        // the bottom row.
        for mode in [ViewMode::Split, ViewMode::Full2D, ViewMode::Full3D] {
            let l = compute(mode, 120, 40, false);
            assert_eq!(
                l.status,
                Rect {
                    x: 0,
                    y: 39,
                    w: 120,
                    h: 1
                },
                "{mode:?} status row"
            );
            for r in [l.grid, l.panel, l.preview, l.elevation, l.tree] {
                if !r.is_empty() {
                    assert!(
                        r.y + r.h <= l.status.y,
                        "{mode:?} pane {r:?} must sit above the status row {}",
                        l.status.y
                    );
                }
            }
        }
    }

    #[test]
    fn toolbar_collapses_on_tiny_terminal() {
        // Below the threshold both chrome rows are empty and panes use full height.
        let l = compute(ViewMode::Split, 40, 4, false);
        assert!(l.toolbar.is_empty(), "no toolbar on a very short terminal");
        assert!(
            l.tool_bar.is_empty(),
            "no tool bar on a very short terminal"
        );
        assert_eq!(l.grid.y, 0, "panes start at the top when no chrome");
    }

    #[test]
    fn retired_tool_bar_is_always_empty() {
        // The old row-1 tool-palette strip is retired: `tool_bar` is ALWAYS empty
        // regardless of size, while the toolbar (Save/Help) is kept whenever the fb
        // is tall enough. (At 40x7 we're below the usable body floor, so the body
        // panes — and the toolbox — are empty too, but the toolbar still shows.)
        let l = compute(ViewMode::Split, 40, 7, false);
        assert!(!l.toolbar.is_empty(), "toolbar kept");
        assert!(l.tool_bar.is_empty(), "retired tool bar always empty");
        assert!(too_small(40, 7), "this size is below the usable body floor");
        assert!(l.grid.is_empty(), "body collapses below the floor");
        assert!(l.toolbox.is_empty(), "toolbox empty below the floor");
        // And at a usable size the strip is still empty.
        let big = compute(ViewMode::Split, 120, 40, false);
        assert!(
            big.tool_bar.is_empty(),
            "retired tool bar empty at usable size"
        );
    }

    #[test]
    fn split_panes_tile_the_width() {
        let l = compute(ViewMode::Split, 120, 40, false);
        // Canonical order, left→right: toolbox | grid | preview | panel — they cover
        // the whole width with no overlap/gap and the PANEL is the far-right column.
        assert_eq!(l.toolbox.x, 0);
        assert_eq!(l.toolbox.w, TOOLBOX_W);
        assert_eq!(
            l.grid.x,
            l.toolbox.x + l.toolbox.w,
            "grid starts past the toolbox"
        );
        assert_eq!(
            l.preview.x,
            l.grid.x + l.grid.w,
            "preview right of the grid"
        );
        assert_eq!(
            l.panel.x,
            l.preview.x + l.preview.w,
            "panel is the far-right column"
        );
        assert_eq!(l.panel.x + l.panel.w, 120, "panel reaches the right edge");
        // Grid fills the body height (fb height minus the one chrome row AND the
        // bottom status row).
        assert_eq!(l.grid.h, 38);
        assert!(!l.preview.is_empty());
    }

    #[test]
    fn split_order_is_toolbox_grid_preview_panel_by_x() {
        // At a roomy size the four body columns are strictly increasing in x:
        // toolbox < grid < preview < panel (the panel is the FAR RIGHT).
        let l = compute(ViewMode::Split, 160, 48, false);
        assert!(!l.toolbox.is_empty() && !l.grid.is_empty());
        assert!(!l.preview.is_empty() && !l.panel.is_empty());
        assert!(
            l.toolbox.x < l.grid.x && l.grid.x < l.preview.x && l.preview.x < l.panel.x,
            "toolbox<grid<preview<panel: tbx={:?} grid={:?} prev={:?} panel={:?}",
            l.toolbox,
            l.grid,
            l.preview,
            l.panel
        );
        // Side-by-side viewports share the same top row and full body height.
        assert_eq!(l.grid.y, l.preview.y, "grid + preview share the top row");
        assert_eq!(
            l.grid.h, l.preview.h,
            "grid + preview share the body height"
        );
    }

    #[test]
    fn swap_flips_grid_and_preview_x_order_in_split() {
        // Un-swapped: grid left of preview. Swapped: preview left of grid. The panel
        // stays the far-right column either way, and the four columns still tile.
        let normal = compute_swapped(ViewMode::Split, 160, 48, false, false);
        let swapped = compute_swapped(ViewMode::Split, 160, 48, false, true);
        // Sanity: both have all four columns.
        for l in [normal, swapped] {
            assert!(!l.toolbox.is_empty() && !l.grid.is_empty());
            assert!(!l.preview.is_empty() && !l.panel.is_empty());
        }
        // Un-swapped x-order: toolbox < grid < preview < panel.
        assert!(
            normal.grid.x < normal.preview.x,
            "un-swapped: grid left of preview"
        );
        // Swapped x-order: the PREVIEW now sits where the grid was; the GRID is right
        // of it. toolbox < preview < grid < panel.
        assert!(
            swapped.preview.x < swapped.grid.x,
            "swapped: preview left of grid (prev={:?} grid={:?})",
            swapped.preview,
            swapped.grid
        );
        assert!(
            swapped.toolbox.x < swapped.preview.x && swapped.grid.x < swapped.panel.x,
            "swapped order toolbox<preview<grid<panel"
        );
        // Swapped still tiles the center with no gap/overlap and reaches the panel.
        assert_eq!(swapped.preview.x, swapped.toolbox.x + swapped.toolbox.w);
        assert_eq!(swapped.grid.x, swapped.preview.x + swapped.preview.w);
        assert_eq!(swapped.panel.x, swapped.grid.x + swapped.grid.w);
        // The panel position is identical whether swapped or not (only the two center
        // viewports trade places).
        assert_eq!(normal.panel, swapped.panel, "panel unaffected by swap");
        assert_eq!(
            normal.toolbox, swapped.toolbox,
            "toolbox unaffected by swap"
        );
    }

    #[test]
    fn swap_is_a_noop_outside_split() {
        // Full2D shows only the grid; Full3D only the preview — swap changes nothing.
        for mode in [ViewMode::Full2D, ViewMode::Full3D] {
            let normal = compute_swapped(mode, 160, 48, false, false);
            let swapped = compute_swapped(mode, 160, 48, false, true);
            assert_eq!(normal, swapped, "{mode:?} ignores swap");
        }
    }

    #[test]
    fn split_narrow_center_stacks_viewports_vertically() {
        // A center too narrow for two side-by-side viewports but tall enough stacks
        // them: grid on top, preview directly below, both full center width.
        // Choose a width whose center is in [MIN_PANE_W, CENTER_STACK_MIN_W).
        let w = MIN_COLS + 4; // small, above the floor
        let h = 30; // tall enough to stack
        let l = compute(ViewMode::Split, w, h, false);
        if !l.preview.is_empty() {
            // When both viewports show at this narrow width, they must be stacked.
            assert_eq!(l.grid.x, l.preview.x, "stacked: same left edge");
            assert_eq!(l.grid.w, l.preview.w, "stacked: same width");
            assert_eq!(
                l.preview.y,
                l.grid.y + l.grid.h,
                "preview stacked below grid"
            );
            assert!(l.grid.w >= MIN_PANE_W, "stacked grid is usable");
        }
        // Either way the grid stays usable.
        assert!(l.grid.w >= MIN_PANE_W);
    }

    #[test]
    fn full2d_hides_preview_and_widens_grid() {
        let split = compute(ViewMode::Split, 120, 40, false);
        let full = compute(ViewMode::Full2D, 120, 40, false);
        assert!(full.preview.is_empty());
        assert!(full.grid.w > split.grid.w, "grid is wider in Full2D");
        assert_eq!(full.toolbox.w + full.grid.w + full.panel.w, 120);
    }

    #[test]
    fn panel_keeps_full_height_no_bottom_strip() {
        // The side panel now hosts the OUTLINER + INSPECTOR organs internally, so it
        // is NOT shortened by a carved bottom strip: the panel fills the body height
        // and both the elevation + tree carve rects are retired (always empty).
        let l = compute(ViewMode::Split, 120, 40, false);
        assert!(
            l.elevation.is_empty(),
            "elevation strip retired from the column"
        );
        assert!(
            l.tree.is_empty(),
            "carved tree strip retired (outliner is in-panel)"
        );
        // Panel reaches the bottom of the body (just above the status row).
        assert_eq!(
            l.panel.y + l.panel.h,
            l.status.y,
            "panel fills the body height"
        );
    }

    #[test]
    fn tree_open_does_not_carve_a_strip() {
        // `tree_open` no longer carves a separate bottom pane — the outliner is the
        // panel's top organ regardless. Both carve rects stay empty either way.
        let open = compute(ViewMode::Split, 120, 40, true);
        let closed = compute(ViewMode::Split, 120, 40, false);
        assert!(open.tree.is_empty() && closed.tree.is_empty());
        assert!(open.elevation.is_empty() && closed.elevation.is_empty());
        assert_eq!(
            open.panel, closed.panel,
            "panel height is tree-open-independent"
        );
    }

    #[test]
    fn full3d_has_no_elevation() {
        let l = compute(ViewMode::Full3D, 120, 40, false);
        assert!(l.elevation.is_empty());
        assert!(l.tree.is_empty());
    }

    #[test]
    fn full3d_hides_grid_and_fills_preview() {
        let l = compute(ViewMode::Full3D, 100, 30, false);
        assert!(l.grid.is_empty());
        assert!(
            l.panel.is_empty(),
            "Full3D has no side panel — readouts go to the status bar"
        );
        // Full3D is a clean walk-through: BOTH the toolbox AND the panel are hidden,
        // so the preview fills the WHOLE body width.
        assert!(
            l.toolbox.is_empty(),
            "Full3D hides the toolbox (walk, not edit)"
        );
        assert_eq!(l.preview.x, 0, "preview fills from the left edge");
        assert_eq!(l.preview.w, 100, "preview spans the full width");
        // Toolbar row 0 (only chrome row), preview from row 1, status bar bottom row.
        assert_eq!(l.toolbar.h, 1);
        assert!(l.tool_bar.is_empty(), "retired row-1 strip");
        assert_eq!(l.preview.y, 1, "preview fills below the one chrome row");
        assert_eq!(
            l.status,
            Rect {
                x: 0,
                y: 29,
                w: 100,
                h: 1
            },
            "status bar bottom row"
        );
        // Preview fills the body between chrome and the status bar (30 - 1 - 1 = 28).
        assert_eq!(l.preview.h, 28);
        assert!(l.preview.y + l.preview.h <= l.status.y);
    }

    #[test]
    fn adapts_to_arbitrary_size_without_overflow() {
        for &(w, h) in &[(40usize, 12usize), (200, 60), (8, 4)] {
            for mode in [ViewMode::Split, ViewMode::Full2D, ViewMode::Full3D] {
                for tree_open in [false, true] {
                    let l = compute(mode, w, h, tree_open);
                    for r in [
                        l.toolbar,
                        l.tool_bar,
                        l.toolbox,
                        l.grid,
                        l.panel,
                        l.preview,
                        l.elevation,
                        l.tree,
                        l.status,
                    ] {
                        assert!(r.x + r.w <= w, "rect {r:?} fits width {w}");
                        assert!(r.y + r.h <= h, "rect {r:?} fits height {h}");
                    }
                }
            }
        }
    }

    #[test]
    fn below_floor_collapses_body_but_keeps_chrome_full_width() {
        // Below the usable floor (cols < MIN_COLS or rows < MIN_ROWS) every body
        // pane is empty so `main` can draw the too-small message; the chrome rows
        // still take the full width when the fb is tall enough to host them.
        for &(w, h) in &[(MIN_COLS - 1, 40usize), (120, MIN_ROWS - 1), (8, 4)] {
            assert!(too_small(w, h), "({w},{h}) is below the floor");
            for mode in [ViewMode::Split, ViewMode::Full2D, ViewMode::Full3D] {
                let l = compute(mode, w, h, false);
                for r in [l.grid, l.panel, l.preview, l.tree, l.elevation, l.toolbox] {
                    assert!(r.is_empty(), "body rect {r:?} empty below the floor");
                }
                // The status bar follows the same fb-height rule as the toolbar: it
                // is present (full width, bottom row) only when the fb is tall enough.
                if !l.status.is_empty() {
                    assert_eq!(l.status.x, 0, "status bar at x=0");
                    assert_eq!(l.status.w, w, "status bar spans full width");
                    assert_eq!(l.status.y, h - 1, "status bar is the bottom row");
                }
                if !l.toolbar.is_empty() {
                    assert_eq!(l.toolbar.x, 0, "toolbar at x=0");
                    assert_eq!(l.toolbar.w, w, "toolbar spans full width");
                }
                if !l.tool_bar.is_empty() {
                    assert_eq!(l.tool_bar.x, 0, "tool palette at x=0");
                    assert_eq!(l.tool_bar.w, w, "tool palette spans full width");
                }
            }
        }
    }

    #[test]
    fn above_floor_grid_is_never_an_unusable_sliver() {
        // Above the floor the grid always gets a usable cell-area width in every
        // mode that shows it; the chrome rows always span the full width.
        for &(w, h) in &[(MIN_COLS, MIN_ROWS), (60, 20), (200, 60)] {
            assert!(!too_small(w, h));
            for mode in [ViewMode::Split, ViewMode::Full2D] {
                for tree_open in [false, true] {
                    let l = compute(mode, w, h, tree_open);
                    assert!(
                        l.grid.w >= MIN_PANE_W,
                        "{mode:?} grid {:?} usable at {w}x{h}",
                        l.grid
                    );
                    // Chrome (when present) spans the full width for the fuller text.
                    if !l.toolbar.is_empty() {
                        assert_eq!(l.toolbar.w, w, "{mode:?} toolbar full width");
                    }
                    if !l.tool_bar.is_empty() {
                        assert_eq!(l.tool_bar.w, w, "{mode:?} tool palette full width");
                    }
                }
            }
        }
    }

    #[test]
    fn split_never_produces_two_slivers_above_the_floor() {
        // Sweep every above-floor width: in Split, the grid is always usable and
        // the preview, when shown at all, is never a sliver. We never split a row
        // into two unusable panes — the preview collapses first.
        for w in MIN_COLS..=120 {
            let h = 20;
            let l = compute(ViewMode::Split, w, h, false);
            assert!(
                l.grid.w >= MIN_PANE_W,
                "grid usable at width {w}: {:?}",
                l.grid
            );
            if !l.preview.is_empty() {
                assert!(
                    l.preview.w >= MIN_PANE_W,
                    "preview not a sliver at width {w}: {:?}",
                    l.preview
                );
            }
        }
    }

    #[test]
    fn toolbox_present_at_a_roomy_size_and_grid_stays_usable() {
        // At a roomy size the toolbox is the fixed-width left column in the EDITING
        // modes (Split / Full2D); the center stays usable to its right. Full3D is a
        // walk-through and hides the toolbox entirely (covered separately).
        for &(w, h) in &[(120usize, 40usize), (200, 60)] {
            assert!(!too_small(w, h));
            for mode in [ViewMode::Split, ViewMode::Full2D] {
                let l = compute(mode, w, h, false);
                assert_eq!(
                    l.toolbox.w, TOOLBOX_W,
                    "{mode:?} toolbox present at {w}x{h}"
                );
                assert_eq!(l.toolbox.x, 0, "{mode:?} toolbox is the left column");
                assert_eq!(
                    l.toolbox.y, l.toolbar.h,
                    "{mode:?} toolbox below the chrome"
                );
                // Toolbox spans the body height (below chrome, above status).
                assert!(
                    l.toolbox.y + l.toolbox.h <= l.status.y,
                    "{mode:?} above status"
                );
                // The grid (center, left of preview/panel) is shifted right past it.
                assert_eq!(l.grid.x, TOOLBOX_W, "{mode:?} grid right of toolbox");
                assert!(l.grid.w >= MIN_PANE_W, "{mode:?} grid usable");
            }
        }
    }

    #[test]
    fn full3d_hides_toolbox_at_a_roomy_size() {
        // Full3D is a clean walk-through: no toolbox, no panel; the preview fills the
        // whole body even when there would have been room for a toolbox.
        for &(w, h) in &[(120usize, 40usize), (200, 60)] {
            let l = compute(ViewMode::Full3D, w, h, false);
            assert!(l.toolbox.is_empty(), "Full3D hides the toolbox at {w}x{h}");
            assert!(l.panel.is_empty(), "Full3D hides the panel at {w}x{h}");
            assert_eq!(l.preview.x, 0, "preview fills from the left at {w}x{h}");
            assert_eq!(l.preview.w, w, "preview spans the full width at {w}x{h}");
        }
    }

    #[test]
    fn toolbox_degrades_12_then_3_then_0_by_width() {
        // The toolbox column width steps down 12 → 3 → 0 as the terminal narrows:
        // the full word column while the center keeps MIN_PANE_W after reserving it
        // plus the panel; the 3-wide icon column when only the narrower strip fits;
        // hidden (0) when even 3 cols would starve the center. Sweep every
        // above-floor width in the editing modes and assert the width matches the
        // SSOT `toolbox_w_for`, and that whenever the toolbox shows the grid stays
        // usable.
        for w in MIN_COLS..=200 {
            for mode in [ViewMode::Split, ViewMode::Full2D] {
                let l = compute(mode, w, 20, false);
                let panel_w = panel_w_for(mode, w);
                let expect = toolbox_w_for(mode, w, panel_w);
                assert_eq!(
                    l.toolbox.w, expect,
                    "{mode:?} toolbox width matches the budget at w={w}"
                );
                assert!(
                    matches!(l.toolbox.w, 0 | TOOLBOX_ICON_W | TOOLBOX_W),
                    "{mode:?} toolbox width is one of 0/3/12 at w={w}: {}",
                    l.toolbox.w
                );
                if !l.toolbox.is_empty() {
                    // When present, the grid was shrunk and stays usable.
                    assert!(
                        l.grid.w >= MIN_PANE_W,
                        "{mode:?} grid usable with toolbox (w={w})"
                    );
                }
            }
        }
    }

    /// The icon-only toolbox actually appears at some narrow-but-above-floor width:
    /// proves the middle rung of the 12→3→0 ladder is reachable, not skipped.
    #[test]
    fn toolbox_icon_mode_is_reachable() {
        let mut saw_icon = false;
        for w in MIN_COLS..=200 {
            for mode in [ViewMode::Split, ViewMode::Full2D] {
                let l = compute(mode, w, 20, false);
                if l.toolbox.w == TOOLBOX_ICON_W {
                    saw_icon = true;
                }
            }
        }
        assert!(
            saw_icon,
            "icon-only toolbox (width 3) is reachable for some width"
        );
    }

    /// The panel SHRINKS toward `PANEL_MIN_W` before collapsing: there is a width band
    /// where the panel is present but narrower than the full PANEL_W.
    #[test]
    fn panel_shrinks_before_collapsing() {
        let mut saw_shrunk = false;
        for w in MIN_COLS..=200 {
            for mode in [ViewMode::Split, ViewMode::Full2D] {
                let pw = panel_w_for(mode, w);
                if pw > 0 {
                    assert!(
                        pw >= PANEL_MIN_W,
                        "{mode:?} non-zero panel >= min at w={w}: {pw}"
                    );
                    if pw < PANEL_W {
                        saw_shrunk = true;
                    }
                }
            }
        }
        assert!(
            saw_shrunk,
            "panel occupies a shrunk (PANEL_MIN_W..PANEL_W) band before collapse"
        );
    }

    // ── THE ROBUSTNESS SWEEP: no overlap / no overflow at every size + mode ──────

    /// `true` when rects `a` and `b` (both non-empty) share at least one cell.
    /// Pure half-open rectangle intersection. Empty rects never overlap anything.
    fn overlaps(a: Rect, b: Rect) -> bool {
        if a.is_empty() || b.is_empty() {
            return false;
        }
        let ax2 = a.x + a.w;
        let ay2 = a.y + a.h;
        let bx2 = b.x + b.w;
        let by2 = b.y + b.h;
        a.x < bx2 && b.x < ax2 && a.y < by2 && b.y < ay2
    }

    /// The widths swept by the robustness test: dense around the floors (every step
    /// near `MIN_COLS` / a few key transition points) plus a coarser tail to ~200.
    fn sweep_widths() -> Vec<usize> {
        let mut v: Vec<usize> = (0..=84).collect(); // 0..84 step 1 (covers 39/40/41)
        v.extend((90..=200).step_by(7)); // coarser tail
        v
    }
    /// The heights swept: every row up to ~30 (covers 11/12/13 around `MIN_ROWS`),
    /// then a coarser tail to ~64.
    fn sweep_heights() -> Vec<usize> {
        let mut v: Vec<usize> = (0..=32).collect();
        v.extend((36..=64).step_by(7));
        v
    }

    // ── ADAPTIVE PANEL MODEL ──────────────────────────────────────────────

    /// Live-editor prefs: all Expanded + adaptive shedding ON.
    fn adaptive() -> PanelPrefs {
        PanelPrefs {
            adaptive: true,
            ..Default::default()
        }
    }

    /// Acceptance: at 240×64 every panel is visible and nothing is railed —
    /// the adaptive layout equals the legacy one on a roomy terminal.
    #[test]
    fn adaptive_240x64_shows_every_panel_no_rail() {
        let p = adaptive();
        let l = compute_with_panels(ViewMode::Split, 240, 64, false, false, p, None);
        assert!(!l.toolbox.is_empty() && !l.grid.is_empty());
        assert!(!l.preview.is_empty() && !l.panel.is_empty());
        assert!(l.rail.is_empty(), "nothing railed at 240 cols");
        assert_eq!(
            effective_panels(ViewMode::Split, 240, p),
            EffPanels::default(),
            "all panels effectively expanded"
        );
        assert_eq!(
            l,
            compute_swapped(ViewMode::Split, 240, 64, false, false),
            "adaptive == legacy above every threshold"
        );
    }

    /// Acceptance: at 100×30 the TOOLBOX + OUTLINER shed to the rail and the
    /// INSPECTOR survives (shedding priority: inspector > outliner > toolbox).
    #[test]
    fn adaptive_100x30_rails_toolbox_and_outliner_keeps_inspector() {
        let p = adaptive();
        let eff = effective_panels(ViewMode::Split, 100, p);
        assert_eq!(eff.toolbox, PanelState::Rail, "toolbox sheds first");
        assert_eq!(eff.outliner, PanelState::Rail, "outliner sheds next");
        assert_eq!(eff.inspector, PanelState::Expanded, "inspector survives");
        let l = compute_with_panels(ViewMode::Split, 100, 30, false, false, p, None);
        assert!(l.toolbox.is_empty(), "no toolbox column in the rail band");
        assert!(
            !l.panel.is_empty(),
            "the right panel (inspector organ) survives"
        );
        assert!(!l.grid.is_empty(), "the grid never sheds");
        assert_eq!(l.rail.w, RAIL_W, "the icon rail is carved");
        assert_eq!(l.rail.x, 100 - RAIL_W, "rail on the right edge");
        assert!(
            l.panel.x + l.panel.w <= l.rail.x,
            "panel never overlaps the rail"
        );
    }

    /// Acceptance: at 80×24 the layout is SINGLE-PANE — grid + toolbar +
    /// status + rail only; nothing panics, nothing overlaps.
    #[test]
    fn adaptive_80x24_single_pane_plus_rail() {
        let p = adaptive();
        let l = compute_with_panels(ViewMode::Split, 80, 24, false, false, p, None);
        assert!(!l.grid.is_empty(), "grid survives (highest priority)");
        assert!(
            l.preview.is_empty() && l.panel.is_empty() && l.toolbox.is_empty(),
            "single-pane: every other body pane shed"
        );
        assert!(!l.rail.is_empty(), "rail shows the collapsed panels' icons");
        assert!(!l.toolbar.is_empty() && !l.status.is_empty(), "chrome kept");
        assert_eq!(l.grid.x + l.grid.w, l.rail.x, "grid tiles up to the rail");
    }

    /// `z` ZOOM: the zoomed pane fullscreens the body (toolbar/status kept);
    /// zoom None restores the exact pre-zoom layout.
    #[test]
    fn zoom_fullscreens_one_pane_and_none_restores() {
        let p = adaptive();
        let z = compute_with_panels(
            ViewMode::Split,
            120,
            40,
            false,
            false,
            p,
            Some(PanelId::Grid),
        );
        assert_eq!(
            z.grid,
            Rect {
                x: 0,
                y: 1,
                w: 120,
                h: 38
            },
            "grid owns the whole body"
        );
        assert!(
            z.preview.is_empty() && z.panel.is_empty() && z.toolbox.is_empty() && z.rail.is_empty()
        );
        assert!(
            !z.toolbar.is_empty() && !z.status.is_empty(),
            "chrome survives zoom"
        );
        let zp = compute_with_panels(
            ViewMode::Split,
            120,
            40,
            false,
            false,
            p,
            Some(PanelId::Preview),
        );
        assert_eq!(
            zp.preview,
            Rect {
                x: 0,
                y: 1,
                w: 120,
                h: 38
            },
            "preview zoom too"
        );
        assert!(zp.grid.is_empty());
        let restored = compute_with_panels(ViewMode::Split, 120, 40, false, false, p, None);
        assert_eq!(
            restored,
            compute_swapped(ViewMode::Split, 120, 40, false, false),
            "un-zoomed adaptive layout at 120 equals the legacy layout"
        );
    }

    /// ACCORDION: with the outliner focused (`tree_open`) the right panel takes
    /// a wider cap than the unfocused layout.
    #[test]
    fn accordion_widens_focused_right_panel() {
        let p = adaptive();
        let normal = compute_with_panels(ViewMode::Split, 200, 48, false, false, p, None);
        let focused = compute_with_panels(ViewMode::Split, 200, 48, true, false, p, None);
        assert!(
            focused.panel.w > normal.panel.w,
            "focused outliner widens the panel ({} > {})",
            focused.panel.w,
            normal.panel.w
        );
    }

    /// A USER pref rails a panel at ANY width (session collapse), and the
    /// sibling organ takes the whole column.
    #[test]
    fn user_pref_rail_collapses_at_any_width() {
        let p = PanelPrefs {
            outliner: PanelState::Rail,
            adaptive: true,
            ..Default::default()
        };
        let eff = effective_panels(ViewMode::Split, 240, p);
        assert_eq!(
            eff.outliner,
            PanelState::Rail,
            "user collapse wins at 240 cols"
        );
        assert_eq!(eff.inspector, PanelState::Expanded);
        let l = compute_with_panels(ViewMode::Split, 240, 64, false, false, p, None);
        assert!(
            !l.rail.is_empty(),
            "the rail appears for the user-collapsed organ"
        );
    }

    /// When the user rails the INSPECTOR pref, the narrow band keeps the
    /// OUTLINER instead — the rail-icon organ swap is reachable.
    #[test]
    fn narrow_band_respects_user_organ_choice() {
        let p = PanelPrefs {
            inspector: PanelState::Rail,
            adaptive: true,
            ..Default::default()
        };
        let eff = effective_panels(ViewMode::Split, 100, p);
        assert_eq!(
            eff.outliner,
            PanelState::Expanded,
            "outliner keeps the column"
        );
        assert_eq!(eff.inspector, PanelState::Rail);
    }

    /// Full3D hides every side panel outright — no rail, no icons (a
    /// walk-through has no chrome), exactly as before.
    #[test]
    fn full3d_has_no_rail_or_panels_under_adaptive() {
        let p = adaptive();
        let l = compute_with_panels(ViewMode::Full3D, 100, 30, false, false, p, None);
        assert!(l.rail.is_empty() && l.panel.is_empty() && l.toolbox.is_empty());
        assert_eq!(l.preview.w, 100, "preview fills the body");
    }

    /// THE ADAPTIVE ROBUSTNESS SWEEP: every size × mode × zoom never overflows,
    /// never overlaps, never panics — the degradation ladder holds everywhere
    /// (acceptance "nothing panics" at any shrink).
    #[test]
    fn adaptive_sweep_no_overlap_no_overflow() {
        let p = adaptive();
        for &w in &sweep_widths() {
            for &h in &sweep_heights() {
                for mode in [ViewMode::Split, ViewMode::Full2D, ViewMode::Full3D] {
                    for zoomed in [None, Some(PanelId::Grid), Some(PanelId::Preview)] {
                        let l = compute_with_panels(mode, w, h, false, false, p, zoomed);
                        let rects = [
                            ("toolbar", l.toolbar),
                            ("toolbox", l.toolbox),
                            ("grid", l.grid),
                            ("preview", l.preview),
                            ("panel", l.panel),
                            ("status", l.status),
                            ("rail", l.rail),
                        ];
                        for (name, r) in rects {
                            if !r.is_empty() {
                                assert!(
                                    r.x + r.w <= w && r.y + r.h <= h,
                                    "{name} {r:?} overflows {w}x{h} ({mode:?}, zoom {zoomed:?})"
                                );
                            }
                        }
                        for i in 0..rects.len() {
                            for j in (i + 1)..rects.len() {
                                let (na, ra) = rects[i];
                                let (nb, rb) = rects[j];
                                assert!(
                                    !overlaps(ra, rb),
                                    "{na} {ra:?} overlaps {nb} {rb:?} at {w}x{h} ({mode:?}, zoom {zoomed:?})"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn sweep_no_overlap_no_overflow_all_modes() {
        for &w in &sweep_widths() {
            for &h in &sweep_heights() {
                for mode in [ViewMode::Split, ViewMode::Full2D, ViewMode::Full3D] {
                    for tree_open in [false, true] {
                        for swap in [false, true] {
                            let l = compute_swapped(mode, w, h, tree_open, swap);
                            // The named body + chrome rects under test (the retired
                            // tree/tool_bar/elevation are asserted empty separately).
                            let rects = [
                                ("toolbar", l.toolbar),
                                ("toolbox", l.toolbox),
                                ("grid", l.grid),
                                ("preview", l.preview),
                                ("panel", l.panel),
                                ("status", l.status),
                            ];
                            // Retired rects are always zero-size.
                            assert!(l.tool_bar.is_empty(), "tool_bar retired {mode:?} {w}x{h}");
                            assert!(l.tree.is_empty(), "tree retired {mode:?} {w}x{h}");
                            assert!(l.elevation.is_empty(), "elevation retired {mode:?} {w}x{h}");

                            // 1. Every non-empty rect is fully inside the framebuffer.
                            for (name, r) in rects {
                                if r.is_empty() {
                                    continue;
                                }
                                assert!(
                                r.x + r.w <= w && r.y + r.h <= h,
                                "{name} {r:?} overflows {w}x{h} ({mode:?}, tree_open={tree_open})"
                            );
                            }
                            // 2. No two non-empty rects overlap (pairwise).
                            for i in 0..rects.len() {
                                for j in (i + 1)..rects.len() {
                                    let (na, ra) = rects[i];
                                    let (nb, rb) = rects[j];
                                    assert!(
                                        !overlaps(ra, rb),
                                        "{na} {ra:?} overlaps {nb} {rb:?} at {w}x{h} \
                                     ({mode:?}, tree_open={tree_open})"
                                    );
                                }
                            }
                            // 3/4. Below the floor: all body rects empty. Above the floor:
                            // the status row is present and clear of every body rect.
                            if too_small(w, h) {
                                for (name, r) in [
                                    ("toolbox", l.toolbox),
                                    ("grid", l.grid),
                                    ("preview", l.preview),
                                    ("panel", l.panel),
                                ] {
                                    assert!(
                                    r.is_empty(),
                                    "{name} {r:?} must be empty below the floor at {w}x{h} ({mode:?})"
                                );
                                }
                            } else {
                                assert!(
                                !l.status.is_empty(),
                                "status row must be present above the floor at {w}x{h} ({mode:?})"
                            );
                                for (name, r) in [
                                    ("toolbar", l.toolbar),
                                    ("toolbox", l.toolbox),
                                    ("grid", l.grid),
                                    ("preview", l.preview),
                                    ("panel", l.panel),
                                ] {
                                    if !r.is_empty() {
                                        assert!(
                                        !overlaps(r, l.status),
                                        "{name} {r:?} overlaps the status row {:?} at {w}x{h} ({mode:?}, swap={swap})",
                                        l.status
                                    );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
