//! Mouse → editor-state transitions. Pure (no terminal, no ECS in
//! [`apply_mouse`]) so the pane hit-testing + grid inversion are unit-tested
//! headlessly; the thin Bevy [`mouse_system`] just feeds it the live
//! framebuffer size and drains the [`EditorMouse`] channel.
//!
//! Routing by pane (using the SAME `layout::compute` the renderer uses, with the
//! live framebuffer size):
//!
//! * **Top-down grid pane**
//!   - LEFT `Press` → start a PAINT stroke: move `cursor` to the cell (inverting
//!     `chrome::draw_grid` via [`view::screen_to_cell`]), take ONE undo snapshot,
//!     and paint ([`edit::apply_tool_at_cursor`]). LEFT `Drag` → extend the stroke
//!     with a gap-free Bresenham line ([`edit::paint_tool_at_cursor`], NO snapshot)
//!     so a fast drag is one continuous stroke and ONE undo step. `Release` ends it.
//!   - RIGHT `Press`/`Drag` → the same stroke flow but ERASING each cell to
//!     `Empty`/`Shape::Full` (one snapshot for the whole erase).
//!   - MIDDLE `Drag` → PAN the view by the world-cell delta since the press anchor
//!     (dragging the map moves the view), clamped to the world.
//!   - `ScrollUp`/`ScrollDown` → ZOOM anchored at the cell under the pointer
//!     ([`view::zoom_in_at`]/[`view::zoom_out_at`]). z-paging moved to PageUp/PageDown,
//!     the Z-slider, and `<`/`>`.
//! * **3D preview pane** — OWNS the mouse for camera. Look is EXCLUSIVELY the
//!   RIGHT-button-HELD path: a right press starts capture, and the only thing
//!   that ends it is right-button RELEASE (`capture == "RMB is held"`, see the
//!   guard in [`apply_mouse`]) — it is hold-to-look, NOT a toggle. A LEFT press
//!   or drag inside the preview is INERT (no drag-look); the wheel dollies the
//!   camera forward/back. Classified up-front by [`pane_at`] so toolbar /
//!   palette / panel / grid hit-tests never intercept a look.
//! * **Panel / elevation / outside any pane** — ignored.
use crate::chrome::{RULER_H, RULER_W, Z_SLIDER_ROW, Z_SLIDER_W};
use crate::edit;
use crate::layout;
use crate::open;
use crate::palette_bar;
use crate::state::{EditorState, ViewMode};
use crate::toolbar::{self, ToolbarButton};
use crate::view;
use bevy::prelude::*;
use re_core::Framebuffer;
use re_present::{EditorKey, EditorMouse, EditorMouseButton, EditorMouseKind};

/// Lines one WHEEL notch scrolls the open `?` help sheet. Three is the usual
/// terminal convention and keeps a notch readable (a full page per notch loses
/// the reader's place); the offset is clamped by `edit::help_scroll_by`.
const HELP_WHEEL_LINES: isize = 3;

/// Which editor pane the pointer is currently over. Derived purely from the
/// layout rects so it can be unit-tested headlessly for every [`ViewMode`].
///
/// Used (at least) to gate the PREVIEW camera-look routing: in Split/Full3D the
/// preview pane OWNS the mouse for free-look, so toolbar/toolbox/panel/grid
/// hit-tests must never steal a click meant for the camera. The chrome toolbar
/// row takes priority over the body panes it overlaps (row 0), since it is drawn
/// over the top of the panes; the [`crate::layout::PanelId::Toolbox`] is a
/// disjoint left body column.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Pane {
    /// The clickable top toolbar row (mode / Open / Save / Help).
    Toolbar,
    /// The vertical left TOOLBOX column (grouped tool buttons).
    Toolbox,
    /// The 3D preview pane — owns the mouse for camera free-look.
    Preview,
    /// The top-down edit grid.
    Grid,
    /// The side panel (chrome / inspector / buttons / z-slider).
    Panel,
    /// The object-tree / outliner side-window.
    Tree,
    /// The 1-col ICON RAIL down the right edge — collapsed panels'
    /// icons; a click reopens the panel.
    Rail,
    /// Not over any interactive pane.
    None,
}

/// CLICK-TO-FOCUS side-effect: a LEFT press inside a center viewport moves the
/// focus cue (and the `z` zoom / swap target) to it. Never consumes the event.
fn click_to_focus(
    state: &mut EditorState,
    ev: EditorMouse,
    l: &layout::Layout,
    col: usize,
    row: usize,
) {
    if ev.kind == EditorMouseKind::Press && ev.button == Some(EditorMouseButton::Left) {
        match pane_at(l, col, row) {
            Pane::Grid => state.focused_viewport = crate::state::ViewportFocus::Grid,
            Pane::Preview => state.focused_viewport = crate::state::ViewportFocus::Preview,
            _ => {}
        }
    }
}

/// The OVERLAY mouse routing, checked BEFORE the classic pane routing.
/// Returns `true` when the event was consumed:
///
/// * **Command palette** (topmost, modal): a LEFT press on a match row selects
///   and RUNS it; every other click is consumed so nothing paints under the
///   overlay.
/// * **Validation drawer**: clicks INSIDE the open drawer belong to it (`[X]`
///   closes; rows are read-only v1) — clicks outside fall through so editing
///   keeps working beside it.
/// * **Icon rail**: a press on a rail icon reopens that panel
///   ([`rail_reopen`]).
fn w3_overlay_mouse(
    state: &mut EditorState,
    ev: EditorMouse,
    l: &layout::Layout,
    fb_w: usize,
    fb_h: usize,
    col: usize,
    row: usize,
) -> bool {
    if state.cmd_palette.is_some() {
        if ev.kind == EditorMouseKind::Press && ev.button == Some(EditorMouseButton::Left) {
            if let Some(idx) = crate::chrome::cmd_palette_row_at(state, fb_w, fb_h, col, row) {
                if let Some(p) = state.cmd_palette.as_mut() {
                    p.sel = idx;
                }
                crate::keymap::execute_selected(state);
            }
        }
        return true;
    }
    if state.validation_open && crate::validate::hit(fb_w, fb_h, col, row) {
        if ev.kind == EditorMouseKind::Press {
            if let Some((brow, bx0, bx1)) = crate::validate::close_button(fb_w, fb_h) {
                if row == brow && col >= bx0 && col < bx1 {
                    crate::validate::toggle(state);
                }
            }
        }
        return true;
    }
    if ev.kind == EditorMouseKind::Press && pane_at(l, col, row) == Pane::Rail {
        let eff = state.panels_eff(fb_w);
        if let Some(pid) = crate::chrome::rail_hit(l.rail, eff, col, row) {
            rail_reopen(state, pid, fb_w);
        }
        return true;
    }
    false
}

/// Reopen a RAIL-collapsed panel from its icon. Sets the user pref back
/// to Expanded; when the width band would immediately re-shed the clicked
/// right-panel organ, its SIBLING rails instead (the accordion organ swap), so
/// the click always does something visible where the budget allows one organ.
/// A panel the band cannot fit at all reports why (honest degradation).
fn rail_reopen(state: &mut EditorState, pid: layout::PanelId, fb_w: usize) {
    use layout::{PanelId, PanelState};
    match pid {
        PanelId::Toolbox => state.panels.toolbox = PanelState::Expanded,
        PanelId::Outliner => {
            state.panels.outliner = PanelState::Expanded;
            if state.panels_eff(fb_w).outliner != PanelState::Expanded {
                // The band keeps ONE organ — hand the column to the outliner.
                state.panels.inspector = PanelState::Rail;
            }
        }
        PanelId::Inspector => {
            state.panels.inspector = PanelState::Expanded;
            if state.panels_eff(fb_w).inspector != PanelState::Expanded {
                state.panels.outliner = PanelState::Rail;
            }
        }
        _ => return,
    }
    let eff = state.panels_eff(fb_w);
    let (name, now) = match pid {
        PanelId::Toolbox => ("toolbox", eff.toolbox),
        PanelId::Outliner => ("outliner", eff.outliner),
        PanelId::Inspector => ("inspector", eff.inspector),
        _ => return,
    };
    state.status = if now == PanelState::Expanded {
        format!("{name} expanded")
    } else {
        format!("too narrow for the {name} — widen the terminal or zoom (z)")
    };
}

/// Return `true` when `(col,row)` lies inside `r` (and `r` is non-empty).
fn in_rect(r: layout::Rect, col: usize, row: usize) -> bool {
    !r.is_empty() && col >= r.x && col < r.x + r.w && row >= r.y && row < r.y + r.h
}

/// The 2D grid SCENE rect: the grid pane minus its 1-row caption header at the top
/// (mirrors `draw_system`, which calls `chrome::draw_grid` one row lower). THE
/// single source of truth shared by the grid hit-test + pan so a click inverts to
/// the cell drawn under the pointer. An empty grid rect stays empty.
pub fn grid_scene_rect(grid: layout::Rect) -> layout::Rect {
    if grid.is_empty() {
        return grid;
    }
    let hh = crate::chrome::VIEWPORT_HEADER_H.min(grid.h);
    layout::Rect {
        x: grid.x,
        y: grid.y + hh,
        w: grid.w,
        h: grid.h - hh,
    }
}

/// Classify which [`Pane`] the pointer `(col,row)` is over, given a computed
/// [`layout::Layout`]. Pure — no state, no terminal — so it is exhaustively
/// unit-tested per view mode. The chrome toolbar row wins over the body panes it
/// is drawn on top of; otherwise body panes (toolbox / grid / panel / preview /
/// tree) are disjoint.
pub fn pane_at(l: &layout::Layout, col: usize, row: usize) -> Pane {
    // Chrome first: the toolbar is painted OVER the body panes (row 0), so a click
    // on that row belongs to the chrome even though a body rect may also cover it.
    if in_rect(l.toolbar, col, row) {
        return Pane::Toolbar;
    }
    // The vertical TOOLBOX is the left body column — disjoint from grid/panel/
    // preview, so it is checked before them.
    if in_rect(l.toolbox, col, row) {
        return Pane::Toolbox;
    }
    if in_rect(l.preview, col, row) {
        return Pane::Preview;
    }
    if in_rect(l.grid, col, row) {
        return Pane::Grid;
    }
    if in_rect(l.tree, col, row) {
        return Pane::Tree;
    }
    if in_rect(l.panel, col, row) {
        return Pane::Panel;
    }
    if in_rect(l.rail, col, row) {
        return Pane::Rail;
    }
    Pane::None
}

/// Apply one mouse event to the editor state given the live framebuffer size.
/// Pure; returns nothing — mutates `state` in place exactly as the keyboard
/// equivalents would.
///
/// Routing order: capture-modal + centered modals first; then a PANE CHECK
/// ([`pane_at`]) routes a pointer inside the PREVIEW pane straight to camera
/// free-look (Split/Full3D — the buttons can't intercept a look); then the
/// TOOLBAR row, tool palette, Z-slider, panel buttons, object-tree, and finally
/// the grid pane (left=paint stroke, right=erase stroke, middle=pan, wheel=zoom).
/// Anything else is inert.
pub fn apply_mouse(state: &mut EditorState, ev: EditorMouse, fb_w: usize, fb_h: usize) {
    // The SAME panel-aware layout `draw_system` renders from, so every
    // hit-test below lands on what is actually drawn.
    let l = state.layout_for(fb_w, fb_h);
    let (col, row) = (ev.col as usize, ev.row as usize);

    // --- OVERLAYS: command palette (modal), validation drawer (in-box
    // clicks), and the icon rail — handled in one helper so each hit resolves
    // against the same geometry its painter uses. Consumed ⇒ done. ---
    if w3_overlay_mouse(state, ev, &l, fb_w, fb_h, col, row) {
        return;
    }

    // --- PREVIEW CAPTURE: while RIGHT-mouse free-look is active the editor is MODAL
    // — clicks/drags must not paint the grid or hit the toolbar. The look itself is
    // driven by `main::preview_capture_look_system` (OS grab delta OR the crossterm
    // cell fallback) — NOT here. The ONLY thing that ends capture is RIGHT-button
    // RELEASE (capture == "RMB is held"); everything else is swallowed. The pointer
    // may have dragged OUTSIDE the preview pane by release time, so the RMB-up check
    // is position-independent. ---
    if state.preview_captured {
        if ev.kind == EditorMouseKind::Release && ev.button == Some(EditorMouseButton::Right) {
            state.capture_request = Some(false);
        }
        return;
    }

    // --- MODAL CLICK GUARD. When any centered modal overlay is open, a
    // click must route to that modal (or be consumed) and NEVER fall through to
    // grid paint underneath. Handle the modal here and return. ---
    // `state.settings.is_some()` and `state.mask_mode` must both be checked here even
    // though `any_modal_active()` includes them — a left-click over the open Settings
    // modal or during part mask editing fell through to grid paint. Both are now
    // guarded so the click is consumed (and, for Settings, routed to its footer).
    // `state.autosave_offer` must be checked here too, and it is the
    // one whose absence LOSES DATA. The offer dispatches at the FRONT of the key
    // chain but the mouse fell straight through it to the toolbar, where
    // `[Save]` runs the save path — which writes the un-recovered document over
    // the original AND calls `io::delete_autosave`, destroying the sidecar. One
    // click on a toolbar the prompt did not even cover and the recovered work was
    // gone. It is guarded FIRST because it dispatches first.
    if state.autosave_offer.is_some()
        || state.help_open
        || state.num_edit.is_some()
        || state.enum_pick.is_some()
        || state.color_edit.is_some()
        || state.dir_picker.is_some()
        || state.wizard.is_some()
        || state.open.is_some()
        || state.saveas.is_some()
        || state.palette.is_some()
        || state.settings.is_some()
        || state.mask_mode
        || state.rename.is_some()
    {
        // (The F2 rename prompt has no clickable controls — its clicks are
        // simply consumed here so they can never paint the grid underneath.)
        apply_modal_click(state, ev, fb_w, fb_h);
        return;
    }

    // --- SWAP CONTROL: the `[<>]` button on the LEFT center viewport's header row
    // toggles `swap_viewports` (flips the 2D/3D sides in Split). It sits on the
    // header row of whichever viewport is physically on the left, so it must be
    // hit-tested BEFORE the preview pane short-circuit (the preview owns the rest of
    // its rect for look) and before the grid paint. Geometry is the SSOT
    // `chrome::viewport_swap_span`, shared with the painter. Only a LEFT press acts. ---
    if ev.kind == EditorMouseKind::Press && ev.button == Some(EditorMouseButton::Left) {
        let both = !l.grid.is_empty() && !l.preview.is_empty();
        if both {
            let left_vp = if l.grid.x <= l.preview.x {
                l.grid
            } else {
                l.preview
            };
            if row == left_vp.y {
                if let Some((s, e)) = crate::chrome::viewport_swap_span(left_vp) {
                    if col >= s && col < e {
                        state.swap_viewports = !state.swap_viewports;
                        state.status = if state.swap_viewports {
                            "viewports swapped (3D | 2D)".into()
                        } else {
                            "viewports normal (2D | 3D)".into()
                        };
                        return;
                    }
                }
            }
        }
    }

    // --- CLICK-TO-FOCUS: a LEFT press inside a CENTER viewport sets the focus cue to
    // it (a VISUAL cue + the swap target only — key routing is UNCHANGED). This is an
    // ADDITIONAL side-effect of the click: it does NOT consume the event, so the grid
    // press still starts a paint stroke (below) and a preview press still follows the
    // RMB-capture contract (below). Done before those paths so both still run. ---
    click_to_focus(state, ev, &l, col, row);

    // --- PANE ROUTING: the PREVIEW pane OWNS the mouse for camera free-look. In
    // Split (right pane) and Full3D (whole view) a click/drag whose (col,row) lands
    // in the preview rect must route to the camera and RETURN *before* any toolbar /
    // palette / panel / grid hit-test, so the buttons can never intercept a look.
    //
    // `pane_at` classifies the pointer: the toolbar (row 0) and tool palette (row 1)
    // are drawn OVER the panes and win on their own rows, so they stay clickable even
    // above the preview; everywhere else inside the preview rect is camera-look. The
    // chrome/panel/grid hit-tests below are unchanged — they own their own rects. ---
    // Toolbar / Palette / Grid / Panel / Tree / None fall through to the existing
    // per-rect hit-tests below (unchanged) so 2D editing + buttons keep working;
    // only Preview is short-circuited here so the camera owns the pointer.
    if pane_at(&l, col, row) == Pane::Preview {
        // PREVIEW CONTRACT: free-look is "RIGHT mouse button HELD". A RIGHT press in
        // the preview requests capture ON; `preview_capture_look_system` then drives
        // the look every frame (OS-grab delta with a crossterm cell fallback that
        // works on Windows Terminal). The matching RIGHT release is handled in the
        // captured branch above (it may land outside the pane). Wheel = dolly.
        match ev.kind {
            // Right Press → request OS free-look capture ON (RMB-held look).
            EditorMouseKind::Press if ev.button == Some(EditorMouseButton::Right) => {
                state.capture_request = Some(true);
            }
            // Wheel over the 3D view = dolly forward/back (one fixed fly step).
            EditorMouseKind::ScrollUp => {
                crate::camera::fly(state, crate::camera::Fly::Forward, 0.0)
            }
            EditorMouseKind::ScrollDown => crate::camera::fly(state, crate::camera::Fly::Back, 0.0),
            // Left press/drag and everything else in the preview is inert now (no
            // drag-look) — look is exclusively the RMB-held path.
            _ => {}
        }
        return;
    }

    // --- TOOLBAR: a Press on the toolbar row routes to a button (before the grid
    // check so toolbar clicks aren't swallowed). ---
    let tb = l.toolbar;
    if !tb.is_empty()
        && ev.kind == EditorMouseKind::Press
        && row == tb.y
        && col >= tb.x
        && col < tb.x + tb.w
    {
        if let Some(btn) = toolbar::toolbar_hit(col - tb.x, tb.w) {
            apply_toolbar(state, btn);
        }
        return;
    }

    // --- TOOLBOX: a Press in the vertical left toolbox column sets the active
    // tool by synthesizing the SAME key the keyboard uses (inherits modal guards +
    // undo). Header/gap rows resolve to None and are consumed without effect. ---
    if ev.kind == EditorMouseKind::Press && pane_at(&l, col, row) == Pane::Toolbox {
        if let Some(btn) = palette_bar::toolbox_hit(&l, col, row) {
            edit::apply_key(
                state,
                EditorKey {
                    code: re_present::EditorKeyCode::Char(btn.key_char()),
                    shift: false,
                },
            );
        }
        return;
    }

    // --- SIDE PANEL ORGANS: the panel hosts the OUTLINER (top) over the INSPECTOR
    // (bottom). Resolve the SAME sub-rects the painters use so a click lands on the
    // organ drawn under it. ---
    let eff = state.panels_eff(fb_w);
    let outliner = crate::chrome::outliner_subrect_eff(state, l.panel, eff);
    let inspector = crate::chrome::inspector_subrect_eff(state, l.panel, eff);

    // --- OUTLINER CONTROL ROWS: `[x] Show objects` toggle (row 1) + `[+ Add object]`
    // (row 2). The two controls live on SEPARATE rows now, so each hit-test keys off
    // its own row to stay in sync with the painter. ---
    if ev.kind == EditorMouseKind::Press && ev.button == Some(EditorMouseButton::Left) {
        if let Some(((cs, ce, crow), (as_, ae, arow))) = crate::chrome::outliner_controls(outliner)
        {
            if row == crow && col >= cs && col < ce {
                // Toggle object visibility (same as the `H` key).
                edit::apply_key(
                    state,
                    EditorKey {
                        code: re_present::EditorKeyCode::Char('H'),
                        shift: false,
                    },
                );
                return;
            }
            if row == arow && col >= as_ && col < ae {
                // Open the create-object palette (the `[+ Add object]` affordance).
                crate::tree::toggle_palette(state);
                return;
            }
        }
    }

    // --- Z-SLIDER: a Press on the slider track (in the INSPECTOR) sets the z-layer.
    if ev.kind == EditorMouseKind::Press && try_z_slider_click(state, inspector, col, row) {
        return;
    }

    // --- INSPECTOR ACTION BUTTONS (P2/P3/P4/P6): a Press on a button runs its
    // action (mostly by synthesizing the matching key, some direct backend). ---
    if ev.kind == EditorMouseKind::Press && ev.button == Some(EditorMouseButton::Left) {
        if let Some(btn) = crate::chrome::panel_button_hit(state, inspector, col, row) {
            apply_panel_button(state, btn);
            return;
        }
    }

    // --- INSPECTOR PROPERTY-SHEET ROWS: a Press on a field row FOCUSES
    // that field (the same focus `;`/`'` page through). The hit-test resolves
    // through the SAME `inspector_rows` + scroll math the painter uses, so the
    // click lands on exactly the row drawn under the pointer. Buttons live BELOW
    // the viewport, so the two hit-tests never overlap. ---
    if ev.kind == EditorMouseKind::Press && ev.button == Some(EditorMouseButton::Left) {
        if let Some((action, zone)) = crate::chrome::inspector_row_hit(state, inspector, col, row) {
            apply_row_click(state, action, zone);
            return;
        }
    }

    // --- OUTLINER OBJECT ROWS: a Press on a tree row sets the cursor + selects
    // it (group headers fold; objects select+jump — via `tree::select_current`). The
    // outliner is ALWAYS visible now, so this no longer gates on `tree.open`. ---
    if ev.kind == EditorMouseKind::Press
        && ev.button == Some(EditorMouseButton::Left)
        && !outliner.is_empty()
    {
        if let Some(ri) = crate::chrome::tree_row_at(state, outliner, col, row) {
            state.tree.cursor = ri;
            crate::tree::select_current(state);
            return;
        }
    }

    // (The 3D PREVIEW pane is routed up-front by the `pane_at` match near the top of
    // this function — it owns the mouse for camera free-look in Split/Full3D, so by
    // here the pointer is NOT over the preview.)

    // The 2D viewport reserves a 1-row caption header at its top (see
    // `chrome::draw_grid_header` + `draw_system`); the grid SCENE — and therefore
    // the clickable cells — begins one row LOWER. Invert against the SAME scene rect
    // the painter draws into so a click lands on the cell under the pointer; a click
    // on the header row resolves to no cell (inert).
    let g = grid_scene_rect(l.grid);
    let in_grid = !g.is_empty() && col >= g.x && col < g.x + g.w && row >= g.y && row < g.y + g.h;

    if !in_grid {
        // Panel/elevation/tree hits are inert (the slider above is the exception).
        return;
    }

    // Map an absolute framebuffer cell to the world cell under it (the grid
    // inversion), or `None` in a ruler gutter / past the world.
    let cell_at = |state: &EditorState, col: usize, row: usize| -> Option<(usize, usize)> {
        let (sx, sy, _sz) = state.map.size();
        view::screen_to_cell(
            col,
            row,
            g.x,
            g.y,
            g.w,
            g.h,
            RULER_W,
            RULER_H,
            state.pan,
            state.zoom,
            (sx, sy),
        )
    };

    match ev.kind {
        EditorMouseKind::Press => match ev.button {
            // LEFT PRESS → apply the active tool at the cell. For STROKE tools
            // (geometry/shape brush, material paint) this also STARTS a paint
            // stroke: one undo snapshot for the whole drag, then the stroke is
            // armed so following drags extend it under that single snapshot.
            // PLACEMENT tools (object placement, `.vox` stamp, staircase stamp)
            // fire ON PRESS ONLY — the stroke is NOT armed, so a drag can never
            // spawn one object per Bresenham cell (the drag-spawn bug).
            Some(EditorMouseButton::Left) => {
                if let Some((wx, wy)) = cell_at(state, col, row) {
                    // Decide BEFORE applying, so the answer describes the tool
                    // that actually fires at this cell.
                    let stroke_tool = edit::tool_paints_on_drag(state);
                    state.cursor = (wx, wy);
                    edit::apply_tool_at_cursor(state); // snapshot + paint/place
                    if stroke_tool {
                        state.stroke_active = true;
                        state.stroke_last_cell = Some((wx, wy, state.z));
                    }
                }
            }
            // RIGHT PRESS → start an ERASE stroke: ONE snapshot, erase the cell to
            // Empty/Full, arm the stroke for Bresenham erase on drag.
            Some(EditorMouseButton::Right) => {
                if let Some((wx, wy)) = cell_at(state, col, row) {
                    state.cursor = (wx, wy);
                    state.push_undo(); // one snapshot for the whole erase stroke
                    erase_cell(state, wx, wy);
                    state.stroke_active = true;
                    state.stroke_last_cell = Some((wx, wy, state.z));
                }
            }
            // MIDDLE PRESS → anchor a pan drag (no edit).
            Some(EditorMouseButton::Middle) => {
                state.last_pan_ptr = Some((ev.col, ev.row));
            }
            _ => {}
        },
        EditorMouseKind::Drag => match ev.button {
            // LEFT DRAG → extend the paint stroke with a gap-free Bresenham line from
            // the last painted cell to the current one (NO snapshot — the press took
            // it), painting each intermediate cell.
            Some(EditorMouseButton::Left) if state.stroke_active => {
                if let Some((wx, wy)) = cell_at(state, col, row) {
                    stroke_to(state, wx, wy, false);
                }
            }
            // RIGHT DRAG → extend the erase stroke the same way.
            Some(EditorMouseButton::Right) if state.stroke_active => {
                if let Some((wx, wy)) = cell_at(state, col, row) {
                    stroke_to(state, wx, wy, true);
                }
            }
            // MIDDLE DRAG → PAN the view by the world-cell delta since the anchor
            // (dragging the map moves the view the opposite way), clamped to the world.
            Some(EditorMouseButton::Middle) => pan_drag(state, &l, ev),
            _ => {}
        },
        // Wheel over the grid ZOOMS, anchored at the cell under the pointer (z-paging
        // moved to PageUp/PageDown, the Z-slider, and `<`/`>`).
        EditorMouseKind::ScrollUp | EditorMouseKind::ScrollDown => {
            let anchor = cell_at(state, col, row).unwrap_or(state.cursor);
            let (sx, sy, _sz) = state.map.size();
            let (nz, np) = if ev.kind == EditorMouseKind::ScrollUp {
                view::zoom_in_at(state.zoom, state.pan, anchor, state.view_cells, (sx, sy))
            } else {
                view::zoom_out_at(state.zoom, state.pan, anchor, state.view_cells, (sx, sy))
            };
            state.zoom = nz;
            state.pan = np;
            state.status = format!("zoom x{}", state.zoom);
        }
        // Release ends any paint/erase stroke and any pan.
        EditorMouseKind::Release => {
            state.stroke_active = false;
            state.stroke_last_cell = None;
            state.last_pan_ptr = None;
        }
    }
}

/// Erase a single grid cell INLINE (geometry → `Empty` + `Shape::Full`), marking
/// the map dirty. Matches the keyboard `x` erase (`edit.rs`) but without taking a
/// snapshot — the caller snapshots once at the start of the erase stroke.
fn erase_cell(state: &mut EditorState, wx: usize, wy: usize) {
    let z = state.z;
    state.map.set(wx, wy, z, re_world::CellKind::Empty);
    state.map.set_shape(wx, wy, z, re_core::Shape::Full);
    state.mark_dirty();
}

/// Walk a gap-free Bresenham line from the stroke's last cell to `(wx, wy)` at the
/// active z and paint (or erase) EACH cell along it, so a fast drag leaves no holes.
/// No snapshot is taken — the press already pushed the single undo step covering the
/// whole stroke. Updates `stroke_last_cell` to the destination.
fn stroke_to(state: &mut EditorState, wx: usize, wy: usize, erase: bool) {
    let z = state.z;
    let (x0, y0) = match state.stroke_last_cell {
        Some((lx, ly, _)) => (lx as i32, ly as i32),
        None => (wx as i32, wy as i32),
    };
    for (cx, cy) in bresenham(x0, y0, wx as i32, wy as i32) {
        if cx < 0 || cy < 0 {
            continue;
        }
        let (cx, cy) = (cx as usize, cy as usize);
        state.cursor = (cx, cy);
        if erase {
            erase_cell(state, cx, cy);
        } else {
            edit::paint_tool_at_cursor(state); // NO snapshot — continuation
        }
    }
    state.stroke_last_cell = Some((wx, wy, z));
}

/// Integer Bresenham line from `(x0, y0)` to `(x1, y1)` INCLUSIVE of both ends, so
/// a stroke continuation paints every cell between the last cell and the new one.
fn bresenham(x0: i32, y0: i32, x1: i32, y1: i32) -> Vec<(i32, i32)> {
    let mut out = Vec::new();
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let (mut x, mut y) = (x0, y0);
    loop {
        out.push((x, y));
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
    out
}

/// MIDDLE-DRAG PAN: shift `state.pan` by the world-cell delta the pointer has moved
/// since `last_pan_ptr`, in the OPPOSITE direction (dragging the map right scrolls
/// the view left), clamped to `[0, map - visible]` per axis. Updates the anchor to
/// the current pointer so the next drag delta is incremental.
fn pan_drag(state: &mut EditorState, l: &layout::Layout, ev: EditorMouse) {
    let Some((ax, ay)) = state.last_pan_ptr else {
        return;
    };
    let zoom = state.zoom.max(1) as i32;
    // Framebuffer-cell delta since the anchor → world cells (one world cell == `zoom`
    // framebuffer cells), in the drag-the-map-moves-the-view (opposite) direction.
    let dcol = ev.col as i32 - ax as i32;
    let drow = ev.row as i32 - ay as i32;
    let dx_cells = dcol / zoom;
    let dy_cells = drow / zoom;
    if dx_cells == 0 && dy_cells == 0 {
        return; // not yet a whole-cell move; keep the anchor so it can accumulate
    }
    let (sx, sy, _sz) = state.map.size();
    let g = grid_scene_rect(l.grid);
    let cells_w = g.w.saturating_sub(RULER_W);
    let cells_h = g.h.saturating_sub(RULER_H);
    let vis_x = view::visible_cells(cells_w, state.zoom);
    let vis_y = view::visible_cells(cells_h, state.zoom);
    let max_x = sx.saturating_sub(vis_x);
    let max_y = sy.saturating_sub(vis_y);
    let new_x = (state.pan.0 as i32 - dx_cells).clamp(0, max_x as i32) as usize;
    let new_y = (state.pan.1 as i32 - dy_cells).clamp(0, max_y as i32) as usize;
    state.pan = (new_x, new_y);
    // Advance the anchor by the consumed whole-cell movement so leftover sub-cell
    // motion accumulates toward the next step instead of being dropped.
    let consumed_col = dx_cells * zoom;
    let consumed_row = dy_cells * zoom;
    state.last_pan_ptr = Some((
        (ax as i32 + consumed_col) as u16,
        (ay as i32 + consumed_row) as u16,
    ));
}

/// Route a click to whichever centered MODAL overlay is open (clicks must
/// never fall through to the grid underneath). Only `Press` events act; other
/// kinds are consumed silently. Each modal's geometry comes from the SAME
/// `chrome::*_geom`/hit helpers the painters use, so a click lands on what's drawn.
fn apply_modal_click(state: &mut EditorState, ev: EditorMouse, fb_w: usize, fb_h: usize) {
    use crate::chrome;
    // HELP WHEEL: the sheet is a ~180-line document, so the wheel scrolls it.
    // This sits ABOVE the press-only guard below, which is what used to swallow
    // every wheel event over the open sheet — the one gesture a reader reaches
    // for first did nothing at all.
    if state.help_open {
        match ev.kind {
            EditorMouseKind::ScrollUp => {
                crate::edit::help_scroll_by(state, -HELP_WHEEL_LINES);
                return;
            }
            EditorMouseKind::ScrollDown => {
                crate::edit::help_scroll_by(state, HELP_WHEEL_LINES);
                return;
            }
            _ => {}
        }
    }
    if ev.kind != EditorMouseKind::Press || ev.button != Some(EditorMouseButton::Left) {
        return; // consume non-left / non-press without leaking to the grid
    }
    let (col, row) = (ev.col as usize, ev.row as usize);

    // CRASH-RECOVERY OFFER: checked FIRST, mirroring `edit::apply_key`'s dispatch
    // order — the offer sits at the front of the key chain, so it must own the
    // click before any other modal too. `[ Restore ]` loads the sidecar,
    // `[ Discard ]` drops the offer (keeping the sidecar file); every other click
    // anywhere on screen is CONSUMED, which is the whole point: this is what stops
    // a stray toolbar `[Save]` from overwriting the un-recovered document.
    if state.autosave_offer.is_some() {
        match chrome::autosave_hit(fb_w, fb_h, col, row) {
            Some(chrome::AutosaveHit::Restore) => open::restore_autosave(state),
            Some(chrome::AutosaveHit::Discard) => open::discard_autosave_offer(state),
            None => {}
        }
        return;
    }

    // ROW EDITORS: the COLOUR overlay has a clickable swatch grid (its geometry
    // is `rowedit::color_layout`, the same one the painter drew from); the
    // numeric editor and the option popup are keyboard-driven, so their clicks
    // are simply consumed rather than leaking to the grid underneath.
    if state.color_edit.is_some() {
        if let Some(hit) = crate::rowedit::color_hit(state, fb_w, fb_h, col, row) {
            crate::rowedit::apply_color_hit(state, hit);
        }
        return;
    }
    if state.num_edit.is_some() || state.enum_pick.is_some() {
        return;
    }

    // HELP: only the [X] close button acts; everything else is consumed. (The
    // WHEEL is handled at the top of this fn, before the press-only guard.)
    if state.help_open {
        if let Some((x0, y0, pw, _ph)) = chrome::help_box_geom(state, fb_w, fb_h) {
            let (bx_y, bx_s, bx_e) = chrome::modal_close_button(x0, y0, pw);
            if row == bx_y && col >= bx_s && col < bx_e {
                state.close_help();
            }
        }
        return;
    }

    // DIRECTION PICKER: keyboard-driven; a click is consumed (no grid paint).
    // The picker grid cells are small and selection is via arrows + Enter, so a
    // stray click simply does nothing rather than leaking to the map underneath.
    if state.dir_picker.is_some() {
        return;
    }

    // WIZARD: field-row focus, [-]/[+] adjust, [Clr]/[Make]/[Cancel] + [X].
    if state.wizard.is_some() {
        if let Some(hit) = chrome::wizard_hit(fb_w, fb_h, col, row) {
            apply_wizard_hit(state, hit);
            return;
        }
        // [X] close on the title border.
        if let Some((x0, y0, pw, _ph)) = chrome::wizard_box_geom(fb_w, fb_h) {
            let (bx_y, bx_s, bx_e) = chrome::modal_close_button(x0, y0, pw);
            if row == bx_y && col >= bx_s && col < bx_e {
                state.wizard = None;
                state.status = "wizard cancelled".into();
            }
        }
        return;
    }

    // OPEN picker: a file-row click selects (and loads) it; footer [Load]/[Cancel];
    // [X] closes.
    if state.open.is_some() {
        if let Some(fi) = chrome::open_file_at(state, fb_w, fb_h, col, row) {
            // Single click on a row BOTH selects AND loads it (documented choice).
            if let Some(o) = state.open.as_mut() {
                o.cursor = fi;
            }
            open::pick(state);
            return;
        }
        if let Some(((ls, le), (cs, ce), frow)) = chrome::open_footer_buttons(state, fb_w, fb_h) {
            if row == frow && col >= ls && col < le {
                open::pick(state);
                return;
            }
            if row == frow && col >= cs && col < ce {
                open::close_overlay(state);
                return;
            }
        }
        if let Some((x0, y0, pw, _ph)) = chrome::open_box_geom(state, fb_w, fb_h) {
            let (bx_y, bx_s, bx_e) = chrome::modal_close_button(x0, y0, pw);
            if row == bx_y && col >= bx_s && col < bx_e {
                open::close_overlay(state);
            }
        }
        return;
    }

    // SAVE-AS: footer [Clr]/[Save]/[Cancel] + the top-border [X]. [Save] confirms
    // (and on success raises the SAVE flag to drive the real write).
    if state.saveas.is_some() {
        if let Some(hit) = chrome::saveas_hit(fb_w, fb_h, col, row) {
            apply_saveas_hit(state, hit);
            return;
        }
        if let Some((x0, y0, pw, _ph)) = chrome::saveas_box_geom(fb_w, fb_h) {
            let (bx_y, bx_s, bx_e) = chrome::modal_close_button(x0, y0, pw);
            if row == bx_y && col >= bx_s && col < bx_e {
                crate::saveas::close_saveas(state);
            }
        }
        return;
    }

    // SETTINGS: footer [Apply]/[Resize]/[Cancel] + the top-border [X]. Any other
    // click inside the box is simply consumed (field rows are keyboard-driven), so
    // nothing leaks to the grid underneath. Footer is mouse-clickable.
    if state.settings.is_some() {
        if let Some(hit) = chrome::settings_hit(fb_w, fb_h, col, row) {
            apply_settings_hit(state, hit);
            return;
        }
        if let Some((x0, y0, pw, _ph)) = chrome::settings_box_geom(fb_w, fb_h) {
            let (bx_y, bx_s, bx_e) = chrome::modal_close_button(x0, y0, pw);
            if row == bx_y && col >= bx_s && col < bx_e {
                state.settings = None;
                state.status = "level settings closed".into();
            }
        }
        return;
    }

    // MASK-EDIT: a part's solid-mask is authored entirely by keyboard (arrows / [ ] /
    // space / x / q). A click is consumed here so it can never leak to the grid.
    if state.mask_mode {
        return;
    }

    // PALETTE: a click on an item row picks (and closes) it; [X] closes.
    if state.palette.is_some() {
        if let Some(i) = chrome::palette_item_at(fb_w, fb_h, col, row) {
            state.palette = Some(i);
            crate::tree::palette_pick(state);
            return;
        }
        if let Some((x0, y0, pw, _ph)) = chrome::palette_box_geom(fb_w, fb_h) {
            let (bx_y, bx_s, bx_e) = chrome::modal_close_button(x0, y0, pw);
            if row == bx_y && col >= bx_s && col < bx_e {
                state.palette = None;
                state.status = "create palette closed".into();
            }
        }
    }
}

/// Apply a wizard button hit by SYNTHESIZING the equivalent key where one exists
/// (so modal guards + the create flow are inherited), or calling the wizard
/// backend directly for buttons with no key (`[Clr]`).
fn apply_wizard_hit(state: &mut EditorState, hit: crate::chrome::WizardHit) {
    use crate::chrome::WizardHit;
    use re_present::EditorKeyCode as K;
    match hit {
        WizardHit::Focus(f) => {
            if let Some(w) = state.wizard.as_mut() {
                w.focus = f;
            }
        }
        // The buttons adjust the focused field directly (calling the SAME
        // `Wizard::adjust` the key path uses) so they work even on the Name field,
        // where typed `+`/`-` would otherwise be free-text.
        WizardHit::Minus => {
            if let Some(w) = state.wizard.as_mut() {
                w.adjust(-1);
            }
        }
        WizardHit::Plus => {
            if let Some(w) = state.wizard.as_mut() {
                w.adjust(1);
            }
        }
        WizardHit::Clear => {
            if let Some(w) = state.wizard.as_mut() {
                w.name_clear();
            }
        }
        WizardHit::Make => edit::apply_key(
            state,
            EditorKey {
                code: K::Enter,
                shift: false,
            },
        ),
        WizardHit::Cancel => {
            state.wizard = None;
            state.status = "wizard cancelled".into();
        }
    }
}

/// Apply a SAVE-AS footer button hit. `[Save]` confirms (and on success raises
/// the SAVE flag so the Bevy `save_system` performs the real disk write through
/// the single choke point); `[Clr]` clears the buffer; `[Cancel]` closes.
fn apply_saveas_hit(state: &mut EditorState, hit: crate::chrome::SaveAsHit) {
    use crate::chrome::SaveAsHit;
    match hit {
        SaveAsHit::Save => {
            if crate::saveas::confirm(state) {
                state.status = "SAVE".into();
            }
        }
        SaveAsHit::Clear => crate::saveas::clear(state),
        SaveAsHit::Cancel => crate::saveas::close_saveas(state),
    }
}

/// Apply a LEVEL SETTINGS footer button hit. Each maps to the SAME backend the
/// keyboard path uses: `[Apply]` → `settings::apply` (confirm + close), `[Resize]`
/// → `settings::resize` (apply staged size, modal stays open), `[Cancel]` → close
/// discarding edits (mirrors the Esc path in `main::quit_on_esc`).
fn apply_settings_hit(state: &mut EditorState, hit: crate::chrome::SettingsHit) {
    use crate::chrome::SettingsHit;
    match hit {
        SettingsHit::Apply => crate::settings::apply(state),
        SettingsHit::Resize => crate::settings::resize(state),
        SettingsHit::Cancel => {
            state.settings = None;
            state.status = "level settings closed".into();
        }
    }
}

/// Run a PROPERTY-SHEET row click.
///
/// WHERE the click landed decides what happens. A click on the
/// LABEL focuses the field — the pre-existing behaviour, unchanged, so
/// `;`/`'` paging and a label click still agree. A click on a WIDGET part acts
/// directly: toggles the checkbox, presses the stepper, or opens the row's
/// editor. Both paths go through `rowedit`, which focuses the field first, so
/// the shared steppers always operate on the row that was actually clicked.
fn apply_row_click(
    state: &mut EditorState,
    action: crate::chrome::RowAction,
    zone: crate::rowvalue::RowZone,
) {
    use crate::rowvalue::{RowZone, WidgetPart as W};
    match zone {
        RowZone::Label => apply_row_focus(state, action),
        RowZone::Widget(W::Toggle) => crate::rowedit::toggle_bool(state, action),
        RowZone::Widget(W::Dec) => crate::rowedit::step_click(state, action, -1),
        RowZone::Widget(W::Inc) => crate::rowedit::step_click(state, action, 1),
        // The value / option / swatch affordances all open the row's editor;
        // `open_for` picks which one from the row's own type.
        RowZone::Widget(W::Value | W::Options | W::Swatch) => {
            if !crate::rowedit::open_for(state, action) {
                apply_row_focus(state, action);
            }
        }
    }
}

/// Focus the clicked row's field. The row action carries exactly which focus
/// register to write (entity / emitter / material), so a click reproduces what
/// `;`/`'` paging would land on.
fn apply_row_focus(state: &mut EditorState, action: crate::chrome::RowAction) {
    use crate::chrome::RowAction as A;
    match action {
        A::None => {}
        A::FocusEntity(f) => {
            state.entity_field = f;
            state.status = format!(
                "focus: {} = {}",
                f.label(),
                crate::entity::entity_field_value_of(state, f)
            );
        }
        A::FocusEmitter(f) => {
            state.emitter_field = f;
            state.status = format!(
                "focus: {} = {}",
                f.label(),
                crate::entity::emitter_field_value_of(state, f)
            );
        }
        A::FocusMaterial(f) => {
            state.material_field = f;
            state.status = format!(
                "focus: {} = {}",
                f.label(),
                crate::material::field_value_of(state, f)
            );
        }
    }
}

/// Run a panel action button. Wherever a key path exists the button SYNTHESIZES
/// that key (inheriting modal guards + undo); the dedicated view controls that
/// must ALWAYS act (zoom regardless of mode, noclip, tree, go-to-cursor, tree
/// actions) call the backend directly.
fn apply_panel_button(state: &mut EditorState, btn: crate::chrome::PanelButton) {
    use crate::chrome::PanelButton as B;
    use re_present::EditorKeyCode as K;
    let synth = |state: &mut EditorState, c: char| {
        edit::apply_key(
            state,
            EditorKey {
                code: K::Char(c),
                shift: false,
            },
        );
    };
    match btn {
        // P2 inspector — same keys the keyboard uses. The field-cycle key is
        // MODE-AWARE: material mode cycles the material field with `b`
        // (material::cycle_field); every other context uses `;` (entity field).
        B::Field => {
            if state.material_mode {
                synth(state, 'b');
            } else {
                synth(state, ';');
            }
        }
        B::Minus => synth(state, '-'),
        B::Plus => synth(state, '+'),
        B::Cycle => synth(state, 'c'),
        B::Delete => synth(state, 'x'),
        // Inspector extras — synthesize the matching key (inherits its guards).
        B::NewMat => synth(state, 'e'),
        B::PipeSwitch => synth(state, 'g'),
        B::TagNode => synth(state, 'n'),
        // Create-object palette toggle (always available).
        B::CreateObj => synth(state, 'i'),
        // P3 view — dedicated zoom always zooms (not mode-gated like the `+`/`-` key).
        B::ZoomIn => {
            state.zoom = view::zoom_in(state.zoom);
            state.status = format!("zoom x{}", state.zoom);
        }
        B::ZoomOut => {
            state.zoom = view::zoom_out(state.zoom);
            state.status = format!("zoom x{}", state.zoom);
        }
        B::FloorUp => synth(state, '>'),
        B::FloorDown => synth(state, '<'),
        B::Noclip => {
            state.noclip = !state.noclip;
            state.status = format!("noclip {}", state.noclip);
        }
        B::Tree => crate::tree::toggle(state),
        B::GoToCursor => crate::camera::jump_to_cursor(state),
        // P4 fill — same keys.
        B::SetA => synth(state, 'k'),
        B::Rect => synth(state, 'K'),
        B::Line => synth(state, 'L'),
        B::Bucket => synth(state, 'J'),
        // P6 tree actions — operate on the current tree cursor (direct backend).
        B::TreeDup => crate::tree::duplicate_current(state),
        B::TreeDel => crate::tree::delete_current(state),
        B::TreeParent => crate::hierarchy::reparent_key(state),
        B::TreeUnparent => crate::hierarchy::unparent_key(state),
    }
}

/// Route a toolbar button press to its action: set the view mode, trigger Save
/// (the same `status="SAVE"` flag the save system reads), open the file picker,
/// or toggle help.
fn apply_toolbar(state: &mut EditorState, btn: ToolbarButton) {
    match btn {
        ToolbarButton::Mode2D => {
            state.view_mode = ViewMode::Full2D;
            state.status = "view: Full2D".into();
        }
        ToolbarButton::Mode3D => {
            state.view_mode = ViewMode::Full3D;
            state.status = "view: Full3D".into();
        }
        ToolbarButton::ModeSplit => {
            state.view_mode = ViewMode::Split;
            state.status = "view: Split".into();
        }
        ToolbarButton::LevelSettings => crate::settings::open(state),
        ToolbarButton::Open => open::open_overlay(state),
        ToolbarButton::Save => state.status = "SAVE".into(),
        ToolbarButton::SaveAs => crate::saveas::open_saveas(state),
        // EDIT cluster: synthesize the SAME keys so the wizard/undo/redo
        // backends + guards are inherited. Undo/Redo are no-ops when their stack is
        // empty (the button is also drawn dim — see `toolbar::is_disabled`).
        ToolbarButton::New => {
            edit::apply_key(
                state,
                EditorKey {
                    code: re_present::EditorKeyCode::Char('W'),
                    shift: false,
                },
            );
        }
        ToolbarButton::Undo => {
            if !toolbar::is_disabled(state, ToolbarButton::Undo) {
                edit::apply_key(
                    state,
                    EditorKey {
                        code: re_present::EditorKeyCode::Char('u'),
                        shift: false,
                    },
                );
            }
        }
        ToolbarButton::Redo => {
            if !toolbar::is_disabled(state, ToolbarButton::Redo) {
                edit::apply_key(
                    state,
                    EditorKey {
                        code: re_present::EditorKeyCode::Char('y'),
                        shift: false,
                    },
                );
            }
        }
        ToolbarButton::Help => {
            // Through the SAME open/close pair the `?` key uses, so the toolbar
            // can never leave a stale scroll offset or filter behind — a sheet
            // reopened mid-document, or still filtered, reads as a broken sheet.
            if state.help_open {
                state.close_help();
            } else {
                state.open_help();
            }
        }
    }
}

/// Hit-test the Z-slider track in the INSPECTOR organ. The slider is painted at
/// `(inspector.x, inspector.y + Z_SLIDER_ROW)`, `Z_SLIDER_W` wide, with `[` / `]`
/// end caps and an inner track of `Z_SLIDER_W - 2` cells (the SAME offsets
/// `chrome::draw_z_slider` uses, relative to the inspector sub-rect origin). A
/// press on an inner-track cell sets `state.z` via [`view::z_from_slider`].
/// Returns `true` when the click was consumed.
fn try_z_slider_click(
    state: &mut EditorState,
    panel: layout::Rect,
    col: usize,
    row: usize,
) -> bool {
    if panel.is_empty() {
        return false;
    }
    let sy = panel.y + Z_SLIDER_ROW;
    if row != sy {
        return false;
    }
    // The inner track sits between the `[` (at panel.x) and `]` end caps.
    let inner = Z_SLIDER_W.saturating_sub(2).max(1);
    let track_x0 = panel.x + 1;
    if col < track_x0 || col >= track_x0 + inner {
        return false;
    }
    let (_sx, _sy, sz) = state.map.size();
    let idx = col - track_x0;
    state.z = view::z_from_slider(idx, inner, sz);
    state.status = format!("z {}/{}", state.z, sz.saturating_sub(1));
    true
}

/// Drain the [`EditorMouse`] channel into [`apply_mouse`], using the live
/// framebuffer size (the same `layout::compute` input as `draw_system`).
pub fn mouse_system(
    mut mouse: MessageReader<EditorMouse>,
    mut state: ResMut<EditorState>,
    fb: Res<Framebuffer>,
) {
    let (w, h) = (fb.cols(), fb.rows());
    for ev in mouse.read() {
        apply_mouse(&mut state, *ev, w, h);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use re_world::CellKind;

    fn press(col: u16, row: u16) -> EditorMouse {
        EditorMouse {
            col,
            row,
            kind: EditorMouseKind::Press,
            button: Some(EditorMouseButton::Left),
        }
    }

    /// A left press inside the grid moves the cursor to that cell AND paints the
    /// active brush there (the `space` action), and marks dirty.
    #[test]
    fn press_in_grid_moves_cursor_and_paints() {
        let mut s = EditorState::new_empty(16, 16, 4);
        s.brush = CellKind::WallSolid;
        // Full2D so the grid pane is large and at the origin.
        s.view_mode = crate::state::ViewMode::Full2D;
        let (w, h) = (120usize, 40usize);
        // Find the framebuffer pixel of world cell (3,2) via the layout + rulers.
        let l = layout::compute(s.view_mode, w, h, false);
        let g = grid_scene_rect(l.grid);
        let px = g.x + RULER_W + 3 * s.zoom as usize;
        let py = g.y + RULER_H + 2 * s.zoom as usize;
        apply_mouse(&mut s, press(px as u16, py as u16), w, h);
        assert_eq!(s.cursor, (3, 2), "cursor moved to the clicked cell");
        assert_eq!(s.map.cell_at(3, 2, 0), CellKind::WallSolid, "brush painted");
        assert!(s.dirty);
    }

    /// A drag continues the stroke: a second cell paints too.
    #[test]
    fn drag_paints_along_the_stroke() {
        let mut s = EditorState::new_empty(16, 16, 4);
        s.brush = CellKind::Floor;
        s.view_mode = crate::state::ViewMode::Full2D;
        let (w, h) = (120usize, 40usize);
        let l = layout::compute(s.view_mode, w, h, false);
        let g = grid_scene_rect(l.grid);
        let zoom = s.zoom as usize;
        let pix = |cx: usize, cy: usize| {
            (
                (g.x + RULER_W + cx * zoom) as u16,
                (g.y + RULER_H + cy * zoom) as u16,
            )
        };
        let (ax, ay) = pix(1, 1);
        apply_mouse(&mut s, press(ax, ay), w, h);
        let (bx, by) = pix(2, 1);
        apply_mouse(
            &mut s,
            EditorMouse {
                col: bx,
                row: by,
                kind: EditorMouseKind::Drag,
                button: Some(EditorMouseButton::Left),
            },
            w,
            h,
        );
        assert_eq!(s.map.cell_at(1, 1, 0), CellKind::Floor);
        assert_eq!(
            s.map.cell_at(2, 1, 0),
            CellKind::Floor,
            "drag extended stroke"
        );
    }

    /// 1.1 DRAG-SPAWN GUARD: in OBJECT mode a left press places exactly ONE
    /// object and does NOT arm a paint stroke, so dragging across further cells
    /// places nothing more. (Pre-fix: `stroke_to` re-ran `place_at_cursor` per
    /// Bresenham cell — one new node per cell dragged over.)
    #[test]
    fn drag_in_object_mode_places_exactly_one_object() {
        use crate::state::EntityKind;
        let mut s = EditorState::new_empty(16, 16, 4);
        s.view_mode = crate::state::ViewMode::Full2D;
        s.place = Some(EntityKind::Node);
        let (w, h) = (120usize, 40usize);
        let lay = layout::compute(s.view_mode, w, h, false);
        let scene = grid_scene_rect(lay.grid);
        let zoom = s.zoom as usize;
        let pix = |cx: usize, cy: usize| {
            (
                (scene.x + RULER_W + cx * zoom) as u16,
                (scene.y + RULER_H + cy * zoom) as u16,
            )
        };
        let (ax, ay) = pix(1, 1);
        apply_mouse(&mut s, press(ax, ay), w, h);
        assert_eq!(s.level.nodes.len(), 1, "press places exactly one node");
        assert!(!s.stroke_active, "placement press does NOT arm a stroke");
        // A fast drag to (4,1) (3 more cells) then release: still exactly one.
        let (dx, dy) = pix(4, 1);
        apply_mouse(
            &mut s,
            EditorMouse {
                col: dx,
                row: dy,
                kind: EditorMouseKind::Drag,
                button: Some(EditorMouseButton::Left),
            },
            w,
            h,
        );
        apply_mouse(
            &mut s,
            EditorMouse {
                col: dx,
                row: dy,
                kind: EditorMouseKind::Release,
                button: Some(EditorMouseButton::Left),
            },
            w,
            h,
        );
        assert_eq!(
            s.level.nodes.len(),
            1,
            "drag must place NO further objects (drag-spawn bug)"
        );
    }

    /// 1.1 DRAG-SPAWN GUARD: the VOX stamp and the STAIRCASE stamp are placement
    /// tools too — a press applies them once but never arms a paint stroke.
    #[test]
    fn vox_and_stair_press_do_not_arm_stroke() {
        for setup in [
            (|s: &mut EditorState| s.vox_mode = true) as fn(&mut EditorState),
            |s: &mut EditorState| s.stair_mode = true,
        ] {
            let mut s = EditorState::new_empty(16, 16, 4);
            s.view_mode = crate::state::ViewMode::Full2D;
            setup(&mut s);
            let (w, h) = (120usize, 40usize);
            let lay = layout::compute(s.view_mode, w, h, false);
            let scene = grid_scene_rect(lay.grid);
            let px = (scene.x + RULER_W + 2 * s.zoom as usize) as u16;
            let py = (scene.y + RULER_H + 2 * s.zoom as usize) as u16;
            apply_mouse(&mut s, press(px, py), w, h);
            assert!(
                !s.stroke_active,
                "vox/stair placement press must not arm a stroke"
            );
        }
    }

    /// 1.1 (kept behavior): a MATERIAL-mode drag still paints the material id
    /// along the stroke — material paint is a legitimate stroke tool.
    #[test]
    fn drag_in_material_mode_still_paints_stroke() {
        let mut s = EditorState::new_empty(16, 16, 4);
        s.view_mode = crate::state::ViewMode::Full2D;
        // Create + select a material and enter material mode (the `e` action).
        s.enter_tool_mode(crate::state::ToolMode::Material);
        crate::material::new_material(&mut s);
        let (w, h) = (120usize, 40usize);
        let lay = layout::compute(s.view_mode, w, h, false);
        let scene = grid_scene_rect(lay.grid);
        let zoom = s.zoom as usize;
        let pix = |cx: usize, cy: usize| {
            (
                (scene.x + RULER_W + cx * zoom) as u16,
                (scene.y + RULER_H + cy * zoom) as u16,
            )
        };
        let (ax, ay) = pix(1, 1);
        apply_mouse(&mut s, press(ax, ay), w, h);
        assert!(s.stroke_active, "material press ARMS the stroke (kept)");
        let (bx, by) = pix(3, 1);
        apply_mouse(
            &mut s,
            EditorMouse {
                col: bx,
                row: by,
                kind: EditorMouseKind::Drag,
                button: Some(EditorMouseButton::Left),
            },
            w,
            h,
        );
        // Every dragged cell carries material id 1 (1-based palette id).
        for x in 1..=3 {
            assert_eq!(
                s.map.side_data_at(x, 1, 0),
                1,
                "cell ({x},1) painted with material id 1 along the drag"
            );
        }
    }

    /// A press outside the grid pane (in the panel / preview region) is a no-op:
    /// neither the cursor nor the map changes.
    #[test]
    fn press_outside_grid_is_noop() {
        let mut s = EditorState::new_empty(16, 16, 4);
        s.view_mode = crate::state::ViewMode::Split;
        let (w, h) = (120usize, 40usize);
        let l = layout::compute(s.view_mode, w, h, false);
        // A column firmly inside the PREVIEW pane (right half), not the grid.
        let col = (l.preview.x + 1) as u16;
        let row = (l.preview.y + 2) as u16;
        let before_cursor = s.cursor;
        apply_mouse(&mut s, press(col, row), w, h);
        assert_eq!(s.cursor, before_cursor, "cursor unchanged outside grid");
        assert!(!s.dirty, "no paint outside grid");
    }

    /// CLICK-TO-FOCUS: a LEFT press in the grid pane sets focus=Grid AND still
    /// paints (focus is an additional side-effect, not a consumer); a LEFT press in
    /// the preview pane sets focus=Preview and stays inert for look (RMB contract).
    #[test]
    fn click_sets_viewport_focus_without_breaking_paint_or_look() {
        use crate::state::ViewportFocus;
        let mut s = EditorState::new_empty(16, 16, 4);
        s.brush = CellKind::WallSolid;
        s.view_mode = crate::state::ViewMode::Split;
        // Start focused on Preview so a grid click is a real change.
        s.focused_viewport = ViewportFocus::Preview;
        let (w, h) = (160usize, 48usize);
        let l = layout::compute_swapped(s.view_mode, w, h, false, s.swap_viewports);
        // A grid cell click (inside the scene, not the header/ruler) → focus Grid + paint.
        let g = grid_scene_rect(l.grid);
        let gx = (g.x + RULER_W + s.zoom as usize) as u16;
        let gy = (g.y + RULER_H + s.zoom as usize) as u16;
        apply_mouse(&mut s, press(gx, gy), w, h);
        assert_eq!(
            s.focused_viewport,
            ViewportFocus::Grid,
            "grid click focuses grid"
        );
        assert!(s.dirty, "grid click still paints");
        // A preview click → focus Preview, no paint, no capture.
        let pv = l.preview;
        let yaw0 = s.cam.yaw;
        apply_mouse(
            &mut s,
            mouse(
                (pv.x + 2) as u16,
                (pv.y + 2) as u16,
                EditorMouseKind::Press,
                Some(EditorMouseButton::Left),
            ),
            w,
            h,
        );
        assert_eq!(
            s.focused_viewport,
            ViewportFocus::Preview,
            "preview click focuses preview"
        );
        assert_eq!(
            s.capture_request, None,
            "left preview click does not capture"
        );
        assert_eq!(s.cam.yaw, yaw0, "left preview click does not rotate");
    }

    /// SWAP CONTROL: a LEFT press on the `[<>]` control of the LEFT viewport header
    /// toggles `swap_viewports`, and after the toggle the grid/preview x-order flips.
    #[test]
    fn swap_control_click_toggles_and_flips_sides() {
        let mut s = EditorState::new_empty(16, 16, 4);
        s.view_mode = crate::state::ViewMode::Split;
        let (w, h) = (160usize, 48usize);
        let l = layout::compute_swapped(s.view_mode, w, h, false, s.swap_viewports);
        assert!(l.grid.x < l.preview.x, "default: grid left of preview");
        // The control is on the LEFT viewport's header (the grid, un-swapped).
        let left_vp = l.grid;
        let (cs, _ce) = crate::chrome::viewport_swap_span(left_vp).expect("swap control fits");
        apply_mouse(&mut s, press(cs as u16, left_vp.y as u16), w, h);
        assert!(s.swap_viewports, "click toggled swap on");
        // Re-compute with the new flag: preview is now on the left.
        let l2 = layout::compute_swapped(s.view_mode, w, h, false, s.swap_viewports);
        assert!(l2.preview.x < l2.grid.x, "after swap: preview left of grid");
        // Clicking it again (now on the LEFT viewport = preview header) toggles back.
        let left2 = l2.preview;
        let (cs2, _e2) = crate::chrome::viewport_swap_span(left2).expect("swap control fits");
        apply_mouse(&mut s, press(cs2 as u16, left2.y as u16), w, h);
        assert!(!s.swap_viewports, "second click toggled swap off");
    }

    /// A press in the left ruler gutter (inside the grid rect but in the gutter)
    /// hits no cell — `screen_to_cell` returns `None`, so nothing happens.
    #[test]
    fn press_in_ruler_gutter_is_noop() {
        let mut s = EditorState::new_empty(16, 16, 4);
        s.view_mode = crate::state::ViewMode::Full2D;
        let (w, h) = (120usize, 40usize);
        let l = layout::compute(s.view_mode, w, h, false);
        // Column 0 of the grid pane is the left row-number ruler.
        let col = l.grid.x as u16;
        let row = (l.grid.y + RULER_H + 1) as u16;
        apply_mouse(&mut s, press(col, row), w, h);
        assert!(!s.dirty, "ruler-gutter press paints nothing");
    }

    fn mouse(
        col: u16,
        row: u16,
        kind: EditorMouseKind,
        button: Option<EditorMouseButton>,
    ) -> EditorMouse {
        EditorMouse {
            col,
            row,
            kind,
            button,
        }
    }

    /// A Press on the [2D]/[3D]/[Split] toolbar buttons sets the matching mode.
    #[test]
    fn toolbar_click_sets_view_mode() {
        let (w, h) = (120usize, 40usize);
        let spans = toolbar::button_spans();
        let l = layout::compute(ViewMode::Split, w, h, false);
        let tb = l.toolbar;
        assert!(!tb.is_empty());
        let click_button = |btn: ToolbarButton| {
            let (_, s, e) = spans.iter().copied().find(|&(b, _, _)| b == btn).unwrap();
            let col = tb.x + (s + e) / 2; // middle of the button span
            (col as u16, tb.y as u16)
        };
        let mut s = EditorState::new_empty(8, 8, 2);
        let (c, r) = click_button(ToolbarButton::Mode2D);
        apply_mouse(&mut s, press(c, r), w, h);
        assert_eq!(s.view_mode, ViewMode::Full2D, "[2D] → Full2D");
        let (c, r) = click_button(ToolbarButton::Mode3D);
        apply_mouse(&mut s, press(c, r), w, h);
        assert_eq!(s.view_mode, ViewMode::Full3D, "[3D] → Full3D");
        let (c, r) = click_button(ToolbarButton::ModeSplit);
        apply_mouse(&mut s, press(c, r), w, h);
        assert_eq!(s.view_mode, ViewMode::Split, "[Split] → Split");
    }

    /// The [Help] and [Open] toolbar buttons toggle help / open the picker.
    #[test]
    fn toolbar_help_and_open_buttons() {
        let (w, h) = (120usize, 40usize);
        let spans = toolbar::button_spans();
        let l = layout::compute(ViewMode::Split, w, h, false);
        let tb = l.toolbar;
        let mid = |btn: ToolbarButton| {
            let (_, sp, e) = spans.iter().copied().find(|&(b, _, _)| b == btn).unwrap();
            ((tb.x + (sp + e) / 2) as u16, tb.y as u16)
        };
        let mut s = EditorState::new_empty(8, 8, 2);
        let (c, r) = mid(ToolbarButton::Help);
        apply_mouse(&mut s, press(c, r), w, h);
        assert!(s.help_open, "[Help] toggles help on");
        // Toggle off.
        s.help_open = false; // (a real second click would also toggle)
        let (c, r) = mid(ToolbarButton::Open);
        apply_mouse(&mut s, press(c, r), w, h);
        assert!(s.open.is_some(), "[Open] opens the file picker");
    }

    /// A Press on the Z-slider track sets the active z-layer (inverse of the knob).
    #[test]
    fn z_slider_click_sets_z() {
        use crate::chrome::{Z_SLIDER_ROW, Z_SLIDER_W};
        let (w, h) = (120usize, 40usize);
        let mut s = EditorState::new_empty(8, 8, 8); // sz = 8
        s.view_mode = ViewMode::Split;
        let l = layout::compute(s.view_mode, w, h, false);
        // The z-slider now lives in the INSPECTOR organ (bottom of the panel split).
        let panel = crate::chrome::inspector_subrect(&s, l.panel);
        assert!(!panel.is_empty(), "inspector organ present at this size");
        let inner = Z_SLIDER_W - 2;
        let track_x0 = panel.x + 1;
        let sy = panel.y + Z_SLIDER_ROW;
        // Click the far-right inner cell → top layer (sz-1 = 7).
        let col = (track_x0 + inner - 1) as u16;
        apply_mouse(&mut s, press(col, sy as u16), w, h);
        assert_eq!(s.z, 7, "right end of the slider selects the top layer");
        // Click the far-left inner cell → floor (0).
        apply_mouse(&mut s, press(track_x0 as u16, sy as u16), w, h);
        assert_eq!(s.z, 0, "left end selects the floor");
    }

    /// RMB-HELD LOOK: a LEFT Press in the preview pane is INERT now — look is the
    /// right button. It must not request capture, must not rotate, must not paint.
    #[test]
    fn preview_left_press_is_inert() {
        let (w, h) = (120usize, 40usize);
        let mut s = EditorState::new_empty(8, 8, 4);
        s.view_mode = ViewMode::Split;
        let l = layout::compute(s.view_mode, w, h, false);
        let p = l.preview;
        let yaw0 = s.cam.yaw;
        apply_mouse(
            &mut s,
            mouse(
                (p.x + 2) as u16,
                (p.y + 2) as u16,
                EditorMouseKind::Press,
                Some(EditorMouseButton::Left),
            ),
            w,
            h,
        );
        assert_eq!(
            s.capture_request, None,
            "left press does NOT request capture"
        );
        assert_eq!(s.cam.yaw, yaw0, "left press does not rotate");
        assert!(!s.dirty, "preview click never paints the grid");
    }

    /// RMB-HELD LOOK: a RIGHT Press in the preview requests capture ON (the look
    /// system then drives rotation). Capture == "RMB is held".
    #[test]
    fn preview_right_press_requests_capture_on() {
        let (w, h) = (120usize, 40usize);
        let mut s = EditorState::new_empty(8, 8, 4);
        s.view_mode = ViewMode::Split;
        let l = layout::compute(s.view_mode, w, h, false);
        let p = l.preview;
        apply_mouse(
            &mut s,
            mouse(
                (p.x + 2) as u16,
                (p.y + 2) as u16,
                EditorMouseKind::Press,
                Some(EditorMouseButton::Right),
            ),
            w,
            h,
        );
        assert_eq!(
            s.capture_request,
            Some(true),
            "right press requests capture ON"
        );
        assert!(!s.dirty, "right press never paints");
    }

    /// STAGE 3: the wheel over the preview dollies the camera forward/back.
    #[test]
    fn preview_wheel_dollies_camera() {
        let (w, h) = (120usize, 40usize);
        let mut s = EditorState::new_empty(16, 16, 4);
        s.view_mode = ViewMode::Split;
        s.noclip = true;
        s.cam.yaw = 0.0; // faces +y
        let l = layout::compute(s.view_mode, w, h, false);
        let p = l.preview;
        let y0 = s.cam.pos.y;
        apply_mouse(
            &mut s,
            mouse(
                (p.x + 2) as u16,
                (p.y + 2) as u16,
                EditorMouseKind::ScrollUp,
                None,
            ),
            w,
            h,
        );
        assert!(s.cam.pos.y > y0, "wheel up dollies forward");
        let y1 = s.cam.pos.y;
        apply_mouse(
            &mut s,
            mouse(
                (p.x + 2) as u16,
                (p.y + 2) as u16,
                EditorMouseKind::ScrollDown,
                None,
            ),
            w,
            h,
        );
        assert!(s.cam.pos.y < y1, "wheel down dollies back");
    }

    /// RMB-HELD LOOK: while captured, a RIGHT-button RELEASE anywhere (even outside
    /// the preview, since a drag may have left the pane) requests RELEASE.
    #[test]
    fn preview_right_release_while_captured_requests_release() {
        let (w, h) = (120usize, 40usize);
        let mut s = EditorState::new_empty(8, 8, 4);
        s.view_mode = ViewMode::Split;
        s.preview_captured = true;
        apply_mouse(
            &mut s,
            mouse(
                5,
                5,
                EditorMouseKind::Release,
                Some(EditorMouseButton::Right),
            ),
            w,
            h,
        );
        assert_eq!(
            s.capture_request,
            Some(false),
            "RMB release while captured releases"
        );
    }

    /// RMB-HELD LOOK: while captured, a LEFT click does NOT release (capture is the
    /// right button only) and never paints the grid.
    #[test]
    fn preview_left_click_while_captured_does_not_release() {
        let (w, h) = (120usize, 40usize);
        let mut s = EditorState::new_empty(8, 8, 4);
        s.view_mode = ViewMode::Split;
        s.preview_captured = true;
        apply_mouse(&mut s, press(5, 5), w, h);
        assert_eq!(
            s.capture_request, None,
            "a left click while captured is swallowed, not a release"
        );
    }

    /// PART 1: while captured the grid is MODAL — a Left Press in the grid does not
    /// paint (dirty stays false).
    #[test]
    fn captured_grid_click_does_not_paint() {
        let (w, h) = (120usize, 40usize);
        let mut s = EditorState::new_empty(16, 16, 4);
        s.view_mode = ViewMode::Full2D;
        s.preview_captured = true;
        let l = layout::compute(s.view_mode, w, h, false);
        let col = (l.grid.x + RULER_W + 1) as u16;
        let row = (l.grid.y + RULER_H + 1) as u16;
        apply_mouse(&mut s, press(col, row), w, h);
        assert!(!s.dirty, "no paint while captured (modal over the grid)");
    }

    /// RMB-HELD LOOK: while captured the pure `apply_mouse` no longer rotates from
    /// drag deltas — the look is driven by `preview_capture_look_system` (OS-grab
    /// delta or the crossterm cell fallback). A drag here is swallowed: the camera
    /// is untouched and nothing paints. (The actual look math is tested in `main.rs`
    /// via `apply_cell_look`.)
    #[test]
    fn captured_drag_is_swallowed_no_rotation() {
        let (w, h) = (120usize, 40usize);
        let mut s = EditorState::new_empty(8, 8, 4);
        s.view_mode = ViewMode::Split;
        s.preview_captured = true;
        let l = layout::compute(s.view_mode, w, h, false);
        let p = l.preview;
        let (ax, ay) = ((p.x + 2) as u16, (p.y + 2) as u16);
        let yaw0 = s.cam.yaw;
        let pitch0 = s.cam.pitch;
        apply_mouse(
            &mut s,
            mouse(ax, ay, EditorMouseKind::Drag, Some(EditorMouseButton::Left)),
            w,
            h,
        );
        apply_mouse(
            &mut s,
            mouse(
                ax + 4,
                ay + 3,
                EditorMouseKind::Drag,
                Some(EditorMouseButton::Left),
            ),
            w,
            h,
        );
        assert_eq!(
            s.cam.yaw, yaw0,
            "captured drag no longer rotates in apply_mouse"
        );
        assert_eq!(
            s.cam.pitch, pitch0,
            "captured drag no longer pitches in apply_mouse"
        );
        assert!(!s.dirty, "captured drag never paints");
    }

    /// A Press on a toolbox tool row sets the active tool via the SAME key path
    /// the keyboard uses (vertical left column).
    #[test]
    fn toolbox_click_sets_tool() {
        use crate::palette_bar::{self, ToolButton, ToolRow};
        let (w, h) = (120usize, 40usize);
        let mut s = EditorState::new_empty(8, 8, 2);
        let l = layout::compute(s.view_mode, w, h, false);
        let tbx = l.toolbox;
        assert!(!tbx.is_empty());
        // Resolve the absolute (col,row) of a tool's row from the SSOT geometry.
        let row_of = |btn: ToolButton| {
            let ry = palette_bar::tool_button_rows()
                .into_iter()
                .find_map(|(r, y)| matches!(r, ToolRow::Tool(b) if b == btn).then_some(y))
                .unwrap();
            ((tbx.x + 1) as u16, (tbx.y + ry) as u16)
        };
        // Click Floor → Floor brush, plain mode.
        let (c, r) = row_of(ToolButton::Floor);
        apply_mouse(&mut s, press(c, r), w, h);
        assert_eq!(
            s.brush,
            re_world::CellKind::Floor,
            "Floor selects the floor brush"
        );
        // Click Wedge → shape mode.
        let (c, r) = row_of(ToolButton::Wedge);
        apply_mouse(&mut s, press(c, r), w, h);
        assert!(s.shape_mode, "Wedge enters shape mode");
    }

    /// A click on a toolbox GROUP HEADER row is consumed but changes no tool.
    #[test]
    fn toolbox_header_click_is_inert() {
        use crate::palette_bar::{self, ToolRow};
        let (w, h) = (120usize, 40usize);
        let mut s = EditorState::new_empty(8, 8, 2);
        s.brush = re_world::CellKind::WallSolid;
        let l = layout::compute(s.view_mode, w, h, false);
        let tbx = l.toolbox;
        // First header row offset.
        let hy = palette_bar::tool_button_rows()
            .into_iter()
            .find_map(|(r, y)| matches!(r, ToolRow::Header(_)).then_some(y))
            .unwrap();
        apply_mouse(&mut s, press((tbx.x + 1) as u16, (tbx.y + hy) as u16), w, h);
        assert_eq!(
            s.brush,
            re_world::CellKind::WallSolid,
            "header click changes nothing"
        );
        assert!(!s.dirty, "header click never paints");
    }

    /// A click anywhere while a modal (help) is open must NOT paint the grid
    /// underneath.
    #[test]
    fn modal_open_click_does_not_paint_grid() {
        let (w, h) = (120usize, 40usize);
        let mut s = EditorState::new_empty(16, 16, 4);
        s.view_mode = ViewMode::Full2D;
        s.brush = re_world::CellKind::WallSolid;
        s.help_open = true;
        let l = layout::compute(s.view_mode, w, h, false);
        let col = (l.grid.x + RULER_W + 1) as u16;
        let row = (l.grid.y + RULER_H + 1) as u16;
        apply_mouse(&mut s, press(col, row), w, h);
        assert!(!s.dirty, "no paint under an open modal");
    }

    /// Clicking a file row in the OPEN picker selects+loads it (here: closes via
    /// pick when the cursor lands on a valid row). We assert the row click is routed
    /// to the modal, not the grid.
    #[test]
    fn open_modal_row_click_routes_to_modal() {
        use crate::state::OpenState;
        let (w, h) = (120usize, 40usize);
        let mut s = EditorState::new_empty(8, 8, 2);
        s.open = Some(OpenState {
            files: vec!["levels/a.lvl".into(), "levels/b.lvl".into()],
            cursor: 0,
        });
        // Compute the open box body row for index 1 and click it.
        let fi = {
            // Find a (col,row) that hits file index 1.
            let mut found = None;
            for row in 0..h {
                for col in 0..w {
                    if crate::chrome::open_file_at(&s, w, h, col, row) == Some(1) {
                        found = Some((col as u16, row as u16));
                        break;
                    }
                }
                if found.is_some() {
                    break;
                }
            }
            found
        };
        let (c, r) = fi.expect("a clickable row for file 1 exists");
        apply_mouse(&mut s, press(c, r), w, h);
        // pick() takes the overlay; a failed load (no such file) still closes it,
        // proving the click routed to the modal rather than painting the grid.
        assert!(!s.dirty, "open-modal click does not paint");
        assert!(
            s.open.is_none(),
            "row click routes through pick() and closes the overlay"
        );
    }

    // ── pane_at: pure pointer→pane classification, per view mode ────────────────

    /// SPLIT: the right pane is Preview; the left body column is the Toolbox, then
    /// the Grid; the panel column is Panel; row 0 is the Toolbar even above the
    /// preview.
    #[test]
    fn pane_at_split_classifies_each_region() {
        let (w, h) = (120usize, 40usize);
        let l = layout::compute(ViewMode::Split, w, h, false);
        // Chrome toolbar wins on row 0, even within the preview's x-range.
        assert_eq!(pane_at(&l, l.preview.x + 1, l.toolbar.y), Pane::Toolbar);
        // The vertical toolbox is the left body column.
        assert_eq!(pane_at(&l, l.toolbox.x + 1, l.toolbox.y + 2), Pane::Toolbox);
        // Deep inside each body pane.
        assert_eq!(pane_at(&l, l.preview.x + 2, l.preview.y + 2), Pane::Preview);
        assert_eq!(pane_at(&l, l.grid.x + 1, l.grid.y + 1), Pane::Grid);
        assert_eq!(pane_at(&l, l.panel.x + 1, l.panel.y + 1), Pane::Panel);
    }

    /// FULL2D: no preview pane exists, so nowhere classifies as Preview; the grid
    /// and panel are present, rows 0/1 are chrome.
    #[test]
    fn pane_at_full2d_has_no_preview() {
        let (w, h) = (120usize, 40usize);
        let l = layout::compute(ViewMode::Full2D, w, h, false);
        assert!(l.preview.is_empty());
        assert_eq!(pane_at(&l, l.grid.x + 1, l.grid.y + 1), Pane::Grid);
        assert_eq!(pane_at(&l, l.panel.x + 1, l.panel.y + 1), Pane::Panel);
        assert_eq!(pane_at(&l, 5, l.toolbar.y), Pane::Toolbar);
        // Scan the whole body: nothing is Preview in Full2D.
        for row in l.grid.y..h {
            for col in 0..w {
                assert_ne!(pane_at(&l, col, row), Pane::Preview, "no preview in Full2D");
            }
        }
    }

    /// FULL3D: the body below the chrome + status strip is all Preview; the grid
    /// never appears.
    #[test]
    fn pane_at_full3d_is_preview_below_chrome() {
        let (w, h) = (120usize, 40usize);
        let l = layout::compute(ViewMode::Full3D, w, h, false);
        assert!(l.grid.is_empty());
        assert_eq!(pane_at(&l, w / 2, l.preview.y + 2), Pane::Preview);
        // The chrome toolbar still wins on row 0.
        assert_eq!(pane_at(&l, w / 2, l.toolbar.y), Pane::Toolbar);
        // Full3D is a clean walk-through: NO toolbox/panel. The whole body below the
        // chrome is the preview, including the left edge.
        assert!(l.toolbox.is_empty(), "Full3D hides the toolbox");
        assert!(l.panel.is_empty(), "Full3D hides the panel");
        assert_eq!(pane_at(&l, l.preview.x + 1, l.preview.y + 2), Pane::Preview);
    }

    /// The OUTLINER now lives INSIDE the side panel (top organ), so a click in its
    /// region classifies as `Pane::Panel` (the carved `Pane::Tree` strip is retired).
    #[test]
    fn pane_at_outliner_region_is_panel() {
        let (w, h) = (120usize, 40usize);
        let l = layout::compute(ViewMode::Split, w, h, true);
        assert!(l.tree.is_empty(), "no carved tree strip any more");
        let mut s = EditorState::new_empty(8, 8, 2);
        s.view_mode = ViewMode::Split;
        let outliner = crate::chrome::outliner_subrect(&s, l.panel);
        assert!(!outliner.is_empty(), "outliner organ present");
        assert_eq!(pane_at(&l, outliner.x + 1, outliner.y + 1), Pane::Panel);
    }

    /// A far-out-of-bounds coordinate classifies as None.
    #[test]
    fn pane_at_out_of_bounds_is_none() {
        let (w, h) = (120usize, 40usize);
        let l = layout::compute(ViewMode::Split, w, h, false);
        assert_eq!(pane_at(&l, w + 50, h + 50), Pane::None);
    }

    /// SPLIT: a left press in the right (preview) pane routes to the camera (anchors
    /// drag-look) and does NOT fall through to any toolbar/panel/grid handler (no
    /// paint, no capture request — capture is the RIGHT button now).
    #[test]
    fn split_preview_press_routes_to_camera_not_buttons() {
        let (w, h) = (120usize, 40usize);
        let mut s = EditorState::new_empty(8, 8, 4);
        s.view_mode = ViewMode::Split;
        let l = layout::compute(s.view_mode, w, h, false);
        let p = l.preview;
        apply_mouse(&mut s, press((p.x + 3) as u16, (p.y + 3) as u16), w, h);
        assert_eq!(
            s.capture_request, None,
            "left preview press does not request capture"
        );
        assert!(
            !s.dirty,
            "preview press never paints the grid (routed to the camera pane, swallowed)"
        );
    }

    /// FULL3D: a left press anywhere in the body routes to the camera (anchors
    /// drag-look across the whole view) rather than painting.
    #[test]
    fn full3d_press_routes_to_camera() {
        let (w, h) = (120usize, 40usize);
        let mut s = EditorState::new_empty(8, 8, 4);
        s.view_mode = ViewMode::Full3D;
        let l = layout::compute(s.view_mode, w, h, false);
        apply_mouse(
            &mut s,
            press((w / 2) as u16, (l.preview.y + 4) as u16),
            w,
            h,
        );
        assert!(
            !s.dirty,
            "Full3D press never paints (routed to the camera pane, swallowed)"
        );
    }

    /// SPLIT: the toolbar stays clickable even though the preview pane owns the
    /// mouse below it — a click on the [3D] button (row 0) still switches mode.
    #[test]
    fn split_toolbar_still_clickable_above_preview() {
        let (w, h) = (120usize, 40usize);
        let mut s = EditorState::new_empty(8, 8, 4);
        s.view_mode = ViewMode::Split;
        let spans = toolbar::button_spans();
        let l = layout::compute(s.view_mode, w, h, false);
        let (_, st, e) = spans
            .iter()
            .copied()
            .find(|&(b, _, _)| b == ToolbarButton::Mode3D)
            .unwrap();
        let col = l.toolbar.x + (st + e) / 2;
        apply_mouse(&mut s, press(col as u16, l.toolbar.y as u16), w, h);
        assert_eq!(
            s.view_mode,
            ViewMode::Full3D,
            "[3D] toolbar click still works in Split"
        );
    }

    /// A press OUTSIDE the preview pane does not rotate the camera.
    #[test]
    fn press_outside_preview_does_not_rotate() {
        let (w, h) = (120usize, 40usize);
        let mut s = EditorState::new_empty(8, 8, 4);
        s.view_mode = ViewMode::Full2D; // no preview pane at all
        let yaw0 = s.cam.yaw;
        let l = layout::compute(s.view_mode, w, h, false);
        // A grid press (paints) must not touch the camera.
        let col = (l.grid.x + RULER_W + 1) as u16;
        let row = (l.grid.y + RULER_H + 1) as u16;
        apply_mouse(&mut s, press(col, row), w, h);
        assert_eq!(s.cam.yaw, yaw0, "grid press leaves the camera alone");
    }

    /// Find the absolute middle (col,row) of a panel action button, or None if it
    /// isn't currently shown.
    fn panel_btn_mid(
        state: &EditorState,
        w: usize,
        h: usize,
        target: crate::chrome::PanelButton,
    ) -> Option<(u16, u16)> {
        let l = layout::compute(state.view_mode, w, h, state.tree.open);
        // Buttons live in the INSPECTOR organ (bottom of the panel split). Span
        // discovery goes through the compatibility accessor.
        let inspector = crate::chrome::inspector_subrect(state, l.panel);
        crate::chrome::button_span(state, inspector, target)
            .map(|(s, e, r)| (((s + e) / 2) as u16, r as u16))
    }

    /// The [Z+] panel button always zooms in.
    #[test]
    fn panel_zoom_button_zooms() {
        use crate::chrome::PanelButton;
        let (w, h) = (120usize, 40usize);
        let mut s = EditorState::new_empty(8, 8, 2);
        s.view_mode = ViewMode::Split;
        let z0 = s.zoom;
        let (c, r) = panel_btn_mid(&s, w, h, PanelButton::ZoomIn).expect("Z+ shown");
        apply_mouse(&mut s, press(c, r), w, h);
        assert!(s.zoom > z0, "[Z+] raises zoom");
    }

    /// The [Ncl] button toggles noclip; [Tree] opens the tree.
    #[test]
    fn panel_noclip_and_tree_buttons() {
        use crate::chrome::PanelButton;
        let (w, h) = (120usize, 40usize);
        let mut s = EditorState::new_empty(8, 8, 2);
        s.view_mode = ViewMode::Split;
        let nc0 = s.noclip;
        let (c, r) = panel_btn_mid(&s, w, h, PanelButton::Noclip).expect("Ncl shown");
        apply_mouse(&mut s, press(c, r), w, h);
        assert_eq!(s.noclip, !nc0, "[Ncl] toggles noclip");
        let (c, r) = panel_btn_mid(&s, w, h, PanelButton::Tree).expect("Tree shown");
        apply_mouse(&mut s, press(c, r), w, h);
        assert!(s.tree.open, "[Tree] opens the outliner");
    }

    /// The [SetA] button sets the fill anchor at the cursor.
    #[test]
    fn panel_set_a_button_sets_fill_anchor() {
        use crate::chrome::PanelButton;
        let (w, h) = (120usize, 40usize);
        let mut s = EditorState::new_empty(8, 8, 2);
        s.view_mode = ViewMode::Split;
        s.cursor = (2, 3);
        let (c, r) = panel_btn_mid(&s, w, h, PanelButton::SetA).expect("SetA shown");
        apply_mouse(&mut s, press(c, r), w, h);
        assert_eq!(s.fill_anchor, Some((2, 3, 0)), "[SetA] sets fill corner A");
    }

    /// With an object selected and a numeric field focused, [+] adjusts it.
    #[test]
    fn panel_plus_button_adjusts_focused_field() {
        use crate::chrome::PanelButton;
        use crate::state::{EntityField, EntityKind};
        let (w, h) = (120usize, 40usize);
        let mut s = EditorState::new_empty(8, 8, 2);
        s.view_mode = ViewMode::Split;
        // Enter Door placement, focus the X position field.
        s.place = Some(EntityKind::Door);
        s.entity_field = EntityField::DoorPosX;
        let x0 = s.level.door.x;
        let (c, r) = panel_btn_mid(&s, w, h, PanelButton::Plus).expect("[+] shown w/ context");
        apply_mouse(&mut s, press(c, r), w, h);
        assert_ne!(s.level.door.x, x0, "[+] adjusted the focused door field");
    }

    /// Clicking an OUTLINER object row sets the cursor and selects it. The
    /// outliner is always visible inside the panel now (no `tree.open` gate).
    #[test]
    fn outliner_row_click_selects() {
        let (w, h) = (120usize, 40usize);
        let mut s = EditorState::new_empty(16, 16, 4);
        s.view_mode = ViewMode::Full2D;
        // A node so the tree has at least one group + object.
        s.level.push_node(2, 2, 0);
        let l = layout::compute(s.view_mode, w, h, s.tree.open);
        let outliner = crate::chrome::outliner_subrect(&s, l.panel);
        assert!(!outliner.is_empty(), "outliner organ present");
        // Find a row that maps to a tree index (the first body row).
        let mut hit = None;
        for row in outliner.y..(outliner.y + outliner.h) {
            if let Some(ri) = crate::chrome::tree_row_at(&s, outliner, outliner.x + 1, row) {
                hit = Some((outliner.x as u16 + 1, row as u16, ri));
                break;
            }
        }
        let (c, r, ri) = hit.expect("a clickable outliner row exists");
        apply_mouse(&mut s, press(c, r), w, h);
        assert_eq!(
            s.tree.cursor, ri,
            "outliner click moves the cursor to the clicked row"
        );
    }

    /// In MATERIAL mode the `[Fld]` button cycles the MATERIAL field (key `b`),
    /// not the entity field (`;`). And `[Fld]` is NOT shown in vox mode.
    #[test]
    fn panel_field_button_is_mode_aware() {
        use crate::chrome::PanelButton;
        let (w, h) = (120usize, 40usize);
        let mut s = EditorState::new_empty(8, 8, 2);
        s.view_mode = ViewMode::Split;
        // Material mode with a real material so the field cycle is meaningful.
        s.material_mode = true;
        crate::material::new_material(&mut s);
        let f0 = s.material_field;
        let (c, r) =
            panel_btn_mid(&s, w, h, PanelButton::Field).expect("[Fld] shown in material mode");
        apply_mouse(&mut s, press(c, r), w, h);
        assert_ne!(
            s.material_field, f0,
            "[Fld] cycles the MATERIAL field in material mode"
        );
        // Vox mode hides [Fld] (vox uses +/- scale, c material, x delete; no field).
        let mut sv = EditorState::new_empty(8, 8, 2);
        sv.view_mode = ViewMode::Split;
        sv.vox_mode = true;
        assert!(
            panel_btn_mid(&sv, w, h, PanelButton::Field).is_none(),
            "[Fld] is not shown in vox mode"
        );
    }

    /// `[+Mat]` is shown in material mode and clicking it creates a material.
    #[test]
    fn panel_new_mat_button_creates_material() {
        use crate::chrome::PanelButton;
        let (w, h) = (120usize, 40usize);
        let mut s = EditorState::new_empty(8, 8, 2);
        s.view_mode = ViewMode::Split;
        s.material_mode = true;
        let n0 = s.level.materials.len();
        let (c, r) =
            panel_btn_mid(&s, w, h, PanelButton::NewMat).expect("[+Mat] shown in material mode");
        apply_mouse(&mut s, press(c, r), w, h);
        assert_eq!(
            s.level.materials.len(),
            n0 + 1,
            "[+Mat] creates a new material"
        );
        // Not shown when material mode is off.
        let mut s2 = EditorState::new_empty(8, 8, 2);
        s2.view_mode = ViewMode::Split;
        assert!(
            panel_btn_mid(&s2, w, h, PanelButton::NewMat).is_none(),
            "[+Mat] hidden outside material mode"
        );
    }

    /// `[Sw]` shows while placing a Pipe; clicking it toggles the selected
    /// pipe's switch cell at the cursor (synthesizing `g`).
    #[test]
    fn panel_pipe_switch_button_toggles_switch() {
        use crate::chrome::PanelButton;
        use crate::state::EntityKind;
        let (w, h) = (120usize, 40usize);
        let mut s = EditorState::new_empty(8, 8, 2);
        s.view_mode = ViewMode::Split;
        s.place = Some(EntityKind::Pipe);
        // Build a pipe with a path so `g` (switch toggle) has a path cell to act on.
        s.level.pipes.push(re_level::Pipe {
            from_node: 0,
            to_node: 1,
            path: vec![(2, 2, 0), (3, 2, 0)],
            switch_cell: None,
            ..Default::default()
        });
        s.selected_pipe = Some(0);
        s.cursor = (2, 2);
        let (c, r) = panel_btn_mid(&s, w, h, PanelButton::PipeSwitch)
            .expect("[Sw] shown while placing a Pipe");
        apply_mouse(&mut s, press(c, r), w, h);
        assert_eq!(
            s.level.pipes[0].switch_cell,
            Some(0),
            "[Sw] sets the switch cell at the cursor path cell"
        );
        // Not shown when not placing a Pipe.
        let mut s2 = EditorState::new_empty(8, 8, 2);
        s2.view_mode = ViewMode::Split;
        s2.place = Some(EntityKind::Node);
        assert!(
            panel_btn_mid(&s2, w, h, PanelButton::PipeSwitch).is_none(),
            "[Sw] hidden when not a Pipe"
        );
    }

    /// `[Tag]` shows while placing a Node; clicking it tags the cursor cell as
    /// an Object carrying node-id 1 (synthesizing `n`).
    #[test]
    fn panel_tag_node_button_tags_cell() {
        use crate::chrome::PanelButton;
        use crate::state::EntityKind;
        let (w, h) = (120usize, 40usize);
        let mut s = EditorState::new_empty(8, 8, 2);
        s.view_mode = ViewMode::Split;
        s.place = Some(EntityKind::Node);
        s.cursor = (1, 1);
        let (c, r) = panel_btn_mid(&s, w, h, PanelButton::TagNode)
            .expect("[Tag] shown while placing a Node");
        apply_mouse(&mut s, press(c, r), w, h);
        assert_eq!(
            s.map.cell_at(1, 1, 0),
            CellKind::Object,
            "[Tag] marks the cell an Object"
        );
        assert_eq!(
            s.map.side_data_at(1, 1, 0),
            1,
            "[Tag] writes node-id 1 in side_data"
        );
        // Not shown when not placing a Node.
        let mut s2 = EditorState::new_empty(8, 8, 2);
        s2.view_mode = ViewMode::Split;
        s2.place = Some(EntityKind::Door);
        assert!(
            panel_btn_mid(&s2, w, h, PanelButton::TagNode).is_none(),
            "[Tag] hidden when not a Node"
        );
    }

    /// `[Create]` is always shown; clicking it opens the create-object palette.
    #[test]
    fn panel_create_button_opens_palette() {
        use crate::chrome::PanelButton;
        let (w, h) = (120usize, 40usize);
        let mut s = EditorState::new_empty(8, 8, 2);
        s.view_mode = ViewMode::Split;
        assert!(s.palette.is_none());
        let (c, r) =
            panel_btn_mid(&s, w, h, PanelButton::CreateObj).expect("[Create] always shown");
        apply_mouse(&mut s, press(c, r), w, h);
        assert!(
            s.palette.is_some(),
            "[Create] opens the create-object palette"
        );
    }

    // ── TYPED-ROW WIDGET CLICKS ──────────────────────────────────────────────

    /// A state with `kind` placed + selected, plus the live inspector rect.
    fn widget_fixture(kind: crate::state::EntityKind) -> (EditorState, crate::layout::Rect) {
        let (w, h) = (120usize, 40usize);
        let mut s = EditorState::new_empty(16, 16, 4);
        s.view_mode = ViewMode::Full2D;
        s.cursor = (2, 2);
        s.z = 1;
        s.place = Some(kind);
        crate::entity::place_at_cursor(&mut s);
        let l = layout::compute(s.view_mode, w, h, s.tree.open);
        let inspector = crate::chrome::inspector_subrect(&s, l.panel);
        (s, inspector)
    }

    /// The absolute `(col, row)` of the first column of `want` in the row for
    /// `field`, resolved through the SAME rows + scroll math the painter uses.
    ///
    /// FOCUSES `field` first: the sheet scrolls to keep the focused row on
    /// screen, so this is also how a user reaches a row that is scrolled off.
    fn widget_at(
        s: &mut EditorState,
        inspector: crate::layout::Rect,
        field: crate::state::EntityField,
        want: crate::rowvalue::RowZone,
    ) -> (u16, u16) {
        s.entity_field = field;
        let rows = crate::chrome::inspector_rows(s);
        let idx = rows
            .iter()
            .position(|r| r.action == crate::chrome::RowAction::FocusEntity(field))
            .expect("field has a row");
        let (visible, _) = crate::chrome::inspector_viewport(s, inspector);
        let focused = rows.iter().position(|r| r.focused).unwrap_or(0);
        let scroll = crate::chrome::settings_scroll(focused, rows.len(), visible);
        assert!(
            idx >= scroll && idx < scroll + visible,
            "{field:?} row is visible"
        );
        let ry = inspector.y + crate::chrome::INSPECTOR_ROWS_TOP + (idx - scroll);
        // Lay out at the SAME width the painter and hit-test use.
        let width = inspector.w.saturating_sub(1);
        let n = crate::rowvalue::row_text_in(&rows[idx].label, &rows[idx].value, width)
            .chars()
            .count();
        let off = (0..n)
            .find(|&i| {
                crate::rowvalue::zone_at_in(&rows[idx].label, &rows[idx].value, width, i) == want
            })
            .unwrap_or_else(|| panic!("{field:?} row has no {want:?} zone"));
        // +1 for the focus-marker cell the painter prefixes.
        ((inspector.x + 1 + off) as u16, ry as u16)
    }

    /// Clicking a BOOL row's checkbox TOGGLES it — it does not merely focus.
    #[test]
    fn clicking_a_checkbox_toggles_the_flag() {
        use crate::rowvalue::{RowZone, WidgetPart as W};
        use crate::state::{EntityField as F, EntityKind};
        let (w, h) = (120usize, 40usize);
        let (mut s, inspector) = widget_fixture(EntityKind::Trigger);
        let before = s.level.triggers[0].once;
        let (c, r) = widget_at(&mut s, inspector, F::TrigOnce, RowZone::Widget(W::Toggle));
        apply_mouse(&mut s, press(c, r), w, h);
        assert_ne!(
            s.level.triggers[0].once, before,
            "the checkbox toggled the flag"
        );
        assert_eq!(
            s.entity_field,
            F::TrigOnce,
            "and focused the row it acted on"
        );
        s.undo();
        assert_eq!(s.level.triggers[0].once, before, "ONE undo step");
    }

    /// Clicking `[+]` / `[-]` steps the number — the same semantics as the keys.
    #[test]
    fn clicking_the_steppers_adjusts_the_number() {
        use crate::rowvalue::{RowZone, WidgetPart as W};
        use crate::state::{EntityField as F, EntityKind};
        let (w, h) = (120usize, 40usize);
        let (mut s, inspector) = widget_fixture(EntityKind::Node);
        let before = s.level.nodes[0].threshold;
        let (c, r) = widget_at(&mut s, inspector, F::NodeThreshold, RowZone::Widget(W::Inc));
        apply_mouse(&mut s, press(c, r), w, h);
        assert_eq!(s.level.nodes[0].threshold, before + 1, "[+] stepped up");
        let (c, r) = widget_at(&mut s, inspector, F::NodeThreshold, RowZone::Widget(W::Dec));
        apply_mouse(&mut s, press(c, r), w, h);
        assert_eq!(s.level.nodes[0].threshold, before, "[-] stepped back down");
    }

    /// Clicking the VALUE of a numeric row opens the inline numeric editor.
    #[test]
    fn clicking_a_number_opens_the_inline_editor() {
        use crate::rowvalue::{RowZone, WidgetPart as W};
        use crate::state::{EntityField as F, EntityKind};
        let (w, h) = (120usize, 40usize);
        let (mut s, inspector) = widget_fixture(EntityKind::Node);
        let (c, r) = widget_at(
            &mut s,
            inspector,
            F::NodeThreshold,
            RowZone::Widget(W::Value),
        );
        apply_mouse(&mut s, press(c, r), w, h);
        let ed = s.num_edit.as_ref().expect("numeric editor opened");
        assert_eq!(
            ed.target,
            crate::chrome::RowAction::FocusEntity(F::NodeThreshold)
        );
        assert!(!ed.bounds.is_float(), "threshold is an integer row");
    }

    /// Clicking `[< >]` opens the option popup, on the CURRENT option.
    #[test]
    fn clicking_the_enum_affordance_opens_the_popup() {
        use crate::rowvalue::{RowZone, WidgetPart as W};
        use crate::state::{EntityField as F, EntityKind};
        let (w, h) = (120usize, 40usize);
        let (mut s, inspector) = widget_fixture(EntityKind::Node);
        let (c, r) = widget_at(&mut s, inspector, F::NodeIo, RowZone::Widget(W::Options));
        apply_mouse(&mut s, press(c, r), w, h);
        let p = s.enum_pick.as_ref().expect("option popup opened");
        assert!(!p.options.is_empty());
        assert_eq!(
            p.options[p.sel],
            crate::rowtype::entity_value(&s, F::NodeIo).display(),
            "opens on the current option"
        );
    }

    /// Clicking a COLOUR swatch opens the colour overlay, seeded with the
    /// current colour and the level's used colours.
    #[test]
    fn clicking_a_swatch_opens_the_colour_overlay() {
        use crate::rowvalue::{RowZone, WidgetPart as W};
        use crate::state::{EntityField as F, EntityKind};
        let (w, h) = (120usize, 40usize);
        let (mut s, inspector) = widget_fixture(EntityKind::Node);
        s.level.nodes[0].color_rgb = 0x00_123456;
        let (c, r) = widget_at(&mut s, inspector, F::NodeColor, RowZone::Widget(W::Swatch));
        apply_mouse(&mut s, press(c, r), w, h);
        let e = s.color_edit.as_ref().expect("colour overlay opened");
        assert_eq!(e.original, 0x00_123456, "seeded with the CURRENT colour");
        assert_eq!(e.buf, "123456", "hex field pre-filled");
        assert!(
            e.used.contains(&0x00_123456),
            "the level's own colours are offered"
        );
    }

    /// The label half of a WIDGET row still only focuses — acting must require
    /// hitting the affordance, or a click meant to select a row would edit it.
    #[test]
    fn clicking_the_label_of_a_widget_row_only_focuses() {
        use crate::rowvalue::RowZone;
        use crate::state::{EntityField as F, EntityKind};
        let (w, h) = (120usize, 40usize);
        let (mut s, inspector) = widget_fixture(EntityKind::Trigger);
        let before = s.level.triggers[0].once;
        let (c, r) = widget_at(&mut s, inspector, F::TrigOnce, RowZone::Label);
        apply_mouse(&mut s, press(c, r), w, h);
        assert_eq!(s.level.triggers[0].once, before, "the flag was NOT toggled");
        assert_eq!(s.entity_field, F::TrigOnce, "it focused instead");
        assert!(s.num_edit.is_none() && s.enum_pick.is_none() && s.color_edit.is_none());
    }

    /// Clicking a PROPERTY-SHEET row focuses that row's field (portal —
    /// the 14-field object; the acceptance's "click row N → focused").
    #[test]
    fn inspector_row_click_focuses_entity_field() {
        use crate::state::{EntityField, EntityKind};
        let (w, h) = (120usize, 40usize);
        let mut s = EditorState::new_empty(16, 16, 4);
        s.view_mode = ViewMode::Full2D;
        s.level.portals.push(re_level::LevelPortal {
            name: "p0".into(),
            ..Default::default()
        });
        s.place = Some(EntityKind::Portal);
        s.selected_portal = Some(0);
        s.entity_field = EntityField::PortalAx0; // focus on row 1 → scroll = 0
        let l = layout::compute(s.view_mode, w, h, s.tree.open);
        let inspector = crate::chrome::inspector_subrect(&s, l.panel);
        let (visible, _) = crate::chrome::inspector_viewport(&s, inspector);
        assert!(visible >= 2, "at least two sheet rows visible at 120x40");
        let rows = crate::chrome::inspector_rows(&s);
        // Click the LAST visible row (a later field than the focused one).
        let target = visible - 1;
        let crate::chrome::RowAction::FocusEntity(expected) = rows[target].action else {
            panic!("expected a field row at viewport row {target}");
        };
        assert_ne!(
            s.entity_field, expected,
            "click target differs from current focus"
        );
        let top = inspector.y + crate::chrome::INSPECTOR_ROWS_TOP;
        apply_mouse(
            &mut s,
            press((inspector.x + 1) as u16, (top + target) as u16),
            w,
            h,
        );
        assert_eq!(
            s.entity_field, expected,
            "click focused the clicked row's field"
        );
        assert!(
            s.status.starts_with("focus:"),
            "status reports the new focus"
        );
    }

    /// A sheet-row click in MATERIAL mode focuses the material field.
    #[test]
    fn inspector_row_click_focuses_material_field() {
        let (w, h) = (120usize, 40usize);
        let mut s = EditorState::new_empty(8, 8, 2);
        s.view_mode = ViewMode::Split;
        s.material_mode = true;
        crate::material::new_material(&mut s);
        let l = layout::compute(s.view_mode, w, h, s.tree.open);
        let inspector = crate::chrome::inspector_subrect(&s, l.panel);
        let (visible, _) = crate::chrome::inspector_viewport(&s, inspector);
        assert!(visible >= 3, "header + at least two material rows visible");
        let rows = crate::chrome::inspector_rows(&s);
        // Row 0 is the header; click viewport row 2 (the SECOND field row —
        // scroll is 0 because the focused Fg field sits on row 1).
        let crate::chrome::RowAction::FocusMaterial(expected) = rows[2].action else {
            panic!("expected a material field row");
        };
        assert_ne!(s.material_field, expected);
        let top = inspector.y + crate::chrome::INSPECTOR_ROWS_TOP;
        apply_mouse(
            &mut s,
            press((inspector.x + 2) as u16, (top + 2) as u16),
            w,
            h,
        );
        assert_eq!(
            s.material_field, expected,
            "material focus follows the click"
        );
    }

    /// A click on the sheet's HEADER row is inert (RowAction::None) — no
    /// paint, no focus change, nothing.
    #[test]
    fn inspector_header_row_click_is_inert() {
        use crate::state::{EntityField, EntityKind};
        let (w, h) = (120usize, 40usize);
        let mut s = EditorState::new_empty(16, 16, 4);
        s.view_mode = ViewMode::Full2D;
        s.level.portals.push(re_level::LevelPortal {
            name: "p0".into(),
            ..Default::default()
        });
        s.place = Some(EntityKind::Portal);
        s.selected_portal = Some(0);
        s.entity_field = EntityField::PortalAx0;
        let l = layout::compute(s.view_mode, w, h, s.tree.open);
        let inspector = crate::chrome::inspector_subrect(&s, l.panel);
        let top = inspector.y + crate::chrome::INSPECTOR_ROWS_TOP;
        let f0 = s.entity_field;
        // Row 0 of the viewport = the summary header (scroll 0, focus on row 1).
        apply_mouse(&mut s, press((inspector.x + 1) as u16, top as u16), w, h);
        assert_eq!(s.entity_field, f0, "header click changes no focus");
        assert!(!s.dirty, "header click edits nothing");
    }

    /// The wizard `[X]` close button closes the wizard, and `draw_wizard` now
    /// paints the `[X]` glyph at the shared close-button hotspot.
    #[test]
    fn wizard_close_button_paints_and_closes() {
        let (w, h) = (120usize, 40usize);
        let mut s = EditorState::new_empty(8, 8, 2);
        s.wizard = Some(crate::wizard::Wizard::default());
        // Compute the close-button hotspot via the shared geometry.
        let (x0, y0, pw, _ph) = crate::chrome::wizard_box_geom(w, h).expect("wizard box fits");
        let (bx_y, bx_s, _bx_e) = crate::chrome::modal_close_button(x0, y0, pw);
        // The painter paints "[X]" at (bx_s, bx_y).
        let mut fb = Framebuffer::new(w, h);
        crate::chrome::draw_wizard(&s, &mut fb, w, h);
        assert_eq!(
            fb.get(bx_s, bx_y).map(|c| c.glyph),
            Some('[' as u32),
            "draw_wizard paints the [X] close glyph"
        );
        // Clicking it closes the wizard.
        apply_mouse(&mut s, press(bx_s as u16, bx_y as u16), w, h);
        assert!(s.wizard.is_none(), "wizard [X] click closes the wizard");
    }

    /// The Save-As `[X]` close button paints and a click on it closes the
    /// prompt (mirrors the wizard close-button test). Also proves `draw_saveas`
    /// renders without panicking for a normal-size fb.
    #[test]
    fn saveas_close_button_paints_and_closes() {
        let (w, h) = (120usize, 40usize);
        let mut s = EditorState::new_empty(8, 8, 2);
        crate::saveas::open_saveas(&mut s);
        let (x0, y0, pw, _ph) = crate::chrome::saveas_box_geom(w, h).expect("saveas box fits");
        let (bx_y, bx_s, _bx_e) = crate::chrome::modal_close_button(x0, y0, pw);
        let mut fb = Framebuffer::new(w, h);
        crate::chrome::draw_saveas(&s, &mut fb, w, h);
        assert_eq!(
            fb.get(bx_s, bx_y).map(|c| c.glyph),
            Some('[' as u32),
            "draw_saveas paints the [X] close glyph"
        );
        apply_mouse(&mut s, press(bx_s as u16, bx_y as u16), w, h);
        assert!(s.saveas.is_none(), "Save-As [X] click closes the prompt");
    }

    /// Clicking the Save-As `[Save]` footer button confirms (sets the path +
    /// raises the SAVE flag) and closes the prompt.
    #[test]
    fn saveas_save_button_confirms() {
        let (w, h) = (120usize, 40usize);
        let mut s = EditorState::new_empty(8, 8, 2);
        s.dirty = true;
        crate::saveas::open_saveas(&mut s); // seeds buf = "untitled"
        let (_clr, (svs, _sve), _cancel, frow) =
            crate::chrome::saveas_footer_spans(w, h).expect("footer fits");
        apply_mouse(&mut s, press((svs + 1) as u16, frow as u16), w, h);
        assert!(s.saveas.is_none(), "[Save] closes the prompt on confirm");
        assert_eq!(s.path.as_deref(), Some("levels/untitled.lvl"), "path set");
        assert_eq!(
            s.status, "SAVE",
            "[Save] raises the SAVE flag for the write"
        );
    }

    /// STAGE 3: the wheel over the grid ZOOMS (no longer z-pages); z is untouched.
    #[test]
    fn scroll_over_grid_zooms_not_z() {
        let mut s = EditorState::new_empty(16, 16, 4);
        s.view_mode = crate::state::ViewMode::Full2D;
        let (w, h) = (120usize, 40usize);
        let l = layout::compute(s.view_mode, w, h, false);
        // The draw system would set view_cells each frame; seed it for the zoom math.
        s.view_cells = (
            l.grid.w.saturating_sub(RULER_W),
            l.grid.h.saturating_sub(RULER_H),
        );
        let col = (l.grid.x + RULER_W + 1) as u16;
        let row = (l.grid.y + RULER_H + 1) as u16;
        let z0 = s.z;
        let zoom0 = s.zoom;
        let up = EditorMouse {
            col,
            row,
            kind: EditorMouseKind::ScrollUp,
            button: None,
        };
        apply_mouse(&mut s, up, w, h);
        assert!(s.zoom > zoom0, "scroll up zooms in");
        assert_eq!(s.z, z0, "wheel no longer pages z");
        let down = EditorMouse {
            col,
            row,
            kind: EditorMouseKind::ScrollDown,
            button: None,
        };
        apply_mouse(&mut s, down, w, h);
        assert_eq!(s.zoom, zoom0, "scroll down zooms back out");
        assert_eq!(s.z, z0, "z still untouched");
    }

    /// STAGE 3: a fast LEFT drag across three cells paints EVERY intermediate cell
    /// (gap-free Bresenham), and the whole press+drag collapses into ONE undo step.
    #[test]
    fn fast_drag_paints_all_cells_one_undo_step() {
        let mut s = EditorState::new_empty(16, 16, 4);
        s.brush = CellKind::Floor;
        s.view_mode = crate::state::ViewMode::Full2D;
        let (w, h) = (120usize, 40usize);
        let l = layout::compute(s.view_mode, w, h, false);
        let g = grid_scene_rect(l.grid);
        let zoom = s.zoom as usize;
        let pix = |cx: usize, cy: usize| {
            (
                (g.x + RULER_W + cx * zoom) as u16,
                (g.y + RULER_H + cy * zoom) as u16,
            )
        };
        // Press at (1,1), then a SINGLE fast drag jumping straight to (4,1): the two
        // skipped cells (2,1) and (3,1) must still be painted.
        let (ax, ay) = pix(1, 1);
        apply_mouse(&mut s, press(ax, ay), w, h);
        let (dx, dy) = pix(4, 1);
        apply_mouse(
            &mut s,
            EditorMouse {
                col: dx,
                row: dy,
                kind: EditorMouseKind::Drag,
                button: Some(EditorMouseButton::Left),
            },
            w,
            h,
        );
        apply_mouse(
            &mut s,
            EditorMouse {
                col: dx,
                row: dy,
                kind: EditorMouseKind::Release,
                button: Some(EditorMouseButton::Left),
            },
            w,
            h,
        );
        for x in 1..=4 {
            assert_eq!(
                s.map.cell_at(x, 1, 0),
                CellKind::Floor,
                "cell ({x},1) painted — no gap"
            );
        }
        assert_eq!(s.history.undo_len(), 1, "the whole stroke is ONE undo step");
    }

    /// STAGE 3: a RIGHT drag ERASES along a gap-free line as one undo step.
    #[test]
    fn right_drag_erases_stroke_one_undo_step() {
        let mut s = EditorState::new_empty(16, 16, 4);
        s.view_mode = crate::state::ViewMode::Full2D;
        // Pre-fill a row of solid cells to erase.
        for x in 1..=4 {
            s.map.set(x, 1, 0, CellKind::WallSolid);
        }
        let (w, h) = (120usize, 40usize);
        let l = layout::compute(s.view_mode, w, h, false);
        let g = grid_scene_rect(l.grid);
        let zoom = s.zoom as usize;
        let pix = |cx: usize, cy: usize| {
            (
                (g.x + RULER_W + cx * zoom) as u16,
                (g.y + RULER_H + cy * zoom) as u16,
            )
        };
        let (ax, ay) = pix(1, 1);
        apply_mouse(
            &mut s,
            mouse(
                ax,
                ay,
                EditorMouseKind::Press,
                Some(EditorMouseButton::Right),
            ),
            w,
            h,
        );
        let (dx, dy) = pix(4, 1);
        apply_mouse(
            &mut s,
            mouse(
                dx,
                dy,
                EditorMouseKind::Drag,
                Some(EditorMouseButton::Right),
            ),
            w,
            h,
        );
        apply_mouse(
            &mut s,
            mouse(
                dx,
                dy,
                EditorMouseKind::Release,
                Some(EditorMouseButton::Right),
            ),
            w,
            h,
        );
        for x in 1..=4 {
            assert_eq!(
                s.map.cell_at(x, 1, 0),
                CellKind::Empty,
                "cell ({x},1) erased — no gap"
            );
        }
        assert_eq!(
            s.history.undo_len(),
            1,
            "the whole erase stroke is ONE undo step"
        );
    }

    /// STAGE 3: a MIDDLE drag PANS the view (shifts `pan`) without painting, and a
    /// release clears the pan anchor.
    #[test]
    fn middle_drag_pans_view() {
        let mut s = EditorState::new_empty(64, 64, 4);
        s.view_mode = crate::state::ViewMode::Full2D;
        s.zoom = 1;
        let (w, h) = (120usize, 40usize);
        let l = layout::compute(s.view_mode, w, h, false);
        let g = grid_scene_rect(l.grid);
        // Anchor deep enough inside the SCENE that the upward drag stays in it (the
        // grid now reserves a 1-row caption header at its top).
        let col = (g.x + RULER_W + 10) as u16;
        let row = (g.y + RULER_H + 8) as u16;
        apply_mouse(
            &mut s,
            mouse(
                col,
                row,
                EditorMouseKind::Press,
                Some(EditorMouseButton::Middle),
            ),
            w,
            h,
        );
        assert!(s.last_pan_ptr.is_some(), "middle press anchors the pan");
        // Drag the map to the right+down by a few cells → the view scrolls the
        // opposite way (pan stays clamped >= 0; here we drag toward negative so pan
        // pins to 0), then drag left+up to move pan positive.
        apply_mouse(
            &mut s,
            mouse(
                col - 8,
                row - 4,
                EditorMouseKind::Drag,
                Some(EditorMouseButton::Middle),
            ),
            w,
            h,
        );
        assert!(
            s.pan.0 > 0 || s.pan.1 > 0,
            "dragging the map scrolls the view (pan moved)"
        );
        assert!(!s.dirty, "panning never paints");
        apply_mouse(
            &mut s,
            mouse(
                col - 8,
                row - 4,
                EditorMouseKind::Release,
                Some(EditorMouseButton::Middle),
            ),
            w,
            h,
        );
        assert!(s.last_pan_ptr.is_none(), "release clears the pan anchor");
    }

    // ── rail icons, command palette, validation drawer ───────────────────────

    /// At 100×30 the outliner sits in the RAIL; clicking its icon reopens it
    /// (the accordion organ swap: the inspector pref rails so the outliner can
    /// take the column). Painter and hit-test share `rail_icon_rows`.
    #[test]
    fn rail_icon_click_reopens_the_outliner_organ() {
        let mut s = EditorState::new_empty(8, 8, 2);
        let (w, h) = (100usize, 30usize);
        let lay = s.layout_for(w, h);
        assert!(!lay.rail.is_empty(), "the rail is present in the shed band");
        let eff = s.panels_eff(w);
        assert_eq!(
            eff.outliner,
            layout::PanelState::Rail,
            "outliner starts railed"
        );
        let (_, _, row) = crate::chrome::rail_icon_rows(lay.rail, eff)
            .into_iter()
            .find(|&(id, _, _)| id == layout::PanelId::Outliner)
            .expect("outliner icon on the rail");
        apply_mouse(&mut s, press(lay.rail.x as u16, row as u16), w, h);
        assert_eq!(
            s.panels_eff(w).outliner,
            layout::PanelState::Expanded,
            "the click reopened the outliner (organ swap)"
        );
        assert_eq!(
            s.panels.inspector,
            layout::PanelState::Rail,
            "the inspector pref railed to make room"
        );
        // The swapped-in organ owns the whole right panel now.
        let lay2 = s.layout_for(w, h);
        let out = crate::chrome::outliner_subrect_eff(&s, lay2.panel, s.panels_eff(w));
        assert_eq!(out, lay2.panel, "outliner takes the full panel column");
    }

    /// A click on a COMMAND-PALETTE match row selects + runs it; while the
    /// palette is open no click reaches the grid underneath.
    #[test]
    fn cmd_palette_row_click_executes_and_clicks_never_paint() {
        let mut s = EditorState::new_empty(8, 8, 2);
        let (w, h) = (120usize, 40usize);
        crate::keymap::open_palette(&mut s);
        s.cmd_palette.as_mut().unwrap().query = "help cheat".into();
        // A click outside the box is consumed, never grid paint.
        apply_mouse(&mut s, press(20, 20), w, h);
        assert!(!s.dirty, "clicks are modal while the palette is open");
        assert!(s.cmd_palette.is_some());
        // Click the TOP match row (the help action) → it executes.
        let (x0, y0, _, _) = crate::chrome::cmd_palette_box_geom(w, h).unwrap();
        let row = y0 + 2; // border + query line
        assert_eq!(
            crate::chrome::cmd_palette_row_at(&s, w, h, x0 + 2, row),
            Some(0),
            "hit-test resolves the drawn row"
        );
        apply_mouse(&mut s, press((x0 + 2) as u16, row as u16), w, h);
        assert!(
            s.cmd_palette.is_none(),
            "the click executed + closed the palette"
        );
        assert!(s.help_open, "the help action ran through apply_key");
    }

    /// VALIDATION DRAWER clicks: [X] closes; a click inside the drawer body is
    /// consumed (read-only); a grid click OUTSIDE the drawer still paints.
    #[test]
    fn validation_drawer_click_zones() {
        let mut s = EditorState::new_empty(8, 8, 2);
        let (w, h) = (120usize, 40usize);
        s.validation_open = true;
        s.last_check = vec!["verdict: PASS".into(), "PASS notes — ok".into()];
        // Inside the drawer body: consumed, nothing painted.
        let (x0, y0, _, _) = crate::validate::box_geom(w, h).unwrap();
        apply_mouse(&mut s, press((x0 + 3) as u16, (y0 + 2) as u16), w, h);
        assert!(!s.dirty, "drawer body clicks are inert (read-only v1)");
        assert!(s.validation_open, "a body click does not close the drawer");
        // A grid click OUTSIDE the drawer still edits (past the header + rulers).
        let grid_r = s.layout_for(w, h).grid;
        apply_mouse(
            &mut s,
            press(
                (grid_r.x + RULER_W + 1) as u16,
                (grid_r.y + 1 + RULER_H + 1) as u16,
            ),
            w,
            h,
        );
        assert!(s.dirty, "editing keeps working beside the open drawer");
        // The [X] button closes it.
        let (brow, bx, _) = crate::validate::close_button(w, h).unwrap();
        apply_mouse(&mut s, press(bx as u16, brow as u16), w, h);
        assert!(!s.validation_open, "[X] closes the drawer");
    }

    // ─── CRASH-RECOVERY OFFER: modal to the MOUSE too ─────────────────────────

    /// Build a state with a pending crash-recovery offer over a real on-disk
    /// original + sidecar, exactly as `open::load_path` would have armed it.
    /// Returns the state and both fixture guards (which delete themselves).
    fn offered_state(
        tag: &str,
    ) -> (
        EditorState,
        crate::testutil::ScratchFile,
        crate::testutil::ScratchFile,
    ) {
        use crate::testutil::ScratchFile;
        use re_world::CellKind;
        let mut a = EditorState::new_empty(4, 4, 2);
        a.map.set(1, 1, 0, CellKind::WallSolid);
        let orig = ScratchFile::write(
            crate::testutil::scratch_path(&format!("h4m-{tag}-orig.lvl")),
            &crate::io::capture_and_serialize(&mut a),
        );
        let mut b = EditorState::new_empty(4, 4, 2);
        b.map.set(2, 2, 0, CellKind::WallSolid);
        let side = ScratchFile::write(
            crate::testutil::scratch_path(&format!("h4m-{tag}-side.lvl")),
            &crate::io::capture_and_serialize(&mut b),
        );
        let mut s = EditorState::new_empty(4, 4, 2);
        assert!(
            crate::open::load_path(&mut s, orig.path()),
            "original loads"
        );
        s.autosave_offer = Some(side.path().to_string());
        (s, orig, side)
    }

    /// THE DATA-LOSS REGRESSION GUARD: making the crash-recovery offer modal to
    /// the keyboard is not enough on its own — the mouse must be guarded too, or
    /// a left click on the toolbar `[Save]` (a button the prompt does not even
    /// cover) raises the SAVE flag, and the save path writes the UN-RECOVERED
    /// document over the original and calls `io::delete_autosave`, destroying
    /// the sidecar — the same data loss reachable through the mouse instead of
    /// the keyboard.
    ///
    /// This is the mouse twin of
    /// `open::tests::autosave_offer_is_modal_and_survives_a_stray_key`.
    #[test]
    fn a_toolbar_save_click_cannot_reach_the_save_path_while_an_offer_is_pending() {
        let (w, h) = (120usize, 40usize);
        let (mut s, _orig, side) = offered_state("save-click");
        s.status = "idle".into();

        // The middle of the toolbar's [Save] button — same construction as
        // `toolbar_click_sets_view_mode`.
        let spans = toolbar::button_spans();
        let l = layout::compute(s.view_mode, w, h, false);
        let tb = l.toolbar;
        assert!(!tb.is_empty(), "the toolbar is on screen at 120x40");
        let (_, bs, be) = spans
            .iter()
            .copied()
            .find(|&(b, _, _)| b == ToolbarButton::Save)
            .expect("[Save] is on the toolbar");
        let col = (tb.x + usize::midpoint(bs, be)) as u16;
        // Premise: this really is the [Save] button's cell (it fires without an offer).
        let mut control = EditorState::new_empty(4, 4, 2);
        apply_mouse(&mut control, press(col, tb.y as u16), w, h);
        assert_eq!(control.status, "SAVE", "premise: that cell IS [Save]");

        apply_mouse(&mut s, press(col, tb.y as u16), w, h);
        assert_ne!(s.status, "SAVE", "the click never reached the save path");
        assert_eq!(
            s.autosave_offer.as_deref(),
            Some(side.path()),
            "the offer is still pending — the click answered nothing"
        );
        assert!(
            std::path::Path::new(side.path()).exists(),
            "the sidecar is still on disk"
        );
    }

    /// While an offer is pending NO click paints the grid either — a press over
    /// the 2D grid is consumed, not turned into an edit.
    #[test]
    fn a_grid_click_under_a_pending_offer_paints_nothing() {
        let (w, h) = (120usize, 40usize);
        let (mut s, _orig, _side) = offered_state("grid-click");
        s.view_mode = ViewMode::Full2D;
        s.dirty = false;
        let l = layout::compute(s.view_mode, w, h, false);
        let col = (l.grid.x + 6) as u16;
        let row = (l.grid.y + RULER_H + 2) as u16;
        apply_mouse(&mut s, press(col, row), w, h);
        assert!(!s.dirty, "a modal click never leaks into a grid paint");
        assert!(s.autosave_offer.is_some(), "and the offer still stands");
    }

    /// `[ Restore ]` loads the sidecar's content into the document, keeps `path`
    /// on the ORIGINAL, and marks it dirty — the same contract the `y` key has.
    #[test]
    fn clicking_restore_restores_the_sidecar() {
        use re_world::CellKind;
        let (w, h) = (120usize, 40usize);
        let (mut s, orig, side) = offered_state("restore");
        let lay = crate::chrome::autosave_layout(w, h).expect("overlay fits at 120x40");
        let col = usize::midpoint(lay.restore.0, lay.restore.1) as u16;
        apply_mouse(&mut s, press(col, lay.button_row as u16), w, h);

        assert_eq!(
            s.map.cell_at(2, 2, 0),
            CellKind::WallSolid,
            "document content replaced by the sidecar's"
        );
        assert_eq!(
            s.path.as_deref(),
            Some(orig.path()),
            "path stays the ORIGINAL"
        );
        assert!(
            s.dirty,
            "restored content is unsaved until an explicit save"
        );
        assert_eq!(s.autosave_offer, None, "offer answered");
        drop(side);
    }

    /// `[ Discard ]` clears the offer, leaves the document as opened, and keeps
    /// the sidecar FILE on disk — the same contract the `n` key has.
    #[test]
    fn clicking_discard_clears_the_offer_and_keeps_the_sidecar() {
        use re_world::CellKind;
        let (w, h) = (120usize, 40usize);
        let (mut s, _orig, side) = offered_state("discard");
        let lay = crate::chrome::autosave_layout(w, h).expect("overlay fits at 120x40");
        let col = usize::midpoint(lay.discard.0, lay.discard.1) as u16;
        apply_mouse(&mut s, press(col, lay.button_row as u16), w, h);

        assert_eq!(s.autosave_offer, None, "offer answered");
        assert_eq!(
            s.map.cell_at(1, 1, 0),
            CellKind::WallSolid,
            "the document is the one that was opened"
        );
        assert!(!s.dirty, "discard leaves a clean document clean");
        assert!(
            std::path::Path::new(side.path()).exists(),
            "discard keeps the sidecar file, as the overlay promises"
        );
        assert_eq!(s.status, crate::open::AUTOSAVE_DISCARDED_STATUS);
    }

    /// A click INSIDE the overlay but on neither button is consumed and answers
    /// nothing — the prompt does not disappear because you missed.
    #[test]
    fn a_click_inside_the_overlay_but_off_the_buttons_answers_nothing() {
        let (w, h) = (120usize, 40usize);
        let (mut s, _orig, side) = offered_state("miss");
        let lay = crate::chrome::autosave_layout(w, h).expect("overlay fits");
        apply_mouse(&mut s, press(lay.ix as u16, lay.lead_row as u16), w, h);
        assert_eq!(
            s.autosave_offer.as_deref(),
            Some(side.path()),
            "a miss leaves the question standing"
        );
        assert!(!s.dirty, "and edits nothing");
    }

    /// The WHEEL scrolls the open help sheet. `apply_modal_click` used to drop
    /// every non-left-press event over an open modal, so the first gesture a
    /// reader reaches for on a 185-line document did nothing at all.
    #[test]
    fn the_wheel_scrolls_the_open_help_sheet() {
        let mut s = EditorState::new_empty(8, 8, 2);
        s.open_help();
        s.help_page = 35;
        let wheel = |kind| EditorMouse {
            kind,
            button: None,
            col: 40,
            row: 12,
        };
        apply_mouse(&mut s, wheel(EditorMouseKind::ScrollDown), 120, 40);
        assert_eq!(s.help_scroll, 3, "a notch scrolls three lines down");
        apply_mouse(&mut s, wheel(EditorMouseKind::ScrollDown), 120, 40);
        assert_eq!(s.help_scroll, 6);
        apply_mouse(&mut s, wheel(EditorMouseKind::ScrollUp), 120, 40);
        assert_eq!(s.help_scroll, 3, "and back up");
        for _ in 0..200 {
            apply_mouse(&mut s, wheel(EditorMouseKind::ScrollDown), 120, 40);
        }
        let last_page = crate::keymap::help_lines().len() - 35;
        assert_eq!(
            s.help_scroll, last_page,
            "the wheel clamps like every other scroll"
        );
        assert!(s.help_open, "and never leaks past the modal");
    }

    /// The toolbar `[Help]` button opens the sheet through the SAME reset path
    /// the `?` key uses — a button that reopened a scrolled, filtered sheet
    /// would show the user a screen of nothing.
    #[test]
    fn the_toolbar_help_button_opens_a_clean_sheet() {
        let mut s = EditorState::new_empty(8, 8, 2);
        apply_toolbar(&mut s, ToolbarButton::Help);
        assert!(s.help_open, "[Help] opens the sheet");
        s.help_scroll = 90;
        s.help_filter = "vox".into();
        apply_toolbar(&mut s, ToolbarButton::Help);
        assert!(!s.help_open, "[Help] closes it again");
        apply_toolbar(&mut s, ToolbarButton::Help);
        assert!(s.help_open);
        assert_eq!(s.help_scroll, 0, "reopened at the top");
        assert_eq!(s.help_filter, "", "and unfiltered");
    }
}
