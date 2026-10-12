//! Pure editor key→state transitions (unit-testable without a terminal).
//!
//! Split of responsibilities:
//! * `edit::apply_key` owns the **2D edit surface** keys only — cursor move
//!   (arrows), z up/down (PageUp/PageDown), paint (space), erase (x), brush
//!   hotkeys (1-6), node tag (n), save (s), and view-mode cycle (Tab).
//! * `camera::fly` / `jump_to_cursor` / noclip-toggle own the **3D preview**
//!   keys (W/A/S/D/R/F, `,`/`.`, G/Enter, Shift+N) and live in
//!   `camera_input_system` (it needs `Res<Time>` for the frame `dt`).
//!
//! The two key sets are disjoint, so a single `EditorKey` is routed to exactly
//! one of them. Camera keys never paint; edit keys never move the camera.
use crate::entity;
use crate::fill;
use crate::material;
use crate::state::{cycle_shape, shape_label, EditorState, ToolMode};
use crate::tree;
use crate::view;
use crate::vox;
use crate::wizard;
use re_core::Shape;
use re_present::{EditorKey, EditorKeyCode as E};
use re_world::CellKind;

/// Apply the ACTIVE TOOL at the current cursor cell — exactly what the keyboard
/// `space` does, routed by mode so the mouse and the keyboard share one code
/// path (no duplicated tool logic):
///
/// * vox mode      → stamp/re-select the selected `.vox` model,
/// * material mode → paint the selected material id,
/// * object mode   → place/select the current entity,
/// * stair tool    → stamp a whole climbable staircase (straight/spiral) at once,
/// * shape brush   → paint a 45° wedge/ramp (solid base + [`Shape`]),
/// * otherwise     → paint the plain geometry brush (clearing any prior shape).
///
/// Each branch already marks dirty + sets a status line. Pure; no I/O.
///
/// THE shared geometry/entity mutation choke point: it snapshots the document for
/// UNDO ([`EditorState::push_undo`]) before applying, so every paint/place — from
/// the keyboard `space` OR a single mouse click — is undoable as one step. This is
/// `push_undo()` + [`paint_tool_at_cursor`]; a mouse DRAG snapshots once
/// at stroke start and then calls [`paint_tool_at_cursor`] per cell so the whole
/// drag collapses into one undo entry.
pub fn apply_tool_at_cursor(state: &mut EditorState) {
    state.push_undo();
    paint_tool_at_cursor(state);
}

/// Apply the ACTIVE TOOL at the current cursor cell WITHOUT taking an undo
/// snapshot — the mode-dispatched paint/place/stamp body shared by the keyboard
/// `space` (via [`apply_tool_at_cursor`], which snapshots first) and the mouse
/// DRAG path (which snapshots ONCE at stroke start and then calls this
/// per painted cell). Keep this signature STABLE — the mouse drag path binds to
/// it. Pure; no I/O. Each branch marks dirty + sets a status line.
pub fn paint_tool_at_cursor(state: &mut EditorState) {
    if state.vox_mode {
        vox::place_at_cursor(state);
    } else if state.material_mode {
        material::paint_id(state);
    } else if state.place.is_some() {
        entity::place_at_cursor(state);
    } else if state.stair_mode {
        // The staircase stamp tool: one action lays a whole climbable straight or
        // spiral staircase at the cursor via the re_level prefab generator. The
        // snapshot above makes the WHOLE stamp a single undo step.
        crate::staircase::stamp_at_cursor(state);
    } else {
        let (x, y, z) = (state.cursor.0, state.cursor.1, state.z);
        if state.shape_mode {
            // A 45° wedge/ramp is a SOLID base cell carrying a non-Full Shape:
            // paint both so it round-trips and casts correctly.
            state.map.set(x, y, z, CellKind::WallSolid);
            state.map.set_shape(x, y, z, state.shape_sel);
            state.mark_dirty();
            state.status = format!(
                "painted {} @ {},{},{}",
                shape_label(state.shape_sel),
                x,
                y,
                z
            );
        } else {
            state.map.set(x, y, z, state.brush);
            // A plain brush stroke clears any prior 45° shape on the cell.
            state.map.set_shape(x, y, z, Shape::Full);
            state.mark_dirty();
            state.status = format!("painted {:?} @ {},{},{}", state.brush, x, y, z);
        }
    }
}

/// Does the ACTIVE TOOL legitimately repeat per cell along a mouse DRAG?
///
/// STROKE tools — the geometry/shape brush and the material paint — answer
/// `true`: dragging them across N cells paints N cells (that is what a brush
/// stroke IS). PLACEMENT tools — the object selector, the `.vox` stamp, and the
/// staircase stamp — answer `false`: they fire ON PRESS ONLY, because repeating
/// a placement per Bresenham cell spawns one object per cell dragged over (the
/// drag-spawn bug). The branch order mirrors [`paint_tool_at_cursor`]'s dispatch
/// exactly, so the answer always describes the branch a paint would take.
pub const fn tool_paints_on_drag(state: &EditorState) -> bool {
    if state.vox_mode {
        false // vox stamp: placement — press only
    } else if state.material_mode {
        true // material paint: a stroke tool
    } else if state.place.is_some() || state.stair_mode {
        false // object placement / staircase stamp: placement — press only
    } else {
        true // geometry / shape brush: THE stroke tool
    }
}

/// Apply one 2D-edit key to the editor state. Pure; no I/O. Camera/fly keys are
/// handled separately by `camera_input_system` (they need frame `dt`).
pub fn apply_key(state: &mut EditorState, key: EditorKey) {
    // Any non-Esc keypress disarms the dirty-quit guard (Esc itself is handled in
    // `quit_on_esc`, never routed here). So an armed "Esc again to quit" is reset
    // by any other action the user takes.
    if key.code != E::Esc {
        state.quit_armed = false;
    }
    // Likewise any non-F5 key disarms the playtest warn-and-confirm —
    // only an immediate second F5 launches over a lvlcheck FAIL.
    if key.code != E::F(5) {
        state.playtest_armed = false;
    }
    // PREVIEW FREE-LOOK is modal over the 2D-edit surface: while `preview_captured`,
    // the editor behaves like the first-person game — WASD walks, `s` moves BACK
    // (NOT save), arrows/`,`/`.` turn, `c` toggles collisions — all handled by
    // `camera_input_system` / `held_preview_mover` in `main.rs`. So suppress EVERY
    // 2D-edit key here: a navigation keystream must never paint, save, change z, or
    // open a modal. Esc is never routed to `apply_key` (it is handled in
    // `quit_on_esc`, which releases capture first), so capture is always escapable.
    if state.preview_captured {
        return;
    }
    // ── CRASH-RECOVERY OFFER ── a MODAL prompt at the very front of the
    // chain: it is set by `open::load_path` when this document's autosave
    // sidecar is strictly newer than the file just opened, and it owns the
    // keyboard until it is answered. `y` restores the sidecar into the editor
    // (path stays the original, document marked dirty — see
    // `open::restore_autosave`); `n` discards the offer (the sidecar file is
    // left on disk); Esc dismisses it via `main::quit_on_esc` →
    // `open::dismiss_autosave_offer`.
    //
    // This prompt must be FULLY MODAL, not transient: a transient prompt that
    // clears itself on any other key would let `s` right after a crash-restart
    // both discard the recovery offer AND (via the save path) delete the
    // sidecar, losing the recovered work for good. So every non-answer key is
    // SWALLOWED and re-states the question instead.
    //
    // DISPATCH SENTINELS: the BEGIN/END markers below fence exactly this block
    // for `keymap.rs`'s `every_dispatched_key_is_in_the_keymap`, which scans
    // the fence for key literals and asserts each has a
    // `Context::AutosaveOffer` row in `KEYMAP` (else the key is invisible in
    // the hint bar, the `?` sheet and the `:` palette). Deleting a sentinel
    // FAILS that test rather than silently disabling it.
    if state.autosave_offer.is_some() {
        // AUTOSAVE-KEYS-BEGIN
        match key.code {
            E::Char('y' | 'Y') => crate::open::restore_autosave(state),
            E::Char('n' | 'N') => crate::open::discard_autosave_offer(state),
            _ => state.status = "AUTOSAVE found (newer) — Y restore / N discard".into(),
        }
        return;
        // AUTOSAVE-KEYS-END
    }
    // The MASK-EDIT modal, when open, captures input modally: arrows move the
    // part-local `(lx, lz)` cursor within the active h-layer, `[`/`]` change the
    // active h-layer (`ly`), `space` sets the voxel SOLID and `x` sets it EMPTY,
    // and `q` exits. Checked before every other modal so it owns the keyboard while
    // authoring a part's mask. All mask edits route through `part.rs` (SSOT).
    if state.mask_mode {
        apply_mask_key(state, key);
        return;
    }
    // ── FOCUS STACK ── F2 RENAME, the HELP popup, and the DIRECTION PICKER
    // dispatch through the input-context stack (`crate::focus`), which walks
    // the open frames top-down in a fixed priority order (rename → help →
    // dir-picker) and consumes the key at the topmost open frame. Every other
    // modal below keeps its own guard + priority in this chain.
    if crate::focus::route_key(state, key) {
        return;
    }
    // The LOAD/OPEN overlay, when open, captures nav modally: Up/Dn move, Enter
    // loads, `O` closes (Esc also closes via `quit_on_esc`).
    if state.open.is_some() {
        apply_open_key(state, key);
        return;
    }
    // The SAVE-AS prompt, when open, captures input modally: printable chars type
    // into the filename buffer, Enter confirms (then triggers the write), Esc
    // cancels (via `quit_on_esc`).
    if state.saveas.is_some() {
        apply_saveas_key(state, key);
        return;
    }
    // The NEW-LEVEL wizard, when open, captures input modally: Tab focus, +/-
    // adjust, Enter create, Esc cancel. Nothing else edits while it is up.
    if state.wizard.is_some() {
        apply_wizard_key(state, key);
        return;
    }
    // The LEVEL SETTINGS form, when open, captures input modally: Tab/arrows
    // focus + navigate, text fields type, list sections add/remove rows/names,
    // Enter applies, Esc discards (via `quit_on_esc`).
    if state.settings.is_some() {
        apply_settings_key(state, key);
        return;
    }
    // ── FUNCTION-KEY COMMANDS (rename / playtest) ──  Handled BEFORE
    // the palette/tree guards so `F2` renames the focused outliner row and `F5`
    // playtests from anywhere; the text-entry modals above never consume F-keys,
    // so nothing is stolen from typing.
    match key.code {
        E::F(2) => {
            crate::rename::open(state);
            return;
        }
        E::F(5) => {
            request_playtest(state);
            return;
        }
        _ => {}
    }
    // The object-TYPE SELECTOR palette captures nav while open (Up/Dn move,
    // Enter pick, `i` close); checked before the tree so only one is focused.
    if state.palette.is_some() {
        apply_palette_key(state, key);
        return;
    }
    // The OBJECT TREE / OUTLINER captures nav while focused (Up/Dn move, Enter
    // select+jump, `h` fold, `z` duplicate, `d` delete, `t` close).
    if state.tree.open {
        apply_tree_key(state, key);
        return;
    }
    // ── TEXT AUTHORING ── while a text field is focused (note body /
    // inset child path / node pool words / trigger event label), keys route to
    // the caret-based text-entry router: printable chars insert AT THE CARET,
    // Backspace deletes before it, Enter adds a newline (note body only), and
    // plain arrows navigate. The router consumes only what it handles —
    // Shift+arrows keep panning, Tab keeps cycling, PageUp/Dn keep z-paging —
    // so no existing keybinding changes when no text field is focused.
    if entity::text_entry_key(state, key) {
        return;
    }
    let (sx, sy, sz) = state.map.size();
    // SHIFT+ARROW pans the top-down window by one cell (clamped to the world);
    // plain arrows move the edit cursor. The pan clamp mirrors `view::follow_pan`:
    // the furthest pan still shows real cells is `map - visible`, where `visible`
    // is the whole-cell window for the current zoom.
    match key.code {
        E::Left if key.shift => {
            state.pan.0 = state.pan.0.saturating_sub(1);
        }
        E::Right if key.shift => {
            let vis = (state.view_cells.0 / (state.zoom.max(1) as usize)).max(1);
            let max_pan = sx.saturating_sub(vis);
            state.pan.0 = (state.pan.0 + 1).min(max_pan);
        }
        E::Up if key.shift => {
            state.pan.1 = state.pan.1.saturating_sub(1);
        }
        E::Down if key.shift => {
            let vis = (state.view_cells.1 / (state.zoom.max(1) as usize)).max(1);
            let max_pan = sy.saturating_sub(vis);
            state.pan.1 = (state.pan.1 + 1).min(max_pan);
        }
        E::Left => state.cursor.0 = state.cursor.0.saturating_sub(1),
        E::Right => state.cursor.0 = (state.cursor.0 + 1).min(sx.saturating_sub(1)),
        E::Up => state.cursor.1 = state.cursor.1.saturating_sub(1),
        E::Down => state.cursor.1 = (state.cursor.1 + 1).min(sy.saturating_sub(1)),
        E::PageUp => state.z = (state.z + 1).min(sz.saturating_sub(1)),
        E::PageDown => state.z = state.z.saturating_sub(1),
        // `<` / `>` page the active z by a WHOLE storey (floor-to-floor), a
        // convenience over PageUp/Down's single-layer step. Clamped to the world.
        // (`,`/`.` stay the camera turn keys; the shifted glyphs are distinct.)
        E::Char('>') => {
            let step = state.level.storey_height() as usize;
            state.z = (state.z + step).min(sz.saturating_sub(1));
            let floor = state.z as u32 / state.level.storey_height();
            state.status = format!("floor {floor} (z={})", state.z);
        }
        E::Char('<') => {
            let step = state.level.storey_height() as usize;
            state.z = state.z.saturating_sub(step);
            let floor = state.z as u32 / state.level.storey_height();
            state.status = format!("floor {floor} (z={})", state.z);
        }
        // `+`/`=`/`-` adjust the SELECTED entity's numeric property when the
        // object selector is active (node threshold / pedestal volume); otherwise
        // they drive the top-down integer zoom. Branching on `place` keeps the
        // two meanings disjoint without a new key.
        E::Char('+') | E::Char('=') => {
            if state.vox_mode {
                state.push_undo_checkpoint();
                vox::adjust_scale(state, 1);
                state.drop_undo_if_unchanged();
            } else if state.material_mode {
                state.push_undo_checkpoint();
                material::adjust_field(state, 1);
                state.drop_undo_if_unchanged();
            } else if state.place.is_some() {
                state.push_undo_checkpoint();
                entity::adjust_property(state, 1);
                state.drop_undo_if_unchanged();
            } else if state.stair_mode {
                // In the staircase tool the spiral RADIUS is grown/shrunk with +/-
                // (mirrors vox scale). This is a TOOL setting, not a map edit — no
                // snapshot; the stamp itself (space) is the undoable action.
                state.stair_radius = (state.stair_radius + 1).max(1);
                state.status = format!("stair {}", crate::staircase::summary(state));
            } else {
                // Pure navigation (zoom): NOT an edit — no snapshot. Anchor the
                // zoom around the cursor cell so it stays put on screen.
                let (sx, sy, _sz) = state.map.size();
                let (z, p) = view::zoom_in_at(
                    state.zoom,
                    state.pan,
                    state.cursor,
                    state.view_cells,
                    (sx, sy),
                );
                state.zoom = z;
                state.pan = p;
                state.status = format!("zoom x{}", state.zoom);
            }
        }
        E::Char('-') => {
            if state.vox_mode {
                state.push_undo_checkpoint();
                vox::adjust_scale(state, -1);
                state.drop_undo_if_unchanged();
            } else if state.material_mode {
                state.push_undo_checkpoint();
                material::adjust_field(state, -1);
                state.drop_undo_if_unchanged();
            } else if state.place.is_some() {
                state.push_undo_checkpoint();
                entity::adjust_property(state, -1);
                state.drop_undo_if_unchanged();
            } else if state.stair_mode {
                state.stair_radius = (state.stair_radius - 1).max(1);
                state.status = format!("stair {}", crate::staircase::summary(state));
            } else {
                // Pure navigation (zoom): anchor around the cursor cell.
                let (sx, sy, _sz) = state.map.size();
                let (z, p) = view::zoom_out_at(
                    state.zoom,
                    state.pan,
                    state.cursor,
                    state.view_cells,
                    (sx, sy),
                );
                state.zoom = z;
                state.pan = p;
                state.status = format!("zoom x{}", state.zoom);
            }
        }
        // `m` enters / cycles MATERIAL mode forward; `M` cycles backward. Entering
        // material mode leaves the object selector (the two are exclusive).
        E::Char('m') => {
            state.enter_tool_mode(ToolMode::Material);
            material::cycle_material(state, 1);
        }
        E::Char('M') => {
            state.enter_tool_mode(ToolMode::Material);
            material::cycle_material(state, -1);
        }
        // `e` creates a NEW material (unique name, default look) and selects it.
        E::Char('e') => {
            state.push_undo();
            state.enter_tool_mode(ToolMode::Material);
            material::new_material(state);
        }
        // `b` cycles which material-def FIELD the `+`/`-`/`c` keys edit (through
        // the full exhaustive list: colours, glyph, recipe + per-variant params,
        // animation, rank, surface, mirror, preset).
        E::Char('b') if state.material_mode => material::cycle_field(state),
        // `v` enters / cycles VOX (box-model) mode: select a `.vox` file, place
        // it as a `[vox]` stamp at the cursor with `space`.
        E::Char('v') => {
            state.enter_tool_mode(ToolMode::Vox);
            vox::cycle_vox(state);
        }
        // `o` enters / cycles the OBJECT SELECTOR (Door→Spawn→Pedestal→Node).
        E::Char('o') => {
            state.enter_tool_mode(ToolMode::Object);
            entity::cycle_selector(state);
        }
        // `P` starts the PIPE-CREATE two-node connect flow (pick source node, then
        // target). Pressing it again while a source is pending CANCELS; otherwise it
        // enters Pipe placement mode and (if the cursor is on a node) picks the
        // source. The TARGET is the next `space` on a node (or tree Enter on a node).
        E::Char('P') => {
            if crate::pipe::cancel(state) {
                // Was pending → cancelled (status set by `cancel`).
            } else {
                state.enter_tool_mode(ToolMode::Object);
                state.place = Some(crate::state::EntityKind::Pipe);
                state.entity_field =
                    crate::state::entity_field_default(crate::state::EntityKind::Pipe)
                        .unwrap_or(state.entity_field);
                crate::pipe::start_pipe_create(state);
            }
        }
        // `g` SETS / CLEARS the selected pipe's `switch_at` cell at the cursor (a
        // path cell). A toggle, like `c` on the PipeSwitch field — bound to its own
        // key so it works whatever field is focused.
        E::Char('g') if state.place == Some(crate::state::EntityKind::Pipe) => {
            state.push_undo();
            let prev = state.entity_field;
            state.entity_field = crate::state::EntityField::PipeSwitch;
            crate::pipe::cycle_enum(state);
            state.entity_field = prev;
        }
        // ENTER opens the FOCUSED inspector row's editor: the
        // inline numeric editor for an Int/Float row, the option popup for an
        // Enum, the colour overlay for a Colour. A Bool row toggles in place
        // (same code path as `c`); a plain Text row has no editor and Enter is
        // inert there.
        //
        // Enter (and the row's own click target) are the ONLY entry points on
        // purpose: bare digits stay bound to the geometry brushes at the root,
        // so `1`-`6`/`9`/`0` never get stolen by a number field.
        E::Enter => {
            let action = crate::chrome::focused_row_action(state);
            match crate::rowedit::value_of(state, action) {
                Some(crate::rowvalue::RowValue::Bool(_)) => {
                    crate::rowedit::toggle_bool(state, action);
                }
                Some(v) if v.is_widget() => {
                    crate::rowedit::open_for(state, action);
                }
                _ => {}
            }
        }
        // `t` toggles the OBJECT TREE / OUTLINER panel (open=focused / close).
        E::Char('t') => tree::toggle(state),
        // `i` toggles the object-TYPE SELECTOR palette (insert/create menu).
        E::Char('i') => tree::toggle_palette(state),
        // `c` cycles the active mode's ENUM property: vox surface material; else
        // the focused MATERIAL enum field (glyph/recipe/anim/surface/mirror/preset);
        // else the active entity's property (Spawn facing, Node io role, focused
        // emitter enum field).
        E::Char('c') if state.vox_mode => {
            state.push_undo_checkpoint();
            vox::cycle_material(state);
            state.drop_undo_if_unchanged();
        }
        E::Char('c') if state.material_mode => {
            state.push_undo_checkpoint();
            material::cycle_enum(state, 1);
            state.drop_undo_if_unchanged();
        }
        E::Char('c') => {
            state.push_undo_checkpoint();
            entity::cycle_property(state);
            state.drop_undo_if_unchanged();
        }
        // `;` / `'` page which FOCUSED FIELD the +/- and c keys target for the
        // active object. For an Emitter this is the emitter parameter list; for
        // Door/Node/Pedestal/Spawn it is that entity's exhaustive field list
        // (pos, glyph, color, io, threshold, anchor, pool, targets, volume, …).
        // In the staircase tool `;`/`'` step the spiral TURN count (kept disjoint
        // from the radius +/- keys). A tool setting, not a map edit — no snapshot.
        E::Char(';') if state.stair_mode => {
            state.stair_turns = state.stair_turns.saturating_add(1).max(1);
            state.status = format!("stair {}", crate::staircase::summary(state));
        }
        E::Char('\'') if state.stair_mode => {
            state.stair_turns = state.stair_turns.saturating_sub(1).max(1);
            state.status = format!("stair {}", crate::staircase::summary(state));
        }
        E::Char(';') => entity::cycle_entity_field(state, 1),
        E::Char('\'') => entity::cycle_entity_field(state, -1),
        E::Char(' ') => apply_tool_at_cursor(state),
        E::Char('x') if state.vox_mode => {
            // Vox mode: delete the vox stamp nearest the cursor.
            state.push_undo();
            vox::delete_at_cursor(state);
        }
        E::Char('x') if state.material_mode => {
            // Material mode: clear the cursor cell's material id back to default.
            state.push_undo();
            material::clear_id(state);
        }
        E::Char('x') if state.place.is_some() => {
            // Object-placement mode: delete the entity under the cursor.
            state.push_undo();
            entity::delete_at_cursor(state);
        }
        E::Char('x') => {
            state.push_undo();
            let (x, y, z) = (state.cursor.0, state.cursor.1, state.z);
            state.map.set(x, y, z, CellKind::Empty);
            // Erase also resets geometry back to Full (no orphan shape on air).
            state.map.set_shape(x, y, z, Shape::Full);
            state.mark_dirty();
        }
        // brush hotkeys — selecting a plain geometry brush leaves shape mode AND
        // the object selector (back to painting geometry).
        E::Char('1') => {
            state.enter_tool_mode(ToolMode::Brush);
            state.brush = CellKind::WallSolid;
        }
        E::Char('2') => {
            state.enter_tool_mode(ToolMode::Brush);
            state.brush = CellKind::Floor;
        }
        E::Char('3') => {
            state.enter_tool_mode(ToolMode::Brush);
            state.brush = CellKind::Ceiling;
        }
        E::Char('4') => {
            state.enter_tool_mode(ToolMode::Brush);
            state.brush = CellKind::WallTrace;
        }
        E::Char('5') => {
            state.enter_tool_mode(ToolMode::Brush);
            state.brush = CellKind::WallMatrix;
        }
        E::Char('6') => {
            state.enter_tool_mode(ToolMode::Brush);
            state.brush = CellKind::Object;
        }
        // `9` selects the HOLOGRAM cell brush (a pass-through transparent cell,
        // glyph `H`). Like `1`-`6` it is a plain geometry brush, so it leaves shape
        // mode AND the object selector.
        E::Char('9') => {
            state.enter_tool_mode(ToolMode::Brush);
            state.brush = CellKind::Hologram;
        }
        // `0` selects the WARP-PORTAL cell brush (a walkable pass-through cell,
        // glyph `@`). Like `1`-`6`/`9` it is a plain geometry brush, so it leaves
        // shape mode AND the object selector. (`1`-`9` are taken; `0` is free.)
        E::Char('0') => {
            state.enter_tool_mode(ToolMode::Brush);
            state.brush = CellKind::Portal;
        }
        // `7` selects the 45° SHAPE brush; `[`/`]` cycle the orientation through
        // the 4 wedges + 4 ramps. The readout shows the current orientation.
        E::Char('7') => {
            state.enter_tool_mode(ToolMode::Shape);
            state.status = format!("shape brush: {}", shape_label(state.shape_sel));
        }
        // `8` selects the STAIRCASE STAMP tool; `\` toggles straight ↔ spiral.
        // `[`/`]` then cycle the straight run's rise direction (mirroring the shape
        // brush's orientation keys); in either sub-mode `space` stamps the whole
        // staircase at the cursor.
        E::Char('8') => {
            state.enter_tool_mode(ToolMode::Stair);
            state.status = format!("stair tool: {}", crate::staircase::summary(state));
        }
        E::Char('\\') if state.stair_mode => {
            state.stair_sub = state.stair_sub.toggled();
            state.status = format!("stair {}", crate::staircase::summary(state));
        }
        // `#` toggles the AUTO-CARVE STAIRWELL option while in the staircase tool:
        // when ON (default), stamping a staircase also punches a stairwell hole
        // through the floor slab(s) above its footprint so the player can climb
        // through. A tool setting, not a map edit — no snapshot here; the stamp
        // (space) is the undoable action that includes the carve.
        E::Char('#') if state.stair_mode => {
            state.auto_carve_stairwell = !state.auto_carve_stairwell;
            state.status = format!(
                "stairwell auto-carve {}",
                if state.auto_carve_stairwell {
                    "ON"
                } else {
                    "OFF"
                }
            );
        }
        E::Char('[') if state.stair_mode => {
            state.stair_dir = crate::staircase::cycle_dir(state.stair_dir, -1);
            state.status = format!("stair {}", crate::staircase::summary(state));
        }
        E::Char(']') if state.stair_mode => {
            state.stair_dir = crate::staircase::cycle_dir(state.stair_dir, 1);
            state.status = format!("stair {}", crate::staircase::summary(state));
        }
        E::Char('[') => {
            state.shape_sel = cycle_shape(state.shape_sel, -1);
            state.shape_mode = true;
            state.status = format!("shape: {}", shape_label(state.shape_sel));
        }
        E::Char(']') => {
            state.shape_sel = cycle_shape(state.shape_sel, 1);
            state.shape_mode = true;
            state.status = format!("shape: {}", shape_label(state.shape_sel));
        }
        // Inspect-mode semantic tag: mark the cursor cell an Object carrying a
        // node-id (demo id = 1) in `side_data`.
        E::Char('n') => {
            state.push_undo();
            let (x, y, z) = (state.cursor.0, state.cursor.1, state.z);
            state.map.set_cell(x, y, z, CellKind::Object, 1);
            state.mark_dirty();
            state.status = format!("tagged node id=1 @ {x},{y},{z}");
        }
        // ── UNDO / REDO ──  `u` undo, `y` redo. Not edits themselves (they don't
        // snapshot); they restore prior snapshots. Status reflects what happened.
        E::Char('u') => {
            if state.undo() {
                state.status = format!(
                    "undo  (undo:{} redo:{})",
                    state.history.undo_len(),
                    state.history.redo_len()
                );
            } else {
                state.status = "nothing to undo".into();
            }
        }
        E::Char('y') => {
            if state.redo() {
                state.status = format!(
                    "redo  (undo:{} redo:{})",
                    state.history.undo_len(),
                    state.history.redo_len()
                );
            } else {
                state.status = "nothing to redo".into();
            }
        }
        // ── FILL tools (active z-slice, current geometry brush) ──
        // `k` sets fill corner A at the cursor; `K` rect-fills A→cursor; `L` draws
        // a cell line A→cursor; `J` bucket-fills the region under the cursor. Each
        // fill is ONE undoable edit (snapshot once, before the whole fill).
        E::Char('k') => {
            state.fill_anchor = Some((state.cursor.0, state.cursor.1, state.z));
            state.status = format!(
                "fill corner A @ {},{},{}  (K rect / L line)",
                state.cursor.0, state.cursor.1, state.z
            );
        }
        E::Char('K') => {
            if let Some((ax, ay, _az)) = state.fill_anchor {
                state.push_undo();
                let n = fill::rect(state, (ax, ay), state.cursor);
                state.fill_anchor = None;
                state.mark_dirty();
                state.status = format!("rect fill {n} cells");
            } else {
                state.status = "set corner A first (k)".into();
            }
        }
        E::Char('L') => {
            if let Some((ax, ay, _az)) = state.fill_anchor {
                state.push_undo();
                let n = fill::line(state, (ax, ay), state.cursor);
                state.fill_anchor = None;
                state.mark_dirty();
                state.status = format!("line fill {n} cells");
            } else {
                state.status = "set corner A first (k)".into();
            }
        }
        E::Char('J') => {
            state.push_undo();
            let n = fill::bucket(state, state.cursor);
            state.mark_dirty();
            state.status = format!("bucket fill {n} cells");
        }
        // ── NEW-LEVEL wizard ──  `W` opens the form (modal; see apply_wizard_key).
        E::Char('W') => {
            state.wizard = Some(wizard::Wizard::default());
            state.status = "new-level wizard (Tab/+/-/Enter/Esc)".into();
        }
        // Save trigger: kept PURE — a flag the Bevy layer's save system acts on.
        // If there is no path yet, `save_system` routes this to the Save-As prompt
        // rather than lying "saved".
        E::Char('s') => state.status = "SAVE".into(),
        // `A` (capital — lowercase `a` is the camera StrafeLeft key) opens the
        // SAVE-AS prompt to name (or rename) the file the next save writes to.
        E::Char('A') => crate::saveas::open_saveas(state),
        // `?` opens the toggleable HELP popup (modal cheat-sheet). Closing it is
        // handled by the help-modal guard at the top of this fn (and Esc).
        E::Char('?') => state.open_help(),
        // `O` (capital — lowercase `o` is the object selector) opens the in-editor
        // LOAD overlay without restarting. Modal once open (see `apply_open_key`).
        E::Char('O') => crate::open::open_overlay(state),
        // `E` (capital — lowercase `e` creates a material) opens the LEVEL
        // SETTINGS form, the same modal the toolbar `[Level]` button opens — the
        // 8 mouse-only .lvl sections (meta/room/effects/palette/sky/texture/
        // inputs/sequence) become keyboard-reachable. Modal once open (see
        // `apply_settings_key`); while ANY earlier modal is up this arm is never
        // reached — the guard chain at the top of this fn early-returns first.
        E::Char('E') => crate::settings::open(state),
        // `D` opens the 3×3×3 DIRECTION PICKER for the selected emitter's aim
        // (capital — lowercase `d` is the camera StrafeRight key). It supersedes
        // the 6-axis `c`-cycle on the DirectionAxis field with a full 26-direction
        // overlay; the `c`-cycle still works as before (this is an ADDITION).
        E::Char('D') => entity::open_dir_picker(state),
        // `H` toggles OBJECT-MARKER visibility on the top-down grid: hide the
        // entity/pipe markers to read the raw geometry, show them again to edit
        // objects. A pure view toggle (no snapshot — it changes nothing in the
        // document). `H` is capital so it doesn't collide with the camera/tree `h`.
        E::Char('H') => {
            state.show_objects = !state.show_objects;
            state.status = if state.show_objects {
                "objects shown".into()
            } else {
                "objects hidden".into()
            };
        }
        // ── VIEW AIDS (`crate::viewaids`) ────────────────────────────────────
        // `G` cycles ONION SKIN: off -> +/-1 -> +/-2 -> off. Ghosts the occupied
        // cells of the neighbouring layer(s) into the 2D pane, dimmed and
        // direction-tinted, so a stairwell lines up without flipping z. A PURE
        // VIEW toggle: no snapshot, no dirty flag, and no preview rebuild — it
        // changes only how `chrome::resolve_cell` paints the 2D pane, exactly like
        // its sibling `H`. (`G` is capital: lowercase `g` is the pipe-route key.)
        E::Char('G') => {
            state.onion = crate::viewaids::cycle_onion(state.onion, 1);
            state.status = if state.onion.is_on() {
                format!(
                    "onion skin {} (ghosts are not clickable)",
                    state.onion.label()
                )
            } else {
                "onion skin off".into()
            };
        }
        // `F` toggles the SOUND-FIELD view aid (`crate::field`): tints each
        // field's footprint by archetype family in the 2D pane. A PURE VIEW
        // toggle — no snapshot, no dirty flag, no preview rebuild — mirrors
        // `G`'s contract exactly.
        E::Char('F') => crate::field::toggle_fields_aid(state),
        // `Shift+U` (capital — lowercase `u` is undo) arms/disarms the OPT-IN LIVE
        // AUDITION: plays the level's ambience + drone through the field
        // preview camera every tick (`audition_tick_system` in `main.rs`). A no-op
        // (with a status explaining why) under `TS_AUDIO_SILENT` or no device —
        // see `crate::field::Audition::start`.
        E::Char('U') => crate::field::toggle_audition(state),
        // `q` ENTERS the MASK-EDIT modal for the selected Part (a no-op hint when no
        // part is selected). Inside the modal `q` exits (handled in `apply_mask_key`).
        E::Char('q') => {
            if state.place == Some(crate::state::EntityKind::Part)
                && state
                    .selected_part
                    .is_some_and(|i| i < state.level.parts.len())
            {
                state.mask_mode = true;
                state.mask_layer = state.mask_layer.min(
                    (state.level.parts[state.selected_part.unwrap()].h as usize).saturating_sub(1),
                );
                state.mask_cursor = (0, 0);
                state.status = "mask: arrows move, [ ] layer, space solid, x empty, q exit".into();
            } else {
                state.status = "select a Part first (o to Part, space to place)".into();
            }
        }
        // `:` opens the COMMAND PALETTE: a one-line fuzzy finder over every
        // keymap action + goto-object + open-level rows. Routed as the
        // TOPMOST focus-stack frame while open (`crate::focus`); Esc closes
        // via `quit_on_esc`.
        E::Char(':') => crate::keymap::open_palette(state),
        // `z` ZOOM-TO-PANE: fullscreen the FOCUSED center viewport
        // (grid or preview — toolbar/status stay); the SAME key restores. The
        // outliner's `z` (duplicate) is unaffected — the tree guard above owns
        // the key while the outliner has focus.
        E::Char('z') => {
            if state.zoomed.is_some() {
                state.zoomed = None;
                state.status = "zoom restored".into();
            } else {
                let target = if state.view_mode == crate::state::ViewMode::Full3D
                    || state.focused_viewport == crate::state::ViewportFocus::Preview
                {
                    crate::layout::PanelId::Preview
                } else {
                    crate::layout::PanelId::Grid
                };
                state.zoomed = Some(target);
                state.status = format!("zoomed {target:?} (z restores)");
            }
        }
        // `V` toggles the READ-ONLY VALIDATION DRAWER: the last save's
        // full lvlcheck report, scrollable. (Capital — lowercase `v` is vox
        // mode.) While open it consumes only its scroll/close keys.
        E::Char('V') => crate::validate::toggle(state),
        // `Shift+R` EXPLICITLY rebuilds the preview map from the level,
        // asset re-decode included — the manual escape hatch for asset-bearing
        // levels whose debounced auto-rebuild is skipped (see
        // `vox::tick_preview_rebuild`). Placed in the main match so the
        // note/inset text-entry branches above still type a literal `R`.
        // (`R` is otherwise free here: the settings modal's `R`-resize lives
        // behind its own guard, and the camera fly key is lowercase `r`.)
        E::Char('R') => vox::manual_rebuild(state),
        // Tab cycles the view mode (Split → Full2D → Full3D → Split).
        E::Tab => {
            state.view_mode = state.view_mode.next();
            state.status = format!("view: {:?}", state.view_mode);
        }
        _ => {}
    }
}

/// F5 PLAYTEST request: lint the CURRENT in-memory level; on a hard FAIL
/// arm a warn-and-confirm (a second F5 launches anyway — never a hard refusal,
/// since a lvlcheck rule can always flip a level to FAIL that used to pass);
/// otherwise raise `playtest_request` for the exclusive `playtest_system` in
/// `main.rs` (it owns the terminal suspend/spawn/resume — this fn stays pure).
fn request_playtest(state: &mut EditorState) {
    let src = state.path.clone().unwrap_or_else(|| "unsaved".into());
    let report = re_lvlcheck::check_level(&state.level, &src);
    if report.failed() && !state.playtest_armed {
        state.playtest_armed = true;
        state.status = format!(
            "lvlcheck FAIL ({}) — F5 again to playtest anyway",
            report.verdict
        );
    } else {
        state.playtest_armed = false;
        state.playtest_request = true;
        state.status = "playtest: launching…".into();
    }
}

/// How many LINES the help sheet currently shows — the filtered body, banners
/// included, because those are lines the reader scrolls past too.
fn help_total(state: &EditorState) -> usize {
    crate::keymap::help_lines_filtered(&state.help_filter).len()
}

/// Move the help sheet to line `target`, CLAMPED to the last page.
///
/// The clamp is `chrome::clamp_help_scroll`, the same one the painter applies,
/// so an offset can never be stored that paints blank rows below the end of the
/// sheet (and a subsequent `Up` then has to undo invisible scrolling before
/// anything moves). `usize::MAX` is therefore the honest spelling of "End".
pub fn help_set_scroll(state: &mut EditorState, target: usize) {
    let total = help_total(state);
    // `help_page` is the body height the popup was last PAINTED with; before the
    // first paint a single line is the only defensible page size.
    let visible = state.help_page.max(1);
    state.help_scroll = crate::chrome::clamp_help_scroll(target, total, visible);
}

/// Scroll the help sheet by `delta` lines (negative = towards the top), clamped.
pub fn help_scroll_by(state: &mut EditorState, delta: isize) {
    let step = delta.unsigned_abs();
    let target = if delta < 0 {
        state.help_scroll.saturating_sub(step)
    } else {
        state.help_scroll.saturating_add(step)
    };
    help_set_scroll(state, target);
}

/// Append one typed character to the help filter and jump back to the top — a
/// filter that keeps the previous query's scroll offset opens on blank space.
fn help_filter_push(state: &mut EditorState, c: char) {
    state.help_filter.push(c);
    state.help_scroll = 0;
    state.status = format!("help filter: {}", state.help_filter);
}

/// Erase the last character of the help filter (no-op when it is already empty).
fn help_filter_pop(state: &mut EditorState) {
    state.help_filter.pop();
    state.help_scroll = 0;
    state.status = if state.help_filter.is_empty() {
        "help filter cleared".into()
    } else {
        format!("help filter: {}", state.help_filter)
    };
}

/// Modal key handling while the HELP SHEET is open (routed by the focus
/// stack, `crate::focus`). The sheet is a ~180-line reference over a
/// ~35-line window, so it is navigable rather than inert:
///
/// * `Up`/`Down` — one line, `PageUp`/`PageDown` — one PAINTED page,
///   `Home`/`End` — the ends. Every one of them clamps to the last page.
/// * printable characters — append to the live FILTER (rows survive on chord,
///   label or group); `Backspace` pops. Esc clears the filter and then closes
///   the sheet, in `main::quit_on_esc` (which owns the whole Esc chain).
/// * `?` — close, from anywhere, filtered or not.
///
/// ## The dispatch sentinels
///
/// The two marker comments immediately inside this function (a BEGIN on the
/// first line of the body, an END on the last) fence EXACTLY this dispatch, and
/// `keymap::every_dispatched_key_is_in_the_keymap` asserts that every key
/// literal between them has a `Context::Help` row in `KEYMAP`. That is what
/// stops this handler from growing a key the sheet itself never mentions —
/// which, for the surface whose whole job is to list the keys, would be the
/// worst place in the editor for it to happen.
pub fn apply_help_key(state: &mut EditorState, key: EditorKey) {
    // HELP-KEYS-BEGIN
    let page = isize::try_from(state.help_page.max(1)).unwrap_or(1);
    match key.code {
        // The key that opens the sheet is the key that shuts it, filter or no
        // filter; it is the one printable that never types into the query.
        E::Char('?') => state.close_help(),
        E::Char(c) => help_filter_push(state, c),
        E::Backspace => help_filter_pop(state),
        E::Up => help_scroll_by(state, -1),
        E::Down => help_scroll_by(state, 1),
        E::PageUp => help_scroll_by(state, -page),
        E::PageDown => help_scroll_by(state, page),
        E::Home => help_set_scroll(state, 0),
        E::End => help_set_scroll(state, usize::MAX),
        _ => {}
    }
    // HELP-KEYS-END
}

/// Modal key handling while the MASK-EDIT overlay is open (a part's solid mask):
/// * arrows — move the part-local `(lx, lz)` cursor within the active h-layer
///   (clamped to the part's `w`/`d`),
/// * `[` / `]` — step the active h-layer (`ly`) down / up (clamped `0..part.h`),
/// * `space` — TOGGLE the voxel under the cursor solid↔empty; `x` — force it EMPTY,
/// * `q` (or Esc, via `quit_on_esc`) — exit the modal.
///
/// Mask edits route through `crate::part::{toggle_mask, set_mask}` (SSOT) and
/// snapshot undo so each edit is one undoable step. A stale/empty part selection
/// exits the modal.
///
/// ## The dispatch sentinels
///
/// The two marker comments immediately inside this function (a BEGIN on the
/// first line of the body, an END on the last) fence EXACTLY this dispatch.
/// `keymap.rs`'s `every_dispatched_key_is_in_the_keymap` reads this file with
/// `include_str!`, scans ONLY that fence for `EditorKeyCode` literals, and
/// asserts each one has a `Context::Mask` binding in `KEYMAP` — otherwise the key
/// would be invisible in the hint bar, the `?` sheet and the `:` palette.
/// Keep the sentinels tight around this body; deleting one does not silently
/// disable the check, it FAILS the test.
fn apply_mask_key(state: &mut EditorState, key: EditorKey) {
    // MASK-KEYS-BEGIN
    let Some(pi) = state.selected_part.filter(|&i| i < state.level.parts.len()) else {
        // The selected part vanished (e.g. deleted from the tree) — leave the modal.
        state.mask_mode = false;
        return;
    };
    let (pw, ph, pd) = {
        let p = &state.level.parts[pi];
        (
            p.w.max(1) as usize,
            p.h.max(1) as usize,
            p.d.max(1) as usize,
        )
    };
    // Keep the layer + cursor in range (a resize may have shrunk the part).
    state.mask_layer = state.mask_layer.min(ph - 1);
    state.mask_cursor.0 = state.mask_cursor.0.min(pw - 1);
    state.mask_cursor.1 = state.mask_cursor.1.min(pd - 1);
    match key.code {
        E::Left => state.mask_cursor.0 = state.mask_cursor.0.saturating_sub(1),
        E::Right => state.mask_cursor.0 = (state.mask_cursor.0 + 1).min(pw - 1),
        E::Up => state.mask_cursor.1 = state.mask_cursor.1.saturating_sub(1),
        E::Down => state.mask_cursor.1 = (state.mask_cursor.1 + 1).min(pd - 1),
        E::Char('[') => {
            state.mask_layer = state.mask_layer.saturating_sub(1);
            state.status = format!("mask layer {}/{}", state.mask_layer, ph - 1);
        }
        E::Char(']') => {
            state.mask_layer = (state.mask_layer + 1).min(ph - 1);
            state.status = format!("mask layer {}/{}", state.mask_layer, ph - 1);
        }
        E::Char(' ') => {
            // `space` TOGGLES the voxel solid↔empty.
            let (lx, lz) = (state.mask_cursor.0 as i32, state.mask_cursor.1 as i32);
            let ly = state.mask_layer as i32;
            state.push_undo_checkpoint();
            let now = crate::part::toggle_mask(&mut state.level.parts[pi], lx, ly, lz);
            if !state.drop_undo_if_unchanged() {
                state.mark_dirty();
            }
            state.status = format!(
                "mask ({},{},{}) = {}",
                lx,
                ly,
                lz,
                if now { "solid" } else { "empty" }
            );
        }
        E::Char('x') => {
            // `x` forces the voxel EMPTY.
            let (lx, lz) = (state.mask_cursor.0 as i32, state.mask_cursor.1 as i32);
            let ly = state.mask_layer as i32;
            state.push_undo_checkpoint();
            crate::part::set_mask(&mut state.level.parts[pi], lx, ly, lz, false);
            if !state.drop_undo_if_unchanged() {
                state.mark_dirty();
            }
            state.status = format!("mask ({lx},{ly},{lz}) = empty");
        }
        E::Char('q') => {
            state.mask_mode = false;
            state.status = "mask edit closed".into();
        }
        _ => {}
    }
    // MASK-KEYS-END
}

/// Modal key handling while the NEW-LEVEL wizard is open. `Tab` focuses the next
/// field; `+`/`-` adjust the focused field; `Enter` creates the level (replacing
/// the document) and closes the form; `Esc` cancels. All other keys are inert so
/// the form can't accidentally edit the (about-to-be-replaced) map.
///
/// ## The dispatch sentinels
///
/// The two marker comments immediately inside this function (a BEGIN on the
/// first line of the body, an END on the last) fence EXACTLY this dispatch.
/// `keymap.rs`'s `every_dispatched_key_is_in_the_keymap` reads this file with
/// `include_str!`, scans ONLY that fence for `EditorKeyCode` literals, and
/// asserts each one has a `Context::Wizard` binding in `KEYMAP` — otherwise the key
/// would be invisible in the hint bar, the `?` sheet and the `:` palette.
/// Keep the sentinels tight around this body; deleting one does not silently
/// disable the check, it FAILS the test.
fn apply_wizard_key(state: &mut EditorState, key: EditorKey) {
    // WIZARD-KEYS-BEGIN
    use crate::wizard::WizardField;
    // FREE-TEXT NAME ENTRY: while the Name field is focused, every printable char
    // (including `+`/`-`/`=`) is typed INTO the name buffer rather than adjusting a
    // preset, so an author can name a level anything instead of picking from a
    // preset cycle. (There is no Backspace in re_present's EditorKeyCode, so
    // deletion is via the wizard's `[Clr]` button.)
    if let Some(w) = state.wizard.as_mut() {
        if w.focus == WizardField::Name {
            match key.code {
                E::Char(c) => {
                    w.name_push(c);
                    return;
                }
                E::Backspace => {
                    w.name_pop();
                    return;
                }
                _ => {}
            }
        }
    }
    match key.code {
        E::Tab => {
            if let Some(w) = state.wizard.as_mut() {
                w.focus_next();
            }
        }
        E::Char('+') | E::Char('=') => {
            if let Some(w) = state.wizard.as_mut() {
                w.adjust(1);
            }
        }
        E::Char('-') => {
            if let Some(w) = state.wizard.as_mut() {
                w.adjust(-1);
            }
        }
        // Enter creates. Take the wizard out so `apply` (which mutates state) does
        // not hold an aliasing borrow, then drop the form.
        E::Enter => {
            if let Some(w) = state.wizard.take() {
                wizard::apply(state, &w);
            }
        }
        // Esc (cancel) is handled by the Bevy `quit_on_esc` system so the SAME
        // Esc keypress can't both close the wizard here AND quit the app there.
        _ => {}
    }
    // WIZARD-KEYS-END
}

/// Modal key handling while the LEVEL SETTINGS form is open. THIN — delegates all
/// logic to [`crate::settings::SettingsState`] methods + [`crate::settings::apply`].
///
/// * `Tab` / `Down` → next field; `Up` → previous field (but in a list SECTION,
///   Up/Down move that section's row cursor instead).
/// * Name / Title (text): `Char` types, `Backspace` pops.
/// * Ambient / StoreyHeight: `Left`/`-` and `Right`/`+`/`=` adjust.
/// * CycleKeep / RenderPipes / Steam: `Space` (or `-`/`+`) toggles.
/// * Snow / Matrix / Glitch / FogNear / FogFar: `Left`/`-` and `Right`/`+`/`=`
///   adjust the f32 (effects in 0.1 steps clamped 0..1; fog distance in 0.5 steps).
/// * `AtmosAmbient` / `FogHeightBase` / `FogHeightFalloff` (`[atmosphere]`):
///   `Left`/`-` and `Right`/`+`/`=` step the f32 (ambient in 0.02 steps clamped
///   0..1 — stepping below 0 CLEARS it back to unset; `fog_height` base in 0.5
///   steps, falloff in 0.1 steps clamped `>= 0`, either half authoring the pair).
/// * `Postfx`: `Left`/`-` and `Right`/`+`/`=` cycle the unset state plus every
///   `re_world::POSTFX_PRESETS` entry.
/// * `FogColor` / `KeyColor` / `SkyCylinder` / `SkyCapTop` / `SkyCapBottom`
///   (text): `Char` types, `Backspace` pops.
/// * Textures section: Up/Down move the row cursor; Left/Right pick the edited
///   column (name/file); printable chars type into it; `a` adds a row, `x` removes.
/// * Inputs section: `Space` toggles the input under the cursor.
/// * Observe section: `a` adds the next candidate name, `x` removes the row.
/// * Steps section: `a` adds a step, `x` removes the step; `Left`/`Right` move the
///   in-step name cursor; `n` adds a name into the step, `r` removes one.
/// * Version / SpecRev / RoomFloor / RoomCeiling / DoorTargetModel (free text):
///   `Char` types, `Backspace` pops.
/// * PowerBudget: `0`-`9` TYPE a digit, `Backspace` pops, `-`/`+` step.
/// * `Enter` ALWAYS confirms + applies the whole form (mirrors the wizard's Enter).
/// * `Esc` is NOT routed here — `main.rs::quit_on_esc` discards the form.
///
/// ## The dispatch sentinels
///
/// The two marker comments immediately inside this function (a BEGIN on the
/// first line of the body and an END on the last) fence EXACTLY this dispatch.
/// `keymap.rs`'s `every_settings_key_is_in_the_keymap` test reads this file with
/// `include_str!` and scans ONLY that fence for `EditorKeyCode` literals,
/// asserting each has a `Context::Settings` binding in `KEYMAP`. That closes a
/// drift direction the old table-only test left open: it proved table →
/// surfaces, but nothing proved dispatch → table. Keep the sentinels tight around
/// this body — widening them would drag unrelated key literals into the scan;
/// deleting one does not silently disable the check, it fails the test.
fn apply_settings_key(state: &mut EditorState, key: EditorKey) {
    // SETTINGS-KEYS-BEGIN
    use crate::settings::SettingsField as F;
    // Enter always applies (the form has many fields; one confirm key is simplest).
    if key.code == E::Enter {
        crate::settings::apply(state);
        return;
    }
    // `R` resizes the live grid to the staged W/H/D (a structural remap, separate
    // from the meta/room Apply). Handled before the `as_mut` borrow because resize
    // mutates the whole `state`. Not a text field, so it never collides with typing.
    if matches!(key.code, E::Char('R')) {
        crate::settings::resize(state);
        return;
    }
    let Some(s) = state.settings.as_mut() else {
        return;
    };
    // Text-field typing takes precedence so `-`/`+`/`a`/`x`/etc. type literally.
    match s.focus {
        F::Name => match key.code {
            E::Char(c) => {
                s.name_push(c);
                return;
            }
            E::Backspace => {
                s.name_pop();
                return;
            }
            _ => {}
        },
        F::Title => match key.code {
            E::Char(c) => {
                s.title_push(c);
                return;
            }
            E::Backspace => {
                s.title_pop();
                return;
            }
            _ => {}
        },
        // FOG COLOUR is a free hex text buffer (`#RRGGBB`); printable chars type so
        // they never fall through to the numeric/cycle handlers below.
        F::FogColor => match key.code {
            E::Char(c) => {
                s.fog_color_push(c);
                return;
            }
            E::Backspace => {
                s.fog_color_pop();
                return;
            }
            _ => {}
        },
        // KEY COLOUR is a free hex text buffer (`#RRGGBB`) like the fog colour;
        // printable chars type so they never reach the numeric handlers below.
        F::KeyColor => match key.code {
            E::Char(c) => {
                s.key_color_push(c);
                return;
            }
            E::Backspace => {
                s.key_color_pop();
                return;
            }
            _ => {}
        },
        // SKY PATHS are free text buffers; printable chars type into the focused one.
        F::SkyCylinder | F::SkyCapTop | F::SkyCapBottom => match key.code {
            E::Char(c) => {
                s.sky_push(s.focus, c);
                return;
            }
            E::Backspace => {
                s.sky_pop(s.focus);
                return;
            }
            _ => {}
        },
        // POWER BUDGET is an INTEGER with TYPED ENTRY: a digit types, Backspace
        // pops. `-`/`+`/arrows deliberately FALL THROUGH to the stepper below, so
        // the field has both input methods: an integer whose only input
        // is pressing `+` N times is not practically authorable.
        F::PowerBudget => match key.code {
            E::Char(c) if c.is_ascii_digit() => {
                s.power_budget_push(c);
                return;
            }
            E::Backspace => {
                s.power_budget_pop();
                return;
            }
            _ => {}
        },
        // FREE-TEXT metadata (`[meta] version`/`spec_rev`, `[room] floor`/
        // `ceiling`, `[door] target_model`): printable chars type so they never
        // fall through to the numeric/cycle handlers below.
        f if crate::settings::SettingsState::is_text_field(f) => match key.code {
            E::Char(c) => {
                s.text_push(f, c);
                return;
            }
            E::Backspace => {
                s.text_pop(f);
                return;
            }
            _ => {}
        },
        // TEXTURES: a printable char types into the focused row's focused column
        // (name/file). Reserve `a`/`x` (add/remove) and the arrows for navigation so
        // they are NOT swallowed as text — every other printable char types.
        F::Textures => match key.code {
            E::Char(c) if c != 'a' && c != 'x' => {
                s.texture_push(c);
                return;
            }
            E::Backspace => {
                s.texture_pop();
                return;
            }
            _ => {}
        },
        _ => {}
    }
    match key.code {
        // Focus movement. In a list section Up/Down move the row cursor; elsewhere
        // they step focus.
        E::Tab => s.focus_next(),
        E::Down => match s.focus {
            F::Inputs => s.move_input_cursor(1),
            F::Observe => s.move_observe_cursor(1),
            F::Steps => s.move_step_cursor(1),
            F::Textures => s.move_texture_cursor(1),
            _ => s.focus_next(),
        },
        E::Up => match s.focus {
            F::Inputs => s.move_input_cursor(-1),
            F::Observe => s.move_observe_cursor(-1),
            F::Steps => s.move_step_cursor(-1),
            F::Textures => s.move_texture_cursor(-1),
            _ => s.focus_prev(),
        },
        // Decrement / left.
        E::Left | E::Char('-') => match s.focus {
            F::Ambient => s.ambient_cycle(-1),
            F::StoreyHeight => s.storey_adjust(-1),
            F::SizeW | F::SizeH | F::SizeD => s.size_adjust(s.focus, -1),
            F::PowerBudget => s.power_budget_adjust(-1),
            F::CycleKeep => s.toggle_cycle_keep(),
            F::RenderPipes => s.toggle_render_pipes(),
            F::Snow => s.snow_adjust(-1),
            F::Steam => s.toggle_steam(),
            F::Matrix => s.matrix_adjust(-1),
            F::Glitch => s.glitch_adjust(-1),
            F::FogNear => s.fog_near_adjust(-1),
            F::FogFar => s.fog_far_adjust(-1),
            F::AtmosAmbient => s.atmos_ambient_adjust(-1),
            F::FogHeightBase => s.fog_height_base_adjust(-1),
            F::FogHeightFalloff => s.fog_height_falloff_adjust(-1),
            F::Postfx => s.postfx_cycle(-1),
            F::Steps => s.move_name_cursor(-1),
            F::Textures => s.move_texture_col(-1),
            _ => {}
        },
        // Increment / right.
        E::Right | E::Char('+') | E::Char('=') => match s.focus {
            F::Ambient => s.ambient_cycle(1),
            F::StoreyHeight => s.storey_adjust(1),
            F::SizeW | F::SizeH | F::SizeD => s.size_adjust(s.focus, 1),
            F::PowerBudget => s.power_budget_adjust(1),
            F::CycleKeep => s.toggle_cycle_keep(),
            F::RenderPipes => s.toggle_render_pipes(),
            F::Snow => s.snow_adjust(1),
            F::Steam => s.toggle_steam(),
            F::Matrix => s.matrix_adjust(1),
            F::Glitch => s.glitch_adjust(1),
            F::FogNear => s.fog_near_adjust(1),
            F::FogFar => s.fog_far_adjust(1),
            F::AtmosAmbient => s.atmos_ambient_adjust(1),
            F::FogHeightBase => s.fog_height_base_adjust(1),
            F::FogHeightFalloff => s.fog_height_falloff_adjust(1),
            F::Postfx => s.postfx_cycle(1),
            F::Steps => s.move_name_cursor(1),
            F::Textures => s.move_texture_col(1),
            _ => {}
        },
        // Space toggles bools / the highlighted input.
        E::Char(' ') => match s.focus {
            F::CycleKeep => s.toggle_cycle_keep(),
            F::RenderPipes => s.toggle_render_pipes(),
            F::Steam => s.toggle_steam(),
            F::Inputs => s.toggle_input(),
            _ => {}
        },
        // Add a row in a list section.
        E::Char('a') => match s.focus {
            F::Observe => s.add_observe(),
            F::Steps => s.add_step(),
            F::Textures => s.add_texture(),
            _ => {}
        },
        // Remove a row in a list section.
        E::Char('x') => match s.focus {
            F::Observe => s.remove_observe(),
            F::Steps => s.remove_step(),
            F::Textures => s.remove_texture(),
            _ => {}
        },
        // Add a name into the selected step.
        E::Char('n') if s.focus == F::Steps => s.add_step_name(),
        // Remove the name under the in-step cursor.
        E::Char('r') if s.focus == F::Steps => s.remove_step_name(),
        _ => {}
    }
    // SETTINGS-KEYS-END
}

/// Modal key handling while the object-TYPE SELECTOR palette is open. Up/Down
/// move the highlight; Enter (or `i`/Tab) picks the type → placement mode; `i`
/// also closes it. Other keys are inert so the palette can't paint/move.
///
/// ## The dispatch sentinels
///
/// The two marker comments immediately inside this function (a BEGIN on the
/// first line of the body, an END on the last) fence EXACTLY this dispatch.
/// `keymap.rs`'s `every_dispatched_key_is_in_the_keymap` reads this file with
/// `include_str!`, scans ONLY that fence for `EditorKeyCode` literals, and
/// asserts each one has a `Context::ObjectPalette` binding in `KEYMAP` — otherwise the key
/// would be invisible in the hint bar, the `?` sheet and the `:` palette.
/// Keep the sentinels tight around this body; deleting one does not silently
/// disable the check, it FAILS the test.
fn apply_palette_key(state: &mut EditorState, key: EditorKey) {
    // OBJPALETTE-KEYS-BEGIN
    match key.code {
        E::Up => tree::palette_navigate(state, -1),
        E::Down => tree::palette_navigate(state, 1),
        E::Enter => tree::palette_pick(state),
        E::Char('i') => tree::toggle_palette(state),
        _ => {}
    }
    // OBJPALETTE-KEYS-END
}

/// Modal key handling while the 3×3×3 DIRECTION PICKER is open:
/// * arrows — move the cursor within the current Z layer (`dx`/`dy`, clamped),
/// * PageUp / `]` — switch to the layer above; PageDown / `[` — the layer below,
/// * Enter / space — PICK the cell under the cursor: set the emitter's
///   `direction` to `normalize((dx, dy, dz))` and close (the centre cell keeps
///   the current aim and closes),
/// * `c` — also picks (keeps the muscle-memory of the enum-cycle key),
/// * `D` — close without changing (Esc also closes, via `quit_on_esc`).
///
/// All other keys are inert so nothing paints under the overlay.
///
/// ## The dispatch sentinels
///
/// The two marker comments immediately inside this function (a BEGIN on the
/// first line of the body, an END on the last) fence EXACTLY this dispatch.
/// `keymap.rs`'s `every_dispatched_key_is_in_the_keymap` reads this file with
/// `include_str!`, scans ONLY that fence for `EditorKeyCode` literals, and
/// asserts each one has a `Context::DirPicker` binding in `KEYMAP` — otherwise the key
/// would be invisible in the hint bar, the `?` sheet and the `:` palette.
/// Keep the sentinels tight around this body; deleting one does not silently
/// disable the check, it FAILS the test.
pub fn apply_dir_picker_key(state: &mut EditorState, key: EditorKey) {
    // DIRPICKER-KEYS-BEGIN
    let Some(mut p) = state.dir_picker else {
        return;
    };
    match key.code {
        E::Left => p.move_cursor(-1, 0),
        E::Right => p.move_cursor(1, 0),
        E::Up => p.move_cursor(0, -1),
        E::Down => p.move_cursor(0, 1),
        E::PageUp | E::Char(']') => p.switch_layer(1),
        E::PageDown | E::Char('[') => p.switch_layer(-1),
        E::Enter | E::Char(' ') | E::Char('c') => {
            state.dir_picker = Some(p);
            entity::pick_dir_picker_cell(state);
            return;
        }
        E::Char('D') => {
            state.dir_picker = None;
            state.status = "direction picker cancelled".into();
            return;
        }
        _ => {}
    }
    state.dir_picker = Some(p);
    state.status = format!("aim picker: layer {}", p.layer_label());
    // DIRPICKER-KEYS-END
}

/// Modal key handling while the LOAD/OPEN overlay is open: Up/Down move the
/// highlighted file, Enter loads the selected level in-place, `O` closes it (Esc
/// also closes, via `quit_on_esc`). Other keys are inert.
///
/// ## The dispatch sentinels
///
/// The two marker comments immediately inside this function (a BEGIN on the
/// first line of the body, an END on the last) fence EXACTLY this dispatch.
/// `keymap.rs`'s `every_dispatched_key_is_in_the_keymap` reads this file with
/// `include_str!`, scans ONLY that fence for `EditorKeyCode` literals, and
/// asserts each one has a `Context::Open` binding in `KEYMAP` — otherwise the key
/// would be invisible in the hint bar, the `?` sheet and the `:` palette.
/// Keep the sentinels tight around this body; deleting one does not silently
/// disable the check, it FAILS the test.
fn apply_open_key(state: &mut EditorState, key: EditorKey) {
    // OPEN-KEYS-BEGIN
    match key.code {
        E::Up => crate::open::navigate(state, -1),
        E::Down => crate::open::navigate(state, 1),
        E::Enter => crate::open::pick(state),
        E::Char('O') => crate::open::close_overlay(state),
        _ => {}
    }
    // OPEN-KEYS-END
}

/// Modal key handling while the SAVE-AS prompt is open: every printable char
/// types into the filename buffer, Enter confirms (and on success triggers the
/// real write by setting the SAVE flag the Bevy `save_system` reads). Esc cancels
/// via `quit_on_esc`. (`[Clr]` clears the buffer via the modal mouse path — there
/// is no Backspace key in `re_present`.) Other keys are inert.
///
/// ## The dispatch sentinels
///
/// The two marker comments immediately inside this function (a BEGIN on the
/// first line of the body, an END on the last) fence EXACTLY this dispatch.
/// `keymap.rs`'s `every_dispatched_key_is_in_the_keymap` reads this file with
/// `include_str!`, scans ONLY that fence for `EditorKeyCode` literals, and
/// asserts each one has a `Context::SaveAs` binding in `KEYMAP` — otherwise the key
/// would be invisible in the hint bar, the `?` sheet and the `:` palette.
/// Keep the sentinels tight around this body; deleting one does not silently
/// disable the check, it FAILS the test.
fn apply_saveas_key(state: &mut EditorState, key: EditorKey) {
    // SAVEAS-KEYS-BEGIN
    match key.code {
        E::Enter => {
            if crate::saveas::confirm(state) {
                // Path is set + the prompt closed — drive the single save choke
                // point so the disk write happens (and clears dirty on success).
                state.status = "SAVE".into();
            }
        }
        E::Backspace => crate::saveas::pop_char(state),
        E::Char(c) => crate::saveas::push_char(state, c),
        _ => {}
    }
    // SAVEAS-KEYS-END
}

/// Modal key handling while the OBJECT TREE / OUTLINER is focused:
/// * Up/Down — move the highlighted row,
/// * Enter — select + jump to the object (group rows fold/unfold),
/// * `h` — collapse/expand the enclosing group,
/// * `z` — duplicate the selected object (offset +1, selects the copy),
/// * `d` — delete the selected object,
/// * `t` — close the panel.
///
/// All edits route through `crate::tree`, which snapshots undo for delete/dup.
fn apply_tree_key(state: &mut EditorState, key: EditorKey) {
    // ── `/` LIVE FILTER ── while the filter line is active, printable
    // chars type into the query (so the command keys h/z/d/t/p/u type
    // literally), Backspace pops, Up/Dn navigate the FILTERED rows, Enter
    // selects the highlighted match and closes the filter. Esc clears via
    // `quit_on_esc` (never routed here). `build_tree` applies the filter, so
    // the painter, the hit-test, and this navigation all see the same rows.
    if state.tree.filter.is_some() {
        match key.code {
            E::Char(c) => {
                if let Some(f) = state.tree.filter.as_mut() {
                    f.push(c);
                }
                tree::clamp_cursor(state);
            }
            E::Backspace => {
                if let Some(f) = state.tree.filter.as_mut() {
                    f.pop();
                }
            }
            E::Enter => {
                tree::select_current(state);
                state.tree.filter = None;
            }
            E::Up => tree::navigate(state, -1),
            E::Down => tree::navigate(state, 1),
            _ => {}
        }
        return;
    }
    match key.code {
        // `/` opens the live row filter (typed query, Esc clears).
        E::Char('/') => {
            state.tree.filter = Some(String::new());
            state.status = "outliner filter: type to filter, Enter selects, Esc clears".into();
        }
        E::Up => tree::navigate(state, -1),
        E::Down => tree::navigate(state, 1),
        E::Enter => tree::select_current(state),
        E::Char('h') => tree::toggle_collapse(state),
        E::Char('z') => tree::duplicate_current(state),
        E::Char('d') => tree::delete_current(state),
        // `p` reparents the SELECTED object under a target (two-step: press p to
        // begin, navigate to the parent, press p to confirm); `u` un-parents the
        // object the cursor is on (or cancels a pending reparent). The nesting
        // makes moving a parent group-move its descendants.
        E::Char('p') => crate::hierarchy::reparent_key(state),
        E::Char('u') => crate::hierarchy::unparent_key(state),
        E::Char('t') => tree::toggle(state),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn key(code: E) -> EditorKey {
        EditorKey { code, shift: false }
    }

    #[test]
    fn space_paints_current_brush() {
        let mut s = EditorState::new_empty(4, 4, 2);
        s.cursor = (2, 2);
        s.brush = CellKind::WallSolid;
        apply_key(&mut s, key(E::Char(' ')));
        assert_eq!(s.map.cell_at(2, 2, 0), CellKind::WallSolid);
    }

    /// GATING: F5 on a level lvlcheck FAILs (an empty document has no
    /// nodes → inventory FAIL) ARMS the warn-and-confirm instead of launching;
    /// an immediate second F5 confirms and raises the playtest request.
    #[test]
    fn f5_on_failing_level_warns_then_confirms() {
        let mut s = EditorState::new_empty(8, 8, 4);
        apply_key(&mut s, key(E::F(5)));
        assert!(s.playtest_armed, "first F5 on a FAIL arms the confirm");
        assert!(!s.playtest_request, "no launch on the warning press");
        assert!(
            s.status.contains("F5 again"),
            "status explains the confirm: {}",
            s.status
        );
        apply_key(&mut s, key(E::F(5)));
        assert!(!s.playtest_armed, "confirm consumed");
        assert!(
            s.playtest_request,
            "second F5 launches anyway (never a hard refusal)"
        );
    }

    /// GATING: any other key between the two F5 presses DISARMS the
    /// confirm (mirrors the dirty-quit guard).
    #[test]
    fn f5_confirm_disarmed_by_any_other_key() {
        let mut s = EditorState::new_empty(8, 8, 4);
        apply_key(&mut s, key(E::F(5)));
        assert!(s.playtest_armed);
        apply_key(&mut s, key(E::Left)); // any non-F5 key
        assert!(!s.playtest_armed, "other keys disarm the confirm");
        apply_key(&mut s, key(E::F(5)));
        assert!(
            s.playtest_armed,
            "next F5 warns again rather than launching"
        );
        assert!(!s.playtest_request);
    }

    /// F5 on a level lvlcheck does NOT fail launches immediately —
    /// no confirm step. (Shipped level 01 passes the FAIL-severity checks.)
    #[test]
    fn f5_on_clean_level_requests_immediately() {
        let mut s = EditorState::new_empty(4, 4, 2);
        assert!(
            crate::open::load_path(&mut s, "../../levels/01.lvl"),
            "shipped level 01 loads"
        );
        apply_key(&mut s, key(E::F(5)));
        assert!(!s.playtest_armed, "no confirm needed on a passing level");
        assert!(s.playtest_request, "playtest requested on the first F5");
    }

    /// F2 with no named selection is a hinted no-op; with a selected
    /// node it opens the rename prompt, which then owns the keyboard (a `w`
    /// keystroke types into the buffer instead of reaching any edit arm).
    #[test]
    fn f2_opens_rename_prompt_which_captures_keys() {
        let mut s = EditorState::new_empty(8, 8, 4);
        apply_key(&mut s, key(E::F(2)));
        assert!(s.rename.is_none(), "no selection: hinted no-op");

        let n = s.level.push_node(2, 2, 0);
        s.place = Some(crate::state::EntityKind::Node);
        s.selected_node = Some(n);
        apply_key(&mut s, key(E::F(2)));
        assert!(
            s.rename.is_some(),
            "F2 opens the prompt on a named selection"
        );
        let before_wizard = s.wizard.is_some();
        apply_key(&mut s, key(E::Char('W'))); // would open the wizard if not modal
        assert_eq!(s.wizard.is_some(), before_wizard, "prompt captures keys");
        assert!(
            s.rename.as_ref().unwrap().buf.ends_with('W'),
            "the char went into the name buffer"
        );
    }

    /// `Shift+R` rebuilds the preview map immediately — a node placed in
    /// the level (not the map) becomes a stamped Object cell.
    #[test]
    fn shift_r_manually_rebuilds_preview() {
        let mut s = EditorState::new_empty(8, 8, 4);
        let _ = s.level.push_node(5, 5, 1);
        assert_eq!(s.map.cell_at(5, 5, 1), CellKind::Empty);
        apply_key(&mut s, key(E::Char('R')));
        assert_eq!(
            s.map.cell_at(5, 5, 1),
            CellKind::Object,
            "manual rebuild ran"
        );
    }

    /// 1.1: [`tool_paints_on_drag`] — stroke tools (geometry, shape, material)
    /// answer true; placement tools (object, vox stamp, stair stamp) answer
    /// false, mirroring `paint_tool_at_cursor`'s dispatch order exactly.
    #[test]
    fn tool_paints_on_drag_stroke_vs_placement() {
        let mut s = EditorState::new_empty(4, 4, 2);
        assert!(tool_paints_on_drag(&s), "geometry brush is a stroke tool");
        s.shape_mode = true;
        assert!(tool_paints_on_drag(&s), "shape brush is a stroke tool");
        s.shape_mode = false;
        s.material_mode = true;
        assert!(tool_paints_on_drag(&s), "material paint is a stroke tool");
        s.material_mode = false;
        s.place = Some(crate::state::EntityKind::Node);
        assert!(
            !tool_paints_on_drag(&s),
            "object placement fires on press only"
        );
        s.place = None;
        s.stair_mode = true;
        assert!(!tool_paints_on_drag(&s), "stair stamp fires on press only");
        s.stair_mode = false;
        s.vox_mode = true;
        assert!(!tool_paints_on_drag(&s), "vox stamp fires on press only");
        // Dispatch precedence: vox wins over material (mirrors paint dispatch).
        s.material_mode = true;
        assert!(
            !tool_paints_on_drag(&s),
            "vox precedence over material mirrors dispatch"
        );
    }

    /// 1.2: the `E` chord opens the LEVEL SETTINGS form — the same modal the
    /// toolbar `[Level]` button opens (previously mouse-only).
    #[test]
    fn chord_e_opens_level_settings() {
        let mut s = EditorState::new_empty(4, 4, 2);
        assert!(s.settings.is_none());
        apply_key(&mut s, key(E::Char('E')));
        assert!(s.settings.is_some(), "`E` opens the Level Settings form");
        assert!(
            s.status.starts_with("level settings"),
            "status announces the settings form"
        );
    }

    /// 1.2: while ANY earlier modal is up, `E` is swallowed by that modal's
    /// guard — the settings form must NOT stack on top of it.
    #[test]
    fn chord_e_blocked_while_another_modal_is_up() {
        // Help popup open → `E` is inert (only `?` acts).
        let mut s = EditorState::new_empty(4, 4, 2);
        s.help_open = true;
        apply_key(&mut s, key(E::Char('E')));
        assert!(s.settings.is_none(), "help modal swallows `E`");
        // New-level wizard open → `E` routes to the wizard, not settings.
        let mut s = EditorState::new_empty(4, 4, 2);
        s.wizard = Some(wizard::Wizard::default());
        apply_key(&mut s, key(E::Char('E')));
        assert!(s.settings.is_none(), "wizard modal swallows `E`");
        // Preview free-look captured → every 2D-edit key is suppressed.
        let mut s = EditorState::new_empty(4, 4, 2);
        s.preview_captured = true;
        apply_key(&mut s, key(E::Char('E')));
        assert!(s.settings.is_none(), "preview capture swallows `E`");
    }

    /// The five `[atmosphere]` rows are REACHABLE by Tab and each one STEPS from
    /// the real key dispatch — a row wired into `ORDER` but not into the
    /// `Left`/`Right` arms would look editable and silently do nothing.
    #[test]
    fn atmosphere_settings_rows_are_reachable_and_step() {
        use crate::settings::SettingsField as F;
        let mut s = EditorState::new_empty(4, 4, 2);
        apply_key(&mut s, key(E::Char('E')));
        assert!(s.settings.is_some(), "the settings form is open");

        // Tab from the first field until each atmosphere row is focused, in the
        // order they sit in the form. One lap is the upper bound.
        let lap = F::ORDER.len();
        for want in [
            F::AtmosAmbient,
            F::KeyColor,
            F::FogHeightBase,
            F::FogHeightFalloff,
            F::Postfx,
        ] {
            let mut hops = 0;
            while s.settings.as_ref().unwrap().focus != want {
                apply_key(&mut s, key(E::Tab));
                hops += 1;
                assert!(hops <= lap, "Tab never reaches {want:?} - is it in ORDER?");
            }
            // The focused row STEPS on Right.
            apply_key(&mut s, key(E::Right));
            let st = s.settings.as_ref().unwrap();
            match want {
                F::AtmosAmbient => assert!(
                    st.atmos_ambient.is_some_and(|a| a > 0.0),
                    "Right steps the atmosphere ambient"
                ),
                F::FogHeightBase => assert!(
                    st.has_fog_height && st.fog_height_base > 0.0,
                    "Right steps the height-fog base and authors the pair"
                ),
                F::FogHeightFalloff => assert!(
                    st.has_fog_height && st.fog_height_falloff > 0.0,
                    "Right steps the height-fog falloff"
                ),
                F::Postfx => assert!(!st.postfx().is_empty(), "Right cycles postfx off (default)"),
                // KeyColor is a TEXT row: Right must not step anything, and a
                // printable char must TYPE rather than fall through to a stepper.
                F::KeyColor => assert!(st.key_color.is_empty(), "Right does not type"),
                _ => unreachable!(),
            }
        }

        // KeyColor typing + Backspace, through the dispatch.
        while s.settings.as_ref().unwrap().focus != F::KeyColor {
            apply_key(&mut s, key(E::Tab));
        }
        for c in "#0AF0C1".chars() {
            apply_key(&mut s, key(E::Char(c)));
        }
        assert_eq!(
            s.settings.as_ref().unwrap().key_color,
            "#0AF0C1",
            "chars type into key_color"
        );
        apply_key(&mut s, key(E::Backspace));
        assert_eq!(
            s.settings.as_ref().unwrap().key_color,
            "#0AF0C",
            "Backspace pops key_color"
        );

        // `Left` steps back down the ambient row (and can clear it).
        while s.settings.as_ref().unwrap().focus != F::AtmosAmbient {
            apply_key(&mut s, key(E::Tab));
        }
        apply_key(&mut s, key(E::Char('-')));
        assert!(
            s.settings
                .as_ref()
                .unwrap()
                .atmos_ambient
                .is_some_and(|a| a.abs() < 1e-6),
            "`-` steps the one applied step back down to 0.0 (still AUTHORED)"
        );
        apply_key(&mut s, key(E::Char('-')));
        assert_eq!(
            s.settings.as_ref().unwrap().atmos_ambient,
            None,
            "one more `-` below 0 clears the ambient back to unset"
        );
    }

    /// 1.2: lowercase `e` KEEPS its meaning (create a new material) — the
    /// settings chord is the capital only.
    #[test]
    fn lowercase_e_still_creates_material_not_settings() {
        let mut s = EditorState::new_empty(4, 4, 2);
        let before = s.level.materials.len();
        apply_key(&mut s, key(E::Char('e')));
        assert_eq!(
            s.level.materials.len(),
            before + 1,
            "`e` creates a material"
        );
        assert!(s.settings.is_none(), "`e` does not open settings");
    }

    #[test]
    fn key_zero_selects_portal_brush_and_space_paints_it() {
        // `0` selects the WARP-PORTAL cell brush; space then paints `CellKind::Portal`.
        let mut s = EditorState::new_empty(4, 4, 2);
        apply_key(&mut s, key(E::Char('0')));
        assert_eq!(
            s.brush,
            CellKind::Portal,
            "`0` selects the Portal cell brush"
        );
        assert!(
            !s.shape_mode && s.place.is_none(),
            "Portal brush leaves special modes"
        );
        s.cursor = (1, 2);
        apply_key(&mut s, key(E::Char(' ')));
        assert_eq!(
            s.map.cell_at(1, 2, 0),
            CellKind::Portal,
            "space paints a Portal cell with the `0` brush"
        );
    }

    #[test]
    fn captured_suppresses_all_2d_edits() {
        // While the 3D preview is captured (free-look), `apply_key` must ignore
        // every 2D-edit key so a WASD/turn navigation keystream never paints, moves
        // the cursor, changes z, or opens a modal.
        let mut s = EditorState::new_empty(4, 4, 2);
        s.preview_captured = true;
        s.cursor = (2, 2);
        s.brush = CellKind::WallSolid;
        let z0 = s.z;
        apply_key(&mut s, key(E::Char(' '))); // would paint
        apply_key(&mut s, key(E::Left)); // would move cursor
        apply_key(&mut s, key(E::PageUp)); // would change z
        apply_key(&mut s, key(E::Char('?'))); // would open help
        assert_eq!(
            s.map.cell_at(2, 2, 0),
            CellKind::Empty,
            "no paint while captured"
        );
        assert_eq!(s.cursor, (2, 2), "cursor frozen while captured");
        assert_eq!(s.z, z0, "z frozen while captured");
        assert!(!s.help_open, "no modal opens while captured");
        assert!(!s.dirty, "captured navigation never dirties the document");
    }

    #[test]
    fn captured_s_does_not_save() {
        // The key bug: while captured, lowercase `s` must NOT raise the SAVE flag
        // (it is the WASD "back" key during free-look, handled in camera_input_system).
        let mut s = EditorState::new_empty(4, 4, 2);
        s.preview_captured = true;
        apply_key(&mut s, key(E::Char('s')));
        assert_ne!(s.status, "SAVE", "captured `s` must not trigger save");
        // And when NOT captured, `s` still arms save (the 2D-edit behaviour is intact).
        let mut s2 = EditorState::new_empty(4, 4, 2);
        apply_key(&mut s2, key(E::Char('s')));
        assert_eq!(s2.status, "SAVE", "uncaptured `s` still saves");
    }
    #[test]
    fn arrows_move_and_clamp() {
        let mut s = EditorState::new_empty(4, 4, 2);
        apply_key(&mut s, key(E::Left)); // clamps at 0
        assert_eq!(s.cursor.0, 0);
        apply_key(&mut s, key(E::Right));
        assert_eq!(s.cursor.0, 1);
    }
    #[test]
    fn pageup_changes_z_clamped() {
        let mut s = EditorState::new_empty(4, 4, 2);
        apply_key(&mut s, key(E::PageUp));
        assert_eq!(s.z, 1);
        apply_key(&mut s, key(E::PageUp)); // clamp at sz-1
        assert_eq!(s.z, 1);
    }
    #[test]
    fn n_tags_object_with_side_data() {
        let mut s = EditorState::new_empty(4, 4, 2);
        s.cursor = (1, 1);
        apply_key(&mut s, key(E::Char('n')));
        assert_eq!(s.map.cell_at(1, 1, 0), CellKind::Object);
        assert_eq!(s.map.side_data_at(1, 1, 0), 1);
    }

    #[test]
    fn zoom_keys_clamp_to_min_max() {
        use crate::view::ZOOM_MAX;
        let mut s = EditorState::new_empty(8, 8, 2);
        assert_eq!(s.zoom, 1);
        apply_key(&mut s, key(E::Char('-'))); // clamp at 1
        assert_eq!(s.zoom, 1);
        apply_key(&mut s, key(E::Char('+')));
        assert_eq!(s.zoom, 2);
        apply_key(&mut s, key(E::Char('=')));
        assert_eq!(s.zoom, 3);
        // Step all the way up to the (raised) max and confirm it clamps there.
        for _ in 0..10 {
            apply_key(&mut s, key(E::Char('+')));
        }
        assert_eq!(s.zoom, ZOOM_MAX, "zoom clamps at the raised max");
        apply_key(&mut s, key(E::Char('-')));
        assert_eq!(s.zoom, ZOOM_MAX - 1);
    }

    #[test]
    fn shape_brush_paints_solid_base_plus_shape_and_marks_dirty() {
        use crate::state::SHAPE_CYCLE;
        let mut s = EditorState::new_empty(4, 4, 2);
        s.cursor = (2, 1);
        apply_key(&mut s, key(E::Char('7'))); // enter shape mode
        assert!(s.shape_mode);
        assert!(!s.dirty);
        apply_key(&mut s, key(E::Char(' '))); // paint the selected shape
        assert_eq!(
            s.map.cell_at(2, 1, 0),
            CellKind::WallSolid,
            "solid base set"
        );
        assert_eq!(s.map.shape_at(2, 1, 0), SHAPE_CYCLE[0], "shape set");
        assert!(s.dirty, "shape paint marks dirty");
    }

    #[test]
    fn orientation_cycle_steps_through_all_eight() {
        use crate::state::SHAPE_CYCLE;
        let mut s = EditorState::new_empty(4, 4, 2);
        apply_key(&mut s, key(E::Char('7')));
        assert_eq!(s.shape_sel, SHAPE_CYCLE[0]);
        // `]` advances one orientation and stays in shape mode.
        apply_key(&mut s, key(E::Char(']')));
        assert_eq!(s.shape_sel, SHAPE_CYCLE[1]);
        assert!(s.shape_mode);
        // `[` steps back.
        apply_key(&mut s, key(E::Char('[')));
        assert_eq!(s.shape_sel, SHAPE_CYCLE[0]);
        // `[` from index 0 wraps to the last (a ramp).
        apply_key(&mut s, key(E::Char('[')));
        assert_eq!(s.shape_sel, SHAPE_CYCLE[7]);
    }

    #[test]
    fn erase_clears_shape_back_to_full() {
        use crate::state::SHAPE_CYCLE;
        let mut s = EditorState::new_empty(4, 4, 2);
        s.cursor = (1, 1);
        apply_key(&mut s, key(E::Char('7')));
        apply_key(&mut s, key(E::Char(' ')));
        assert_ne!(s.map.shape_at(1, 1, 0), Shape::Full);
        apply_key(&mut s, key(E::Char('x')));
        assert_eq!(s.map.cell_at(1, 1, 0), CellKind::Empty);
        assert_eq!(s.map.shape_at(1, 1, 0), Shape::Full, "erase resets shape");
        // sanity: not accidentally painting the same shape index forever.
        let _ = SHAPE_CYCLE;
    }

    #[test]
    fn plain_brush_paint_clears_prior_shape() {
        let mut s = EditorState::new_empty(4, 4, 2);
        s.cursor = (0, 0);
        // First paint a wedge, then a plain brush over it.
        apply_key(&mut s, key(E::Char('7')));
        apply_key(&mut s, key(E::Char(' ')));
        assert_ne!(s.map.shape_at(0, 0, 0), Shape::Full);
        apply_key(&mut s, key(E::Char('1'))); // WallSolid plain brush, leaves shape mode
        assert!(!s.shape_mode);
        apply_key(&mut s, key(E::Char(' ')));
        assert_eq!(
            s.map.shape_at(0, 0, 0),
            Shape::Full,
            "plain paint clears shape"
        );
    }

    #[test]
    fn selecting_plain_brush_exits_shape_mode() {
        let mut s = EditorState::new_empty(4, 4, 2);
        apply_key(&mut s, key(E::Char('7')));
        assert!(s.shape_mode);
        apply_key(&mut s, key(E::Char('2')));
        assert!(!s.shape_mode);
        assert_eq!(s.brush, CellKind::Floor);
    }

    #[test]
    fn hash_toggles_stairwell_autocarve_only_in_stair_mode() {
        let mut s = EditorState::new_empty(8, 8, 8);
        assert!(s.auto_carve_stairwell, "default ON");
        // Outside the stair tool, `#` is inert (no guard match).
        apply_key(&mut s, key(E::Char('#')));
        assert!(s.auto_carve_stairwell, "# inert outside stair mode");
        // Enter the stair tool, then `#` toggles OFF and back ON.
        apply_key(&mut s, key(E::Char('8')));
        apply_key(&mut s, key(E::Char('#')));
        assert!(!s.auto_carve_stairwell, "# toggles auto-carve OFF");
        assert!(s.status.contains("OFF"), "status reflects OFF");
        apply_key(&mut s, key(E::Char('#')));
        assert!(s.auto_carve_stairwell, "# toggles auto-carve back ON");
        assert!(s.status.contains("ON"), "status reflects ON");
    }

    #[test]
    fn show_objects_defaults_true_and_h_toggles_it() {
        let mut s = EditorState::new_empty(4, 4, 2);
        assert!(s.show_objects, "objects shown by default");
        apply_key(&mut s, key(E::Char('H')));
        assert!(!s.show_objects, "H hides objects");
        assert_eq!(s.status, "objects hidden");
        apply_key(&mut s, key(E::Char('H')));
        assert!(s.show_objects, "H shows objects again");
        assert_eq!(s.status, "objects shown");
        // It is a pure view toggle — it must never dirty the document.
        assert!(
            !s.dirty,
            "toggling object visibility does not dirty the map"
        );
    }

    #[test]
    fn tab_cycles_view_mode() {
        use crate::state::ViewMode;
        let mut s = EditorState::new_empty(4, 4, 2);
        assert_eq!(s.view_mode, ViewMode::Split);
        apply_key(&mut s, key(E::Tab));
        assert_eq!(s.view_mode, ViewMode::Full2D);
        apply_key(&mut s, key(E::Tab));
        assert_eq!(s.view_mode, ViewMode::Full3D);
        apply_key(&mut s, key(E::Tab));
        assert_eq!(s.view_mode, ViewMode::Split);
    }

    #[test]
    fn paint_then_undo_restores_map_and_redo_reapplies() {
        let mut s = EditorState::new_empty(4, 4, 2);
        s.cursor = (1, 1);
        s.brush = CellKind::WallSolid;
        assert_eq!(s.map.cell_at(1, 1, 0), CellKind::Empty);
        apply_key(&mut s, key(E::Char(' '))); // paint
        assert_eq!(s.map.cell_at(1, 1, 0), CellKind::WallSolid);
        apply_key(&mut s, key(E::Char('u'))); // undo
        assert_eq!(
            s.map.cell_at(1, 1, 0),
            CellKind::Empty,
            "undo reverts the paint"
        );
        apply_key(&mut s, key(E::Char('y'))); // redo
        assert_eq!(
            s.map.cell_at(1, 1, 0),
            CellKind::WallSolid,
            "redo re-applies"
        );
    }

    #[test]
    fn noop_checkpoint_key_preserves_redo_stack() {
        // B1 regression: an innocent no-op `c` (nothing selected) must NOT wipe the
        // redo branch, and must leave no dead undo step.
        let mut s = EditorState::new_empty(4, 4, 2);
        s.cursor = (1, 1);
        s.brush = CellKind::WallSolid;
        apply_key(&mut s, key(E::Char(' '))); // paint        → undo:1 redo:0
        apply_key(&mut s, key(E::Char('u'))); // undo         → undo:0 redo:1
        assert_eq!(s.history.redo_len(), 1);
        apply_key(&mut s, key(E::Char('c'))); // no-op (no selection)
        assert_eq!(s.history.redo_len(), 1, "no-op c must not wipe redo");
        assert_eq!(s.history.undo_len(), 0, "no-op c leaves no dead undo step");
        apply_key(&mut s, key(E::Char('y'))); // redo still works
        assert_eq!(
            s.map.cell_at(1, 1, 0),
            CellKind::WallSolid,
            "redo re-applies the paint"
        );
    }

    #[test]
    fn real_checkpoint_edit_clears_redo() {
        // The flip side of B1: a checkpoint edit that REALLY changes the document
        // invalidates the redo branch, exactly like a plain push_undo edit.
        let mut s = editor_with_selected_emitter();
        s.emitter_field = crate::state::EmitterField::Kind;
        apply_key(&mut s, key(E::Char('c'))); // real cycle  → undo:1 redo:0
        assert_eq!(s.history.undo_len(), 1);
        apply_key(&mut s, key(E::Char('u'))); // undo        → redo:1
        assert_eq!(s.history.redo_len(), 1);
        apply_key(&mut s, key(E::Char('c'))); // real cycle again
        assert_eq!(
            s.history.redo_len(),
            0,
            "a real checkpoint edit invalidates redo"
        );
    }

    #[test]
    fn undo_with_empty_stack_is_noop() {
        let mut s = EditorState::new_empty(4, 4, 2);
        apply_key(&mut s, key(E::Char('u')));
        assert_eq!(s.status, "nothing to undo");
        apply_key(&mut s, key(E::Char('y')));
        assert_eq!(s.status, "nothing to redo");
    }

    #[test]
    fn new_edit_clears_redo_stack() {
        let mut s = EditorState::new_empty(4, 4, 2);
        s.cursor = (0, 0);
        s.brush = CellKind::WallSolid;
        apply_key(&mut s, key(E::Char(' '))); // edit A
        apply_key(&mut s, key(E::Char('u'))); // undo → redo has 1
        assert_eq!(s.history.redo_len(), 1);
        s.cursor = (2, 2);
        apply_key(&mut s, key(E::Char(' '))); // a NEW edit
        assert_eq!(s.history.redo_len(), 0, "new edit clears redo");
    }

    #[test]
    fn history_is_capped() {
        use crate::history::HISTORY_CAP;
        let mut s = EditorState::new_empty(64, 4, 1);
        s.brush = CellKind::WallSolid;
        // Make more edits than the cap; the undo stack must not exceed it.
        for x in 0..(HISTORY_CAP + 10) {
            s.cursor = (x % 64, 0);
            apply_key(&mut s, key(E::Char(' ')));
        }
        assert_eq!(s.history.undo_len(), HISTORY_CAP, "undo stack capped");
    }

    #[test]
    fn rect_fill_is_one_undoable_edit() {
        let mut s = EditorState::new_empty(8, 8, 1);
        s.brush = CellKind::WallSolid;
        s.cursor = (1, 1);
        apply_key(&mut s, key(E::Char('k'))); // corner A at (1,1)
        s.cursor = (3, 3);
        apply_key(&mut s, key(E::Char('K'))); // rect fill A→(3,3)
                                              // 3x3 = 9 cells painted.
        for y in 1..=3 {
            for x in 1..=3 {
                assert_eq!(s.map.cell_at(x, y, 0), CellKind::WallSolid);
            }
        }
        assert_eq!(s.history.undo_len(), 1, "fill is a SINGLE undo step");
        apply_key(&mut s, key(E::Char('u'))); // one undo reverts the WHOLE fill
        for y in 1..=3 {
            for x in 1..=3 {
                assert_eq!(s.map.cell_at(x, y, 0), CellKind::Empty);
            }
        }
    }

    #[test]
    fn line_fill_paints_and_single_undo_reverts() {
        let mut s = EditorState::new_empty(8, 8, 1);
        s.brush = CellKind::Floor;
        s.cursor = (0, 0);
        apply_key(&mut s, key(E::Char('k')));
        s.cursor = (4, 0);
        apply_key(&mut s, key(E::Char('L')));
        for x in 0..=4 {
            assert_eq!(s.map.cell_at(x, 0, 0), CellKind::Floor);
        }
        apply_key(&mut s, key(E::Char('u')));
        for x in 0..=4 {
            assert_eq!(s.map.cell_at(x, 0, 0), CellKind::Empty);
        }
    }

    #[test]
    fn bucket_fill_via_key_floods_region() {
        let mut s = EditorState::new_empty(4, 4, 1);
        s.brush = CellKind::WallSolid;
        s.cursor = (0, 0);
        apply_key(&mut s, key(E::Char('J')));
        // Whole empty slice becomes solid.
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(s.map.cell_at(x, y, 0), CellKind::WallSolid);
            }
        }
        assert_eq!(s.history.undo_len(), 1);
    }

    #[test]
    fn rect_without_anchor_is_inert() {
        let mut s = EditorState::new_empty(4, 4, 1);
        apply_key(&mut s, key(E::Char('K')));
        assert!(!s.dirty, "no anchor ⇒ rect does nothing");
        assert_eq!(s.history.undo_len(), 0);
    }

    #[test]
    fn p_key_starts_and_cancels_pipe_create() {
        use crate::state::EntityKind;
        let mut s = EditorState::new_empty(12, 12, 2);
        s.level.push_node(1, 1, 0); // node 0
        s.cursor = (1, 1);
        s.z = 0;
        apply_key(&mut s, key(E::Char('P'))); // start: picks source node 0
        assert_eq!(s.place, Some(EntityKind::Pipe));
        assert_eq!(s.pipe_pending, Some(0));
        apply_key(&mut s, key(E::Char('P'))); // press again → cancel
        assert!(
            s.pipe_pending.is_none(),
            "second P cancels the pending source"
        );
    }

    #[test]
    fn space_completes_pipe_between_two_nodes() {
        let mut s = EditorState::new_empty(12, 12, 2);
        s.level.push_node(1, 1, 0); // node 0
        s.level.push_node(4, 3, 0); // node 1
        s.cursor = (1, 1);
        s.z = 0;
        apply_key(&mut s, key(E::Char('P'))); // source = node 0
        assert_eq!(s.pipe_pending, Some(0));
        s.cursor = (4, 3); // target node 1
        apply_key(&mut s, key(E::Char(' '))); // connect
        assert_eq!(
            s.level.pipes.len(),
            1,
            "pipe wired via space on the target node"
        );
        assert!(s.pipe_pending.is_none());
        assert_eq!(s.selected_pipe, Some(0));
    }

    #[test]
    fn g_key_toggles_pipe_switch_at_cursor() {
        use crate::state::EntityKind;
        let mut s = EditorState::new_empty(12, 12, 2);
        s.level.push_node(1, 1, 0);
        s.level.push_node(4, 1, 0);
        s.level.push_pipe_between(0, 1).unwrap();
        s.place = Some(EntityKind::Pipe);
        s.selected_pipe = Some(0);
        // Cursor on a path cell (2,1,0).
        s.cursor = (2, 1);
        s.z = 0;
        apply_key(&mut s, key(E::Char('g'))); // set switch here
        assert!(s.level.pipes[0].switch_cell.is_some());
        apply_key(&mut s, key(E::Char('g'))); // clear it
        assert!(s.level.pipes[0].switch_cell.is_none());
    }

    #[test]
    fn wizard_open_create_replaces_level() {
        let mut s = EditorState::new_empty(16, 16, 4);
        apply_key(&mut s, key(E::Char('W'))); // open
        assert!(s.wizard.is_some());
        // Tab to size Y and bump it; then create.
        // Default 16x16x4; shrink X by stepping down a couple times.
        for _ in 0..6 {
            apply_key(&mut s, key(E::Char('-'))); // size X 16 → 10
        }
        apply_key(&mut s, key(E::Enter)); // create
        assert!(s.wizard.is_none(), "wizard closed after create");
        assert_eq!(s.map.size().0, 10, "X shrunk via the form");
        assert!(s.dirty);
    }

    #[test]
    fn help_toggles_open_then_closed() {
        let mut s = EditorState::new_empty(8, 8, 2);
        assert!(!s.help_open);
        apply_key(&mut s, key(E::Char('?'))); // open
        assert!(s.help_open, "? opens help");
        apply_key(&mut s, key(E::Char('?'))); // ? closes (modal guard)
        assert!(!s.help_open, "? closes help when open");
        // Document the Esc-closes path: Esc is handled in `quit_on_esc`, not here,
        // so opening then sending Esc through that system closes it. We assert the
        // toggle invariant the modal guard enforces.
    }

    /// The sheet SCROLLS: a line at a time, a page at a time, and to the ends —
    /// and every one of them clamps to the last page, so no key can scroll the
    /// window off the end of the document into blank rows.
    #[test]
    fn help_keys_scroll_and_clamp() {
        let mut s = EditorState::new_empty(8, 8, 2);
        s.open_help();
        s.help_page = 35; // the body height a 40-row terminal paints
        let total = crate::keymap::help_lines().len();
        let last_page = total - 35;

        apply_help_key(&mut s, key(E::Down));
        assert_eq!(s.help_scroll, 1, "Down scrolls one line");
        apply_help_key(&mut s, key(E::Up));
        assert_eq!(s.help_scroll, 0, "Up scrolls back");
        apply_help_key(&mut s, key(E::Up));
        assert_eq!(s.help_scroll, 0, "and cannot go above the first line");

        apply_help_key(&mut s, key(E::PageDown));
        assert_eq!(s.help_scroll, 35, "PageDown moves one PAINTED page");
        apply_help_key(&mut s, key(E::PageUp));
        assert_eq!(s.help_scroll, 0, "PageUp moves back a page");

        apply_help_key(&mut s, key(E::End));
        assert_eq!(
            s.help_scroll, last_page,
            "End lands on the last page, not past it"
        );
        for _ in 0..20 {
            apply_help_key(&mut s, key(E::Down));
        }
        assert_eq!(
            s.help_scroll, last_page,
            "Down cannot walk past the last page"
        );
        apply_help_key(&mut s, key(E::Up));
        assert_eq!(
            s.help_scroll,
            last_page - 1,
            "so one Up moves visibly at once"
        );
        apply_help_key(&mut s, key(E::Home));
        assert_eq!(s.help_scroll, 0, "Home returns to the first line");
    }

    /// Typing FILTERS the sheet (the outliner's `/` idiom, minus the `/`),
    /// Backspace pops, and the offset returns to the top on every edit so a
    /// filtered sheet never opens on blank space.
    #[test]
    fn typing_filters_the_help_sheet() {
        let mut s = EditorState::new_empty(8, 8, 2);
        s.open_help();
        s.help_page = 35;
        apply_help_key(&mut s, key(E::End));
        assert!(s.help_scroll > 0, "premise: scrolled away from the top");

        for c in "mask".chars() {
            apply_help_key(&mut s, key(E::Char(c)));
        }
        assert_eq!(s.help_filter, "mask");
        assert_eq!(s.help_scroll, 0, "a filter edit returns to the first match");
        let filtered = crate::keymap::help_lines_filtered(&s.help_filter);
        assert!(
            filtered.len() < crate::keymap::help_lines().len(),
            "the sheet really narrows"
        );
        assert!(
            s.status.contains("mask"),
            "the status echoes the query: {}",
            s.status
        );

        apply_help_key(&mut s, key(E::Backspace));
        assert_eq!(s.help_filter, "mas", "Backspace pops one character");
        for _ in 0..3 {
            apply_help_key(&mut s, key(E::Backspace));
        }
        assert_eq!(s.help_filter, "", "and empties");
        apply_help_key(&mut s, key(E::Backspace));
        assert_eq!(s.help_filter, "", "popping an empty filter is a no-op");
        assert!(s.help_open, "none of that closed the sheet");
    }

    /// `?` still closes the sheet — from anywhere, filter or no filter. It is
    /// the one printable that does not type into the query, because it is the
    /// key the sheet's own footer tells the user to press.
    #[test]
    fn question_mark_closes_the_sheet_even_while_filtering() {
        let mut s = EditorState::new_empty(8, 8, 2);
        s.open_help();
        for c in "vox".chars() {
            apply_help_key(&mut s, key(E::Char(c)));
        }
        assert_eq!(s.help_filter, "vox");
        apply_help_key(&mut s, key(E::Char('?')));
        assert!(!s.help_open, "? closes the sheet");
        assert_eq!(s.help_filter, "", "and drops the filter with it");
        assert_eq!(s.help_scroll, 0, "and the scroll offset");
    }

    /// REOPENING is always a clean sheet: a stale offset from the previous visit
    /// would open the popup on blank rows, and a stale filter would open it
    /// looking like a sheet that had lost most of its rows.
    #[test]
    fn reopening_help_resets_scroll_and_filter() {
        let mut s = EditorState::new_empty(8, 8, 2);
        s.open_help();
        s.help_page = 35;
        apply_help_key(&mut s, key(E::End));
        apply_help_key(&mut s, key(E::Char('z')));
        assert!(s.help_scroll == 0 && !s.help_filter.is_empty());
        s.help_scroll = 120;
        apply_key(&mut s, key(E::Char('?'))); // the sheet's own close key
        assert!(!s.help_open);
        apply_key(&mut s, key(E::Char('?'))); // and open it again
        assert!(s.help_open, "? reopens it");
        assert_eq!(s.help_scroll, 0, "at the top");
        assert_eq!(s.help_filter, "", "unfiltered");
    }

    #[test]
    fn help_open_is_modal_no_paint() {
        let mut s = EditorState::new_empty(8, 8, 2);
        s.cursor = (2, 2);
        s.brush = CellKind::WallSolid;
        apply_key(&mut s, key(E::Char('?'))); // open help
        apply_key(&mut s, key(E::Char(' '))); // would paint normally
        apply_key(&mut s, key(E::Left)); // would move normally
        assert_eq!(s.cursor, (2, 2), "cursor frozen while help open");
        assert!(!s.dirty, "no paint while help open");
    }

    #[test]
    fn capital_o_opens_load_overlay_modal() {
        let mut s = EditorState::new_empty(8, 8, 2);
        s.cursor = (2, 2);
        apply_key(&mut s, key(E::Char('O'))); // open the file picker
        assert!(s.open.is_some(), "O opens the load overlay");
        // Modal: an arrow does NOT move the edit cursor (it navigates the list).
        apply_key(&mut s, key(E::Left));
        assert_eq!(s.cursor, (2, 2), "edit cursor frozen while open overlay up");
        // `O` again closes it.
        apply_key(&mut s, key(E::Char('O')));
        assert!(s.open.is_none(), "O closes the overlay");
    }

    #[test]
    fn capital_a_opens_saveas_prompt() {
        let mut s = EditorState::new_empty(8, 8, 2);
        apply_key(&mut s, key(E::Char('A')));
        assert!(s.saveas.is_some(), "A opens the Save-As prompt");
        // Modal: typing a printable char appends to the filename buffer, not paint.
        s.cursor = (2, 2);
        s.wizard = None;
        let before = s.saveas.as_ref().unwrap().buf.clone();
        apply_key(&mut s, key(E::Char('z')));
        assert_eq!(
            s.saveas.as_ref().unwrap().buf,
            format!("{before}z"),
            "chars type into the buffer while the prompt is open"
        );
        assert_eq!(s.cursor, (2, 2), "edit cursor frozen while Save-As open");
    }

    #[test]
    fn non_esc_key_disarms_quit_guard() {
        let mut s = EditorState::new_empty(8, 8, 2);
        s.quit_armed = true;
        apply_key(&mut s, key(E::Right));
        assert!(
            !s.quit_armed,
            "any non-Esc key disarms the dirty-quit guard"
        );
    }

    #[test]
    fn wizard_name_field_accepts_free_text() {
        use crate::wizard::WizardField;
        let mut s = EditorState::new_empty(8, 8, 2);
        apply_key(&mut s, key(E::Char('W'))); // open wizard
                                              // Tab to the Name field (X→Y→Z→Name = 3 tabs).
        for _ in 0..3 {
            apply_key(&mut s, key(E::Tab));
        }
        assert_eq!(s.wizard.as_ref().unwrap().focus, WizardField::Name);
        // Clear the preset then type "hi".
        s.wizard.as_mut().unwrap().name_clear();
        apply_key(&mut s, key(E::Char('h')));
        apply_key(&mut s, key(E::Char('i')));
        assert!(
            s.wizard.as_ref().unwrap().name().ends_with("hi"),
            "typed chars append to the name buffer"
        );
    }

    #[test]
    fn wizard_navigation_keys_do_not_paint() {
        // With the wizard open, an arrow/space must not move the cursor or paint.
        let mut s = EditorState::new_empty(8, 8, 2);
        s.cursor = (2, 2);
        apply_key(&mut s, key(E::Char('W')));
        apply_key(&mut s, key(E::Char(' '))); // would paint normally
        apply_key(&mut s, key(E::Left));
        assert_eq!(s.cursor, (2, 2), "cursor frozen while wizard open");
        assert!(!s.dirty, "no paint while wizard open");
    }

    /// Put the editor in Emitter-placement mode with one emitter selected, so the
    /// direction-picker keymap tests have a target to aim.
    fn editor_with_selected_emitter() -> EditorState {
        let mut s = EditorState::new_empty(8, 8, 4);
        s.level.emitters.push(re_world::EmitterSpec::new());
        s.place = Some(crate::state::EntityKind::Emitter);
        s.selected_emitter = Some(0);
        s
    }

    #[test]
    fn capital_d_opens_direction_picker_on_selected_emitter() {
        let mut s = editor_with_selected_emitter();
        apply_key(&mut s, key(E::Char('D')));
        assert!(s.dir_picker.is_some(), "D opens the aim picker");
    }

    #[test]
    fn capital_d_without_emitter_is_a_hinted_noop() {
        let mut s = EditorState::new_empty(8, 8, 2);
        apply_key(&mut s, key(E::Char('D')));
        assert!(s.dir_picker.is_none(), "no emitter -> no picker");
    }

    #[test]
    fn picker_is_modal_no_paint_then_pick_sets_direction() {
        let mut s = editor_with_selected_emitter();
        s.cursor = (3, 3);
        apply_key(&mut s, key(E::Char('D'))); // open
        let before_cursor = s.cursor;
        // While open, arrows move the CUBE cursor, never the edit cursor, and
        // space does NOT paint geometry.
        apply_key(&mut s, key(E::Char(' '))); // would paint normally — but PICKS
                                              // (space picks + closes, so re-open to test the modal-no-paint property)
        assert!(s.dir_picker.is_none(), "space picks + closes the picker");
        assert_eq!(
            s.cursor, before_cursor,
            "edit cursor frozen while picker open"
        );

        // Now aim deliberately: open, move to the +X (east) cell on the mid layer,
        // pick, and assert direction == (1,0,0).
        apply_key(&mut s, key(E::Char('D'))); // re-open (starts on current aim cell)
                                              // Force the cube cursor to the centre then step east once.
        if let Some(p) = s.dir_picker.as_mut() {
            p.cx = 0;
            p.cy = 0;
            p.cz = 0;
        }
        apply_key(&mut s, key(E::Right)); // dx -> +1 (east, '→')
        apply_key(&mut s, key(E::Enter)); // pick
        assert!(s.dir_picker.is_none(), "Enter picks + closes");
        let dir = s.level.emitters[0].direction;
        assert!(
            (dir.x - 1.0).abs() < 1e-5 && dir.y.abs() < 1e-5 && dir.z.abs() < 1e-5,
            "picked east cell sets direction +X, got {dir:?}"
        );
        assert!(s.dirty, "picking an aim marks dirty");
    }

    #[test]
    fn picker_layer_switch_then_pick_sets_diagonal_up() {
        let mut s = editor_with_selected_emitter();
        apply_key(&mut s, key(E::Char('D')));
        if let Some(p) = s.dir_picker.as_mut() {
            p.cx = 0;
            p.cy = 0;
            p.cz = 0;
        }
        apply_key(&mut s, key(E::Char(']'))); // top layer (dz = +1)
        apply_key(&mut s, key(E::Right)); // dx = +1
        apply_key(&mut s, key(E::Down)); // dy = +1
        apply_key(&mut s, key(E::Enter)); // pick (1,1,1) normalized
        let dir = s.level.emitters[0].direction;
        let expect = glam::Vec3::new(1.0, 1.0, 1.0).normalize();
        assert!(
            (dir - expect).length() < 1e-5,
            "got {dir:?}, want {expect:?}"
        );
    }

    // ── commit-if-changed (no dead undo steps for c / + / -) ──

    #[test]
    fn c_in_noop_context_leaves_undo_unchanged() {
        // Fresh state: no place, no mode ⇒ `c` routes to entity::cycle_property
        // with place=None, a documented no-op. No undo entry must be stacked.
        let mut s = EditorState::new_empty(8, 8, 2);
        assert_eq!(s.history.undo_len(), 0);
        apply_key(&mut s, key(E::Char('c')));
        assert_eq!(
            s.history.undo_len(),
            0,
            "no-op `c` must not stack an undo step"
        );
        // And repeated presses still never grow the stack.
        apply_key(&mut s, key(E::Char('c')));
        apply_key(&mut s, key(E::Char('c')));
        assert_eq!(
            s.history.undo_len(),
            0,
            "exploring with `c` stacks no dead steps"
        );
    }

    #[test]
    fn c_real_cycle_increments_undo_by_one() {
        use crate::state::EmitterField;
        let mut s = editor_with_selected_emitter();
        s.emitter_field = EmitterField::Kind;
        assert_eq!(s.history.undo_len(), 0);
        apply_key(&mut s, key(E::Char('c'))); // Standard -> GodRays (a real change)
        assert_eq!(
            s.history.undo_len(),
            1,
            "a real `c` cycle is exactly one undo step"
        );
        assert_eq!(s.level.emitters[0].kind, re_world::EmitterKind::GodRays);
    }

    #[test]
    fn plus_on_numeric_field_increments_undo_but_enum_field_does_not() {
        use crate::state::EmitterField;
        let mut s = editor_with_selected_emitter();
        // A numeric field (rate): `+` changes the document → one undo step.
        s.emitter_field = EmitterField::Rate;
        let before_rate = s.level.emitters[0].rate;
        apply_key(&mut s, key(E::Char('+')));
        assert!(s.level.emitters[0].rate > before_rate, "rate increased");
        assert_eq!(s.history.undo_len(), 1, "a real `+` is one undo step");
        // An enum field (kind): `+` is a documented no-op → NO new undo step.
        s.emitter_field = EmitterField::Kind;
        apply_key(&mut s, key(E::Char('+')));
        assert_eq!(
            s.history.undo_len(),
            1,
            "`+` on an enum field changes nothing and stacks no undo step"
        );
    }

    // ── mode exclusivity via enter_tool_mode ──

    #[test]
    fn selecting_shape_brush_clears_selected_emitter() {
        // Put the editor in Emitter placement with one selected, then press `7`.
        let mut s = editor_with_selected_emitter();
        assert_eq!(s.place, Some(crate::state::EntityKind::Emitter));
        assert_eq!(s.selected_emitter, Some(0));
        apply_key(&mut s, key(E::Char('7'))); // shape brush
        assert!(s.shape_mode, "7 enters shape mode");
        assert_eq!(s.place, None, "shape mode is not entity placement");
        assert_eq!(s.selected_emitter, None, "stale emitter selection cleared");
    }

    #[test]
    fn v_clears_material_mode() {
        // The old `v` arm forgot to clear material_mode; enter_tool_mode fixes it.
        let mut s = EditorState::new_empty(8, 8, 2);
        s.material_mode = true;
        apply_key(&mut s, key(E::Char('v')));
        assert!(s.vox_mode, "v enters vox mode");
        assert!(!s.material_mode, "v clears material mode");
    }

    // ── Shift+Arrow pans, plain arrow moves the cursor ──

    #[test]
    fn shift_right_pans_and_plain_right_moves_cursor() {
        let mut s = EditorState::new_empty(32, 32, 2);
        // A measured view window so the pan clamp has room to scroll.
        s.view_cells = (10, 10);
        s.zoom = 1;
        s.cursor = (5, 5);
        let cursor_before = s.cursor;
        // Shift+Right pans, leaving the cursor put.
        apply_key(
            &mut s,
            EditorKey {
                code: E::Right,
                shift: true,
            },
        );
        assert_eq!(s.pan.0, 1, "Shift+Right increases pan.0");
        assert_eq!(
            s.cursor, cursor_before,
            "Shift+Right does not move the cursor"
        );
        // Plain Right moves the cursor, leaving the pan put.
        let pan_before = s.pan;
        apply_key(&mut s, key(E::Right));
        assert_eq!(
            s.cursor.0,
            cursor_before.0 + 1,
            "plain Right moves the cursor"
        );
        assert_eq!(s.pan, pan_before, "plain Right does not pan");
    }

    #[test]
    fn shift_pan_clamps_to_world_window() {
        let mut s = EditorState::new_empty(4, 4, 2);
        s.view_cells = (4, 4); // whole 4-cell world fits ⇒ max_pan = 0.
        s.zoom = 1;
        apply_key(
            &mut s,
            EditorKey {
                code: E::Right,
                shift: true,
            },
        );
        assert_eq!(s.pan.0, 0, "pan clamps at 0 when the world fully fits");
    }

    // ── Backspace in the Save-As prompt ──

    #[test]
    fn saveas_backspace_removes_last_char() {
        let mut s = EditorState::new_empty(8, 8, 2);
        apply_key(&mut s, key(E::Char('A'))); // open Save-As
        s.saveas.as_mut().unwrap().buf.clear();
        apply_key(&mut s, key(E::Char('a')));
        apply_key(&mut s, key(E::Char('b')));
        assert_eq!(s.saveas.as_ref().unwrap().buf, "ab");
        apply_key(&mut s, key(E::Backspace));
        assert_eq!(
            s.saveas.as_ref().unwrap().buf,
            "a",
            "Backspace pops the last char"
        );
        // Backspace on an empty buffer is a no-op (no panic).
        apply_key(&mut s, key(E::Backspace));
        apply_key(&mut s, key(E::Backspace));
        assert_eq!(s.saveas.as_ref().unwrap().buf, "");
    }

    // ── Backspace in the wizard Name field ──

    #[test]
    fn wizard_name_backspace_removes_last_char() {
        use crate::wizard::WizardField;
        let mut s = EditorState::new_empty(8, 8, 2);
        apply_key(&mut s, key(E::Char('W'))); // open wizard
        for _ in 0..3 {
            apply_key(&mut s, key(E::Tab)); // focus Name
        }
        assert_eq!(s.wizard.as_ref().unwrap().focus, WizardField::Name);
        s.wizard.as_mut().unwrap().name_clear();
        apply_key(&mut s, key(E::Char('h')));
        apply_key(&mut s, key(E::Char('i')));
        assert_eq!(s.wizard.as_ref().unwrap().name(), "hi");
        apply_key(&mut s, key(E::Backspace));
        assert_eq!(
            s.wizard.as_ref().unwrap().name_buf,
            "h",
            "Backspace pops the last char of the name buffer"
        );
    }

    #[test]
    fn emitter_kind_field_cycles_standard_godrays() {
        use crate::state::EmitterField;
        let mut s = editor_with_selected_emitter();
        s.emitter_field = EmitterField::Kind;
        assert_eq!(s.level.emitters[0].kind, re_world::EmitterKind::Standard);
        apply_key(&mut s, key(E::Char('c'))); // cycle the focused enum field
        assert_eq!(s.level.emitters[0].kind, re_world::EmitterKind::GodRays);
        apply_key(&mut s, key(E::Char('c')));
        assert_eq!(s.level.emitters[0].kind, re_world::EmitterKind::Standard);
    }

    // ── panels / palette / validation bindings ──────────────────────────────

    /// `:` opens the COMMAND PALETTE; while open it is the topmost focus frame,
    /// so a further printable key types into the query, not the grid.
    #[test]
    fn colon_opens_command_palette_and_captures_keys() {
        let mut s = EditorState::new_empty(4, 4, 2);
        s.cursor = (2, 2);
        apply_key(&mut s, key(E::Char(':')));
        assert!(s.cmd_palette.is_some(), "':' opens the palette");
        apply_key(&mut s, key(E::Char(' ')));
        assert_eq!(
            s.map.cell_at(2, 2, 0),
            CellKind::Empty,
            "space typed into the query — never painted the grid"
        );
        assert_eq!(s.cmd_palette.as_ref().unwrap().query, " ");
    }

    /// `z` ZOOMS the focused viewport and the SAME key restores.
    #[test]
    fn z_zooms_focused_pane_and_restores() {
        let mut s = EditorState::new_empty(4, 4, 2);
        apply_key(&mut s, key(E::Char('z')));
        assert_eq!(
            s.zoomed,
            Some(crate::layout::PanelId::Grid),
            "grid focus zooms the grid"
        );
        apply_key(&mut s, key(E::Char('z')));
        assert_eq!(s.zoomed, None, "same key restores");
        s.focused_viewport = crate::state::ViewportFocus::Preview;
        apply_key(&mut s, key(E::Char('z')));
        assert_eq!(
            s.zoomed,
            Some(crate::layout::PanelId::Preview),
            "preview focus zooms it"
        );
    }

    /// `V` toggles the read-only VALIDATION DRAWER; while open it consumes
    /// ONLY its scroll/close keys — editing keys still work underneath.
    #[test]
    fn v_toggles_validation_drawer_without_blocking_edits() {
        let mut s = EditorState::new_empty(4, 4, 2);
        apply_key(&mut s, key(E::Char('V')));
        assert!(s.validation_open, "V opens the drawer");
        s.cursor = (1, 1);
        apply_key(&mut s, key(E::Char(' ')));
        assert_eq!(
            s.map.cell_at(1, 1, 0),
            CellKind::WallSolid,
            "painting still works under the open drawer"
        );
        apply_key(&mut s, key(E::Char('V')));
        assert!(!s.validation_open, "V closes it again");
    }

    /// `/` in the outliner opens a LIVE FILTER: typed chars narrow the rows
    /// (fuzzy), Enter selects the highlighted match and closes the filter.
    #[test]
    fn slash_filters_outliner_rows_live_and_enter_selects() {
        let mut s = EditorState::new_empty(8, 8, 2);
        s.level.push_node(1, 1, 0);
        s.level.push_node(5, 5, 0);
        s.level.nodes[0].name = "alpha".into();
        s.level.nodes[1].name = "furnace".into();
        s.tree.open = true;
        apply_key(&mut s, key(E::Char('/')));
        assert_eq!(
            s.tree.filter.as_deref(),
            Some(""),
            "'/' opens an empty filter"
        );
        for c in "furn".chars() {
            apply_key(&mut s, key(E::Char(c)));
        }
        assert_eq!(s.tree.filter.as_deref(), Some("furn"));
        let rows = tree::build_tree(&s);
        let objects: Vec<String> = rows
            .iter()
            .filter_map(|r| match r {
                crate::tree::TreeRow::Object { text, .. } => Some(text.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(objects.len(), 1, "filter keeps only the match: {objects:?}");
        assert!(objects[0].contains("furnace"));
        // The command keys type literally while filtering (no accidental delete).
        let nodes_before = s.level.nodes.len();
        apply_key(&mut s, key(E::Char('d')));
        assert_eq!(
            s.level.nodes.len(),
            nodes_before,
            "'d' typed into the filter"
        );
        apply_key(&mut s, key(E::Backspace));
        assert_eq!(
            s.tree.filter.as_deref(),
            Some("furn"),
            "Backspace popped the 'd'"
        );
        // Navigate to the (only) object row and select it.
        apply_key(&mut s, key(E::Down));
        apply_key(&mut s, key(E::Down));
        apply_key(&mut s, key(E::Enter));
        assert!(s.tree.filter.is_none(), "Enter closes the filter");
        assert_eq!(s.selected_node, Some(1), "the furnace node got selected");
    }

    // ── ENTER opens the focused row's editor ────────────────────────────────

    /// A state with `kind` placed + selected and `field` focused.
    fn row_state(kind: crate::state::EntityKind, field: crate::state::EntityField) -> EditorState {
        let mut s = EditorState::new_empty(16, 16, 4);
        s.cursor = (2, 2);
        s.z = 1;
        s.place = Some(kind);
        crate::entity::place_at_cursor(&mut s);
        s.entity_field = field;
        s
    }

    /// Enter on a focused NUMERIC row opens the inline numeric editor.
    #[test]
    fn enter_on_a_numeric_row_opens_the_inline_editor() {
        use crate::state::EntityField as F;
        let mut s = row_state(crate::state::EntityKind::Node, F::NodeThreshold);
        apply_key(&mut s, key(E::Enter));
        assert!(s.num_edit.is_some(), "Enter opened the numeric editor");
        assert!(s.enum_pick.is_none() && s.color_edit.is_none());
    }

    /// Enter on a focused ENUM row opens the option popup.
    #[test]
    fn enter_on_an_enum_row_opens_the_option_popup() {
        use crate::state::EntityField as F;
        let mut s = row_state(crate::state::EntityKind::Node, F::NodeIo);
        apply_key(&mut s, key(E::Enter));
        assert!(s.enum_pick.is_some(), "Enter opened the option popup");
    }

    /// Enter on a focused COLOUR row opens the colour overlay.
    #[test]
    fn enter_on_a_colour_row_opens_the_colour_overlay() {
        use crate::state::EntityField as F;
        let mut s = row_state(crate::state::EntityKind::Node, F::NodeColor);
        apply_key(&mut s, key(E::Enter));
        assert!(s.color_edit.is_some(), "Enter opened the colour overlay");
    }

    /// Enter on a focused BOOL row TOGGLES it in place — no overlay.
    #[test]
    fn enter_on_a_bool_row_toggles_in_place() {
        use crate::state::EntityField as F;
        let mut s = row_state(crate::state::EntityKind::Trigger, F::TrigOnce);
        let before = s.level.triggers[0].once;
        apply_key(&mut s, key(E::Enter));
        assert_ne!(s.level.triggers[0].once, before, "the flag toggled");
        assert!(s.num_edit.is_none() && s.enum_pick.is_none() && s.color_edit.is_none());
    }

    /// Enter on a plain TEXT row is inert — no overlay, no mutation.
    #[test]
    fn enter_on_a_text_row_is_inert() {
        use crate::state::EntityField as F;
        let mut s = row_state(crate::state::EntityKind::Node, F::NodePool);
        let before = s.level.nodes[0].pool.clone();
        apply_key(&mut s, key(E::Enter));
        assert!(s.num_edit.is_none() && s.enum_pick.is_none() && s.color_edit.is_none());
        assert_eq!(s.level.nodes[0].pool, before);
    }

    /// BARE DIGITS stay bound to the geometry brushes: opening a numeric editor
    /// must never be reachable by typing a number at the root, or `1`-`6`/`9`/`0`
    /// would be stolen from the brush set.
    #[test]
    fn bare_digits_at_root_still_set_the_geometry_brush() {
        use crate::state::EntityField as F;
        let mut s = row_state(crate::state::EntityKind::Node, F::NodeThreshold);
        for c in ['1', '2', '3', '4', '5', '6', '9', '0'] {
            apply_key(&mut s, key(E::Char(c)));
            assert!(
                s.num_edit.is_none(),
                "'{c}' must not open a number field at the root"
            );
        }
    }

    /// While a row editor is open, the ROOT chain never sees the key — a typed
    /// `s` goes into the number, it does not save the level.
    #[test]
    fn an_open_row_editor_swallows_root_keys() {
        use crate::state::EntityField as F;
        let mut s = row_state(crate::state::EntityKind::Node, F::NodeThreshold);
        apply_key(&mut s, key(E::Enter));
        assert!(s.num_edit.is_some());
        let before = s.status.clone();
        apply_key(&mut s, key(E::Char('s')));
        assert!(s.num_edit.is_some(), "the editor is still open");
        assert_eq!(s.status, before, "'s' did not reach the root save handler");
    }
}
