//! PIPE authoring (M4b): the two-node connect flow, the focused-field editor
//! (state / switch_at / density / color), pipe selection by a path cell, and
//! deletion.
//!
//! Pure functions over [`EditorState`] — no I/O, no terminal. Pipes live in the
//! typed `re_level::Level.pipes` field, which `re_voximport::to_lvl_text` already
//! serializes (state / path / switch_at / density / color), so wiring them here and
//! saving via `io::save_lvl_text` round-trips them with no extra format work.
//!
//! Validate-before-mutate: a self-pipe or a duplicate `A->B` is REFUSED (the level
//! is left untouched) with a status hint; the routing + the refusal both live in
//! `re_level::Level::push_pipe_between` (one source of truth with the `.lvl`
//! parser's `expand_path`). There is no panic and no half-wired pipe.
use crate::state::{EditorState, EntityField, EntityKind};
use re_level::PipeState;

/// Step `color` to the next/prev palette swatch. Delegates to
/// [`crate::material::step_palette`] (see there): on-palette values wrap ±1,
/// off-palette values land on the NEAREST swatch rather than snapping to white.
fn step_color(color: u32, delta: i32) -> u32 {
    crate::material::step_palette(color, delta)
}

/// Index of the pipe whose expanded `path` passes through cell `(x, y, z)`, if any.
/// The SELECTED pipe is preferred when several overlap (so the highlighted wire
/// stays selected when its path crosses another). Used by the top-down render and
/// by re-selecting a pipe from a path cell.
pub fn pipe_at(state: &EditorState, x: i32, y: i32, z: i32) -> Option<usize> {
    let on_path = |p: &re_level::Pipe| p.path.contains(&(x, y, z));
    if let Some(sel) = state.selected_pipe {
        if state.level.pipes.get(sel).is_some_and(on_path) {
            return Some(sel);
        }
    }
    state.level.pipes.iter().position(on_path)
}

/// `true` when this pipe is authored as a SPLINE (its `(from, to)` pair is in
/// [`EditorState::spline_pipes`]). The SSOT for the `path`/`spline` distinction —
/// every site that decides "spline or path" routes through here so there is one
/// source of truth (`re_level::Pipe` carries no flag).
pub fn is_spline(state: &EditorState, p: &re_level::Pipe) -> bool {
    state.spline_pipes.contains(&(p.from.clone(), p.to.clone()))
}

/// Find the node index at the cursor cell (the pipe-create flow connects nodes by
/// the cells they sit on). `None` when no node sits there.
fn node_at_cursor(state: &EditorState) -> Option<usize> {
    let (x, y, z) = (state.cursor.0 as i32, state.cursor.1 as i32, state.z as i32);
    state.level.node_at(x, y, z)
}

/// START a pipe-create: the next two node picks (cursor on a node + `space`, or a
/// tree Enter on a node) become the SOURCE then the TARGET. Requires the cursor to
/// be on a node (the source). A no-op (with a hint) when the cursor is not on a
/// node. Snapshots nothing yet — the pipe is appended only when the target is
/// picked (in [`pick_node`]).
pub fn start_pipe_create(state: &mut EditorState) {
    // Entering pipe-create leaves the other modes so `space` routes to `pick_node`.
    state.vox_mode = false;
    state.material_mode = false;
    state.selected_pipe = None;
    match node_at_cursor(state) {
        Some(src) => {
            state.pipe_pending = Some(src);
            let name = state.level.nodes[src].name.clone();
            state.status = format!(
                "pipe: source '{name}' — move to TARGET node, space to connect (Esc cancels)"
            );
        }
        None => {
            state.pipe_pending = None;
            state.status = "pipe: put the cursor on a SOURCE node first, then P".into();
        }
    }
}

/// Handle a node pick while a pipe-create is pending. Called with the node index
/// the user selected (cursor cell or tree). Picks the SOURCE if none is pending
/// yet (defensive — `start_pipe_create` normally sets it), else wires
/// SOURCE -> this node via [`re_level::Level::push_pipe_between`] (which refuses a
/// self-pipe / duplicate and auto-routes the L-shaped path). Returns `true` when a
/// pipe was actually appended.
pub fn pick_node(state: &mut EditorState, node: usize) -> bool {
    let Some(src) = state.pipe_pending else {
        // No source yet: this pick becomes the source.
        state.pipe_pending = Some(node);
        let name = state
            .level
            .nodes
            .get(node)
            .map(|n| n.name.clone())
            .unwrap_or_default();
        state.status = format!("pipe: source '{name}' — pick TARGET node");
        return false;
    };
    if src == node {
        state.status = "pipe: source = target (pick a DIFFERENT node)".into();
        return false;
    }
    state.push_undo();
    match state.level.push_pipe_between(src, node) {
        Some(pi) => {
            state.pipe_pending = None;
            state.selected_pipe = Some(pi);
            // Point the object selector at Pipe so the focused-field editors target
            // the new pipe immediately.
            state.place = Some(EntityKind::Pipe);
            state.entity_field =
                crate::state::entity_field_default(EntityKind::Pipe).unwrap_or(state.entity_field);
            state.mark_dirty();
            let p = &state.level.pipes[pi];
            state.status = format!(
                "pipe '{}'->'{}' wired ({} cells)",
                p.from,
                p.to,
                p.path.len()
            );
            true
        }
        None => {
            // Refused (duplicate A->B). Undo the just-pushed snapshot (nothing
            // changed) by popping it back off the stack, and keep the source pending
            // so the user can pick a different target.
            state.undo();
            let from = state
                .level
                .nodes
                .get(src)
                .map(|n| n.name.clone())
                .unwrap_or_default();
            let to = state
                .level
                .nodes
                .get(node)
                .map(|n| n.name.clone())
                .unwrap_or_default();
            state.status = format!("pipe '{from}'->'{to}' already exists (pick another target)");
            false
        }
    }
}

/// CANCEL an in-flight pipe-create (Esc handling routes here from the key layer).
/// A no-op when nothing is pending.
pub fn cancel(state: &mut EditorState) -> bool {
    if state.pipe_pending.take().is_some() {
        state.status = "pipe-create cancelled".into();
        true
    } else {
        false
    }
}

// ───────────────────────────── focused-field editing ─────────────────────────────

/// `c` on a Pipe focused field: toggle `state` (Off ↔ On) or set/clear the
/// `switch_at` cell. Density/color are numeric (`+`/`-`) and are hinted as such.
/// A no-op (with a hint) when no pipe is selected.
pub fn cycle_enum(state: &mut EditorState) {
    let field = state.entity_field;
    let cursor = (state.cursor.0 as i32, state.cursor.1 as i32, state.z as i32);
    let Some(idx) = state.selected_pipe.filter(|&i| i < state.level.pipes.len()) else {
        state.status = "no pipe selected".into();
        return;
    };
    match field {
        EntityField::PipeMode => {
            // Toggle path ↔ spline. SSOT: membership in `state.spline_pipes` keyed
            // by the pipe's (from, to) node pair. A spline needs ≥2 control points;
            // a wired pipe's expanded path always has ≥2 cells, so switching is
            // always valid — but guard defensively so a degenerate pipe can't
            // produce an unparseable `spline =` line.
            let p = &state.level.pipes[idx];
            let key = (p.from.clone(), p.to.clone());
            let points = p.path.len();
            if state.spline_pipes.contains(&key) {
                state.spline_pipes.remove(&key);
                state.mark_dirty();
                state.status = format!("pipe #{idx} mode PATH");
            } else if points >= 2 {
                state.spline_pipes.insert(key);
                state.mark_dirty();
                state.status = format!("pipe #{idx} mode SPLINE ({points} control points)");
            } else {
                state.status = "pipe spline needs ≥2 points (wire a longer pipe first)".into();
            }
        }
        EntityField::PipeState => {
            let p = &mut state.level.pipes[idx];
            p.state = match p.state {
                PipeState::Off => PipeState::On,
                PipeState::On => PipeState::Off,
            };
            let s = if p.state == PipeState::On {
                "ON"
            } else {
                "OFF"
            };
            state.mark_dirty();
            state.status = format!("pipe #{idx} state {s}");
        }
        EntityField::PipeSwitch => {
            let p = &mut state.level.pipes[idx];
            // If the cursor cell is on the path, SET the switch there. If the cursor
            // is already the switch cell (or off-path), CLEAR it. This makes `c` a
            // place-then-clear toggle keyed off the cursor.
            let cursor_on_path = p.path.iter().position(|&c| c == cursor);
            match (p.switch_cell, cursor_on_path) {
                (Some(cur), Some(pos)) if cur == pos => {
                    // Cursor is the current switch cell ⇒ clear.
                    p.switch_cell = None;
                    state.status = format!("pipe #{idx} switch cleared");
                }
                (_, Some(pos)) => {
                    p.switch_cell = Some(pos);
                    state.status = format!(
                        "pipe #{idx} switch @ {},{},{}",
                        cursor.0, cursor.1, cursor.2
                    );
                }
                (Some(_), None) => {
                    // Cursor off-path but a switch exists ⇒ clear it.
                    p.switch_cell = None;
                    state.status = format!("pipe #{idx} switch cleared (cursor off-path)");
                }
                (None, None) => {
                    state.status =
                        "pipe switch: put the cursor on a PATH cell of this pipe, then c".into();
                    return;
                }
            }
            state.mark_dirty();
        }
        _ => {
            state.status = format!("'{}' is numeric: use +/-", field.label());
        }
    }
}

/// `+`/`-` on a Pipe focused field: step density (clamped `0..=1`), the palette
/// colour, or move the switch ALONG the path. `state` is enum (hint to use `c`).
/// A no-op (with a hint) when no pipe is selected.
pub fn adjust_numeric(state: &mut EditorState, delta: i32) {
    let field = state.entity_field;
    if field == EntityField::PipeState {
        state.status = "'state' is an enum: use c".into();
        return;
    }
    let Some(idx) = state.selected_pipe.filter(|&i| i < state.level.pipes.len()) else {
        state.status = "no pipe selected".into();
        return;
    };
    let p = &mut state.level.pipes[idx];
    let msg = match field {
        EntityField::PipeDensity => {
            // Step by 0.1, clamped to [0,1].
            let d = (p.density + delta as f32 * 0.1).clamp(0.0, 1.0);
            // Round to one decimal so repeated steps stay clean (0.30000004 → 0.3).
            p.density = (d * 10.0).round() / 10.0;
            format!("pipe #{idx} density {:.1}", p.density)
        }
        EntityField::PipeColor => {
            p.color_rgb = step_color(p.color_rgb, delta);
            format!("pipe #{idx} color #{:06X}", p.color_rgb & 0xFF_FFFF)
        }
        EntityField::PipeSwitch => {
            if p.path.is_empty() {
                "pipe has no path".to_string()
            } else {
                let n = p.path.len() as i32;
                // Move the switch index along the path, wrapping. No switch yet ⇒
                // start at index 0.
                let cur = p.switch_cell.map(|c| c as i32).unwrap_or(0);
                let next = (cur + delta).rem_euclid(n) as usize;
                p.switch_cell = Some(next);
                let (cx, cy, cz) = p.path[next];
                format!("pipe #{idx} switch @ {cx},{cy},{cz}")
            }
        }
        _ => "use c for this field".to_string(),
    };
    state.mark_dirty();
    state.status = msg;
}

/// Render an ARBITRARY pipe field of the selected pipe (the row model: the
/// inspector lists every field, not just the focused one). `"-"` when no pipe is
/// selected or `field` is not a pipe field.
pub fn field_value_of(state: &EditorState, field: EntityField) -> String {
    let Some(idx) = state.selected_pipe.filter(|&i| i < state.level.pipes.len()) else {
        return "-".to_string();
    };
    let p = &state.level.pipes[idx];
    match field {
        EntityField::PipeMode => {
            if is_spline(state, p) {
                "spline".into()
            } else {
                "path".into()
            }
        }
        EntityField::PipeState => {
            if p.state == PipeState::On {
                "ON".into()
            } else {
                "OFF".into()
            }
        }
        EntityField::PipeSwitch => match p.switch_cell.and_then(|c| p.path.get(c)) {
            Some(&(x, y, z)) => format!("{x},{y},{z}"),
            None => "none".into(),
        },
        EntityField::PipeDensity => format!("{:.1}", p.density),
        EntityField::PipeColor => format!("#{:06X}", p.color_rgb & 0xFF_FFFF),
        _ => "-".into(),
    }
}

/// A compact one-line readout describing the selected pipe (for the panel header).
pub fn selected_summary(state: &EditorState) -> Option<String> {
    let idx = state.selected_pipe?;
    let p = state.level.pipes.get(idx)?;
    let s = if p.state == PipeState::On {
        "ON"
    } else {
        "OFF"
    };
    let sw = if p.switch_cell.is_some() {
        " switch"
    } else {
        ""
    };
    Some(format!("pipe '{}'->'{}' {s}{sw}", p.from, p.to))
}

#[cfg(test)]
mod tests {
    use super::*;
    /// The swatch list the pipe colour stepper shares with every other colour
    /// field — one colour vocabulary across the editor.
    use crate::material::PALETTE;

    /// A small level with two nodes the pipe flow can connect.
    fn two_node_state() -> EditorState {
        let mut s = EditorState::new_empty(16, 16, 4);
        s.level.push_node(1, 1, 0); // node 0
        s.level.push_node(4, 3, 0); // node 1
        s
    }

    #[test]
    fn create_flow_connects_two_nodes_with_auto_path() {
        let mut s = two_node_state();
        // Cursor on node 0 → start (picks source).
        s.cursor = (1, 1);
        s.z = 0;
        start_pipe_create(&mut s);
        assert_eq!(s.pipe_pending, Some(0));
        // Cursor on node 1 → pick target → pipe wired.
        s.cursor = (4, 3);
        let made = pick_node(&mut s, 1);
        assert!(made, "a pipe was wired");
        assert_eq!(s.level.pipes.len(), 1);
        let p = &s.level.pipes[0];
        assert_eq!((p.from_node, p.to_node), (0, 1));
        assert_eq!(p.path.first(), Some(&(1, 1, 0)));
        assert_eq!(p.path.last(), Some(&(4, 3, 0)));
        assert_eq!(s.selected_pipe, Some(0));
        assert_eq!(s.place, Some(EntityKind::Pipe));
        assert!(s.pipe_pending.is_none(), "pending cleared after wiring");
    }

    #[test]
    fn create_flow_refuses_self_pipe() {
        let mut s = two_node_state();
        s.pipe_pending = Some(0);
        let made = pick_node(&mut s, 0);
        assert!(!made);
        assert!(s.level.pipes.is_empty());
        assert!(s.status.contains("DIFFERENT"));
    }

    #[test]
    fn create_flow_refuses_duplicate() {
        let mut s = two_node_state();
        s.pipe_pending = Some(0);
        assert!(pick_node(&mut s, 1));
        // A second 0->1 is refused; the source stays pending so a retry is easy.
        s.pipe_pending = Some(0);
        let made = pick_node(&mut s, 1);
        assert!(!made);
        assert_eq!(s.level.pipes.len(), 1, "no duplicate appended");
        assert!(s.status.contains("already exists"));
    }

    #[test]
    fn cancel_clears_pending() {
        let mut s = two_node_state();
        s.pipe_pending = Some(0);
        assert!(cancel(&mut s));
        assert!(s.pipe_pending.is_none());
        // A second cancel is a no-op.
        assert!(!cancel(&mut s));
    }

    #[test]
    fn pipe_at_finds_path_cell() {
        let mut s = two_node_state();
        s.level.push_pipe_between(0, 1).unwrap();
        // A mid-path cell (along the x leg of the L) is on the pipe.
        assert_eq!(pipe_at(&s, 2, 1, 0), Some(0));
        // An off-path cell is not.
        assert_eq!(pipe_at(&s, 10, 10, 0), None);
    }

    #[test]
    fn mode_toggle_flips_path_and_spline() {
        let mut s = two_node_state();
        s.level.push_pipe_between(0, 1).unwrap();
        s.selected_pipe = Some(0);
        s.place = Some(EntityKind::Pipe);
        s.entity_field = EntityField::PipeMode;
        let p = &s.level.pipes[0];
        let key = (p.from.clone(), p.to.clone());
        // Default is PATH.
        assert!(!is_spline(&s, &s.level.pipes[0]));
        assert_eq!(field_value_of(&s, s.entity_field), "path");
        // Toggle → SPLINE.
        cycle_enum(&mut s);
        assert!(s.spline_pipes.contains(&key));
        assert_eq!(field_value_of(&s, s.entity_field), "spline");
        // Toggle → PATH.
        cycle_enum(&mut s);
        assert!(!s.spline_pipes.contains(&key));
        assert_eq!(field_value_of(&s, s.entity_field), "path");
    }

    #[test]
    fn edit_state_toggles() {
        let mut s = two_node_state();
        s.level.push_pipe_between(0, 1).unwrap();
        s.selected_pipe = Some(0);
        s.place = Some(EntityKind::Pipe);
        s.entity_field = EntityField::PipeState;
        assert_eq!(s.level.pipes[0].state, PipeState::Off);
        cycle_enum(&mut s);
        assert_eq!(s.level.pipes[0].state, PipeState::On);
        cycle_enum(&mut s);
        assert_eq!(s.level.pipes[0].state, PipeState::Off);
    }

    #[test]
    fn edit_switch_set_and_clear_at_cursor() {
        let mut s = two_node_state();
        s.level.push_pipe_between(0, 1).unwrap();
        s.selected_pipe = Some(0);
        s.entity_field = EntityField::PipeSwitch;
        // Cursor on a path cell sets the switch.
        s.cursor = (2, 1);
        s.z = 0;
        cycle_enum(&mut s);
        let pos = s.level.pipes[0]
            .path
            .iter()
            .position(|&c| c == (2, 1, 0))
            .unwrap();
        assert_eq!(s.level.pipes[0].switch_cell, Some(pos));
        // `c` again at the same cell clears it.
        cycle_enum(&mut s);
        assert_eq!(s.level.pipes[0].switch_cell, None);
    }

    #[test]
    fn edit_density_clamps_0_to_1() {
        let mut s = two_node_state();
        s.level.push_pipe_between(0, 1).unwrap();
        s.selected_pipe = Some(0);
        s.entity_field = EntityField::PipeDensity;
        // Default density is 1.0; stepping up clamps at 1.0.
        adjust_numeric(&mut s, 1);
        assert!((s.level.pipes[0].density - 1.0).abs() < 1e-6);
        // Step down toward 0.
        for _ in 0..20 {
            adjust_numeric(&mut s, -1);
        }
        assert!((s.level.pipes[0].density - 0.0).abs() < 1e-6);
    }

    #[test]
    fn edit_color_steps_palette() {
        let mut s = two_node_state();
        s.level.push_pipe_between(0, 1).unwrap();
        s.selected_pipe = Some(0);
        s.entity_field = EntityField::PipeColor;
        let before = s.level.pipes[0].color_rgb & 0xFF_FFFF;
        adjust_numeric(&mut s, 1);
        let after = s.level.pipes[0].color_rgb & 0xFF_FFFF;
        // It landed on a palette swatch (and generally moved off `before`).
        assert!(PALETTE.iter().any(|&p| (p & 0xFF_FFFF) == after));
        let _ = before;
    }

    #[test]
    fn delete_removes_pipe_and_fixes_selection() {
        let mut s = two_node_state();
        s.level.push_node(8, 8, 0); // node 2
        s.level.push_pipe_between(0, 1).unwrap(); // pipe 0
        s.level.push_pipe_between(0, 2).unwrap(); // pipe 1
        s.selected_pipe = Some(1);
        s.place = Some(EntityKind::Pipe);
        // Cursor on a path cell of pipe 1, then the shared entity deleter.
        s.cursor = (0, 1);
        s.z = 0;
        crate::entity::delete_at_cursor(&mut s);
        assert_eq!(s.level.pipes.len(), 1);
        assert!(s.selected_pipe.is_none());
    }

    #[test]
    fn delete_with_nothing_is_noop() {
        let mut s = two_node_state();
        s.place = Some(EntityKind::Pipe);
        s.cursor = (10, 10);
        crate::entity::delete_at_cursor(&mut s);
        assert!(s.status.contains("no pipe"));
    }
}
