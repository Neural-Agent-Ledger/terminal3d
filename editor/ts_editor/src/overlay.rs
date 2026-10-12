//! SCALABLE MATERIAL OVERLAY line-polyline authoring (the waypoint model).
//!
//! Pure functions over [`EditorState`] — no I/O, no terminal. The overlay zones /
//! lines / insets live in `level.overlay.{zones,lines,insets}`, which
//! `re_voximport::to_lvl_text` already serializes (`[zone]`/`[line]`/`[inset]`), so
//! authoring them here and saving via `io::save_lvl_text` round-trips them with no
//! extra format work.
//!
//! This module owns the LINE's polyline model. A [`re_level::OverlayLine`] stores a
//! [`re_core::LinePath`] (`points: Vec<Vec3>`); the editor appends a waypoint at the
//! cursor cell (cell CENTRE, so the world-space polyline rides through the middle of
//! each authored cell) on `space`, and removes the last on `-` while the `points`
//! field is focused — REUSING the pipe-path waypoint approach
//! (`re_world`/`re_level` differ only in storage: a pipe path is `Vec<(i32,i32,i32)>`
//! cells, a line path is `Vec<Vec3>` world points). The zone/inset region authoring
//! lives in `entity.rs` next to the other region helpers; only the line's distinct
//! polyline storage is split out here.
use crate::state::{EditorState, EntityKind};
use re_core::Vec3;

/// The world-space point for cell `(x, y, z)` a line waypoint is stamped at: the
/// cell CENTRE. Keeps the authored polyline visibly aligned with the grid cells it
/// passes through (the marker hit-test floors each point back to its cell).
fn waypoint_for_cell(x: i32, y: i32, z: i32) -> Vec3 {
    Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5)
}

/// `space` on the LINE kind: if the cursor lies on an EXISTING line's path, SELECT
/// that line for editing; otherwise APPEND a waypoint at the cursor to the SELECTED
/// line (creating a new line when none is selected). Mirrors the pipe path-cell
/// select / waypoint-append flow.
///
/// THE single source of truth for line waypoint APPEND + select-by-path. Called from
/// `entity::place_at_cursor`'s `Line` arm. Marks dirty + sets a status line.
pub fn line_place_or_extend(state: &mut EditorState, x: i32, y: i32, z: i32) {
    // Re-selecting an existing line's path cell keeps editing that line (and does NOT
    // append) so a stray click on a drawn line doesn't grow it.
    if let Some(idx) = crate::entity::line_at(state, x, y, z) {
        // If the cursor is on the CURRENTLY-SELECTED line's path, treat the click as
        // a no-op select-confirm; if it's a DIFFERENT line, switch selection to it.
        if state.selected_line != Some(idx) {
            state.selected_line = Some(idx);
            state.status = format!("selected line #{idx}");
            return;
        }
        state.status = format!("line #{idx}: cursor already on this path (- to remove last)");
        return;
    }
    // Append to the selected line, or create a fresh one anchored at the cursor.
    let idx = match state
        .selected_line
        .filter(|&i| i < state.level.overlay.lines.len())
    {
        Some(i) => i,
        None => {
            let name = unique_line_name(state);
            state.level.overlay.lines.push(re_level::OverlayLine {
                name,
                ..Default::default()
            });
            let i = state.level.overlay.lines.len() - 1;
            state.selected_line = Some(i);
            state.place = Some(EntityKind::Line);
            i
        }
    };
    state.level.overlay.lines[idx]
        .path
        .points
        .push(waypoint_for_cell(x, y, z));
    let n = state.level.overlay.lines[idx].path.points.len();
    state.mark_dirty();
    state.status = format!("line #{idx} waypoint {n} @ {x},{y},{z}");
}

/// `+`/`-` on the LINE `points` field: `-` REMOVES the last waypoint (the editing
/// affordance for shortening a polyline; new points are APPENDED with `space`); `+`
/// is a hint to append at the cursor with `space`. A defined no-op when the line has
/// no waypoints.
pub fn line_adjust_points(state: &mut EditorState, idx: usize, delta: i32) {
    let Some(line) = state.level.overlay.lines.get_mut(idx) else {
        state.status = "no line selected".into();
        return;
    };
    if delta < 0 {
        if line.path.points.pop().is_some() {
            let n = line.path.points.len();
            state.mark_dirty();
            state.status = format!("line #{idx}: removed last waypoint ({n} left)");
        } else {
            state.status = format!("line #{idx}: no waypoints to remove");
        }
    } else {
        state.status = format!("line #{idx}: move the cursor + press space to append a waypoint");
    }
}

/// A unique auto name (`line1`, `line2`, …) for a freshly-placed overlay line,
/// skipping any name already taken so a `[line]` never collides on its key. Kept here
/// (next to the append flow) since line creation lives in this module.
fn unique_line_name(state: &EditorState) -> String {
    let mut k = state.level.overlay.lines.len() + 1;
    loop {
        let name = format!("line{k}");
        if !state.level.overlay.lines.iter().any(|l| l.name == name) {
            return name;
        }
        k += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn space_appends_waypoints_to_a_new_line() {
        let mut s = EditorState::new_empty(16, 16, 4);
        s.place = Some(EntityKind::Line);
        // First space creates a line + one waypoint.
        line_place_or_extend(&mut s, 1, 2, 0);
        assert_eq!(s.level.overlay.lines.len(), 1);
        assert_eq!(s.selected_line, Some(0));
        assert_eq!(s.level.overlay.lines[0].path.points.len(), 1);
        // Two more waypoints extend the SAME line.
        line_place_or_extend(&mut s, 4, 2, 0);
        line_place_or_extend(&mut s, 4, 5, 0);
        assert_eq!(s.level.overlay.lines.len(), 1, "no duplicate line");
        assert_eq!(s.level.overlay.lines[0].path.points.len(), 3);
        // The waypoints ride through cell centres.
        let p0 = s.level.overlay.lines[0].path.points[0];
        assert!((p0.x - 1.5).abs() < 1e-6 && (p0.y - 2.5).abs() < 1e-6);
    }

    #[test]
    fn space_on_existing_path_selects_not_appends() {
        let mut s = EditorState::new_empty(16, 16, 4);
        s.place = Some(EntityKind::Line);
        line_place_or_extend(&mut s, 1, 1, 0);
        line_place_or_extend(&mut s, 3, 1, 0);
        // Drop the selection, then click a path cell — it RE-SELECTS rather than
        // appending a duplicate.
        s.selected_line = None;
        line_place_or_extend(&mut s, 1, 1, 0);
        assert_eq!(s.selected_line, Some(0));
        assert_eq!(
            s.level.overlay.lines[0].path.points.len(),
            2,
            "no append on re-select"
        );
    }

    #[test]
    fn minus_removes_last_waypoint() {
        let mut s = EditorState::new_empty(16, 16, 4);
        s.place = Some(EntityKind::Line);
        line_place_or_extend(&mut s, 1, 1, 0);
        line_place_or_extend(&mut s, 2, 1, 0);
        line_adjust_points(&mut s, 0, -1);
        assert_eq!(s.level.overlay.lines[0].path.points.len(), 1);
        line_adjust_points(&mut s, 0, -1);
        assert_eq!(s.level.overlay.lines[0].path.points.len(), 0);
        // A further removal is a defined no-op.
        line_adjust_points(&mut s, 0, -1);
        assert!(s.status.contains("no waypoints"));
    }
}
