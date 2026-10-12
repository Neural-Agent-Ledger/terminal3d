//! Geometry FILL tools on the active z-slice: RECT, LINE, BUCKET.
//!
//! All three paint the CURRENT geometry brush — the plain [`CellKind`] brush, or,
//! when `shape_mode` is on, a solid base cell carrying the selected 45° [`Shape`]
//! (matching `edit::apply_tool_at_cursor`). Each fill is ONE undoable edit: the
//! caller takes a single [`EditorState::push_undo`] snapshot before invoking it,
//! never per-cell.
//!
//! Anchor model: `k` marks corner A at the cursor (`state.fill_anchor`). Then:
//! * `K` → RECT-fills the box A→cursor,
//! * `L` → draws a cell LINE A→cursor (2D Bresenham),
//!
//! and clears the anchor. BUCKET (`J`) needs no anchor — it flood-fills the
//! contiguous same-kind region under the cursor (4-connected, slice-bounded).
//!
//! Pure: no I/O, no terminal. Operates only on `state.map` at `state.z`.
use crate::state::EditorState;
use re_core::Shape;
use re_world::CellKind;

/// Paint one cell on the active slice with the current geometry brush (plain or
/// 45° shape), mirroring the non-entity branch of `edit::apply_tool_at_cursor`.
fn paint_cell(state: &mut EditorState, x: usize, y: usize, z: usize) {
    if state.shape_mode {
        state.map.set(x, y, z, CellKind::WallSolid);
        state.map.set_shape(x, y, z, state.shape_sel);
    } else {
        state.map.set(x, y, z, state.brush);
        state.map.set_shape(x, y, z, Shape::Full);
    }
}

/// RECT-fill the inclusive box `a`→`b` on the active z-slice with the current
/// brush. Corners are normalised and clamped to the map; out-of-range corners are
/// pulled in rather than panicking. Returns the number of cells painted.
pub fn rect(state: &mut EditorState, a: (usize, usize), b: (usize, usize)) -> usize {
    let (sx, sy, _sz) = state.map.size();
    if sx == 0 || sy == 0 {
        return 0;
    }
    let z = state.z;
    let x0 = a.0.min(b.0).min(sx - 1);
    let x1 = a.0.max(b.0).min(sx - 1);
    let y0 = a.1.min(b.1).min(sy - 1);
    let y1 = a.1.max(b.1).min(sy - 1);
    let mut n = 0;
    for y in y0..=y1 {
        for x in x0..=x1 {
            paint_cell(state, x, y, z);
            n += 1;
        }
    }
    n
}

/// Draw a straight cell LINE from `a` to `b` on the active z-slice with the
/// current brush (integer 2D Bresenham). Endpoints are clamped to the map.
/// Returns the number of cells painted.
pub fn line(state: &mut EditorState, a: (usize, usize), b: (usize, usize)) -> usize {
    let (sx, sy, _sz) = state.map.size();
    if sx == 0 || sy == 0 {
        return 0;
    }
    let z = state.z;
    let clamp = |p: (usize, usize)| (p.0.min(sx - 1), p.1.min(sy - 1));
    let (ax, ay) = clamp(a);
    let (bx, by) = clamp(b);

    // Bresenham over i32; coordinates are in-bounds so casts are safe.
    let (mut x, mut y) = (ax as i32, ay as i32);
    let (x1, y1) = (bx as i32, by as i32);
    let dx = (x1 - x).abs();
    let dy = -(y1 - y).abs();
    let sx_step = if x < x1 { 1 } else { -1 };
    let sy_step = if y < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let mut n = 0;
    loop {
        paint_cell(state, x as usize, y as usize, z);
        n += 1;
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx_step;
        }
        if e2 <= dx {
            err += dx;
            y += sy_step;
        }
    }
    n
}

/// BUCKET flood-fill the contiguous region of cells sharing the SAME [`CellKind`]
/// as the cell under `start`, on the active z-slice, with the current brush
/// (4-connected, bounded to the slice). If painting wouldn't change the seed cell
/// (same kind, and either plain brush matching kind or shape mode that would just
/// re-stamp), the region is still filled so a shape/brush swap propagates. Returns
/// the number of cells painted. Iterative stack — no recursion, no overflow.
pub fn bucket(state: &mut EditorState, start: (usize, usize)) -> usize {
    let (sx, sy, _sz) = state.map.size();
    if start.0 >= sx || start.1 >= sy {
        return 0;
    }
    let zu = state.z;
    let z = zu as i32;
    let target = state.map.cell_at(start.0 as i32, start.1 as i32, z);

    let mut visited = vec![false; sx * sy];
    let idx = |x: usize, y: usize| y * sx + x;
    let mut stack = vec![start];
    visited[idx(start.0, start.1)] = true;
    let mut n = 0;
    while let Some((x, y)) = stack.pop() {
        // Only spread across cells of the target kind.
        if state.map.cell_at(x as i32, y as i32, z) != target {
            continue;
        }
        paint_cell(state, x, y, zu);
        n += 1;
        // 4-connected neighbours within the slice.
        let push =
            |nx: usize, ny: usize, stack: &mut Vec<(usize, usize)>, visited: &mut Vec<bool>| {
                if !visited[idx(nx, ny)] && state.map.cell_at(nx as i32, ny as i32, z) == target {
                    visited[idx(nx, ny)] = true;
                    stack.push((nx, ny));
                }
            };
        if x > 0 {
            push(x - 1, y, &mut stack, &mut visited);
        }
        if x + 1 < sx {
            push(x + 1, y, &mut stack, &mut visited);
        }
        if y > 0 {
            push(x, y - 1, &mut stack, &mut visited);
        }
        if y + 1 < sy {
            push(x, y + 1, &mut stack, &mut visited);
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rect_fills_inclusive_box() {
        let mut s = EditorState::new_empty(8, 8, 2);
        s.brush = CellKind::WallSolid;
        let painted = rect(&mut s, (1, 1), (3, 2));
        // (3-1+1) * (2-1+1) = 3 * 2 = 6 cells.
        assert_eq!(painted, 6);
        for y in 1..=2 {
            for x in 1..=3 {
                assert_eq!(s.map.cell_at(x, y, 0), CellKind::WallSolid);
            }
        }
        // A corner outside the box is untouched.
        assert_eq!(s.map.cell_at(0, 0, 0), CellKind::Empty);
        assert_eq!(s.map.cell_at(4, 2, 0), CellKind::Empty);
    }

    #[test]
    fn rect_normalises_and_clamps_corners() {
        let mut s = EditorState::new_empty(4, 4, 1);
        s.brush = CellKind::Floor;
        // Reversed + out-of-range corners: should clamp to the 4x4 map.
        let painted = rect(&mut s, (99, 99), (2, 2));
        // Box (2,2)..(3,3) = 2x2 = 4 cells.
        assert_eq!(painted, 4);
        assert_eq!(s.map.cell_at(3, 3, 0), CellKind::Floor);
        assert_eq!(s.map.cell_at(2, 2, 0), CellKind::Floor);
    }

    #[test]
    fn line_draws_diagonal() {
        let mut s = EditorState::new_empty(8, 8, 1);
        s.brush = CellKind::WallSolid;
        let painted = line(&mut s, (0, 0), (3, 3));
        assert_eq!(painted, 4, "diagonal of length 4");
        for i in 0..4 {
            assert_eq!(s.map.cell_at(i, i, 0), CellKind::WallSolid);
        }
    }

    #[test]
    fn line_draws_horizontal() {
        let mut s = EditorState::new_empty(8, 8, 1);
        s.brush = CellKind::WallTrace;
        let painted = line(&mut s, (1, 4), (5, 4));
        assert_eq!(painted, 5);
        for x in 1..=5 {
            assert_eq!(s.map.cell_at(x, 4, 0), CellKind::WallTrace);
        }
        assert_eq!(s.map.cell_at(0, 4, 0), CellKind::Empty);
    }

    #[test]
    fn bucket_fills_contiguous_region_only() {
        let mut s = EditorState::new_empty(6, 6, 1);
        // Carve a wall barrier down column 3 so flood can't cross it.
        for y in 0..6 {
            s.map.set(3, y, 0, CellKind::WallSolid);
        }
        // Fill the empty region on the LEFT of the barrier with Floor.
        s.brush = CellKind::Floor;
        let painted = bucket(&mut s, (0, 0));
        // Left region is columns 0..=2 over 6 rows = 18 cells.
        assert_eq!(painted, 18);
        assert_eq!(s.map.cell_at(0, 0, 0), CellKind::Floor);
        assert_eq!(s.map.cell_at(2, 5, 0), CellKind::Floor);
        // Right of the barrier is untouched (still Empty).
        assert_eq!(s.map.cell_at(4, 0, 0), CellKind::Empty);
        // The barrier itself is untouched.
        assert_eq!(s.map.cell_at(3, 2, 0), CellKind::WallSolid);
    }

    #[test]
    fn bucket_paints_shape_when_shape_mode() {
        let mut s = EditorState::new_empty(4, 4, 1);
        s.shape_mode = true;
        s.shape_sel = crate::state::SHAPE_CYCLE[0];
        let painted = bucket(&mut s, (0, 0));
        assert_eq!(painted, 16, "whole empty slice is one region");
        assert_eq!(s.map.cell_at(0, 0, 0), CellKind::WallSolid);
        assert_eq!(s.map.shape_at(0, 0, 0), crate::state::SHAPE_CYCLE[0]);
    }
}
