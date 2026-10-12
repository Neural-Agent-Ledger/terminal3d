//! Pure top-down viewport math: integer zoom, X/Y pan, ruler tick placement,
//! z-slider fill, and elevation cross-section extraction. No Bevy, no
//! Framebuffer — every function is input→output so the readability features
//! (zoom clamp, pan auto-follow, ruler ticks, slider ratio) are unit-tested
//! without a terminal.
use re_world::{CellKind, Map};

/// Min/max integer zoom (block size in framebuffer cells per world cell). The
/// max was raised from 3 to 6 so the author can zoom in far enough that each
/// world cell becomes a large block — that extra room is used by the painter to
/// draw sub-cell detail (see `chrome::draw_grid`), so "zoom = render detail".
pub const ZOOM_MIN: u8 = 1;
pub const ZOOM_MAX: u8 = 6;
/// Chunk-line spacing in world cells (faint orientation guides + ruler ticks).
pub const CHUNK: usize = 8;

/// Clamp a desired zoom into the legal `ZOOM_MIN..=ZOOM_MAX` range.
pub fn clamp_zoom(z: i32) -> u8 {
    z.clamp(ZOOM_MIN as i32, ZOOM_MAX as i32) as u8
}

/// Ruler tick SPACING in world cells, which tightens (gets denser) as zoom
/// rises: at zoom 1 ticks sit every [`CHUNK`] cells, at 2 every `CHUNK/2`, and
/// from zoom 4 up every single cell. Never zero. Pure; unit-tested. Used by
/// [`ruler_ticks`] so high zoom shows a finer coordinate scale.
pub fn ruler_tick_step(zoom: u8) -> usize {
    match zoom {
        0 | 1 => CHUNK,     // 8
        2 | 3 => CHUNK / 2, // 4
        _ => 1,             // every cell from 4x up
    }
}

/// Inverse of [`z_slider_knob`]: map a clicked track index `click_idx`
/// (0..inner_cells-1) back to a world z-layer in `0..sz`. Rounds to the nearest
/// layer so a click anywhere along a layer's segment selects it. A single-layer
/// world always returns 0. Pure; unit-tested as the painter's inverse.
pub fn z_from_slider(click_idx: usize, inner_cells: usize, sz: usize) -> usize {
    if sz <= 1 || inner_cells <= 1 {
        return 0;
    }
    let cells = inner_cells - 1;
    let idx = click_idx.min(cells);
    // ratio in [0,1] → nearest layer.
    let ratio = idx as f32 / cells as f32;
    (ratio * (sz - 1) as f32).round() as usize
}

/// One step zoom in (toward 3×), clamped.
pub fn zoom_in(z: u8) -> u8 {
    clamp_zoom(z as i32 + 1)
}

/// One step zoom out (toward 1×), clamped.
pub fn zoom_out(z: u8) -> u8 {
    clamp_zoom(z as i32 - 1)
}

/// Clamp a candidate pan origin to the legal range on one axis: `0` at the near
/// edge, `map.saturating_sub(visible)` at the far edge so the window never scrolls
/// past the world (and pins to 0 when the whole world fits). `visible` is the
/// whole-world-cell window width for the *new* zoom (see [`visible_cells`]).
fn clamp_pan_axis(pan: usize, visible: usize, map: usize) -> usize {
    pan.min(map.saturating_sub(visible))
}

/// Recompute the pan origin so the cell currently under `anchor_cell` keeps
/// (approximately) the same on-screen position after the zoom changes from
/// `old_zoom` to `new_zoom`.
///
/// The painter places world cell `(pan + v)` at framebuffer block `v*zoom`, so a
/// cell's on-screen offset in WORLD cells is `anchor - pan`. We preserve that
/// offset as a fraction of the visible window: the anchor sat `frac` of the way
/// across the old window, so after the zoom we place it `frac` of the way across
/// the new (differently sized) window. The result is clamped to the legal pan
/// range for the new zoom so it is never negative and never past the world edge.
///
/// `view_cells` is the cell-area pane size in framebuffer cells `(w, h)` (the grid
/// pane minus the ruler gutters — the same `cells_w/cells_h` `screen_to_cell`
/// computes). `map_size` is the world `(sx, sy)`.
fn pan_anchored(
    old_zoom: u8,
    new_zoom: u8,
    pan: (usize, usize),
    anchor_cell: (usize, usize),
    view_cells: (usize, usize),
    map_size: (usize, usize),
) -> (usize, usize) {
    let axis = |pane: usize, old_pan: usize, anchor: usize, map: usize| -> usize {
        let vis_old = visible_cells(pane, old_zoom);
        let vis_new = visible_cells(pane, new_zoom);
        // On-screen offset of the anchor in the OLD window, as a fraction.
        let off_old = anchor
            .saturating_sub(old_pan)
            .min(vis_old.saturating_sub(1));
        let frac = if vis_old > 1 {
            off_old as f32 / (vis_old - 1) as f32
        } else {
            0.0
        };
        // Place the anchor at the same fraction of the NEW (smaller/larger) window.
        let off_new = (frac * (vis_new.saturating_sub(1)) as f32).round() as usize;
        let candidate = anchor.saturating_sub(off_new);
        clamp_pan_axis(candidate, vis_new, map)
    };
    (
        axis(view_cells.0, pan.0, anchor_cell.0, map_size.0),
        axis(view_cells.1, pan.1, anchor_cell.1, map_size.1),
    )
}

/// Zoom IN one step (toward [`ZOOM_MAX`]) while keeping the cell under
/// `anchor_cell` at roughly its current on-screen position. Returns the new
/// `(zoom, pan)`; the pan is always clamped to `[0, map_size - visible_cells]`
/// per axis so it is never negative and never scrolls past the world. The
/// non-anchored [`zoom_in`] is kept for callers that don't pass an anchor.
///
/// `view_cells` = cell-area pane size in framebuffer cells `(w, h)` (grid pane
/// minus the ruler gutters). `map_size` = world `(sx, sy)`.
pub fn zoom_in_at(
    zoom: u8,
    pan: (usize, usize),
    anchor_cell: (usize, usize),
    view_cells: (usize, usize),
    map_size: (usize, usize),
) -> (u8, (usize, usize)) {
    let new_zoom = zoom_in(zoom);
    let new_pan = pan_anchored(zoom, new_zoom, pan, anchor_cell, view_cells, map_size);
    (new_zoom, new_pan)
}

/// Zoom OUT one step (toward [`ZOOM_MIN`]) while keeping the cell under
/// `anchor_cell` at roughly its current on-screen position. Same contract and
/// clamping as [`zoom_in_at`].
pub fn zoom_out_at(
    zoom: u8,
    pan: (usize, usize),
    anchor_cell: (usize, usize),
    view_cells: (usize, usize),
    map_size: (usize, usize),
) -> (u8, (usize, usize)) {
    let new_zoom = zoom_out(zoom);
    let new_pan = pan_anchored(zoom, new_zoom, pan, anchor_cell, view_cells, map_size);
    (new_zoom, new_pan)
}

/// How many WHOLE world cells fit across a pane of `pane` framebuffer cells at
/// `zoom`. At least 1 so the viewport is never degenerate.
pub fn visible_cells(pane: usize, zoom: u8) -> usize {
    (pane / zoom.max(1) as usize).max(1)
}

/// Auto-follow pan: given the current pan origin `pan` (world cell at the
/// pane's top-left edge along one axis), the `cursor` world coordinate on that
/// axis, the pane size in framebuffer cells, the `zoom`, and the world `extent`
/// (size) on that axis, return the new pan origin that keeps the cursor inside
/// the visible window with the smallest scroll.
///
/// Pure 1-D math; called once per axis. Scrolls left when the cursor falls
/// below the window, right when it passes the last visible column, and never
/// scrolls past the world so trailing empty space is not shown unnecessarily.
pub fn follow_pan(pan: usize, cursor: usize, pane: usize, zoom: u8, extent: usize) -> usize {
    let vis = visible_cells(pane, zoom);
    // The furthest-right pan that still shows real world cells.
    let max_pan = extent.saturating_sub(vis);
    let mut new = pan.min(max_pan);
    if cursor < new {
        // Cursor scrolled off the near edge: align window to the cursor.
        new = cursor;
    } else if cursor >= new + vis {
        // Cursor scrolled off the far edge: bring it to the last column.
        new = cursor + 1 - vis;
    }
    new.min(max_pan)
}

/// Fill ratio of the Z slider in `[0,1]`: how high the active layer `z` sits in
/// the world's `sz` layers. `z=0` → 0.0 (floor), `z=sz-1` → 1.0 (top). A
/// single-layer world reports 0.0 (no travel).
pub fn z_slider_ratio(z: usize, sz: usize) -> f32 {
    if sz <= 1 {
        return 0.0;
    }
    (z.min(sz - 1) as f32) / ((sz - 1) as f32)
}

/// Knob index (0..=cells-1) of the horizontal Z slider's filled cell: `z=0`
/// sits at index 0 (floor end), `z=sz-1` at the last cell (top end).
pub fn z_slider_knob(z: usize, sz: usize, cells: usize) -> usize {
    if cells == 0 {
        return 0;
    }
    let ratio = z_slider_ratio(z, sz);
    (ratio * (cells.saturating_sub(1)) as f32).round() as usize
}

/// World coordinate for a given ruler tick: the first chunk-aligned world cell
/// at or after `pan`, then every `CHUNK` cells, while still inside the window.
/// Returns `(world_coord, offset_in_cells_from_pan)` pairs. Tracks pan + zoom
/// because the offset is in WORLD cells (the caller multiplies by zoom to place
/// the label in framebuffer columns).
pub fn ruler_ticks(pan: usize, pane: usize, zoom: u8, extent: usize) -> Vec<(usize, usize)> {
    let vis = visible_cells(pane, zoom);
    let last = pan.saturating_add(vis).min(extent); // exclusive
                                                    // Tick spacing tightens with zoom so the scale reads finer when zoomed in.
    let step = ruler_tick_step(zoom).max(1);
    // First step-aligned world cell >= pan.
    let first = pan.div_ceil(step) * step;
    let mut out = Vec::new();
    let mut w = first;
    while w < last {
        out.push((w, w - pan));
        w += step;
    }
    out
}

/// Whether a WORLD coordinate sits on a chunk boundary (faint guide line).
pub fn is_chunk_line(world: usize) -> bool {
    world.is_multiple_of(CHUNK)
}

/// Invert the top-down grid projection: map an ABSOLUTE framebuffer cell
/// `(col, row)` back to the world cell `(wx, wy)` it paints, or `None` when the
/// point is outside the drawable cell area (in a ruler gutter, past the visible
/// window, or outside the world extent).
///
/// This is the exact inverse of `chrome::draw_grid`'s forward mapping, which
/// places world cell `(pan.0 + vx, pan.1 + vy)` into the framebuffer block
/// `[cells_x0 + vx*zoom, cells_x0 + (vx+1)*zoom)` (and likewise on y), where
/// `cells_x0 = grid_x0 + ruler_w` and `cells_y0 = grid_y0 + ruler_h`. The caller
/// passes the live grid rect (`grid_x0, grid_y0, grid_w, grid_h`) and the ruler
/// gutter sizes so this stays free of chrome constants and round-trips against
/// the painter for any `zoom`/`pan`.
///
/// * `col`, `row` — absolute framebuffer cell of the mouse event.
/// * `grid_x0`, `grid_y0`, `grid_w`, `grid_h` — the grid pane rect.
/// * `ruler_w`, `ruler_h` — gutter columns/rows reserved on the left/top.
/// * `pan` — world cell at the top-left of the cell area.
/// * `zoom` — integer block size.
/// * `extent` — world `(sx, sy)` size, to reject cells past the map edge.
#[allow(clippy::too_many_arguments)]
pub fn screen_to_cell(
    col: usize,
    row: usize,
    grid_x0: usize,
    grid_y0: usize,
    grid_w: usize,
    grid_h: usize,
    ruler_w: usize,
    ruler_h: usize,
    pan: (usize, usize),
    zoom: u8,
    extent: (usize, usize),
) -> Option<(usize, usize)> {
    let zoom = zoom.max(1) as usize;
    let cells_x0 = grid_x0 + ruler_w;
    let cells_y0 = grid_y0 + ruler_h;
    let cells_w = grid_w.saturating_sub(ruler_w);
    let cells_h = grid_h.saturating_sub(ruler_h);

    // In a ruler gutter or left/above the grid: no cell.
    if col < cells_x0 || row < cells_y0 {
        return None;
    }
    // Past the right/bottom edge of the grid pane: no cell.
    if col >= grid_x0 + grid_w || row >= grid_y0 + grid_h {
        return None;
    }

    let (pan_x, pan_y) = pan;
    let (sx, sy) = extent;
    // How many whole world cells the painter draws (pane window ∩ remaining world).
    let vis_x = visible_cells(cells_w, zoom as u8).min(sx.saturating_sub(pan_x));
    let vis_y = visible_cells(cells_h, zoom as u8).min(sy.saturating_sub(pan_y));

    let vx = (col - cells_x0) / zoom;
    let vy = (row - cells_y0) / zoom;
    // Beyond the painted window (e.g. the partial trailing block past `vis`).
    if vx >= vis_x || vy >= vis_y {
        return None;
    }

    let wx = pan_x + vx;
    let wy = pan_y + vy;
    // Defensive: the visible-cell clamp already keeps us in-world, but guard so
    // an out-of-extent cursor is never returned.
    if wx >= sx || wy >= sy {
        return None;
    }
    Some((wx, wy))
}

/// A single cell sample for the elevation cross-section.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ElevSample {
    pub z: usize,
    pub kind: CellKind,
    pub side_data: u16,
    /// The cell's 45° geometry shape (Full for ordinary cubes). The elevation
    /// window uses this so a `RampZ` reads as a SLOPE rather than a flat block.
    pub shape: re_core::Shape,
}

/// Extract the vertical Z-stack at the cursor `(cx, cy)`: one sample per layer
/// from `z=0` (bottom) up to `sz-1` (top). The caller renders it top-down (high
/// z first) so the column reads like an elevation profile. Out-of-bounds cursor
/// yields all-`Empty` (matches `Map::cell_at`).
pub fn elevation_column(map: &Map, cx: usize, cy: usize) -> Vec<ElevSample> {
    let (_sx, _sy, sz) = map.size();
    (0..sz)
        .map(|z| ElevSample {
            z,
            kind: map.cell_at(cx as i32, cy as i32, z as i32),
            side_data: map.side_data_at(cx as i32, cy as i32, z as i32),
            shape: map.shape_at(cx as i32, cy as i32, z as i32),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zoom_clamps_to_min_max() {
        assert_eq!(ZOOM_MIN, 1);
        assert_eq!(ZOOM_MAX, 6);
        assert_eq!(clamp_zoom(0), 1);
        assert_eq!(clamp_zoom(1), 1);
        assert_eq!(clamp_zoom(3), 3);
        assert_eq!(clamp_zoom(6), 6);
        assert_eq!(clamp_zoom(9), 6); // clamp at the new max
        assert_eq!(zoom_in(1), 2);
        assert_eq!(zoom_in(6), 6); // clamp at max
        assert_eq!(zoom_out(1), 1); // clamp at min
        assert_eq!(zoom_out(6), 5);
    }

    /// On-screen WORLD-cell offset of `anchor` in the window starting at `pan`
    /// (the painter's forward offset). `None` when the anchor is not inside the
    /// visible window for that zoom.
    fn on_screen_offset(pan: usize, anchor: usize, pane: usize, zoom: u8) -> Option<usize> {
        if anchor < pan {
            return None;
        }
        let off = anchor - pan;
        if off < visible_cells(pane, zoom) {
            Some(off)
        } else {
            None
        }
    }

    #[test]
    fn zoom_in_at_keeps_anchor_visible_across_range() {
        // 60x40 cell pane, 64x64 world. Walk the whole zoom range from the min,
        // anchoring on a cell that starts near a grid edge (where naive zoom
        // would throw it off-screen).
        let view_cells = (60usize, 40usize);
        let map_size = (64usize, 64usize);
        let anchor = (2usize, 1usize); // near the near edge
        let mut zoom = ZOOM_MIN;
        let mut pan = (0usize, 0usize);
        while zoom < ZOOM_MAX {
            let (nz, np) = zoom_in_at(zoom, pan, anchor, view_cells, map_size);
            assert_eq!(nz, zoom_in(zoom), "zoom advanced one step");
            // Pan clamped: never negative (usize) and never past the world.
            assert!(np.0 <= map_size.0 - visible_cells(view_cells.0, nz));
            assert!(np.1 <= map_size.1 - visible_cells(view_cells.1, nz));
            // Anchor remains inside the visible window after zooming.
            assert!(
                on_screen_offset(np.0, anchor.0, view_cells.0, nz).is_some(),
                "anchor x visible after zoom-in to {nz}: pan={np:?}"
            );
            assert!(
                on_screen_offset(np.1, anchor.1, view_cells.1, nz).is_some(),
                "anchor y visible after zoom-in to {nz}: pan={np:?}"
            );
            zoom = nz;
            pan = np;
        }
    }

    #[test]
    fn zoom_at_anchor_in_middle_stays_near_prior_screen_position() {
        // A centred anchor should keep roughly the same screen offset.
        let view_cells = (40usize, 40usize);
        let map_size = (64usize, 64usize);
        let anchor = (32usize, 32usize);
        let zoom = 2u8;
        // Start centred on the anchor.
        let pan = (
            anchor.0 - visible_cells(view_cells.0, zoom) / 2,
            anchor.1 - visible_cells(view_cells.1, zoom) / 2,
        );
        let off_before = (anchor.0 - pan.0, anchor.1 - pan.1);
        let frac_before = (
            off_before.0 as f32 / visible_cells(view_cells.0, zoom) as f32,
            off_before.1 as f32 / visible_cells(view_cells.1, zoom) as f32,
        );
        let (nz, np) = zoom_in_at(zoom, pan, anchor, view_cells, map_size);
        let off_after = (anchor.0 - np.0, anchor.1 - np.1);
        let frac_after = (
            off_after.0 as f32 / visible_cells(view_cells.0, nz) as f32,
            off_after.1 as f32 / visible_cells(view_cells.1, nz) as f32,
        );
        // Same fractional screen position within ~10% (rounding + smaller window).
        assert!((frac_before.0 - frac_after.0).abs() < 0.1, "x frac drift");
        assert!((frac_before.1 - frac_after.1).abs() < 0.1, "y frac drift");
    }

    #[test]
    fn zoom_out_at_clamps_pan_and_keeps_anchor() {
        // Start zoomed in near the far edge, then zoom out repeatedly.
        let view_cells = (60usize, 40usize);
        let map_size = (64usize, 64usize);
        let anchor = (62usize, 62usize); // near the far edge
        let mut zoom = ZOOM_MAX;
        // Pan that places the anchor at the far edge of the current window.
        let mut pan = (
            anchor
                .0
                .saturating_sub(visible_cells(view_cells.0, zoom) - 1),
            anchor
                .1
                .saturating_sub(visible_cells(view_cells.1, zoom) - 1),
        );
        while zoom > ZOOM_MIN {
            let (nz, np) = zoom_out_at(zoom, pan, anchor, view_cells, map_size);
            assert_eq!(nz, zoom_out(zoom));
            // Pan never past the world (and usize ⇒ never negative).
            assert!(np.0 <= map_size.0 - visible_cells(view_cells.0, nz));
            assert!(np.1 <= map_size.1 - visible_cells(view_cells.1, nz));
            // Anchor still visible.
            assert!(on_screen_offset(np.0, anchor.0, view_cells.0, nz).is_some());
            assert!(on_screen_offset(np.1, anchor.1, view_cells.1, nz).is_some());
            zoom = nz;
            pan = np;
        }
    }

    #[test]
    fn zoom_at_window_larger_than_world_pins_pan_to_zero() {
        // World smaller than the visible window on both axes: pan must pin to 0.
        let view_cells = (60usize, 40usize);
        let map_size = (4usize, 3usize);
        let anchor = (2usize, 1usize);
        let (_nz, np) = zoom_in_at(1, (0, 0), anchor, view_cells, map_size);
        assert_eq!(np, (0, 0), "tiny world pins pan to origin");
        let (_nz, np) = zoom_out_at(ZOOM_MAX, (0, 0), anchor, view_cells, map_size);
        assert_eq!(np, (0, 0));
    }

    #[test]
    fn zoom_at_clamps_at_zoom_extremes() {
        // At ZOOM_MAX, zoom_in_at must not advance zoom; at ZOOM_MIN, zoom_out_at
        // must not retreat — pan stays in range either way.
        let view_cells = (60usize, 40usize);
        let map_size = (64usize, 64usize);
        let anchor = (30usize, 30usize);
        let (nz, np) = zoom_in_at(ZOOM_MAX, (10, 10), anchor, view_cells, map_size);
        assert_eq!(nz, ZOOM_MAX);
        assert!(np.0 <= map_size.0 - visible_cells(view_cells.0, nz));
        let (nz, _np) = zoom_out_at(ZOOM_MIN, (0, 0), anchor, view_cells, map_size);
        assert_eq!(nz, ZOOM_MIN);
    }

    #[test]
    fn ruler_tick_step_tightens_with_zoom() {
        // Denser ticks as zoom rises; never zero.
        assert_eq!(ruler_tick_step(1), CHUNK);
        assert_eq!(ruler_tick_step(2), CHUNK / 2);
        assert_eq!(ruler_tick_step(3), CHUNK / 2);
        assert_eq!(ruler_tick_step(4), 1);
        assert_eq!(ruler_tick_step(6), 1);
        // Strictly non-increasing across the legal range.
        for z in ZOOM_MIN..ZOOM_MAX {
            assert!(ruler_tick_step(z) >= ruler_tick_step(z + 1));
            assert!(ruler_tick_step(z) >= 1);
        }
    }

    #[test]
    fn z_from_slider_inverts_z_slider_knob() {
        // For each representative (z, sz), the knob index maps back to z.
        for &(sz, cells) in &[(4usize, 8usize), (6, 16), (10, 12), (1, 8)] {
            for z in 0..sz {
                let knob = z_slider_knob(z, sz, cells);
                let inv = z_from_slider(knob, cells, sz);
                assert_eq!(inv, z, "round-trip z={z} sz={sz} cells={cells} knob={knob}");
            }
        }
        // Out-of-range click clamps to the last layer; single-layer is always 0.
        assert_eq!(z_from_slider(999, 8, 4), 3);
        assert_eq!(z_from_slider(3, 8, 1), 0);
    }

    #[test]
    fn visible_cells_divides_by_zoom() {
        assert_eq!(visible_cells(30, 1), 30);
        assert_eq!(visible_cells(30, 2), 15);
        assert_eq!(visible_cells(30, 3), 10);
        assert_eq!(visible_cells(0, 1), 1); // never zero
    }

    #[test]
    fn follow_pan_keeps_cursor_in_window() {
        // Window shows 10 cells; world is 40 wide.
        let pane = 10;
        let zoom = 1;
        let extent = 40;
        // Cursor at 0 with pan 0: no scroll.
        assert_eq!(follow_pan(0, 0, pane, zoom, extent), 0);
        // Cursor steps to 9 (last visible col): still no scroll.
        assert_eq!(follow_pan(0, 9, pane, zoom, extent), 0);
        // Cursor steps to 10 (off far edge): scroll so 10 is the last col.
        assert_eq!(follow_pan(0, 10, pane, zoom, extent), 1);
        // Cursor jumps far right: clamps to max_pan (40-10=30).
        assert_eq!(follow_pan(0, 39, pane, zoom, extent), 30);
        // Cursor scrolls back to near edge: window aligns to cursor.
        assert_eq!(follow_pan(20, 5, pane, zoom, extent), 5);
    }

    #[test]
    fn follow_pan_respects_zoom() {
        // At zoom 2 a 20-cell pane shows only 10 world cells.
        assert_eq!(visible_cells(20, 2), 10);
        assert_eq!(follow_pan(0, 10, 20, 2, 40), 1);
    }

    #[test]
    fn follow_pan_never_exceeds_world() {
        // Window larger than world: pan pinned to 0.
        assert_eq!(follow_pan(0, 3, 100, 1, 5), 0);
    }

    #[test]
    fn z_slider_ratio_spans_layers() {
        assert_eq!(z_slider_ratio(0, 4), 0.0);
        assert!((z_slider_ratio(3, 4) - 1.0).abs() < 1e-6);
        assert!((z_slider_ratio(1, 4) - 1.0 / 3.0).abs() < 1e-6);
        assert_eq!(z_slider_ratio(0, 1), 0.0); // single layer: no travel
    }

    #[test]
    fn z_slider_knob_spans_track() {
        // Floor at index 0, top layer at the last cell.
        assert_eq!(z_slider_knob(0, 4, 8), 0);
        assert_eq!(z_slider_knob(3, 4, 8), 7);
        // Mid layer sits proportionally.
        assert_eq!(z_slider_knob(1, 4, 7), 2); // 1/3 * 6 = 2
        assert_eq!(z_slider_knob(0, 1, 8), 0); // single layer
    }

    #[test]
    fn ruler_ticks_align_to_chunks_and_track_pan() {
        // zoom 1 ⇒ step = CHUNK (8). pan 0, pane 20 cells: ticks at 0, 8, 16.
        let t = ruler_ticks(0, 20, 1, 40);
        assert_eq!(t, vec![(0, 0), (8, 8), (16, 16)]);
        // pan 5: window covers world 5..25, chunks at 8,16,24.
        let t = ruler_ticks(5, 20, 1, 40);
        assert_eq!(t, vec![(8, 3), (16, 11), (24, 19)]);
        // zoom 2 halves the world-cell window to 10 AND tightens the step to 4:
        // world 8 fits past pan 5 (also 12 at offset 7), so ticks are 8 and 12.
        let t = ruler_ticks(5, 20, 2, 40);
        assert_eq!(t, vec![(8, 3), (12, 7)]);
    }

    #[test]
    fn ruler_ticks_clip_to_world_extent() {
        // World only 10 wide: tick at 8 fits, 16 does not.
        let t = ruler_ticks(0, 40, 1, 10);
        assert_eq!(t, vec![(0, 0), (8, 8)]);
    }

    #[test]
    fn chunk_lines_every_n() {
        assert!(is_chunk_line(0));
        assert!(is_chunk_line(8));
        assert!(is_chunk_line(16));
        assert!(!is_chunk_line(4));
    }

    #[test]
    fn elevation_column_reads_the_z_stack() {
        let mut map = Map::new(4, 4, 3);
        map.set(2, 1, 0, CellKind::Floor);
        map.set(2, 1, 2, CellKind::WallSolid);
        let col = elevation_column(&map, 2, 1);
        assert_eq!(col.len(), 3);
        assert_eq!(col[0].kind, CellKind::Floor);
        assert_eq!(col[1].kind, CellKind::Empty);
        assert_eq!(col[2].kind, CellKind::WallSolid);
        assert_eq!(col[0].z, 0);
        assert_eq!(col[2].z, 2);
    }

    #[test]
    fn elevation_column_out_of_bounds_is_empty() {
        let map = Map::new(4, 4, 3);
        let col = elevation_column(&map, 99, 99);
        assert!(col.iter().all(|s| s.kind == CellKind::Empty));
    }

    /// The FORWARD mapping `chrome::draw_grid` uses for a visible cell's
    /// top-left framebuffer pixel — duplicated here so the round-trip test pins
    /// `screen_to_cell` to the exact inverse of the painter.
    fn cell_to_screen(
        cell: (usize, usize),
        grid0: (usize, usize),
        ruler: (usize, usize),
        pan: (usize, usize),
        zoom: u8,
    ) -> (usize, usize) {
        let (wx, wy) = cell;
        let (grid_x0, grid_y0) = grid0;
        let (ruler_w, ruler_h) = ruler;
        let zoom = zoom.max(1) as usize;
        let cells_x0 = grid_x0 + ruler_w;
        let cells_y0 = grid_y0 + ruler_h;
        let px = cells_x0 + (wx - pan.0) * zoom;
        let py = cells_y0 + (wy - pan.1) * zoom;
        (px, py)
    }

    #[test]
    fn screen_to_cell_round_trips_against_forward_mapping() {
        // A generous grid pane and world; exercise several zoom/pan values.
        let (gx0, gy0, gw, gh) = (0usize, 0usize, 60usize, 40usize);
        let (rw, rh) = (3usize, 1usize); // RULER_W / RULER_H
        let extent = (32usize, 32usize);
        for &zoom in &[1u8, 2, 3, ZOOM_MAX] {
            for &pan in &[(0usize, 0usize), (5, 2), (10, 7)] {
                // Visible window of whole cells (same clamp as the painter).
                let cells_w = gw - rw;
                let cells_h = gh - rh;
                let vis_x = visible_cells(cells_w, zoom).min(extent.0 - pan.0);
                let vis_y = visible_cells(cells_h, zoom).min(extent.1 - pan.1);
                for vy in 0..vis_y {
                    for vx in 0..vis_x {
                        let (wx, wy) = (pan.0 + vx, pan.1 + vy);
                        // Forward: cell → its block's top-left pixel.
                        let (px, py) = cell_to_screen((wx, wy), (gx0, gy0), (rw, rh), pan, zoom);
                        // Inverse of the top-left pixel returns the same cell.
                        assert_eq!(
                            screen_to_cell(px, py, gx0, gy0, gw, gh, rw, rh, pan, zoom, extent),
                            Some((wx, wy)),
                            "round-trip zoom={zoom} pan={pan:?} cell=({wx},{wy})"
                        );
                        // Any pixel WITHIN the block maps to the same cell.
                        let (qx, qy) = (px + (zoom as usize - 1), py + (zoom as usize - 1));
                        assert_eq!(
                            screen_to_cell(qx, qy, gx0, gy0, gw, gh, rw, rh, pan, zoom, extent),
                            Some((wx, wy)),
                            "block-interior round-trip zoom={zoom} pan={pan:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn screen_to_cell_rejects_gutters_and_out_of_window() {
        let (gx0, gy0, gw, gh) = (0usize, 0usize, 40usize, 20usize);
        let (rw, rh) = (3usize, 1usize);
        let extent = (16usize, 16usize);
        let pan = (0, 0);
        // Top-left gutter columns/rows are not cells.
        assert_eq!(
            screen_to_cell(0, 0, gx0, gy0, gw, gh, rw, rh, pan, 1, extent),
            None
        );
        assert_eq!(
            screen_to_cell(2, 0, gx0, gy0, gw, gh, rw, rh, pan, 1, extent),
            None,
            "left ruler column"
        );
        // First real cell sits at (ruler_w, ruler_h).
        assert_eq!(
            screen_to_cell(rw, rh, gx0, gy0, gw, gh, rw, rh, pan, 1, extent),
            Some((0, 0))
        );
        // Far past the pane right edge: no cell.
        assert_eq!(
            screen_to_cell(gw + 5, rh, gx0, gy0, gw, gh, rw, rh, pan, 1, extent),
            None
        );
        // Inside the pane but past the world extent (16 wide): no cell.
        let beyond = rw + 16; // world x would be 16 == sx, out of range
        assert_eq!(
            screen_to_cell(beyond, rh, gx0, gy0, gw, gh, rw, rh, pan, 1, extent),
            None
        );
    }

    #[test]
    fn screen_to_cell_honours_grid_offset() {
        // A grid pane not anchored at the origin (e.g. centred / shifted layout).
        let (gx0, gy0, gw, gh) = (8usize, 3usize, 30usize, 18usize);
        let (rw, rh) = (3usize, 1usize);
        let extent = (16usize, 16usize);
        let pan = (2, 1);
        // The first cell of the pane is world (pan) at pixel (gx0+rw, gy0+rh).
        assert_eq!(
            screen_to_cell(gx0 + rw, gy0 + rh, gx0, gy0, gw, gh, rw, rh, pan, 1, extent),
            Some(pan)
        );
        // A point left of the pane is rejected.
        assert_eq!(
            screen_to_cell(gx0 - 1, gy0 + rh, gx0, gy0, gw, gh, rw, rh, pan, 1, extent),
            None
        );
    }
}
