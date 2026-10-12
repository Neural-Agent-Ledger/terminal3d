//! 3×3×3 cube DIRECTION PICKER for an emitter's aim (`EmitterSpec.direction`).
//!
//! Pure logic + a small modal-state struct. NO terminal, NO Bevy: the picker is
//! driven by `apply_key` (key transitions) and drawn by `chrome::draw_dir_picker`
//! from the state held here, mirroring the editor's other modal overlays (help /
//! wizard / open / saveas / palette).
//!
//! ## What it is
//!
//! The 26 non-centre cells of a 3×3×3 cube map to the 26 unit directions
//! `normalize((dx, dy, dz))` for `dx, dy, dz ∈ {-1, 0, +1}` minus `(0,0,0)`.
//! Picking a cell sets `emitter.direction` to that unit vector. The centre cell
//! shows an ARROW approximating the CURRENT direction.
//!
//! ## Axis → screen convention (matches the editor's top-down 2D view)
//!
//! The 2D grid uses `cursor.0 = x` (column, +X to the RIGHT) and `cursor.1 = y`
//! (row, +Y DOWN / south — consistent with `re_world`'s
//! `EmitterSpec.direction` doc: `(0,1,0)` = south). `+Z` is UP out of the floor.
//! So on screen:
//!
//! * `dx = +1` → `→` (right / east),   `dx = -1` → `←` (left / west)
//! * `dy = +1` → `↓` (down / south),   `dy = -1` → `↑` (up / north)
//! * `dz = +1` → `▲` (toward the ceiling), `dz = -1` → `▼` (toward the floor)
//! * `dz = 0`  → `·` (level), shown for the horizontal arrows where needed.
//!
//! Each picker LAYER is one Z plane drawn as a 3×3 of `(dx, dy)` cells, `dx`
//! across (col), `dy` down (row). The three layers are bottom (`dz = -1`), mid
//! (`dz = 0`) and top (`dz = +1`).

use glam::Vec3;

/// One cell coordinate inside the 3×3×3 cube: each axis is `-1`, `0` or `+1`.
/// `(0, 0, 0)` is the centre (no direction; shows the current-aim arrow).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CubeCell {
    pub dx: i32,
    pub dy: i32,
    pub dz: i32,
}

impl CubeCell {
    /// The unit direction this cell represents, or `None` for the centre cell.
    /// `normalize((dx, dy, dz))` — 45°-aligned by construction.
    pub fn direction(self) -> Option<Vec3> {
        if self.dx == 0 && self.dy == 0 && self.dz == 0 {
            return None;
        }
        Some(Vec3::new(self.dx as f32, self.dy as f32, self.dz as f32).normalize())
    }
}

/// The modal state of the direction picker while it is open. Held by
/// `EditorState::dir_picker` (`Some` ⇒ overlay shown + input modal). Records the
/// emitter being aimed and the in-cube cursor.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct DirPickerState {
    /// Index into `level.emitters` of the emitter being aimed.
    pub emitter: usize,
    /// Cursor column within the displayed layer: `dx ∈ {-1, 0, +1}`.
    pub cx: i32,
    /// Cursor row within the displayed layer: `dy ∈ {-1, 0, +1}`.
    pub cy: i32,
    /// Displayed layer: `dz ∈ {-1, 0, +1}` (bottom / mid / top).
    pub cz: i32,
}

impl DirPickerState {
    /// Open the picker for `emitter`, placing the cube cursor on the cell that
    /// best matches `dir` (so the user starts on the current aim).
    pub fn for_direction(emitter: usize, dir: Vec3) -> Self {
        let c = direction_to_cell(dir);
        DirPickerState {
            emitter,
            cx: c.dx,
            cy: c.dy,
            cz: c.dz,
        }
    }

    /// The cube cell under the cursor.
    pub fn cursor_cell(self) -> CubeCell {
        CubeCell {
            dx: self.cx,
            dy: self.cy,
            dz: self.cz,
        }
    }

    /// Move the cursor within the current layer, clamped to `-1..=+1` (no wrap —
    /// the 3×3 grid has hard edges so the user feels the bounds).
    pub fn move_cursor(&mut self, ddx: i32, ddy: i32) {
        self.cx = (self.cx + ddx).clamp(-1, 1);
        self.cy = (self.cy + ddy).clamp(-1, 1);
    }

    /// Switch the displayed layer by `delta` (±1), clamped to `-1..=+1`
    /// (bottom ↔ mid ↔ top).
    pub fn switch_layer(&mut self, delta: i32) {
        self.cz = (self.cz + delta).clamp(-1, 1);
    }

    /// Short label for the current layer.
    pub fn layer_label(self) -> &'static str {
        match self.cz {
            -1 => "bottom (z-)",
            1 => "top (z+)",
            _ => "mid (z0)",
        }
    }
}

/// Map an arbitrary unit (or near-unit) direction to the nearest cube cell by
/// snapping each component to `-1 / 0 / +1`. A zero vector snaps to the centre.
///
/// Snap threshold: a component whose magnitude exceeds `0.5 / sqrt(3) ≈ 0.289`
/// counts as ±1. This recovers the exact cell for any of the 26 canonical
/// directions (the smallest non-zero canonical component is `1/sqrt(3) ≈ 0.577`)
/// and gives a sensible nearest cell for off-axis input.
pub fn direction_to_cell(dir: Vec3) -> CubeCell {
    const T: f32 = 0.2887; // 0.5 / sqrt(3)
    let snap = |v: f32| -> i32 {
        if v > T {
            1
        } else if v < -T {
            -1
        } else {
            0
        }
    };
    CubeCell {
        dx: snap(dir.x),
        dy: snap(dir.y),
        dz: snap(dir.z),
    }
}

/// A compact arrow string approximating `dir` for the picker's centre cell, using
/// the 8 horizontal arrows for the `(x, y)` component plus a vertical marker for
/// the `z` component. See the module-level axis→screen convention.
///
/// Examples (per the convention above): `(1,0,0)→"→"`, `(0,-1,0)→"↑"` (north),
/// `(0,1,0)→"↓"` (south), `(1,1,0)→"↘"`, straight up `(0,0,1)→"▲"`, straight
/// down `(0,0,-1)→"▼"`.
pub fn direction_to_arrow_str(dir: Vec3) -> String {
    let c = direction_to_cell(dir);
    match (c.dx, c.dy, c.dz) {
        // Pure vertical (no horizontal component): a single vertical glyph.
        (0, 0, 1) => "▲".to_string(),
        (0, 0, -1) => "▼".to_string(),
        (0, 0, 0) => "·".to_string(),
        // Horizontal component present: the 8-way arrow, suffixed with the
        // vertical marker when the aim also tilts up/down.
        _ => {
            let h = horizontal_arrow(c.dx, c.dy);
            match c.dz {
                1 => format!("{h}▲"),
                -1 => format!("{h}▼"),
                _ => h.to_string(),
            }
        }
    }
}

/// The single dominant arrow glyph for `dir` (the centre-cell glyph when only one
/// char fits). For a tilted aim this is the horizontal arrow if there is a
/// horizontal component, else the vertical glyph. Pure; unit-tested.
pub fn direction_to_arrow(dir: Vec3) -> char {
    let c = direction_to_cell(dir);
    if c.dx == 0 && c.dy == 0 {
        match c.dz {
            1 => '▲',
            -1 => '▼',
            _ => '·',
        }
    } else {
        horizontal_arrow(c.dx, c.dy)
    }
}

/// The 8-way horizontal arrow for a snapped `(dx, dy)` (each `-1/0/+1`, not both
/// zero). `+x` = `→` east, `-x` = `←` west, `+y` = `↓` south, `-y` = `↑` north;
/// diagonals combine. `(0,0)` (no horizontal component) returns `·`.
fn horizontal_arrow(dx: i32, dy: i32) -> char {
    match (dx, dy) {
        (0, -1) => '↑',
        (0, 1) => '↓',
        (1, 0) => '→',
        (-1, 0) => '←',
        (1, -1) => '↗',
        (-1, -1) => '↖',
        (1, 1) => '↘',
        (-1, 1) => '↙',
        _ => '·',
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3::new(x, y, z)
    }

    #[test]
    fn cardinal_horizontal_arrows() {
        // Per the axis→screen convention documented at the module head.
        assert_eq!(direction_to_arrow(v(1.0, 0.0, 0.0)), '→');
        assert_eq!(direction_to_arrow(v(-1.0, 0.0, 0.0)), '←');
        assert_eq!(direction_to_arrow(v(0.0, 1.0, 0.0)), '↓'); // +y = south
        assert_eq!(direction_to_arrow(v(0.0, -1.0, 0.0)), '↑'); // -y = north
    }

    #[test]
    fn diagonal_horizontal_arrows() {
        assert_eq!(direction_to_arrow(v(1.0, -1.0, 0.0)), '↗'); // NE
        assert_eq!(direction_to_arrow(v(-1.0, -1.0, 0.0)), '↖'); // NW
        assert_eq!(direction_to_arrow(v(1.0, 1.0, 0.0)), '↘'); // SE
        assert_eq!(direction_to_arrow(v(-1.0, 1.0, 0.0)), '↙'); // SW
    }

    #[test]
    fn vertical_arrows() {
        assert_eq!(direction_to_arrow(v(0.0, 0.0, 1.0)), '▲'); // straight up
        assert_eq!(direction_to_arrow(v(0.0, 0.0, -1.0)), '▼'); // straight down
    }

    #[test]
    fn normalized_inputs_recover_the_same_arrow() {
        // The real emitter direction is a unit vector; the snap must survive it.
        assert_eq!(direction_to_arrow(v(1.0, 1.0, 0.0).normalize()), '↘');
        assert_eq!(direction_to_arrow(v(1.0, 1.0, 1.0).normalize()), '↘');
        assert_eq!(direction_to_arrow(v(0.0, 0.0, 1.0).normalize()), '▲');
    }

    #[test]
    fn arrow_str_shows_vertical_tilt() {
        // A diagonal that also tilts up appends the up marker.
        assert_eq!(direction_to_arrow_str(v(1.0, 1.0, 1.0).normalize()), "↘▲");
        assert_eq!(direction_to_arrow_str(v(1.0, 0.0, -1.0).normalize()), "→▼");
        assert_eq!(direction_to_arrow_str(v(0.0, 0.0, 1.0)), "▲");
        assert_eq!(direction_to_arrow_str(v(1.0, 0.0, 0.0)), "→");
    }

    #[test]
    fn cell_round_trips_through_direction_for_all_26() {
        // Every one of the 26 canonical cells maps to a direction that snaps
        // back to the SAME cell — the picker is lossless on its own outputs.
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if dx == 0 && dy == 0 && dz == 0 {
                        continue;
                    }
                    let cell = CubeCell { dx, dy, dz };
                    let dir = cell.direction().expect("non-centre has a direction");
                    // Unit length (45°-aligned, normalized).
                    assert!((dir.length() - 1.0).abs() < 1e-5);
                    let back = direction_to_cell(dir);
                    assert_eq!(back, cell, "cell {cell:?} -> {dir:?} -> {back:?}");
                }
            }
        }
    }

    #[test]
    fn centre_cell_has_no_direction() {
        assert_eq!(
            CubeCell {
                dx: 0,
                dy: 0,
                dz: 0
            }
            .direction(),
            None
        );
    }

    #[test]
    fn cursor_clamps_to_layer_bounds() {
        let mut p = DirPickerState::for_direction(0, v(1.0, 0.0, 0.0));
        // Start on (+1, 0, 0).
        assert_eq!(
            p.cursor_cell(),
            CubeCell {
                dx: 1,
                dy: 0,
                dz: 0
            }
        );
        // Push right past the edge — clamps, does not wrap.
        p.move_cursor(5, 0);
        assert_eq!(p.cx, 1);
        // Push up-left to the corner.
        p.move_cursor(-2, -2);
        assert_eq!((p.cx, p.cy), (-1, -1));
    }

    #[test]
    fn layer_switch_clamps() {
        let mut p = DirPickerState::for_direction(0, v(0.0, 0.0, 0.0));
        assert_eq!(p.cz, 0);
        p.switch_layer(1);
        assert_eq!(p.cz, 1);
        p.switch_layer(5); // clamp at top
        assert_eq!(p.cz, 1);
        p.switch_layer(-3); // clamp at bottom
        assert_eq!(p.cz, -1);
    }

    #[test]
    fn opens_on_the_current_aim_cell() {
        // Opening on a south-east-up aim starts the cursor on that cube cell.
        let p = DirPickerState::for_direction(2, v(1.0, 1.0, 1.0).normalize());
        assert_eq!(p.emitter, 2);
        assert_eq!(
            p.cursor_cell(),
            CubeCell {
                dx: 1,
                dy: 1,
                dz: 1
            }
        );
    }
}
