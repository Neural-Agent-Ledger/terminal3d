//! `re_render::lines` — the AUTHORED `[line]` overlay post-pass (Scalable Material
//! Overlay, spec §6 **P2 — Lines**; DP2-B authored vector strokes + DP5-C post-pass
//! render).
//!
//! This pass draws the overlay's authored polylines (`Map::overlay().lines`) as crisp
//! box-drawing VECTOR strokes over the already-finalized [`Framebuffer`], RIGHT AFTER
//! the auto-contour pass ([`crate::contour::apply`]) — so explicit authored lines
//! layer over the auto silhouettes. Today those polylines are parsed (P0) and carried
//! but never drawn; this is the render that makes them visible.
//!
//! ## Mechanism — projected-rasterize (spec DP2-B "author polyline, render as
//! post-pass"; DP5-C "lines = post-pass after sample")
//!
//! For each consecutive pair of world-space `LinePath.points`:
//! 1. **Project** both endpoints through the SAME camera/projection the DDA cast uses
//!    ([`world_to_screen`]) → screen `(col, row)` + perpendicular `depth`. A mismatch
//!    here would float the line off the geometry, so the projection is derived from
//!    the cast's own per-column ray setup (see [`world_to_screen`] docs).
//! 2. **Rasterize** a 2-D stroke between the two projected screen endpoints with a
//!    screen-space DDA (Bresenham-style major-axis walk), interpolating `depth`
//!    linearly along the segment in screen space.
//! 3. **Occlusion:** at each stroke cell draw ONLY if no geometry on that stroke
//!    point's own view ray is nearer than the point (minus a small bias) — asked of the
//!    grid on every plane ([`crate::offslice::query_row`]), not of the framebuffer's
//!    stylised backdrop depths. So nearer geometry hides the line (a line behind a
//!    wall, on any storey, does not draw; a line lying ON a surface does).
//! 4. **Draw:** write a box-drawing line glyph chosen for the stroke's screen
//!    orientation — REUSING the auto-contour vocabulary
//!    ([`crate::contour::contour_glyph_weighted`]) so the look matches — tinted by the
//!    resolved line-material colour (fallback: the contour OUTER ink).
//!
//! Cost is **O(total stroke cell-length)**, not O(cells × segments): we walk only the
//! cells a stroke actually crosses. This is why projected-rasterize is preferred over
//! a per-cell distance-to-curve query (which would be O(cells × segments) without a
//! spatial index).
//!
//! ## Perf / dirty-flag (raycaster discipline)
//!
//! Authored lines depend ONLY on `(pose, map.overlay, projection)` — there is NO
//! animation in P2. The pass hooks into the SAME post-pass stage as the contour pass,
//! which already runs only on a (re)cast: an idle static frame reuses the cached
//! framebuffer and never re-enters this pass, so a static frame after the first cast
//! skips the re-work for free. The empty-overlay case is a hard no-op (see
//! [`apply`]): a map with no lines leaves the framebuffer byte-identical to the
//! pre-pass state.
//!
//! Pure compute, no terminal, no per-frame heap allocation in the draw loop (the only
//! state is a couple of stack scalars per segment). Engine-pure like the rest of
//! `re_render`.

use re_core::{Framebuffer, Vec3};
use re_world::{CameraPose, Map};

use crate::ascii_shader::{pack_rgb, MaterialRegistry};
use crate::contour::{self, EdgeClass};
use crate::{
    gameplay_eye_ref_z, offslice, pitch_row_shift, view_params, Fov, SlopeDir, SlopeTheme, PERP_MIN,
};

/// Occlusion depth bias (world cells). A stroke cell draws when the first geometry on
/// the stroke point's own view ray is no nearer than the point minus this slack — so a
/// line authored to lie exactly ON a surface (the ray meets that surface at the
/// point itself, up to the screen-space interpolation of the segment's depth) still
/// draws, while a line genuinely BEHIND nearer geometry is hidden. Small on purpose: a
/// fraction of a cell, so a line never punches through a wall it sits behind.
pub const LINE_DEPTH_BIAS: f32 = 0.25;

/// Near clip plane for line segments (world cells of perpendicular depth). A segment
/// that crosses the camera plane is cut here instead of being dropped: a floor cable
/// running away from a player standing on or beside it starts BEHIND the camera, and
/// dropping every such segment left the cable invisible exactly where it was nearest.
pub const LINE_NEAR: f32 = 0.2;

/// Result of projecting a world point through the cast's camera/projection.
///
/// `col` / `row` are CELL-CENTRE coordinates: the cell a point lands in is
/// `(col.round(), row.round())`, i.e. the cell whose area contains the projected point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScreenPoint {
    /// Fractional screen column (`0.0` = the centre of column 0).
    pub col: f32,
    /// Fractional screen row (`0.0` = the centre of row 0).
    pub row: f32,
    /// Camera-plane-PERPENDICULAR distance (the fisheye-free depth the cast stores in
    /// [`Cell::depth`]). Always `> 0` for a returned point (behind-camera is clipped).
    ///
    /// [`Cell::depth`]: re_core::Cell::depth
    pub depth: f32,
}

/// The per-cast view basis the line projection shares — derived ONCE from the camera +
/// FOV + framebuffer size, then reused for every endpoint so a whole overlay's worth
/// of lines costs only the per-endpoint projection. Mirrors the head of the cast
/// ([`crate::raycast_full_memo`] / [`crate::SubViewBasis`]) so a projected point lands
/// EXACTLY where the corresponding cast column/row would.
#[derive(Clone, Copy, Debug)]
pub struct LineViewBasis {
    /// Camera world position (xy used for the lateral/perp split).
    pos: Vec3,
    /// Facing unit direction in the xy-plane: `(sin yaw, cos yaw)`.
    dir_x: f32,
    dir_y: f32,
    /// Camera-plane UNIT axis `(cos yaw, -sin yaw)` (the screen-x direction in world
    /// space). The cast's plane vector is this scaled by `plane_scale`; we keep it
    /// unit and divide by `plane_scale` in the inverse so the math reads cleanly.
    plane_x: f32,
    plane_y: f32,
    /// `tan(hfov/2)` — the camera-plane half-width. `camera_x = lateral/(perp*scale)`.
    plane_scale: f32,
    /// `(rows/2)/tan(vfov/2)` — vertical projection-plane distance, in screen rows.
    proj_plane_dist: f32,
    /// Pitch-shifted horizon row (`rows/2 + pitch_row_shift`), as the continuous row
    /// coordinate of the horizon line (`0.0` = the top edge of row 0).
    horizon: f32,
    /// The eye's WORLD height in the cast's projection. The gameplay cast renders slice
    /// `cell_z` with `eye = cell_z + ½ + vshift/proj` and `vshift = (cam.z − ref)·proj`,
    /// `ref = cell_z + ½` ([`gameplay_eye_ref_z`]) — which is exactly `cam.z`. One
    /// perspective for walls, every other plane and these lines
    /// (`crate::offslice::VProj`).
    eye: f32,
    /// The slice plane the gameplay cast walks (for the occlusion query's roles).
    cell_z: i32,
    /// Framebuffer width in columns (for the `camera_x → col` map).
    cols: f32,
}

impl LineViewBasis {
    /// Build the shared projection basis for a `cols`×`rows` frame, in the GAMEPLAY
    /// vertical model the composite / `--dump` world layers are cast with.
    pub fn new(cam: &CameraPose, fov: &Fov, cols: usize, rows: usize) -> Self {
        let view = view_params(cols.max(1), rows.max(1), fov);
        let (dir_x, dir_y) = cam.dir();
        let (yc, ys) = (cam.yaw.cos(), cam.yaw.sin());
        let horizon = (rows as i32 / 2 + pitch_row_shift(cam.pitch, rows)) as f32;
        LineViewBasis {
            pos: cam.pos,
            dir_x,
            dir_y,
            plane_x: yc,
            plane_y: -ys,
            plane_scale: view.plane_scale,
            proj_plane_dist: view.proj_plane_dist,
            horizon,
            eye: cam.pos.z,
            cell_z: gameplay_eye_ref_z(cam.pos.z).floor() as i32,
            cols: cols.max(1) as f32,
        }
    }

    /// Perpendicular (camera-plane) depth of a world point; negative behind.
    #[inline]
    pub fn perp(&self, world: Vec3) -> f32 {
        (world.x - self.pos.x) * self.dir_x + (world.y - self.pos.y) * self.dir_y
    }

    /// The segment `a → b` clipped to the near plane [`LINE_NEAR`], or `None` when it
    /// lies wholly behind it.
    pub fn clip_near(&self, a: Vec3, b: Vec3) -> Option<(Vec3, Vec3)> {
        let (pa, pb) = (self.perp(a), self.perp(b));
        match (pa >= LINE_NEAR, pb >= LINE_NEAR) {
            (true, true) => Some((a, b)),
            (false, false) => None,
            (a_in, _) => {
                let t = (LINE_NEAR - pa) / (pb - pa);
                let cut = a + (b - a) * t;
                Some(if a_in { (a, cut) } else { (cut, b) })
            }
        }
    }

    /// Project a world point → screen `(col, row, depth)`, or `None` if the point is
    /// AT/BEHIND the camera plane (`perp <= 0`). [`apply`] clips every segment to the
    /// near plane first ([`Self::clip_near`]), so a segment running from behind the
    /// camera still draws its visible part.
    ///
    /// Inverse of the cast's per-column ray model:
    /// * `rel = world.xy - cam.xy`
    /// * `perp   = rel · dir`        (the facing component — the fisheye-free depth)
    /// * `lateral = rel · plane`     (the screen-x component, plane axis is unit)
    /// * `camera_x = lateral / (perp * plane_scale)` — the cast builds the column ray
    ///   as `dir + plane_scale*plane * camera_x`, so a world point at screen `camera_x`
    ///   has `lateral/perp = plane_scale * camera_x`; invert it.
    /// * `col = (camera_x + 1)/2 * cols - 0.5` — inverse of the cast's
    ///   `camera_x = 2*(col+0.5)/cols - 1`.
    /// * `row = horizon + proj·(eye − world.z)/perp − ½` — the SAME vertical projection
    ///   every surface of the cast uses, `z` in world units (cell `(x, y, k)` spans
    ///   `[k, k+1]`, floor top `1.0`, standing eye `1.5`), shifted to cell-centre
    ///   coordinates. So a point at a wall's top edge lands in the row that edge passes
    ///   through, and the wall's own drawn rows (the rows whose centres lie on the wall,
    ///   `offslice::span_rows`) start right there.
    #[inline]
    pub fn project(&self, world: Vec3) -> Option<ScreenPoint> {
        let rx = world.x - self.pos.x;
        let ry = world.y - self.pos.y;
        let perp = rx * self.dir_x + ry * self.dir_y;
        if perp <= PERP_MIN {
            return None; // at or behind the camera plane → clip the endpoint.
        }
        let lateral = rx * self.plane_x + ry * self.plane_y;
        let camera_x = lateral / (perp * self.plane_scale);
        let col = (camera_x + 1.0) * 0.5 * self.cols - 0.5;
        let row = self.horizon + self.proj_plane_dist * (self.eye - world.z) / perp - 0.5;
        Some(ScreenPoint {
            col,
            row,
            depth: perp,
        })
    }

    /// Vertical projection-plane distance (rows), i.e. `(rows/2)/tan(vfov/2)` — the
    /// same value [`Self::project`] uses to turn a world height into a screen row.
    /// Exposed so a sibling post-pass ([`crate::node_sprite`]) can size a
    /// screen-space footprint (`diameter_rows = 2·R·proj_plane_dist/depth`) with the
    /// exact same projection the cast and the lines use — one basis, no drift.
    #[inline]
    pub(crate) fn proj_plane_dist(&self) -> f32 {
        self.proj_plane_dist
    }

    /// As [`Self::project`], but for a small SPHERE of `radius` around `world`
    /// rather than a dimensionless point (`crate::node_sprite`'s node/switch orb,
    /// D5 "edge culling"). [`Self::project`] alone drops a point whose exact centre
    /// sits at/behind the camera plane (`perp <= PERP_MIN`) — correct for a
    /// dimensionless line endpoint, but a puzzle-node sphere whose centre is just
    /// barely behind that plane still has most of its bulk in front of it (the
    /// plane passes through the CAMERA position, not the near clip distance), and
    /// dropping it wholesale is a visible pop rather than a graceful partial reveal.
    ///
    /// Returns `None` only when the WHOLE sphere is behind the plane
    /// (`perp + radius <= PERP_MIN`). Otherwise, if the raw centre already clears
    /// the plane, this is identical to [`Self::project`]; if not, the point is
    /// nudged forward along the view direction just far enough to clear it (a
    /// bounded approximation — the sphere's near cap, not an exact silhouette —
    /// good enough for a sub-cell-radius orb, and it is only asked to do
    /// anything the moment the plain projection would have vanished the node
    /// entirely).
    pub(crate) fn project_sphere(&self, world: Vec3, radius: f32) -> Option<ScreenPoint> {
        let perp = self.perp(world);
        if perp + radius <= PERP_MIN {
            return None; // the whole sphere is behind the camera plane.
        }
        if perp > PERP_MIN {
            return self.project(world);
        }
        let nudge = PERP_MIN - perp + 1e-3;
        let eff = world + Vec3::new(self.dir_x, self.dir_y, 0.0) * nudge;
        self.project(eff)
    }

    /// Is a stroke point at screen `(col, row)` (cell-centre coordinates) and depth
    /// `depth` in front of every piece of geometry on its own view ray?
    ///
    /// Asks the grid, not the framebuffer: the framebuffer's floor / ceiling rows carry
    /// the backdrop's stylised depths, which are not distances to anything, so a depth
    /// test against them would hide lines above the second storey and let every storey
    /// slab leak. Walking the exact ray the point lies on through every plane
    /// (`offslice::query_row`) finds the true first surface — a wall on any plane, a
    /// slab, a table top — and a point ON a surface meets it at its own depth.
    ///
    /// `pub(crate)`: also the occlusion test [`crate::node_sprite`] uses for a
    /// sprite's rim/body cells (at the sphere-front depth `depth − R`) and for its
    /// pipe-connection stubs — the same "ask the grid on every plane" contract.
    pub(crate) fn visible(&self, map: &Map, col: f32, row: f32, depth: f32) -> bool {
        self.visible_except(map, col, row, depth, None)
    }

    /// As [`Self::visible`], but the grid cell `except` (if any) never counts as an
    /// occluder — the puzzle-node sprite pass ([`crate::node_sprite`]) uses this for
    /// an EMBEDDED node (D4): the node's own voxel is deliberately left SOLID (so it
    /// paints a recessed socket face instead of a hole), but that same voxel must
    /// never hide the small orb drawn on top of it, or the sprite would occlude
    /// itself into invisibility the instant its own cell becomes the nearest
    /// surface on its own view ray.
    pub(crate) fn visible_except(
        &self,
        map: &Map,
        col: f32,
        row: f32,
        depth: f32,
        except: Option<(i32, i32, i32)>,
    ) -> bool {
        let camera_x = 2.0 * (col + 0.5) / self.cols - 1.0;
        let rdx = self.dir_x + self.plane_x * self.plane_scale * camera_x;
        let rdy = self.dir_y + self.plane_y * self.plane_scale * camera_x;
        let k = (self.horizon - (row + 0.5)) / self.proj_plane_dist;
        match offslice::query_row(
            map,
            self.cell_z,
            self.pos.x,
            self.pos.y,
            rdx,
            rdy,
            self.eye,
            k,
            offslice::MAX_STEPS,
            |hit| match except {
                Some((ex, ey, ez)) => !(hit.x == ex && hit.y == ey && hit.plane == ez),
                None => true,
            },
        ) {
            Some(hit) => depth <= hit.s + LINE_DEPTH_BIAS,
            None => true,
        }
    }
}

/// Standalone projection helper (test/diagnostic entry): project `world` through the
/// camera/projection for a `cols`×`rows` frame (see [`LineViewBasis::project`]).
/// Returns `None` for a point at/behind the camera plane.
pub fn world_to_screen(
    cam: &CameraPose,
    fov: &Fov,
    cols: usize,
    rows: usize,
    world: Vec3,
) -> Option<ScreenPoint> {
    LineViewBasis::new(cam, fov, cols, rows).project(world)
}

/// Resolve a line's stroke colour → packed `0x00RRGGBB`.
///
/// 1. `palette_id > 0` — the LEVEL-AUTHORED `[material NAME]` the line binds, resolved
///    by `re_level` to its 1-based colour-palette id ([`Map::line_palette_id`]) and
///    registered by [`crate::registry_for_map`] as that palette slot. This is the
///    set-dressing E0 fix: the registry's name table only holds `'static` built-in and
///    catalog names, so a level-authored name never resolved through
///    [`MaterialRegistry::id_of`] and every authored stroke drew in the fallback white.
/// 2. Otherwise the NAME against the registry's named entries (built-ins + catalog,
///    e.g. `rust`, `neon`).
/// 3. Otherwise the contour OUTER ink.
///
/// The representative colour is the material's foreground palette evaluated at its
/// mid-point (`eval(0.5)`; for an authored `fg` that is a flat palette, so exactly the
/// authored `fg`) — a stable single hue for a crisp ink stroke (a line is one glyph with
/// no per-cell texture coordinate to sample).
#[inline]
pub fn resolve_line_color(name: &str, palette_id: u16, registry: &MaterialRegistry) -> u32 {
    if let Some(id) = registry.color_material(palette_id) {
        return pack_rgb(registry.get(id).palette.eval(0.5));
    }
    let name = name.trim();
    if !name.is_empty() {
        if let Some(id) = registry.id_of(name) {
            return pack_rgb(registry.get(id).palette.eval(0.5));
        }
    }
    contour::CONTOUR_FG
}

/// The authored-line post-pass: draw every `map.overlay().lines` stroke over `fb`.
///
/// **No-op guarantee (back-compat, MANDATORY):** an empty `overlay.lines` returns
/// immediately, leaving `fb` byte-identical to the pre-pass state — every existing
/// (no-line) level renders exactly as today. A degenerate path (< 2 points) draws
/// nothing.
///
/// Runs AFTER [`crate::contour::apply`] in every pipeline path that casts (the
/// single-shot dump AND the composite live path), so authored lines overdraw the auto
/// silhouettes.
pub fn apply(
    fb: &mut Framebuffer,
    map: &Map,
    cam: &CameraPose,
    fov: &Fov,
    registry: &MaterialRegistry,
) {
    let overlay = map.overlay();
    if overlay.lines.is_empty() {
        return; // hard no-op: byte-identical to the pre-pass framebuffer.
    }
    let (cols, rows) = (fb.cols(), fb.rows());
    if cols == 0 || rows == 0 {
        return;
    }
    let basis = LineViewBasis::new(cam, fov, cols, rows);

    for (index, line) in overlay.lines.iter().enumerate() {
        let pts = &line.path.points;
        if pts.len() < 2 {
            continue; // a degenerate/inert path draws nothing (carried, never drawn).
        }
        let color = resolve_line_color(&line.material.0, map.line_palette_id(index), registry);
        // Each consecutive pair is one segment, clipped to the near plane (a segment
        // wholly behind it draws nothing; one crossing it is cut at `LINE_NEAR`).
        for win in pts.windows(2) {
            let Some((wa, wb)) = basis.clip_near(win[0], win[1]) else {
                continue;
            };
            let (Some(a), Some(b)) = (basis.project(wa), basis.project(wb)) else {
                continue;
            };
            draw_segment(fb, cols, rows, a, b, color, |c, r, d| {
                basis.visible(map, c, r, d)
            });
        }
    }
}

/// Rasterize one projected segment `a → b` into `fb`: a screen-space DDA walking the
/// MAJOR axis one cell at a time, interpolating depth linearly, and stamping an
/// orientation-correct box-drawing glyph tinted `color` on every cell whose stroke
/// point `visible(col, row, depth)` accepts (the occlusion test — see
/// [`LineViewBasis::visible`]). Off-grid cells are skipped (the segment may project
/// partly outside the viewport).
/// `pub(crate)`: reused by [`crate::node_sprite`] for the pipe-connection stubs (a
/// short stroke from a node's rim to the shared face of an adjacent pipe cell) —
/// same occlusion-aware box-drawing stroke, same visual vocabulary.
pub(crate) fn draw_segment(
    fb: &mut Framebuffer,
    cols: usize,
    rows: usize,
    a: ScreenPoint,
    b: ScreenPoint,
    color: u32,
    visible: impl Fn(f32, f32, f32) -> bool,
) {
    // Cell-centre sampling: a fractional col/row `c` lies in cell `c.round()`. Work in
    // cell-index space from the rounded endpoints so the walk lands on whole cells.
    let dx = b.col - a.col;
    let dy = b.row - a.row;
    // The whole segment shares ONE orientation glyph, from its full screen direction.
    let glyph = u32::from(stroke_glyph(dx, dy));
    // Clip the screen segment to the viewport first (parametric, Liang–Barsky): a
    // segment clipped to the near plane can project thousands of rows off-screen, and
    // walking those rows one by one is wasted work.
    let Some((t0, t1)) = clip_to_rect(a.col, a.row, dx, dy, cols as f32, rows as f32) else {
        return;
    };
    let (x0, y0) = (a.col + dx * t0, a.row + dy * t0);
    let (sdx, sdy) = (dx * (t1 - t0), dy * (t1 - t0));
    // Number of steps = the longer screen extent, so we never skip a cell on the major
    // axis (the classic DDA). `+1` so both endpoints are sampled. A zero-length segment
    // (both endpoints in one cell) still stamps that one cell.
    let steps = sdx.abs().max(sdy.abs()).floor() as i32 + 1;
    let inv = if steps > 1 {
        1.0 / (steps - 1) as f32
    } else {
        0.0
    };
    // PERSPECTIVE-CORRECT depth: `1/depth` (not depth) is linear in screen space. A
    // segment running away from the camera spends most of its screen length near its
    // close end, so a linear depth ran far too deep there and the occlusion test hid
    // the stroke behind the very floor it lies on.
    let (ia, ib) = (1.0 / a.depth.max(PERP_MIN), 1.0 / b.depth.max(PERP_MIN));

    // (The glyph comes from the SAME marching-mask box-drawing vocabulary the
    // auto-contour uses: a near-horizontal run → `━`, near-vertical → `┃`, diagonal →
    // a half-cell slope.)
    for s in 0..steps {
        let u = s as f32 * inv; // 0..1 along the CLIPPED segment in screen space.
        let cf = x0 + sdx * u;
        let rf = y0 + sdy * u;
        let t = t0 + (t1 - t0) * u; // 0..1 along the whole projected segment.
        let depth = 1.0 / (ia + (ib - ia) * t);
        let ci = cf.round() as i32;
        let ri = rf.round() as i32;
        if ci < 0 || ri < 0 || ci as usize >= cols || ri as usize >= rows {
            continue;
        }
        let Some(cell) = fb.get(ci as usize, ri as usize) else {
            continue;
        };
        // OCCLUSION: draw only if the line is at/in front of the geometry here.
        if !visible(cf, rf, depth) {
            continue; // behind nearer geometry → hidden.
        }

        let mut out = *cell;
        out.glyph = glyph;
        out.fg = color;
        // Keep the cell's own bg so the stroke reads as ink ON the surface, not a
        // sticker; a BOLD attribute so an authored line reads as crisp primary ink
        // (matching the contour OUTER hierarchy weight).
        out.style = re_core::CellStyle {
            flags: re_core::CellStyle::BOLD,
            ..re_core::CellStyle::NONE
        };
        fb.set(ci as usize, ri as usize, out);
    }
}

/// Parametric clip of the screen segment `p + t·d`, `t ∈ [0, 1]`, to the cell-centre
/// viewport `[-0.5, cols-0.5] × [-0.5, rows-0.5]` (Liang–Barsky). Returns the visible
/// `(t0, t1)`, or `None` when the segment misses the viewport.
fn clip_to_rect(px: f32, py: f32, dx: f32, dy: f32, cols: f32, rows: f32) -> Option<(f32, f32)> {
    let (mut t0, mut t1) = (0.0f32, 1.0f32);
    for (p, q) in [
        (-dx, px + 0.5),
        (dx, cols - 0.5 - px),
        (-dy, py + 0.5),
        (dy, rows - 0.5 - py),
    ] {
        if p == 0.0 {
            if q < 0.0 {
                return None;
            }
        } else {
            let r = q / p;
            if p < 0.0 {
                t0 = t0.max(r);
            } else {
                t1 = t1.min(r);
            }
        }
    }
    (t0 <= t1).then_some((t0, t1))
}

/// Choose a box-drawing glyph for a stroke whose screen-space direction is `(dx, dy)`
/// (cols, rows). Reuses the auto-contour vocabulary via
/// [`contour::contour_glyph_weighted`] with the OUTER (heavy) weight so authored lines
/// read as crisp primary strokes.
///
/// IMPORTANT — the contour mask convention is "which orthogonal neighbour has an EDGE",
/// and the drawn glyph is the rule PERPENDICULAR to that edge direction (an edge to the
/// N/S draws a horizontal rule `─`; an edge to the E/W draws a vertical rule `│` — see
/// [`contour::contour_glyph_weighted`]). So to draw a stroke that RUNS horizontally we
/// pass the `N|S` mask (horizontal rule `━`), and a vertical run passes `E|W` (`┃`):
/// * predominantly horizontal run (`|dx| > |dy|·2`) → `N|S` → `━`,
/// * predominantly vertical run  (`|dy| > |dx|·2`) → `E|W` → `┃`,
/// * a diagonal → a TRUE half-cell diagonal glyph from the shared slope vocabulary
///   ([`SlopeTheme::UNICODE`], the same `╱`/`╲` the auto-contour de-angularization uses)
///   so an authored diagonal reads as a crisp straight line, NOT a box-corner staircase:
///   a down-right (or up-left) screen run → `╲`; an up-right (or down-left) run → `╱`.
#[inline]
fn stroke_glyph(dx: f32, dy: f32) -> char {
    let (ax, ay) = (dx.abs(), dy.abs());
    if ax > ay * 2.0 {
        // Horizontal run → horizontal rule: pass the N|S (horizontal-seam) mask.
        contour::contour_glyph_weighted(contour::N | contour::S, EdgeClass::Outer)
    } else if ay > ax * 2.0 {
        // Vertical run → vertical rule: pass the E|W (vertical-seam) mask.
        contour::contour_glyph_weighted(contour::E | contour::W, EdgeClass::Outer)
    } else {
        // Diagonal: a true half-cell diagonal. Screen rows increase DOWNWARD, so a run
        // with the same sign on dx,dy descends to the right → `╲`; opposite signs ascend
        // to the right → `╱`. (A zero diagonal — both small — falls here too and reads
        // as `╲`; a degenerate single-cell segment is rare and visually inconsequential.)
        let theme = SlopeTheme::UNICODE;
        if (dx >= 0.0) == (dy >= 0.0) {
            theme.line_for(SlopeDir::DownRight) // `╲`
        } else {
            theme.line_for(SlopeDir::DownLeft) // `╱`
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::STANDING_EYE_Z;
    use re_core::{Cell, SurfaceKind};
    use re_world::Map;

    fn cam_at(x: f32, y: f32, z: f32, yaw: f32) -> CameraPose {
        CameraPose {
            pos: Vec3::new(x, y, z),
            yaw,
            pitch: 0.0,
        }
    }

    /// A solid wall cell at `depth`, for occlusion fixtures.
    fn wall(depth: f32) -> Cell {
        Cell {
            glyph: u32::from('.'),
            fg: 0x0040_4040,
            bg: 0x0010_1010,
            depth,
            surface: SurfaceKind::WallSolid,
            ..Cell::default()
        }
    }

    /// `draw_segment` against the framebuffer's own depths (the fixture occluder).
    fn draw_fb(fb: &mut Framebuffer, cols: usize, rows: usize, a: ScreenPoint, b: ScreenPoint) {
        let depths: Vec<f32> = fb.cells().iter().map(|c| c.depth).collect();
        draw_segment(fb, cols, rows, a, b, 0x00FF_FFFF, |c, r, d| {
            d <= depths[r.round() as usize * cols + c.round() as usize] + LINE_DEPTH_BIAS
        });
    }

    // (a) world_to_screen projects a known world point to the expected screen cell + depth.
    #[test]
    fn project_point_dead_ahead_lands_on_centre_column_at_true_depth() {
        // Camera at origin looking +y (yaw 0 ⇒ dir = (0,1)). A point 5 cells straight
        // ahead at eye height projects to the centre column and depth == 5.
        let cols = 40;
        let rows = 24;
        let cam = cam_at(0.0, 0.0, STANDING_EYE_Z, 0.0);
        let fov = Fov::default();
        // A point at the eye's own world height makes the vertical term 0: it lands on
        // the horizon LINE, i.e. half a row above the centre of the row below it.
        let world = Vec3::new(0.0, 5.0, STANDING_EYE_Z);
        let sp = world_to_screen(&cam, &fov, cols, rows, world).expect("in front");
        assert!(
            (sp.depth - 5.0).abs() < 1e-3,
            "perp depth {} != 5",
            sp.depth
        );
        // Centre column: camera_x == 0 ⇒ col == cols/2 - 0.5.
        assert!(
            (sp.col - (cols as f32 / 2.0 - 0.5)).abs() < 1e-3,
            "dead-ahead point must land on the centre column, got col {}",
            sp.col
        );
        let horizon = rows as f32 / 2.0;
        assert!(
            (sp.row - (horizon - 0.5)).abs() < 1e-3,
            "eye-height point must land on the horizon row, got {}",
            sp.row
        );
    }

    #[test]
    fn project_matches_cast_column_ray_for_an_off_centre_point() {
        // A point offset laterally must invert the cast's `camera_x` map: project it,
        // rebuild the column ray for that fractional column, and confirm the ray points
        // back through the world point (lateral/perp ratio matches plane_scale*camera_x).
        let (cols, rows) = (60usize, 30usize);
        let cam = cam_at(2.0, 1.0, STANDING_EYE_Z, 0.3);
        let fov = Fov::default();
        let world = Vec3::new(5.0, 7.0, 1.0);
        let sp = world_to_screen(&cam, &fov, cols, rows, world).expect("in front");
        let view = view_params(cols, rows, &fov);
        let (dir_x, dir_y) = cam.dir();
        let plane_x = cam.yaw.cos() * view.plane_scale;
        let plane_y = -cam.yaw.sin() * view.plane_scale;
        let camera_x = 2.0 * (sp.col + 0.5) / cols as f32 - 1.0;
        let rdx = dir_x + plane_x * camera_x;
        let rdy = dir_y + plane_y * camera_x;
        // The world point lies along this ray from the camera: (world - pos) ∥ ray.
        let wx = world.x - cam.pos.x;
        let wy = world.y - cam.pos.y;
        let cross = wx * rdy - wy * rdx; // ∥ ⇒ cross ≈ 0
        assert!(
            cross.abs() < 1e-3,
            "projected column ray must pass through the world point (cross {cross})"
        );
    }

    #[test]
    fn behind_camera_point_is_clipped() {
        let cam = cam_at(0.0, 0.0, STANDING_EYE_Z, 0.0); // looking +y
        let fov = Fov::default();
        // A point behind the camera (−y) is at/behind the plane ⇒ None.
        assert!(world_to_screen(&cam, &fov, 40, 24, Vec3::new(0.0, -3.0, 1.0)).is_none());
    }

    // (b) a line segment writes stroke cells along the expected screen path.
    #[test]
    fn segment_writes_stroke_cells_along_its_screen_path() {
        let (cols, rows) = (20usize, 9usize);
        let mut fb = Framebuffer::new(cols, rows);
        // Fill with FAR geometry so nothing occludes (depth large).
        for r in 0..rows {
            for c in 0..cols {
                fb.set(c, r, wall(100.0));
            }
        }
        // A horizontal stroke across the middle row, depth 5 (well in front).
        let a = ScreenPoint {
            col: 3.0,
            row: 4.0,
            depth: 5.0,
        };
        let b = ScreenPoint {
            col: 15.0,
            row: 4.0,
            depth: 5.0,
        };
        draw_fb(&mut fb, cols, rows, a, b);
        // Every cell from col 3..=15 on row 4 must now carry a horizontal rule glyph.
        for c in 3..=15 {
            let g = char::from_u32(fb.get(c, 4).unwrap().glyph).unwrap();
            assert_eq!(g, '━', "stroke cell at col {c} must be a horizontal rule");
        }
        // A cell OFF the stroke path is untouched (still the '.' body glyph).
        assert_eq!(char::from_u32(fb.get(3, 0).unwrap().glyph).unwrap(), '.');
    }

    #[test]
    fn diagonal_segment_is_a_crisp_line_not_a_stairstep_gap() {
        // A 45° stroke must stamp ONE cell per column with NO gaps (crisp diagonal).
        let (cols, rows) = (12usize, 12usize);
        let mut fb = Framebuffer::new(cols, rows);
        for r in 0..rows {
            for c in 0..cols {
                fb.set(c, r, wall(100.0));
            }
        }
        let a = ScreenPoint {
            col: 1.0,
            row: 1.0,
            depth: 5.0,
        };
        let b = ScreenPoint {
            col: 9.0,
            row: 9.0,
            depth: 5.0,
        };
        draw_fb(&mut fb, cols, rows, a, b);
        // Each step (1,1)..(9,9) must be a drawn corner glyph — a continuous diagonal.
        for i in 1..=9 {
            let g = char::from_u32(fb.get(i, i).unwrap().glyph).unwrap();
            assert_ne!(g, '.', "diagonal cell ({i},{i}) must be drawn (no gap)");
        }
    }

    // (c) a stroke cell BEHIND nearer geometry is NOT drawn (occlusion).
    #[test]
    fn stroke_behind_nearer_geometry_is_occluded() {
        let (cols, rows) = (10usize, 3usize);
        let mut fb = Framebuffer::new(cols, rows);
        // Row 1: a NEAR wall (depth 2) over the whole row.
        for c in 0..cols {
            fb.set(c, 1, wall(2.0));
        }
        // A stroke at depth 5 (BEHIND the depth-2 wall) across row 1.
        let a = ScreenPoint {
            col: 1.0,
            row: 1.0,
            depth: 5.0,
        };
        let b = ScreenPoint {
            col: 8.0,
            row: 1.0,
            depth: 5.0,
        };
        draw_fb(&mut fb, cols, rows, a, b);
        // The wall is nearer (2 < 5 - bias) ⇒ the line is hidden: cells keep '.'.
        for c in 1..=8 {
            assert_eq!(
                char::from_u32(fb.get(c, 1).unwrap().glyph).unwrap(),
                '.',
                "occluded stroke cell {c} must NOT draw over nearer geometry"
            );
        }
    }

    #[test]
    fn stroke_on_a_surface_at_equal_depth_draws() {
        // A line lying ON the surface (same depth) must draw (bias tolerates it).
        let (cols, rows) = (10usize, 3usize);
        let mut fb = Framebuffer::new(cols, rows);
        for c in 0..cols {
            fb.set(c, 1, wall(5.0));
        }
        let a = ScreenPoint {
            col: 1.0,
            row: 1.0,
            depth: 5.0,
        };
        let b = ScreenPoint {
            col: 8.0,
            row: 1.0,
            depth: 5.0,
        };
        draw_fb(&mut fb, cols, rows, a, b);
        for c in 1..=8 {
            assert_eq!(
                char::from_u32(fb.get(c, 1).unwrap().glyph).unwrap(),
                '━',
                "a line on the surface (equal depth) must draw"
            );
        }
    }

    // (d) a map with NO lines leaves the framebuffer byte-identical (no-op proof).
    #[test]
    fn empty_overlay_is_a_strict_noop() {
        let (cols, rows) = (16usize, 8usize);
        let mut fb = Framebuffer::new(cols, rows);
        for r in 0..rows {
            for c in 0..cols {
                fb.set(c, r, wall((c + r) as f32 + 1.0));
            }
        }
        let before = fb.cells().to_vec();
        let map = Map::new(8, 8, 4); // default ⇒ empty overlay (no lines).
        let cam = cam_at(4.0, 4.0, STANDING_EYE_Z, 0.0);
        let registry = MaterialRegistry::new();
        apply(&mut fb, &map, &cam, &Fov::default(), &registry);
        assert_eq!(
            fb.cells(),
            &before[..],
            "a map with no overlay lines must leave the framebuffer byte-identical"
        );
    }

    #[test]
    fn resolve_color_falls_back_to_contour_ink_for_unknown_name() {
        let registry = MaterialRegistry::new();
        assert_eq!(resolve_line_color("", 0, &registry), contour::CONTOUR_FG);
        assert_eq!(
            resolve_line_color("no_such_material", 0, &registry),
            contour::CONTOUR_FG
        );
        // A palette id past the (empty) palette also falls back.
        assert_eq!(resolve_line_color("x", 3, &registry), contour::CONTOUR_FG);
    }

    // ── Set-dressing E0 (F5): an authored line material reaches the framebuffer ──

    /// A minimal level: a 10x10 room, camera at (1,5) facing +x, and two strokes
    /// across the floor in front of it — one bound to a LEVEL-AUTHORED material, one
    /// to a CATALOG name. `extra` is appended verbatim (more sections).
    fn line_level(extra: &str) -> re_level::Level {
        const BASE: &str = "\
[meta]
name = line-colour-fixture
[room]
size = 10 10 4
ambient = solid
[material ink_red]
fg = #FF2040
bg = #200408
[line authored]
path = 4,3,1.0; 4,7,1.0
material = ink_red
[line catalog]
path = 6,3,1.0; 6,7,1.0
material = rust
[node p]
pos = 8 1 1
glyph = P
is_input = yes
[node q]
pos = 8 8 1
glyph = Q
is_output = yes
threshold = 1
[pipe p->q]
state = ON
path = (8,1,1) (8,8,1)
[pedestal]
center = 8 8 1
volume = 1 1 1
[door]
pos = 8 6 1
[player]
spawn = 1 5 1
facing = +x
";
        let src = format!("{BASE}{extra}");
        re_level::load_string(&src, "line-colour-fixture").expect("fixture parses")
    }

    fn render_with_lines(level: &re_level::Level) -> Framebuffer {
        let map = level.build_map();
        let cam = level.spawn_camera();
        let mut fb = Framebuffer::new(120, 40);
        crate::raycast_fov(&map, &cam, &Fov::default(), &mut fb);
        fb
    }

    /// Count cells carrying an authored-stroke glyph in colour `rgb`.
    fn stroke_cells_in(fb: &Framebuffer, rgb: u32) -> usize {
        let strokes = ['━', '┃', '╱', '╲'];
        fb.cells()
            .iter()
            .filter(|c| {
                c.fg == rgb && char::from_u32(c.glyph).is_some_and(|g| strokes.contains(&g))
            })
            .count()
    }

    /// E0: a `[line]` bound to a level-authored `[material]` draws in THAT material's
    /// colour. Before the fix the name never resolved (the registry's name table only
    /// holds `'static` built-in / catalog names) and every stroke was contour white.
    #[test]
    fn authored_line_material_colour_reaches_the_framebuffer() {
        let level = line_level("");
        let map = level.build_map();
        assert_eq!(
            map.line_palette_id(0),
            1,
            "ink_red is the first authored material"
        );
        assert_eq!(
            map.line_palette_id(1),
            0,
            "a catalog name is not level-authored"
        );
        let fb = render_with_lines(&level);
        let red = stroke_cells_in(&fb, 0x00FF_2040);
        assert!(
            red >= 3,
            "the authored stroke must draw in #FF2040 (found {red} cells)"
        );
    }

    /// The catalog-name route still resolves (`rust` is a registry-named material),
    /// and neither stroke falls back to the contour ink.
    #[test]
    fn catalog_line_material_still_resolves_by_name() {
        let level = line_level("");
        let map = level.build_map();
        let registry = crate::registry_for_map(&map);
        let rust = resolve_line_color("rust", 0, &registry);
        assert_ne!(rust, contour::CONTOUR_FG, "catalog `rust` must resolve");
        let fb = render_with_lines(&level);
        assert!(
            stroke_cells_in(&fb, rust) >= 3,
            "the catalog stroke draws in rust"
        );
    }

    /// A material bound to a SURFACE (not only line-referenced) also tints a stroke:
    /// resolution goes through the authored name, whatever else the material is for.
    #[test]
    fn surface_bound_material_tints_a_line_too() {
        let level = line_level(
            "[material mix_wall]\nfg = #30C0FF\nbg = #04121A\nsurface = wall_solid\n\
             [line on_floor]\npath = 5,3,1.0; 5,7,1.0\nmaterial = mix_wall\n",
        );
        let fb = render_with_lines(&level);
        assert!(
            stroke_cells_in(&fb, 0x0030_C0FF) >= 3,
            "surface-bound material tints"
        );
    }

    /// A floor cable running STRAIGHT AWAY from the camera, starting behind it (the
    /// player stands on the cable, as a `cable_run` under the walkway). It used to
    /// draw nothing: the behind-camera segment was dropped, and on the part in front
    /// the screen-linear depth ran so deep that the floor it lies on hid it. It must
    /// draw a run of stroke cells below the horizon, down to the bottom of the frame.
    #[test]
    fn an_end_on_floor_cable_from_behind_the_camera_draws() {
        let level = line_level(
            "[material cable]\nfg = #FF00FF\nbg = #100010\n\
             [line under_foot]\npath = 0.5,5.5,1.02; 7.5,5.5,1.02\nmaterial = cable\n",
        );
        let fb = render_with_lines(&level);
        let cells: Vec<usize> = (0..fb.rows())
            .filter(|&r| (0..fb.cols()).any(|c| fb.get(c, r).unwrap().fg == 0x00FF_00FF))
            .collect();
        assert!(
            cells.len() >= 6,
            "an end-on floor cable drew on {} row(s)",
            cells.len()
        );
        assert!(
            cells.iter().all(|&r| r >= fb.rows() / 2),
            "a floor cable stays below the horizon"
        );
        assert!(
            cells.iter().any(|&r| r >= fb.rows() - 4),
            "the cable must reach the bottom of the frame (it passes under the camera)"
        );
    }

    /// A segment wholly behind the camera draws nothing, and the viewport clip keeps a
    /// near-plane cut (which projects thousands of rows off-screen) cheap and exact.
    #[test]
    fn near_clip_and_viewport_clip() {
        assert_eq!(
            clip_to_rect(-10.0, 5.0, 5.0, 0.0, 10.0, 10.0),
            None,
            "misses left"
        );
        let (t0, t1) = clip_to_rect(5.0, 5.0, 0.0, 1000.0, 10.0, 10.0).unwrap();
        assert!(
            t0 == 0.0 && (t1 * 1000.0 - 4.5).abs() < 1e-3,
            "clipped at the bottom row"
        );
        let cam = cam_at(5.0, 5.0, 1.5, 0.0);
        let basis = LineViewBasis::new(&cam, &Fov::default(), 100, 32);
        let (ahead, behind) = (Vec3::new(5.0, 9.0, 1.0), Vec3::new(5.0, 1.0, 1.0));
        assert!(basis.clip_near(behind, Vec3::new(5.0, 2.0, 1.0)).is_none());
        let (a, b) = basis.clip_near(behind, ahead).unwrap();
        assert!(
            (basis.perp(a) - LINE_NEAR).abs() < 1e-4 && b == ahead,
            "cut at the near plane"
        );
    }

    // ── Set-dressing F4: `[line]` z is WORLD z ───────────────────────────────────

    /// A stroke at WORLD floor height (z = 1.0) lands on cells the cast filled as
    /// FLOOR, below the horizon; a low table top at world 1.45 also stays below the
    /// horizon.
    #[test]
    fn world_z_floor_and_table_heights_draw_below_the_horizon() {
        let (cols, rows) = (100usize, 32usize);
        let map = room(9, 9, 1);
        let cam = cam_at(1.5, 4.5, STANDING_EYE_Z, std::f32::consts::FRAC_PI_2); // +x
        let fov = Fov::default();
        let mut fb = Framebuffer::new(cols, rows);
        crate::raycast_fov(&map, &cam, &fov, &mut fb);
        let basis = LineViewBasis::new(&cam, &fov, cols, rows);
        let horizon = rows as f32 / 2.0;

        let floor = basis.project(Vec3::new(4.5, 4.5, 1.0)).expect("ahead");
        let cell = fb
            .get(floor.col.round() as usize, floor.row.round() as usize)
            .unwrap();
        assert_eq!(
            cell.surface,
            re_core::SurfaceKind::Floor,
            "world z=1.0 is the floor"
        );
        assert!(floor.row > horizon + 1.0, "floor point below the horizon");
        assert!(
            basis.visible(&map, floor.col, floor.row, floor.depth),
            "on the floor ⇒ drawn"
        );

        let table = basis.project(Vec3::new(4.5, 4.5, 1.45)).expect("ahead");
        assert!(
            table.row > horizon - 0.5 && table.row < floor.row,
            "a table top at world 1.45 sits between the floor and the horizon (row {}, floor {}, horizon {horizon})",
            table.row,
            floor.row
        );
    }

    // ── One convention: lines and walls agree at every height ───────────────────
    use re_world::CellKind;

    /// An `sx`×`sy` room: a floor slab at z=0 and perimeter walls `height` cells tall
    /// (z = 1..=height, i.e. world `[1, 1+height]`).
    fn room(sx: usize, sy: usize, height: usize) -> Map {
        let mut m = Map::new(sx, sy, height + 2);
        for x in 0..sx {
            for y in 0..sy {
                if x == 0 || y == 0 || x == sx - 1 || y == sy - 1 {
                    for z in 1..=height {
                        m.set(x, y, z, CellKind::WallSolid);
                    }
                } else {
                    m.set(x, y, 0, CellKind::Floor);
                }
            }
        }
        m
    }

    /// A line at a wall's TOP and BASE edge lands in the row that edge passes through,
    /// and the cast drew that wall starting / ending exactly at the same edge — on a
    /// wall four cells tall, so three of its planes are off the eye slice. This is
    /// the one-convention contract: cell `(x, y, k)` spans world `[k, k+1]` for the
    /// walls on every plane and for the `[line]`s alike.
    #[test]
    fn a_line_on_a_wall_edge_lands_in_the_row_that_edge_crosses() {
        let (cols, rows) = (100usize, 60usize);
        let map = room(7, 12, 4); // far +y wall: cells y=11, face at world y=11.0
        let cam = cam_at(3.5, 1.5, STANDING_EYE_Z, 0.0); // looking +y
        let fov = Fov::default();
        let mut fb = Framebuffer::new(cols, rows);
        crate::raycast_fov(&map, &cam, &fov, &mut fb);
        let basis = LineViewBasis::new(&cam, &fov, cols, rows);

        let top = basis.project(Vec3::new(3.5, 11.0, 5.0)).expect("ahead");
        let base = basis.project(Vec3::new(3.5, 11.0, 1.0)).expect("ahead");
        let col = top.col.round() as usize;
        let wall_rows: Vec<usize> = (0..rows)
            .filter(|&r| fb.get(col, r).unwrap().surface == re_core::SurfaceKind::WallSolid)
            .collect();
        let (first, last) = (wall_rows[0], *wall_rows.last().unwrap());
        assert_eq!(
            last + 1 - first,
            wall_rows.len(),
            "one contiguous wall run {wall_rows:?}"
        );
        assert!(
            first > 0 && last + 1 < rows,
            "the whole 4-cell wall is on screen"
        );
        // The cast's drawn rows are exactly the rows whose centres lie on the wall
        // between the two projected edges (cell-centre coordinates ⇒ `ceil`).
        assert_eq!(
            first,
            top.row.ceil() as usize,
            "wall top row vs the top-edge projection"
        );
        assert_eq!(
            last + 1,
            base.row.ceil() as usize,
            "wall end row vs the base-edge projection"
        );
        // The lines sit in the rows the edges cross: at most one row outside the wall.
        let (top_row, base_row) = (top.row.round() as usize, base.row.round() as usize);
        assert!(
            top_row + 1 >= first && top_row <= first,
            "top line row {top_row}, wall from {first}"
        );
        assert!(
            base_row >= last && base_row <= last + 1,
            "base line row {base_row}, wall to {last}"
        );
        // Both edges are ON the wall, so neither is occluded by it.
        assert!(basis.visible(&map, top.col, top.row, top.depth));
        assert!(basis.visible(&map, base.col, base.row, base.depth));
    }

    /// Occlusion reads the GRID along the stroke point's own ray: a stroke hanging at
    /// z = 3.5 behind the upper courses of a wall is hidden, the same stroke in front
    /// of it or painted on its face draws. (The framebuffer's backdrop depths could not
    /// decide this — the rows above the eye slice there are a stylised ceiling.)
    #[test]
    fn a_line_behind_an_upper_course_is_hidden_and_one_in_front_draws() {
        let (cols, rows) = (100usize, 60usize);
        let mut map = room(7, 16, 5);
        for x in 1..6 {
            for z in 1..=4 {
                map.set(x, 8, z, CellKind::WallSolid); // a partition 4 cells tall at y=8
            }
        }
        let cam = cam_at(3.5, 1.5, STANDING_EYE_Z, 0.0); // looking +y
        let basis = LineViewBasis::new(&cam, &Fov::default(), cols, rows);
        let behind = basis.project(Vec3::new(3.5, 11.0, 3.5)).expect("ahead");
        let on_face = basis.project(Vec3::new(3.5, 8.0, 3.5)).expect("ahead");
        let in_front = basis.project(Vec3::new(3.5, 6.0, 3.5)).expect("ahead");
        assert!(
            !basis.visible(&map, behind.col, behind.row, behind.depth),
            "behind ⇒ hidden"
        );
        assert!(
            basis.visible(&map, on_face.col, on_face.row, on_face.depth),
            "on the face"
        );
        assert!(
            basis.visible(&map, in_front.col, in_front.row, in_front.depth),
            "in front"
        );
    }
}
