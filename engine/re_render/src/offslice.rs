//! `re_render::offslice` — the VERTICAL GEOMETRY pass: every z-plane, not just the
//! eye slice.
//!
//! ## Why this exists (the root cause it fixes)
//!
//! The caster is 2.5D: the primary ray walks ONE horizontal grid slice, `cell_z =
//! gameplay_eye_ref_z(eye).floor()` (plane 1 for a player grounded on a normal floor,
//! see [`crate::gameplay_eye_ref_z`]), and paints ONE cell-height wall band per column.
//! Everything authored on any other plane used to be reached only through a narrow
//! special case — a scan that looked a few planes UP for a hand-maintained allowlist of
//! "content" kinds and painted the single nearest one. The result, visible from any
//! standing eye:
//!
//! * a wall five cells tall rendered one cell tall;
//! * gallery decks, cable trays, hanging panels, pendant lamps, a CCTV bracket — any
//!   geometry mounted two or more planes up — never rendered at all;
//! * the treads of a staircase above the first never rendered from below;
//! * `[line]` strokes (which project with true perspective) could not agree with walls
//!   that did not.
//!
//! ## What it does now
//!
//! Per screen column, after the primary slice and the floor/ceiling backdrop fill, this
//! pass walks the column's ray through the grid FRONT TO BACK and, at every grid column
//! it crosses, resolves every z-plane inside the view frustum (pitch included). Each
//! solid cell is intersected EXACTLY with the ray's vertical plane: that cut is a convex
//! polygon in `(s, z)` — `s` the perpendicular ray parameter, `z` world height — a
//! rectangle for a `Full` cell, a trapezoid for a 45° ramp, a clipped rectangle for a
//! wedge. Each screen row is a line `z = eye + k·s` through that plane, so the visible
//! surface for a row is simply where that line first enters a polygon: the near
//! vertical face, a top face (looking down), a bottom face (looking up) or a ramp's
//! slope. Rows are painted once, nearest first, with a per-column coverage mask.
//!
//! The projection is the one the primary slice uses:
//!
//! ```text
//!   row(h, s) = horizon + proj · (eye − h) / s
//! ```
//!
//! with `eye = cell_z + 0.5 + vshift_per_perp / proj` — so the eye-slice band, every
//! other plane, the ramps and the authored `[line]`s are one perspective, and a grid
//! cell `(x, y, k)` spans world `z ∈ [k, k+1]` on every one of them.
//!
//! ## Who owns what
//!
//! * The PRIMARY slice keeps plane `cell_z` for every non-ramp solid (mirrors, portals,
//!   image cut-outs, wedges and plain walls all have per-hit special paths there). A
//!   ray reaching such a cell here is covered but not painted ([`Role::Slice`]).
//! * RAMPS (`Shape::RampZ`) are this pass's on EVERY plane, the eye plane included.
//!   The primary DDA passes through them (`dda_hit`), so a ramp is one exact polygon —
//!   the old per-column silhouette + backdrop re-cast (which measured the backdrop from
//!   a nudged origin and drew it floor-to-sky) is gone.
//! * FLOOR / CEILING slabs are the storey lids. Their horizontal faces COVER rows without
//!   painting them — the backdrop floor/ceiling fill already shaded those rows — so a
//!   slab hides the storey beyond it without changing how a floor or ceiling looks. A
//!   slab's vertical EDGE (the rim of a stairwell or an atrium) is real geometry and is
//!   painted.
//! * HOLOGRAM cells are transparent here; `crate::holo` draws them.
//!
//! Pure grid reads and float maths, no allocation, no terminal.

use crate::{floori, PERP_MIN};
use re_core::{shape_surface, RampDir, Shape};
use re_world::{CellKind, Map};

/// Max grid steps for one column walk — the same bound class as the primary DDA.
pub(crate) const MAX_STEPS: i32 = 256;

/// Geometric slack (world cells) for the polygon edge tests.
const EPS: f32 = 1e-4;

// ───────────────────────────── projection ─────────────────────────────────────

/// The vertical projection of one cast column. `row(h, s)` is the (continuous) screen
/// row of world height `h` at perpendicular distance `s`; `0.0` is the TOP edge of row
/// 0, so row `r` covers `[r, r+1)` and its centre is `r + 0.5`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct VProj {
    /// Pitch-shifted horizon row.
    pub(crate) horizon: f32,
    /// `(rows/2)/tan(vfov/2)` — the vertical projection-plane distance, in rows.
    pub(crate) proj: f32,
    /// World z of the eye in the caster's model: `cell_z + 0.5 + vshift/proj`.
    pub(crate) eye: f32,
}

impl VProj {
    /// The projection for a column cast on slice `cell_z` with the given horizon, plane
    /// distance and vertical eye shear (see [`crate::vshift_per_perp`]).
    #[inline]
    pub(crate) fn new(cell_z: i32, horizon: i32, proj: f32, vshift_per_perp: f32) -> Self {
        let proj = proj.max(1e-6);
        VProj {
            horizon: horizon as f32,
            proj,
            eye: cell_z as f32 + 0.5 + vshift_per_perp / proj,
        }
    }

    /// Continuous screen row of world height `h` at perpendicular distance `s`.
    #[inline]
    pub(crate) fn row(&self, h: f32, s: f32) -> f32 {
        self.horizon + self.proj * (self.eye - h) / s.max(PERP_MIN)
    }

    /// World-z rise per unit of perpendicular distance of the view ray through the
    /// screen-row coordinate `row` (a row CENTRE is `r + 0.5`).
    #[inline]
    pub(crate) fn slope(&self, row: f32) -> f32 {
        (self.horizon - row) / self.proj
    }
}

/// The integer rows `[first, end)` whose CENTRES lie inside the continuous span
/// `[top, bottom)` — the one rasterisation rule every vertical span in the caster uses
/// (primary wall band, off-slice faces, image panes), so abutting spans tile exactly
/// and a `[line]` placed at a span edge lands on the row that edge crosses.
#[inline]
pub(crate) fn span_rows(top: f32, bottom: f32) -> (i32, i32) {
    let first = (top - 0.5).ceil();
    let end = (bottom - 0.5).ceil();
    // `as` saturates, so an effectively infinite span clamps instead of wrapping.
    (first as i32, end as i32)
}

/// Texture coordinate of the continuous screen row `y` on a face DRAWN over rows
/// `[first, end)` (`n` rows): `(y − ½ − first) / n`, so the row whose centre is `r + ½`
/// samples at `(r − first)/n` — its TOP edge, the caster's historical texture
/// convention (`v` runs `0, 1/n, …, (n−1)/n` down the drawn rows, and a face centred on
/// the horizon keeps `v = ½` on the horizon row at every depth).
///
/// Mapping the drawn rows, not the continuous projected span, keeps a face's texture
/// fixed while its sub-row extent drifts with depth (turning in place changes every
/// column's perpendicular depth a little): the sampled texel only moves when a row
/// actually enters or leaves the face, which is what keeps a turn inside the shimmer
/// budget. The drawn rows are exactly where the projection puts the face, so its edges
/// still agree with every other surface and with the `[line]`s.
#[inline]
pub(crate) fn drawn_v(first: i32, end: i32, y: f32) -> f32 {
    let n = end.saturating_sub(first).max(1) as f32;
    ((y - 0.5 - first as f32) / n).clamp(0.0, 1.0)
}

// ───────────────────────────── cell roles ─────────────────────────────────────

/// What the vertical pass does with one cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Role {
    /// Air, or a transparent hologram voxel: the ray passes through.
    Air,
    /// Solid geometry this pass shades.
    Paint,
    /// A storey slab (`Floor` / `Ceiling`): horizontal faces cover without painting (the
    /// backdrop fill owns the floor/ceiling look), the vertical rim is painted.
    Slab,
    /// A non-ramp solid on the primary slice plane: the primary cast owns its shading;
    /// here it only stops the ray.
    Slice,
}

/// Classify the cell at `(x, y, z)` for a column cast on slice `cell_z`.
///
/// **Exhaustive over [`CellKind`], no wildcard** — a new kind is a compile error here
/// until its vertical role is decided, so nothing can silently vanish above the eye
/// again (the failure mode of the old allowlist).
#[inline]
pub(crate) fn role_of(map: &Map, x: i32, y: i32, z: i32, cell_z: i32) -> Role {
    let kind = map.cell_at(x, y, z);
    match kind {
        CellKind::Empty | CellKind::Hologram => Role::Air,
        // NODE SPRITE: a FREE-STANDING puzzle node stays collidable but is
        // render-transparent off the primary slice too — the vertical pass must
        // not paint its underside/dome or contour-frame it (`crate::is_sprite_node`).
        // It is drawn separately, after the cast, by `node_sprite::apply`. An
        // EMBEDDED node (`crate::node_sprite::is_embedded`) is deliberately
        // EXCLUDED here (falls through to the ordinary `Object` arm below) — making
        // it air would expose a cavity in the solid/pipe/slab it's set into.
        CellKind::Object
            if crate::is_sprite_node(map, x, y, z, kind, map.side_data_at(x, y, z)) =>
        {
            Role::Air
        }
        CellKind::Floor | CellKind::Ceiling => Role::Slab,
        CellKind::WallSolid
        | CellKind::WallTrace
        | CellKind::WallMatrix
        | CellKind::Object
        | CellKind::Portal => {
            if z == cell_z && !matches!(map.shape_at(x, y, z), Shape::RampZ(_)) {
                Role::Slice
            } else {
                Role::Paint
            }
        }
    }
}

// ───────────────────────────── the grid walk ──────────────────────────────────

/// One grid column the ray crosses: `(x, y)` and the perpendicular parameter range
/// `[t_in, t_out]` it spends inside it. `side` is the axis of the face it ENTERED
/// through (`1` = x-const, `0` = y-const); `start` marks the camera's own column,
/// which has no near face.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Step {
    pub(crate) x: i32,
    pub(crate) y: i32,
    pub(crate) t_in: f32,
    pub(crate) t_out: f32,
    pub(crate) side: u8,
    pub(crate) start: bool,
}

/// The front-to-back grid walk of one ray, as an iterator of [`Step`]s — the same
/// continuous DDA the primary cast uses, so `t_in`/`t_out` are apples-to-apples with
/// [`crate::RayHit::perp`]. Stops when the ray leaves the grid's XY footprint or after
/// `max_steps`.
pub(crate) struct GridWalk {
    map_x: i32,
    map_y: i32,
    step_x: i32,
    step_y: i32,
    side_dist_x: f32,
    side_dist_y: f32,
    delta_x: f32,
    delta_y: f32,
    t: f32,
    side: u8,
    first: bool,
    left: i32,
    sx: i32,
    sy: i32,
}

impl GridWalk {
    /// Start a walk from world origin `(ox, oy)` along direction `(dx, dy)`,
    /// bounded to `map`'s footprint and at most `max_steps` grid crossings.
    pub(crate) fn new(map: &Map, ox: f32, oy: f32, dx: f32, dy: f32, max_steps: i32) -> Self {
        let map_x = floori(ox);
        let map_y = floori(oy);
        let delta_x = if dx == 0.0 { 1e30 } else { (1.0 / dx).abs() };
        let delta_y = if dy == 0.0 { 1e30 } else { (1.0 / dy).abs() };
        let (step_x, side_dist_x) = if dx < 0.0 {
            (-1, (ox - map_x as f32) * delta_x)
        } else {
            (1, (map_x as f32 + 1.0 - ox) * delta_x)
        };
        let (step_y, side_dist_y) = if dy < 0.0 {
            (-1, (oy - map_y as f32) * delta_y)
        } else {
            (1, (map_y as f32 + 1.0 - oy) * delta_y)
        };
        let (sx, sy, _) = map.size();
        GridWalk {
            map_x,
            map_y,
            step_x,
            step_y,
            side_dist_x,
            side_dist_y,
            delta_x,
            delta_y,
            t: 0.0,
            side: 0,
            first: true,
            left: max_steps,
            sx: sx as i32,
            sy: sy as i32,
        }
    }
}

impl Iterator for GridWalk {
    type Item = Step;

    fn next(&mut self) -> Option<Step> {
        if self.left <= 0 {
            return None;
        }
        self.left -= 1;
        let start = self.first;
        if !start {
            // Step into the next column across the nearer boundary.
            if self.side_dist_x < self.side_dist_y {
                self.t = self.side_dist_x;
                self.side_dist_x += self.delta_x;
                self.map_x += self.step_x;
                self.side = 1;
            } else {
                self.t = self.side_dist_y;
                self.side_dist_y += self.delta_y;
                self.map_y += self.step_y;
                self.side = 0;
            }
        }
        self.first = false;
        if self.map_x < 0 || self.map_y < 0 || self.map_x >= self.sx || self.map_y >= self.sy {
            return None;
        }
        Some(Step {
            x: self.map_x,
            y: self.map_y,
            t_in: self.t,
            t_out: self.side_dist_x.min(self.side_dist_y),
            side: self.side,
            start,
        })
    }
}

// ───────────────────────────── cell polygons ──────────────────────────────────

/// The near (entry) face of a cell polygon.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Near {
    /// No visible near face (the camera's own column, or a ramp's zero-height foot).
    None,
    /// An axis-aligned face on the stepped axis ([`Step::side`]).
    Axis,
    /// A wedge's 45° diagonal, with its outward (air-facing) world normal.
    Diag(glam::Vec3),
}

/// The ray's vertical-plane cut through one cell on one plane: solid for
/// `s ∈ [sa, sb]`, `z` from the cell base up to a top line running from height `h0`
/// (at `sa`) to `h1` (at `sb`), both relative to the base. Convex by construction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Poly {
    pub(crate) sa: f32,
    pub(crate) sb: f32,
    pub(crate) h0: f32,
    pub(crate) h1: f32,
    pub(crate) near: Near,
}

/// Rise of a ramp (`0` at the foot edge, `1` at the head edge) at ray parameter `s`.
#[inline]
fn ramp_rise(dir: RampDir, cx: i32, cy: i32, ox: f32, oy: f32, dx: f32, dy: f32, s: f32) -> f32 {
    let lx = ox + s * dx - cx as f32;
    let ly = oy + s * dy - cy as f32;
    match dir {
        RampDir::PosX => lx,
        RampDir::NegX => 1.0 - lx,
        RampDir::PosY => ly,
        RampDir::NegY => 1.0 - ly,
    }
}

/// Build the polygon of a `shape` cell at grid `(cx, cy)` for the ray crossing the
/// column over `[step.t_in, step.t_out]`. `None` when the ray misses the solid part of
/// the footprint entirely (a wedge's open triangle).
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn cell_poly(
    shape: Shape,
    step: &Step,
    ox: f32,
    oy: f32,
    dx: f32,
    dy: f32,
) -> Option<Poly> {
    let (t_in, t_out) = (step.t_in, step.t_out);
    let axis_near = if step.start { Near::None } else { Near::Axis };
    match shape {
        Shape::Full => Some(Poly {
            sa: t_in,
            sb: t_out,
            h0: 1.0,
            h1: 1.0,
            near: axis_near,
        }),
        Shape::RampZ(dir) => {
            let h0 = ramp_rise(dir, step.x, step.y, ox, oy, dx, dy, t_in).clamp(0.0, 1.0);
            let h1 = ramp_rise(dir, step.x, step.y, ox, oy, dx, dy, t_out).clamp(0.0, 1.0);
            let near = if h0 > EPS { axis_near } else { Near::None };
            Some(Poly {
                sa: t_in,
                sb: t_out,
                h0,
                h1,
                near,
            })
        }
        Shape::WedgeXY(_) => {
            // Solid half-plane: (P − point)·n ≤ 0 with `n` pointing into the air. Along
            // the ray that is a linear function f(s) = f0 + s·fd.
            let plane = shape_surface(shape)?;
            let n = plane.normal;
            let f0 = (ox - step.x as f32 - plane.point.x) * n.x
                + (oy - step.y as f32 - plane.point.y) * n.y;
            let fd = dx * n.x + dy * n.y;
            let f_in = f0 + t_in * fd;
            let f_out = f0 + t_out * fd;
            let s_d = if fd.abs() > 1e-12 {
                -f0 / fd
            } else {
                f32::INFINITY
            };
            match (f_in <= 0.0, f_out <= 0.0) {
                (true, true) => Some(Poly {
                    sa: t_in,
                    sb: t_out,
                    h0: 1.0,
                    h1: 1.0,
                    near: axis_near,
                }),
                (true, false) => Some(Poly {
                    sa: t_in,
                    sb: s_d.clamp(t_in, t_out),
                    h0: 1.0,
                    h1: 1.0,
                    near: axis_near,
                }),
                (false, true) => Some(Poly {
                    sa: s_d.clamp(t_in, t_out),
                    sb: t_out,
                    h0: 1.0,
                    h1: 1.0,
                    near: Near::Diag(n),
                }),
                (false, false) => None,
            }
        }
    }
}

/// Which surface of a polygon a view ray entered through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Face {
    /// The near vertical face (axis-aligned or a wedge diagonal).
    Near,
    /// The top line: a flat top (`Full` / wedge) or a ramp's slope.
    Top,
    /// The flat underside.
    Bottom,
}

/// Where the view ray `z = eye + k·s` first enters `poly` (base at world `base`):
/// `(s, z, face)`, or `None` if it passes by.
#[inline]
pub(crate) fn enter(poly: &Poly, base: f32, eye: f32, k: f32) -> Option<(f32, f32, Face)> {
    let Poly {
        sa,
        sb,
        h0,
        h1,
        near,
    } = *poly;
    let za = eye + k * sa;
    let top_a = base + h0;
    if za >= base - EPS && za <= top_a + EPS {
        return match near {
            Near::None => None, // inside the band with no face: the camera's own column.
            _ => Some((sa, za.clamp(base, top_a), Face::Near)),
        };
    }
    if za > top_a {
        // Above the near top edge: meet the top line `base + h0 + m·(s − sa)`.
        let len = sb - sa;
        let m = if len > EPS { (h1 - h0) / len } else { 0.0 };
        let den = k - m;
        if den.abs() < 1e-9 {
            return None;
        }
        let s = (top_a - m * sa - eye) / den;
        if s >= sa - EPS && s <= sb + EPS {
            let s = s.clamp(sa, sb);
            return Some((s, eye + k * s, Face::Top));
        }
        return None;
    }
    // Below the base: meet the underside.
    if k <= 0.0 {
        return None;
    }
    let s = (base - eye) / k;
    if s >= sa - EPS && s <= sb + EPS {
        let s = s.clamp(sa, sb);
        return Some((s, base, Face::Bottom));
    }
    None
}

/// Screen-row bounds `(min, max)` (continuous) of `poly` on plane `base`.
#[inline]
pub(crate) fn poly_row_bounds(poly: &Poly, base: f32, vp: &VProj) -> (f32, f32) {
    let r = [
        vp.row(base, poly.sa),
        vp.row(base + poly.h0, poly.sa),
        vp.row(base + poly.h1, poly.sb),
        vp.row(base, poly.sb),
    ];
    let lo = r.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = r.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (lo, hi)
}

/// One resolved vertical hit: which cell, which face, where.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct VHit {
    pub(crate) s: f32,
    pub(crate) z: f32,
    pub(crate) face: Face,
    pub(crate) x: i32,
    pub(crate) y: i32,
    pub(crate) plane: i32,
    pub(crate) role: Role,
    pub(crate) poly: Poly,
    pub(crate) shape: Shape,
    /// The axis the ray entered the grid column through ([`Step::side`]) — the normal
    /// axis of an [`Near::Axis`] face.
    pub(crate) side: u8,
}

/// Resolve the nearest surface the view ray of slope `k` meets inside the grid column
/// of `step`, walking the planes in the order the ray crosses them. `opaque` is asked
/// about each candidate before it is accepted; returning `false` lets the ray pass
/// through (an image pane's transparent texel).
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn resolve_in_column(
    map: &Map,
    cell_z: i32,
    step: &Step,
    ox: f32,
    oy: f32,
    dx: f32,
    dy: f32,
    eye: f32,
    k: f32,
    mut opaque: impl FnMut(&VHit) -> bool,
) -> Option<VHit> {
    let sz = map.size().2 as i32;
    let za = eye + k * step.t_in;
    let zb = eye + k * step.t_out;
    let (lo, hi) = if za <= zb { (za, zb) } else { (zb, za) };
    let p_lo = (lo.floor() as i32).max(0);
    let p_hi = (hi.floor() as i32).min(sz - 1);
    if p_lo > p_hi {
        return None;
    }
    let ascending = k >= 0.0;
    let count = p_hi - p_lo + 1;
    for i in 0..count {
        let p = if ascending { p_lo + i } else { p_hi - i };
        let role = role_of(map, step.x, step.y, p, cell_z);
        if role == Role::Air {
            continue;
        }
        let shape = map.shape_at(step.x, step.y, p);
        let Some(poly) = cell_poly(shape, step, ox, oy, dx, dy) else {
            continue;
        };
        // @todo off-slice Object prop smear at grazing angles — needs a
        // screen-space footprint bound. A prior fix here (`narrow_prop_footprint`)
        // was reverted: it narrowed along the RAY parameter `s`, not across the
        // screen, so it changed no on-screen width and regressed several levels'
        // golden frames for nothing.
        let base = p as f32;
        if let Some((s, z, face)) = enter(&poly, base, eye, k) {
            let hit = VHit {
                s,
                z,
                face,
                x: step.x,
                y: step.y,
                plane: p,
                role,
                poly,
                shape,
                side: step.side,
            };
            if opaque(&hit) {
                return Some(hit);
            }
        }
    }
    None
}

/// The inclusive plane range the frustum rows `[0, rows)` can see inside `step`.
#[inline]
pub(crate) fn frustum_planes(step: &Step, vp: &VProj, rows: i32, sz: i32) -> (i32, i32) {
    let k_top = vp.slope(0.0);
    let k_bot = vp.slope(rows as f32);
    let z_hi = vp.eye + (k_top * step.t_in).max(k_top * step.t_out);
    let z_lo = vp.eye + (k_bot * step.t_in).min(k_bot * step.t_out);
    (
        (z_lo.floor() as i32).max(0),
        (z_hi.floor() as i32).min(sz - 1),
    )
}

/// Full single-row query: walk the ray from the camera and return the nearest vertical
/// hit for the view ray of slope `k`, or `None`. The sub-cell silhouette refinement
/// uses this so a refined edge agrees with the column pass.
#[allow(clippy::too_many_arguments)]
pub(crate) fn query_row(
    map: &Map,
    cell_z: i32,
    ox: f32,
    oy: f32,
    dx: f32,
    dy: f32,
    eye: f32,
    k: f32,
    max_steps: i32,
    mut opaque: impl FnMut(&VHit) -> bool,
) -> Option<VHit> {
    for step in GridWalk::new(map, ox, oy, dx, dy, max_steps) {
        if let Some(h) = resolve_in_column(map, cell_z, &step, ox, oy, dx, dy, eye, k, &mut opaque)
        {
            return Some(h);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use re_core::DiagXY;
    use re_world::{stamp_geometry, GeometrySpec};

    /// A campaign-shaped room: floor slab z=0, habitable z=1..sz-2, ceiling slab on top.
    fn room(sx: usize, sy: usize, sz: usize) -> Map {
        stamp_geometry(&GeometrySpec {
            size: (sx, sy, sz),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[],
        })
    }

    const VP: VProj = VProj {
        horizon: 20.0,
        proj: 20.0,
        eye: 1.5,
    };

    #[test]
    fn projection_puts_a_cell_band_between_its_world_edges() {
        // A plane-k cell at distance 4 spans row(k+1) .. row(k); the eye sits at 1.5.
        assert!(
            (VP.row(1.5, 4.0) - 20.0).abs() < 1e-5,
            "eye height is the horizon"
        );
        assert!((VP.row(2.0, 4.0) - 17.5).abs() < 1e-5);
        assert!((VP.row(1.0, 4.0) - 22.5).abs() < 1e-5);
        // Five storeys up is five bands up — no clipping to "one band at eye level".
        assert!((VP.row(6.0, 4.0) - (20.0 - 4.5 * 5.0)).abs() < 1e-4);
        // slope() inverts row(): the ray through row(h, s) is at height h at s.
        let r = VP.row(3.25, 7.0);
        assert!((VP.eye + VP.slope(r) * 7.0 - 3.25).abs() < 1e-4);
    }

    #[test]
    fn span_rows_samples_row_centres() {
        assert_eq!(span_rows(10.3, 12.7), (10, 13)); // centres 10.5, 11.5, 12.5 inside
        assert_eq!(span_rows(10.6, 12.4), (11, 12)); // 10.5 above, 12.5 below
        assert_eq!(span_rows(0.0, 0.4), (0, 0)); // no centre inside ⇒ empty
                                                 // Abutting spans tile with no gap and no overlap.
        let (_, a_end) = span_rows(3.3, 7.8);
        let (b_first, _) = span_rows(7.8, 9.9);
        assert_eq!(a_end, b_first);
    }

    #[test]
    fn walk_reports_entry_and_exit_parameters() {
        let map = room(8, 8, 5);
        let steps: Vec<Step> = GridWalk::new(&map, 4.5, 6.5, 0.0, -1.0, 64).collect();
        assert!(steps[0].start && steps[0].t_in == 0.0 && (steps[0].t_out - 0.5).abs() < 1e-5);
        assert_eq!((steps[1].x, steps[1].y), (4, 5));
        assert!((steps[1].t_in - 0.5).abs() < 1e-5 && (steps[1].t_out - 1.5).abs() < 1e-5);
        assert_eq!(steps.last().map(|s| s.y), Some(0), "stops at the grid edge");
    }

    #[test]
    fn a_tall_wall_is_seen_on_every_plane_it_occupies() {
        // A 4-high free-standing wall at y=2 (near face y=3), camera at y=6.5 looking
        // -y. Rows aimed at z = 2.5 .. 4.5 all hit its near face at s = 3.5 on their own\n        // plane.
        let mut map = room(8, 8, 7);
        for z in 1..=4 {
            map.set_cell(4, 2, z, CellKind::WallSolid, 0);
        }
        for z in [2.5f32, 3.5, 4.5] {
            let k = (z - 1.5) / 3.5;
            let h = query_row(&map, 1, 4.5, 6.5, 0.0, -1.0, 1.5, k, 64, |_| true).expect("hit");
            assert_eq!((h.x, h.y, h.plane), (4, 2, z.floor() as i32));
            assert_eq!(h.face, Face::Near);
            assert!((h.s - 3.5).abs() < 1e-4);
            assert_eq!(h.role, Role::Paint);
        }
        // The eye-plane band belongs to the primary slice: covered, not painted here.
        let h = query_row(&map, 1, 4.5, 6.5, 0.0, -1.0, 1.5, 0.0, 64, |_| true).expect("hit");
        assert_eq!((h.plane, h.role), (1, Role::Slice));
    }

    #[test]
    fn a_deck_overhead_shows_its_underside_and_hides_what_is_above() {
        // A deck slab of solid cells at z=3 spanning y=2..=4; a node above it at z=4.
        let mut map = room(8, 10, 7);
        for y in 2..=4 {
            map.set_cell(4, y, 3, CellKind::WallSolid, 0);
        }
        map.set_cell(4, 3, 4, CellKind::Object, 1);
        // Looking up steeply from under the deck's near edge: the ray meets z=3 inside
        // the deck's footprint ⇒ the underside.
        let k = (3.0 - 1.5) / 3.0; // reaches z=3 at s=3 ⇒ y = 7.5-3 = 4.5 → cell y=4
        let h = query_row(&map, 1, 4.5, 7.5, 0.0, -1.0, 1.5, k, 64, |_| true).expect("hit");
        assert_eq!((h.y, h.plane, h.face), (4, 3, Face::Bottom));
        assert!((h.z - 3.0).abs() < 1e-4);
        // The node above the deck is never reached from below.
        for i in 0..40 {
            let k = 0.1 + i as f32 * 0.05;
            if let Some(h) = query_row(&map, 1, 4.5, 7.5, 0.0, -1.0, 1.5, k, 64, |_| true) {
                assert_ne!(h.plane, 4, "the deck must occlude the node (k={k})");
            }
        }
    }

    #[test]
    fn a_top_face_is_seen_from_above() {
        // Eye upstairs at 5.5, a block at z=2 below a hole: looking down hits its top.
        let mut map = room(8, 10, 9);
        map.set_cell(4, 3, 2, CellKind::WallSolid, 0);
        let k = (3.0 - 5.5) / 4.0; // z=3 (the block top) at s = 4 ⇒ y = 7.5-4 = 3.5
        let h = query_row(&map, 5, 4.5, 7.5, 0.0, -1.0, 5.5, k, 64, |_| true).expect("hit");
        assert_eq!((h.y, h.plane, h.face), (3, 2, Face::Top));
    }

    #[test]
    fn slabs_cover_their_faces_and_stop_the_ray() {
        let mut map = room(8, 10, 9);
        for y in 0..10 {
            for x in 0..8 {
                map.set(x, y, 4, CellKind::Floor); // storey slab
            }
        }
        map.set_cell(4, 3, 5, CellKind::Object, 2); // a node on the floor above
        let k = (5.5 - 1.5) / 4.0;
        let h = query_row(&map, 1, 4.5, 7.5, 0.0, -1.0, 1.5, k, 64, |_| true).expect("hit");
        assert_eq!((h.plane, h.role, h.face), (4, Role::Slab, Face::Bottom));
    }

    #[test]
    fn a_ramp_is_a_trapezoid_not_a_column() {
        // RampZ(PosY) at (4,3,1): rises toward +y, i.e. its HEAD faces the camera at
        // y=7.5 looking -y. The near face is full height; rays above it pass over the
        // descending slope and on to whatever is behind — never a floor-to-sky column.
        let mut map = room(8, 10, 5);
        map.set_cell(4, 3, 1, CellKind::WallSolid, 0);
        map.set_shape(4, 3, 1, Shape::RampZ(RampDir::PosY));
        let step = Step {
            x: 4,
            y: 3,
            t_in: 3.5,
            t_out: 4.5,
            side: 0,
            start: false,
        };
        let poly = cell_poly(Shape::RampZ(RampDir::PosY), &step, 4.5, 7.5, 0.0, -1.0).unwrap();
        assert!(
            (poly.h0 - 1.0).abs() < 1e-5 && poly.h1.abs() < 1e-5,
            "{poly:?}"
        );
        // A ray aimed at z=1.5 at s=3.5 enters the near face.
        let (s, _, f) = enter(&poly, 1.0, 1.5, 0.0).unwrap();
        assert_eq!(f, Face::Near);
        assert!((s - 3.5).abs() < 1e-5);
        // A ray aimed at z=2.2 at s=3.5 is above the head and rising: it misses.
        assert!(enter(&poly, 1.0, 1.5, 0.7 / 3.5).is_none());
        // From the FOOT side the ray lands on the slope, and the hit rises with s.
        // NegY rises toward -y: seen from y=5 looking -y the FOOT (y=4) faces us.
        let foot = Step {
            x: 4,
            y: 3,
            t_in: 1.0,
            t_out: 2.0,
            side: 0,
            start: false,
        };
        let poly = cell_poly(Shape::RampZ(RampDir::NegY), &foot, 4.5, 5.0, 0.0, -1.0).unwrap();
        assert!(
            poly.h0.abs() < 1e-5 && (poly.h1 - 1.0).abs() < 1e-5,
            "{poly:?}"
        );
        assert_eq!(poly.near, Near::None, "a zero-height foot has no near face");
        // z = 1.5 - 0.2 s meets the slope z = 1 + (s - 1) at s = 1.25.
        let (s, z, f) = enter(&poly, 1.0, 1.5, -0.2).unwrap();
        assert_eq!(f, Face::Top);
        assert!(
            (s - 1.25).abs() < 1e-4 && (z - 1.25).abs() < 1e-4,
            "slope hit s={s} z={z}"
        );
    }

    #[test]
    fn a_wedge_is_cut_to_its_solid_half() {
        // WedgeXY(NE) — solid where local x+y ≥ 1. A +y ray along local x=0.25 crosses
        // air until local y = 0.75, then solid: the near face is the diagonal.
        let step = Step {
            x: 2,
            y: 2,
            t_in: 1.0,
            t_out: 2.0,
            side: 0,
            start: false,
        };
        let poly = cell_poly(Shape::WedgeXY(DiagXY::NE), &step, 2.25, 1.0, 0.0, 1.0).unwrap();
        assert!(
            (poly.sa - 1.75).abs() < 1e-4 && (poly.sb - 2.0).abs() < 1e-4,
            "{poly:?}"
        );
        assert!(matches!(poly.near, Near::Diag(_)));
        // Along local x=0.9 the air triangle is thinner: the diagonal at local y=0.1.
        let poly = cell_poly(Shape::WedgeXY(DiagXY::NE), &step, 2.9, 1.0, 0.0, 1.0).unwrap();
        assert!(
            (poly.sa - 1.1).abs() < 1e-4 && matches!(poly.near, Near::Diag(_)),
            "{poly:?}"
        );
        // Entering from the solid side (a -y ray through the north face at local
        // x=0.25, where x+y = 1.25) it is solid from the entry face on.
        let back = Step {
            x: 2,
            y: 2,
            t_in: 1.0,
            t_out: 2.0,
            side: 0,
            start: false,
        };
        let poly = cell_poly(Shape::WedgeXY(DiagXY::NE), &back, 2.25, 4.0, 0.0, -1.0).unwrap();
        assert!(
            (poly.sa - 1.0).abs() < 1e-4 && poly.near == Near::Axis,
            "{poly:?}"
        );
    }

    #[test]
    fn the_nearest_element_wins_regardless_of_plane() {
        // A solid Object (the door/inert tag — NOT a sprite-node tag, which is now
        // `Role::Air` here and drawn separately by `node_sprite`) at z=2 in front of
        // a taller wall: rows over the object see the object.
        let mut map = room(8, 10, 7);
        for z in 1..=5 {
            map.set_cell(4, 1, z, CellKind::WallSolid, 0);
        }
        map.set_cell(4, 4, 2, CellKind::Object, re_core::DOOR_OBJECT_TAG);
        let k = (2.5 - 1.5) / 3.0; // z=2.5 at the node's face (s=3)
        let h = query_row(&map, 1, 4.5, 7.5, 0.0, -1.0, 1.5, k, 64, |_| true).expect("hit");
        assert_eq!((h.y, h.plane), (4, 2));
        // A pass-through predicate (a transparent pane) exposes the wall behind.
        let h = query_row(&map, 1, 4.5, 7.5, 0.0, -1.0, 1.5, k, 64, |h| h.y != 4).expect("hit");
        assert_eq!(h.y, 1);
    }

    #[test]
    fn frustum_planes_grow_with_distance_and_follow_pitch() {
        let step = Step {
            x: 0,
            y: 0,
            t_in: 2.0,
            t_out: 3.0,
            side: 0,
            start: false,
        };
        let (lo, hi) = frustum_planes(&step, &VP, 40, 20);
        assert_eq!(lo, 0);
        assert!(
            hi >= 4,
            "at s=3 the top row sees z = 1.5 + 3 = 4.5 (hi={hi})"
        );
        // Pitching up (horizon lower on screen) raises the visible top.
        let up = VProj {
            horizon: 35.0,
            ..VP
        };
        assert!(frustum_planes(&step, &up, 40, 20).1 > hi);
    }

    #[test]
    fn role_table_is_exhaustive() {
        let mut map = room(8, 8, 5);
        // `Object`'s side_data is 0 here (`re_core::DOOR_OBJECT_TAG` for the door,
        // `INERT_OBJECT_TAG` for set-dressing) — a RESERVED tag, so `is_sprite_node`
        // is false and it still paints as solid wall-voxel geometry (the door/inert
        // case). The NODE case (tag < RESERVED_TAG_BASE) is Role::Air and is tested
        // separately below (`a_sprite_node_is_air_on_every_plane`).
        let object_side_data = re_core::DOOR_OBJECT_TAG;
        for kind in [
            CellKind::Empty,
            CellKind::WallSolid,
            CellKind::WallTrace,
            CellKind::WallMatrix,
            CellKind::Floor,
            CellKind::Ceiling,
            CellKind::Object,
            CellKind::Hologram,
            CellKind::Portal,
        ] {
            let sd = if kind == CellKind::Object {
                object_side_data
            } else {
                0
            };
            map.set_cell(4, 4, 2, kind, sd);
            let got = role_of(&map, 4, 4, 2, 1);
            let on_slice = {
                map.set_cell(4, 4, 1, kind, sd);
                role_of(&map, 4, 4, 1, 1)
            };
            // Exhaustive over `CellKind` — a new variant is a compile error HERE as well.
            let (want, want_slice) = match kind {
                CellKind::Empty | CellKind::Hologram => (Role::Air, Role::Air),
                CellKind::Floor | CellKind::Ceiling => (Role::Slab, Role::Slab),
                CellKind::WallSolid
                | CellKind::WallTrace
                | CellKind::WallMatrix
                | CellKind::Object
                | CellKind::Portal => (Role::Paint, Role::Slice),
            };
            assert_eq!((got, on_slice), (want, want_slice), "role for {kind:?}");
        }
        // A ramp on the slice plane is this pass's, not the primary's.
        map.set_cell(4, 4, 1, CellKind::WallSolid, 0);
        map.set_shape(4, 4, 1, Shape::RampZ(RampDir::PosX));
        assert_eq!(role_of(&map, 4, 4, 1, 1), Role::Paint);
    }

    /// NODE SPRITE: a puzzle-node `Object` (`side_data` = a node index,
    /// below `RESERVED_TAG_BASE`) is `Role::Air` on EVERY plane — on the primary
    /// slice too — so the vertical pass never paints its calm-fill dome/underside
    /// or contour-frames it. It is drawn separately by `node_sprite::apply`.
    #[test]
    fn a_sprite_node_is_air_on_every_plane() {
        let mut map = room(8, 8, 5);
        map.set_cell(4, 4, 2, CellKind::Object, 0); // node 0, off-slice
        assert_eq!(role_of(&map, 4, 4, 2, 1), Role::Air);
        map.set_cell(4, 4, 1, CellKind::Object, 3); // node 3, on the primary slice
        assert_eq!(role_of(&map, 4, 4, 1, 1), Role::Air);
        // The door/inert reserved tags keep painting as solid geometry.
        map.set_cell(4, 4, 1, CellKind::Object, re_core::DOOR_OBJECT_TAG);
        assert_eq!(role_of(&map, 4, 4, 1, 1), Role::Slice);
    }
}
