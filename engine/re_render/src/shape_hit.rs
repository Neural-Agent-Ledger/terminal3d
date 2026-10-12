//! Per-cell 45° geometry intersection for the DDA.
//!
//! The continuous-DDA in [`crate::dda_hit`] walks the grid on a fixed `cell_z`
//! plane. When it steps onto a *solid* cell whose [`re_core::Shape`] is not
//! [`Shape::Full`], it must decide what the ray actually sees inside that cell:
//!
//! * the cell's **diagonal / slope face** (a real 45° surface with a tilted
//!   normal), if the ray crosses into the solid half across that face;
//! * a remaining **axis-aligned solid face** of the cell (a wedge still has its
//!   two orthogonal back faces solid), if the ray entered the cell already on the
//!   solid side; or
//! * **nothing** — the ray passes through the open triangle / above the ramp and
//!   the DDA must continue to whatever is behind (the open part is see-through).
//!
//! [`Shape::Full`] never reaches this module — the DDA keeps its original
//! axis-aligned fast path, so an all-`Full` map renders byte-identically.
//!
//! All math is in world coordinates. The cell occupies
//! `[cx, cx+1] × [cy, cy+1]` in XY and `[cell_z, cell_z+1]` in Z. The ray is the
//! 2D horizontal ray `(ox, oy) + t·(dx, dy)` cast at the eye's z-plane (`z_eye`).

use glam::Vec3;
use re_core::{shape_solid_contains, shape_surface, RampDir, Shape};

/// A 45° surface hit inside one cell, in world coordinates.
pub(crate) struct ShapeHit {
    /// Perpendicular (camera-plane) distance the DDA reports: for this 2D cast the
    /// ray-parameter `t` already equals the perpendicular distance because the ray
    /// direction is the un-normalized `(dx, dy)` whose facing component is 1 (the
    /// same convention `dda_hit` uses for `side_dist`).
    pub(crate) t: f32,
    /// World hit point XY on the surface.
    pub(crate) hit_x: f32,
    pub(crate) hit_y: f32,
    /// World-space outward unit normal of the surface. For a wedge this is the
    /// vertical diagonal normal; for a ramp it is the tilted slope normal; for a
    /// remaining axis-aligned solid face it is the axis normal (so the shading path
    /// can treat it like an ordinary wall face).
    pub(crate) normal: Vec3,
    /// Horizontal texture coordinate along the surface, in `[0,1]`.
    pub(crate) tex_u: f32,
    /// `side` byte for the shading path's foreshorten/tangent fallback: 1 ⇒ the
    /// normal is "more X-ish", 0 ⇒ "more Y-ish". Only consulted when `normal` is
    /// not supplied to a downstream consumer; the real `normal` drives lighting.
    pub(crate) side: u8,
}

/// The cell-local Z plane the 2D cast samples the geometry on. The continuous-DDA
/// walks a fixed integer `cell_z` plane and draws each wall slice full-height,
/// centred on the horizon; it carries no sub-cell eye height into the per-cell hit
/// test (threading one would perturb the near/far cache keys and the pitch/pan
/// slice-plane logic, which this module must not touch). We therefore sample the ramp at
/// the cell mid-height — a deterministic, cache-neutral reference that yields the
/// correct tilted slope normal. A wedge is z-independent and ignores this.
const SAMPLE_LZ: f32 = 0.5;

/// Intersect the 2D ray with the non-`Full` cell `(cx, cy, cell_z)` carrying
/// `shape`. `t_enter` is the ray parameter at which the DDA stepped onto this
/// cell's near face (the perpendicular distance to that face). `(ox, oy)` is the
/// ray origin, `(dx, dy)` the (un-normalized) ray direction the DDA marches.
///
/// Returns:
/// * `Some(hit)` — the ray sees a solid surface inside this cell; the DDA stops.
/// * `None` — the ray passes through the open part; the DDA must continue.
///
/// `shape` MUST be non-`Full` (the caller keeps the Full fast path).
#[allow(clippy::too_many_arguments)]
pub(crate) fn intersect_cell(
    shape: Shape,
    cx: i32,
    cy: i32,
    cell_z: i32,
    ox: f32,
    oy: f32,
    dx: f32,
    dy: f32,
    t_enter: f32,
) -> Option<ShapeHit> {
    // Entry point in world + cell-local coords.
    let t0 = t_enter.max(0.0);
    let ex = ox + t0 * dx;
    let ey = oy + t0 * dy;
    let lz = SAMPLE_LZ;
    let enter_local = Vec3::new(ex - cx as f32, ey - cy as f32, lz);

    // If the entry point is already on the SOLID side, the ray crossed an
    // axis-aligned solid face of this cell — treat it as an ordinary wall hit on
    // that face (the wedge's orthogonal back face, or the ramp's vertical back /
    // its base). Report the entry face with its axis normal.
    if shape_solid_contains(shape, enter_local) {
        return Some(axis_face_hit(t0, ex, ey, cx, cy, dx, dy));
    }

    // Otherwise the ray entered the OPEN half. Intersect it with the shape's
    // diagonal / slope plane and accept the crossing only if it lands inside the
    // cell footprint on the solid boundary.
    let plane = shape_surface(shape)?; // None only for Full, excluded by contract.
                                       // World-space plane point (cell origin + local point) and unit normal (already
                                       // world-correct under the unit-cell uniform scale).
    let p = Vec3::new(
        cx as f32 + plane.point.x,
        cy as f32 + plane.point.y,
        cell_z as f32 + plane.point.z,
    );
    let n = plane.normal;

    match shape {
        Shape::WedgeXY(_) => intersect_wedge(cx, cy, ox, oy, dx, dy, t0, p, n),
        Shape::RampZ(dir) => intersect_ramp(dir, cx, cy, ox, oy, dx, dy, t0, p, n),
        Shape::Full => None,
    }
}

/// Build an axis-aligned wall hit for the entry face the DDA just crossed (the
/// solid orthogonal back-face case). The normal points back toward the camera
/// along whichever axis the ray was advancing on. `t0` is the entry parameter: the
/// hit IS the entry face, so its distance is the entry distance. (Reporting `0` here
/// used to put the face at the camera, which the wall path then drew as a
/// floor-to-sky column: `line_h = proj / PERP_MIN`.)
#[inline]
fn axis_face_hit(t0: f32, ex: f32, ey: f32, cx: i32, cy: i32, dx: f32, dy: f32) -> ShapeHit {
    // Which face did we enter through? The entry point sits on a cell boundary;
    // pick the axis whose coordinate is closest to an integer edge of this cell.
    let fx = ex - cx as f32;
    let fy = ey - cy as f32;
    let dist_x_edge = fx.min(1.0 - fx);
    let dist_y_edge = fy.min(1.0 - fy);
    if dist_x_edge <= dist_y_edge {
        // Crossed an x-const face: normal is -sign(dx) along x.
        let nx = -dx.signum();
        ShapeHit {
            t: t0,
            hit_x: ex,
            hit_y: ey,
            normal: Vec3::new(nx, 0.0, 0.0),
            tex_u: fy.clamp(0.0, 1.0),
            side: 1,
        }
    } else {
        let ny = -dy.signum();
        ShapeHit {
            t: t0,
            hit_x: ex,
            hit_y: ey,
            normal: Vec3::new(0.0, ny, 0.0),
            tex_u: fx.clamp(0.0, 1.0),
            side: 0,
        }
    }
}

/// Vertical-diagonal wedge: intersect the ray with the diagonal vertical plane and
/// accept it inside the cell footprint. The wedge spans the full Z height so the
/// hit needs no z test.
#[allow(clippy::too_many_arguments)]
fn intersect_wedge(
    cx: i32,
    cy: i32,
    ox: f32,
    oy: f32,
    dx: f32,
    dy: f32,
    t0: f32,
    p: Vec3,
    n: Vec3,
) -> Option<ShapeHit> {
    let t = ray_plane_t(ox, oy, dx, dy, p, n)?;
    if t < t0 {
        return None; // plane is behind the entry — open side, pass through.
    }
    let hx = ox + t * dx;
    let hy = oy + t * dy;
    // Inside the cell footprint? (small tolerance for the diagonal endpoints).
    let fx = hx - cx as f32;
    let fy = hy - cy as f32;
    if !(-EPS..=1.0 + EPS).contains(&fx) || !(-EPS..=1.0 + EPS).contains(&fy) {
        return None; // crosses the diagonal LINE outside this cell → open, pass through.
    }
    // Texture coordinate runs along the diagonal (use the larger-varying axis for
    // stability — both run 0→1 across the diagonal).
    let tex_u = fx.clamp(0.0, 1.0);
    Some(ShapeHit {
        t,
        hit_x: hx,
        hit_y: hy,
        normal: n,
        tex_u,
        side: if n.x.abs() >= n.y.abs() { 1 } else { 0 },
    })
}

/// 45° ramp slope: intersect the ray with the sloped plane. The slope surface at a
/// given XY has height `z = slope_height(dir, x, y)`; the ray is cast at the fixed
/// eye z-plane `z_eye`, so the slope is "hit" at the XY point where the slope
/// surface reaches `z_eye` along the ray — i.e. the 2D ray's crossing of the
/// iso-height contour `slope_height == z_eye` within the cell. Below that contour
/// the cell is solid (the ray would already be inside solid); beyond it the slope
/// has dropped under the eye and the cell is open at this z.
#[allow(clippy::too_many_arguments)]
fn intersect_ramp(
    dir: RampDir,
    cx: i32,
    cy: i32,
    ox: f32,
    oy: f32,
    dx: f32,
    dy: f32,
    t0: f32,
    _p: Vec3,
    n: Vec3,
) -> Option<ShapeHit> {
    let lz = SAMPLE_LZ;
    // The slope surface reaches the eye height `lz` along its rise axis at a local
    // coordinate `target` on that axis (PosX `z=x` ⇒ `x=lz`; NegX `z=1−x` ⇒
    // `x=1−lz`; PosY `z=y` ⇒ `y=lz`; NegY `z=1−y` ⇒ `y=1−lz`). Solve the 2D ray for
    // the parameter `t` at which the rise axis hits `target` — the XY point where
    // the slope crosses the eye plane.
    let (axis_is_x, origin_local, d, target) = match dir {
        RampDir::PosX => (true, ox - cx as f32, dx, lz),
        RampDir::NegX => (true, ox - cx as f32, dx, 1.0 - lz),
        RampDir::PosY => (false, oy - cy as f32, dy, lz),
        RampDir::NegY => (false, oy - cy as f32, dy, 1.0 - lz),
    };
    if d.abs() < EPS {
        return None; // ray parallel to the rise axis — never crosses this contour.
    }
    let t = (target - origin_local) / d;
    if t < t0 {
        return None;
    }
    let fx = (ox + t * dx) - cx as f32;
    let fy = (oy + t * dy) - cy as f32;
    if !(-EPS..=1.0 + EPS).contains(&fx) || !(-EPS..=1.0 + EPS).contains(&fy) {
        return None;
    }
    Some(ShapeHit {
        t,
        hit_x: ox + t * dx,
        hit_y: oy + t * dy,
        normal: n,
        tex_u: if axis_is_x {
            fx.clamp(0.0, 1.0)
        } else {
            fy.clamp(0.0, 1.0)
        },
        side: if n.x.abs() >= n.y.abs() { 1 } else { 0 },
    })
}

/// Intersect the 2D ray with a vertical plane through world point `p` with
/// horizontal normal `n` (z component ignored — the wedge plane is vertical).
/// Returns the ray parameter `t`, or `None` if the ray is parallel.
#[inline]
fn ray_plane_t(ox: f32, oy: f32, dx: f32, dy: f32, p: Vec3, n: Vec3) -> Option<f32> {
    let denom = dx * n.x + dy * n.y;
    if denom.abs() < EPS {
        return None;
    }
    let t = ((p.x - ox) * n.x + (p.y - oy) * n.y) / denom;
    Some(t)
}

const EPS: f32 = 1e-5;
