//! Per-cell geometry shapes for true 45° geometry on all axes.
//!
//! # Local-coordinate convention
//!
//! Every geometry query uses **cell-local coordinates** `local ∈ [0, 1)³`
//! where `local.x/y/z = world_pos - cell_origin`.  A point `local = (0,0,0)`
//! is the west-north-floor corner of the cell; `(1,1,1)` is the east-south-
//! ceiling corner (exclusive — the cell is half-open on the upper bound, but
//! the geometry functions accept the full `[0, 1]` closed interval for ray
//! endpoint probing).
//!
//! # Shape semantics
//!
//! * [`Shape::Full`] — the current solid cube.  Every point in the cell is
//!   inside the solid.  **Default** — a world where every cell is `Full`
//!   behaves byte-identically to before this module existed.
//!
//! * `Shape::WedgeXY(DiagXY)` — a **vertical diagonal wall**.  The cell is
//!   split by a vertical half-plane (infinite in Z); one triangular prism in
//!   the XY footprint is solid, the other is open (air).  The wall face is
//!   the 45° diagonal; the two axis-aligned faces on the solid side are the
//!   ordinary wall faces the DDA already handles.
//!
//!   Orientation variants and the inequality that defines the SOLID half:
//!
//!   | Variant | Solid half (in local x,y ∈ \[0,1\]) |
//!   |---------|-----------------------------------|
//!   | `NE` | `x + y ≥ 1`  (north-east triangle) |
//!   | `NW` | `x ≤ y`      (north-west triangle; solid where x is not larger than y) |
//!   | `SE` | `x ≥ y`      (south-east triangle; solid where x is at least y) |
//!   | `SW` | `x + y ≤ 1`  (south-west triangle) |
//!
//!   The diagonal face normals (pointing OUT of the solid half, into the air half):
//!
//!   | Variant | Outward normal (unit vector) |
//!   |---------|------------------------------|
//!   | `NE` | `(-1/√2, -1/√2, 0)` — pointing SW into the air half |
//!   | `NW` | `( 1/√2, -1/√2, 0)` — pointing SE into the air half |
//!   | `SE` | `(-1/√2,  1/√2, 0)` — pointing NW into the air half |
//!   | `SW` | `( 1/√2,  1/√2, 0)` — pointing NE into the air half |
//!
//!   The `(x + y ≥ 1)` / `(x + y ≤ 1)` split is the NE45°/SW45° diagonal; the
//!   `(x ≥ y)` / `(x ≤ y)` split is the NW45°/SE45° diagonal (x=y line).
//!
//! * `Shape::RampZ(RampDir)` — a **45° slope** rising across the cell.  The
//!   solid region is *below* a 45° plane, so the top surface is the slope face.
//!   At the low edge the solid is zero-height (the ramp just touches the floor);
//!   at the high edge the cell is full-height solid.
//!
//!   Orientation variants and the inequality that defines the SOLID half:
//!
//!   | Variant | Rising direction | Solid where … |
//!   |---------|-----------------|---------------|
//!   | `PosX`  | ramp rises as x increases | `z ≤ x`      (floor at x=0, full at x=1) |
//!   | `NegX`  | ramp rises as x decreases | `z ≤ 1 − x`  (floor at x=1, full at x=0) |
//!   | `PosY`  | ramp rises as y increases | `z ≤ y`      (floor at y=0, full at y=1) |
//!   | `NegY`  | ramp rises as y decreases | `z ≤ 1 − y`  (floor at y=1, full at y=0) |
//!
//!   The outward slope normal (pointing away from the solid, out of the ramp
//!   surface upward into the air above it) is `(-1/√2, 0, 1/√2)` etc., depending
//!   on direction — see [`shape_surface`].

use glam::Vec3;

// ---------------------------------------------------------------------------
// Orientation enums.
// ---------------------------------------------------------------------------

/// Diagonal orientation for [`Shape::WedgeXY`]: which triangular half of the
/// XY footprint is solid.
///
/// Naming convention: the variant name identifies the **compass quadrant that
/// contains the solid triangle** when you look at the cell top-down with +X
/// pointing east and +Y pointing south (screen convention).
///
/// ```text
///  NW solid  │  NE solid
///  (x ≤ y)   │  (x+y ≥ 1)
/// ───────────┼───────────
///  SW solid  │  SE solid
///  (x+y ≤ 1) │  (x ≥ y)
/// ```
///
/// The cell-local diagonal runs from corner (0,1)–(1,0) for the
/// `NE`/`SW` split, and from corner (0,0)–(1,1) for the `NW`/`SE` split.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DiagXY {
    /// Solid where `x + y ≥ 1` (north-east triangle).
    /// The diagonal face goes from corner `(0,1)` to `(1,0)`.
    /// Outward normal: `(-1/√2, -1/√2, 0)` pointing SW.
    #[default]
    NE,
    /// Solid where `x ≤ y` (north-west triangle).
    /// The diagonal face goes from corner `(0,0)` to `(1,1)`.
    /// Outward normal: `(1/√2, -1/√2, 0)` pointing SE.
    NW,
    /// Solid where `x ≥ y` (south-east triangle).
    /// The diagonal face goes from corner `(0,0)` to `(1,1)`.
    /// Outward normal: `(-1/√2, 1/√2, 0)` pointing NW.
    SE,
    /// Solid where `x + y ≤ 1` (south-west triangle).
    /// The diagonal face goes from corner `(0,1)` to `(1,0)`.
    /// Outward normal: `(1/√2, 1/√2, 0)` pointing NE.
    SW,
}

/// Rising direction for [`Shape::RampZ`]: which horizontal axis the ramp
/// rises along, and whether it rises toward the positive or negative end.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RampDir {
    /// Ramp rises in the +X direction.  Solid where `z ≤ x`.
    /// Low edge at x=0 (floor), full-height at x=1.
    /// Outward slope normal: `(-1/√2, 0, 1/√2)`.
    #[default]
    PosX,
    /// Ramp rises in the −X direction.  Solid where `z ≤ 1 − x`.
    /// Low edge at x=1 (floor), full-height at x=0.
    /// Outward slope normal: `(1/√2, 0, 1/√2)`.
    NegX,
    /// Ramp rises in the +Y direction.  Solid where `z ≤ y`.
    /// Low edge at y=0 (floor), full-height at y=1.
    /// Outward slope normal: `(0, -1/√2, 1/√2)`.
    PosY,
    /// Ramp rises in the −Y direction.  Solid where `z ≤ 1 − y`.
    /// Low edge at y=1 (floor), full-height at y=0.
    /// Outward slope normal: `(0, 1/√2, 1/√2)`.
    NegY,
}

// ---------------------------------------------------------------------------
// The shape enum.
// ---------------------------------------------------------------------------

/// Per-cell geometry shape.  Shared between `re_world` (collision) and
/// `re_render` (DDA/raymarch hit + normals).
///
/// `Default` is [`Shape::Full`] so every existing cell without an explicit
/// shape set behaves exactly as today — the world is back-compatible.
///
/// See the module-level documentation for the precise solid-region inequalities
/// and surface normals for each variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Shape {
    /// A full solid cube — the current default.  All points inside the cell
    /// are solid.  [`shape_solid_contains`] always returns `true` for this.
    #[default]
    Full,
    /// A vertical 45° diagonal wall cutting the cell footprint.  The cell is
    /// solid on one triangular half in the XY plane, open on the other.
    /// Solid for the full Z height (the wall is vertical).
    WedgeXY(DiagXY),
    /// A 45° slope (ramp) rising across the cell vertically.  The solid
    /// region is *below* the slope surface; above it is open air.
    RampZ(RampDir),
}

// ---------------------------------------------------------------------------
// Surface plane.
// ---------------------------------------------------------------------------

/// The diagonal / slope face of a non-`Full` cell, expressed as a point on
/// the plane and its outward (air-facing) unit normal.
///
/// Used by the M3b DDA to compute the exact ray–face intersection and the
/// surface normal for shading.  Both `point` and `normal` are in **cell-local
/// coordinates** (`[0,1]³`); the DDA scales them to world coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfacePlane {
    /// A point that lies on the plane surface, in cell-local coordinates.
    pub point: Vec3,
    /// Outward unit normal (pointing from the solid side into the open/air side).
    pub normal: Vec3,
}

// ---------------------------------------------------------------------------
// Geometry functions.
// ---------------------------------------------------------------------------

const FRAC_1_SQRT_2: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// Is the cell-local point `local` (each component in `[0, 1]`) inside the
/// **solid** part of `shape`?
///
/// For [`Shape::Full`] this is always `true`.  For wedge / ramp shapes the
/// solid half is defined by the inequalities documented on [`DiagXY`] and
/// [`RampDir`].  Points exactly ON the diagonal boundary are considered solid
/// (the `≥` / `≤` inequalities are non-strict).
///
/// # Panics
///
/// Never panics.  Components outside `[0, 1]` are accepted as-is (the caller
/// converts world positions to local ones; small floating-point overshoot is
/// tolerated rather than clamped — the inequalities are still well-defined).
#[inline]
pub fn shape_solid_contains(shape: Shape, local: Vec3) -> bool {
    match shape {
        Shape::Full => true,
        Shape::WedgeXY(diag) => {
            let (x, y) = (local.x, local.y);
            match diag {
                DiagXY::NE => x + y >= 1.0,
                DiagXY::NW => x <= y,
                DiagXY::SE => x >= y,
                DiagXY::SW => x + y <= 1.0,
            }
        }
        Shape::RampZ(dir) => {
            let (x, y, z) = (local.x, local.y, local.z);
            match dir {
                RampDir::PosX => z <= x,
                RampDir::NegX => z <= 1.0 - x,
                RampDir::PosY => z <= y,
                RampDir::NegY => z <= 1.0 - y,
            }
        }
    }
}

/// Return the diagonal / slope [`SurfacePlane`] for `shape`, or `None` for
/// [`Shape::Full`] (whose faces are axis-aligned and handled by the DDA
/// without this helper).
///
/// The returned plane is in **cell-local coordinates** (`[0,1]³`):
/// * `point` — a canonical point on the face (chosen as the face's geometric
///   mid-point for numerical stability).
/// * `normal` — the outward (air-side) unit normal.
///
/// Both `normal.length()` is exactly `1.0` (within f32 precision), which the
/// renderer relies on for shading dot products.
pub fn shape_surface(shape: Shape) -> Option<SurfacePlane> {
    match shape {
        Shape::Full => None,

        Shape::WedgeXY(diag) => {
            // The diagonal face is vertical (normal has z=0) and runs along the
            // diagonal of the cell footprint at z = [0,1].
            //
            // For NE/SW the diagonal is the line x+y=1 (from (0,1,.) to (1,0,.)).
            // For NW/SE the diagonal is the line x=y (from (0,0,.) to (1,1,.)).
            let (point, normal) = match diag {
                DiagXY::NE => {
                    // Solid: x+y≥1. Air side is SW (x+y<1).
                    // Diagonal at x+y=1, representative point mid-diagonal = (0.5,0.5,0.5).
                    // Normal points INTO the air (SW) side: (-1,-1,0)/√2.
                    (
                        Vec3::new(0.5, 0.5, 0.5),
                        Vec3::new(-FRAC_1_SQRT_2, -FRAC_1_SQRT_2, 0.0),
                    )
                }
                DiagXY::NW => {
                    // Solid: x≤y. Air side is SE (x>y).
                    // Diagonal at x=y, representative point = (0.5,0.5,0.5).
                    // Normal points INTO the air (SE) side: (+1,-1,0)/√2.
                    (
                        Vec3::new(0.5, 0.5, 0.5),
                        Vec3::new(FRAC_1_SQRT_2, -FRAC_1_SQRT_2, 0.0),
                    )
                }
                DiagXY::SE => {
                    // Solid: x≥y. Air side is NW (x<y).
                    // Diagonal at x=y, representative point = (0.5,0.5,0.5).
                    // Normal points INTO the air (NW) side: (-1,+1,0)/√2.
                    (
                        Vec3::new(0.5, 0.5, 0.5),
                        Vec3::new(-FRAC_1_SQRT_2, FRAC_1_SQRT_2, 0.0),
                    )
                }
                DiagXY::SW => {
                    // Solid: x+y≤1. Air side is NE (x+y>1).
                    // Diagonal at x+y=1, representative point = (0.5,0.5,0.5).
                    // Normal points INTO the air (NE) side: (+1,+1,0)/√2.
                    (
                        Vec3::new(0.5, 0.5, 0.5),
                        Vec3::new(FRAC_1_SQRT_2, FRAC_1_SQRT_2, 0.0),
                    )
                }
            };
            Some(SurfacePlane { point, normal })
        }

        Shape::RampZ(dir) => {
            // The slope surface is a 45° plane.  Solid is BELOW it; air is above.
            // The outward normal points upward (+z component) and away from the
            // rising direction (negative along the rise axis).
            //
            // Derivation (PosX as example):
            //   Slope plane: z = x  →  z − x = 0  →  normal direction (−1,0,+1)/√2.
            //   "Outward" = away from the solid, into the air above: same (−1,0,+1)/√2.
            //   Representative point: centre of the cell (0.5,0.5,0.5) lies ON z=x? No:
            //     z=0.5, x=0.5 → on the plane. Good.
            //
            //  PosX:  z=x   →  (-1/√2, 0,  +1/√2)
            //  NegX:  z=1-x →  (+1/√2, 0,  +1/√2)   (normal in +x and +z)
            //  PosY:  z=y   →  (0, -1/√2,  +1/√2)
            //  NegY:  z=1-y →  (0, +1/√2,  +1/√2)
            //
            // All normals are unit vectors: each has two components ±1/√2 → magnitude = 1.
            let (point, normal) = match dir {
                RampDir::PosX => {
                    // Slope z=x. Mid-slope point: x=0.5, y=0.5, z=0.5.
                    (
                        Vec3::new(0.5, 0.5, 0.5),
                        Vec3::new(-FRAC_1_SQRT_2, 0.0, FRAC_1_SQRT_2),
                    )
                }
                RampDir::NegX => {
                    // Slope z=1-x. Mid-slope: x=0.5, y=0.5, z=0.5.
                    (
                        Vec3::new(0.5, 0.5, 0.5),
                        Vec3::new(FRAC_1_SQRT_2, 0.0, FRAC_1_SQRT_2),
                    )
                }
                RampDir::PosY => {
                    // Slope z=y. Mid-slope: x=0.5, y=0.5, z=0.5.
                    (
                        Vec3::new(0.5, 0.5, 0.5),
                        Vec3::new(0.0, -FRAC_1_SQRT_2, FRAC_1_SQRT_2),
                    )
                }
                RampDir::NegY => {
                    // Slope z=1-y. Mid-slope: x=0.5, y=0.5, z=0.5.
                    (
                        Vec3::new(0.5, 0.5, 0.5),
                        Vec3::new(0.0, FRAC_1_SQRT_2, FRAC_1_SQRT_2),
                    )
                }
            };
            Some(SurfacePlane { point, normal })
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Back-compat: Full ----

    #[test]
    fn full_contains_all_corners() {
        for &corner in &[
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(0.5, 0.5, 0.5),
        ] {
            assert!(
                shape_solid_contains(Shape::Full, corner),
                "Full must contain {corner:?}"
            );
        }
    }

    #[test]
    fn full_surface_is_none() {
        assert_eq!(shape_surface(Shape::Full), None);
    }

    #[test]
    fn full_default() {
        let s: Shape = Default::default();
        assert_eq!(s, Shape::Full);
    }

    // ---- WedgeXY NE (solid: x+y≥1) ----

    #[test]
    fn wedge_ne_solid_half() {
        let s = Shape::WedgeXY(DiagXY::NE);
        // Clearly inside solid (x+y > 1).
        assert!(shape_solid_contains(s, Vec3::new(0.8, 0.8, 0.5)));
        assert!(shape_solid_contains(s, Vec3::new(1.0, 0.0, 0.5)));
        assert!(shape_solid_contains(s, Vec3::new(0.0, 1.0, 0.5)));
        // Exactly on the boundary (x+y=1) — solid.
        assert!(shape_solid_contains(s, Vec3::new(0.5, 0.5, 0.0)));
        assert!(shape_solid_contains(s, Vec3::new(0.3, 0.7, 0.9)));
    }

    #[test]
    fn wedge_ne_empty_half() {
        let s = Shape::WedgeXY(DiagXY::NE);
        // Clearly in the air half (x+y < 1).
        assert!(!shape_solid_contains(s, Vec3::new(0.1, 0.1, 0.5)));
        assert!(!shape_solid_contains(s, Vec3::new(0.0, 0.0, 0.5)));
        assert!(!shape_solid_contains(s, Vec3::new(0.3, 0.3, 0.5)));
    }

    // ---- WedgeXY SW (solid: x+y≤1) — complement of NE ----

    #[test]
    fn wedge_sw_solid_half() {
        let s = Shape::WedgeXY(DiagXY::SW);
        assert!(shape_solid_contains(s, Vec3::new(0.0, 0.0, 0.5)));
        assert!(shape_solid_contains(s, Vec3::new(0.3, 0.3, 0.5)));
        // Boundary: solid.
        assert!(shape_solid_contains(s, Vec3::new(0.5, 0.5, 0.5)));
    }

    #[test]
    fn wedge_sw_empty_half() {
        let s = Shape::WedgeXY(DiagXY::SW);
        assert!(!shape_solid_contains(s, Vec3::new(0.8, 0.8, 0.5)));
        assert!(!shape_solid_contains(s, Vec3::new(1.0, 1.0, 0.5)));
    }

    // ---- WedgeXY NW (solid: x≤y) ----

    #[test]
    fn wedge_nw_solid_half() {
        let s = Shape::WedgeXY(DiagXY::NW);
        // x < y → solid.
        assert!(shape_solid_contains(s, Vec3::new(0.2, 0.8, 0.5)));
        assert!(shape_solid_contains(s, Vec3::new(0.0, 1.0, 0.5)));
        // Boundary x=y → solid.
        assert!(shape_solid_contains(s, Vec3::new(0.5, 0.5, 0.5)));
    }

    #[test]
    fn wedge_nw_empty_half() {
        let s = Shape::WedgeXY(DiagXY::NW);
        // x > y → air.
        assert!(!shape_solid_contains(s, Vec3::new(0.9, 0.1, 0.5)));
        assert!(!shape_solid_contains(s, Vec3::new(1.0, 0.0, 0.5)));
    }

    // ---- WedgeXY SE (solid: x≥y) — complement of NW ----

    #[test]
    fn wedge_se_solid_half() {
        let s = Shape::WedgeXY(DiagXY::SE);
        assert!(shape_solid_contains(s, Vec3::new(0.9, 0.1, 0.5)));
        assert!(shape_solid_contains(s, Vec3::new(1.0, 0.0, 0.5)));
        // Boundary x=y → solid.
        assert!(shape_solid_contains(s, Vec3::new(0.5, 0.5, 0.5)));
    }

    #[test]
    fn wedge_se_empty_half() {
        let s = Shape::WedgeXY(DiagXY::SE);
        assert!(!shape_solid_contains(s, Vec3::new(0.2, 0.8, 0.5)));
        assert!(!shape_solid_contains(s, Vec3::new(0.0, 1.0, 0.5)));
    }

    // ---- WedgeXY: z-independence ----

    #[test]
    fn wedge_xy_is_z_independent() {
        // The solid region for WedgeXY is the same at every z.
        let s = Shape::WedgeXY(DiagXY::NE);
        let inside_xy = Vec3::new(0.8, 0.8, 0.0);
        let outside_xy = Vec3::new(0.1, 0.1, 0.0);
        for &z in &[0.0_f32, 0.25, 0.5, 0.75, 1.0] {
            assert!(
                shape_solid_contains(s, Vec3::new(inside_xy.x, inside_xy.y, z)),
                "NE solid must be z-independent at z={z}"
            );
            assert!(
                !shape_solid_contains(s, Vec3::new(outside_xy.x, outside_xy.y, z)),
                "NE air must be z-independent at z={z}"
            );
        }
    }

    // ---- WedgeXY surface planes: normals are unit-length, point to air ----

    #[test]
    fn wedge_ne_surface_normal_unit_and_outward() {
        let sp = shape_surface(Shape::WedgeXY(DiagXY::NE)).unwrap();
        assert!(
            (sp.normal.length() - 1.0).abs() < 1e-6,
            "normal must be unit"
        );
        // NE solid half is where x+y≥1; normal should point SW (into x+y<1 region).
        // SW direction has negative x and negative y.
        assert!(
            sp.normal.x < 0.0 && sp.normal.y < 0.0,
            "NE normal must point SW"
        );
        assert_eq!(sp.normal.z, 0.0, "WedgeXY normal has no z component");
        // The representative point lies on the plane: x+y=1 → 0.5+0.5=1.
        assert!((sp.point.x + sp.point.y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn wedge_sw_surface_normal_unit_and_outward() {
        let sp = shape_surface(Shape::WedgeXY(DiagXY::SW)).unwrap();
        assert!((sp.normal.length() - 1.0).abs() < 1e-6);
        // SW solid half is where x+y≤1; normal points NE.
        assert!(
            sp.normal.x > 0.0 && sp.normal.y > 0.0,
            "SW normal must point NE"
        );
        assert_eq!(sp.normal.z, 0.0);
    }

    #[test]
    fn wedge_nw_surface_normal_unit_and_outward() {
        let sp = shape_surface(Shape::WedgeXY(DiagXY::NW)).unwrap();
        assert!((sp.normal.length() - 1.0).abs() < 1e-6);
        // NW solid half is where x≤y; air is in SE (x>y). Normal points +x, -y.
        assert!(
            sp.normal.x > 0.0 && sp.normal.y < 0.0,
            "NW normal must point SE"
        );
        assert_eq!(sp.normal.z, 0.0);
    }

    #[test]
    fn wedge_se_surface_normal_unit_and_outward() {
        let sp = shape_surface(Shape::WedgeXY(DiagXY::SE)).unwrap();
        assert!((sp.normal.length() - 1.0).abs() < 1e-6);
        // SE solid half is where x≥y; air is in NW (x<y). Normal points -x, +y.
        assert!(
            sp.normal.x < 0.0 && sp.normal.y > 0.0,
            "SE normal must point NW"
        );
        assert_eq!(sp.normal.z, 0.0);
    }

    // ---- RampZ: solid below, air above ----

    #[test]
    fn ramp_posx_solid_below_slope() {
        let s = Shape::RampZ(RampDir::PosX);
        // z=x plane. At x=0.8 the slope is at z=0.8; below (z=0.2) is solid.
        assert!(shape_solid_contains(s, Vec3::new(0.8, 0.5, 0.2)));
        // Above the slope (z=0.9 > x=0.8): air.
        assert!(!shape_solid_contains(s, Vec3::new(0.8, 0.5, 0.9)));
        // Exactly on the slope (z=x): solid (boundary included).
        assert!(shape_solid_contains(s, Vec3::new(0.6, 0.5, 0.6)));
        // At x=0.0 the slope is at z=0; any z>0 is air.
        assert!(!shape_solid_contains(s, Vec3::new(0.0, 0.5, 0.1)));
        // At x=1.0 the slope is at z=1; z=0.9 is solid.
        assert!(shape_solid_contains(s, Vec3::new(1.0, 0.5, 0.9)));
    }

    #[test]
    fn ramp_negx_solid_below_slope() {
        let s = Shape::RampZ(RampDir::NegX);
        // z=1-x plane. At x=0.2, slope at z=0.8. Below (z=0.5) is solid.
        assert!(shape_solid_contains(s, Vec3::new(0.2, 0.5, 0.5)));
        // Above (z=0.9 > 1-0.2=0.8): air.
        assert!(!shape_solid_contains(s, Vec3::new(0.2, 0.5, 0.9)));
        // At x=1.0, slope at z=0: any z>0 is air.
        assert!(!shape_solid_contains(s, Vec3::new(1.0, 0.5, 0.1)));
    }

    #[test]
    fn ramp_posy_solid_below_slope() {
        let s = Shape::RampZ(RampDir::PosY);
        // z=y. At y=0.7, slope at z=0.7; z=0.3 is solid.
        assert!(shape_solid_contains(s, Vec3::new(0.5, 0.7, 0.3)));
        // z=0.9 is above slope: air.
        assert!(!shape_solid_contains(s, Vec3::new(0.5, 0.7, 0.9)));
    }

    #[test]
    fn ramp_negy_solid_below_slope() {
        let s = Shape::RampZ(RampDir::NegY);
        // z=1-y. At y=0.3, slope at z=0.7; z=0.5 is solid.
        assert!(shape_solid_contains(s, Vec3::new(0.5, 0.3, 0.5)));
        // z=0.9 > 0.7: air.
        assert!(!shape_solid_contains(s, Vec3::new(0.5, 0.3, 0.9)));
    }

    // ---- RampZ: surface normals point up and away from solid ----

    #[test]
    fn ramp_posx_surface_normal_unit_and_upward() {
        let sp = shape_surface(Shape::RampZ(RampDir::PosX)).unwrap();
        assert!(
            (sp.normal.length() - 1.0).abs() < 1e-6,
            "normal must be unit"
        );
        // Outward normal points AWAY from solid (above the slope) and against +X
        // (the rising direction). So z > 0 and x < 0.
        assert!(
            sp.normal.z > 0.0,
            "RampZ PosX normal must point upward (+z)"
        );
        assert!(
            sp.normal.x < 0.0,
            "RampZ PosX normal must point against rise axis (-x)"
        );
        assert_eq!(sp.normal.y, 0.0, "RampZ PosX normal has no y component");
        // Point lies on the slope: z=x → 0.5=0.5. ✓
        assert!((sp.point.z - sp.point.x).abs() < 1e-6);
    }

    #[test]
    fn ramp_negx_surface_normal_unit_and_upward() {
        let sp = shape_surface(Shape::RampZ(RampDir::NegX)).unwrap();
        assert!((sp.normal.length() - 1.0).abs() < 1e-6);
        assert!(sp.normal.z > 0.0);
        assert!(
            sp.normal.x > 0.0,
            "RampZ NegX normal must point +x (against fall direction)"
        );
        assert_eq!(sp.normal.y, 0.0);
        // Point on slope: z=1-x → 0.5=1-0.5. ✓
        assert!((sp.point.z - (1.0 - sp.point.x)).abs() < 1e-6);
    }

    #[test]
    fn ramp_posy_surface_normal_unit_and_upward() {
        let sp = shape_surface(Shape::RampZ(RampDir::PosY)).unwrap();
        assert!((sp.normal.length() - 1.0).abs() < 1e-6);
        assert!(sp.normal.z > 0.0);
        assert!(sp.normal.y < 0.0, "RampZ PosY normal must point -y");
        assert_eq!(sp.normal.x, 0.0);
        assert!((sp.point.z - sp.point.y).abs() < 1e-6);
    }

    #[test]
    fn ramp_negy_surface_normal_unit_and_upward() {
        let sp = shape_surface(Shape::RampZ(RampDir::NegY)).unwrap();
        assert!((sp.normal.length() - 1.0).abs() < 1e-6);
        assert!(sp.normal.z > 0.0);
        assert!(sp.normal.y > 0.0, "RampZ NegY normal must point +y");
        assert_eq!(sp.normal.x, 0.0);
        assert!((sp.point.z - (1.0 - sp.point.y)).abs() < 1e-6);
    }

    // ---- All normals are unit-length ----

    #[test]
    fn all_surface_normals_are_unit_length() {
        let shapes = [
            Shape::WedgeXY(DiagXY::NE),
            Shape::WedgeXY(DiagXY::NW),
            Shape::WedgeXY(DiagXY::SE),
            Shape::WedgeXY(DiagXY::SW),
            Shape::RampZ(RampDir::PosX),
            Shape::RampZ(RampDir::NegX),
            Shape::RampZ(RampDir::PosY),
            Shape::RampZ(RampDir::NegY),
        ];
        for s in &shapes {
            let sp = shape_surface(*s).expect("non-Full shape must return Some");
            let len = sp.normal.length();
            assert!(
                (len - 1.0).abs() < 1e-6,
                "normal for {s:?} has length {len}, expected 1.0"
            );
        }
    }

    // ---- Back-compat: NE/SW and NW/SE are complementary ----

    #[test]
    fn ne_and_sw_are_complements() {
        // Every point is in exactly one of NE or SW (boundary on both).
        let ne = Shape::WedgeXY(DiagXY::NE);
        let sw = Shape::WedgeXY(DiagXY::SW);
        let probes = [
            Vec3::new(0.1, 0.1, 0.5), // SW solid
            Vec3::new(0.9, 0.9, 0.5), // NE solid
            Vec3::new(0.5, 0.5, 0.5), // boundary: both solid (x+y=1)
        ];
        for p in &probes {
            if p.x + p.y < 1.0 - 1e-6 {
                // strictly SW: NE=false, SW=true
                assert!(!shape_solid_contains(ne, *p));
                assert!(shape_solid_contains(sw, *p));
            } else if p.x + p.y > 1.0 + 1e-6 {
                // strictly NE: NE=true, SW=false
                assert!(shape_solid_contains(ne, *p));
                assert!(!shape_solid_contains(sw, *p));
            } else {
                // boundary: both solid
                assert!(shape_solid_contains(ne, *p));
                assert!(shape_solid_contains(sw, *p));
            }
        }
    }

    #[test]
    fn nw_and_se_are_complements() {
        let nw = Shape::WedgeXY(DiagXY::NW);
        let se = Shape::WedgeXY(DiagXY::SE);
        let probes = [
            Vec3::new(0.2, 0.8, 0.5), // x<y: NW solid
            Vec3::new(0.8, 0.2, 0.5), // x>y: SE solid
            Vec3::new(0.5, 0.5, 0.5), // x=y: both solid
        ];
        for p in &probes {
            if p.x < p.y - 1e-6 {
                assert!(shape_solid_contains(nw, *p));
                assert!(!shape_solid_contains(se, *p));
            } else if p.x > p.y + 1e-6 {
                assert!(!shape_solid_contains(nw, *p));
                assert!(shape_solid_contains(se, *p));
            } else {
                assert!(shape_solid_contains(nw, *p));
                assert!(shape_solid_contains(se, *p));
            }
        }
    }

    // ---- Ramp surface points lie on the slope plane ----

    #[test]
    fn ramp_surface_points_lie_on_plane() {
        // For PosX: slope is z=x. The representative point must satisfy z-x=0.
        let sp_posx = shape_surface(Shape::RampZ(RampDir::PosX)).unwrap();
        assert!((sp_posx.point.z - sp_posx.point.x).abs() < 1e-6);

        let sp_negx = shape_surface(Shape::RampZ(RampDir::NegX)).unwrap();
        assert!((sp_negx.point.z - (1.0 - sp_negx.point.x)).abs() < 1e-6);

        let sp_posy = shape_surface(Shape::RampZ(RampDir::PosY)).unwrap();
        assert!((sp_posy.point.z - sp_posy.point.y).abs() < 1e-6);

        let sp_negy = shape_surface(Shape::RampZ(RampDir::NegY)).unwrap();
        assert!((sp_negy.point.z - (1.0 - sp_negy.point.y)).abs() < 1e-6);
    }

    // ---- DiagXY and RampDir defaults ----

    #[test]
    fn diagxy_default_is_ne() {
        assert_eq!(DiagXY::default(), DiagXY::NE);
    }

    #[test]
    fn rampdir_default_is_posx() {
        assert_eq!(RampDir::default(), RampDir::PosX);
    }
}
