//! `re_render::ripple` — expanding-ring math for `re_world::MotionKind::Ripple`.
//!
//! Pipeline stage: pure per-particle math called from `emitter.rs`'s motion
//! integration. Inputs: a particle's fixed angle/seed + its age. Output: the
//! particle's world position ([`ring_pos`]) each frame. Knobs: none authored
//! here — `speed`/`life`/`rate` live on the `[emitter]` section.
//!
//! ## Why a ring is a MOTION kind, not a new `.lvl` section
//!
//! Ripples are NOT their own authoring surface — reusing the existing
//! `[emitter]` section (`re_render/src/emitter.rs`, `re_world/src/emitter_spec.rs`,
//! and the `[emitter]` section of `re_level`'s parser) avoids adding a new
//! top-level `.lvl` section, which would touch section-dispatch code this
//! module does not own. So a ripple here is authored as an ordinary
//! `[emitter]` whose `motion = ripple speed=…` replaces the usual vel/accel
//! integration: each spawned particle is one point on the ring, at a FIXED
//! angle drawn from its own deterministic seed, whose radius grows linearly
//! with age. A steady `rate` therefore keeps refilling the ring at fresh
//! angles while old points die at `lifetime`, and because particles of many
//! different ages coexist, several concentric rings show at once from ONE
//! emitter — a dripping-tap read with no periodic re-trigger needed. This
//! reuses 100% of the existing world-projection + occlusion + colour-ramp
//! splat path (a ripple particle occludes/gets-occluded exactly like any
//! other), so walls correctly hide a ring behind them.
//!
//! ## The underline hem
//!
//! The ring's leading edge wants a `curly` UNDERLINE hem. The particle IS
//! given that style (see `emitter::particle_cell_styled`-adjacent code in
//! `emitter.rs`), and the `emitters` composite slot's `GlyphOnly` blend
//! (`re_core::layer.rs::apply_emitter_blend`) carries a source cell's
//! `style` through to the accumulator whenever that style is non-empty
//! (`CellStyle::is_empty()` false) — an ordinary particle splat still has no
//! style, so the common case is an unchanged no-op, but a ripple's hem cell
//! does, and reaches the final composited frame. The ring's visible
//! glyph+colour fade (`'∘' → '·' → ' '`) is unchanged and still carries the
//! read when the hem style is dim relative to the fade.

use glam::Vec3;

/// The world position of a ripple point at `age` seconds, given its fixed
/// per-particle `angle` (radians) and the ring's `origin` (world cells, `z`
/// held constant — a flat ring on the floor/water plane) and `speed`
/// (cells/sec of radial growth). Pure.
#[inline]
pub fn ring_pos(origin: Vec3, angle: f32, age: f32, speed: f32) -> Vec3 {
    let r = (speed.max(0.0) * age).max(0.0);
    let (s, c) = angle.sin_cos();
    Vec3::new(origin.x + r * c, origin.y + r * s, origin.z)
}

/// A deterministic angle in `[0, 2π)` from a particle's seed — every ring
/// point gets its own fixed angle, so a steady spawn rate scatters points all
/// the way around the circle rather than piling them on one bearing. Pure.
#[inline]
pub fn ring_angle(unit: f32) -> f32 {
    unit.clamp(0.0, 1.0) * std::f32::consts::TAU
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_pos_grows_linearly_with_age() {
        let origin = Vec3::new(5.0, 5.0, 1.0);
        let p0 = ring_pos(origin, 0.0, 0.0, 2.0);
        assert_eq!(p0, origin, "age 0 is exactly the origin");
        let p1 = ring_pos(origin, 0.0, 1.0, 2.0);
        // angle 0 => +x direction, radius = speed*age = 2.0.
        assert!((p1.x - (origin.x + 2.0)).abs() < 1e-5);
        assert!((p1.y - origin.y).abs() < 1e-5);
        let p2 = ring_pos(origin, 0.0, 2.0, 2.0);
        assert!(
            (p2.x - (origin.x + 4.0)).abs() < 1e-5,
            "radius doubles when age doubles"
        );
    }

    #[test]
    fn ring_pos_z_is_held_constant() {
        let origin = Vec3::new(1.0, 1.0, 2.5);
        let p = ring_pos(origin, 1.234, 3.0, 0.5);
        assert_eq!(p.z, 2.5, "a ripple stays on its origin's horizontal plane");
    }

    #[test]
    fn ring_pos_negative_speed_is_clamped_to_zero_radius() {
        let origin = Vec3::new(0.0, 0.0, 0.0);
        let p = ring_pos(origin, 0.7, 5.0, -3.0);
        assert_eq!(
            p, origin,
            "a negative speed must not shrink the ring inward"
        );
    }

    #[test]
    fn ring_angle_spans_full_circle_and_is_deterministic() {
        assert_eq!(ring_angle(0.0), 0.0);
        assert!((ring_angle(1.0) - std::f32::consts::TAU).abs() < 1e-5);
        assert_eq!(ring_angle(0.25), ring_angle(0.25), "pure function");
    }

    #[test]
    fn different_angles_spread_particles_around_the_ring() {
        let origin = Vec3::ZERO;
        let a = ring_pos(origin, ring_angle(0.0), 2.0, 1.0);
        let b = ring_pos(origin, ring_angle(0.5), 2.0, 1.0);
        // Opposite sides of the ring at the same age/radius.
        assert!((a.x - (-b.x)).abs() < 1e-4);
        assert!((a.y - (-b.y)).abs() < 1e-4);
    }
}
