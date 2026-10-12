//! Log-polar self-similar field — the shared core of the infinite-zoom material
//! and the tunnel set-piece. Pure compute, no per-cell allocation.
//!
//! The "infinite" illusion is a log-polar coordinate warp whose pattern is
//! PERIODIC over one zoom-octave ([`LN2`] in log-radius) and one turn ([`TAU`] in
//! angle), so scrolling either axis loops seamlessly. No escape-time fractal: the
//! field is built from a cosine ring/arm wave plus the crate's existing
//! [`fbm`](crate::ascii_shader) noise, sampled in `(ring, arm)` space so detail
//! repeats across octaves (self-similar across scale).
use glam::Vec2;
use noise::Perlin;

use crate::ascii_shader::fbm;

/// One zoom-octave in log-radius: a whole-octave zoom scroll returns the same
/// field, so [`log_polar_field`] loops seamlessly when `phase.zoom` advances by
/// this much.
pub const LN2: f32 = std::f32::consts::LN_2;
/// One full turn in angle: a `phase.spin` scroll of this much loops seamlessly.
pub const TAU: f32 = std::f32::consts::TAU;
/// Centre clamp: below this radius `ln`/`atan2` are ill-defined; clamp so the
/// centre cell reads a finite field (no NaN/inf). A unit test sweeps the exact
/// centre (0.5,0.5) to prove this.
const R_EPS: f32 = 1e-4;

/// The fixed shape of the log-polar field for one material/tunnel instance — the
/// arm count, fBm enrichment, and arm curl. Distinct from the time-varying
/// [`LogPolarPhase`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FractalParams {
    /// Spiral arms; `0` = pure concentric rings.
    pub arms: u8,
    /// fBm enrichment strength `[0,1]` (self-similar across octaves).
    pub octave_warp: f32,
    /// Arm curl: shears angle by log-radius (log spiral). `0` = straight rays.
    ///
    /// NOTE: a non-zero `twist` makes a PURE zoom-only loop seam unless paired
    /// with the coupled spin of [`AnimKind::SpinZoom`](crate::ascii_shader::AnimKind::SpinZoom)
    /// — over one [`LN2`] zoom the spin advances a whole [`TAU`], so both seams
    /// coincide. Zoom-only materials use `twist = 0`.
    pub twist: f32,
}

/// The time-varying scroll of the field: how far the pattern has zoomed and spun.
/// The caller reduces each axis into its loop period before sampling.
#[derive(Clone, Copy, Debug, Default)]
pub struct LogPolarPhase {
    /// Added to log-radius (the zoom scroll). Loops over [`LN2`].
    pub zoom: f32,
    /// Added to angle (the spin scroll). Loops over [`TAU`].
    pub spin: f32,
}

/// Seamless SpinZoom phase: couples spin to zoom AND cancels the field's
/// `twist*log_radius` per-octave drift so the loop is seamless for ANY twist.
///
/// The field angle is `theta = atan2 + phase.spin + twist*(ln r + phase.zoom)`,
/// so advancing the zoom by one octave ([`LN2`]) injects `twist*LN2` of extra
/// angle. A naive `spin = (zoom/LN2)*TAU` coupling does NOT cancel that, leaving
/// a per-loop angular hitch (visible at the shipped `twist = 0.9`). Subtracting
/// `twist*zoom` from the spin makes the total per-octave angle advance exactly
/// `TAU`: Δθ = (TAU - twist*LN2) + twist*LN2 = TAU, so the [`arms`]-wrapped angle
/// advances a whole integer (`arms`) → seamless for any `twist`.
///
/// `zoom_scroll` is the raw (un-wrapped) zoom advance; the returned zoom is
/// reduced into `[0, LN2)`. At `twist = 0` this is byte-identical to the old
/// `(zoom/LN2)*TAU` coupling, so pure-zoom/spin materials are unchanged.
///
/// [`arms`]: FractalParams::arms
#[inline]
pub fn spinzoom_phase(zoom_scroll: f32, twist: f32) -> LogPolarPhase {
    let z = zoom_scroll.rem_euclid(LN2);
    LogPolarPhase {
        zoom: z,
        spin: (z / LN2) * TAU - twist * z,
    }
}

/// One sample of the log-polar field.
#[derive(Clone, Copy, Debug)]
pub struct FractalSample {
    /// Height/density `[0,1]` — drives the body glyph ramp (like `height_at`).
    pub height: f32,
    /// Position within the current octave `[0,1)` — drives the ring palette.
    pub ring_phase: f32,
    /// Angular position `[0,1)` — drives the arm tint.
    pub angle_idx: f32,
}

/// Sample the field at `(u,v)` in `[0,1]` measured from the zone centre `(0.5,0.5)`.
///
/// Periodicity is the seamlessness guarantee: the field is a function periodic in
/// the [`LN2`]-wrapped log-radius and the [`TAU`]-wrapped angle, so any whole-octave
/// zoom or full-turn spin returns an identical field (proven by the loop tests).
/// Pure compute, no allocation; a single `ln` + `atan2` (+ a 2-octave fBm only when
/// `octave_warp > 0`).
#[inline]
pub fn log_polar_field(
    u: f32,
    v: f32,
    phase: LogPolarPhase,
    p: &FractalParams,
    perlin: &Perlin,
) -> FractalSample {
    let dx = u - 0.5;
    let dy = v - 0.5;
    // Centre clamp: at r→0, ln/atan2 are undefined — clamp so the centre is finite.
    let r = (dx * dx + dy * dy).sqrt().max(R_EPS);
    let lr = r.ln() + phase.zoom;
    let theta = dy.atan2(dx) + phase.spin + p.twist * lr;

    let ring = (lr / LN2).rem_euclid(1.0); // [0,1) within the octave
    let arms = p.arms as f32;
    let ang = if arms > 0.0 {
        (theta * arms / TAU).rem_euclid(1.0)
    } else {
        (theta / TAU).rem_euclid(1.0)
    };

    let ring_wave = 0.5 - 0.5 * (TAU * ring).cos(); // nested rings
    let arm_wave = if arms > 0.0 {
        0.5 - 0.5 * (TAU * ang).cos()
    } else {
        1.0
    };
    let mut h = ring_wave * arm_wave;

    if p.octave_warp > 0.0 {
        // Sample noise in (ring, ang) space so detail repeats across octaves
        // (self-similar). Only 2 octaves — cheap; LOD trims further at call sites.
        let n = fbm(perlin, Vec2::new(ring * 4.0, ang * 4.0), 2, 2.0, 0.5);
        h = h * (1.0 - p.octave_warp) + n * p.octave_warp;
    }

    FractalSample {
        height: h.clamp(0.0, 1.0),
        ring_phase: ring,
        angle_idx: ang,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn perlin() -> Perlin {
        Perlin::new(crate::ascii_shader::SHADER_SEED)
    }

    #[test]
    fn zoom_loops_seamlessly_over_one_octave() {
        // twist = 0 ⇒ a whole-octave zoom returns an identical field (no seam).
        let p = FractalParams {
            arms: 0,
            octave_warp: 0.0,
            twist: 0.0,
        };
        let pl = perlin();
        for &(u, v) in &[(0.30, 0.42), (0.71, 0.18), (0.55, 0.83)] {
            let a = log_polar_field(
                u,
                v,
                LogPolarPhase {
                    zoom: 0.0,
                    spin: 0.0,
                },
                &p,
                &pl,
            );
            let b = log_polar_field(
                u,
                v,
                LogPolarPhase {
                    zoom: LN2,
                    spin: 0.0,
                },
                &p,
                &pl,
            );
            assert!((a.height - b.height).abs() < 1e-4, "zoom seam at ({u},{v})");
            assert!((a.ring_phase - b.ring_phase).abs() < 1e-4);
        }
        // The seam test is only meaningful if the field actually MOVES across the
        // octave: a constant field would pass the loop check trivially. Assert the
        // mid-octave phase differs from the seam phase at a sample point.
        let (u, v) = (0.71, 0.18);
        let edge = log_polar_field(
            u,
            v,
            LogPolarPhase {
                zoom: 0.0,
                spin: 0.0,
            },
            &p,
            &pl,
        );
        let mid = log_polar_field(
            u,
            v,
            LogPolarPhase {
                zoom: LN2 / 2.0,
                spin: 0.0,
            },
            &p,
            &pl,
        );
        assert!(
            (edge.height - mid.height).abs() > 1e-3,
            "zoom field is constant — the seamless-loop check would pass trivially"
        );
    }

    #[test]
    fn spin_loops_seamlessly_over_full_turn() {
        let p = FractalParams {
            arms: 3,
            octave_warp: 0.0,
            twist: 0.0,
        };
        let pl = perlin();
        for &(u, v) in &[(0.30, 0.42), (0.71, 0.18), (0.55, 0.83)] {
            let a = log_polar_field(
                u,
                v,
                LogPolarPhase {
                    zoom: 0.0,
                    spin: 0.0,
                },
                &p,
                &pl,
            );
            let b = log_polar_field(
                u,
                v,
                LogPolarPhase {
                    zoom: 0.0,
                    spin: TAU,
                },
                &p,
                &pl,
            );
            assert!((a.height - b.height).abs() < 1e-4, "spin seam at ({u},{v})");
        }
        // Guard against a trivially-constant field: half a turn must change the
        // field at a sample point, so the full-turn loop check is non-vacuous.
        let (u, v) = (0.30, 0.42);
        let edge = log_polar_field(
            u,
            v,
            LogPolarPhase {
                zoom: 0.0,
                spin: 0.0,
            },
            &p,
            &pl,
        );
        let half = log_polar_field(
            u,
            v,
            LogPolarPhase {
                zoom: 0.0,
                spin: TAU / 2.0,
            },
            &p,
            &pl,
        );
        assert!(
            (edge.height - half.height).abs() > 1e-3,
            "spin field is constant — the seamless-loop check would pass trivially"
        );
    }

    #[test]
    fn octave_warp_is_self_similar_across_octaves() {
        // With octave_warp > 0 the fBm enrichment is sampled in (ring, ang) space,
        // which the LN2-wrapped log-radius makes periodic per octave. Two samples
        // whose (ring, ang) inputs coincide but sit a whole octave apart in absolute
        // log-radius must therefore agree — that periodicity IS the self-similarity.
        // We realise "one octave apart" via a zoom phase of LN2 (twist = 0 so the
        // angle is untouched), at the same (u,v): the ring/ang wrap identically.
        let p = FractalParams {
            arms: 4,
            octave_warp: 0.6,
            twist: 0.0,
        };
        let pl = perlin();
        for &(u, v) in &[(0.33, 0.61), (0.78, 0.27), (0.5, 0.9)] {
            let a = log_polar_field(
                u,
                v,
                LogPolarPhase {
                    zoom: 0.0,
                    spin: 0.0,
                },
                &p,
                &pl,
            );
            let b = log_polar_field(
                u,
                v,
                LogPolarPhase {
                    zoom: LN2,
                    spin: 0.0,
                },
                &p,
                &pl,
            );
            // Same (ring, ang) one octave up ⇒ identical enriched height.
            assert!(
                (a.height - b.height).abs() < 1e-4,
                "octave_warp not self-similar across one octave at ({u},{v})"
            );
            assert!((a.angle_idx - b.angle_idx).abs() < 1e-4);
        }
    }

    /// The un-wrapped SpinZoom phase for a continuous scroll `s`. This is what
    /// `spinzoom_phase` produces BEFORE the `[0,LN2)` reduction; sampling the field
    /// at `s` and `s + LN2` with it is the genuine seamlessness invariant (the wrapped
    /// `spinzoom_phase(0)` vs `spinzoom_phase(LN2)` both collapse to z=0 and so would
    /// pass trivially even on the broken coupling). `spinzoom_phase` is the reduction
    /// of this, verified by `spinzoom_phase_reduces_unwrapped`.
    fn unwrapped_spinzoom(s: f32, twist: f32) -> LogPolarPhase {
        LogPolarPhase {
            zoom: s,
            spin: (s / LN2) * TAU - twist * s,
        }
    }

    #[test]
    fn spinzoom_loops_seamlessly_at_shipped_twist() {
        // [C1] regression: at a NON-ZERO twist the naive `(zoom/LN2)*TAU` coupling
        // failed to cancel the field's `twist*log_radius` per-octave drift, leaving a
        // visible angular hitch every loop (~half an arm-period). `spinzoom_phase`
        // absorbs that drift, so advancing the (un-wrapped) scroll by one octave LN2
        // returns an IDENTICAL field. These are the SHIPPED portal_spiral params
        // (arms=5, twist=0.9, octave_warp=0.30).
        //
        // This test FAILS on the pre-fix coupling: `unwrapped_spinzoom` here matches
        // the new `spin = (s/LN2)*TAU - twist*s`; replace it with the old
        // `(s/LN2)*TAU` and the per-octave angle drifts by `twist*LN2`, breaking the
        // assert (verified: old coupling jumps ~0.496 of an arm-period at the seam).
        let p = FractalParams {
            arms: 5,
            octave_warp: 0.30,
            twist: 0.9,
        };
        let pl = perlin();
        for &(u, v) in &[(0.30, 0.42), (0.71, 0.18), (0.55, 0.83), (0.18, 0.66)] {
            for &s in &[0.0_f32, 0.137, 0.4, 0.6] {
                let a = log_polar_field(u, v, unwrapped_spinzoom(s, p.twist), &p, &pl);
                let b = log_polar_field(u, v, unwrapped_spinzoom(s + LN2, p.twist), &p, &pl);
                assert!(
                    (a.height - b.height).abs() < 1e-3,
                    "SpinZoom height seam at ({u},{v}) s={s} twist=0.9: {} vs {}",
                    a.height,
                    b.height
                );
                // angle_idx wraps in [0,1): compare on the circle.
                let da = (a.angle_idx - b.angle_idx).abs();
                assert!(
                    da.min(1.0 - da) < 1e-3,
                    "SpinZoom arm seam at ({u},{v}) s={s}: {} vs {}",
                    a.angle_idx,
                    b.angle_idx
                );
            }
        }
        // Non-vacuous: the field must actually MOVE at the half-period, else the seam
        // check would pass trivially on a constant field.
        let (u, v) = (0.71, 0.18);
        let edge = log_polar_field(u, v, unwrapped_spinzoom(0.0, p.twist), &p, &pl);
        let mid = log_polar_field(u, v, unwrapped_spinzoom(LN2 / 2.0, p.twist), &p, &pl);
        assert!(
            (edge.height - mid.height).abs() > 1e-3,
            "SpinZoom field is constant — the seamless-loop check would pass trivially"
        );
    }

    #[test]
    fn spinzoom_phase_reduces_unwrapped() {
        // `spinzoom_phase` IS the un-wrapped phase with the zoom reduced into [0,LN2);
        // the spin uses the reduced zoom, which is exactly what `log_polar_field` needs
        // (the field is periodic per octave). Tie the helper to the tested invariant.
        for &s in &[0.0, 0.137, LN2 / 3.0, LN2, 1.7, 5.0] {
            let z = s.rem_euclid(LN2);
            let ph = spinzoom_phase(s, 0.9);
            assert_eq!(ph.zoom, z);
            assert_eq!(ph.spin, (z / LN2) * TAU - 0.9 * z);
        }
    }

    #[test]
    fn spinzoom_phase_matches_old_coupling_at_zero_twist() {
        // At twist=0 the new helper must be byte-identical to the old
        // `(zoom/LN2)*TAU` coupling so pure-zoom/spin materials are unchanged.
        for &zs in &[0.0, 0.1, LN2 / 3.0, LN2, 1.7, 5.0] {
            let ph = spinzoom_phase(zs, 0.0);
            let z = zs.rem_euclid(LN2);
            assert_eq!(ph.zoom, z);
            assert_eq!(ph.spin, (z / LN2) * TAU);
        }
    }

    #[test]
    fn output_is_bounded_and_finite_including_centre() {
        let p = FractalParams {
            arms: 5,
            octave_warp: 0.6,
            twist: 0.8,
        };
        let pl = perlin();
        // sweep, INCLUDING the exact centre (0.5,0.5) where r→0.
        for iy in 0..=10 {
            for ix in 0..=10 {
                let u = ix as f32 / 10.0;
                let v = iy as f32 / 10.0;
                let s = log_polar_field(
                    u,
                    v,
                    LogPolarPhase {
                        zoom: 1.3,
                        spin: 0.7,
                    },
                    &p,
                    &pl,
                );
                assert!(
                    s.height.is_finite() && (0.0..=1.0).contains(&s.height),
                    "height oob at ({u},{v})"
                );
                assert!(s.ring_phase.is_finite() && s.angle_idx.is_finite());
            }
        }
        let c = log_polar_field(0.5, 0.5, LogPolarPhase::default(), &p, &pl);
        assert!(c.height.is_finite(), "centre singularity not clamped");
    }
}
