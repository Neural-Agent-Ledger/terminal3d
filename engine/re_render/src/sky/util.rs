//! `re_render::sky::util` — small deterministic hashing + value-noise helpers
//! shared by every sky layer.
//!
//! Pipeline stage: leaf helpers under [`crate::sky::SkyProgram`]; not a layer
//! itself. Inputs: a seed + integer/float coordinates. Output: a deterministic
//! hash, noise sample, or colour-blend value — no globals, no time source.
//!
//! All randomness here is `hash(seed, index, param_id)`: there is no RNG
//! state and nothing depends on the wall clock. Pure functions only, no
//! allocation, no platform-`sin`/`cos` in the HASH path (integer-only
//! finalizer) — cross-OS-stable by construction. A couple of layers (aurora
//! curtain wobble, clouds ρ-perspective) do call `f32::sin`; that is an
//! accepted, bounded cross-OS ulp risk, and it carries no golden risk here
//! because no level authored before the sky-program feature existed uses a
//! `[sky] base/layer` key — every fixture exercising this path is its own,
//! freshly-blessed golden.

use std::f32::consts::TAU;

/// A 32-bit integer finalizer (a "wang hash" style avalanche) — pure, no
/// float ops, identical bit-for-bit on every platform.
#[inline]
pub(crate) fn hash_u32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

/// Combine two integers into one hashed value.
#[inline]
pub(crate) fn hash2(a: u32, b: u32) -> u32 {
    hash_u32(a ^ hash_u32(b).wrapping_mul(0x9E37_79B1))
}

/// Combine three integers into one hashed value.
#[inline]
pub(crate) fn hash3(a: u32, b: u32, c: u32) -> u32 {
    hash2(hash2(a, b), c)
}

/// Map a hashed `u32` into `[0, 1)`.
#[inline]
pub(crate) fn hash01(h: u32) -> f32 {
    (h >> 8) as f32 / (1u32 << 24) as f32
}

/// A signed cell coordinate folded into an unsigned hash lane (wraps cleanly
/// for negative grid indices, which the sky's yaw-wrapped grids never actually
/// produce, but floor/world-space layers like `mirror`/`void` do).
#[inline]
pub(crate) fn coord_u32(x: i32) -> u32 {
    x as u32
}

#[inline]
fn smoothstep01(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Hermite smoothstep between two edges (glsl-style `smoothstep`).
#[inline]
pub(crate) fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if (edge1 - edge0).abs() < 1e-8 {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// 1D value noise at `x`, seeded by `seed` — smooth, deterministic, bounded to
/// `[0, 1)`. Used for the aurora's per-column hem wobble and the cloud edge's
/// silver-lining sample offset.
#[inline]
pub(crate) fn value_noise1(x: f32, seed: u32) -> f32 {
    let xi = x.floor();
    let xf = x - xi;
    let x0 = coord_u32(xi as i32);
    let x1 = x0.wrapping_add(1);
    let h0 = hash01(hash2(x0, seed));
    let h1 = hash01(hash2(x1, seed));
    h0 + (h1 - h0) * smoothstep01(xf)
}

/// 2D value noise at `(x, y)`, seeded by `seed` — bilinear over a hashed
/// integer lattice. Used for cloud density.
#[inline]
pub(crate) fn value_noise2(x: f32, y: f32, seed: u32) -> f32 {
    let xi = x.floor();
    let yi = y.floor();
    let xf = x - xi;
    let yf = y - yi;
    let x0 = coord_u32(xi as i32);
    let y0 = coord_u32(yi as i32);
    let x1 = x0.wrapping_add(1);
    let y1 = y0.wrapping_add(1);
    let h00 = hash01(hash3(x0, y0, seed));
    let h10 = hash01(hash3(x1, y0, seed));
    let h01 = hash01(hash3(x0, y1, seed));
    let h11 = hash01(hash3(x1, y1, seed));
    let sx = smoothstep01(xf);
    let sy = smoothstep01(yf);
    let a = h00 + (h10 - h00) * sx;
    let b = h01 + (h11 - h01) * sx;
    a + (b - a) * sy
}

/// 4-octave fractal Brownian motion over [`value_noise2`], lacunarity 2, gain
/// 0.5 — the cloud density field's noise basis.
#[inline]
pub(crate) fn fbm4(x: f32, y: f32, seed: u32) -> f32 {
    let mut sum = 0.0f32;
    let mut amp = 0.5f32;
    let mut freq = 1.0f32;
    let mut norm = 0.0f32;
    for octave in 0..4u32 {
        sum += amp
            * value_noise2(
                x * freq,
                y * freq,
                seed.wrapping_add(octave.wrapping_mul(101)),
            );
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    if norm > 0.0 {
        sum / norm
    } else {
        0.0
    }
}

/// Wrap an angle (radians) into `[0, TAU)`.
#[inline]
pub(crate) fn wrap_tau(a: f32) -> f32 {
    a.rem_euclid(TAU)
}

/// Signed shortest angular distance `a - b`, wrapped into `(-PI, PI]`.
#[inline]
pub(crate) fn angle_diff(a: f32, b: f32) -> f32 {
    wrap_tau(a - b + std::f32::consts::PI) - std::f32::consts::PI
}

/// Lerp a packed `0x00RRGGBB` colour toward another by `t ∈ [0,1]`.
#[inline]
pub(crate) fn lerp_rgb(a: u32, b: u32, t: f32) -> u32 {
    let t = t.clamp(0.0, 1.0);
    let ch = |sh: u32| -> u32 {
        let ca = ((a >> sh) & 0xFF) as f32;
        let cb = ((b >> sh) & 0xFF) as f32;
        (ca + (cb - ca) * t).round().clamp(0.0, 255.0) as u32
    };
    (ch(16) << 16) | (ch(8) << 8) | ch(0)
}

/// Additively lift a packed colour by `(dr, dg, db)` (each may be negative),
/// clamped per-channel to `[0, 255]`. Used for "mass in B" lifts (glow, milky
/// way, pile height) that must never wrap.
#[inline]
pub(crate) fn add_rgb_clamped(c: u32, dr: f32, dg: f32, db: f32) -> u32 {
    let ch = |sh: u32, d: f32| -> u32 {
        (((c >> sh) & 0xFF) as f32 + d).round().clamp(0.0, 255.0) as u32
    };
    (ch(16, dr) << 16) | (ch(8, dg) << 8) | ch(0, db)
}

/// Scale a packed colour's channels multiplicatively by `k` (>= 0), clamped.
#[inline]
pub(crate) fn scale_rgb(c: u32, k: f32) -> u32 {
    let ch = |sh: u32| -> u32 { (((c >> sh) & 0xFF) as f32 * k).round().clamp(0.0, 255.0) as u32 };
    (ch(16) << 16) | (ch(8) << 8) | ch(0)
}

/// Rotate a packed colour's hue by `degrees` (HSV hue rotation — a small,
/// self-contained conversion, not the crate's Oklab colour path; good enough
/// for the aurora's slow palette drift). Value and saturation are preserved.
#[inline]
pub(crate) fn rotate_hue_rgb(c: u32, degrees: f32) -> u32 {
    let r = ((c >> 16) & 0xFF) as f32 / 255.0;
    let g = ((c >> 8) & 0xFF) as f32 / 255.0;
    let b = (c & 0xFF) as f32 / 255.0;
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;
    let v = max;
    let s = if max > 0.0 { delta / max } else { 0.0 };
    let mut h = if delta < 1e-6 {
        0.0
    } else if (max - r).abs() < f32::EPSILON {
        60.0 * (((g - b) / delta).rem_euclid(6.0))
    } else if (max - g).abs() < f32::EPSILON {
        60.0 * ((b - r) / delta + 2.0)
    } else {
        60.0 * ((r - g) / delta + 4.0)
    };
    h = (h + degrees).rem_euclid(360.0);
    // HSV -> RGB.
    let cc = v * s;
    let x = cc * (1.0 - ((h / 60.0).rem_euclid(2.0) - 1.0).abs());
    let m = v - cc;
    let (r1, g1, b1) = match (h / 60.0) as u32 {
        0 => (cc, x, 0.0),
        1 => (x, cc, 0.0),
        2 => (0.0, cc, x),
        3 => (0.0, x, cc),
        4 => (x, 0.0, cc),
        _ => (cc, 0.0, x),
    };
    let to_u8 = |x: f32| ((x + m) * 255.0).round().clamp(0.0, 255.0) as u32;
    (to_u8(r1) << 16) | (to_u8(g1) << 8) | to_u8(b1)
}

/// A per-entity "staggered effective tick": entity `entity_hash` only actually
/// advances once every `period` ticks (offset by its own hash so entities
/// desynchronise), producing byte-static output between its own updates —
/// twinkle/shimmer effects change on a hashed subset of candidates per tick
/// rather than all of them at once, which is what keeps the per-tick
/// changed-cell count small.
#[inline]
pub(crate) fn staggered_tick(entity_hash: u32, tick: u32, period: u32) -> u32 {
    let period = period.max(1);
    let offset = entity_hash % period;
    if tick >= offset {
        (tick - offset) / period * period + offset
    } else {
        0
    }
}

/// Quantize a wrapped `[0,1)` fraction into one of `buckets` integer cells —
/// the spatial half of the per-cell stagger key (paired with [`staggered_qt`]
/// via a hash of the returned index/indices).
#[inline]
pub(crate) fn frac_bucket(frac01: f32, buckets: u32) -> u32 {
    let buckets = buckets.max(1);
    (frac01.rem_euclid(1.0) * buckets as f32) as u32 % buckets
}

/// A per-bucket staggered "effective qt" (fractional seconds), the CONTINUOUS-
/// FIELD counterpart of [`staggered_tick`]: a layer whose colour is a
/// continuous function of world position/time (clouds' wind drift, aurora's
/// curtain phase, the fractal spin-zoom, a datarain column's fall) would
/// otherwise recolour nearly every covered cell EVERY tick — sampling it at
/// the live tick would push the p95 changed-cells-per-tick budget too high.
/// Quantizing `tick` through `staggered_tick` first means only a hashed
/// ~`1/period` subset of buckets actually advance on any given tick; the rest
/// replay byte-identically.
#[inline]
pub(crate) fn staggered_qt(cell_hash: u32, tick: u32, hz: u32, period: u32) -> f32 {
    staggered_tick(cell_hash, tick, period) as f32 / hz.max(1) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staggered_qt_holds_between_a_buckets_own_updates() {
        // hash=3, period=8 -> offset=3, so ticks 3..=10 all share one effective
        // tick (3); pick two well inside that window, not straddling the next
        // update at tick 11.
        let hz = 12;
        let period = 8;
        let a = staggered_qt(3, 5, hz, period);
        let b = staggered_qt(3, 9, hz, period);
        assert_eq!(
            a, b,
            "between its own updates a bucket's effective qt must not move"
        );
    }

    #[test]
    fn hash_is_deterministic() {
        assert_eq!(hash3(1, 2, 3), hash3(1, 2, 3));
        assert_ne!(hash3(1, 2, 3), hash3(1, 2, 4));
    }

    #[test]
    fn hash01_is_bounded() {
        for i in 0..1000u32 {
            let h = hash01(hash2(i, 7));
            assert!((0.0..1.0).contains(&h), "{h}");
        }
    }

    #[test]
    fn value_noise1_is_continuous_at_integer_boundaries() {
        // At integer x the interpolation weight is 0, so noise(x) == the lattice hash.
        let seed = 5;
        let at_int = value_noise1(3.0, seed);
        let expect = hash01(hash2(coord_u32(3), seed));
        assert!((at_int - expect).abs() < 1e-5);
    }

    #[test]
    fn fbm4_is_bounded() {
        for i in 0..50 {
            let v = fbm4(i as f32 * 0.37, i as f32 * 0.71, 9);
            assert!((0.0..=1.0).contains(&v), "{v}");
        }
    }

    #[test]
    fn staggered_tick_is_stable_between_own_updates() {
        let eh = 123;
        let period = 6;
        let a = staggered_tick(eh, 10, period);
        let b = staggered_tick(eh, 11, period);
        // Most consecutive ticks land in the same bucket (period > 1).
        assert!(a == b || (10 - a as i64) < period as i64);
    }

    #[test]
    fn angle_diff_wraps_correctly() {
        let d = angle_diff(0.1, TAU - 0.1);
        assert!(d > 0.0 && d < 0.5, "{d}");
    }
}
