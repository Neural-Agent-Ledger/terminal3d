//! Deterministic seeded hashing shared by every effect in this crate.
//!
//! Same splitmix64-style mix as [`crate::overgrowth`]'s private
//! `hash64`, generalized to a free `salt: u64` third argument instead of that
//! module's small const-enum of salts — every effect here needs MANY
//! independent random draws per instance (size, gaze, hue, and one distinct
//! frequency per oscillating part), so a closed enum of salts would just be
//! re-invented per effect. There is still exactly one rule: two different
//! logical random draws must use two different `salt` values, so that "should
//! this blink now" and "which way should the eye look" never accidentally
//! read the same hash stream and correlate.
//!
//! No wall clock, no `rand` crate, no thread-local RNG anywhere in this
//! module or any of its callers — every "random" number is a pure function of
//! its integer inputs, which is what lets every effect's `paint` be a pure
//! function of `(self, t)` and every test below assert exact equality instead
//! of a statistical property.

/// A splitmix64-style avalanche mix of three integers. Same inputs, same
/// output, on any machine, forever.
#[inline]
#[must_use]
pub fn hash64(a: u64, b: u64, c: u64) -> u64 {
    let mut x = a ^ b.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ c.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    x
}

/// A value in `[0, 1)`, deterministic in `(a, b, c)`. Top 53 bits of the mix
/// feed an `f64` mantissa's worth of precision — the standard
/// splitmix-to-uniform-float trick.
#[inline]
#[must_use]
pub fn hash_unit(a: u64, b: u64, c: u64) -> f64 {
    ((hash64(a, b, c) >> 11) as f64) * (1.0 / (1u64 << 53) as f64)
}

/// A value in `[-1, 1)`, deterministic in `(a, b, c)`.
#[inline]
#[must_use]
pub fn hash_signed(a: u64, b: u64, c: u64) -> f64 {
    hash_unit(a, b, c) * 2.0 - 1.0
}

/// Uniform-`f32` convenience over [`hash_unit`] scaled into `[lo, hi)` —
/// what every effect's `random()` constructor calls for a size / hue /
/// frequency / phase parameter. `id` is usually the effect's own seed-derived
/// per-instance id (`0` for a singleton effect; a per-particle/per-eye index
/// for a swarm), `salt` names WHICH parameter this draw is for.
#[inline]
#[must_use]
pub fn range(seed: u64, id: u64, salt: u64, lo: f32, hi: f32) -> f32 {
    lo + (hash_unit(seed, id, salt) as f32) * (hi - lo)
}

/// Uniform-`f32` in `[0, 2π)` — a recurring need (initial phase / heading /
/// hue-angle offset), spelled out once so callers don't re-derive `2.0 *
/// PI` at every call site.
#[inline]
#[must_use]
pub fn angle(seed: u64, id: u64, salt: u64) -> f32 {
    range(seed, id, salt, 0.0, std::f32::consts::TAU)
}

/// A boolean draw with probability `p` of `true`.
#[inline]
#[must_use]
pub fn chance(seed: u64, id: u64, salt: u64, p: f32) -> bool {
    (hash_unit(seed, id, salt) as f32) < p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_unit_stays_in_its_documented_range() {
        for i in 0..2000u64 {
            let u = hash_unit(7, i, 3);
            assert!((0.0..1.0).contains(&u), "hash_unit out of [0,1): {u}");
        }
    }

    #[test]
    fn same_inputs_same_output() {
        assert_eq!(hash64(1, 2, 3), hash64(1, 2, 3));
        assert_eq!(range(9, 4, 2, -1.0, 1.0), range(9, 4, 2, -1.0, 1.0));
    }

    #[test]
    fn different_salts_decorrelate() {
        // Not a statistical proof, just the property every call site relies
        // on: asking two different questions of the same (seed, id) does not
        // silently reuse the same stream.
        let a = hash_unit(1, 1, 10);
        let b = hash_unit(1, 1, 11);
        assert_ne!(a, b);
    }

    #[test]
    fn range_respects_bounds() {
        for i in 0..500u64 {
            let v = range(42, i, 5, 3.0, 9.0);
            assert!((3.0..9.0).contains(&v), "range() escaped bounds: {v}");
        }
    }
}
