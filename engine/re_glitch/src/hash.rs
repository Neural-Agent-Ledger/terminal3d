//! Deterministic hash / value-noise primitives for the glitch passes.
//!
//! Mirrors the FNV-ish integer mix in `re_render::ascii_shader` so the look is
//! reproducible across platforms and golden-test stable. Everything here is a
//! pure function of its integer/seed arguments — no global state, no `rand`.
//!
//! The seed plumbing convention used by every effect:
//! * `seed: u64` — the master per-run seed (fixed; identical every frame).
//! * `seed_off: u64` — an effect-local salt so two effects never share a
//!   decision stream for the same cell.
//! * a *time slice* `tslice(time, rate)` — `time` quantized to an integer step,
//!   folded into the hash so visuals EVOLVE with time without reseeding the RNG.

/// 64→64 bit avalanche mix (SplitMix64 finalizer). Deterministic everywhere.
#[inline]
pub fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    x
}

/// Combine an arbitrary list of `u64` lanes into one well-mixed hash. The first
/// lane seeds the accumulator; each subsequent lane is folded with a rotate so
/// argument ORDER matters (so `(x, y)` ≠ `(y, x)`).
#[inline]
pub fn hash(lanes: &[u64]) -> u64 {
    let mut acc = 0x9E37_79B9_7F4A_7C15;
    for &l in lanes {
        acc = mix64(acc ^ l)
            .rotate_left(17)
            .wrapping_add(0x2545_F491_4F6C_DD1D);
        acc ^= mix64(l);
    }
    mix64(acc)
}

/// A `[0,1)` float from a hash word — top 53 bits for full f64-ish spread, cast
/// to `f32` (we only ever need cell-granular randomness).
#[inline]
pub fn unit(h: u64) -> f32 {
    // 24 mantissa bits is plenty for an f32 in [0,1); take the high bits which
    // are the best-mixed.
    ((h >> 40) as f32) / ((1u64 << 24) as f32)
}

/// A signed `[-1,1)` float from a hash word.
#[inline]
pub fn signed(h: u64) -> f32 {
    unit(h) * 2.0 - 1.0
}

/// Quantize `time` (seconds) into an integer slice advancing at `rate` slices
/// per second. Folding this into a hash makes an effect flicker/evolve at `rate`
/// Hz deterministically — the SAME `time` always yields the SAME slice, so two
/// `apply_glitch` calls at one `time` are identical, while a later `time`
/// advances the look. `rate <= 0` freezes the effect (slice 0 forever).
#[inline]
pub fn tslice(time: f32, rate: f32) -> u64 {
    if rate <= 0.0 {
        return 0;
    }
    // floor toward -inf so negative times (rare) still bucket monotonically.
    (time * rate).floor() as i64 as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_in_range() {
        for i in 0..10_000u64 {
            let u = unit(hash(&[i, i.wrapping_mul(7)]));
            assert!((0.0..1.0).contains(&u), "unit out of range: {u}");
        }
    }

    #[test]
    fn hash_is_order_sensitive_and_deterministic() {
        assert_eq!(hash(&[1, 2, 3]), hash(&[1, 2, 3]), "must be deterministic");
        assert_ne!(hash(&[1, 2]), hash(&[2, 1]), "order must matter");
    }

    #[test]
    fn tslice_advances_with_time_and_freezes_at_zero_rate() {
        assert_eq!(tslice(0.0, 10.0), tslice(0.05, 10.0), "same 0.1s bucket");
        assert_ne!(
            tslice(0.0, 10.0),
            tslice(0.5, 10.0),
            "later time, later slice"
        );
        assert_eq!(tslice(0.0, 0.0), tslice(99.0, 0.0), "rate 0 freezes");
    }
}
