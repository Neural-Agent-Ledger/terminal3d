//! Packed-color helpers shared by every post-process pass. Colors are packed
//! `0x00RRGGBB`, matching the [`Cell`](re_core::Cell) `fg`/`bg` contract.
//!
//! Everything here is a pure, branch-light `f32`/`u8` function — no global
//! state — so the passes stay deterministic and golden-test stable.

/// Unpack a packed `0x00RRGGBB` color into `(r, g, b)` byte components.
#[inline]
pub fn unpack(c: u32) -> (u8, u8, u8) {
    (
        ((c >> 16) & 0xFF) as u8,
        ((c >> 8) & 0xFF) as u8,
        (c & 0xFF) as u8,
    )
}

/// Pack `(r, g, b)` byte components into a `0x00RRGGBB` color.
#[inline]
pub fn pack(r: u8, g: u8, b: u8) -> u32 {
    ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
}

/// Unpack into a linear-ish `[0,1]` float triple (we treat the stored bytes as
/// the working color space — no gamma round-trip, which keeps the passes cheap
/// and reversible for tests).
#[inline]
pub fn unpack_f(c: u32) -> [f32; 3] {
    let (r, g, b) = unpack(c);
    [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0]
}

/// Pack a float triple back to `0x00RRGGBB`, clamping each channel to `[0,1]`.
#[inline]
pub fn pack_f(c: [f32; 3]) -> u32 {
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    pack(q(c[0]), q(c[1]), q(c[2]))
}

/// Perceptual luma in `[0,1]` of a packed color (Rec. 601 weights). Used by the
/// bloom threshold and a couple of grades.
#[inline]
pub fn luma(c: u32) -> f32 {
    let [r, g, b] = unpack_f(c);
    0.299 * r + 0.587 * g + 0.114 * b
}

/// Scale every channel of a packed color by `k` (clamped). `k < 1` darkens,
/// `k > 1` brightens. Used by SSAO, vignette and bloom.
#[inline]
pub fn scale(c: u32, k: f32) -> u32 {
    let [r, g, b] = unpack_f(c);
    pack_f([r * k, g * k, b * k])
}

/// Per-channel additive `a + b` (clamped) — the bloom accumulation operator.
#[inline]
pub fn add(a: u32, b: u32) -> u32 {
    let fa = unpack_f(a);
    let fb = unpack_f(b);
    pack_f([fa[0] + fb[0], fa[1] + fb[1], fa[2] + fb[2]])
}
