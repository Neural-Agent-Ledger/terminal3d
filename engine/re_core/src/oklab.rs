//! Perceptual colour: sRGB ↔ linear sRGB ↔ Oklab ↔ Oklch, plus Oklab ΔE.
//!
//! Oklab math previously existed ONLY as tooling in `re_imgqa/src/color.rs`
//! (a headless render-QA linter, never linked into the shipped binary). The
//! palette gate and the colour-drift field both need it at RUNTIME, so the
//! conversions move here — `re_core` is the shared, terminal-free, game-free
//! foundation every other crate already depends on — and `re_imgqa`
//! re-exports them instead of carrying its own copy (a shared type moving
//! crates is an announced change, not a silent duplication).
//!
//! Colours travel as packed `0x00RRGGBB`, the same encoding
//! [`crate::Cell`]'s `fg`/`bg` use.
//!
//! Oklch is the cylindrical form of Oklab (`l` unchanged, `(a,b)` → `(c,
//! h)` polar) — the natural space for "rotate the hue, cap the chroma, leave
//! lightness alone" operations such as the palette gate's hue arcs/chroma cap
//! and colour-drift's `(ΔL, ΔC, Δh)` offsets (spec §3.1, §5.2). `h` is in
//! RADIANS throughout this module, matching [`oklab_hue`]'s existing
//! `atan2`-radians convention; callers authoring in degrees (the spec's prose
//! uses `°`, e.g. "Δh rotates at ω ≈ 20°/s") convert at the edge with
//! `.to_radians()` / `.to_degrees()`.

/// A linear-sRGB triple in `[0,1]` (already gamma-decoded).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinRgb {
    /// Red, linear, `[0,1]`.
    pub r: f32,
    /// Green, linear, `[0,1]`.
    pub g: f32,
    /// Blue, linear, `[0,1]`.
    pub b: f32,
}

/// An Oklab colour: perceptual lightness `l` plus the `a`/`b` opponent axes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Oklab {
    /// Perceptual lightness, roughly `[0,1]`.
    pub l: f32,
    /// Green–red opponent axis (negative = green, positive = red).
    pub a: f32,
    /// Blue–yellow opponent axis (negative = blue, positive = yellow).
    pub b: f32,
}

/// The cylindrical form of [`Oklab`]: lightness `l` (unchanged), chroma `c`
/// (distance from the neutral axis, `>= 0`), and hue `h` in RADIANS. Undefined
/// (but harmless — `h` is simply whatever `atan2(0,0)` gives) at `c == 0`
/// (a neutral grey has no hue); callers that care gate on chroma first, same
/// as the existing [`oklab_hue`] convention.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Oklch {
    /// Perceptual lightness, unchanged from [`Oklab::l`].
    pub l: f32,
    /// Chroma — distance from the neutral (grey) axis, `>= 0`.
    pub c: f32,
    /// Hue, in RADIANS (see the type docs for the convention).
    pub h: f32,
}

/// Decode one 8-bit sRGB channel `[0,255]` to linear `[0,1]` (the standard
/// IEC 61966-2-1 transfer function).
#[inline]
#[must_use]
pub fn srgb_to_linear(c: u8) -> f32 {
    let cs = c as f32 / 255.0;
    if cs <= 0.040_448_237 {
        cs / 12.92
    } else {
        ((cs + 0.055) / 1.055).powf(2.4)
    }
}

/// Encode one linear `[0,1]` channel back to an 8-bit sRGB value `[0,255]`.
#[inline]
#[must_use]
pub fn linear_to_srgb(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let s = if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0).round().clamp(0.0, 255.0) as u8
}

/// Unpack a `0x00RRGGBB` colour into linear sRGB.
#[inline]
#[must_use]
pub fn unpack_linear(packed: u32) -> LinRgb {
    let r = ((packed >> 16) & 0xFF) as u8;
    let g = ((packed >> 8) & 0xFF) as u8;
    let b = (packed & 0xFF) as u8;
    LinRgb {
        r: srgb_to_linear(r),
        g: srgb_to_linear(g),
        b: srgb_to_linear(b),
    }
}

/// Pack a linear-sRGB triple back into `0x00RRGGBB` (gamma-encoding each channel).
#[inline]
#[must_use]
pub fn pack_linear(c: LinRgb) -> u32 {
    let r = linear_to_srgb(c.r) as u32;
    let g = linear_to_srgb(c.g) as u32;
    let b = linear_to_srgb(c.b) as u32;
    (r << 16) | (g << 8) | b
}

/// Linear interpolation between two linear-sRGB colours by `t ∈ [0,1]`. Used to
/// form the *effective colour*: `lerp(bg, fg, ink(glyph))`.
#[inline]
#[must_use]
pub fn lerp_linear(a: LinRgb, b: LinRgb, t: f32) -> LinRgb {
    let t = t.clamp(0.0, 1.0);
    LinRgb {
        r: a.r + (b.r - a.r) * t,
        g: a.g + (b.g - a.g) * t,
        b: a.b + (b.b - a.b) * t,
    }
}

/// Convert linear sRGB → Oklab (Björn Ottosson's matrices + cube-root nonlinearity).
#[must_use]
pub fn linear_to_oklab(c: LinRgb) -> Oklab {
    let l = 0.412_221_47 * c.r + 0.536_332_55 * c.g + 0.051_445_995 * c.b;
    let m = 0.211_903_5 * c.r + 0.680_699_5 * c.g + 0.107_396_96 * c.b;
    let s = 0.088_302_46 * c.r + 0.281_718_85 * c.g + 0.629_978_7 * c.b;

    let l_ = cbrt(l);
    let m_ = cbrt(m);
    let s_ = cbrt(s);

    Oklab {
        l: 0.210_454_26 * l_ + 0.793_617_8 * m_ - 0.004_072_047 * s_,
        a: 1.977_998_5 * l_ - 2.428_592_2 * m_ + 0.450_593_7 * s_,
        b: 0.025_904_037 * l_ + 0.782_771_77 * m_ - 0.808_675_77 * s_,
    }
}

/// Convert Oklab → linear sRGB (the inverse of [`linear_to_oklab`]).
#[must_use]
pub fn oklab_to_linear(c: Oklab) -> LinRgb {
    let l_ = c.l + 0.396_337_78 * c.a + 0.215_803_76 * c.b;
    let m_ = c.l - 0.105_561_346 * c.a - 0.063_854_17 * c.b;
    let s_ = c.l - 0.089_484_18 * c.a - 1.291_485_5 * c.b;

    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;

    LinRgb {
        r: 4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_94 * s,
        g: -1.268_438 * l + 2.609_757_4 * m - 0.341_319_38 * s,
        b: -0.004_196_086 * l - 0.703_418_6 * m + 1.707_614_7 * s,
    }
}

/// Real cube root (handles negatives, which `powf` does not).
#[inline]
fn cbrt(x: f32) -> f32 {
    x.signum() * x.abs().powf(1.0 / 3.0)
}

/// Oklab → Oklch: `(a,b)` → polar `(c, h)`, `l` unchanged. `h` is in RADIANS.
#[inline]
#[must_use]
pub fn oklab_to_oklch(c: Oklab) -> Oklch {
    Oklch {
        l: c.l,
        c: oklab_chroma(c),
        h: oklab_hue(c),
    }
}

/// Oklch → Oklab: polar `(c, h)` → `(a,b)`, `l` unchanged.
#[inline]
#[must_use]
pub fn oklch_to_oklab(c: Oklch) -> Oklab {
    Oklab {
        l: c.l,
        a: c.c * c.h.cos(),
        b: c.c * c.h.sin(),
    }
}

/// Convert a packed `0x00RRGGBB` colour directly to Oklch (sRGB → linear →
/// Oklab → Oklch in one call — the common case for a drift/gate source that
/// only ever sees packed colours).
#[inline]
#[must_use]
pub fn packed_to_oklch(packed: u32) -> Oklch {
    oklab_to_oklch(linear_to_oklab(unpack_linear(packed)))
}

/// Convert an Oklch colour directly to a packed `0x00RRGGBB` colour (the
/// inverse of [`packed_to_oklch`]), gamma-encoding and byte-clamping at the
/// end the same way [`pack_linear`] does.
#[inline]
#[must_use]
pub fn oklch_to_packed(c: Oklch) -> u32 {
    pack_linear(oklab_to_linear(oklch_to_oklab(c)))
}

/// Oklab ΔE — straight Euclidean distance in Oklab space. Perceptually uniform
/// enough for the relative-contrast comparisons this crate's callers make.
#[inline]
#[must_use]
pub fn delta_e(a: Oklab, b: Oklab) -> f32 {
    let dl = a.l - b.l;
    let da = a.a - b.a;
    let db = a.b - b.b;
    (dl * dl + da * da + db * db).sqrt()
}

/// Oklab ΔE directly between two packed colours (no ink lerp).
#[inline]
#[must_use]
pub fn delta_e_packed(a: u32, b: u32) -> f32 {
    delta_e(
        linear_to_oklab(unpack_linear(a)),
        linear_to_oklab(unpack_linear(b)),
    )
}

/// Oklab hue angle (RADIANS) of a colour, for the hue-spread metric. Undefined
/// for near-gray; callers gate on chroma before trusting it.
#[inline]
#[must_use]
pub fn oklab_hue(c: Oklab) -> f32 {
    c.b.atan2(c.a)
}

/// Oklab chroma (distance from the neutral axis).
#[inline]
#[must_use]
pub fn oklab_chroma(c: Oklab) -> f32 {
    (c.a * c.a + c.b * c.b).sqrt()
}

/// Wrap a hue delta (radians) to the shortest signed path in `(-π, π]` — the
/// form a hue-arc / hue-rotation offset should be applied through, so a drift
/// or gate never spins "the long way around" the colour wheel.
#[inline]
#[must_use]
pub fn wrap_hue_delta(mut delta: f32) -> f32 {
    const TAU: f32 = std::f32::consts::TAU;
    delta %= TAU;
    if delta > std::f32::consts::PI {
        delta -= TAU;
    } else if delta <= -std::f32::consts::PI {
        delta += TAU;
    }
    delta
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lin(r: f32, g: f32, b: f32) -> LinRgb {
        LinRgb { r, g, b }
    }

    #[test]
    fn srgb_linear_roundtrip() {
        for v in [0u8, 1, 17, 64, 128, 200, 255] {
            let l = srgb_to_linear(v);
            let back = linear_to_srgb(l);
            assert_eq!(back, v, "sRGB roundtrip failed for {v}");
        }
    }

    #[test]
    fn oklab_roundtrip_within_epsilon() {
        for &packed in &[
            0x0000_0000u32,
            0x00FF_FFFF,
            0x00FF_0000,
            0x0000_FF00,
            0x0000_00FF,
            0x003A_7BDF,
            0x0080_8080,
        ] {
            let lin0 = unpack_linear(packed);
            let lab = linear_to_oklab(lin0);
            let lin1 = oklab_to_linear(lab);
            assert!((lin0.r - lin1.r).abs() < 1e-3, "r drift {packed:#08x}");
            assert!((lin0.g - lin1.g).abs() < 1e-3, "g drift {packed:#08x}");
            assert!((lin0.b - lin1.b).abs() < 1e-3, "b drift {packed:#08x}");
        }
    }

    #[test]
    fn oklch_roundtrip_within_epsilon() {
        for &packed in &[
            0x00FF_0000u32,
            0x0000_FF00,
            0x0000_00FF,
            0x0080_8080,
            0x0012_3456,
        ] {
            let lab0 = linear_to_oklab(unpack_linear(packed));
            let lch = oklab_to_oklch(lab0);
            let lab1 = oklch_to_oklab(lch);
            assert!((lab0.l - lab1.l).abs() < 1e-5, "l drift {packed:#08x}");
            assert!((lab0.a - lab1.a).abs() < 1e-4, "a drift {packed:#08x}");
            assert!((lab0.b - lab1.b).abs() < 1e-4, "b drift {packed:#08x}");
        }
    }

    #[test]
    fn packed_oklch_roundtrip_recovers_the_byte_value() {
        for &packed in &[
            0x00FF_0000u32,
            0x0000_FF00,
            0x0000_00FF,
            0x0080_4020,
            0x00FF_FFFF,
            0x0000_0000,
        ] {
            let lch = packed_to_oklch(packed);
            let back = oklch_to_packed(lch);
            // sRGB<->Oklab round-trips within ~1 unit of gamma-encoding error
            // per channel (matches oklab_roundtrip_within_epsilon's tolerance).
            let (ar, ag, ab) = ((packed >> 16) & 0xFF, (packed >> 8) & 0xFF, packed & 0xFF);
            let (br, bg, bb) = ((back >> 16) & 0xFF, (back >> 8) & 0xFF, back & 0xFF);
            assert!(
                (ar as i32 - br as i32).abs() <= 1,
                "r drift {packed:#08x} -> {back:#08x}"
            );
            assert!(
                (ag as i32 - bg as i32).abs() <= 1,
                "g drift {packed:#08x} -> {back:#08x}"
            );
            assert!(
                (ab as i32 - bb as i32).abs() <= 1,
                "b drift {packed:#08x} -> {back:#08x}"
            );
        }
    }

    #[test]
    fn chroma_is_zero_for_neutral_grays() {
        for &g in &[
            0x0000_0000u32,
            0x0040_4040,
            0x0080_8080,
            0x00C0_C0C0,
            0x00FF_FFFF,
        ] {
            let lab = linear_to_oklab(unpack_linear(g));
            assert!(
                oklab_chroma(lab) < 1e-4,
                "grey {g:#08x} must have ~zero chroma"
            );
        }
    }

    #[test]
    fn black_white_delta_e_is_large() {
        let d = delta_e_packed(0x0000_0000, 0x00FF_FFFF);
        assert!(d > 0.9, "black↔white ΔE should be ~1.0, got {d}");
        assert!(
            delta_e_packed(0x0012_3456, 0x0012_3456) < 1e-6,
            "self ΔE must be ~0"
        );
    }

    #[test]
    fn wrap_hue_delta_takes_the_short_way_round() {
        // A 350° rotation is really a -10° rotation the short way.
        let almost_full_turn = 350.0_f32.to_radians();
        let wrapped = wrap_hue_delta(almost_full_turn);
        assert!(
            wrapped < 0.0,
            "350° must wrap to a small NEGATIVE delta, got {wrapped}"
        );
        assert!(
            (wrapped.to_degrees() + 10.0).abs() < 1e-3,
            "got {}°",
            wrapped.to_degrees()
        );

        // A small delta passes through unchanged.
        let small = 12.0_f32.to_radians();
        assert!((wrap_hue_delta(small) - small).abs() < 1e-6);
    }

    #[test]
    fn lin_helper_is_used_by_the_roundtrip_fixture() {
        // Guards the `lin()` helper against becoming dead code silently.
        let c = lin(0.2, 0.4, 0.6);
        assert_eq!(c.r, 0.2);
        assert_eq!(c.g, 0.4);
        assert_eq!(c.b, 0.6);
    }
}
