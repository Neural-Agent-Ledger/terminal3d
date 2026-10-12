//! PERCEPTUAL LEGIBILITY FLOOR — "very dim", never "invisible".
//!
//! ## The defect this exists for
//!
//! The relief lighting in [`crate::ascii_shader`] is a plain multiply:
//! `fg = albedo * lit` and `bg = dim_rgb(material_bg, shadow)`. With a murk level's
//! ambient (L1 `[atmosphere] ambient = 0.07`) and an authored near-black palette
//! (`pulse_wall` bg `#160505`, floor bg `#08111E`, ceiling bg `#050608`) the two
//! darknesses COMPOUND: away from a lamp the cell's bytes land around `(2..24)`,
//! then fog mixes toward an equally near-black fog colour (`#100406`). A `--dump`
//! still shows a dense glyph grid, but `--dump --ansi` shows fg and bg both under
//! ~24/255 over ~40 % of the L1 spawn view — on a real screen that is solid black.
//! The mood the author wanted was *dark*; what shipped was *absent*.
//!
//! ## The rule
//!
//! Applied once in `Material::apply_lighting` — right after distance/height fog
//! (step 4a, before the opt-in depth-contrast policy), or for a SKY/ceiling recipe
//! right after its depth band (step 5a) — to every LIT cell except the exempt
//! recipes below:
//!
//! 1. **Ground floor (bg).** The bg's Oklab lightness `L` is soft-lifted:
//!    `L' = L + F·(1 − L/K)²` for `L < K`, identity for `L ≥ K`
//!    ([`LEGIBILITY_BG_L_FLOOR`] = F, [`LEGIBILITY_BG_L_KNEE`] = K). With `K ≥ 2F`
//!    the map is STRICTLY MONOTONIC, so every relative darkness (crevice < face,
//!    far < near, shadow < lit) survives — the darkest ground becomes `F`, not
//!    grey-flat. Above the knee (a brightly lit ground) it is the exact identity.
//! 2. **Ink contrast (fg).** On a dark ground (lifted `bg L ≤`
//!    [`LEGIBILITY_FG_ACTIVE_MAX_BG_L`]) whose ink is the LIGHTER channel and does
//!    not already clear the lifted ground by [`LEGIBILITY_FG_IDENTITY_DL`], the ink's
//!    `L` is remapped linearly onto `[bg L' + MIN_INK_DL, bg L' + IDENTITY_DL]`
//!    ([`LEGIBILITY_MIN_INK_DL`]) — every glyph gets a perceptible step over its own
//!    ground, darker ink stays darker, and bright (lamp-lit) ink is untouched.
//!    Dark-ink-on-light cells (fg darker than bg by authorship) are left alone.
//! 3. **Hue.** The lift is ADDED along the material's own authored colour direction
//!    (its `(a, b)` per unit `L`, from the pre-lighting base texel), capped at
//!    [`LEGIBILITY_MAX_TINT_RATIO`] chroma per unit lightness. A red-brick wall's
//!    shadow becomes a dim maroon, a steel floor a dim navy — never neutral grey,
//!    never neon. An authored pure-black material lifts neutrally.
//!
//!    A near-neutral authored ground (chroma/L below
//!    [`LEGIBILITY_NEUTRAL_TINT_RATIO`]) borrows its ink's hue instead.
//!
//! Exempt: TEXT and VOID self-lit recipes (code wall, note placard, hologram — the
//! descriptor's `self_lit && (is_code_wall || transparent)`), whose black is
//! authored meaning, and cells shaded without a `LightCtx` (focused unit tests).
//! Self-lit PANEL recipes (scanline/`metal`, truchet, fractal) are NOT exempt: they
//! are used as ceilings and walls, and their near-black ground is the same defect.
//! Pure, allocation-free, deterministic.

use re_core::oklab::{linear_to_oklab, oklab_to_linear, pack_linear, unpack_linear, Oklab};

/// Oklab `L` the darkest possible ground (bg) is lifted to.
///
/// `L = 0.22` is a
/// ~`#1C1C1C`-class neutral (≈ 26/255), a touch brighter on a tinted material: the
/// dimmest thing on screen, but distinguishable from the terminal's own black.
pub const LEGIBILITY_BG_L_FLOOR: f32 = 0.22;

/// Oklab `L` at and above which the bg floor is the exact identity.
///
/// Must be
/// `≥ 2 × LEGIBILITY_BG_L_FLOOR` for the soft lift to stay monotonic. Set to
/// ~2.8 × the floor rather than the minimum 2 × so the slope at black stays ≈ 0.3:
/// the murk keeps a visible ground gradient (a lamp's dim ground stays brighter
/// than the unlit ground beside it) instead of collapsing to one flat grey.
pub const LEGIBILITY_BG_L_KNEE: f32 = 0.62;

/// Minimum Oklab `ΔL` of ink over its own (lifted) dark ground.
///
/// `0.14` reads as a
/// dim glyph on a dim ground (≈ 55/255 ink on ≈ 26/255 ground at the very bottom).
pub const LEGIBILITY_MIN_INK_DL: f32 = 0.14;

/// The ink floor only engages on a DARK ground: lifted `bg L` at or below this.
///
/// Above it the ground itself is legible and the authored ink contrast stands.
pub const LEGIBILITY_FG_ACTIVE_MAX_BG_L: f32 = 0.36;

/// Ink that already clears its (lifted) ground by this `ΔL` is left EXACTLY as shaded.
///
/// This is what keeps lamp-lit glyphs byte-identical. Below it, the ink's
/// original `L` range `[bg L, bg L' + span]` is remapped linearly (monotonic, so
/// the relief ramp's relative ink darkness survives) onto
/// `[bg L' + MIN_INK_DL, bg L' + span]`.
pub const LEGIBILITY_FG_IDENTITY_DL: f32 = 0.28;

/// Cap on the tint added per unit of lifted lightness (Oklab chroma / L).
///
/// Keeps a
/// saturated authored colour from turning its shadow into a neon glow.
pub const LEGIBILITY_MAX_TINT_RATIO: f32 = 0.45;

/// An authored ground whose chroma per unit `L` is below this reads as NEUTRAL.
///
/// (`#050608`, `#020304` — "near-black slate".) Its lift borrows the tint of the
/// material's authored INK instead, so a slate ceiling lifts to dim slate rather
/// than to flat charcoal.
pub const LEGIBILITY_NEUTRAL_TINT_RATIO: f32 = 0.06;

/// Monotonic soft floor: `L + F·(1 − L/K)²` below the knee `K`, identity above.
/// Returns `F` at `L = 0`; requires `K ≥ 2F` for monotonicity (asserted in tests).
#[inline]
#[must_use]
pub fn soft_floor(l: f32, floor: f32, knee: f32) -> f32 {
    if l >= knee || knee <= 0.0 {
        return l;
    }
    let t = 1.0 - l.max(0.0) / knee;
    l.max(0.0) + floor * t * t
}

#[inline]
fn to_lab(packed: u32) -> Oklab {
    linear_to_oklab(unpack_linear(packed))
}

/// The authored hue direction of `hue_src` as `(a/L, b/L)` (uncapped).
#[inline]
fn tint_ratio(hue_src: Oklab) -> (f32, f32) {
    let l = hue_src.l.max(0.05);
    (hue_src.a / l, hue_src.b / l)
}

/// The authored hue direction of `hue_src` as `(a/L, b/L)`, chroma-ratio capped.
#[inline]
fn tint_dir(hue_src: Oklab) -> (f32, f32) {
    let (mut ra, mut rb) = tint_ratio(hue_src);
    let r = ra.hypot(rb);
    if r > LEGIBILITY_MAX_TINT_RATIO {
        let s = LEGIBILITY_MAX_TINT_RATIO / r;
        ra *= s;
        rb *= s;
    }
    (ra, rb)
}

/// Raise `c` to lightness `new_l` along the `dir` tint; returns `c` untouched
/// (bit-exact) when there is nothing to lift.
#[inline]
fn lift(packed: u32, c: Oklab, new_l: f32, dir: (f32, f32)) -> u32 {
    let dl = new_l - c.l;
    if dl <= 1e-4 {
        return packed;
    }
    pack_linear(oklab_to_linear(Oklab {
        l: new_l,
        a: c.a + dl * dir.0,
        b: c.b + dl * dir.1,
    }))
}

/// Apply the legibility floor to a shaded cell's `(fg, bg)`.
///
/// `fg_hue` / `bg_hue`
/// are the material's AUTHORED (pre-lighting) colours, used only for the tint
/// direction of the lift. Returns the new `(fg, bg)`; a cell already legible (bg
/// above the knee, ink already clear of its ground) comes back byte-identical.
#[must_use]
pub fn legible_pair(fg: u32, bg: u32, fg_hue: u32, bg_hue: u32) -> (u32, u32) {
    let bg_lab = to_lab(bg);
    let fg_lab = to_lab(fg);

    // 1. Ground floor.
    let bg_l = soft_floor(bg_lab.l, LEGIBILITY_BG_L_FLOOR, LEGIBILITY_BG_L_KNEE);
    let bg_src = to_lab(bg_hue);
    let (ra, rb) = tint_ratio(bg_src);
    let bg_tint = if ra.hypot(rb) >= LEGIBILITY_NEUTRAL_TINT_RATIO {
        tint_dir(bg_src)
    } else {
        tint_dir(to_lab(fg_hue))
    };
    let ground = lift(bg, bg_lab, bg_l, bg_tint);

    // 2. Ink contrast — only on a dark ground, only for light-ink-on-dark cells,
    //    only when the ink does not already clear its lifted ground.
    let mut ink = fg;
    let hi = bg_l + LEGIBILITY_FG_IDENTITY_DL;
    if bg_l <= LEGIBILITY_FG_ACTIVE_MAX_BG_L && fg_lab.l >= bg_lab.l && fg_lab.l < hi {
        let lo = bg_lab.l;
        let t = ((fg_lab.l - lo) / (hi - lo).max(1e-4)).clamp(0.0, 1.0);
        let fg_l =
            bg_l + LEGIBILITY_MIN_INK_DL + t * (LEGIBILITY_FG_IDENTITY_DL - LEGIBILITY_MIN_INK_DL);
        ink = lift(fg, fg_lab, fg_l, tint_dir(to_lab(fg_hue)));
    }
    (ink, ground)
}

/// [`legible_pair`] for a cell that authored POINT LIGHTS reach: floor the cell as
/// it would be WITHOUT the lamps, then add the lamps' contribution back on top.
///
/// `(fg, bg)` is the fully lit cell and `unlamped` the same cell shaded with no
/// point lights (every other term identical). The floor is a lift toward a dim
/// minimum, so it compresses exactly the dark range a lamp pool rises out of:
/// flooring the LIT pair lifted the unlit room almost as high as the pool, and the
/// ink rule's `bg L' ≤` [`LEGIBILITY_FG_ACTIVE_MAX_BG_L`] gate switched the ink
/// lift OFF where a lamp raised the ground — so on the `22-lit-cellar` demo the
/// lamps made the frame DARKER (mean ink 70.2 lit vs 88.5 unlit, /255). Here the
/// murk is floored once, and the lamp's own light rides on top at full strength,
/// per channel, in the shader's additive byte space:
/// `out = floor(unlamped) + max(lit − unlamped, 0)`. So a lamp can only brighten a
/// cell — by at least the amount it brightened it before the floor — and a cell no
/// lamp reaches (`unlamped == lit`) is exactly [`legible_pair`].
///
/// One exception, for the INK only: a lamp adds more to the ground than to the ink
/// (`LAMP_BG_GAIN`), so inside a pool the lit ink can end up DARKER than its lit
/// ground — dark relief strokes on a lamp-lit wall. That cell is already legible
/// by inversion, and lifting its ink by the murk rule (which makes ink LIGHTER than
/// the ground) would walk it back across its ground and fade the strokes (L1's
/// lamp-lit `pulse_wall` relief went from `#070102` strokes on a `#4E1C1D` ground to
/// a near-invisible `#4x` ink on a `#5D1F23` one). On an `inked` cell whose LIT ink
/// is darker than its lit ground, the lit ink itself is kept when it stands further
/// off the (floored, lamp-lit) ground than the floored ink does — exactly what
/// [`legible_pair`] does for dark-ink-on-light. (Lifting the dark ink by the
/// ground's byte delta instead keeps the byte gap but crushes the contrast ratio.)
/// A blank cell (`inked == false`) shows no ink, so it keeps the plain rule and its
/// fg stays monotonic in the lamp.
#[must_use]
pub fn legible_pair_lamped(
    fg: u32,
    bg: u32,
    unlamped: (u32, u32),
    fg_hue: u32,
    bg_hue: u32,
    inked: bool,
) -> (u32, u32) {
    let (murk_ink, murk_ground) = unlamped;
    let (floored_ink, floored_ground) = legible_pair(murk_ink, murk_ground, fg_hue, bg_hue);
    let ground = add_lamp_delta(floored_ground, bg, murk_ground);
    let lifted = add_lamp_delta(floored_ink, fg, murk_ink);
    let ink = if inked && to_lab(fg).l < to_lab(bg).l {
        // Keep whichever ink stands further off the ground: the lit pair's own
        // dark strokes, or the floored ink riding the lamp.
        let g = to_lab(ground).l;
        if g - to_lab(fg).l >= (to_lab(lifted).l - g).abs() {
            fg
        } else {
            lifted
        }
    } else {
        lifted
    };
    (ink, ground)
}

/// `base + max(lit − dark, 0)` per 8-bit channel, saturating (packed `0xRRGGBB`).
#[inline]
fn add_lamp_delta(base: u32, lit: u32, dark: u32) -> u32 {
    let ch = |c: u32, s: u32| (c >> s) & 0xFF;
    [16u32, 8, 0].iter().fold(0u32, |acc, &s| {
        let delta = ch(lit, s).saturating_sub(ch(dark, s));
        acc | ((ch(base, s) + delta).min(0xFF) << s)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn l_of(c: u32) -> f32 {
        to_lab(c).l
    }
    fn max_ch(c: u32) -> u32 {
        ((c >> 16) & 0xFF).max((c >> 8) & 0xFF).max(c & 0xFF)
    }

    // Compile-time: the knee must sit at least 2x the floor (a constant condition,
    // so it is checked by the compiler instead of a runtime `assert!`).
    const _: () = assert!(LEGIBILITY_BG_L_KNEE >= 2.0 * LEGIBILITY_BG_L_FLOOR);

    #[test]
    fn knees_keep_the_soft_floor_monotonic() {
        for (f, k) in [(LEGIBILITY_BG_L_FLOOR, LEGIBILITY_BG_L_KNEE), (0.36, 0.72)] {
            let mut prev = -1.0;
            for i in 0..=1000 {
                let l = i as f32 / 1000.0;
                let v = soft_floor(l, f, k);
                assert!(
                    v > prev,
                    "soft_floor not strictly increasing at L={l} (F={f} K={k})"
                );
                assert!(v >= l && v >= f - 1e-6, "floor violated at L={l}");
                prev = v;
            }
            assert_eq!(soft_floor(k, f, k), k, "identity at the knee");
        }
    }

    #[test]
    fn pure_black_is_lifted_off_black_but_stays_dim() {
        let (fg, bg) = legible_pair(0x000000, 0x000000, 0x000000, 0x000000);
        assert!(max_ch(bg) >= 24, "bg must leave the black band: {bg:06X}");
        assert!(max_ch(bg) <= 40, "bg must stay dim: {bg:06X}");
        assert!(l_of(fg) - l_of(bg) >= LEGIBILITY_MIN_INK_DL - 0.01);
        assert!(max_ch(fg) <= 80, "ink must stay dim: {fg:06X}");
    }

    #[test]
    fn l1_murk_wall_keeps_its_red_hue() {
        // L1 pulse_wall authored fg #6A2424 / bg #160505, shaded down near black.
        let (fg, bg) = legible_pair(0x070202, 0x030101, 0x6A2424, 0x160505);
        let (r, g, b) = ((bg >> 16) & 0xFF, (bg >> 8) & 0xFF, bg & 0xFF);
        assert!(
            r >= 24 && r > g && r > b,
            "bg should be a dim maroon: {bg:06X}"
        );
        let (fr, fgc, fb) = ((fg >> 16) & 0xFF, (fg >> 8) & 0xFF, fg & 0xFF);
        assert!(fr > fgc && fr > fb, "ink should stay red: {fg:06X}");
    }

    #[test]
    fn neutral_ground_borrows_the_ink_hue() {
        // L1 pulse_ceiling: bg #050608 (neutral), fg #3A4658 (slate blue).
        let (_, bg) = legible_pair(0x010101, 0x010101, 0x3A4658, 0x050608);
        assert!(
            (bg & 0xFF) > ((bg >> 16) & 0xFF),
            "lifted ceiling should lean slate-blue: {bg:06X}"
        );
    }

    #[test]
    fn relative_darkness_survives() {
        let (_, a) = legible_pair(0x000000, 0x020304, 0x3A4658, 0x050608);
        let (_, b) = legible_pair(0x000000, 0x0A0C10, 0x3A4658, 0x050608);
        assert!(
            l_of(a) < l_of(b),
            "a darker ground must stay darker after the lift"
        );
    }

    #[test]
    fn lamp_lit_cells_are_byte_identical() {
        // Bright lamp-pool grounds (above the bg knee) with bright ink.
        for (fg, bg) in [
            (0xFFD0D0u32, 0xD07878u32),
            (0xF0F4FC, 0x98A8C0),
            (0xF0F0F0, 0x9A9A9A),
        ] {
            assert!(l_of(bg) >= LEGIBILITY_BG_L_KNEE, "{bg:06X} L={}", l_of(bg));
            assert_eq!(legible_pair(fg, bg, 0x6A2424, 0x160505), (fg, bg));
        }
    }

    #[test]
    fn lamp_lit_ink_on_a_dark_ground_keeps_its_ink() {
        // The common murk-level lamp cell: bright red ink over a still-dark ground.
        // The ground is lifted (that is the floor) but the ink is byte-identical.
        let (fg, bg) = (0xE05050u32, 0x2A0C0Cu32);
        let (nfg, nbg) = legible_pair(fg, bg, 0x6A2424, 0x160505);
        assert_eq!(nfg, fg, "lamp-lit ink must not move");
        assert!(l_of(nbg) >= l_of(bg));
    }

    #[test]
    fn darker_ink_stays_darker() {
        let (a, _) = legible_pair(0x0A0A0A, 0x020202, 0x808080, 0x101010);
        let (b, _) = legible_pair(0x202020, 0x020202, 0x808080, 0x101010);
        assert!(l_of(a) < l_of(b));
    }

    #[test]
    fn dark_ink_on_light_ground_is_not_touched() {
        let (fg, bg) = (0x101010, 0x909090);
        assert_eq!(legible_pair(fg, bg, fg, bg), (fg, bg));
    }

    /// No lamp reaches the cell (`unlamped == lit`) ⇒ exactly the plain floor.
    #[test]
    fn lamped_floor_without_lamp_light_is_the_plain_floor() {
        for &(fg, bg) in &[
            (0x0A0304u32, 0x050102u32),
            (0x303840, 0x101418),
            (0xE0C080, 0x402010),
        ] {
            assert_eq!(
                legible_pair_lamped(fg, bg, (fg, bg), 0xC04040, 0x160505, true),
                legible_pair(fg, bg, 0xC04040, 0x160505)
            );
        }
    }

    /// The defect: flooring the LIT pair made a lamp pool darker than the same murk
    /// cell unlit (the ink lift switched off once the lamp raised the ground). With
    /// the lamp added on top of the floored murk, lit ≥ unlit on every channel, and
    /// strictly brighter by at least the lamp's own raw contribution. (The ground
    /// always; the ink on a blank cell — an inked one may keep darker lit strokes,
    /// see `inverted_ink_in_a_lamp_pool_keeps_its_contrast`.)
    #[test]
    fn a_lamp_never_darkens_a_floored_cell() {
        let (hue_fg, hue_bg) = (0xB0A080u32, 0x201810u32);
        let dark = (0x0C0A08u32, 0x040302u32);
        let base = legible_pair(dark.0, dark.1, hue_fg, hue_bg);
        for lamp in [0x08u32, 0x20, 0x40, 0x80] {
            let lit_fg = dark.0 + lamp * 0x010101;
            let lit_bg = dark.1 + (lamp + lamp / 2) * 0x010101; // the ground takes MORE (LAMP_BG_GAIN)
            let (fg, bg) = legible_pair_lamped(lit_fg, lit_bg, dark, hue_fg, hue_bg, false);
            for s in [16u32, 8, 0] {
                let c = |v: u32| (v >> s) & 0xFF;
                assert!(
                    c(fg) >= (c(base.0) + lamp).min(0xFF),
                    "ink {fg:06X} vs {:06X} lamp {lamp}",
                    base.0
                );
                assert!(
                    c(bg) >= (c(base.1) + lamp).min(0xFF),
                    "ground {bg:06X} vs {:06X} lamp {lamp}",
                    base.1
                );
            }
        }
    }

    #[test]
    fn lamp_delta_saturates_per_channel() {
        assert_eq!(add_lamp_delta(0xF01010, 0x402020, 0x101010), 0xFF2020);
        assert_eq!(
            add_lamp_delta(0x202020, 0x101010, 0x303030),
            0x202020,
            "never subtracts"
        );
    }

    /// A lamp pool that leaves dark relief ink on a lit ground keeps its strokes:
    /// the lit ink is kept instead of being floored across its ground.
    #[test]
    fn inverted_ink_in_a_lamp_pool_keeps_its_contrast() {
        let dark = (0x1A0808u32, 0x0C0404u32); // murk: ink a touch over its ground
        let (fg, bg) = (0x301010u32, 0x803030u32); // lit: the ground overtook the ink
        let (ink, ground) = legible_pair_lamped(fg, bg, dark, 0xC04040, 0x401010, true);
        let l = |c: u32| to_lab(c).l;
        assert!(l(ground) >= l(bg), "the ground is never dimmed");
        assert_eq!(ink, fg, "the dark strokes are kept as lit");
        // A blank cell has no strokes to keep: its fg takes the plain lamped rule.
        let (blank_fg, _) = legible_pair_lamped(fg, bg, dark, 0xC04040, 0x401010, false);
        assert!(l(blank_fg) > l(fg));
    }
}
