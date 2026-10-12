//! Colour math: packed-u32 sRGB ↔ linear sRGB ↔ Oklab, ΔE, and CVD simulation.
//!
//! The sRGB/Oklab conversions used to be defined HERE, but the palette gate
//! and the colour-drift field need the same math at RUNTIME, in a crate with
//! no terminal/game dependency — so they moved to `re_core::oklab`, the shared,
//! dependency-light foundation every crate already builds on. This module
//! now RE-EXPORTS them (an announced move, not a silent duplication) so every
//! existing `color::Foo` call site in this crate
//! keeps compiling unchanged, and adds the one thing that stays imgqa-only:
//! colour-vision-deficiency (CVD) simulation, which is a tooling/analysis
//! concern with no runtime caller.
//!
//! The pipeline is: unpack `u32` → gamma-decode each channel to **linear sRGB**
//! → (optionally lerp by ink coverage to form an *effective colour*) → convert
//! to **Oklab** (perceptually uniform). All colour-contrast metrics are Oklab
//! ΔE on those effective colours, so "fg vs bg", "glyph weight", and "hue" all
//! collapse into one comparable distance.

pub use re_core::oklab::{
    delta_e, delta_e_packed, lerp_linear, linear_to_oklab, linear_to_srgb, oklab_chroma, oklab_hue,
    oklab_to_linear, pack_linear, srgb_to_linear, unpack_linear, LinRgb, Oklab, Oklch,
};

// ─────────────────────────── colour-blindness (CVD) ─────────────────────────
//
// Simulated dichromacy stays HERE, not in `re_core::oklab`: it is an
// imgqa-only analysis concern (the render-QA linter's CVD-safety metric) with
// no runtime caller, unlike the Oklab/Oklch conversions above.

/// The three dichromacy types simulated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cvd {
    /// Protanopia — missing L cones (red-blind).
    Protan,
    /// Deuteranopia — missing M cones (green-blind).
    Deutan,
    /// Tritanopia — missing S cones (blue-blind).
    Tritan,
}

impl Cvd {
    /// All three, for iterating the worst-case search.
    pub const ALL: [Cvd; 3] = [Cvd::Protan, Cvd::Deutan, Cvd::Tritan];
}

/// Simulate a CVD on a linear-sRGB colour using the Viénot/Brettel-style
/// dichromat projection matrices (operating in linear-RGB space). These are the
/// widely-used machado/viénot approximations; they collapse the confusion axis
/// of each dichromacy so a red/green pair, say, maps to near-identical outputs
/// under deuteranopia. Gray (r==g==b) is preserved by construction.
#[must_use]
pub fn simulate_cvd(c: LinRgb, kind: Cvd) -> LinRgb {
    // Row-major 3×3 matrices (linear RGB → linear RGB).
    let m: [f32; 9] = match kind {
        Cvd::Protan => [
            0.152_286, 1.052_583, -0.204_868, 0.114_503, 0.786_281, 0.099_216, -0.003_882,
            -0.048_116, 1.051_998,
        ],
        Cvd::Deutan => [
            0.367_322, 0.860_646, -0.227_968, 0.280_085, 0.672_501, 0.047_413, -0.011_820,
            0.042_940, 0.968_881,
        ],
        Cvd::Tritan => [
            1.255_528, -0.076_749, -0.178_779, -0.078_411, 0.930_809, 0.147_602, 0.004_733,
            0.691_367, 0.303_900,
        ],
    };
    let r = m[0] * c.r + m[1] * c.g + m[2] * c.b;
    let g = m[3] * c.r + m[4] * c.g + m[5] * c.b;
    let b = m[6] * c.r + m[7] * c.g + m[8] * c.b;
    LinRgb {
        r: r.clamp(0.0, 1.0),
        g: g.clamp(0.0, 1.0),
        b: b.clamp(0.0, 1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lin(r: f32, g: f32, b: f32) -> LinRgb {
        LinRgb { r, g, b }
    }

    // The sRGB<->Oklab roundtrip / ΔE / chroma tests now live with the
    // implementation in `re_core::oklab::tests` — this crate only re-tests the
    // CVD simulation, which is the part that actually still lives here.

    #[test]
    fn cvd_matrices_preserve_gray() {
        for &g in &[0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let gray = lin(g, g, g);
            for kind in Cvd::ALL {
                let out = simulate_cvd(gray, kind);
                assert!((out.r - g).abs() < 0.02, "{kind:?} shifted gray r: {out:?}");
                assert!((out.g - g).abs() < 0.02, "{kind:?} shifted gray g: {out:?}");
                assert!((out.b - g).abs() < 0.02, "{kind:?} shifted gray b: {out:?}");
            }
        }
    }

    #[test]
    fn red_green_collapse_under_deuteranopia() {
        // A saturated red/green pair is highly separable normally, but should
        // collapse (much smaller ΔE) under deuteranopia.
        let red = unpack_linear(0x00FF_0000);
        let green = unpack_linear(0x0000_FF00);
        let normal = delta_e(linear_to_oklab(red), linear_to_oklab(green));

        let red_d = simulate_cvd(red, Cvd::Deutan);
        let green_d = simulate_cvd(green, Cvd::Deutan);
        let collapsed = delta_e(linear_to_oklab(red_d), linear_to_oklab(green_d));

        assert!(
            collapsed < normal * 0.5,
            "red/green should collapse under deuteranopia (normal={normal}, cvd={collapsed})"
        );
    }
}
