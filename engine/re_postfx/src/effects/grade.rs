//! Color-grade / tonemap: a filmic tonemap followed by contrast, saturation,
//! lift/gamma/gain and an optional hue rotation — the mood grade applied to every
//! cell's fg AND bg in place. Pure per-cell color op; glyph and all other fields
//! are preserved.
//!
//! Order within the pass (fixed): tonemap → contrast (around mid-grey) → lift/
//! gamma/gain → saturation → hue. The master `intensity` blends the graded result
//! back toward the original, so a low master is a gentle grade.

use re_core::Framebuffer;

use crate::color;
use crate::config::GradeFx;

/// Reinhard-ish filmic tonemap: `x / (1 + x)` rescaled so 1.0 maps near 1.0.
#[inline]
fn tonemap_reinhard(c: [f32; 3]) -> [f32; 3] {
    let f = |x: f32| {
        let m = x / (1.0 + x);
        // Renormalize so a unit input lands at ~1.0 (preserves bright whites).
        m * 2.0
    };
    [f(c[0]).min(1.0), f(c[1]).min(1.0), f(c[2]).min(1.0)]
}

/// Narkowicz ACES filmic approximation: a richer shoulder/toe than Reinhard for a
/// proper cinematic highlight rolloff and crushed-but-not-clipped shadows. Cheap
/// (one rational per channel), and `f(0)=0`, `f(1)≈1` so the identity neighbourhood
/// is preserved.
#[inline]
fn tonemap_aces(c: [f32; 3]) -> [f32; 3] {
    const A: f32 = 2.51;
    const B: f32 = 0.03;
    const C: f32 = 2.43;
    const D: f32 = 0.59;
    const E: f32 = 0.14;
    let f = |x: f32| {
        let n = x * (A * x + B);
        let d = x * (C * x + D) + E;
        (n / d).clamp(0.0, 1.0)
    };
    [f(c[0]), f(c[1]), f(c[2])]
}

/// Dispatch the selected tonemap curve.
#[inline]
fn tonemap(c: [f32; 3], aces: bool) -> [f32; 3] {
    if aces {
        tonemap_aces(c)
    } else {
        tonemap_reinhard(c)
    }
}

/// Hue-rotate an RGB triple by `deg` degrees using the YIQ chroma-rotation
/// approximation (cheap, no full HSV round-trip).
#[inline]
fn hue_rotate(c: [f32; 3], deg: f32) -> [f32; 3] {
    if deg == 0.0 {
        return c;
    }
    let a = deg.to_radians();
    let (s, co) = a.sin_cos();
    // YIQ rotation matrix coefficients for a hue shift.
    let m = [
        [
            0.299 + 0.701 * co + 0.168 * s,
            0.587 - 0.587 * co + 0.330 * s,
            0.114 - 0.114 * co - 0.497 * s,
        ],
        [
            0.299 - 0.299 * co - 0.328 * s,
            0.587 + 0.413 * co + 0.035 * s,
            0.114 - 0.114 * co + 0.292 * s,
        ],
        [
            0.299 - 0.300 * co + 1.250 * s,
            0.587 - 0.588 * co - 1.050 * s,
            0.114 + 0.886 * co - 0.203 * s,
        ],
    ];
    [
        (m[0][0] * c[0] + m[0][1] * c[1] + m[0][2] * c[2]).clamp(0.0, 1.0),
        (m[1][0] * c[0] + m[1][1] * c[1] + m[1][2] * c[2]).clamp(0.0, 1.0),
        (m[2][0] * c[0] + m[2][1] * c[1] + m[2][2] * c[2]).clamp(0.0, 1.0),
    ]
}

/// Apply the full grade chain to one float color.
fn grade_one(c0: [f32; 3], e: &GradeFx) -> [f32; 3] {
    let mut c = c0;
    if e.tonemap {
        c = tonemap(c, e.aces);
    }
    // Contrast around mid-grey 0.5.
    if (e.contrast - 1.0).abs() > 1e-4 {
        for v in &mut c {
            *v = (*v - 0.5) * e.contrast + 0.5;
        }
    }
    // Lift / gamma / gain.
    let gamma = e.gamma.max(1e-3);
    for v in &mut c {
        let mut x = (*v + e.lift * (1.0 - *v)).max(0.0); // lift the shadows
        if (gamma - 1.0).abs() > 1e-4 {
            x = x.powf(1.0 / gamma);
        }
        x *= e.gain;
        *v = x;
    }
    // Saturation around luma.
    if (e.saturation - 1.0).abs() > 1e-4 {
        let l = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
        for v in &mut c {
            *v = l + (*v - l) * e.saturation;
        }
    }
    c = hue_rotate(c, e.hue_shift);
    [
        c[0].clamp(0.0, 1.0),
        c[1].clamp(0.0, 1.0),
        c[2].clamp(0.0, 1.0),
    ]
}

pub(crate) fn apply(fb: &mut Framebuffer, e: &GradeFx, master: f32) {
    if !e.enabled || master <= 0.0 {
        return;
    }
    let mix = master.clamp(0.0, 1.0);
    let cols = fb.cols();
    let rows = fb.rows();
    for row in 0..rows {
        for col in 0..cols {
            let Some(s) = fb.get(col, row) else { continue };
            let mut c = *s;
            let fg0 = color::unpack_f(c.fg);
            let bg0 = color::unpack_f(c.bg);
            let fg1 = grade_one(fg0, e);
            let bg1 = grade_one(bg0, e);
            // Blend graded → original by the master mix.
            let blend = |a: [f32; 3], b: [f32; 3]| {
                [
                    a[0] + (b[0] - a[0]) * mix,
                    a[1] + (b[1] - a[1]) * mix,
                    a[2] + (b[2] - a[2]) * mix,
                ]
            };
            c.fg = color::pack_f(blend(fg0, fg1));
            c.bg = color::pack_f(blend(bg0, bg1));
            fb.set(col, row, c);
        }
    }
}
