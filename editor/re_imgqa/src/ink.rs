//! Ink-coverage table: a glyph → coverage fraction `[0,1]` lookup.
//!
//! The eye reads a terminal cell as an **ink mask** over the background. A solid
//! block `█` covers the whole cell (1.0); a space covers nothing (0.0); a dot or
//! a thin line is somewhere in between. This module is the source of truth for
//! that mapping for every glyph the engine's ASCII raycaster can emit.
//!
//! Coverage is used two ways: it forms each cell's *effective colour*
//! (`lerp(bg, fg, ink)`) and it is the **shape-contrast** signal
//! (`|ink(a) − ink(b)|`).
//!
//! Coverage gaps are made loud, not silent: an unmapped glyph falls back to a
//! mid value AND is reported via [`lookup`]'s `mapped` flag so the caller
//! can count it in `unmapped_glyphs`.

use re_render::body_ramp;

/// A glyph's ink coverage plus whether it was found in the table.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InkLookup {
    /// Coverage fraction in `[0,1]`.
    pub coverage: f32,
    /// `true` if the glyph had an explicit table entry; `false` ⇒ the mid-value
    /// fallback (counted as an `unmapped_glyph` diagnostic).
    pub mapped: bool,
}

/// Fallback coverage for an unmapped glyph: a deliberate mid value so an unknown
/// printable glyph still scores plausibly rather than reading as blank.
pub const UNMAPPED_COVERAGE: f32 = 0.5;

/// Coverage for the wall-body ramp characters, densest → sparsest. Indexed in
/// lockstep with [`re_render::body_ramp`] (`@%#*+=~:-.  `), evenly spaced from
/// 1.0 down to 0.0 so the ramp is monotonic by construction. Kept in sync by the
/// `ramp_glyphs_all_mapped` test.
fn ramp_coverage(idx: usize, len: usize) -> f32 {
    if len <= 1 {
        return 1.0;
    }
    // densest (idx 0) → 1.0, sparsest (idx len-1) → 0.0
    1.0 - (idx as f32) / ((len - 1) as f32)
}

/// Look up a glyph's ink coverage.
///
/// Order of resolution:
/// 1. The wall-body ramp (`re_render::body_ramp`), evenly spaced 1.0→0.0.
/// 2. The named special glyphs the renderer emits (box-drawing mortar/circuit,
///    floor grain, sky stars).
/// 3. A generic printable-ASCII coverage estimate (covers code-wall source text).
/// 4. The mid-value fallback (reported `mapped = false`).
pub fn lookup(glyph: u32) -> InkLookup {
    // Space is unambiguous and frequent: short-circuit it.
    if glyph == u32::from(' ') {
        return InkLookup {
            coverage: 0.0,
            mapped: true,
        };
    }

    // 1. Wall-body ramp.
    let ramp = body_ramp();
    if let Some(idx) = ramp.iter().position(|&b| u32::from(b) == glyph) {
        return InkLookup {
            coverage: ramp_coverage(idx, ramp.len()),
            mapped: true,
        };
    }

    // 2. Named special glyphs.
    if let Some(c) = special_coverage(glyph) {
        return InkLookup {
            coverage: c,
            mapped: true,
        };
    }

    // 3. Generic printable ASCII (code-wall source + any literal glyph).
    if let Some(c) = ascii_coverage(glyph) {
        return InkLookup {
            coverage: c,
            mapped: true,
        };
    }

    // 4. Unknown — loud fallback.
    InkLookup {
        coverage: UNMAPPED_COVERAGE,
        mapped: false,
    }
}

/// Just the coverage value (mid-fallback for unknowns), when the `mapped` flag
/// is not needed.
#[inline]
pub fn coverage(glyph: u32) -> f32 {
    lookup(glyph).coverage
}

/// Coverage for the renderer's named non-ramp glyphs: the box-drawing mortar /
/// circuit lines, floor grain specks, and sky stars. Codepoints mirror the
/// `const … _GLYPH` set in `re_render::ascii_shader`.
fn special_coverage(glyph: u32) -> Option<f32> {
    Some(match glyph {
        // Box-drawing lines (mortar ─ │, circuit bus/rung) — thin strokes.
        0x2500 => 0.45, // ─ MORTAR_H / CIRCUIT_RUNG
        0x2502 => 0.45, // │ MORTAR_V / CIRCUIT_BUS
        0x253C => 0.70, // ┼ CIRCUIT_NODE (cross — heavier)
        0x25CB => 0.55, // ○ CIRCUIT_PAD (ring)
        0x2726 => 0.65, // ✦ STAR_GLYPH_BRIGHT
        // Block-element shades (not in the ASCII ramp but conventional coverage).
        0x2591 => 0.25, // ░ light shade
        0x2592 => 0.50, // ▒ medium shade
        0x2593 => 0.75, // ▓ dark shade
        0x2588 => 1.00, // █ full block
        _ => return None,
    })
}

/// A coverage estimate for printable ASCII (0x20..=0x7E). Code walls lay the
/// engine's own source onto a wall, so any printable character can appear; this
/// gives each a plausible ink weight grouped by visual density. Control / non-
/// printable codepoints return `None` (→ unmapped fallback).
fn ascii_coverage(glyph: u32) -> Option<f32> {
    let c = char::from_u32(glyph)?;
    if !c.is_ascii() {
        return None;
    }
    let b = c as u8;
    Some(match b {
        b' ' => 0.0,
        // Very light: punctuation that is mostly whitespace.
        b'\'' | b'`' | b'.' | b',' => 0.12,
        b'-' | b'_' | b'~' | b':' | b';' | b'"' => 0.22,
        // Light operators / brackets.
        b'!' | b'|' | b'/' | b'\\' | b'(' | b')' | b'[' | b']' | b'{' | b'}' | b'<' | b'>'
        | b'^' | b'i' | b'l' | b'j' | b't' | b'f' | b'r' => 0.33,
        // Medium: most lowercase + common operators.
        b'=' | b'+' | b'*' | b'?' | b'%' | b'&' | b'$' => 0.45,
        b'a'..=b'z' => 0.45,
        // Digits + uppercase read a touch heavier.
        b'0'..=b'9' => 0.55,
        b'A'..=b'Z' => 0.58,
        // Heaviest ASCII ink.
        b'#' | b'@' => 0.85,
        // Control / non-printable ASCII (DEL, etc.) → unmapped fallback.
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn space_is_zero_and_block_is_full() {
        assert_eq!(coverage(u32::from(' ')), 0.0);
        assert_eq!(coverage(0x2588), 1.0); // █
    }

    #[test]
    fn ramp_glyphs_all_mapped_and_monotonic() {
        // Every glyph the renderer's wall ramp can emit must have an entry, and
        // the coverage must be (weakly) monotonic densest → sparsest.
        let ramp = body_ramp();
        let mut prev = f32::INFINITY;
        for &b in ramp {
            let l = lookup(u32::from(b));
            assert!(l.mapped, "ramp glyph {:?} unmapped", b as char);
            assert!(
                l.coverage <= prev + 1e-6,
                "ramp coverage not monotonic at {:?}: {} > {}",
                b as char,
                l.coverage,
                prev
            );
            prev = l.coverage;
        }
        // The ramp's densest glyph is full-ish ink, the sparsest is ~0.
        assert!(coverage(u32::from(ramp[0])) > 0.9);
        assert!(coverage(u32::from(ramp[ramp.len() - 1])) < 0.1);
    }

    #[test]
    fn every_renderer_emittable_glyph_is_mapped() {
        // The full set of glyphs the renderer can emit, gathered from the
        // `const … _GLYPH` constants + the ramp + code-wall ASCII source. This is
        // the spec §9 "every emittable glyph has a coverage entry" guard: it fails
        // loudly if a renderer glyph slips through to the unmapped fallback.
        let mut glyphs: Vec<u32> = body_ramp().iter().map(|&b| u32::from(b)).collect();
        glyphs.extend([
            0x2500,
            0x2502,
            0x253C,
            0x25CB,
            0x2726,          // box / circuit / star
            u32::from('.'),  // BRICK_SPECK / GRAIN_SPECK_A / STAR_FAINT
            u32::from('\''), // GRAIN_SPECK_B
            u32::from(':'),  // GRAIN_GRID
            u32::from('*'),  // STAR_MID
            u32::from(' '),  // code-wall void
        ]);
        // Code walls emit arbitrary printable ASCII source.
        glyphs.extend((0x20u32..=0x7E).collect::<Vec<_>>());

        for g in glyphs {
            let l = lookup(g);
            assert!(
                l.mapped,
                "renderer-emittable glyph {:#06x} ({:?}) is unmapped",
                g,
                char::from_u32(g)
            );
            assert!(
                (0.0..=1.0).contains(&l.coverage),
                "coverage out of range for {g:#06x}"
            );
        }
    }

    #[test]
    fn unknown_glyph_falls_back_and_is_flagged() {
        // A codepoint the renderer never emits (a CJK ideograph).
        let l = lookup(0x4E2D);
        assert!(!l.mapped, "unknown glyph should report mapped=false");
        assert_eq!(l.coverage, UNMAPPED_COVERAGE);
    }
}
