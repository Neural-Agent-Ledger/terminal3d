//! `re_render::quadrant` — the LOCAL 2×2 quadrant blitter (sub-cell detail).
//!
//! The raycaster makes the normal one-ray-per-cell G-buffer. Most cells are flat
//! (a wall face, a floor band) and read fine as a single full glyph. But a *cell
//! that straddles a silhouette* — the edge of a pillar against the floor behind it,
//! the lip of a doorway — packs two very different surfaces into one terminal cell;
//! a single glyph there reads as a jagged staircase. This module adds detail ONLY
//! to those cells, by sampling the cell's four sub-centres (a 2×2) and emitting one
//! of the sixteen Unicode QUADRANT BLOCK glyphs plus TWO colours (the near pair as
//! foreground, the far pair as background) so the half-cell that is "pillar" and the
//! half that is "floor behind" each keep their own colour and the boundary lands on
//! a sub-cell edge instead of a whole-cell jump.
//!
//! ## Economy
//!
//! The expensive part — casting four sub-rays — happens ONLY for cells the cheap
//! [`needs_subcell`] classifier marks. A flat cell stays a single full glyph (one
//! ray, as before). Marking is driven by either:
//!
//! * a SILHOUETTE — the cell's depth differs from a neighbour by more than a
//!   ratio threshold ([`SUBCELL_DEPTH_RATIO`]); this is the depth-discontinuity
//!   edge that benefits from sub-cell detail, OR
//! * a MATERIAL opt-in — a surface that always wants the extra resolution.
//!
//! Far / frozen cells are NEVER marked (the caller gates that), so distant
//! background never pays the 4× and never noises. The marked fraction on a normal
//! room is small (a handful of edge cells per frame), so the whole-frame overhead
//! stays in the 1.15–1.45× band, never 4×.
//!
//! ## Purity
//!
//! Everything here is pure, allocation-free, and Bevy-free: [`needs_subcell`] is a
//! handful of comparisons; [`blit_quadrant`] is a mask build + a 2-colour split.
//! The sub-ray CASTING lives in `composite.rs` (it owns the cast loop + the
//! `crate::shade_subray` sampler); this module only classifies and composes.

use re_core::Cell;

/// The sixteen QUADRANT glyphs, indexed by a 4-bit mask `TL|TR|BL|BR`
/// (bit 0 = top-left, bit 1 = top-right, bit 2 = bottom-left, bit 3 = bottom-right).
/// A set bit means that sub-cell belongs to the NEAR (foreground) pair, so the glyph
/// inks that quadrant in the foreground colour over the background colour.
///
/// `0b0000` → space (all four sub-cells are "far"; the cell is a flat background),
/// `0b1111` → `█` (all four are "near"; a flat foreground). The fourteen in between
/// are the genuine half/quarter/diagonal block glyphs from Block Elements
/// (U+2580–U+259F), all width-1 in any monospace font.
pub const QUADRANT_GLYPHS: [char; 16] = [
    ' ', // 0000 — none near
    '▘', // 0001 — TL
    '▝', // 0010 — TR
    '▀', // 0011 — TL+TR  (upper half)
    '▖', // 0100 — BL
    '▌', // 0101 — TL+BL  (left half)
    '▞', // 0110 — TR+BL  (anti-diagonal)
    '▛', // 0111 — TL+TR+BL
    '▗', // 1000 — BR
    '▚', // 1001 — TL+BR  (main diagonal)
    '▐', // 1010 — TR+BR  (right half)
    '▜', // 1011 — TL+TR+BR
    '▄', // 1100 — BL+BR  (lower half)
    '▙', // 1101 — TL+BL+BR
    '▟', // 1110 — TR+BL+BR
    '█', // 1111 — all near
];

/// The depth-ratio threshold a [`needs_subcell`] SILHOUETTE crossing must exceed:
/// a cell whose depth differs from a neighbour by more than this *fraction of the
/// nearer depth* is a silhouette edge and is marked for sub-cell detail. Expressed
/// as a ratio (not an absolute distance) so a far wall and a near wall are judged on
/// the same relative scale — a 0.5-cell step at depth 1 is a strong edge, the same
/// step at depth 20 is not. Tuned so flat walls (whose neighbour depths vary
/// smoothly by foreshortening) stay UNMARKED while a real occlusion boundary marks.
pub const SUBCELL_DEPTH_RATIO: f32 = 0.18;

/// One shaded sub-ray sample inside a marked cell: the glyph/colours the
/// `crate::shade_subray` cast produced for that sub-centre, plus its depth (the
/// near/far split key). The quadrant blitter reads only `fg`/`bg`/`depth`/`glyph`.
///
/// Built by `composite.rs` from a `crate::Cell` (`SubSample::from_cell`), kept as a
/// small `Copy` POD so the four-sample array lives on the stack — no allocation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SubSample {
    /// Perpendicular depth at this sub-centre (the near/far split key).
    pub depth: f32,
    /// Foreground colour the sub-ray shaded (`0x00RRGGBB`).
    pub fg: u32,
    /// Background colour the sub-ray shaded (`0x00RRGGBB`).
    pub bg: u32,
    /// The full glyph the sub-ray would have drawn (used to pick a representative
    /// ink colour: a glyph cell contributes its `fg`, a blank cell its `bg`).
    pub glyph: u32,
}

impl SubSample {
    /// Build a sub-sample from a fully-shaded [`Cell`] (the output of
    /// `crate::shade_subray`).
    #[inline]
    pub fn from_cell(c: &Cell) -> Self {
        SubSample {
            depth: c.depth,
            fg: c.fg,
            bg: c.bg,
            glyph: c.glyph,
        }
    }

    /// The "ink" colour this sub-cell contributes to its pair's average: a sub-cell
    /// that drew a visible glyph contributes its foreground; a blank one its
    /// background. So a quadrant of "wall glyph" averages toward the wall's ink and a
    /// quadrant of "open floor" toward the floor's fill.
    #[inline]
    fn ink(&self) -> u32 {
        if self.glyph == u32::from(' ') {
            self.bg
        } else {
            self.fg
        }
    }
}

/// The result of [`blit_quadrant`]: the sub-cell glyph + the two-colour split to
/// write into the cell. `fg` inks the NEAR quadrants (the mask bits), `bg` the FAR
/// quadrants. `depth` is the nearest of the four sub-samples (the cell's depth cue
/// must reflect its closest surface for honest occlusion).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadrantBlit {
    /// One of the sixteen [`QUADRANT_GLYPHS`].
    pub glyph: u32,
    /// Foreground colour — the average ink of the NEAR (foreground) sub-samples.
    pub fg: u32,
    /// Background colour — the average ink of the FAR (background) sub-samples.
    pub bg: u32,
    /// The cell's depth cue: the NEAREST sub-sample depth (closest surface wins).
    pub depth: f32,
}

/// The cheap per-cell CLASSIFIER: should this cell cast the four sub-rays?
///
/// Returns `true` when either:
/// * `material_opt_in` is set (a material that always wants sub-cell detail), OR
/// * a SILHOUETTE crossing is present — `depth_here` differs from ANY neighbour in
///   `neighbour_depths` by more than [`SUBCELL_DEPTH_RATIO`] of the nearer depth.
///
/// Pure + branch-light (a few abs/compares over ≤4 neighbours) so running it on
/// every cell is far cheaper than a single ray. A non-finite `depth_here` (a cell
/// that hit nothing — open sky) is NEVER marked: there is no surface to detail.
///
/// The CALLER is responsible for not marking far / frozen cells (pass an empty
/// neighbour set and `material_opt_in = false` for those), so distant background
/// never pays the sub-cell cost.
#[inline]
pub fn needs_subcell(depth_here: f32, neighbour_depths: &[f32], material_opt_in: bool) -> bool {
    if material_opt_in {
        return true;
    }
    if !depth_here.is_finite() {
        return false; // open sky / miss — nothing to refine.
    }
    for &nd in neighbour_depths {
        if !nd.is_finite() {
            // A neighbour that hit nothing while THIS cell hit a surface is itself a
            // silhouette (surface edge against open space) — mark it.
            return true;
        }
        let near = depth_here.min(nd).max(1e-4);
        if (depth_here - nd).abs() > SUBCELL_DEPTH_RATIO * near {
            return true;
        }
    }
    false
}

/// Map the four shaded sub-samples (`[TL, TR, BL, BR]`) to one [`QuadrantBlit`]:
/// a quadrant glyph + a two-colour near/far split.
///
/// Algorithm (pure, deterministic):
/// 1. Find the SPLIT depth — the midpoint between the nearest and farthest of the
///    four sub-samples. A sub-sample nearer than the split is "near" (foreground);
///    the rest are "far" (background).
/// 2. Build the 4-bit mask from which sub-cells are near → pick the glyph.
/// 3. `fg` = average ink of the near sub-samples; `bg` = average ink of the far
///    sub-samples. A degenerate split (all four on one side — a flat cell that was
///    marked by a neighbour but is itself uniform) puts every sub-cell in the near
///    set, yielding `█` with the surface colour as both fg and the (equal) bg, so a
///    falsely-marked flat cell still reads correctly.
///
/// The split is by DEPTH (not colour) so the boundary tracks the silhouette: the
/// near pair is the closer surface (e.g. the pillar), the far pair the surface
/// behind it (e.g. the floor) — exactly the edge the single-glyph cast smeared.
#[inline]
pub fn blit_quadrant(samples: [SubSample; 4]) -> QuadrantBlit {
    // 1. Near/far split at the depth midpoint.
    let mut near_d = f32::INFINITY;
    let mut far_d = f32::NEG_INFINITY;
    for s in &samples {
        let d = if s.depth.is_finite() {
            s.depth
        } else {
            f32::MAX
        };
        near_d = near_d.min(d);
        far_d = far_d.max(d);
    }
    // Midpoint; a tiny epsilon below `far_d` so the FARTHEST sample is never counted
    // "near" when the spread is real. When all four are equal (`near_d == far_d`) the
    // threshold sits at the depth itself and every sub-cell counts as near (flat cell).
    let split = if (far_d - near_d) <= f32::EPSILON {
        far_d + 1.0 // everything strictly less → all near.
    } else {
        0.5 * (near_d + far_d)
    };

    // 2. Mask + 3. accumulate the two colour groups.
    let mut mask = 0u8;
    let mut near_rgb = [0u32; 3];
    let mut far_rgb = [0u32; 3];
    let mut near_n = 0u32;
    let mut far_n = 0u32;
    let mut nearest = f32::INFINITY;
    for (i, s) in samples.iter().enumerate() {
        let d = if s.depth.is_finite() {
            s.depth
        } else {
            f32::MAX
        };
        nearest = nearest.min(s.depth);
        let ink = s.ink();
        if d < split {
            mask |= 1 << i;
            accumulate(&mut near_rgb, ink);
            near_n += 1;
        } else {
            accumulate(&mut far_rgb, ink);
            far_n += 1;
        }
    }

    let fg = average(near_rgb, near_n).unwrap_or_else(|| average(far_rgb, far_n).unwrap_or(0));
    let bg = average(far_rgb, far_n).unwrap_or(fg);

    QuadrantBlit {
        glyph: QUADRANT_GLYPHS[(mask & 0x0F) as usize] as u32,
        fg,
        bg,
        depth: if nearest.is_finite() {
            nearest
        } else {
            f32::INFINITY
        },
    }
}

/// One INTERIOR sub-texel for the luminance-split image samplers ([`blit_quadrant_luminance`]
/// / [`blit_braille_luminance`]): the texture colour at the sub-position and whether it is
/// OPAQUE (inside the alpha silhouette). A TRANSPARENT sub-texel is OUTSIDE the figure — it
/// always belongs to the background and contributes the `backdrop` colour, preserving the
/// sharp alpha EDGE regardless of the interior tonal split.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageSub {
    /// `true` ⇒ the sub-texel is opaque (figure interior); `false` ⇒ transparent (backdrop).
    pub opaque: bool,
    /// The texel's colour (`0x00RRGGBB`). Meaningful only when `opaque`.
    pub rgb: u32,
}

/// Rec.601 relative luminance of a packed `0x00RRGGBB`, scaled 0..=255*… (we keep it in the
/// same fixed-point as the channels so a mean over sub-texels stays integer-exact and the
/// blit is byte-deterministic). Returns a `u32` in `[0, 255]`.
#[inline]
fn luma255(packed: u32) -> u32 {
    let r = (packed >> 16) & 0xFF;
    let g = (packed >> 8) & 0xFF;
    let b = packed & 0xFF;
    // 0.299/0.587/0.114 in 1/256 fixed point (77,150,29 ≈ 256). Integer ⇒ deterministic.
    (r * 77 + g * 150 + b * 29) >> 8
}

/// QUAD luminance split (image interior detail). Given the four sub-texels `[TL,TR,BL,BR]`
/// and the `backdrop` a transparent sub-texel reveals, resolve INTERIOR tonal detail at 2×2
/// while keeping the alpha silhouette sharp.
///
/// Algorithm (pure, deterministic):
/// 1. threshold = MEAN luminance over the four sub-texels, where a TRANSPARENT sub-texel
///    contributes the `backdrop`'s luminance (so the figure edge against the backdrop still
///    falls out of the comparison the same way it did under the old opaque/transparent split).
/// 2. A sub-texel sets its mask bit (→ foreground, the LIGHTER side) iff it is OPAQUE AND its
///    luminance ≥ threshold. Everything else (opaque-but-dark, OR transparent) is background.
/// 3. `fg` = mean rgb of the foreground sub-texels; `bg` = mean rgb of the background set,
///    where transparent members contribute the `backdrop` colour (sharp edge) and
///    opaque-but-dark members contribute their own rgb (interior dark tone).
///
/// Net: a light+dark interior cell splits on the real tonal boundary (a half/quarter block);
/// a silhouette cell still resolves transparent→backdrop; a flat-colour interior cell yields
/// `█` in that colour (all four equal → all foreground).
#[inline]
pub fn blit_quadrant_luminance(subs: [ImageSub; 4], backdrop: u32) -> QuadrantBlit {
    let backdrop_luma = luma255(backdrop);
    // 1. Mean luminance threshold (transparent ⇒ backdrop luminance).
    let mut luma_sum = 0u32;
    for s in &subs {
        luma_sum += if s.opaque {
            luma255(s.rgb)
        } else {
            backdrop_luma
        };
    }
    let threshold = luma_sum / 4;

    // 2 + 3. Mask + two colour groups.
    let mut mask = 0u8;
    let mut fg_rgb = [0u32; 3];
    let mut bg_rgb = [0u32; 3];
    let mut fg_n = 0u32;
    let mut bg_n = 0u32;
    for (i, s) in subs.iter().enumerate() {
        if s.opaque && luma255(s.rgb) >= threshold {
            mask |= 1 << i;
            accumulate(&mut fg_rgb, s.rgb);
            fg_n += 1;
        } else {
            // background: opaque-but-dark contributes its own rgb; transparent the backdrop.
            let c = if s.opaque { s.rgb } else { backdrop };
            accumulate(&mut bg_rgb, c);
            bg_n += 1;
        }
    }

    let fg = average(fg_rgb, fg_n).unwrap_or_else(|| average(bg_rgb, bg_n).unwrap_or(backdrop));
    let bg = average(bg_rgb, bg_n).unwrap_or(fg);
    QuadrantBlit {
        glyph: QUADRANT_GLYPHS[(mask & 0x0F) as usize] as u32,
        fg,
        bg,
        // Image sub-cells carry no depth split; the caster owns the cell's depth cue.
        depth: 0.0,
    }
}

/// BRAILLE luminance halftone (image interior detail). Given the eight sub-texels
/// `grid[row][col]` (`row ∈ 0..4` top→bottom, `col ∈ 0..2` left→right) and the `backdrop`,
/// resolve a 2×4 halftone of the figure's INTERIOR luminance plus the sharp alpha edge.
///
/// Algorithm (pure, deterministic):
/// 1. threshold = MEAN luminance over the eight (transparent ⇒ backdrop luminance).
/// 2. A dot is ON (ink) iff it is OPAQUE AND DARKER than the threshold — the darker interior
///    detail becomes the ink (reads portraits / line-art well; ink-polarity = darker=ink).
///    A transparent OR lighter sub-texel is OFF (shows the backdrop / lighter fill).
/// 3. `fg` = mean rgb of the ON (ink) dots; `bg` = mean rgb of the OFF set, where transparent
///    members contribute the `backdrop` (sharp edge) and lighter-opaque members their own rgb.
///
/// SOLID-FIGURE SAFETY: if the darker=ink rule inks NO dot yet the cell DOES contain opaque
/// texels (a flat or uniformly-bright opaque patch — no interior tonal detail, e.g. a bright
/// figure on a dark backdrop), every opaque dot is inked instead. This guarantees a solid
/// figure is never rendered as a blank (invisible) cell: with no interior detail to halftone,
/// the cell falls back to a full-coverage silhouette in the figure's colour, and the
/// transparent texels still stay OFF so the alpha edge survives.
#[inline]
pub fn blit_braille_luminance(grid: [[ImageSub; 2]; 4], backdrop: u32) -> BrailleBlit {
    let backdrop_luma = luma255(backdrop);
    // 1. Mean luminance threshold + count opaque texels (for the solid-figure fallback).
    let mut luma_sum = 0u32;
    let mut opaque_n = 0u32;
    for cols in &grid {
        for s in cols {
            if s.opaque {
                luma_sum += luma255(s.rgb);
                opaque_n += 1;
            } else {
                luma_sum += backdrop_luma;
            }
        }
    }
    let threshold = luma_sum / 8;

    // 2. First pass: would the darker=ink rule light any dot?
    let mut any_ink = false;
    for cols in &grid {
        for s in cols {
            if s.opaque && luma255(s.rgb) < threshold {
                any_ink = true;
            }
        }
    }
    // Solid-figure fallback: opaque texels exist but none are below the mean → ink ALL
    // opaque dots so the figure stays visible (full-coverage silhouette, not blank).
    let ink_all_opaque = !any_ink && opaque_n > 0;

    // 3. Mask + ink/backdrop colour groups.
    let mut mask = 0u8;
    let mut ink = [0u32; 3];
    let mut off_rgb = [0u32; 3];
    let mut on_n = 0u32;
    let mut off_n = 0u32;
    for (row, cols) in grid.iter().enumerate() {
        for (col, s) in cols.iter().enumerate() {
            let on = s.opaque && (ink_all_opaque || luma255(s.rgb) < threshold);
            if on {
                mask |= BRAILLE_DOT_BITS[row][col];
                accumulate(&mut ink, s.rgb);
                on_n += 1;
            } else {
                let c = if s.opaque { s.rgb } else { backdrop };
                accumulate(&mut off_rgb, c);
                off_n += 1;
            }
        }
    }
    let fg = average(ink, on_n).unwrap_or(backdrop);
    let bg = average(off_rgb, off_n).unwrap_or(backdrop);
    BrailleBlit {
        glyph: braille_glyph(mask) as u32,
        fg,
        bg,
    }
}

// ───────────────────── the rest of the sub-cell ladder ───────────────────────
//
// `notcurses` — the reference terminal-blitting implementation — defines the
// degradation ladder as Pixel > Sextant(3×2) > Quadrant(2×2) > Half(2×1) > ASCII,
// and documents two facts this module is built around:
//
//   * 1×1 and 2×2 "can be perfectly represented with two colors per cell";
//     higher geometries "are increasingly likely to require some degree of
//     interpolation".
//   * Braille "doesn't tend to work out very well for images", though it is good
//     for plots.
//
// We shipped Quadrant and Braille and skipped the two rungs between them. That left
// image work with a bad choice: 2×2 (two colours, coarse) or 2×4 Braille (fine, but
// ONE ink colour for all eight dots — see `BrailleBlit`). SEXTANT is the rung that
// resolves it: 3× the sub-texels of a quadrant while still being a BLOCK geometry,
// so it keeps the two-colour split. HALF is the cheap universal rung below quadrant,
// and it matters for a reason beyond resolution: a terminal cell is roughly 2:1
// tall, so a 2×1 split is the one that makes a sub-texel approximately SQUARE.

/// The four HALF-BLOCK glyphs, indexed by a 2-bit mask `T|B` (bit 0 = top,
/// bit 1 = bottom). A set bit means that half belongs to the NEAR / ink group.
///
/// All four are Block Elements (U+2580–U+259F) or ASCII space, i.e. present in every
/// monospace font that can draw the quadrant set — this rung needs no capability
/// gate.
pub const HALF_GLYPHS: [char; 4] = [
    ' ', // 00 — neither
    '▀', // 01 — top
    '▄', // 10 — bottom
    '█', // 11 — both
];

/// First BLOCK SEXTANT codepoint (`U+1FB00` = BLOCK SEXTANT-1).
const SEXTANT_BASE: u32 = 0x1FB00;

/// The four sextant masks that are NOT encoded in the Symbols for Legacy Computing
/// block, because Unicode already had them:
///
/// | mask | sub-cells | existing character |
/// |------|-----------|--------------------|
/// | `0b000000` | none | `U+0020` SPACE |
/// | `0b010101` | left column (TL+ML+BL) | `U+258C` ▌ LEFT HALF BLOCK |
/// | `0b101010` | right column (TR+MR+BR) | `U+2590` ▐ RIGHT HALF BLOCK |
/// | `0b111111` | all | `U+2588` █ FULL BLOCK |
///
/// The other 60 masks occupy `U+1FB00..=U+1FB3B` **in ascending mask order with these
/// four skipped** — which is why [`sextant_glyph`] computes the codepoint instead of
/// carrying a 60-entry table nobody could proofread.
const SEXTANT_SKIPPED: [u8; 4] = [0b000000, 0b010101, 0b101010, 0b111111];

/// The BLOCK SEXTANT glyph for a 6-bit sub-cell mask.
///
/// Bit order follows the Unicode sextant NUMBERING, so the arithmetic below is the
/// encoding rule rather than a coincidence:
///
/// ```text
///   bit 0 (value  1) = top-left       bit 1 (value  2) = top-right
///   bit 2 (value  4) = middle-left    bit 3 (value  8) = middle-right
///   bit 4 (value 16) = bottom-left    bit 5 (value 32) = bottom-right
/// ```
///
/// A set bit means that sub-cell is inked in the FOREGROUND colour. `mask` is taken
/// modulo 64; the four masks in [`SEXTANT_SKIPPED`] return their pre-existing Block
/// Elements character, every other mask indexes `U+1FB00` by how many un-skipped
/// masks precede it.
///
/// Verified against the standard at three points: mask 1 → `U+1FB00`
/// (BLOCK SEXTANT-1), mask 29 → `U+1FB1B` (BLOCK SEXTANT-1345), and mask 62 →
/// `U+1FB3B` (BLOCK SEXTANT-23456, the last of the 60).
#[inline]
pub fn sextant_glyph(mask: u8) -> char {
    let m = mask & 0b0011_1111;
    match m {
        0b000000 => ' ',
        0b010101 => '▌',
        0b101010 => '▐',
        0b111111 => '█',
        _ => {
            // How many un-skipped masks are strictly below `m`? `m` itself is not
            // skipped here, so this is `m` minus the skipped values below it.
            let skipped_below = SEXTANT_SKIPPED.iter().filter(|&&s| s < m).count() as u32;
            let idx = u32::from(m) - skipped_below;
            // Always a valid scalar: idx ≤ 59, so the codepoint is ≤ U+1FB3B.
            char::from_u32(SEXTANT_BASE + idx).unwrap_or('█')
        }
    }
}

/// HALF-BLOCK (2×1) geometry blit from two shaded sub-samples `[top, bottom]`.
///
/// The depth-split, colour-averaging and degenerate-cell behaviour are identical to
/// [`blit_quadrant`] — only the geometry differs. See [`blit_blocks`].
#[inline]
pub fn blit_half(samples: [SubSample; 2]) -> QuadrantBlit {
    blit_blocks(&samples, |m| HALF_GLYPHS[(m & 0b11) as usize])
}

/// SEXTANT (3×2) geometry blit from six shaded sub-samples, in Unicode sextant bit
/// order: `[TL, TR, ML, MR, BL, BR]`.
///
/// Same depth-split model as [`blit_quadrant`], at three times the sub-texels, and
/// still only two colours per cell — which is the whole reason to prefer it over
/// Braille for image and silhouette work.
#[inline]
pub fn blit_sextant(samples: [SubSample; 6]) -> QuadrantBlit {
    blit_blocks(&samples, sextant_glyph)
}

/// The shared BLOCK-geometry depth blit behind [`blit_half`] and [`blit_sextant`].
///
/// Generalises [`blit_quadrant`]'s algorithm to any sub-cell count: split the samples
/// at the midpoint between nearest and farthest depth, ink the near group, and give
/// each group the mean of its members' [`SubSample::ink`].
///
/// [`blit_quadrant`] itself is deliberately NOT rewritten in terms of this function:
/// it is pinned by the golden corpus, and a "surely identical" refactor of a hashed
/// code path is exactly the change that turns out not to be. `blit_blocks` is instead
/// proven equivalent to it by test (`blit_blocks_matches_blit_quadrant_on_four`).
#[inline]
fn blit_blocks(samples: &[SubSample], glyph_of: impl Fn(u8) -> char) -> QuadrantBlit {
    let mut near_d = f32::INFINITY;
    let mut far_d = f32::NEG_INFINITY;
    for s in samples {
        let d = if s.depth.is_finite() {
            s.depth
        } else {
            f32::MAX
        };
        near_d = near_d.min(d);
        far_d = far_d.max(d);
    }
    let split = if (far_d - near_d) <= f32::EPSILON {
        far_d + 1.0 // flat cell → everything strictly less → all near.
    } else {
        0.5 * (near_d + far_d)
    };

    let mut mask = 0u8;
    let mut near_rgb = [0u32; 3];
    let mut far_rgb = [0u32; 3];
    let mut near_n = 0u32;
    let mut far_n = 0u32;
    let mut nearest = f32::INFINITY;
    for (i, s) in samples.iter().enumerate() {
        let d = if s.depth.is_finite() {
            s.depth
        } else {
            f32::MAX
        };
        nearest = nearest.min(s.depth);
        let ink = s.ink();
        if d < split {
            mask |= 1 << i;
            accumulate(&mut near_rgb, ink);
            near_n += 1;
        } else {
            accumulate(&mut far_rgb, ink);
            far_n += 1;
        }
    }

    let fg = average(near_rgb, near_n).unwrap_or_else(|| average(far_rgb, far_n).unwrap_or(0));
    let bg = average(far_rgb, far_n).unwrap_or(fg);
    QuadrantBlit {
        glyph: glyph_of(mask) as u32,
        fg,
        bg,
        depth: if nearest.is_finite() {
            nearest
        } else {
            f32::INFINITY
        },
    }
}

/// HALF-BLOCK luminance blit for image interiors — the 2×1 sibling of
/// [`blit_quadrant_luminance`]. `subs` is `[top, bottom]`.
#[inline]
pub fn blit_half_luminance(subs: [ImageSub; 2], backdrop: u32) -> QuadrantBlit {
    blit_blocks_luminance(&subs, backdrop, |m| HALF_GLYPHS[(m & 0b11) as usize])
}

/// SEXTANT luminance blit for image interiors — the 3×2 sibling of
/// [`blit_quadrant_luminance`]. `grid[row][col]`, `row ∈ 0..3` top→bottom,
/// `col ∈ 0..2` left→right.
///
/// This is the rung intended to REPLACE Braille for image work: the same 6-of-8
/// order of detail, but a real two-colour split instead of one ink for the cell.
#[inline]
pub fn blit_sextant_luminance(grid: [[ImageSub; 2]; 3], backdrop: u32) -> QuadrantBlit {
    // Flatten in Unicode sextant bit order: row-major already gives TL,TR,ML,MR,BL,BR.
    let flat = [
        grid[0][0], grid[0][1], grid[1][0], grid[1][1], grid[2][0], grid[2][1],
    ];
    blit_blocks_luminance(&flat, backdrop, sextant_glyph)
}

/// The shared BLOCK-geometry luminance blit behind [`blit_half_luminance`] and
/// [`blit_sextant_luminance`].
///
/// Identical in policy to [`blit_quadrant_luminance`], generalised over sub-texel
/// count: threshold at the MEAN luminance (a transparent texel counts as the
/// backdrop's luminance), ink every opaque texel at or above it, and give each group
/// the mean of its members.
///
/// Note the ink POLARITY is the quadrant one — *lighter* is foreground — not the
/// Braille one (*darker* is ink). These are block glyphs filling area, not dots
/// laying ink on paper, so a bright figure must fill its sub-cells rather than
/// hollow them out.
#[inline]
fn blit_blocks_luminance(
    subs: &[ImageSub],
    backdrop: u32,
    glyph_of: impl Fn(u8) -> char,
) -> QuadrantBlit {
    let backdrop_luma = luma255(backdrop);
    let n = subs.len().max(1) as u32;
    let mut luma_sum = 0u32;
    for s in subs {
        luma_sum += if s.opaque {
            luma255(s.rgb)
        } else {
            backdrop_luma
        };
    }
    let threshold = luma_sum / n;

    let mut mask = 0u8;
    let mut fg_rgb = [0u32; 3];
    let mut bg_rgb = [0u32; 3];
    let mut fg_n = 0u32;
    let mut bg_n = 0u32;
    for (i, s) in subs.iter().enumerate() {
        if s.opaque && luma255(s.rgb) >= threshold {
            mask |= 1 << i;
            accumulate(&mut fg_rgb, s.rgb);
            fg_n += 1;
        } else {
            let c = if s.opaque { s.rgb } else { backdrop };
            accumulate(&mut bg_rgb, c);
            bg_n += 1;
        }
    }

    let fg = average(fg_rgb, fg_n).unwrap_or_else(|| average(bg_rgb, bg_n).unwrap_or(backdrop));
    let bg = average(bg_rgb, bg_n).unwrap_or(fg);
    QuadrantBlit {
        glyph: glyph_of(mask) as u32,
        fg,
        bg,
        // Image sub-cells carry no depth split; the caster owns the cell's depth cue.
        depth: 0.0,
    }
}

/// Accumulate one packed `0x00RRGGBB` colour into a running per-channel sum.
#[inline]
fn accumulate(acc: &mut [u32; 3], packed: u32) {
    acc[0] += (packed >> 16) & 0xFF;
    acc[1] += (packed >> 8) & 0xFF;
    acc[2] += packed & 0xFF;
}

/// Average a per-channel sum over `n` samples back into a packed colour, or `None`
/// if the group was empty.
#[inline]
fn average(acc: [u32; 3], n: u32) -> Option<u32> {
    if n == 0 {
        return None;
    }
    let r = acc[0] / n;
    let g = acc[1] / n;
    let b = acc[2] / n;
    Some((r << 16) | (g << 8) | b)
}

// ─────────────────────────── 2×4 BRAILLE blitter ────────────────────────────
//
// The quadrant blitter above doubles resolution (2×2, two colours). For an even
// crisper SILHOUETTE the Braille blitter packs a 2-wide × 4-tall coverage grid into
// one glyph — eight ON/OFF dots in the Unicode BRAILLE PATTERNS block (U+2800–28FF).
// It carries SHAPE at 2×4 with ONE ink colour (fg = the average of the ON dots) over
// a backdrop (bg), so an alpha cut-out reads as a sharp monochrome figure at four
// sub-rows / two sub-columns per cell — finer than the 2×2 quadrant, at the cost of
// the per-quadrant second colour.

/// The Unicode BRAILLE-PATTERN dot-bit for a `(col, row)` cell of the 2×4 grid
/// (`col ∈ {0,1}` left/right, `row ∈ {0,1,2,3}` top→bottom). The braille glyph is
/// `char::from_u32(0x2800 + mask)`, where `mask` ORs these bits for every ON dot.
///
/// The historical braille dot numbering is column-major then the two low cells last:
/// ```text
///   dot1 (0x01)  dot4 (0x08)
///   dot2 (0x02)  dot5 (0x10)
///   dot3 (0x04)  dot6 (0x20)
///   dot7 (0x40)  dot8 (0x80)
/// ```
/// so the left column is rows 0..3 → bits 0x01,0x02,0x04,0x40 and the right column is
/// rows 0..3 → bits 0x08,0x10,0x20,0x80. This table encodes exactly that, indexed
/// `[row][col]`.
pub const BRAILLE_DOT_BITS: [[u8; 2]; 4] = [
    [0x01, 0x08], // row 0 (top):    dot1, dot4
    [0x02, 0x10], // row 1:          dot2, dot5
    [0x04, 0x20], // row 2:          dot3, dot6
    [0x40, 0x80], // row 3 (bottom): dot7, dot8
];

/// Map an 8-dot Braille coverage `mask` (built from [`BRAILLE_DOT_BITS`]) to its
/// glyph in the Braille Patterns block. `mask == 0` is the blank pattern `U+2800`
/// (no dots). Always valid: `0x2800..=0x28FF` are all assigned, so the `from_u32`
/// never yields `None` (a defensive fallback to space keeps it total).
#[inline]
pub fn braille_glyph(mask: u8) -> char {
    char::from_u32(0x2800 + u32::from(mask)).unwrap_or(' ')
}

/// One ON/OFF coverage dot for the [`blit_braille`] grid: whether the sub-texel is
/// OPAQUE (ON — part of the figure) and, if so, the colour it contributes to the
/// single-ink average. A transparent (OFF) dot contributes nothing to the ink and
/// leaves its position showing the backdrop (`bg`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BrailleDot {
    /// `true` ⇒ the sub-texel is opaque (a lit dot); `false` ⇒ transparent (backdrop).
    pub on: bool,
    /// The opaque sub-texel's colour (`0x00RRGGBB`), averaged into `fg`. Ignored when
    /// `on` is false.
    pub rgb: u32,
}

/// The result of [`blit_braille`]: a 2×4 coverage glyph + a single-ink/backdrop
/// colour split. `glyph` is one of the 256 Braille patterns; `fg` inks the ON dots
/// (the average of their colours); `bg` is the backdrop the OFF dots reveal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BrailleBlit {
    /// One of the 256 Braille-pattern glyphs (`0x2800 + mask`).
    pub glyph: u32,
    /// Foreground ink — the average colour of the ON (opaque) dots. When no dot is ON
    /// (the all-transparent cell) this equals `bg` (the glyph is blank anyway).
    pub fg: u32,
    /// Backdrop colour the OFF (transparent) dots show — what is behind the figure.
    pub bg: u32,
}

/// Map the EIGHT shaded sub-dots (`grid[row][col]`, `row ∈ 0..4`, `col ∈ 0..2`,
/// top-to-bottom / left-to-right) plus a `backdrop` colour to one [`BrailleBlit`]:
/// a 2×4 monochrome coverage glyph + a single ink / backdrop split.
///
/// Algorithm (pure, deterministic):
/// 1. For each ON dot, set its [`BRAILLE_DOT_BITS`] bit in the mask and add its colour
///    to the ink average.
/// 2. `glyph = 0x2800 + mask`; `fg` = the average of the ON dots (or `backdrop` if
///    none are ON); `bg = backdrop`.
///
/// This carries the figure's SHAPE at 2×4 sub-resolution with one ink colour, exactly
/// the alpha-cut-out high-detail mode: a transparent sub-texel is an OFF dot showing
/// the backdrop, an opaque one a lit dot in the figure's average colour.
#[inline]
pub fn blit_braille(grid: [[BrailleDot; 2]; 4], backdrop: u32) -> BrailleBlit {
    let mut mask = 0u8;
    let mut ink = [0u32; 3];
    let mut on_n = 0u32;
    for (row, cols) in grid.iter().enumerate() {
        for (col, dot) in cols.iter().enumerate() {
            if dot.on {
                mask |= BRAILLE_DOT_BITS[row][col];
                accumulate(&mut ink, dot.rgb);
                on_n += 1;
            }
        }
    }
    let fg = average(ink, on_n).unwrap_or(backdrop);
    BrailleBlit {
        glyph: braille_glyph(mask) as u32,
        fg,
        bg: backdrop,
    }
}

// ─────────────────────────────── tests ──────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(depth: f32, fg: u32, bg: u32, glyph: char) -> SubSample {
        SubSample {
            depth,
            fg,
            bg,
            glyph: u32::from(glyph),
        }
    }

    // ───────────────── half-block + sextant (the added ladder rungs) ─────────────

    /// The three points of the sextant encoding that were verified against the
    /// published standard. If the skip rule is ever mis-implemented these are the
    /// assertions that catch it, because each one sits on the far side of a
    /// different number of skipped masks (0, 1 and 3 respectively).
    #[test]
    fn sextant_codepoints_match_the_unicode_standard() {
        assert_eq!(
            sextant_glyph(0b000001),
            '\u{1FB00}',
            "mask 1 = BLOCK SEXTANT-1"
        );
        assert_eq!(
            sextant_glyph(29),
            '\u{1FB1B}',
            "mask 29 = BLOCK SEXTANT-1345"
        );
        assert_eq!(
            sextant_glyph(62),
            '\u{1FB3B}',
            "mask 62 = BLOCK SEXTANT-23456"
        );
    }

    /// The four masks Unicode did NOT encode in the sextant block, because they
    /// already existed. Getting any of these wrong yields a hole in the middle of an
    /// otherwise-correct mosaic.
    #[test]
    fn sextant_reuses_the_four_pre_existing_block_elements() {
        assert_eq!(sextant_glyph(0b000000), ' ', "empty");
        assert_eq!(
            sextant_glyph(0b010101),
            '▌',
            "left column = LEFT HALF BLOCK"
        );
        assert_eq!(
            sextant_glyph(0b101010),
            '▐',
            "right column = RIGHT HALF BLOCK"
        );
        assert_eq!(sextant_glyph(0b111111), '█', "full");
    }

    /// Every mask maps to a distinct glyph, all 60 encoded ones land inside
    /// `U+1FB00..=U+1FB3B`, and nothing escapes the block. This is the property the
    /// computed-index approach exists to guarantee — a hand-typed 60-entry table
    /// could satisfy the spot checks above and still collide somewhere in the middle.
    #[test]
    fn sextant_is_a_bijection_into_the_encoded_range() {
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        for m in 0u8..64 {
            let g = sextant_glyph(m);
            assert!(seen.insert(g), "mask {m} collides with an earlier glyph");
            let cp = u32::from(g);
            let special = matches!(g, ' ' | '▌' | '▐' | '█');
            assert!(
                special || (SEXTANT_BASE..=0x1FB3B).contains(&cp),
                "mask {m} produced U+{cp:04X}, outside the sextant range"
            );
        }
        assert_eq!(seen.len(), 64, "all 64 masks are distinct");
        // High bits are ignored rather than aliasing into a wrong glyph.
        assert_eq!(sextant_glyph(0b1100_0001), sextant_glyph(0b0000_0001));
    }

    /// `blit_blocks` is the generalisation of `blit_quadrant`, and `blit_quadrant` is
    /// pinned by the golden corpus — so rather than refactor the hashed path, the two
    /// are proven equivalent here over a spread of depth/colour configurations.
    #[test]
    fn blit_blocks_matches_blit_quadrant_on_four() {
        let cases: [[SubSample; 4]; 4] = [
            // A real silhouette: two near, two far.
            [
                sample(1.0, 0xFF0000, 0x100000, '#'),
                sample(1.0, 0xFF0000, 0x100000, '#'),
                sample(9.0, 0x00FF00, 0x001000, '.'),
                sample(9.0, 0x00FF00, 0x001000, '.'),
            ],
            // Flat cell (falsely marked) — all four equal.
            [sample(3.0, 0x808080, 0x101010, '#'); 4],
            // One near corner against three far.
            [
                sample(0.5, 0xFFFFFF, 0x000000, '@'),
                sample(7.0, 0x202020, 0x010101, ' '),
                sample(7.0, 0x202020, 0x010101, ' '),
                sample(7.0, 0x202020, 0x010101, ' '),
            ],
            // A miss (open sky) mixed with real hits.
            [
                sample(2.0, 0x334455, 0x111111, '#'),
                sample(f32::INFINITY, 0x000000, 0x000000, ' '),
                sample(2.0, 0x334455, 0x111111, '#'),
                sample(f32::INFINITY, 0x000000, 0x000000, ' '),
            ],
        ];
        for (i, c) in cases.into_iter().enumerate() {
            let want = blit_quadrant(c);
            let got = blit_blocks(&c, |m| QUADRANT_GLYPHS[(m & 0x0F) as usize]);
            assert_eq!(got, want, "case {i}: the generalisation must not drift");
        }
    }

    /// Half-block splits top/bottom on depth and keeps a colour per half — the two
    /// properties that make it worth having below quadrant.
    #[test]
    fn blit_half_splits_top_from_bottom() {
        let b = blit_half([
            sample(1.0, 0xFF0000, 0x000000, '#'),
            sample(8.0, 0x0000FF, 0x000000, '#'),
        ]);
        assert_eq!(
            b.glyph,
            u32::from('▀'),
            "near half is the TOP → upper half block"
        );
        assert_eq!(b.fg, 0xFF0000, "the near half keeps its own colour");
        assert_eq!(b.bg, 0x0000FF, "the far half keeps its own colour");
        assert_eq!(
            b.depth, 1.0,
            "the cell's depth cue is the nearest sub-sample"
        );

        // Reversed: near is the bottom.
        let b = blit_half([
            sample(8.0, 0x0000FF, 0x000000, '#'),
            sample(1.0, 0xFF0000, 0x000000, '#'),
        ]);
        assert_eq!(
            b.glyph,
            u32::from('▄'),
            "near half is the BOTTOM → lower half block"
        );
        assert_eq!(b.fg, 0xFF0000);

        // Flat cell → full block, both colours equal (never a blank hole).
        let b = blit_half([sample(4.0, 0x123456, 0x000000, '#'); 2]);
        assert_eq!(b.glyph, u32::from('█'));
        assert_eq!(b.fg, b.bg);
    }

    /// Sextant resolves a horizontal band a quadrant cannot: a middle row distinct
    /// from top and bottom. This is the concrete reason for the extra rung.
    #[test]
    fn blit_sextant_resolves_a_middle_band_quadrant_cannot() {
        // Near MIDDLE row only, far top and bottom → mask 0b001100 = ML+MR.
        let near = sample(1.0, 0xFFFFFF, 0x000000, '#');
        let far = sample(9.0, 0x202020, 0x000000, '.');
        let b = blit_sextant([far, far, near, near, far, far]);
        assert_eq!(b.glyph, u32::from(sextant_glyph(0b001100)));
        assert_eq!(b.fg, 0xFFFFFF, "the middle band keeps its own colour");
        assert_eq!(
            b.bg, 0x202020,
            "and the rest keeps its own — TWO colours, not one"
        );
        assert_ne!(
            b.glyph,
            u32::from(' '),
            "a middle-only band must not vanish"
        );
    }

    /// The claim that justifies preferring sextant over Braille for images: it is a
    /// BLOCK geometry, so it still carries two independent colours per cell, where
    /// `BrailleBlit` gives all eight dots a single ink.
    #[test]
    fn sextant_luminance_keeps_two_colours_where_braille_keeps_one() {
        let lit = ImageSub {
            rgb: 0xF0F0F0,
            opaque: true,
        };
        let dark = ImageSub {
            rgb: 0x203040,
            opaque: true,
        };
        let b = blit_sextant_luminance([[lit, lit], [dark, dark], [dark, dark]], 0x000000);
        assert_eq!(b.fg, 0xF0F0F0, "bright texels ink the foreground");
        assert_eq!(
            b.bg, 0x203040,
            "dark texels keep their OWN colour as background"
        );
        assert_ne!(b.fg, b.bg, "two distinct colours survive the cell");
        assert_eq!(b.glyph, u32::from(sextant_glyph(0b000011)), "top row inked");
    }

    /// A transparent texel is outside the figure: it takes the backdrop and never
    /// inks, so the alpha edge stays sharp — same contract as the quadrant sampler.
    #[test]
    fn block_luminance_keeps_transparent_texels_out_of_the_figure() {
        let solid = ImageSub {
            rgb: 0xFFFFFF,
            opaque: true,
        };
        let clear = ImageSub {
            rgb: 0xFFFFFF,
            opaque: false,
        };
        let b = blit_half_luminance([solid, clear], 0x123456);
        assert_eq!(b.glyph, u32::from('▀'), "only the opaque half inks");
        assert_eq!(b.bg, 0x123456, "the transparent half shows the backdrop");
    }

    // ── classifier ───────────────────────────────────────────────────────────

    #[test]
    fn classifier_marks_a_silhouette_edge() {
        // A near pillar (depth 2) against a far floor (depth 10) — a strong
        // depth discontinuity must be marked.
        assert!(
            needs_subcell(2.0, &[2.0, 10.0, 2.1], false),
            "a near surface beside a far one must be marked"
        );
    }

    #[test]
    fn classifier_skips_a_flat_region() {
        // A flat wall: neighbour depths vary only slightly (foreshortening), well
        // under the ratio threshold — must NOT be marked.
        assert!(
            !needs_subcell(5.0, &[5.0, 5.05, 4.97, 5.02], false),
            "a flat region must not be marked"
        );
    }

    #[test]
    fn classifier_marks_surface_against_open_sky() {
        // This cell hit a wall (finite depth); a neighbour hit nothing (sky) — a
        // silhouette against open space, must be marked.
        assert!(needs_subcell(4.0, &[4.0, f32::INFINITY], false));
        // But a cell that ITSELF hit nothing is never marked (no surface).
        assert!(!needs_subcell(f32::INFINITY, &[4.0, 4.0], false));
    }

    #[test]
    fn classifier_honours_material_opt_in() {
        // Even a perfectly flat cell is marked if the material opts in.
        assert!(needs_subcell(5.0, &[5.0, 5.0], true));
    }

    #[test]
    fn classifier_ratio_is_relative_not_absolute() {
        // The SAME 0.5-cell absolute step is a silhouette up close but smooth far
        // away — the ratio threshold judges them differently.
        assert!(
            needs_subcell(1.0, &[1.5], false),
            "0.5 step at depth 1 (50%) is an edge"
        );
        assert!(
            !needs_subcell(20.0, &[20.5], false),
            "0.5 step at depth 20 (2.5%) is smooth"
        );
    }

    // ── blit: known patterns → glyph + colours ───────────────────────────────

    #[test]
    fn blit_upper_half_pattern() {
        // TL+TR near (depth 1), BL+BR far (depth 10) → upper-half ▀.
        // fg = near ink (red glyphs), bg = far ink (blue fill).
        let red = 0x00FF0000;
        let blue = 0x000000FF;
        let s = [
            sample(1.0, red, 0, '#'),   // TL near, glyph → ink = fg = red
            sample(1.0, red, 0, '#'),   // TR near
            sample(10.0, 0, blue, ' '), // BL far, blank → ink = bg = blue
            sample(10.0, 0, blue, ' '), // BR far
        ];
        let out = blit_quadrant(s);
        assert_eq!(out.glyph, u32::from('▀'), "TL+TR near → upper half");
        assert_eq!(out.fg, red, "fg = near (red glyph ink)");
        assert_eq!(out.bg, blue, "bg = far (blue fill ink)");
        assert_eq!(out.depth, 1.0, "depth = nearest sub-sample");
    }

    #[test]
    fn blit_left_half_pattern() {
        // TL+BL near, TR+BR far → left half ▌.
        let s = [
            sample(2.0, 0x00112233, 0, 'x'),
            sample(9.0, 0, 0x00445566, ' '),
            sample(2.0, 0x00112233, 0, 'x'),
            sample(9.0, 0, 0x00445566, ' '),
        ];
        let out = blit_quadrant(s);
        assert_eq!(out.glyph, u32::from('▌'), "TL+BL near → left half");
    }

    #[test]
    fn blit_main_diagonal_pattern() {
        // TL+BR near, TR+BL far → main diagonal ▚.
        let s = [
            sample(1.0, 0x00AABBCC, 0, '#'),
            sample(8.0, 0, 0x00010203, ' '),
            sample(8.0, 0, 0x00010203, ' '),
            sample(1.0, 0x00AABBCC, 0, '#'),
        ];
        let out = blit_quadrant(s);
        assert_eq!(out.glyph, u32::from('▚'), "TL+BR near → main diagonal");
    }

    #[test]
    fn blit_single_corner_pattern() {
        // Only BR near → ▗.
        let s = [
            sample(9.0, 0, 0x00111111, ' '),
            sample(9.0, 0, 0x00111111, ' '),
            sample(9.0, 0, 0x00111111, ' '),
            sample(1.0, 0x00FF00FF, 0, '@'),
        ];
        let out = blit_quadrant(s);
        assert_eq!(
            out.glyph,
            u32::from('▗'),
            "only BR near → lower-right corner"
        );
        assert_eq!(out.fg, 0x00FF00FF, "fg = the single near ink");
    }

    #[test]
    fn blit_flat_cell_is_full_block() {
        // All four equal depth (a uniform cell that was marked by a neighbour but is
        // itself flat) → every sub-cell near → █, fg == bg == the surface colour.
        let c = 0x00334455;
        let s = [
            sample(5.0, c, 0, 'o'),
            sample(5.0, c, 0, 'o'),
            sample(5.0, c, 0, 'o'),
            sample(5.0, c, 0, 'o'),
        ];
        let out = blit_quadrant(s);
        assert_eq!(out.glyph, u32::from('█'), "uniform depth → full block");
        assert_eq!(out.fg, c);
        assert_eq!(out.bg, c, "no far group → bg falls back to fg");
    }

    #[test]
    fn blit_colour_split_averages_each_pair() {
        // Two near samples with different inks → fg is their average; likewise bg.
        let s = [
            sample(1.0, 0x00FF0000, 0, '#'), // near, ink red
            sample(1.0, 0x00000000, 0, '#'), // near, ink black
            sample(9.0, 0, 0x0000FF00, ' '), // far, ink green
            sample(9.0, 0, 0x00000000, ' '), // far, ink black
        ];
        let out = blit_quadrant(s);
        // near average = (255+0)/2 in R = 127.
        assert_eq!((out.fg >> 16) & 0xFF, 127, "near fg averages the two inks");
        // far average = (255+0)/2 in G = 127.
        assert_eq!((out.bg >> 8) & 0xFF, 127, "far bg averages the two inks");
    }

    #[test]
    fn blit_is_deterministic() {
        let s = [
            sample(1.0, 0x00ABCDEF, 0, '#'),
            sample(7.0, 0, 0x00123456, ' '),
            sample(1.0, 0x00ABCDEF, 0, '#'),
            sample(7.0, 0, 0x00123456, ' '),
        ];
        assert_eq!(
            blit_quadrant(s),
            blit_quadrant(s),
            "blit must be deterministic"
        );
    }

    #[test]
    fn every_quadrant_glyph_is_distinct_for_distinct_mask() {
        // The 16 entries: only the two "flat" ends repeat structure; all glyphs are
        // the intended block-element set (or space). Sanity: index 0 is space, 15 is
        // full block, and no interior index is space/full-block.
        assert_eq!(QUADRANT_GLYPHS[0], ' ');
        assert_eq!(QUADRANT_GLYPHS[15], '█');
        for g in &QUADRANT_GLYPHS[1..15] {
            assert!(
                *g != ' ' && *g != '█',
                "interior glyph must be a partial block: {g}"
            );
        }
    }

    // ── braille: 8-dot mask → U+28xx glyph + ink/backdrop ─────────────────────

    fn dot(on: bool, rgb: u32) -> BrailleDot {
        BrailleDot { on, rgb }
    }

    #[test]
    fn braille_blank_mask_is_u2800() {
        assert_eq!(braille_glyph(0x00), '\u{2800}');
    }

    #[test]
    fn braille_full_mask_is_u28ff() {
        assert_eq!(braille_glyph(0xFF), '\u{28FF}');
    }

    #[test]
    fn braille_dot_bits_cover_every_distinct_bit() {
        // The eight dot positions must map to the eight DISTINCT braille bits 0x01..0x80.
        let mut seen = 0u8;
        for row in &BRAILLE_DOT_BITS {
            for &bit in row {
                assert_eq!(bit.count_ones(), 1, "each dot is a single bit");
                assert_eq!(seen & bit, 0, "dot bits must not overlap");
                seen |= bit;
            }
        }
        assert_eq!(seen, 0xFF, "the eight dots cover all eight braille bits");
    }

    #[test]
    fn braille_top_left_dot_only_is_u2801() {
        // Only the top-left dot ON (dot1, bit 0x01) → U+2801.
        let backdrop = 0x00112233;
        let mut grid = [[dot(false, 0); 2]; 4];
        grid[0][0] = dot(true, 0x00FF0000);
        let out = blit_braille(grid, backdrop);
        assert_eq!(out.glyph, u32::from('\u{2801}'), "only dot1 → U+2801");
        assert_eq!(out.fg, 0x00FF0000, "fg = the single ON dot's colour");
        assert_eq!(out.bg, backdrop, "bg = the backdrop");
    }

    #[test]
    fn braille_bottom_right_dot_is_dot8() {
        // Bottom-right (row 3, col 1) is dot8 (0x80) → U+2880.
        let mut grid = [[dot(false, 0); 2]; 4];
        grid[3][1] = dot(true, 0x00ABCDEF);
        let out = blit_braille(grid, 0);
        assert_eq!(out.glyph, u32::from('\u{2880}'), "row3/col1 → dot8 (0x80)");
    }

    #[test]
    fn braille_left_column_is_dots_1_2_3_7() {
        // The whole LEFT column ON → bits 0x01|0x02|0x04|0x40 = 0x47 → U+2847.
        let mut grid = [[dot(false, 0); 2]; 4];
        for row in grid.iter_mut() {
            row[0] = dot(true, 0x00808080);
        }
        let out = blit_braille(grid, 0);
        assert_eq!(out.glyph, u32::from('\u{2847}'), "left column → 0x47");
    }

    #[test]
    fn braille_transparent_dot_reads_as_backdrop() {
        // A mix: one ON dot, the rest transparent — the OFF dots must NOT ink, and the
        // backdrop must survive as bg.
        let backdrop = 0x00204060;
        let mut grid = [[dot(false, 0x00FFFFFF); 2]; 4]; // OFF dots carry junk rgb
        grid[1][1] = dot(true, 0x0000FF00);
        let out = blit_braille(grid, backdrop);
        assert_eq!(
            out.fg, 0x0000FF00,
            "fg ignores OFF-dot colours, = the lone ON dot"
        );
        assert_eq!(out.bg, backdrop, "OFF dots reveal the backdrop");
    }

    #[test]
    fn braille_ink_averages_the_on_dots() {
        // Two ON dots with different colours → fg is their per-channel average.
        let mut grid = [[dot(false, 0); 2]; 4];
        grid[0][0] = dot(true, 0x00FF0000); // red
        grid[0][1] = dot(true, 0x00000000); // black
        let out = blit_braille(grid, 0x00123456);
        assert_eq!(
            (out.fg >> 16) & 0xFF,
            127,
            "ON-dot ink averages (255+0)/2 = 127"
        );
    }

    #[test]
    fn braille_all_transparent_is_blank_glyph_over_backdrop() {
        let backdrop = 0x00777777;
        let grid = [[dot(false, 0x00FFFFFF); 2]; 4];
        let out = blit_braille(grid, backdrop);
        assert_eq!(
            out.glyph,
            u32::from('\u{2800}'),
            "no ON dots → blank braille"
        );
        assert_eq!(
            out.fg, backdrop,
            "fg falls back to backdrop when nothing inks"
        );
        assert_eq!(out.bg, backdrop);
    }

    #[test]
    fn braille_blit_is_deterministic() {
        let mut grid = [[dot(false, 0); 2]; 4];
        grid[0][0] = dot(true, 0x00ABCDEF);
        grid[2][1] = dot(true, 0x00123456);
        assert_eq!(
            blit_braille(grid, 0x00010203),
            blit_braille(grid, 0x00010203),
            "braille blit must be deterministic"
        );
    }

    // ── image interior-detail: luminance-split quad / braille ─────────────────

    fn isub(opaque: bool, rgb: u32) -> ImageSub {
        ImageSub { opaque, rgb }
    }

    #[test]
    fn quad_lum_two_tones_split_not_flat_block() {
        // Two distinct OPAQUE tones — TL/TR light (white), BL/BR dark (near black). The
        // mean luminance lands between them, so the light pair is foreground and the dark
        // pair background → an UPPER-HALF block, not a flat █. This is interior detail.
        let white = 0x00FFFFFF;
        let dark = 0x00101010;
        let subs = [
            isub(true, white), // TL
            isub(true, white), // TR
            isub(true, dark),  // BL
            isub(true, dark),  // BR
        ];
        let out = blit_quadrant_luminance(subs, 0x00000000);
        assert_eq!(
            out.glyph,
            u32::from('▀'),
            "light-top/dark-bottom → upper half"
        );
        assert_ne!(out.glyph, u32::from('█'), "must NOT be a flat block");
        assert_eq!(out.fg, white, "fg = the light interior tone");
        assert_eq!(out.bg, dark, "bg = the dark interior tone");
    }

    #[test]
    fn quad_lum_flat_interior_is_full_block() {
        // A uniform opaque interior → every sub-texel ≥ mean → all foreground → █.
        let c = 0x00446688;
        let subs = [isub(true, c); 4];
        let out = blit_quadrant_luminance(subs, 0x00000000);
        assert_eq!(out.glyph, u32::from('█'), "flat interior → full block");
        assert_eq!(out.fg, c, "fg = the interior colour");
    }

    #[test]
    fn quad_lum_transparent_goes_to_backdrop_edge() {
        // The alpha EDGE: TL/TR opaque (figure), BL/BR transparent (outside). The
        // transparent pair must read the backdrop in bg and NOT be foreground, so the
        // silhouette stays sharp.
        let body = 0x00C0C0C0;
        let backdrop = 0x00203040;
        let subs = [
            isub(true, body), // TL
            isub(true, body), // TR
            isub(false, 0),   // BL transparent
            isub(false, 0),   // BR transparent
        ];
        let out = blit_quadrant_luminance(subs, backdrop);
        assert_eq!(
            out.glyph,
            u32::from('▀'),
            "opaque-top/transparent-bottom → upper half"
        );
        assert_eq!(
            out.bg, backdrop,
            "transparent sub-texels reveal the backdrop"
        );
    }

    #[test]
    fn quad_lum_is_deterministic() {
        let subs = [
            isub(true, 0x00FFEEDD),
            isub(true, 0x00112233),
            isub(false, 0),
            isub(true, 0x00445566),
        ];
        assert_eq!(
            blit_quadrant_luminance(subs, 0x00010203),
            blit_quadrant_luminance(subs, 0x00010203),
            "luminance quad blit must be deterministic"
        );
    }

    #[test]
    fn braille_lum_light_and_dark_gives_nontrivial_mask() {
        // Half the opaque texels are dark (below mean → ink/ON), half light (above → OFF).
        // The result must be a non-blank, non-full mask: real interior halftone.
        let light = 0x00F0F0F0;
        let dark = 0x00202020;
        let mut grid = [[isub(true, light); 2]; 4];
        // Make the bottom two rows dark (ink).
        grid[2] = [isub(true, dark); 2];
        grid[3] = [isub(true, dark); 2];
        let out = blit_braille_luminance(grid, 0x00000000);
        let mask = (out.glyph - 0x2800) as u8;
        assert_ne!(mask, 0x00, "dark interior must ink some dots");
        assert_ne!(mask, 0xFF, "light interior must leave some dots off");
        // Darker = ink: the dark (bottom) rows are the ON dots → fg ≈ dark tone.
        assert_eq!(out.fg, dark, "fg = the dark ink tone (darker=ink polarity)");
    }

    #[test]
    fn braille_lum_transparent_dots_off_and_reveal_backdrop() {
        // Alpha edge: one opaque-dark texel inks; the rest transparent → OFF, backdrop bg.
        let backdrop = 0x00305070;
        let mut grid = [[isub(false, 0x00FFFFFF); 2]; 4]; // transparent w/ junk rgb
        grid[1][1] = isub(true, 0x00101010); // a dark opaque interior dot
        let out = blit_braille_luminance(grid, backdrop);
        let mask = (out.glyph - 0x2800) as u8;
        assert_eq!(
            mask, BRAILLE_DOT_BITS[1][1],
            "only the opaque-dark dot inks"
        );
        assert_eq!(out.bg, backdrop, "transparent dots reveal the backdrop");
        assert_eq!(out.fg, 0x00101010, "fg = the lone ink dot");
    }

    #[test]
    fn braille_lum_flat_solid_figure_inks_full_block_not_blank() {
        // A uniform OPAQUE interior has no tonal detail (nothing below the mean), so the
        // darker=ink rule would ink nothing. The SOLID-FIGURE SAFETY fallback then inks
        // ALL opaque dots → a full braille block (U+28FF) in the figure colour, so a flat
        // bright figure on a dark backdrop is never rendered invisible.
        let c = 0x00808080;
        let grid = [[isub(true, c); 2]; 4];
        let out = blit_braille_luminance(grid, 0x00000000);
        assert_eq!(
            out.glyph,
            u32::from('\u{28FF}'),
            "flat solid figure → full coverage, not blank"
        );
        assert_eq!(out.fg, c, "fg = the figure colour");
    }

    #[test]
    fn braille_lum_solid_figure_on_dark_backdrop_stays_visible() {
        // The inversion case the polarity note warns about: a bright RED figure (luma ≈ 76)
        // on a BLACK backdrop. Darker=ink alone would blank it; the fallback inks all the
        // opaque dots so the silhouette survives, with the transparent edge still OFF.
        let red = 0x00FF0000;
        let mut grid = [[isub(false, 0); 2]; 4]; // right column transparent (edge)
        for cols in &mut grid {
            cols[0] = isub(true, red); // left column = solid red figure
        }
        let out = blit_braille_luminance(grid, 0x00000000);
        let mask = (out.glyph - 0x2800) as u8;
        assert_ne!(mask, 0x00, "a solid figure must light dots, never blank");
        assert_eq!(out.fg, red, "fg = the figure colour");
        // Only the LEFT column inks (the right column is transparent edge).
        assert_eq!(
            mask,
            0x01 | 0x02 | 0x04 | 0x40,
            "only the opaque left column inks"
        );
        assert_eq!(
            out.bg, 0x00000000,
            "transparent right column reveals the backdrop"
        );
    }

    #[test]
    fn braille_lum_is_deterministic() {
        let mut grid = [[isub(true, 0x00C0C0C0); 2]; 4];
        grid[0][0] = isub(true, 0x00101010);
        grid[3][1] = isub(false, 0);
        assert_eq!(
            blit_braille_luminance(grid, 0x00010203),
            blit_braille_luminance(grid, 0x00010203),
            "luminance braille blit must be deterministic"
        );
    }
}
