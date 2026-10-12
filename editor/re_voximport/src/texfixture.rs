//! A procedurally-generated, freely-licensed PNG test plate for the image-texture
//! sampler.
//!
//! # Why this exists
//!
//! The only golden anchors for the `[texture] → recipe = image` wall sampler
//! (`levels/TX1-asuka.lvl`, `levels/TX4-asuka-subcell.lvl`) bind
//! `assets/texture/asuka_tex.png` — copyrighted fan art that does not ship with the
//! game. Removing that one file does **not** make those levels fail to load: the
//! `[texture]` decode is best-effort (see
//! [`re_level::Level::build_map`](re_level::Level::build_map)), so the level renders
//! its untextured procedural fallback and the golden hashes keep matching *each
//! other*. The sampler loses all coverage and nothing goes red.
//!
//! This module is the free replacement anchor. It generates
//! `assets/texture/genplate_tex.png` from code that lives in the repository, so the
//! asset is reproducible rather than an opaque blob, and it can never be a licensing
//! problem.
//!
//! # Regenerating the asset
//!
//! ```sh
//! cargo run -p re_voximport --bin gen_texfixture -- assets/texture/genplate_tex.png
//! ```
//!
//! Run from the workspace root. The generator is a pure function of nothing —
//! [`crate::texfixture::plate_rgba`] takes no input and uses only integer
//! arithmetic, so it produces byte-identical pixels on every platform and every
//! run. The crate's unit test `the_committed_png_matches_the_generator` decodes
//! the committed file and compares it pixel-for-pixel with a fresh
//! `plate_rgba()`, so the asset cannot drift away from its source.
//!
//! (The test compares DECODED PIXELS, not file bytes: the PNG container's exact
//! compressed bytes are an artefact of the `png` encoder's version and settings,
//! whereas the pixels are what the game actually samples.)
//!
//! # What the plate contains, and why
//!
//! The three properties below are not aesthetic choices — each one is what makes a
//! specific rendering path measurable in a dumped frame. A generic small texture
//! would carry none of them.
//!
//! 1. **An alpha channel with a non-trivial silhouette.** The plate is a figure
//!    (round head, tapering body, a rectangular window punched clean through it) on a
//!    fully transparent field. The image-cut-out path in `re_render` re-casts the ray
//!    behind any transparent texel, so this is what turns a wall into a cut-out pane
//!    and what makes the `quad`/`braille` blitters composite against a backdrop. A
//!    rectangular or fully-opaque plate exercises none of that.
//! 2. **Saturated, non-grey colour** — specifically a warm (red-dominant) ramp over
//!    most of its area. The untextured procedural fallback for these wall surfaces
//!    produces *zero* warm-foreground cells, so "there are warm cells on screen" is
//!    the single measurement that distinguishes "the image sampler ran" from "a
//!    procedural recipe ran". A greyscale plate would be indistinguishable from the
//!    fallback in exactly that channel, i.e. it would recreate the hole it is meant
//!    to close.
//! 3. **Detail at two scales.** The left half is a coarse 16-px checker; the right
//!    half is a fine 2-px checker (finer than one screen cell's texel footprint at
//!    normal viewing distance), and a hard-edged bright bar crosses both. So
//!    `subcell = off` (one solid block per cell), `subcell = quad` (2×2) and
//!    `subcell = braille` (2×4) resolve visibly *different* densities — the fine half
//!    yields many distinct partial coverage patterns while the coarse half yields
//!    solid runs. A single smooth blob would render as three copies of itself.
//!
//! Pure compute: integer pixel math plus a PNG encode. No terminal, no renderer.

use std::io::Cursor;

// ─────────────────────────────────────────────────────────────────────────────────
// `gen_anim.gif` — the free, generated ANIMATED test clip
// ─────────────────────────────────────────────────────────────────────────────────
//
// The only anchors for `[texture] frames = <clip.gif>` (`re_level`'s animated
// image-texture path) used to bind a real downloaded GIF of unknown origin,
// untracked and unlicensed for redistribution. Deleting it left the tests that
// named it unable to run at all on a clean clone (the comment claiming it was
// "repo-tracked" was false). This is the free replacement, generated from code
// exactly like
// [`plate_rgba`]/[`plate_png`] above.

/// Edge length of each frame, in texels. Small and non-square, so a `max_size` cap
/// test can assert BOTH the cap and aspect preservation without decoding megabytes.
pub const ANIM_W: u32 = 48;
/// Edge length of each frame, in texels (see [`ANIM_W`]).
pub const ANIM_H: u32 = 24;

/// Number of frames in the generated clip. More than one (so the animation is
/// genuinely animated) and small enough that 8 distinct, verifiably-different
/// frames are cheap to assert over.
pub const ANIM_FRAMES: u32 = 8;

/// Frame `i`'s RGBA8 buffer (tightly packed, top-left origin, row-major).
///
/// Deterministic and platform-independent: integer arithmetic only. Each frame is a
/// solid colour that steps across the clip (so `frame[i] != frame[j]` for `i != j`,
/// the property every frame-count/frame-index test needs) plus a single-texel-wide
/// bright column that sweeps left-to-right across the frame sequence, so a naive
/// "always return frame 0" decoder is caught by BOTH the colour and the column
/// position.
#[must_use]
pub fn anim_frame_rgba(i: u32) -> Vec<u8> {
    debug_assert!(i < ANIM_FRAMES);
    let v = (i * 255 / (ANIM_FRAMES - 1)) as u8;
    let sweep_x = i * ANIM_W / ANIM_FRAMES;
    let mut out = Vec::with_capacity((ANIM_W * ANIM_H * 4) as usize);
    for _row in 0..ANIM_H {
        for x in 0..ANIM_W {
            if x == sweep_x {
                out.extend_from_slice(&[255, 255, 255, 255]);
            } else {
                out.extend_from_slice(&[v, 255 - v, 128, 255]);
            }
        }
    }
    out
}

/// Encode all [`ANIM_FRAMES`] via [`anim_frame_rgba`] as an animated GIF.
///
/// # Errors
///
/// Returns the encoder's message on failure (malformed buffer or I/O error into
/// the in-memory `Vec`, which cannot actually happen, but the generator binary
/// reports it rather than panicking, matching [`plate_png`]).
pub fn anim_gif() -> Result<Vec<u8>, String> {
    use image::codecs::gif::GifEncoder;
    let mut out: Vec<u8> = Vec::new();
    {
        let mut enc = GifEncoder::new(&mut out);
        for i in 0..ANIM_FRAMES {
            let buf: image::RgbaImage =
                image::RgbaImage::from_raw(ANIM_W, ANIM_H, anim_frame_rgba(i)).ok_or_else(
                    || "anim_frame_rgba() produced a buffer of the wrong length".to_string(),
                )?;
            enc.encode_frame(image::Frame::new(buf))
                .map_err(|e| format!("GIF frame {i} encode failed: {e}"))?;
        }
    }
    Ok(out)
}

/// Edge length of the generated plate, in texels. Square.
///
/// 64 is the smallest size that still carries the fine/coarse split: the fine half
/// needs a 2-px feature (16 alternations across 32 columns) that is smaller than one
/// screen cell's texel footprint, and the coarse half needs a 16-px feature that is
/// comfortably larger than it. The whole PNG lands in a few KB.
pub const PLATE_SIZE: u32 = 64;

/// Number of bytes in the generated RGBA8 buffer.
const PLATE_BYTES: usize = (PLATE_SIZE * PLATE_SIZE * 4) as usize;

// ── Silhouette geometry (property 1) ────────────────────────────────────────────
// All in texel units on the 64×64 plate, top-left origin.

/// Centre column of the figure (also the plate's centre).
const HEAD_CX: i32 = 32;
/// Centre row of the round head.
const HEAD_CY: i32 = 15;
/// Head radius.
const HEAD_R: i32 = 12;

/// First and last row of the body (inclusive).
const BODY_TOP: i32 = 24;
const BODY_BOTTOM: i32 = 61;
/// Body half-width at [`BODY_TOP`] and at [`BODY_BOTTOM`]; interpolated linearly
/// between, which is what makes the left/right edges slope instead of being a
/// straight vertical cut.
const BODY_HALF_TOP: i32 = 9;
const BODY_HALF_BOTTOM: i32 = 19;

/// The window punched clean through the body: `x` in `[WINDOW_X0, WINDOW_X1]`,
/// `y` in `[WINDOW_Y0, WINDOW_Y1]`, fully transparent. An interior hole makes the
/// silhouette genuinely non-convex, so a cut-out pane shows the backdrop *inside*
/// the figure and not only around it.
const WINDOW_X0: i32 = 26;
const WINDOW_X1: i32 = 37;
const WINDOW_Y0: i32 = 43;
const WINDOW_Y1: i32 = 51;

// ── Colour ramp (property 2) ────────────────────────────────────────────────────
// A diagonal ramp keyed on `x + y` (range 0..=126), from hot orange-red at the
// top-left to deep crimson-magenta at the bottom-right. Red stays dominant over BOTH
// other channels by a wide margin at every point on the ramp (r-g >= 152, r-b >= 80),
// so the plate still reads WARM after the renderer's distance shading has scaled every
// channel down. That margin is the measurement in property 2 — a ramp that crossed
// over into blue at one end would leave part of the figure indistinguishable from a
// procedural wall.
const RAMP_MAX: i32 = (PLATE_SIZE as i32 - 1) * 2; // 126
const R_START: i32 = 255;
const R_END: i32 = 200;
const G_START: i32 = 24;
const G_END: i32 = 48;
const B_START: i32 = 16;
const B_END: i32 = 120;

/// The hard-edged bright bar that crosses both detail regions: a hot yellow band.
/// Its edges are single-texel hard transitions, which is what the sub-cell blitters
/// turn into partial-coverage glyphs.
const BAR_Y0: i32 = 28;
const BAR_Y1: i32 = 33;
const BAR_X0: i32 = 10;
const BAR_X1: i32 = 53;
const BAR_RGB: [u8; 3] = [255, 236, 72];

/// Checker period of the COARSE half (left, `x < 32`), in texels.
const COARSE_PERIOD: i32 = 16;
/// Checker period of the FINE half (right, `x >= 32`), in texels.
const FINE_PERIOD: i32 = 2;
/// Brightness (percent) applied to the dark squares of either checker. Hard-edged
/// (no interpolation) so the contrast survives sub-cell averaging.
const DARK_PERCENT: i32 = 42;

/// True when texel `(x, y)` is part of the figure (alpha 255); false ⇒ alpha 0.
fn is_opaque(x: i32, y: i32) -> bool {
    let dx = x - HEAD_CX;
    let dy = y - HEAD_CY;
    let in_head = dx * dx + dy * dy <= HEAD_R * HEAD_R;

    let in_body = if (BODY_TOP..=BODY_BOTTOM).contains(&y) {
        // Linear taper, integer-only: half-width grows from BODY_HALF_TOP to
        // BODY_HALF_BOTTOM across the body's rows.
        let span = BODY_BOTTOM - BODY_TOP;
        let half = BODY_HALF_TOP + (BODY_HALF_BOTTOM - BODY_HALF_TOP) * (y - BODY_TOP) / span;
        (x - HEAD_CX).abs() <= half
    } else {
        false
    };

    let in_window = (WINDOW_X0..=WINDOW_X1).contains(&x) && (WINDOW_Y0..=WINDOW_Y1).contains(&y);

    (in_head || in_body) && !in_window
}

/// Linear interpolation on the diagonal ramp, integer-only.
const fn ramp(start: i32, end: i32, t: i32) -> i32 {
    start + (end - start) * t / RAMP_MAX
}

/// The opaque colour of texel `(x, y)` before the checker modulation.
fn base_rgb(x: i32, y: i32) -> [i32; 3] {
    if (BAR_X0..=BAR_X1).contains(&x) && (BAR_Y0..=BAR_Y1).contains(&y) {
        return [
            i32::from(BAR_RGB[0]),
            i32::from(BAR_RGB[1]),
            i32::from(BAR_RGB[2]),
        ];
    }
    let t = x + y;
    [
        ramp(R_START, R_END, t),
        ramp(G_START, G_END, t),
        ramp(B_START, B_END, t),
    ]
}

/// True when texel `(x, y)` falls on a DARK square of its half's checker.
///
/// The two halves use different periods, which is the whole point: the same screen
/// cell covers many fine squares (they average, and the sub-cell blitters resolve
/// some of them) but only part of one coarse square (it reads solid).
const fn is_dark_square(x: i32, y: i32) -> bool {
    let period = if x < PLATE_SIZE as i32 / 2 {
        COARSE_PERIOD
    } else {
        FINE_PERIOD
    };
    ((x / period) + (y / period)) % 2 == 1
}

/// Generate the plate as a tightly-packed RGBA8 buffer, top-left origin, row-major.
///
/// Deterministic and platform-independent: integer arithmetic only, no floats, no
/// randomness, no I/O.
#[must_use]
pub fn plate_rgba() -> Vec<u8> {
    let mut out = Vec::with_capacity(PLATE_BYTES);
    for y in 0..PLATE_SIZE as i32 {
        for x in 0..PLATE_SIZE as i32 {
            if !is_opaque(x, y) {
                out.extend_from_slice(&[0, 0, 0, 0]);
                continue;
            }
            let [r, g, b] = base_rgb(x, y);
            let scale = if is_dark_square(x, y) {
                DARK_PERCENT
            } else {
                100
            };
            out.push((r * scale / 100) as u8);
            out.push((g * scale / 100) as u8);
            out.push((b * scale / 100) as u8);
            out.push(255);
        }
    }
    debug_assert_eq!(out.len(), PLATE_BYTES);
    out
}

/// Encode [`plate_rgba`] as PNG bytes.
///
/// # Errors
///
/// Returns the encoder's message if the `image` crate refuses the buffer. That
/// cannot happen for a well-formed 64×64 RGBA8 buffer, but the generator binary
/// reports it rather than panicking.
pub fn plate_png() -> Result<Vec<u8>, String> {
    let buf: image::RgbaImage = image::RgbaImage::from_raw(PLATE_SIZE, PLATE_SIZE, plate_rgba())
        .ok_or_else(|| "plate_rgba() produced a buffer of the wrong length".to_string())?;
    let mut out: Vec<u8> = Vec::new();
    buf.write_to(&mut Cursor::new(&mut out), image::ImageFormat::Png)
        .map_err(|e| format!("PNG encode failed: {e}"))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Path of the committed asset, derived at COMPILE time so the test does not
    /// depend on the process working directory.
    fn committed_png() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("assets")
            .join("texture")
            .join("genplate_tex.png")
    }

    /// Index of the first byte of texel `(x, y)` in the RGBA buffer.
    fn at(rgba: &[u8], x: u32, y: u32) -> [u8; 4] {
        let i = ((y * PLATE_SIZE + x) * 4) as usize;
        [rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3]]
    }

    /// THE POINT OF THE MODULE: the asset on disk is exactly what this source
    /// produces. If someone edits the plate by hand, or the generator changes without
    /// the asset being regenerated, this fails.
    #[test]
    fn the_committed_png_matches_the_generator() {
        let path = committed_png();
        let bytes = std::fs::read(&path).unwrap_or_else(|e| {
            panic!(
                "the generated fixture must be committed at {}: {e}\n\
                 Regenerate with: cargo run -p re_voximport --bin gen_texfixture -- \
                 assets/texture/genplate_tex.png",
                path.display()
            )
        });
        let decoded =
            image::load_from_memory(&bytes).expect("committed fixture must decode as PNG");
        let rgba = decoded.to_rgba8();
        assert_eq!(
            (rgba.width(), rgba.height()),
            (PLATE_SIZE, PLATE_SIZE),
            "committed fixture is the wrong size"
        );
        assert_eq!(
            rgba.as_raw().as_slice(),
            plate_rgba().as_slice(),
            "the committed {} does not match plate_rgba(); regenerate it with\n  \
             cargo run -p re_voximport --bin gen_texfixture -- assets/texture/genplate_tex.png",
            path.display()
        );
    }

    /// Generation is a pure function: two calls agree byte-for-byte.
    #[test]
    fn generation_is_deterministic() {
        assert_eq!(plate_rgba(), plate_rgba());
        assert_eq!(plate_png().expect("encodes"), plate_png().expect("encodes"));
    }

    /// PROPERTY 1 (source side): the alpha channel is a real silhouette — both fully
    /// transparent and fully opaque texels exist, alpha is strictly binary, the four
    /// corners are cut away, and the punched window is transparent while its
    /// surroundings are not. (The rendered-frame proof lives in the host
    /// game's `image_texture_content` suite; this only guards the generator.)
    #[test]
    fn the_plate_carries_a_non_trivial_binary_alpha_silhouette() {
        let rgba = plate_rgba();
        assert!(
            rgba.as_chunks::<4>()
                .0
                .iter()
                .all(|p| p[3] == 0 || p[3] == 255),
            "alpha is binary"
        );
        let texels = rgba.len() / 4;
        let opaque = rgba
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|p| p[3] == 255)
            .count();
        let clear = texels - opaque;
        assert!(
            opaque > 800,
            "enough figure to sample: {opaque} opaque texels"
        );
        assert!(
            clear > 800,
            "enough cut-out to see through: {clear} clear texels"
        );

        for (x, y) in [(0, 0), (63, 0), (0, 63), (63, 63)] {
            assert_eq!(at(&rgba, x, y)[3], 0, "corner ({x},{y}) is cut away");
        }
        assert_eq!(at(&rgba, 32, 47)[3], 0, "the punched window is transparent");
        assert_eq!(
            at(&rgba, 32, 38)[3],
            255,
            "the body above the window is opaque"
        );
        assert_eq!(
            at(&rgba, 32, 57)[3],
            255,
            "the body below the window is opaque"
        );

        // Non-trivial ⇒ the topmost opaque row differs across columns (a rectangle
        // would give one value for every column it covers).
        let mut tops: Vec<u32> = Vec::new();
        for x in 0..PLATE_SIZE {
            if let Some(y) = (0..PLATE_SIZE).find(|y| at(&rgba, x, *y)[3] == 255) {
                tops.push(y);
            }
        }
        tops.sort_unstable();
        tops.dedup();
        assert!(
            tops.len() >= 8,
            "silhouette top edge varies: {} distinct rows",
            tops.len()
        );
    }

    /// PROPERTY 2 (source side): opaque texels are saturated and red-dominant, never
    /// grey. A greyscale plate would be indistinguishable from the procedural
    /// fallback in the one channel that proves the sampler ran.
    #[test]
    fn every_opaque_texel_is_saturated_and_mostly_warm() {
        let rgba = plate_rgba();
        let mut warm = 0usize;
        let mut opaque = 0usize;
        for p in rgba.as_chunks::<4>().0 {
            if p[3] == 0 {
                continue;
            }
            opaque += 1;
            let (r, g, b) = (i32::from(p[0]), i32::from(p[1]), i32::from(p[2]));
            let sat = r.max(g).max(b) - r.min(g).min(b);
            assert!(sat >= 40, "texel {p:?} is nearly grey (sat {sat})");
            if r >= g + 40 && r >= b + 40 {
                warm += 1;
            }
        }
        assert!(
            warm * 2 > opaque,
            "the majority of the figure is warm: {warm} of {opaque}"
        );
    }

    /// PROPERTY 3 (source side): the two halves really do carry different feature
    /// sizes. Counting horizontal light/dark transitions per row separates them
    /// unambiguously — the fine half flips every 2 texels, the coarse half every 16.
    #[test]
    fn the_two_halves_carry_detail_at_two_different_scales() {
        let rgba = plate_rgba();
        let half = PLATE_SIZE / 2;
        // A row well below the head and clear of the bar and the window.
        let y = 60;
        let lum = |x: u32| -> i32 {
            let p = at(&rgba, x, y);
            i32::from(p[0]) + i32::from(p[1]) + i32::from(p[2])
        };
        let count_flips = |range: std::ops::Range<u32>| -> usize {
            range
                .skip(1)
                .filter(|x| (lum(*x) - lum(x - 1)).abs() > 30)
                .count()
        };
        // Restrict to the columns the body actually covers on this row.
        let coarse = count_flips(14..half);
        let fine = count_flips(half..50);
        assert!(fine >= 6, "the fine half alternates often: {fine} flips");
        assert!(
            coarse <= 2,
            "the coarse half is nearly flat: {coarse} flips"
        );
        assert!(
            fine > coarse * 3,
            "fine={fine} is a different scale from coarse={coarse}"
        );
    }

    /// The encoded PNG is small enough to belong in the repository.
    #[test]
    fn the_encoded_png_is_a_few_kilobytes() {
        let png = plate_png().expect("encodes");
        assert!(png.len() < 16 * 1024, "fixture PNG is {} bytes", png.len());
        assert_eq!(&png[1..4], b"PNG", "it really is a PNG");
    }

    // ── `gen_anim.gif` ──────────────────────────────────────────────────────

    /// Path of the committed animated fixture, derived at COMPILE time.
    fn committed_gif() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("assets")
            .join("texture")
            .join("gen_anim.gif")
    }

    /// THE POINT OF THIS HALF OF THE MODULE: the committed clip decodes to exactly
    /// the frames [`anim_frame_rgba`] produces, so the asset can never silently
    /// drift from its generator.
    #[test]
    fn the_committed_gif_matches_the_generator() {
        use image::codecs::gif::GifDecoder;
        use image::AnimationDecoder;

        let path = committed_gif();
        let f = std::fs::File::open(&path).unwrap_or_else(|e| {
            panic!(
                "the generated GIF fixture must be committed at {}: {e}\n\
                 Regenerate with: cargo run -p re_voximport --bin gen_texfixture -- --anim \
                 assets/texture/gen_anim.gif",
                path.display()
            )
        });
        let decoder =
            GifDecoder::new(std::io::BufReader::new(f)).expect("committed GIF must decode");
        let frames = decoder
            .into_frames()
            .collect_frames()
            .expect("every frame must decode");
        assert_eq!(
            frames.len(),
            ANIM_FRAMES as usize,
            "frame count must match the generator"
        );
        for (i, frame) in frames.iter().enumerate() {
            let buf = frame.buffer();
            assert_eq!(
                (buf.width(), buf.height()),
                (ANIM_W, ANIM_H),
                "frame {i} size"
            );
            assert_eq!(
                buf.as_raw().as_slice(),
                anim_frame_rgba(i as u32).as_slice(),
                "frame {i} pixels must match anim_frame_rgba({i}); regenerate with\n  \
                 cargo run -p re_voximport --bin gen_texfixture -- --anim assets/texture/gen_anim.gif"
            );
        }
    }

    /// Generation is a pure function: two calls agree byte-for-byte.
    #[test]
    fn anim_generation_is_deterministic() {
        for i in 0..ANIM_FRAMES {
            assert_eq!(anim_frame_rgba(i), anim_frame_rgba(i));
        }
        assert_eq!(anim_gif().expect("encodes"), anim_gif().expect("encodes"));
    }

    /// Every frame is genuinely distinct from every other — the property the
    /// frame-count / frame-index tests in `re_level`/`tex_import` rely on.
    #[test]
    fn every_anim_frame_is_distinct() {
        let frames: Vec<Vec<u8>> = (0..ANIM_FRAMES).map(anim_frame_rgba).collect();
        for i in 0..frames.len() {
            for j in (i + 1)..frames.len() {
                assert_ne!(frames[i], frames[j], "frame {i} and frame {j} must differ");
            }
        }
    }

    /// The encoded GIF is small enough to belong in the repository.
    #[test]
    fn the_encoded_gif_is_a_few_kilobytes() {
        let gif = anim_gif().expect("encodes");
        assert!(gif.len() < 32 * 1024, "fixture GIF is {} bytes", gif.len());
        assert_eq!(&gif[0..3], b"GIF", "it really is a GIF");
    }
}
