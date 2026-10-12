//! `re_render::edge_field` — precomputed per-texture edge data for the ASCII /
//! contour / lines image samplers (quality-textures spec, Components 2 + 2b).
//!
//! The contour pass is **not** a per-frame convolution. When a texture is attached
//! to a [`crate::ascii_shader::Material`] (at registry build time,
//! `crate::registry_for_map`) we precompute, ONCE, an [`EdgeField`] from the texture
//! luminance:
//!
//! 1. optional Gaussian pre-blur,
//! 2. **Difference of Gaussians** (two sigmas, subtract) → clean thin contours,
//! 3. **Sobel on the DoG result** → gradient magnitude + angle,
//! 4. quantise the angle to 5 direction bins (`- / | \ _`),
//!
//! and a small **mip pyramid** (×½ downsamples) of BOTH the luminance and the edge
//! field, so the sampler can pick the level whose texel size matches a cell's
//! `tex_footprint` (minification → no flicker). The per-frame path then only
//! *samples* this baked data — O(1), allocation-free, cache friendly.
//!
//! Pure compute: no Bevy, no terminal, no `unsafe`. The cost is a handful of
//! separable convolution passes over the texels, paid once at load.

use std::sync::Arc;

use re_core::TextureImage;

// ─────────────────────────── tunables (Component 2) ──────────────────────────

/// DoG sigma pair (narrow, wide). The Difference of Gaussians subtracts a wide
/// blur from a narrow one, leaving a thin band of response exactly on intensity
/// edges. The ratio (~1.6) approximates a Laplacian-of-Gaussian — the classic
/// Marr–Hildreth edge operator the Acerola DoG look is built on.
pub const DOG_SIGMA_NARROW: f32 = 1.0;
/// Wide DoG sigma (see [`DOG_SIGMA_NARROW`]).
pub const DOG_SIGMA_WIDE: f32 = 1.6;

/// Optional Gaussian PRE-blur sigma applied to the raw luminance before the DoG,
/// to suppress single-pixel noise so the contours are clean. `0.0` disables it.
pub const PRE_BLUR_SIGMA: f32 = 0.8;

/// Edge-magnitude threshold (in normalised `[0,1]` Sobel-on-DoG magnitude) above
/// which a texel is considered "on an edge". Below it the sampler draws the
/// luminance-ramp fill; at/above it the directional contour glyph. A named knob
/// for manual tuning — raise it for sparser, bolder lines.
pub const EDGE_MAG_THRESHOLD: f32 = 0.18;

/// How many ×½ mip levels to build beyond level 0 (inclusive of level 0 the pyramid
/// holds `1 + MIP_EXTRA_LEVELS`). Five extra halvings cover a 417×598 texture down to
/// ~13×19 — plenty for the far LOD; the source data is cheap so this is generous.
pub const MIP_EXTRA_LEVELS: usize = 5;

// ───────────────────────────── direction bins ───────────────────────────────

/// The five quantised gradient-direction bins and their line glyphs. The bin index
/// is stored in [`EdgeLevel::dir`]; the glyph is the contour stroke drawn ACROSS the
/// gradient (perpendicular to it — a stroke runs along the edge, not down-gradient).
///
/// `0 = '-'  1 = '/'  2 = '|'  3 = '\'  4 = '_'`
pub const DIR_GLYPHS: [char; 5] = ['-', '/', '|', '\\', '_'];

/// Map a gradient ANGLE (radians, `atan2(gy, gx)`) to a direction bin `0..=4`.
///
/// The Sobel gradient points ACROSS the edge (down the brightness slope); the
/// drawn stroke runs ALONG the edge, i.e. perpendicular to the gradient. We bin the
/// gradient angle into 4 orientations and pick the glyph whose stroke is
/// perpendicular:
///
/// * gradient horizontal (`→`) ⇒ edge is vertical ⇒ `|`
/// * gradient vertical (`↑`)   ⇒ edge is horizontal ⇒ `-`
/// * gradient at +45°          ⇒ edge at −45° ⇒ `\`
/// * gradient at −45°          ⇒ edge at +45° ⇒ `/`
///
/// Bin 4 (`_`) is reserved for a near-horizontal edge biased low — we fold it into
/// bin 0 (`-`) here so the four Sobel orientations map deterministically; the `_`
/// glyph is still reachable by samplers that want a low-baseline variant.
#[inline]
pub fn angle_to_bin(angle: f32) -> u8 {
    // Normalise the gradient angle to [0, PI) — a line has no head/tail, so a
    // gradient and its 180° opposite describe the same edge orientation.
    let pi = std::f32::consts::PI;
    let mut a = angle % pi;
    if a < 0.0 {
        a += pi;
    }
    // Quarter-turn sectors centred on 0, 45, 90, 135 degrees.
    // sector boundaries at 22.5, 67.5, 112.5, 157.5 degrees.
    let deg = a * 180.0 / pi;
    if !(22.5..157.5).contains(&deg) {
        // gradient ~horizontal → vertical edge → '|'
        2
    } else if (22.5..67.5).contains(&deg) {
        // gradient ~+45 → '\'
        3
    } else if (67.5..112.5).contains(&deg) {
        // gradient ~vertical → horizontal edge → '-'
        0
    } else {
        // gradient ~+135 (−45) → '/'
        1
    }
}

// ───────────────────────────── data structures ──────────────────────────────

/// One mip level of the precomputed edge data: luminance + Sobel/DoG magnitude +
/// quantised direction bin, all at this level's resolution. All vectors are
/// `width * height` long, row-major, top-left origin (matching [`TextureImage`]).
#[derive(Clone, Debug, PartialEq)]
pub struct EdgeLevel {
    /// Level width in texels.
    pub width: u32,
    /// Level height in texels.
    pub height: u32,
    /// Luminance per texel `[0,1]`.
    pub lum: Vec<f32>,
    /// Edge magnitude per texel `[0,1]` (`0` = flat, `1` = strong edge).
    pub mag: Vec<f32>,
    /// Quantised gradient-direction bin per texel: `0='-' 1='/' 2='|' 3='\' 4='_'`.
    pub dir: Vec<u8>,
}

impl EdgeLevel {
    /// Sample the level at UV `(u, v)` in `[0,1]` (clamped), nearest texel. Returns
    /// `(lum, mag, dir_bin)`. A zero-area level returns `(0, 0, 0)`.
    #[inline]
    pub fn sample(&self, u: f32, v: f32) -> (f32, f32, u8) {
        if self.width == 0 || self.height == 0 {
            return (0.0, 0.0, 0);
        }
        let uc = u.clamp(0.0, 1.0);
        let vc = v.clamp(0.0, 1.0);
        let px = ((uc * self.width as f32) as u32).min(self.width - 1);
        let py = ((vc * self.height as f32) as u32).min(self.height - 1);
        let idx = (py * self.width + px) as usize;
        (self.lum[idx], self.mag[idx], self.dir[idx])
    }
}

/// Precomputed edge data for ONE texture: level 0 at full resolution plus a
/// small ×½ mip pyramid. Attached to a `Material` as `Option<Arc<EdgeField>>`
/// next to the texture. The hot path samples
/// [`level_for_texels`](EdgeField::level_for_texels) then [`EdgeLevel::sample`].
#[derive(Clone, Debug, PartialEq)]
pub struct EdgeField {
    /// Pyramid levels, index 0 = full resolution, each subsequent ≈ half size.
    pub levels: Vec<EdgeLevel>,
}

impl EdgeField {
    /// Full-resolution width (level 0). `0` for an empty field.
    #[inline]
    pub fn width(&self) -> u32 {
        self.levels.first().map_or(0, |l| l.width)
    }

    /// Full-resolution height (level 0). `0` for an empty field.
    #[inline]
    pub fn height(&self) -> u32 {
        self.levels.first().map_or(0, |l| l.height)
    }

    /// Pick the mip level whose texel size best matches the cell's `tex_footprint`
    /// in TEXEL units: `texels_per_cell` is how many level-0 texels one screen cell
    /// covers (≈ footprint × texture resolution along the tighter axis). A cell
    /// covering `n` texels wants level `floor(log2(n))` so one mip texel ≈ one cell —
    /// the standard minification rule that kills shimmer. Clamped to the built range.
    #[inline]
    pub fn level_for_texels(&self, texels_per_cell: f32) -> &EdgeLevel {
        let n = texels_per_cell.max(1.0);
        let lvl = n.log2().floor().max(0.0) as usize;
        let lvl = lvl.min(self.levels.len().saturating_sub(1));
        &self.levels[lvl]
    }

    /// Build the edge field + mip pyramid from a texture's RGBA8 pixels.
    ///
    /// Returns `None` for a zero-area texture (nothing to sample). Otherwise the
    /// pipeline is: luminance → optional pre-blur → DoG (two sigmas) → Sobel on the
    /// DoG → magnitude/direction, all on level 0; then ×½ downsample for each mip
    /// level (magnitude averaged, direction = dominant bin of the 2×2 quad).
    pub fn build(img: &TextureImage) -> Option<Arc<EdgeField>> {
        let w = img.width;
        let h = img.height;
        if w == 0 || h == 0 {
            return None;
        }
        let level0 = build_level0(img);
        let mut levels = Vec::with_capacity(MIP_EXTRA_LEVELS + 1);
        levels.push(level0);
        for _ in 0..MIP_EXTRA_LEVELS {
            let last = levels.last().unwrap();
            if last.width <= 1 && last.height <= 1 {
                break;
            }
            levels.push(downsample(last));
        }
        Some(Arc::new(EdgeField { levels }))
    }
}

// ─────────────────────────── build pipeline (load-time) ──────────────────────

/// Rec.601 luminance of an `[r,g,b]` 0..255 texel, normalised to `[0,1]`.
#[inline]
fn luma(r: u8, g: u8, b: u8) -> f32 {
    (0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32) / 255.0
}

/// Level-0 luminance + Sobel-on-DoG edge field.
fn build_level0(img: &TextureImage) -> EdgeLevel {
    let w = img.width as usize;
    let h = img.height as usize;
    let n = w * h;

    // Luminance.
    let mut lum = vec![0.0f32; n];
    for (i, px) in img.rgba.as_chunks::<4>().0.iter().enumerate().take(n) {
        lum[i] = luma(px[0], px[1], px[2]);
    }

    // Optional pre-blur, then DoG = blur(narrow) − blur(wide) on the pre-blurred
    // luminance. Each Gaussian is a separable box-approx (3 box passes ≈ Gaussian),
    // kept cheap and allocation-bounded.
    let base = if PRE_BLUR_SIGMA > 0.0 {
        gaussian(&lum, w, h, PRE_BLUR_SIGMA)
    } else {
        lum.clone()
    };
    let narrow = gaussian(&base, w, h, DOG_SIGMA_NARROW);
    let wide = gaussian(&base, w, h, DOG_SIGMA_WIDE);
    let mut dog = vec![0.0f32; n];
    for i in 0..n {
        dog[i] = narrow[i] - wide[i];
    }

    // Sobel on the DoG result → gradient (gx, gy) → magnitude + angle.
    let (mut mag, dir) = sobel(&dog, w, h);

    // Normalise magnitude to [0,1] by the field max (a stable per-texture scale so
    // EDGE_MAG_THRESHOLD is meaningful regardless of the source contrast).
    let max = mag.iter().copied().fold(0.0f32, f32::max);
    if max > 1e-6 {
        let inv = 1.0 / max;
        for m in &mut mag {
            *m *= inv;
        }
    }

    EdgeLevel {
        width: img.width,
        height: img.height,
        lum,
        mag,
        dir,
    }
}

/// Separable Gaussian blur approximated by THREE successive box blurs (the standard
/// fast Gaussian). `sigma <= 0` returns the input unchanged. Radius from sigma via
/// the box-blur equivalence `w ≈ sigma * sqrt(12/3 + 1)`.
fn gaussian(src: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    if sigma <= 0.0 || w == 0 || h == 0 {
        return src.to_vec();
    }
    // Box radius approximating the Gaussian; clamp to keep small textures sane.
    let radius = ((sigma * 1.7).round() as usize).clamp(1, 16);
    let mut buf = src.to_vec();
    for _ in 0..3 {
        buf = box_blur_h(&buf, w, h, radius);
        buf = box_blur_v(&buf, w, h, radius);
    }
    buf
}

/// Horizontal box blur with clamped edges.
fn box_blur_h(src: &[f32], w: usize, h: usize, radius: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; src.len()];
    let denom = (2 * radius + 1) as f32;
    for y in 0..h {
        let row = y * w;
        for x in 0..w {
            let mut acc = 0.0f32;
            for k in 0..=(2 * radius) {
                let xi = (x + k).saturating_sub(radius).min(w - 1);
                acc += src[row + xi];
            }
            out[row + x] = acc / denom;
        }
    }
    out
}

/// Vertical box blur with clamped edges.
fn box_blur_v(src: &[f32], w: usize, h: usize, radius: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; src.len()];
    let denom = (2 * radius + 1) as f32;
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for k in 0..=(2 * radius) {
                let yi = (y + k).saturating_sub(radius).min(h - 1);
                acc += src[yi * w + x];
            }
            out[y * w + x] = acc / denom;
        }
    }
    out
}

/// Sobel operator on a scalar field → `(magnitude, direction_bin)` per texel. Edges
/// are detected on the (DoG) field; the gradient angle picks the direction bin via
/// [`angle_to_bin`]. Clamped edges (replicate the border texel).
fn sobel(field: &[f32], w: usize, h: usize) -> (Vec<f32>, Vec<u8>) {
    let n = w * h;
    let mut mag = vec![0.0f32; n];
    let mut dir = vec![0u8; n];
    let at = |x: isize, y: isize| -> f32 {
        let xi = x.clamp(0, w as isize - 1) as usize;
        let yi = y.clamp(0, h as isize - 1) as usize;
        field[yi * w + xi]
    };
    for y in 0..h as isize {
        for x in 0..w as isize {
            // Sobel kernels.
            let gx = (at(x + 1, y - 1) + 2.0 * at(x + 1, y) + at(x + 1, y + 1))
                - (at(x - 1, y - 1) + 2.0 * at(x - 1, y) + at(x - 1, y + 1));
            let gy = (at(x - 1, y + 1) + 2.0 * at(x, y + 1) + at(x + 1, y + 1))
                - (at(x - 1, y - 1) + 2.0 * at(x, y - 1) + at(x + 1, y - 1));
            let idx = (y as usize) * w + (x as usize);
            mag[idx] = (gx * gx + gy * gy).sqrt();
            dir[idx] = angle_to_bin(gy.atan2(gx));
        }
    }
    (mag, dir)
}

/// ×½ downsample of an [`EdgeLevel`]: 2×2 box-average of luminance + magnitude;
/// direction = the dominant (most frequent, ties → strongest-magnitude) bin of the
/// quad. Odd dimensions round UP so the smallest level is never empty.
fn downsample(src: &EdgeLevel) -> EdgeLevel {
    let sw = src.width as usize;
    let sh = src.height as usize;
    let dw = sw.div_ceil(2).max(1);
    let dh = sh.div_ceil(2).max(1);
    let mut lum = vec![0.0f32; dw * dh];
    let mut mag = vec![0.0f32; dw * dh];
    let mut dir = vec![0u8; dw * dh];
    for dy in 0..dh {
        for dx in 0..dw {
            let mut lsum = 0.0f32;
            let mut msum = 0.0f32;
            let mut count = 0.0f32;
            // dominant-bin vote weighted by magnitude.
            let mut bin_weight = [0.0f32; 5];
            for oy in 0..2 {
                for ox in 0..2 {
                    let sx = dx * 2 + ox;
                    let sy = dy * 2 + oy;
                    if sx >= sw || sy >= sh {
                        continue;
                    }
                    let si = sy * sw + sx;
                    lsum += src.lum[si];
                    msum += src.mag[si];
                    count += 1.0;
                    let b = src.dir[si] as usize;
                    bin_weight[b] += src.mag[si] + 1e-3; // +eps so flat quads still vote
                }
            }
            let di = dy * dw + dx;
            if count > 0.0 {
                lum[di] = lsum / count;
                mag[di] = msum / count;
            }
            // dominant direction bin.
            let mut best = 0usize;
            for (b, &wgt) in bin_weight.iter().enumerate() {
                if wgt > bin_weight[best] {
                    best = b;
                }
            }
            dir[di] = best as u8;
        }
    }
    EdgeLevel {
        width: dw as u32,
        height: dh as u32,
        lum,
        mag,
        dir,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small RGBA8 texture from a grayscale closure `f(x,y) -> 0..255`.
    fn gray(w: u32, h: u32, f: impl Fn(u32, u32) -> u8) -> TextureImage {
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                let v = f(x, y);
                rgba.extend_from_slice(&[v, v, v, 255]);
            }
        }
        TextureImage::new(w, h, rgba)
    }

    #[test]
    fn angle_bins_are_perpendicular_to_gradient() {
        // gradient pointing right (horizontal) → vertical edge → '|' (bin 2).
        assert_eq!(angle_to_bin(0.0), 2);
        // gradient pointing up (vertical) → horizontal edge → '-' (bin 0).
        assert_eq!(angle_to_bin(std::f32::consts::FRAC_PI_2), 0);
        // gradient +45° → '\' (bin 3).
        assert_eq!(angle_to_bin(std::f32::consts::FRAC_PI_4), 3);
        // gradient +135° (−45°) → '/' (bin 1).
        assert_eq!(angle_to_bin(3.0 * std::f32::consts::FRAC_PI_4), 1);
    }

    #[test]
    fn empty_texture_builds_nothing() {
        assert!(EdgeField::build(&TextureImage::new(0, 0, vec![])).is_none());
    }

    #[test]
    fn vertical_step_detects_vertical_edge() {
        // Left half black, right half white → a single VERTICAL edge down the middle.
        // The gradient there points horizontally → the edge orientation is vertical
        // → direction bin 2 ('|').
        let w = 16;
        let h = 16;
        let img = gray(w, h, |x, _| if x < w / 2 { 0 } else { 255 });
        let field = EdgeField::build(&img).expect("field");
        let l0 = &field.levels[0];
        // Find the strongest-magnitude texel; it must sit near the centre column and
        // be classified as a vertical edge.
        let mut best = 0usize;
        for i in 0..l0.mag.len() {
            if l0.mag[i] > l0.mag[best] {
                best = i;
            }
        }
        assert!(
            l0.mag[best] > EDGE_MAG_THRESHOLD,
            "the step must read as an edge"
        );
        let bx = (best % w as usize) as i32;
        assert!(
            (bx - (w as i32 / 2)).abs() <= 2,
            "strongest edge near the middle column, got x={bx}"
        );
        assert_eq!(l0.dir[best], 2, "a vertical step is a '|' edge (bin 2)");
    }

    #[test]
    fn horizontal_step_detects_horizontal_edge() {
        let w = 16;
        let h = 16;
        let img = gray(w, h, |_, y| if y < h / 2 { 0 } else { 255 });
        let field = EdgeField::build(&img).expect("field");
        let l0 = &field.levels[0];
        let mut best = 0usize;
        for i in 0..l0.mag.len() {
            if l0.mag[i] > l0.mag[best] {
                best = i;
            }
        }
        assert_eq!(l0.dir[best], 0, "a horizontal step is a '-' edge (bin 0)");
    }

    #[test]
    fn flat_texture_has_no_edges() {
        let img = gray(8, 8, |_, _| 128);
        let field = EdgeField::build(&img).expect("field");
        let l0 = &field.levels[0];
        assert!(
            l0.mag.iter().all(|&m| m < EDGE_MAG_THRESHOLD),
            "a flat field must have no edges"
        );
    }

    #[test]
    fn pyramid_halves_each_level() {
        let img = gray(16, 16, |x, _| (x * 16) as u8);
        let field = EdgeField::build(&img).expect("field");
        assert!(field.levels.len() >= 2, "pyramid must have mips");
        assert_eq!(field.levels[0].width, 16);
        assert_eq!(field.levels[1].width, 8);
        assert_eq!(field.levels[1].height, 8);
    }

    #[test]
    fn level_for_texels_picks_coarser_for_bigger_footprint() {
        let img = gray(32, 32, |x, y| ((x ^ y) * 8) as u8);
        let field = EdgeField::build(&img).expect("field");
        // 1 texel/cell → level 0 (full res).
        assert_eq!(field.level_for_texels(1.0).width, 32);
        // 4 texels/cell → level 2 (×¼).
        assert_eq!(field.level_for_texels(4.0).width, 8);
    }

    #[test]
    fn luminance_is_recorded() {
        let img = gray(4, 4, |_, _| 255);
        let field = EdgeField::build(&img).expect("field");
        let l0 = &field.levels[0];
        assert!(
            l0.lum.iter().all(|&v| (v - 1.0).abs() < 1e-3),
            "white → lum 1"
        );
    }
}
