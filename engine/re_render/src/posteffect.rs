//! `posteffect` — a screen-space POST-PROCESS pass over the shaded
//! [`Framebuffer`], distinct from the cinematic `re_postfx` crate.
//!
//! This module owns the player-facing **radial focus** look the design calls for:
//! a sharp central disc that degrades outward in three concentric bands (a
//! *stylized ASCII depth-of-field* — symbol + colour detail-reduction with
//! radius, NOT a gaussian pixel blur), plus two simple global colour transforms
//! (grayscale and a limited-palette / posterize). It runs AFTER the shade /
//! contour pass produced a clean frame and BEFORE the terminal present.
//!
//! # Why it lives here, not in `re_postfx`
//!
//! `re_postfx` is the cinematic *grade* stack (edge/SSAO/bloom/grade/dither). This
//! module is the user's separate "focus/blur" feature and the shared spatial
//! foundation ([`screen_radius`] / [`spiral_index`]) that the upcoming
//! infinite-zoom feature reuses. Keeping it in `re_render` lets the zoom code
//! share the geometry without a cross-crate dependency.
//!
//! # Determinism & purity
//!
//! Every transform is integer-indexed `f32`/integer math — no Bevy, no clock, no
//! `rand`, no allocation in the per-cell path (the only `Vec` is one reused
//! snapshot for neighbourhood colour averaging). Same `(fb, cfg)` →
//! byte-identical output. Depends only on `re_core`.
//!
//! # Apply order
//!
//! [`apply_posteffects`] runs **blur first** (it averages neighbourhoods and
//! collapses glyphs, so it must see the original colours), then **grayscale**,
//! then **palette** (the colour-only passes are commutative-free and applied in a
//! fixed order so a composed `TS_POSTFX=blur,gray,palette` is deterministic).
//!
//! A [`PostFxConfig::default`] is a complete no-op: it leaves the framebuffer
//! byte-identical, so the shipped levels and every golden frame are unchanged
//! unless a caller (the `--dump` env flag today, a key toggle later) opts in.

use re_core::{Cell, Framebuffer};

// ───────────────────────────── spatial foundation ───────────────────────────

/// Vertical-to-horizontal cell aspect of a typical terminal: a character cell is
/// roughly twice as tall as it is wide, so a row step covers about twice the
/// visual distance a column step does. We scale the vertical component of the
/// radius by this so a "circle" measured in cells reads as a visual circle on
/// screen, not a tall ellipse. Named constant, not a scattered literal.
pub const CELL_ASPECT: f32 = 2.0;

/// Aspect-corrected, screen-centred distance of cell `(col, row)` from the centre
/// of a `cols`×`rows` grid, in **horizontal-cell units**.
///
/// `0.0` at the exact centre, growing outward. The vertical component is scaled by
/// [`CELL_ASPECT`] so the locus of a constant radius is a visual circle on a
/// terminal whose cells are ~2:1 tall — the basis for the radial focus bands and
/// the future zoom reveal.
///
/// Defined for every input: a `0`-width or `0`-height grid yields `0.0` (there is
/// no "outside" of an empty screen).
#[inline]
pub fn screen_radius(col: usize, row: usize, cols: usize, rows: usize) -> f32 {
    if cols == 0 || rows == 0 {
        return 0.0;
    }
    // Centre in fractional cell coordinates (cell centres at integer+0.5).
    let cx = cols as f32 * 0.5;
    let cy = rows as f32 * 0.5;
    let dx = (col as f32 + 0.5) - cx;
    // Scale the vertical delta into horizontal-cell units so a circle is circular.
    let dy = ((row as f32 + 0.5) - cy) * CELL_ASPECT;
    (dx * dx + dy * dy).sqrt()
}

/// A monotonic-from-centre **spiral ordering** of cell `(col, row)`, for a future
/// spiral reveal/zoom. Combines the aspect-corrected [`screen_radius`] (the
/// dominant term, so the value strictly grows outward ring by ring) with a small
/// angle term that breaks ties within a ring into a smooth spiral sweep.
///
/// Properties (unit-tested):
/// * `0.0` (or near it) at the centre, increasing with distance — radius
///   dominates, so a farther cell always sorts after a nearer one across rings.
/// * within roughly one ring, the angle (`atan2`, normalised to `[0,1)`) advances
///   the order so a reveal driven by this index unwinds as a spiral, not a plain
///   expanding disc.
///
/// Pure; defined for every input (empty grid → `0.0`).
#[inline]
pub fn spiral_index(col: usize, row: usize, cols: usize, rows: usize) -> f32 {
    if cols == 0 || rows == 0 {
        return 0.0;
    }
    let r = screen_radius(col, row, cols, rows);
    let cx = cols as f32 * 0.5;
    let cy = rows as f32 * 0.5;
    let dx = (col as f32 + 0.5) - cx;
    let dy = ((row as f32 + 0.5) - cy) * CELL_ASPECT;
    // atan2 in (-π, π] → normalise to [0, 1). The angle term is weighted well
    // below 1.0 so radius stays the dominant (ring-ordering) component and the
    // index never decreases across rings — only swirls within one.
    let ang = dy.atan2(dx); // (-π, π]
    let ang01 = (ang + std::f32::consts::PI) / std::f32::consts::TAU;
    r + ang01 * SPIRAL_ANGLE_WEIGHT
}

/// How much the angle term contributes to [`spiral_index`]. Kept below `1.0` so a
/// full angular sweep advances the index by less than one radial ring — radius
/// remains the dominant ordering term (the index is monotonic across rings).
pub const SPIRAL_ANGLE_WEIGHT: f32 = 0.9;

// ───────────────────────────── glyph ink density ────────────────────────────

/// Ordered glyph ramp from sparse (low ink coverage) to dense (high ink),
/// borrowed from the classic ASCII luminance ramp. Used to estimate a glyph's
/// **ink coverage** so the blur bands can collapse a cell to the nearest member of
/// a small symbol set by density (a heavy glyph snaps to the dense replacement, a
/// light one to the sparse replacement). Index = density rank (0 = lightest).
const INK_RAMP: &[char] = &[' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];

/// Box-drawing / contour / quadrant glyph ranges that the strongest band drops.
/// `(start, end_inclusive)` in Unicode scalar values:
/// * `U+2500..=U+257F` — Box Drawing (the contour line glyphs).
/// * `U+2580..=U+259F` — Block Elements (quadrants, shades, full/half blocks).
/// * `U+2596..=U+259F` is inside the block range above; listed once.
const CONTOUR_RANGES: &[(u32, u32)] = &[(0x2500, 0x257F), (0x2580, 0x259F)];

/// True if `glyph` is a box-drawing / block / quadrant glyph — the contour family
/// the strongest blur band removes so structural lines vanish in the far rim.
#[inline]
fn is_contour_glyph(glyph: u32) -> bool {
    CONTOUR_RANGES
        .iter()
        .any(|&(lo, hi)| glyph >= lo && glyph <= hi)
}

/// Estimate a glyph's ink coverage in `[0,1]` (0 = blank, 1 = solid). Contour /
/// block glyphs are treated as dense (they carry a lot of ink); a glyph found in
/// [`INK_RAMP`] uses its ramp position; anything else (letters, digits, symbols)
/// gets a medium-high default so it collapses to the denser member of a symbol set.
#[inline]
fn ink_density(glyph: u32) -> f32 {
    if glyph == u32::from(' ') {
        return 0.0;
    }
    if is_contour_glyph(glyph) {
        // Block elements are near-solid; box lines are moderately heavy.
        return if glyph >= 0x2580 { 0.95 } else { 0.6 };
    }
    if let Some(ch) = char::from_u32(glyph) {
        if let Some(pos) = INK_RAMP.iter().position(|&c| c == ch) {
            return pos as f32 / (INK_RAMP.len() - 1) as f32;
        }
    }
    // Letters/digits/unknown: treat as fairly dense so they read as "ink".
    0.7
}

/// Collapse `glyph` to the nearest member of `set` by ink density. `set` is ordered
/// sparse→dense; the glyph's [`ink_density`] picks the bucket. An empty set returns
/// the glyph unchanged (defensive — config validation keeps sets non-empty).
#[inline]
fn collapse_to_set(glyph: u32, set: &[char]) -> u32 {
    if set.is_empty() {
        return glyph;
    }
    if set.len() == 1 {
        return u32::from(set[0]);
    }
    let d = ink_density(glyph);
    // Map density [0,1] onto a set index, nearest bucket.
    let idx = (d * (set.len() - 1) as f32).round() as usize;
    u32::from(set[idx.min(set.len() - 1)])
}

// ───────────────────────────── colour helpers ───────────────────────────────

/// Unpack a packed `0x00RRGGBB` colour into `(r, g, b)` bytes.
#[inline]
fn unpack(rgb: u32) -> (u8, u8, u8) {
    (
        ((rgb >> 16) & 0xFF) as u8,
        ((rgb >> 8) & 0xFF) as u8,
        (rgb & 0xFF) as u8,
    )
}

/// Pack `(r, g, b)` bytes into a `0x00RRGGBB` colour.
#[inline]
fn pack(r: u8, g: u8, b: u8) -> u32 {
    ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
}

/// Rec.601 luma of a packed colour, as a byte. `0.299R + 0.587G + 0.114B`.
#[inline]
fn luma(rgb: u32) -> u8 {
    let (r, g, b) = unpack(rgb);
    (0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32).round() as u8
}

/// Map a packed colour to its grayscale (luma → equal RGB) equivalent.
#[inline]
fn to_gray(rgb: u32) -> u32 {
    let y = luma(rgb);
    pack(y, y, y)
}

/// Quantise one channel to `levels` evenly-spaced steps (≥2). `levels == 2` snaps
/// to `{0, 255}`; higher `levels` give a finer posterise. Rounds to the nearest
/// step so mid-greys land on the closest palette entry rather than truncating dark.
#[inline]
fn quantize_channel(v: u8, levels: u8) -> u8 {
    let levels = levels.max(2) as f32;
    let step = (levels - 1.0).max(1.0);
    let q = (v as f32 / 255.0 * step).round() / step;
    (q * 255.0).round().clamp(0.0, 255.0) as u8
}

/// Quantise a packed colour to `levels` steps per channel.
#[inline]
fn quantize_rgb(rgb: u32, levels: u8) -> u32 {
    let (r, g, b) = unpack(rgb);
    pack(
        quantize_channel(r, levels),
        quantize_channel(g, levels),
        quantize_channel(b, levels),
    )
}

/// Map a packed colour onto a fixed duotone ramp: its luma indexes a linear
/// interpolation between `dark` and `light`. Gives the "gameboy"/amber-mono look —
/// every cell becomes a tint of one hue, brightness preserved.
#[inline]
fn map_duotone(rgb: u32, dark: (u8, u8, u8), light: (u8, u8, u8)) -> u32 {
    let t = luma(rgb) as f32 / 255.0;
    let lerp = |a: u8, b: u8| {
        (a as f32 + (b as f32 - a as f32) * t)
            .round()
            .clamp(0.0, 255.0) as u8
    };
    pack(
        lerp(dark.0, light.0),
        lerp(dark.1, light.1),
        lerp(dark.2, light.2),
    )
}

// ───────────────────────────── config types ─────────────────────────────────

/// One degradation band of the radial focus-blur. Configurable so `/iterate-render`
/// can tune the look without touching code.
#[derive(Clone, Debug, PartialEq)]
pub struct BlurBand {
    /// Symbol set this band collapses glyphs into, ordered sparse→dense (by ink
    /// density). A 2-element set keeps two variants; a 1-element set is a single
    /// symbol (the strongest reduction). Never empty.
    pub symbols: Vec<char>,
    /// Half-extent (in cells) of the neighbourhood whose colours are averaged into
    /// each cell. `0` = no averaging (colour untouched). Larger = blurrier colour.
    pub blur_radius: usize,
    /// Posterise level applied to the (averaged) colour: `0` = no quantise, else
    /// N levels per channel. Combined with `blur_radius` this is the band's colour
    /// reduction strength.
    pub color_levels: u8,
    /// If true, box-drawing / block / quadrant (contour) glyphs are dropped and
    /// replaced by this band's symbol set before collapse — structural lines vanish.
    pub drop_contours: bool,
}

/// Radial focus-blur configuration: a sharp central disc surrounded by three
/// concentric degradation bands of increasing strength.
#[derive(Clone, Debug, PartialEq)]
pub struct BlurCfg {
    /// Radius (in horizontal-cell units, see [`screen_radius`]) of the sharp
    /// central disc. Cells within this radius are left UNCHANGED.
    pub center_radius: f32,
    /// Outer radius of band 1 (weak). Cells in `(center_radius, band1]`.
    pub band1_radius: f32,
    /// Outer radius of band 2 (medium). Cells in `(band1, band2]`.
    pub band2_radius: f32,
    /// The three bands, weakest→strongest. `[0]` weak, `[1]` medium, `[2]` strong.
    /// Everything beyond `band2_radius` falls into band 3 (`[2]`).
    pub bands: [BlurBand; 3],
}

impl Default for BlurCfg {
    /// The default radial focus look: a ~6-cell sharp disc (≈12×12 region), then
    /// three bands that drop from two symbol variants to one dot, averaging and
    /// quantising colours more aggressively outward, with contour lines removed in
    /// the strongest band. "The bigger the circle, the stronger the change."
    fn default() -> Self {
        BlurCfg {
            center_radius: 6.0,
            band1_radius: 12.0,
            band2_radius: 20.0,
            bands: [
                // Band 1 (weak): two variants (sparse/dense), light colour mean.
                BlurBand {
                    symbols: vec![':', '#'],
                    blur_radius: 1,
                    color_levels: 0,
                    drop_contours: false,
                },
                // Band 2 (medium): collapse toward one/two, stronger averaging.
                BlurBand {
                    symbols: vec!['.', '+'],
                    blur_radius: 1,
                    color_levels: 6,
                    drop_contours: false,
                },
                // Band 3 (strong): a single dot, contours dropped, large mean.
                BlurBand {
                    symbols: vec![DOT_GLYPH],
                    blur_radius: 2,
                    color_levels: 4,
                    drop_contours: true,
                },
            ],
        }
    }
}

/// The single dot the strongest band reduces every glyph to: `·` (U+00B7, middle
/// dot). Configurable via the band's `symbols`; this is the default.
pub const DOT_GLYPH: char = '\u{00B7}';

impl BlurCfg {
    /// Classify `radius` into a band: `None` = inside the sharp central disc
    /// (untouched), `Some(i)` = band `i` (0 weak, 1 medium, 2 strong). Anything
    /// beyond `band2_radius` is band 2 (the strong band has no outer bound).
    #[inline]
    pub fn band_for_radius(&self, radius: f32) -> Option<usize> {
        if radius <= self.center_radius {
            None
        } else if radius <= self.band1_radius {
            Some(0)
        } else if radius <= self.band2_radius {
            Some(1)
        } else {
            Some(2)
        }
    }
}

/// A limited-palette / posterise configuration.
#[derive(Clone, Debug, PartialEq)]
pub enum PaletteCfg {
    /// Quantise each channel to N evenly-spaced levels (≥2). Lower N = chunkier.
    Posterize {
        /// Levels per channel (≥2).
        levels: u8,
    },
    /// Map every cell onto a duotone ramp between `dark` and `light` by luma — the
    /// "gameboy"/amber-mono look. Brightness preserved, hue collapsed to one axis.
    Duotone {
        /// The dark end of the ramp, `(r, g, b)`.
        dark: (u8, u8, u8),
        /// The light end of the ramp, `(r, g, b)`.
        light: (u8, u8, u8),
    },
}

impl PaletteCfg {
    /// The classic 4-shade green Game Boy palette as a duotone ramp (dark olive →
    /// pale green).
    pub fn gameboy() -> Self {
        PaletteCfg::Duotone {
            dark: (0x0F, 0x38, 0x0F),
            light: (0x9B, 0xBC, 0x0F),
        }
    }

    /// An amber monochrome terminal ramp (near-black → amber).
    pub fn amber() -> Self {
        PaletteCfg::Duotone {
            dark: (0x1A, 0x0E, 0x00),
            light: (0xFF, 0xB0, 0x00),
        }
    }

    /// Apply this palette to a single packed colour.
    #[inline]
    fn map(&self, rgb: u32) -> u32 {
        match *self {
            PaletteCfg::Posterize { levels } => quantize_rgb(rgb, levels),
            PaletteCfg::Duotone { dark, light } => map_duotone(rgb, dark, light),
        }
    }
}

/// The composable post-effect stack. Each field is independently toggleable; a
/// [`Default`] is all-off (a byte-identical no-op).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PostFxConfig {
    /// Radial focus-blur (sharp centre, degrading rim). `None` = off.
    pub blur: Option<BlurCfg>,
    /// Map fg/bg to grayscale (Rec.601 luma). `false` = off.
    pub grayscale: bool,
    /// Limited-palette / posterise of fg/bg. `None` = off.
    pub palette: Option<PaletteCfg>,
}

impl PostFxConfig {
    /// True if no effect is enabled — [`apply_posteffects`] is a guaranteed no-op.
    #[inline]
    pub fn is_noop(&self) -> bool {
        self.blur.is_none() && !self.grayscale && self.palette.is_none()
    }

    /// Parse a `PostFxConfig` from the `TS_POSTFX` environment variable, the
    /// dump-path gate (a key toggle will drive the same struct later). The value is
    /// a comma-separated list of effect names:
    ///
    /// * `blur`    → default radial focus-blur.
    /// * `gray` / `grayscale` → grayscale.
    /// * `palette` → default posterise (8 levels/channel).
    /// * `gameboy` → Game Boy duotone palette.
    /// * `amber`   → amber-mono duotone palette.
    ///
    /// e.g. `TS_POSTFX=blur`, `TS_POSTFX=gray`, `TS_POSTFX=blur,gray`,
    /// `TS_POSTFX=blur,gameboy`. Unknown tokens are ignored. Absent / empty →
    /// `None` (no post-fx; the frame is unchanged).
    pub fn from_env() -> Option<Self> {
        let raw = std::env::var("TS_POSTFX").ok()?;
        Self::parse(&raw)
    }

    /// Parse the comma-separated effect list (the body of `from_env`, exposed for
    /// tests + a future key-binding). Returns `None` if no recognised effect is
    /// present (so an empty or junk value is a clean no-op, not a half-config).
    pub fn parse(raw: &str) -> Option<Self> {
        let mut cfg = PostFxConfig::default();
        let mut any = false;
        for tok in raw.split(',') {
            match tok.trim().to_ascii_lowercase().as_str() {
                "blur" => {
                    cfg.blur = Some(BlurCfg::default());
                    any = true;
                }
                "gray" | "grey" | "grayscale" | "greyscale" | "bw" => {
                    cfg.grayscale = true;
                    any = true;
                }
                "palette" | "posterize" => {
                    cfg.palette = Some(PaletteCfg::Posterize { levels: 8 });
                    any = true;
                }
                "gameboy" | "gb" => {
                    cfg.palette = Some(PaletteCfg::gameboy());
                    any = true;
                }
                "amber" | "mono" => {
                    cfg.palette = Some(PaletteCfg::amber());
                    any = true;
                }
                _ => {}
            }
        }
        if any {
            Some(cfg)
        } else {
            None
        }
    }
}

// ───────────────────────────── the pass ─────────────────────────────────────

/// Apply the post-effect stack to `fb` IN PLACE. Order: **blur → grayscale →
/// palette**. A no-op config ([`PostFxConfig::is_noop`]) leaves `fb`
/// byte-identical and returns early (no snapshot, no work).
///
/// Pure: a deterministic function of `(fb, cfg)`. The only allocation is one
/// reused snapshot when the blur needs neighbourhood colour averaging.
pub fn apply_posteffects(fb: &mut Framebuffer, cfg: &PostFxConfig) {
    if cfg.is_noop() {
        return;
    }
    if let Some(blur) = &cfg.blur {
        apply_blur(fb, blur);
    }
    if cfg.grayscale {
        apply_grayscale(fb);
    }
    if let Some(pal) = &cfg.palette {
        apply_palette(fb, pal);
    }
}

/// Average a band's colour neighbourhood for cell `(col,row)` from the read-only
/// `snapshot`, returning the mean `(fg, bg)`. The neighbourhood is the square of
/// half-extent `band.blur_radius`; a radius of `0` returns the cell's own colours.
#[inline]
fn averaged_colors(snapshot: &Framebuffer, col: usize, row: usize, radius: usize) -> (u32, u32) {
    if radius == 0 {
        let c = snapshot.get(col, row).copied().unwrap_or_default();
        return (c.fg, c.bg);
    }
    let (mut fr, mut fg_, mut fb_, mut br, mut bg_, mut bb, mut n) =
        (0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32);
    let r = radius as isize;
    for dr in -r..=r {
        for dc in -r..=r {
            let nc = col as isize + dc;
            let nr = row as isize + dr;
            if nc < 0 || nr < 0 {
                continue;
            }
            if let Some(c) = snapshot.get(nc as usize, nr as usize) {
                let (r1, g1, b1) = unpack(c.fg);
                fr += r1 as u32;
                fg_ += g1 as u32;
                fb_ += b1 as u32;
                let (r2, g2, b2) = unpack(c.bg);
                br += r2 as u32;
                bg_ += g2 as u32;
                bb += b2 as u32;
                n += 1;
            }
        }
    }
    if n == 0 {
        let c = snapshot.get(col, row).copied().unwrap_or_default();
        return (c.fg, c.bg);
    }
    let avg = |s: u32| (s / n) as u8;
    (
        pack(avg(fr), avg(fg_), avg(fb_)),
        pack(avg(br), avg(bg_), avg(bb)),
    )
}

/// The radial focus-blur: glyph collapse + colour reduction per band, sharp centre.
fn apply_blur(fb: &mut Framebuffer, cfg: &BlurCfg) {
    let cols = fb.cols();
    let rows = fb.rows();
    if cols == 0 || rows == 0 {
        return;
    }
    // Colour averaging samples neighbours, so read from an immutable snapshot taken
    // before any mutation — otherwise the mean would smear already-blurred cells.
    let snapshot = fb.clone();
    for row in 0..rows {
        for col in 0..cols {
            let radius = screen_radius(col, row, cols, rows);
            let band_idx = match cfg.band_for_radius(radius) {
                None => continue, // sharp central disc — untouched
                Some(i) => i,
            };
            let band = &cfg.bands[band_idx];
            let mut cell: Cell = *snapshot.get(col, row).unwrap();

            // Glyph: optionally drop contour lines, then collapse to the band's set.
            let src_glyph = if band.drop_contours && is_contour_glyph(cell.glyph) {
                // A dropped contour collapses to the band's densest symbol so a
                // structural edge becomes ink, not a hole.
                u32::from(*band.symbols.last().unwrap_or(&DOT_GLYPH))
            } else {
                cell.glyph
            };
            cell.glyph = collapse_to_set(src_glyph, &band.symbols);

            // Colour: neighbourhood mean, then optional posterise.
            let (mut fg, mut bg) = averaged_colors(&snapshot, col, row, band.blur_radius);
            if band.color_levels > 0 {
                fg = quantize_rgb(fg, band.color_levels);
                bg = quantize_rgb(bg, band.color_levels);
            }
            cell.fg = fg;
            cell.bg = bg;

            fb.set(col, row, cell);
        }
    }
}

/// Map every cell's fg + bg to grayscale (Rec.601 luma). Glyph + geometry untouched.
fn apply_grayscale(fb: &mut Framebuffer) {
    let cols = fb.cols();
    let rows = fb.rows();
    for row in 0..rows {
        for col in 0..cols {
            if let Some(c) = fb.get(col, row).copied().as_mut() {
                c.fg = to_gray(c.fg);
                c.bg = to_gray(c.bg);
                fb.set(col, row, *c);
            }
        }
    }
}

/// Map every cell's fg + bg through the limited palette. Glyph + geometry untouched.
fn apply_palette(fb: &mut Framebuffer, pal: &PaletteCfg) {
    let cols = fb.cols();
    let rows = fb.rows();
    for row in 0..rows {
        for col in 0..cols {
            if let Some(c) = fb.get(col, row).copied().as_mut() {
                c.fg = pal.map(c.fg);
                c.bg = pal.map(c.bg);
                fb.set(col, row, *c);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use re_core::{Cell, Framebuffer, SurfaceKind};

    // ── foundation: screen_radius ──────────────────────────────────────────

    #[test]
    fn screen_radius_is_zero_at_center() {
        // Even-sized grid: the four central cells are closest to the centre and
        // share the minimal radius; an odd grid has a true single centre.
        let r = screen_radius(10, 10, 21, 21); // centre cell of a 21x21 grid
        assert!(r < 1.0, "centre cell radius should be ~0, got {r}");
    }

    #[test]
    fn screen_radius_grows_to_corner_max() {
        let cols = 40;
        let rows = 20;
        let center = screen_radius(cols / 2, rows / 2, cols, rows);
        let corner = screen_radius(0, 0, cols, rows);
        assert!(corner > center, "corner must be farther than centre");
        // The corner is the farthest cell — no interior cell exceeds it.
        for row in 0..rows {
            for col in 0..cols {
                assert!(screen_radius(col, row, cols, rows) <= corner + 1e-3);
            }
        }
    }

    #[test]
    fn screen_radius_is_aspect_circular() {
        // A horizontal offset of N cells and a vertical offset of N/CELL_ASPECT
        // cells should give (nearly) equal radii — that is what "circular on a 2:1
        // terminal" means.
        let cols = 101;
        let rows = 101;
        let cc = 50;
        let horiz = screen_radius(cc + 10, cc, cols, rows);
        // 10 horizontal cells ≈ 5 vertical cells at CELL_ASPECT=2.
        let vert = screen_radius(cc, cc + 5, cols, rows);
        assert!(
            (horiz - vert).abs() < 0.5,
            "aspect correction should make a circle: horiz={horiz}, vert={vert}"
        );
    }

    #[test]
    fn screen_radius_empty_grid_is_zero() {
        assert_eq!(screen_radius(0, 0, 0, 0), 0.0);
        assert_eq!(screen_radius(3, 4, 0, 10), 0.0);
    }

    // ── foundation: spiral_index ───────────────────────────────────────────

    #[test]
    fn spiral_index_monotonic_from_center() {
        // Sample radii outward along a ray; the spiral index must strictly increase
        // (radius dominates the angle term, so it never decreases across rings).
        let cols = 81;
        let rows = 81;
        let cc = 40;
        let mut prev = spiral_index(cc, cc, cols, rows);
        for step in 1..30 {
            // Walk diagonally outward (changes radius by > one ring each step).
            let v = spiral_index(cc + step, cc + step, cols, rows);
            assert!(
                v > prev,
                "spiral index must grow outward: step {step} gave {v} <= {prev}"
            );
            prev = v;
        }
    }

    #[test]
    fn spiral_index_center_is_small() {
        let cols = 51;
        let rows = 51;
        let center = spiral_index(25, 25, cols, rows);
        let edge = spiral_index(50, 25, cols, rows);
        assert!(center < edge);
        // The centre's index is bounded by the angle term alone (radius ~0).
        assert!(center <= SPIRAL_ANGLE_WEIGHT + 1.0);
    }

    // ── ink density + contour classification ───────────────────────────────

    #[test]
    fn ink_density_orders_ramp() {
        assert_eq!(ink_density(u32::from(' ')), 0.0);
        assert!(ink_density(u32::from('.')) < ink_density(u32::from('#')));
        assert!(ink_density(u32::from('@')) > ink_density(u32::from(':')));
    }

    #[test]
    fn contour_glyphs_detected() {
        assert!(is_contour_glyph(0x2500)); // ─ box light horizontal
        assert!(is_contour_glyph(0x250C)); // ┌ corner
        assert!(is_contour_glyph(0x2588)); // █ full block
        assert!(is_contour_glyph(0x2580)); // ▀ upper half block
        assert!(!is_contour_glyph(u32::from('#')));
        assert!(!is_contour_glyph(u32::from('A')));
    }

    #[test]
    fn collapse_to_set_picks_by_density() {
        let set = ['.', '#'];
        // A heavy glyph collapses to the dense member, a light one to the sparse.
        assert_eq!(collapse_to_set(u32::from('@'), &set), u32::from('#'));
        assert_eq!(collapse_to_set(u32::from('.'), &set), u32::from('.'));
        // A single-element set is always that symbol.
        let dot = [DOT_GLYPH];
        assert_eq!(collapse_to_set(u32::from('@'), &dot), u32::from(DOT_GLYPH));
        assert_eq!(collapse_to_set(u32::from(' '), &dot), u32::from(DOT_GLYPH));
    }

    // ── colour helpers ─────────────────────────────────────────────────────

    #[test]
    fn luma_matches_rec601() {
        // Pure green is the brightest channel weight (0.587).
        assert_eq!(luma(0x0000_FF00), (0.587_f32 * 255.0).round() as u8);
        // Pure red.
        assert_eq!(luma(0x00FF_0000), (0.299_f32 * 255.0).round() as u8);
        // White → 255, black → 0.
        assert_eq!(luma(0x00FF_FFFF), 255);
        assert_eq!(luma(0x0000_0000), 0);
    }

    #[test]
    fn grayscale_makes_equal_channels() {
        let g = to_gray(0x00FF_0000); // red → its luma in all channels
        let (r, gg, b) = unpack(g);
        assert_eq!(r, gg);
        assert_eq!(gg, b);
        assert_eq!(r, luma(0x00FF_0000));
    }

    #[test]
    fn quantize_channel_snaps_to_levels() {
        // 2 levels → only 0 or 255.
        assert_eq!(quantize_channel(10, 2), 0);
        assert_eq!(quantize_channel(200, 2), 255);
        // 3 levels → 0, ~128, 255.
        assert_eq!(quantize_channel(0, 3), 0);
        assert_eq!(quantize_channel(255, 3), 255);
        let mid = quantize_channel(128, 3);
        assert!((120..=135).contains(&mid), "mid level was {mid}");
    }

    #[test]
    fn duotone_preserves_brightness_axis() {
        let gb = PaletteCfg::gameboy();
        // Black maps to the dark anchor, white to the light anchor.
        let dark = gb.map(0x0000_0000);
        let light = gb.map(0x00FF_FFFF);
        assert_eq!(dark, pack(0x0F, 0x38, 0x0F));
        assert_eq!(light, pack(0x9B, 0xBC, 0x0F));
    }

    // ── band classification ────────────────────────────────────────────────

    #[test]
    fn band_for_radius_thresholds() {
        let cfg = BlurCfg::default();
        assert_eq!(cfg.band_for_radius(0.0), None); // centre
        assert_eq!(cfg.band_for_radius(cfg.center_radius), None); // boundary inclusive
        assert_eq!(cfg.band_for_radius(cfg.center_radius + 0.1), Some(0));
        assert_eq!(cfg.band_for_radius(cfg.band1_radius + 0.1), Some(1));
        assert_eq!(cfg.band_for_radius(cfg.band2_radius + 0.1), Some(2));
        assert_eq!(cfg.band_for_radius(9999.0), Some(2)); // strong band unbounded
    }

    // ── helpers to build a test frame ──────────────────────────────────────

    fn cell_with(glyph: char, fg: u32, bg: u32) -> Cell {
        Cell {
            glyph: u32::from(glyph),
            fg,
            bg,
            surface: SurfaceKind::WallSolid,
            ..Cell::default()
        }
    }

    /// A frame filled with a heavy glyph + a contour glyph at a far corner.
    fn filled_frame(cols: usize, rows: usize) -> Framebuffer {
        let mut fb = Framebuffer::new(cols, rows);
        for row in 0..rows {
            for col in 0..cols {
                fb.set(col, row, cell_with('@', 0x00C0_4020, 0x0010_2030));
            }
        }
        fb
    }

    // ── default config is a strict no-op ───────────────────────────────────

    #[test]
    fn default_config_is_byte_identical_noop() {
        let before = filled_frame(48, 24);
        let mut after = before.clone();
        apply_posteffects(&mut after, &PostFxConfig::default());
        assert_eq!(
            before.cells(),
            after.cells(),
            "Default PostFxConfig must leave the framebuffer byte-identical"
        );
        assert!(PostFxConfig::default().is_noop());
    }

    // ── blur: sharp centre, degrading rim ──────────────────────────────────

    #[test]
    fn blur_leaves_center_untouched() {
        let cols = 60;
        let rows = 30;
        let before = filled_frame(cols, rows);
        let mut after = before.clone();
        apply_blur(&mut after, &BlurCfg::default());
        // A centre cell (radius ~0) must be unchanged.
        let cc = (cols / 2, rows / 2);
        assert_eq!(
            before.get(cc.0, cc.1),
            after.get(cc.0, cc.1),
            "sharp central disc must be untouched"
        );
    }

    #[test]
    fn blur_band3_collapses_to_single_dot_and_drops_contours() {
        let cols = 80;
        let rows = 40;
        let mut fb = filled_frame(cols, rows);
        // Put a contour glyph at the far corner (deep in band 3).
        fb.set(0, 0, cell_with('\u{2500}', 0x00FF_FFFF, 0x0000_0000)); // ─
                                                                       // also a normal heavy glyph next to it (still band 3).
        let r00 = screen_radius(0, 0, cols, rows);
        assert_eq!(BlurCfg::default().band_for_radius(r00), Some(2));

        apply_blur(&mut fb, &BlurCfg::default());

        // The contour glyph is gone — replaced by the band's single dot.
        let corner = fb.get(0, 0).unwrap();
        assert_eq!(
            corner.glyph,
            u32::from(DOT_GLYPH),
            "band 3 must drop contour glyphs and collapse to ONE dot"
        );
        // Any other band-3 cell is also the dot (exactly one symbol variant).
        let other = fb.get(1, 0).unwrap();
        assert_eq!(other.glyph, u32::from(DOT_GLYPH));
    }

    #[test]
    fn blur_band1_keeps_two_variants() {
        let cols = 80;
        let rows = 40;
        let cc = (cols / 2, rows / 2);
        let mut fb = filled_frame(cols, rows);
        // Find a cell that lands in band 1 and set a LIGHT glyph there, plus a
        // HEAVY one elsewhere in band 1 — they must collapse to DIFFERENT symbols
        // (two variants kept).
        let cfg = BlurCfg::default();
        // A cell just outside the centre disc, horizontally (radius in band 1).
        let light_col = cc.0 + (cfg.center_radius as usize) + 2;
        assert_eq!(
            cfg.band_for_radius(screen_radius(light_col, cc.1, cols, rows)),
            Some(0)
        );
        fb.set(light_col, cc.1, cell_with('.', 0x0080_8080, 0x0000_0000)); // light
        fb.set(
            light_col,
            cc.1 + 1,
            cell_with('@', 0x0080_8080, 0x0000_0000),
        ); // heavy

        apply_blur(&mut fb, &cfg);

        let sparse = fb.get(light_col, cc.1).unwrap().glyph;
        let dense = fb.get(light_col, cc.1 + 1).unwrap().glyph;
        assert_ne!(
            sparse, dense,
            "band 1 must keep TWO symbol variants (light vs heavy glyph differ)"
        );
        // Both come from band 1's two-symbol set.
        let set: Vec<u32> = cfg.bands[0].symbols.iter().map(|&c| u32::from(c)).collect();
        assert!(set.contains(&sparse));
        assert!(set.contains(&dense));
    }

    #[test]
    fn blur_outward_reduces_variety() {
        // The COUNT of distinct glyphs used should drop band by band: band 1 has 2,
        // band 3 has 1. We verify the band 3 region is a single glyph value.
        let cols = 100;
        let rows = 50;
        let mut fb = filled_frame(cols, rows);
        // Sprinkle varied glyphs everywhere.
        let palette = ['.', ':', '+', '#', '@', '\u{2500}'];
        for row in 0..rows {
            for col in 0..cols {
                let g = palette[(col + row) % palette.len()];
                fb.set(col, row, cell_with(g, 0x0040_8060, 0x0000_0000));
            }
        }
        let cfg = BlurCfg::default();
        apply_blur(&mut fb, &cfg);

        // Collect distinct glyphs in band 3 (corner region).
        use std::collections::HashSet;
        let mut band3: HashSet<u32> = HashSet::new();
        for row in 0..rows {
            for col in 0..cols {
                if cfg.band_for_radius(screen_radius(col, row, cols, rows)) == Some(2) {
                    band3.insert(fb.get(col, row).unwrap().glyph);
                }
            }
        }
        assert_eq!(band3.len(), 1, "band 3 must use exactly ONE glyph variant");
        assert!(band3.contains(&u32::from(DOT_GLYPH)));
    }

    // ── grayscale ──────────────────────────────────────────────────────────

    #[test]
    fn grayscale_pass_makes_fg_gray() {
        let mut fb = Framebuffer::new(4, 4);
        fb.set(0, 0, cell_with('#', 0x00FF_0000, 0x0000_FF00)); // red fg, green bg
        let cfg = PostFxConfig {
            grayscale: true,
            ..PostFxConfig::default()
        };
        apply_posteffects(&mut fb, &cfg);
        let c = fb.get(0, 0).unwrap();
        // fg becomes the luma of red in all channels.
        assert_eq!(c.fg, to_gray(0x00FF_0000));
        let (r, g, b) = unpack(c.fg);
        assert_eq!(r, g);
        assert_eq!(g, b);
        // bg likewise gray.
        let (br, bg, bb) = unpack(c.bg);
        assert_eq!(br, bg);
        assert_eq!(bg, bb);
        // glyph untouched.
        assert_eq!(c.glyph, u32::from('#'));
    }

    // ── palette ────────────────────────────────────────────────────────────

    #[test]
    fn palette_posterize_quantizes() {
        let mut fb = Framebuffer::new(2, 2);
        fb.set(0, 0, cell_with('#', 0x0012_3456, 0x00AB_CDEF));
        let cfg = PostFxConfig {
            palette: Some(PaletteCfg::Posterize { levels: 2 }),
            ..PostFxConfig::default()
        };
        apply_posteffects(&mut fb, &cfg);
        let c = fb.get(0, 0).unwrap();
        // Every channel snaps to 0 or 255.
        for ch in [c.fg, c.bg] {
            let (r, g, b) = unpack(ch);
            for v in [r, g, b] {
                assert!(
                    v == 0 || v == 255,
                    "posterize@2 must snap to 0/255, got {v}"
                );
            }
        }
    }

    #[test]
    fn palette_gameboy_duotone_one_hue_axis() {
        let mut fb = Framebuffer::new(2, 2);
        fb.set(0, 0, cell_with('#', 0x00FF_0000, 0x0000_0000));
        let cfg = PostFxConfig {
            palette: Some(PaletteCfg::gameboy()),
            ..PostFxConfig::default()
        };
        apply_posteffects(&mut fb, &cfg);
        let c = fb.get(0, 0).unwrap();
        // bg was black → maps to the dark gameboy anchor.
        assert_eq!(c.bg, pack(0x0F, 0x38, 0x0F));
    }

    // ── env parsing ────────────────────────────────────────────────────────

    #[test]
    fn parse_recognises_effects() {
        assert!(PostFxConfig::parse("blur").unwrap().blur.is_some());
        assert!(PostFxConfig::parse("gray").unwrap().grayscale);
        assert!(PostFxConfig::parse("bw").unwrap().grayscale);
        let combo = PostFxConfig::parse("blur,gray").unwrap();
        assert!(combo.blur.is_some() && combo.grayscale);
        assert!(matches!(
            PostFxConfig::parse("gameboy").unwrap().palette,
            Some(PaletteCfg::Duotone { .. })
        ));
        assert!(matches!(
            PostFxConfig::parse("palette").unwrap().palette,
            Some(PaletteCfg::Posterize { .. })
        ));
    }

    #[test]
    fn parse_empty_or_junk_is_none() {
        assert!(PostFxConfig::parse("").is_none());
        assert!(PostFxConfig::parse("   ").is_none());
        assert!(PostFxConfig::parse("nonsense,foo").is_none());
        // A junk token alongside a real one keeps only the real one.
        let cfg = PostFxConfig::parse("foo,blur").unwrap();
        assert!(cfg.blur.is_some());
        assert!(!cfg.grayscale);
    }

    // ── composition order is deterministic ─────────────────────────────────

    #[test]
    fn full_stack_is_deterministic() {
        let base = filled_frame(50, 26);
        let cfg = PostFxConfig {
            blur: Some(BlurCfg::default()),
            grayscale: true,
            palette: Some(PaletteCfg::Posterize { levels: 4 }),
        };
        let mut a = base.clone();
        let mut b = base.clone();
        apply_posteffects(&mut a, &cfg);
        apply_posteffects(&mut b, &cfg);
        assert_eq!(a.cells(), b.cells(), "post-fx must be deterministic");
        // And it actually changed the rim (not a silent no-op).
        assert_ne!(base.cells(), a.cells());
    }
}
