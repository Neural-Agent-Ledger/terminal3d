//! `zband` — a colour material that fills only PART of its cell's height, and paints
//! the top of that part in a second colour.
//!
//! ## Why
//!
//! Every solid cell is one storey tall, which is taller than the eye. That made two
//! things impossible to say at cell scale:
//!
//! * a lit `[emit]` reward figure could only be a stack of storey-tall neon columns — a
//!   gate across the door's line of sight, not a figure;
//! * a chair, a bench, a low stack or a table could only be a floor-level outline (a
//!   `furniture_trace`), which nobody sees from the spawn.
//!
//! A **z-band** keeps the cell a normal solid for the grid (collision, the DDA, the
//! keep-out rules) and makes the RENDERER treat it as solid only between two heights
//! `lo..hi` (fractions of the cell, `0` = its floor). Above and below the band the cell
//! is see-through, exactly like a transparent texel of a cut-out image pane, so it rides
//! the same cut-out path (`crate::image_cutout_column` on the eye plane,
//! `crate::vhit_opaque` on every other plane). The band has no lid of its own: from
//! standing eye height the lid of a band that ends at or below the eye is edge-on, and
//! its top edge is drawn by the contour pass as a rim.
//!
//! A band may carry a **top tone**: from `split` (a fraction of the band, measured up
//! from `lo`) to `hi` the face shades in a second colour — the head over the stem,
//! the bright top of a seat. The recolour keeps the shaded brightness, so light, fog
//! and the material's own pattern survive it.
//!
//! ## How a material asks for it (the name suffix)
//!
//! `re_world::MaterialDef` has no field for it, and the materials that need it are
//! synthesized by two other crates (`re_game::emit_parts`, `re_level`'s prop kit), so the
//! request travels in the material NAME, the same way a preset selection does:
//!
//! ```text
//! <name>~z<LO>_<HI>                 band from LO % to HI % of the cell
//! <name>~z<LO>_<HI>~t<RRGGBB>_<S>   … whose top (from S % of the band up) is #RRGGBB
//! ```
//!
//! `LO`/`HI`/`S` are integer percents. `~` cannot appear in an authored `[material]`
//! header, so an ordinary material never parses as a band. [`suffix`] writes the form
//! and [`parse`] reads it; `re_level` and `re_game` mirror [`suffix`] (a cross-crate test
//! in `re_game` pins them equal).

/// A partial-height colour material: solid from `lo` to `hi` (cell fractions), with an
/// optional top tone.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZBand {
    /// Bottom of the solid band, as a fraction of the cell height (`0` = the floor).
    pub lo: f32,
    /// Top of the solid band (`1` = the cell's ceiling).
    pub hi: f32,
    /// The top tone: `(rgb, split)` — from `lo + split·(hi − lo)` up, the face shades in
    /// `rgb`. `None` = one colour.
    pub top: Option<(u32, f32)>,
}

/// The smallest band the renderer honours, in cell fractions. A thinner one would be a
/// sub-row sliver at any distance; it is ignored (the cell stays a full solid).
pub const MIN_BAND: f32 = 0.05;

impl ZBand {
    /// Is local height `z` (a fraction of the cell, `0` = floor) inside the solid band?
    #[inline]
    #[must_use]
    pub fn contains(&self, z: f32) -> bool {
        z >= self.lo && z <= self.hi
    }

    /// Does the band reach the cell's floor / ceiling (so that face is a real lid)?
    #[inline]
    #[must_use]
    pub fn touches_floor(&self) -> bool {
        self.lo <= 1e-3
    }

    /// See [`ZBand::touches_floor`].
    #[inline]
    #[must_use]
    pub fn touches_ceiling(&self) -> bool {
        self.hi >= 1.0 - 1e-3
    }

    /// The top-tone colour at local height `z`, if `z` is in the band's top part.
    #[inline]
    #[must_use]
    pub fn tone_at(&self, z: f32) -> Option<u32> {
        let (rgb, split) = self.top?;
        (z >= self.lo + split * (self.hi - self.lo)).then_some(rgb)
    }
}

/// Parse a material name's z-band suffix. `None` for any name without a well-formed
/// `~z` segment, or with a band thinner than [`MIN_BAND`] or outside `0..=1`. Pure.
#[must_use]
pub fn parse(name: &str) -> Option<ZBand> {
    let at = name.find("~z")?;
    let mut segs = name[at + 1..].split('~');
    let z = segs.next()?.strip_prefix('z')?;
    let (lo, hi) = z.split_once('_')?;
    let pct = |s: &str| {
        s.parse::<u16>()
            .ok()
            .filter(|&v| v <= 100)
            .map(|v| f32::from(v) / 100.0)
    };
    let (lo, hi) = (pct(lo)?, pct(hi)?);
    if hi - lo < MIN_BAND {
        return None;
    }
    let top = segs.next().and_then(|t| {
        let t = t.strip_prefix('t')?;
        let (hex, split) = t.split_once('_')?;
        let rgb = (hex.len() == 6)
            .then(|| u32::from_str_radix(hex, 16).ok())
            .flatten()?;
        Some((rgb, pct(split)?))
    });
    Some(ZBand { lo, hi, top })
}

/// Write the z-band suffix for `lo..hi` (cell fractions) and an optional top tone
/// `(rgb, split)`. Values are rounded to whole percents and clamped to `0..=100`. Pure.
#[must_use]
pub fn suffix(lo: f32, hi: f32, top: Option<(u32, f32)>) -> String {
    let pct = |v: f32| (v.clamp(0.0, 1.0) * 100.0).round() as u32;
    let mut s = format!("~z{}_{}", pct(lo), pct(hi));
    if let Some((rgb, split)) = top {
        use std::fmt::Write as _;
        let _ = write!(s, "~t{:06x}_{}", rgb & 0xFF_FFFF, pct(split));
    }
    s
}

/// Recolour a shaded colour `c` from `base` to `to`, keeping its brightness relative to
/// `base` (a texel at 60 % of the base colour comes out at 60 % of the new one). Pure.
#[inline]
#[must_use]
pub fn recolour(c: u32, base: u32, to: u32) -> u32 {
    let peak = |v: u32| ((v >> 16) & 0xFF).max((v >> 8) & 0xFF).max(v & 0xFF);
    let pb = peak(base).max(1) as f32;
    let k = (peak(c) as f32 / pb).min(1.5);
    crate::ascii_shader::brighten_rgb(to, k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_plain_name_is_not_a_band() {
        assert_eq!(parse("__prop_lab_body"), None);
        assert_eq!(parse("\u{1f}emit\u{1f}0"), None);
        assert_eq!(parse("x~q1_2"), None);
    }

    #[test]
    fn suffix_round_trips_through_parse() {
        let s = format!("\u{1f}emit\u{1f}3{}", suffix(0.0, 0.5, None));
        assert_eq!(s, "\u{1f}emit\u{1f}3~z0_50");
        assert_eq!(
            parse(&s),
            Some(ZBand {
                lo: 0.0,
                hi: 0.5,
                top: None
            })
        );
        let t = format!("__prop_seat{}", suffix(0.0, 0.45, Some((0xFF_D080, 0.7))));
        assert_eq!(t, "__prop_seat~z0_45~tffd080_70");
        let b = parse(&t).expect("parses");
        assert_eq!(b.top, Some((0xFF_D080, 0.7)));
        assert!((b.hi - 0.45).abs() < 1e-6);
    }

    #[test]
    fn degenerate_or_out_of_range_bands_are_ignored() {
        assert_eq!(parse("m~z40_42"), None, "thinner than MIN_BAND");
        assert_eq!(parse("m~z50_40"), None, "inverted");
        assert_eq!(parse("m~z0_140"), None, "above the cell");
        assert_eq!(parse("m~z0_"), None);
    }

    #[test]
    fn band_membership_and_tone() {
        let b = ZBand {
            lo: 0.0,
            hi: 0.5,
            top: Some((0x00_FF00, 0.6)),
        };
        assert!(b.contains(0.0) && b.contains(0.49) && !b.contains(0.51));
        assert!(b.touches_floor() && !b.touches_ceiling());
        assert_eq!(b.tone_at(0.2), None, "below the split: the base colour");
        assert_eq!(b.tone_at(0.35), Some(0x00_FF00), "0.35 ≥ 0 + 0.6·0.5");
    }

    #[test]
    fn recolour_keeps_relative_brightness() {
        // A texel at half the base's peak comes out at half the new colour.
        let out = recolour(0x40_4040, 0x80_8080, 0x00_FF00);
        assert_eq!(out & 0xFF_00FF, 0);
        let g = (out >> 8) & 0xFF;
        assert!((126..=129).contains(&g), "got {g}");
    }

    // ── rendered: the band, its lid, and what stays visible behind it ──

    use re_core::{Framebuffer, SurfaceKind};
    use re_world::{CameraPose, CellKind, Map, MaterialDef};

    const COLS: usize = 80;
    const ROWS: usize = 40;
    /// The block cell, the slab one storey up behind it, and the eye (+y, 3 cells off).
    const BLOCK: (usize, usize, usize) = (4, 5, 1);
    const SLAB: (usize, usize, usize) = (4, 7, 2);

    /// A walled 9×12×5 room; `block` = the colour-1 material name of the cell at
    /// [`BLOCK`] (`None` = no block), plus a plain slab at [`SLAB`].
    fn frame(block: Option<&str>) -> Framebuffer {
        let (sx, sy, sz) = (9usize, 12usize, 5usize);
        let mut m = Map::new(sx, sy, sz);
        for z in 0..sz {
            for y in 0..sy {
                for x in 0..sx {
                    let edge = x == 0 || y == 0 || x == sx - 1 || y == sy - 1;
                    let k = if edge {
                        CellKind::WallSolid
                    } else if z == 0 {
                        CellKind::Floor
                    } else if z == sz - 1 {
                        CellKind::Ceiling
                    } else {
                        CellKind::Empty
                    };
                    m.set(x, y, z, k);
                }
            }
        }
        m.set(SLAB.0, SLAB.1, SLAB.2, CellKind::WallSolid);
        let mut mats: Vec<MaterialDef> = (0..5).map(|_| MaterialDef::fallback()).collect();
        mats.push(MaterialDef {
            name: block.unwrap_or("blk").to_string(),
            fg: 0xC0_3030,
            bg: 0x30_0C0C,
            fallback: false,
            ..MaterialDef::fallback()
        });
        m.set_materials(mats);
        if block.is_some() {
            m.set_cell(BLOCK.0, BLOCK.1, BLOCK.2, CellKind::WallSolid, 1);
        }
        let cam = CameraPose::spawn_at(4, 1, 1, 0);
        let reg = crate::registry_for_map(&m);
        let mut fb = Framebuffer::new(COLS, ROWS);
        crate::composite::composite_dump_once(&m, &cam, &reg, &crate::Fov::default(), &[], &mut fb);
        fb
    }

    fn block_rows(fb: &Framebuffer, col: usize) -> Vec<usize> {
        (0..ROWS)
            .filter(|&r| {
                fb.get(col, r).is_some_and(|c| {
                    c.material == 1
                        && matches!(c.surface, SurfaceKind::WallSolid | SurfaceKind::WallTrace)
                })
            })
            .collect()
    }

    /// Rows of `col` that show the slab's near face (the `y = 7` plane).
    fn slab_rows(fb: &Framebuffer, col: usize) -> Vec<usize> {
        (0..ROWS)
            .filter(|&r| {
                fb.get(col, r).is_some_and(|c| {
                    c.material != 1
                        && c.surface == SurfaceKind::WallSolid
                        && (c.hit.y - 7.0).abs() < 0.05
                })
            })
            .collect()
    }

    /// A knee-high band renders strictly inside the full cell's silhouette (fewer
    /// rows, all of them rows the full cell paints), shows its top tone (face top +
    /// lid), and no longer hides off-slice geometry behind its see-through rows: every
    /// slab row visible with no block at all is still the slab, except where the band
    /// itself covers it.
    #[test]
    fn a_band_renders_inside_the_full_silhouette_with_a_toned_lid() {
        let col = COLS / 2;
        let full = block_rows(&frame(Some("blk")), col);
        let banded_fb = frame(Some(&format!(
            "blk{}",
            suffix(0.0, 0.3, Some((0xFF_FFFF, 0.7)))
        )));
        let band = block_rows(&banded_fb, col);
        assert!(
            !full.is_empty() && !band.is_empty(),
            "full {full:?} band {band:?}"
        );
        assert!(band.len() < full.len(), "band {band:?} vs full {full:?}");
        assert!(
            band.iter().all(|r| full.contains(r)),
            "band {band:?} outside full {full:?}"
        );
        assert_eq!(
            band.last(),
            full.last(),
            "the band stands on the cell floor"
        );

        let whitish = |r: &usize| {
            banded_fb.get(col, *r).is_some_and(|c| {
                let (rr, g) = ((c.fg >> 16) & 0xFF, (c.fg >> 8) & 0xFF);
                rr > 0 && g * 2 > rr
            })
        };
        let toned = band.iter().filter(|r| whitish(r)).count();
        assert!(
            toned > 0 && toned < band.len(),
            "top tone over a red body: {toned}/{}",
            band.len()
        );
        assert!(
            whitish(&band[0]),
            "the band's topmost row (its lid / rim) is the top tone"
        );

        let clear = slab_rows(&frame(None), col);
        assert!(
            !clear.is_empty(),
            "control: the slab is visible with no block"
        );
        let behind = slab_rows(&banded_fb, col);
        let lost: Vec<_> = clear
            .iter()
            .filter(|r| !behind.contains(r) && !band.contains(r))
            .collect();
        assert!(
            lost.is_empty(),
            "slab rows hidden by the band's see-through rows: {lost:?}"
        );
    }
}
