//! Every level exercised here (`TX5-genplate.lvl` and the inline
//! `subcell_probe`s over the same generated plate) is an engine fixture, and
//! the sampler under test — `re_level`'s `[texture]` decode feeding
//! `re_render`'s image material — is entirely engine behaviour, using only
//! `re_level` and `re_render`, both regular dependencies of `re_runtime`
//! already.
//!
//! POSITIVE CONTENT assertions for the `.lvl` image-texture sampler.
//!
//! # The hole this closes
//!
//! The golden-framebuffer suite (`re_golden`) pins the image-sampler levels by
//! SHA-256. A hash pins CONSISTENCY, not correctness: it is equally happy with a
//! correctly textured frame and with a fallback frame, as long as neither changes.
//! And a fallback frame is exactly what you get for free here — `re_level`'s
//! `[texture]` decode is best-effort (`Level::build_map` skips a texture it cannot
//! read; only the unused `build_map_checked` reports it), so deleting the bound
//! PNG does NOT make the level `UNREADABLE`. It renders its untextured procedural
//! walls, silently, with no output on stderr, and the suite stays green while the
//! sampler loses all of its coverage.
//!
//! Measured on this tree, `levels/TX5-genplate.lvl` at the golden 100×32 dump grid:
//!
//! | measurement           | texture present | texture renamed away |
//! |-----------------------|-----------------|----------------------|
//! | warm foreground cells | 325             | **0**                |
//! | Braille `U+28xx` cells | 262            | **0**                |
//!
//! The hash moves, so a *blessed* golden catches it — but only if somebody has
//! already blessed the textured frame and nobody re-blesses the fallback. Neither is
//! a check. The assertions below are: they name a property of the *content* that only
//! the image sampler can produce, so a missing texture is a red test with a message
//! that says which asset went missing, not a hash that quietly agrees with itself.
//!
//! # What is asserted, and why each one is specific to the sampler
//!
//! * **Warm, saturated foreground cells exist.** The procedural fallback for these
//!   wall surfaces produces ZERO red-dominant cells (measured above). This is the one
//!   signal that separates "the image sampler ran" from "a procedural recipe ran" —
//!   which is also why the free fixture had to be a *coloured* plate and not a
//!   greyscale one.
//! * **The lit silhouette is non-trivial.** The topmost image cell varies from column
//!   to column, and many columns have an interior GAP (image, then non-image, then
//!   image again) where the alpha channel is transparent and the ray continued to the
//!   backdrop behind. Neither a flat-topped rectangle nor a procedurally filled wall
//!   does that.
//! * **Braille `U+28xx` glyphs appear** on the `subcell = braille` levels, in many
//!   distinct patterns. A Braille coverage glyph can only come from the sub-cell
//!   blitter fed by real texels.
//! * **The three sub-cell modes resolve DIFFERENT densities** on the same texture,
//!   and the plate's coarse half reads solid where its fine half reads partial —
//!   the two-scale property, measured on the frame.
//!
//! Pure compute: `composite_dump_once` at time 0, no terminal, no Bevy `App`, no
//! audio device — the same entry point `--dump` and the golden suite use.

use re_core::{Cell, Framebuffer};
use re_level::{load, load_string};
use re_render::{composite::composite_dump_once, registry_for_map, Emitter, Fov};
use re_world::CameraPose;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

/// Dump grid — identical to the golden-framebuffer suite and to `--dump`, so
/// these numbers describe the very frame the goldens hash.
const COLS: usize = 100;
const ROWS: usize = 32;

/// First and last code point of the Unicode Braille Patterns block. A glyph in this
/// range can only have come from `re_render::quadrant::blit_braille*`.
const BRAILLE_LO: u32 = 0x2800;
const BRAILLE_HI: u32 = 0x28FF;

/// The workspace root, from `CARGO_MANIFEST_DIR` (`crates/re_runtime`) stepped up
/// two. Compile-time, so nothing here depends on the process working directory.
fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

/// Render one shipped level at the golden grid, panicking with a useful message if
/// it will not load.
fn render_level(file: &str) -> Framebuffer {
    let path = workspace_root().join("levels").join(file);
    let lv = load(&path).unwrap_or_else(|e| panic!("{} must parse: {e}", path.display()));
    let map = lv.build_map();
    let cam: CameraPose = lv.spawn_camera();
    let registry = registry_for_map(&map);
    let emitters: Vec<Emitter> = lv.emitters.iter().map(Emitter::from_spec).collect();
    let mut fb = Framebuffer::new(COLS, ROWS);
    composite_dump_once(&map, &cam, &registry, &Fov::default(), &emitters, &mut fb);
    fb
}

/// Render a level authored as a STRING (used by the sub-cell probes, which differ
/// from each other only in one key).
fn render_text(text: &str) -> Framebuffer {
    let lv = load_string(text, "<probe>").expect("probe level must parse");
    let map = lv.build_map();
    let cam: CameraPose = lv.spawn_camera();
    let registry = registry_for_map(&map);
    let mut fb = Framebuffer::new(COLS, ROWS);
    composite_dump_once(&map, &cam, &registry, &Fov::default(), &[], &mut fb);
    fb
}

/// A cell painted by the image sampler: a visible glyph whose foreground is bright
/// and clearly RED-DOMINANT.
///
/// The margins (`>= 100` on red, `>= 40` over both other channels) are wide enough
/// that the renderer's distance shading cannot demote a lit image cell out of the
/// class, and narrow enough that the untextured fallback — which scores ZERO such
/// cells on every level asserted here — cannot fake one.
fn is_warm(cell: &Cell) -> bool {
    let r = i32::from(((cell.fg >> 16) & 0xFF) as u8);
    let g = i32::from(((cell.fg >> 8) & 0xFF) as u8);
    let b = i32::from((cell.fg & 0xFF) as u8);
    cell.glyph != u32::from(b' ') && r >= 100 && r >= g + 40 && r >= b + 40
}

/// True for the 256 Braille Patterns glyphs.
fn is_braille(glyph: u32) -> bool {
    (BRAILLE_LO..=BRAILLE_HI).contains(&glyph)
}

/// How many cells satisfy `pred`.
fn count(fb: &Framebuffer, pred: impl Fn(&Cell) -> bool) -> usize {
    (0..ROWS)
        .flat_map(|row| (0..COLS).map(move |col| (col, row)))
        .filter(|(col, row)| pred(fb.get(*col, *row).expect("in-bounds")))
        .count()
}

/// Every distinct glyph in the frame, with its cell count.
fn glyph_histogram(fb: &Framebuffer) -> BTreeMap<u32, usize> {
    let mut out = BTreeMap::new();
    for row in 0..ROWS {
        for col in 0..COLS {
            *out.entry(fb.get(col, row).expect("in-bounds").glyph)
                .or_insert(0) += 1;
        }
    }
    out
}

/// The shape of the image sampler's warm footprint, read column by column.
struct Silhouette {
    /// Columns containing at least one warm cell.
    columns: usize,
    /// Distinct row indices at which a column's warm run STARTS. A flat-topped
    /// rectangle yields one; a real silhouette yields several.
    distinct_tops: usize,
    /// Columns whose warm cells are INTERRUPTED — warm, then not, then warm again.
    /// Only a transparent texel between two opaque ones does that, so this counts
    /// the interior holes of the alpha silhouette.
    interrupted_columns: usize,
}

fn silhouette(fb: &Framebuffer) -> Silhouette {
    let mut tops: BTreeSet<usize> = BTreeSet::new();
    let mut columns = 0;
    let mut interrupted = 0;
    for col in 0..COLS {
        let warm: Vec<bool> = (0..ROWS)
            .map(|row| is_warm(fb.get(col, row).expect("in-bounds")))
            .collect();
        let (Some(first), Some(last)) =
            (warm.iter().position(|w| *w), warm.iter().rposition(|w| *w))
        else {
            continue;
        };
        columns += 1;
        tops.insert(first);
        if warm[first..=last].iter().any(|w| !*w) {
            interrupted += 1;
        }
    }
    Silhouette {
        columns,
        distinct_tops: tops.len(),
        interrupted_columns: interrupted,
    }
}

/// The message every colour assertion carries. Naming the asset is the whole point:
/// the failure mode being guarded is "somebody removed a texture file and nothing
/// said so".
fn missing_texture_hint(level: &str, asset: &str) -> String {
    format!(
        "\n\n  {level} binds `{asset}`. A count of ZERO means the image sampler did NOT run:\n  \
         `re_level::Level::build_map` skips a `[texture]` it cannot decode and renders the\n  \
         untextured procedural fallback WITHOUT reporting anything, so a missing or\n  \
         unreadable {asset} looks exactly like this. Check that the file exists and decodes.\n  \
         (This assertion exists because the golden hash alone cannot tell those apart.)"
    )
}

// ─────────────────────────────────────────────────────────────────────────────────
// TX5-genplate — the FREE, generated anchor. It is the only anchor: a
// copyrighted-asset anchor level and its positive-content assertions have no
// place in this suite, since every asset under `assets/` must be self-made or
// properly licensed (see `CONTRIBUTING.md`). TX5 alone carries this coverage.
// ─────────────────────────────────────────────────────────────────────────────────

/// The generated plate must be in the tree at all. Checked separately so its absence
/// reports the regeneration command instead of showing up as four confusing content
/// failures.
#[test]
fn the_generated_plate_asset_is_committed() {
    let p = workspace_root()
        .join("assets")
        .join("texture")
        .join("genplate_tex.png");
    assert!(
        p.is_file(),
        "missing {}. Regenerate it with:\n  \
         cargo run -p re_voximport --bin gen_texfixture -- assets/texture/genplate_tex.png\n\
         (source: crates/re_voximport/src/texfixture.rs — deterministic, integer-only)",
        p.display()
    );
}

#[test]
fn tx5_genplate_shows_real_image_colour() {
    let fb = render_level("TX5-genplate.lvl");
    let warm = count(&fb, is_warm);
    assert!(
        warm >= 100,
        "TX5-genplate: expected >=100 warm image cells, got {warm} (measured 325 with the \
         texture, 0 without). PROPERTY (b): the plate is saturated and red-dominant \
         precisely so that this number separates the image sampler from a procedural \
         recipe.{}",
        missing_texture_hint("TX5-genplate", "assets/texture/genplate_tex.png")
    );
}

#[test]
fn tx5_genplate_emits_braille_glyphs() {
    let fb = render_level("TX5-genplate.lvl");
    let hist = glyph_histogram(&fb);
    let cells: usize = hist
        .iter()
        .filter(|(g, _)| is_braille(**g))
        .map(|(_, n)| *n)
        .sum();
    let distinct = hist.keys().filter(|g| is_braille(**g)).count();
    assert!(
        cells >= 50,
        "TX5-genplate: expected >=50 Braille U+28xx cells from the `subcell = braille` back \
         wall, got {cells} (measured 262 with the texture, 0 without).{}",
        missing_texture_hint("TX5-genplate", "assets/texture/genplate_tex.png")
    );
    assert!(
        distinct >= 8,
        "TX5-genplate: only {distinct} distinct Braille patterns (measured 18)"
    );
}

#[test]
fn tx5_genplate_shows_a_non_trivial_alpha_silhouette() {
    let fb = render_level("TX5-genplate.lvl");
    let s = silhouette(&fb);
    assert!(
        s.columns >= 20,
        "TX5-genplate: only {} columns carry image cells.{}",
        s.columns,
        missing_texture_hint("TX5-genplate", "assets/texture/genplate_tex.png")
    );
    assert!(
        s.distinct_tops >= 3,
        "TX5-genplate: the image's top edge is flat ({} distinct top rows, measured 5). \
         PROPERTY (a): the plate's round head and tapering body exist to make it ragged.",
        s.distinct_tops
    );
    assert!(
        s.interrupted_columns >= 8,
        "TX5-genplate: only {} columns show the backdrop THROUGH the figure (measured 26). \
         PROPERTY (a): the window punched clean through the plate's body is what produces \
         these interior gaps, and it is what the image cut-out path rides on.",
        s.interrupted_columns
    );
}

// ─────────────────────────────────────────────────────────────────────────────────
// Sub-cell probes: the same generated plate on the same wall, differing ONLY in the
// `subcell` key, so every difference between the three frames IS the sub-cell mode.
// ─────────────────────────────────────────────────────────────────────────────────

/// One probe level. The texture path is ABSOLUTE, because a level parsed from a
/// string has no file to anchor a relative asset path to (`re_level::assets`
/// documents this) — an absolute path is its own single candidate.
fn subcell_probe(mode: &str) -> String {
    let plate = workspace_root()
        .join("assets")
        .join("texture")
        .join("genplate_tex.png");
    let tex = plate
        .canonicalize()
        .unwrap_or_else(|e| {
            panic!(
                "cannot read {}: {e}\nRegenerate it with:\n  \
                 cargo run -p re_voximport --bin gen_texfixture -- \
                 assets/texture/genplate_tex.png",
                plate.display()
            )
        })
        .to_string_lossy()
        .replace('\\', "/");
    format!(
        "[meta]\nname = subcell_probe_{mode}\n\
         [room]\nsize = 14 10 6\nambient = solid\n\
         [texture t]\nfile = {tex}\n\
         [material m]\nrecipe = image\ntexture = t\nsurface = wall_solid\n\
         span = 0,14 5,4\nsubcell = {mode}\n\
         [node a]\npos = 2,2,2\nis_input = yes\n\
         [node b]\npos = 11,2,2\nis_output = yes\n\
         [pipe a->b]\npath = (2,2,2) (11,2,2)\n\
         [pedestal]\ncenter = 11 2 1\nvolume = 1 1 1\n\
         [door]\npos = 13 5 1\n\
         [player]\nspawn = 5 3 1\nfacing = +y\n"
    )
}

/// How many cells hold a different glyph in the two frames.
fn glyph_diff(a: &Framebuffer, b: &Framebuffer) -> usize {
    (0..ROWS)
        .flat_map(|row| (0..COLS).map(move |col| (col, row)))
        .filter(|(col, row)| {
            a.get(*col, *row).expect("in-bounds").glyph
                != b.get(*col, *row).expect("in-bounds").glyph
        })
        .count()
}

/// PROPERTY (c), half one: the three sub-cell modes are three different renders of
/// the same plate, not three copies of one blob. Measured: 14 / 20 / 33 distinct
/// glyphs for off / quad / braille, and 83 / 271 / 265 differing cells pairwise.
#[test]
fn tx5_genplate_subcell_modes_resolve_different_densities() {
    let off = render_text(&subcell_probe("off"));
    let quad = render_text(&subcell_probe("quad"));
    let braille = render_text(&subcell_probe("braille"));

    let (n_off, n_quad, n_braille) = (
        glyph_histogram(&off).len(),
        glyph_histogram(&quad).len(),
        glyph_histogram(&braille).len(),
    );
    assert!(
        n_off < n_quad && n_quad < n_braille,
        "sub-cell modes must resolve strictly finer: off={n_off}, quad={n_quad}, \
         braille={n_braille} distinct glyphs (measured 14 / 20 / 33). Equal counts mean the \
         plate carries no detail below one screen cell — property (c) of the fixture."
    );

    for (name, d) in [
        ("off vs quad", glyph_diff(&off, &quad)),
        ("off vs braille", glyph_diff(&off, &braille)),
        ("quad vs braille", glyph_diff(&quad, &braille)),
    ] {
        assert!(
            d >= 30,
            "{name}: only {d} cells differ (measured 83 / 271 / 265). The three modes must \
             render VISIBLY differently on this plate."
        );
    }
}

/// PROPERTY (c), half two: the plate's COARSE half (16-texel checker) reads mostly as
/// fully-inked Braille cells while its FINE half (2-texel checker) reads mostly as
/// PARTIAL coverage patterns. That is detail at two scales, measured on the frame.
///
/// The wall spans the full view, so the plate's left half lands on the left half of
/// the frame; the split at column 50 is the frame's midpoint, not a tuned constant.
#[test]
fn tx5_genplate_carries_detail_at_two_scales() {
    let fb = render_text(&subcell_probe("braille"));
    let mid = COLS / 2;
    let mut coarse = (0usize, 0usize); // (fully inked, partial)
    let mut fine = (0usize, 0usize);
    for row in 0..ROWS {
        for col in 0..COLS {
            let glyph = fb.get(col, row).expect("in-bounds").glyph;
            if !is_braille(glyph) {
                continue;
            }
            let mask = glyph - BRAILLE_LO;
            let saturated = mask == 0x00 || mask == 0xFF;
            let bucket = if col < mid { &mut coarse } else { &mut fine };
            if saturated {
                bucket.0 += 1;
            } else {
                bucket.1 += 1;
            }
        }
    }
    assert!(
        coarse.0 + coarse.1 >= 20 && fine.0 + fine.1 >= 20,
        "not enough Braille cells to compare halves: coarse {coarse:?}, fine {fine:?}"
    );
    assert!(
        coarse.0 > coarse.1,
        "the plate's COARSE half must read mostly SOLID: {} fully-inked vs {} partial \
         (measured 109 vs 25)",
        coarse.0,
        coarse.1
    );
    assert!(
        fine.1 > fine.0,
        "the plate's FINE half must read mostly PARTIAL: {} partial vs {} fully-inked \
         (measured 83 vs 45)",
        fine.1,
        fine.0
    );
}
