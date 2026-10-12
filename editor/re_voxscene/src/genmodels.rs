//! Procedurally generated `.vox` building models — the free, self-made
//! replacement for a removed, unlicensed third-party voxel kit.
//!
//! # Why this exists
//!
//! `re_voxscene`'s only content used to be 9 buildings copied from a downloaded
//! MagicaVoxel store pack of unknown licence, deleted with owner approval.
//! Deleting that kit left the crate's loader and its
//! `full_pipeline_assembles_and_renders` test with nothing to load.
//!
//! This module is the replacement anchor, following the same "the generator is
//! the source of truth" pattern as `re_voximport::texfixture`: every model is
//! produced by pure, deterministic integer arithmetic (no RNG, no external
//! asset, no floats) via [`generated_models`], then written to real `.vox`
//! files under `authoring/voxelmodels/generated/` by the `gen_voxmodels`
//! binary, so the geometry is reproducible from source rather than an opaque
//! downloaded blob and can never be a licensing question again.
//!
//! # Regenerating the assets
//!
//! ```sh
//! cargo run -p re_voxscene --bin gen_voxmodels
//! ```
//!
//! Run from the workspace root. Output is byte-stable: [`generated_models`]
//! takes no input, so re-running never produces a diff. The crate's unit test
//! `every_committed_vox_matches_its_generator` decodes each committed file and
//! compares its geometry pixel-for-pixel (voxel-for-voxel) against the
//! generator, so an asset can never silently drift from its source.
//!
//! # The five models
//!
//! Five is comfortably inside the "3-5 models" the replacement kit needs, and
//! each exercises a different silhouette so [`crate::layout_town`] still has
//! genuinely different footprints/heights to lay out and the town preview
//! reads as a real skyline rather than five copies of one box:
//!
//! | stem            | shape                                    | notable feature |
//! |------------------|-------------------------------------------|-----------------|
//! | `house01`        | small hollow box house                    | one door, one back window |
//! | `house02`        | larger hollow box house                   | door + 4 windows (front pair + both sides) |
//! | `tower01`        | narrow, tall hollow shaft                 | a repeating window band up all four walls |
//! | `lshape01`       | two boxes unioned into an L footprint     | non-convex silhouette, a door on each wing |
//! | `warehouse01`    | wide, flat, low hollow box                | a wide door + periodic roof skylights |
//!
//! Every model is a **hollow shell**: solid walls (one voxel thick) around the
//! footprint plus a solid roof cap, with rectangular gaps cut for doors,
//! windows, and (on `warehouse01`) roof skylights. That is deliberately the
//! same "geometry only, no interior detail" level of fidelity the original
//! third-party kit's models were treated at by [`crate::load_building`] — it
//! discards palette information and only ever reads "is this voxel solid",
//! so the replacement only has to be honest about which cells are solid.
//!
//! Palettes are real but decorative: `re_voxscene` (like the deleted kit)
//! discards palette data entirely, keeping GEOMETRY ONLY. A distinct
//! (wall, roof) colour pair per model is still written so the `.vox` files are
//! inspectable and visually distinguishable in a MagicaVoxel-compatible viewer.

use dot_vox::{Color, DotVoxData, Model, Size, Voxel};

/// A generated building, ready to serialise as a standalone `.vox` file.
#[derive(Clone, Debug)]
pub struct GeneratedModel {
    /// File stem (no extension) — the committed asset is
    /// `authoring/voxelmodels/generated/<stem>.vox`.
    pub stem: &'static str,
    /// Footprint + height in voxels (width, depth, height).
    pub size: (u32, u32, u32),
    /// Solid voxels as `(x, y, z, palette_index)`, already tight to `(0,0,0)`.
    pub voxels: Vec<(u32, u32, u32, u8)>,
    /// A real (if decorative) 256-entry palette.
    pub palette: Vec<Color>,
}

impl GeneratedModel {
    /// Build the `dot_vox` container for this model (one model, no scene graph
    /// — `load_building_from_bytes` merges whatever models it finds, and one
    /// per file is exactly what the original kit shipped too).
    #[must_use]
    pub fn to_dot_vox(&self) -> DotVoxData {
        let (sx, sy, sz) = self.size;
        DotVoxData {
            version: 150,
            index_map: Vec::new(),
            models: vec![Model {
                size: Size {
                    x: sx,
                    y: sy,
                    z: sz,
                },
                voxels: self
                    .voxels
                    .iter()
                    .map(|&(x, y, z, i)| Voxel {
                        x: x as u8,
                        y: y as u8,
                        z: z as u8,
                        i,
                    })
                    .collect(),
            }],
            palette: self.palette.clone(),
            materials: Vec::new(),
            scenes: Vec::new(),
            layers: Vec::new(),
        }
    }

    /// Encode this model as `.vox` bytes.
    ///
    /// # Errors
    ///
    /// Returns the encoder's message on failure. `dot_vox`'s writer cannot
    /// actually fail for a well-formed model like these, but the generator
    /// binary reports it rather than panicking, matching `texfixture::plate_png`.
    pub fn write_bytes(&self) -> Result<Vec<u8>, String> {
        let mut bytes = Vec::new();
        self.to_dot_vox()
            .write_vox(&mut bytes)
            .map_err(|e| format!("{}.vox encode failed: {e}", self.stem))?;
        Ok(bytes)
    }
}

/// Palette index used for wall voxels.
const WALL_I: u8 = 0;
/// Palette index used for roof voxels (takes priority over `WALL_I` where a
/// voxel is both — the top ring of the walls).
const ROOF_I: u8 = 1;

/// A real but decorative 256-entry palette: index 0 = wall colour, index 1 =
/// roof colour, the rest a flat neutral grey (unused — `re_voxscene` discards
/// palette data entirely; this only keeps the file well-formed and viewable).
fn palette_for(wall: [u8; 3], roof: [u8; 3]) -> Vec<Color> {
    let neutral = Color {
        r: 110,
        g: 110,
        b: 110,
        a: 255,
    };
    let mut p = vec![neutral; 256];
    p[WALL_I as usize] = Color {
        r: wall[0],
        g: wall[1],
        b: wall[2],
        a: 255,
    };
    p[ROOF_I as usize] = Color {
        r: roof[0],
        g: roof[1],
        b: roof[2],
        a: 255,
    };
    p
}

/// Build a hollow box shell: solid walls (footprint boundary, one voxel thick)
/// at every `z`, plus a fully solid roof cap at `z == h - 1`. Pure integer
/// arithmetic, deterministic, no floats, no randomness.
fn box_shell(w: u32, d: u32, h: u32) -> Vec<(u32, u32, u32, u8)> {
    let mut v = Vec::new();
    for z in 0..h {
        let is_roof = z == h - 1;
        for y in 0..d {
            for x in 0..w {
                let is_wall = x == 0 || x == w - 1 || y == 0 || y == d - 1;
                if is_wall || is_roof {
                    v.push((x, y, z, if is_roof { ROOF_I } else { WALL_I }));
                }
            }
        }
    }
    v
}

/// Remove every voxel matching `pred(x, y, z)` — used to cut door/window/
/// skylight gaps out of a shell.
fn cut_gap(voxels: &mut Vec<(u32, u32, u32, u8)>, pred: impl Fn(u32, u32, u32) -> bool) {
    voxels.retain(|&(x, y, z, _)| !pred(x, y, z));
}

/// `house01` — a small hollow house: one door on the front wall (`y == 0`),
/// one window on the back wall (`y == d - 1`).
fn house01() -> GeneratedModel {
    const W: u32 = 9;
    const D: u32 = 7;
    const H: u32 = 6;
    let mut v = box_shell(W, D, H);
    // Door: width 3, ground level + 1 (z in 0..=1), centred on the front wall.
    cut_gap(&mut v, |x, y, z| y == 0 && (3..=5).contains(&x) && z <= 1);
    // Back window: width 3, mid-height.
    cut_gap(&mut v, |x, y, z| {
        y == D - 1 && (3..=5).contains(&x) && (3..=4).contains(&z)
    });
    GeneratedModel {
        stem: "house01",
        size: (W, D, H),
        voxels: v,
        palette: palette_for([196, 168, 120], [120, 70, 60]),
    }
}

/// `house02` — a larger hollow house: a door flanked by two front windows,
/// plus one window on each side wall.
fn house02() -> GeneratedModel {
    const W: u32 = 13;
    const D: u32 = 10;
    const H: u32 = 8;
    let mut v = box_shell(W, D, H);
    // Door: width 3, ground level + 1, centred.
    cut_gap(&mut v, |x, y, z| y == 0 && (5..=7).contains(&x) && z <= 1);
    // Two front windows flanking the door, mid-height.
    cut_gap(&mut v, |x, y, z| {
        y == 0 && (1..=2).contains(&x) && (3..=4).contains(&z)
    });
    cut_gap(&mut v, |x, y, z| {
        y == 0 && (10..=11).contains(&x) && (3..=4).contains(&z)
    });
    // One window per side wall, upper storey.
    cut_gap(&mut v, |x, y, z| {
        x == 0 && (3..=4).contains(&y) && (4..=5).contains(&z)
    });
    cut_gap(&mut v, |x, y, z| {
        x == W - 1 && (3..=4).contains(&y) && (4..=5).contains(&z)
    });
    GeneratedModel {
        stem: "house02",
        size: (W, D, H),
        voxels: v,
        palette: palette_for([150, 160, 168], [70, 78, 90]),
    }
}

/// `tower01` — a narrow, tall shaft with a repeating window band up all four
/// walls (a genuinely different height/footprint ratio from the houses, so
/// [`re_voxscene::layout_town`] lays it out distinctly).
fn tower01() -> GeneratedModel {
    const W: u32 = 6;
    const D: u32 = 6;
    const H: u32 = 20;
    let mut v = box_shell(W, D, H);
    // A door at the base (front wall only).
    cut_gap(&mut v, |x, y, z| y == 0 && (2..=3).contains(&x) && z <= 1);
    // A small window band on every wall, repeated every 4 levels, stopping
    // short of the roof cap.
    let mut z0 = 3u32;
    while z0 + 1 < H - 1 {
        let (z0c, z1c) = (z0, z0 + 1);
        cut_gap(&mut v, move |x, y, z| {
            y == 0 && x == W / 2 && (z0c..=z1c).contains(&z)
        });
        cut_gap(&mut v, move |x, y, z| {
            y == D - 1 && x == W / 2 && (z0c..=z1c).contains(&z)
        });
        cut_gap(&mut v, move |x, y, z| {
            x == 0 && y == D / 2 && (z0c..=z1c).contains(&z)
        });
        cut_gap(&mut v, move |x, y, z| {
            x == W - 1 && y == D / 2 && (z0c..=z1c).contains(&z)
        });
        z0 += 4;
    }
    GeneratedModel {
        stem: "tower01",
        size: (W, D, H),
        voxels: v,
        palette: palette_for([90, 100, 130], [50, 55, 70]),
    }
}

/// `lshape01` — two hollow boxes unioned into an L footprint: box A (the main
/// wing) and box B (the side wing) offset so their footprints share an edge
/// but the outer silhouette is non-convex. Each wing keeps its own door.
fn lshape01() -> GeneratedModel {
    // Wing A: the long top bar of the L.
    const AW: u32 = 10;
    const AD: u32 = 6;
    // Wing B: the side wing extending down from A's right side.
    const BW: u32 = 5;
    const BD: u32 = 9;
    const H: u32 = 7;
    // B is offset so it starts under A's right portion, extending the
    // footprint further in +y — the union is an L, not a rectangle.
    const B_OFF_X: u32 = 5;
    const B_OFF_Y: u32 = AD; // B begins exactly where A's depth ends

    let mut a = box_shell(AW, AD, H);
    // Door on wing A's front (north) wall.
    cut_gap(&mut a, |x, y, z| y == 0 && (3..=5).contains(&x) && z <= 1);

    let mut b = box_shell(BW, BD, H);
    // Door on wing B's outer (east) wall, near the bottom of the wing.
    cut_gap(&mut b, |x, y, z| {
        x == BW - 1 && (BD - 3..=BD - 1).contains(&y) && z <= 1
    });
    // A window on wing B's outer (west) wall, mid-height.
    cut_gap(&mut b, |x, y, z| {
        x == 0 && (3..=5).contains(&y) && (3..=4).contains(&z)
    });

    let mut v = a;
    v.extend(
        b.into_iter()
            .map(|(x, y, z, i)| (x + B_OFF_X, y + B_OFF_Y, z, i)),
    );
    // The union may double-write the shared boundary voxels; dedup so the
    // model has no duplicate (x,y,z) entries (mirrors `load_building`'s own
    // normalise/dedup step for real files).
    v.sort_unstable();
    v.dedup_by(|k, o| (k.0, k.1, k.2) == (o.0, o.1, o.2));

    let total_w = AW.max(B_OFF_X + BW);
    let total_d = B_OFF_Y + BD;
    GeneratedModel {
        stem: "lshape01",
        size: (total_w, total_d, H),
        voxels: v,
        palette: palette_for([180, 130, 90], [100, 60, 45]),
    }
}

/// `warehouse01` — a wide, flat, low hollow box with a wide door and a row of
/// periodic roof skylight cutouts.
fn warehouse01() -> GeneratedModel {
    const W: u32 = 20;
    const D: u32 = 10;
    const H: u32 = 5;
    let mut v = box_shell(W, D, H);
    // Wide front door (width 6).
    cut_gap(&mut v, |x, y, z| y == 0 && (7..=12).contains(&x) && z <= 2);
    // Periodic 2x2 roof skylights every 4 columns, clear of the door span.
    let mut x0 = 2u32;
    while x0 + 1 < W - 2 {
        let (x0c, x1c) = (x0, x0 + 1);
        cut_gap(&mut v, move |x, y, z| {
            z == H - 1 && (x0c..=x1c).contains(&x) && (4..=5).contains(&y)
        });
        x0 += 4;
    }
    GeneratedModel {
        stem: "warehouse01",
        size: (W, D, H),
        voxels: v,
        palette: palette_for([140, 140, 120], [80, 80, 65]),
    }
}

/// All 5 generated models, in a fixed deterministic order (the order
/// `re_voxscene::layout_town` lays them out in matters for reproducibility).
#[must_use]
pub fn generated_models() -> Vec<GeneratedModel> {
    vec![house01(), house02(), tower01(), lshape01(), warehouse01()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Path of a committed generated asset, derived at COMPILE time so the
    /// test does not depend on the process working directory.
    fn committed_vox(stem: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("authoring")
            .join("voxelmodels")
            .join("generated")
            .join(format!("{stem}.vox"))
    }

    #[test]
    fn generation_is_deterministic() {
        for m in generated_models() {
            let a = m.write_bytes().expect("encodes");
            let b = m.write_bytes().expect("encodes");
            assert_eq!(a, b, "{} must encode identically on repeat calls", m.stem);
        }
    }

    #[test]
    fn exactly_five_models_with_distinct_stems_and_real_geometry() {
        let models = generated_models();
        assert_eq!(models.len(), 5, "expected exactly 5 generated models");
        let mut stems: Vec<&str> = models.iter().map(|m| m.stem).collect();
        stems.sort_unstable();
        stems.dedup();
        assert_eq!(stems.len(), 5, "stems must be distinct");
        for m in &models {
            assert!(!m.voxels.is_empty(), "{} has no solid voxels", m.stem);
            let (w, d, h) = m.size;
            assert!(w > 0 && d > 0 && h > 0, "{} has a degenerate size", m.stem);
            // Every voxel must sit within the declared size.
            for &(x, y, z, _) in &m.voxels {
                assert!(x < w && y < d && z < h, "{} voxel out of bounds", m.stem);
            }
        }
    }

    #[test]
    fn models_have_varied_footprints_and_heights() {
        // The whole point of 5 distinct shapes: no two models share a
        // (width, depth, height) triple.
        let models = generated_models();
        let mut sizes: Vec<(u32, u32, u32)> = models.iter().map(|m| m.size).collect();
        sizes.sort_unstable();
        sizes.dedup();
        assert_eq!(sizes.len(), 5, "every model must have a distinct size");
    }

    #[test]
    fn doors_and_windows_are_real_gaps_in_the_shell() {
        // house01: the door gap at (4, 0, 0) must be ABSENT (not solid).
        let h1 = house01();
        assert!(
            !h1.voxels.iter().any(|&(x, y, z, _)| (x, y, z) == (4, 0, 0)),
            "house01 door must be a real gap"
        );
        // But the wall right beside the door, same wall, must still be solid.
        assert!(
            h1.voxels.iter().any(|&(x, y, z, _)| (x, y, z) == (0, 0, 0)),
            "house01 front wall corner must remain solid"
        );
    }

    #[test]
    fn lshape01_footprint_is_genuinely_non_convex() {
        // The far corner of wing A's footprint extension (beyond wing B's
        // width) must have NO voxel at any z for the row just past wing A's
        // depth — that empty notch is what makes the silhouette an L rather
        // than a rectangle.
        let l = lshape01();
        let (_, _, h) = l.size;
        // Wing B occupies x in [5, 10), y in [6, 15). The notch is x in
        // [0, 5), y in [6, 15) — should be entirely empty.
        for z in 0..h {
            for y in 6..15 {
                for x in 0..5 {
                    assert!(
                        !l.voxels.contains(&(x, y, z, WALL_I))
                            && !l.voxels.contains(&(x, y, z, ROOF_I)),
                        "L-shape notch at ({x},{y},{z}) must be empty"
                    );
                }
            }
        }
    }

    #[test]
    fn every_committed_vox_matches_its_generator() {
        for m in generated_models() {
            let path = committed_vox(m.stem);
            let bytes = std::fs::read(&path).unwrap_or_else(|e| {
                panic!(
                    "the generated fixture must be committed at {}: {e}\n\
                     Regenerate with: cargo run -p re_voxscene --bin gen_voxmodels",
                    path.display()
                )
            });
            let generated = m.write_bytes().expect("encodes");
            // Compare DECODED geometry, not raw file bytes: `dot_vox`'s own
            // encoder output is what we control, but comparing through a
            // round-trip parse is the strongest guarantee it stays loadable.
            let committed_parsed = dot_vox::load_bytes(&bytes)
                .unwrap_or_else(|e| panic!("committed {} must parse: {e}", path.display()));
            let generated_parsed =
                dot_vox::load_bytes(&generated).expect("generated bytes must parse");
            assert_eq!(
                committed_parsed.models.len(),
                generated_parsed.models.len(),
                "{}: model count must match",
                m.stem
            );
            let mut committed_voxels = committed_parsed.models[0].voxels.clone();
            let mut generated_voxels = generated_parsed.models[0].voxels.clone();
            let key = |v: &dot_vox::Voxel| (v.x, v.y, v.z, v.i);
            committed_voxels.sort_by_key(key);
            generated_voxels.sort_by_key(key);
            assert_eq!(
                committed_voxels.iter().map(key).collect::<Vec<_>>(),
                generated_voxels.iter().map(key).collect::<Vec<_>>(),
                "the committed {} does not match its generator; regenerate it with\n  \
                 cargo run -p re_voxscene --bin gen_voxmodels",
                path.display()
            );
        }
    }

    /// Each encoded `.vox` stays small enough to belong in the repository.
    #[test]
    fn every_encoded_vox_is_a_few_kilobytes() {
        for m in generated_models() {
            let bytes = m.write_bytes().expect("encodes");
            assert!(
                bytes.len() < 32 * 1024,
                "{} is {} bytes",
                m.stem,
                bytes.len()
            );
        }
    }
}
