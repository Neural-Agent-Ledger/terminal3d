//! Integration tests against the REAL committed generated `.vox` assets (see
//! `re_voxscene::genmodels` — the free, self-made replacement for a removed,
//! unlicensed third-party voxel kit): all 5 models load, and the end-to-end
//! load → layout → assemble → map → render pipeline produces a non-empty,
//! in-bounds, overlap-free town.

use std::path::PathBuf;

use re_voxscene::{
    assemble_town, layout_town, load_generated_town, overview_camera, render_ascii,
    street_level_camera, town_to_map,
};

/// The authoring root = `<workspace>/authoring/`.
/// (`editor/re_voxscene` → `editor/` → workspace root → `authoring/`).
fn authoring_root() -> PathBuf {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    crate_dir
        .parent() // editor/
        .and_then(|p| p.parent()) // workspace root
        .expect("workspace root")
        .join("authoring")
}

#[test]
fn all_five_models_load() {
    let buildings =
        load_generated_town(&authoring_root()).expect("all 5 generated models must load");
    assert_eq!(buildings.len(), 5, "expected exactly 5 buildings");
    for b in &buildings {
        assert!(b.solid_count() > 0, "{} has no solid voxels", b.name);
        assert!(
            b.footprint_x() > 0 && b.footprint_y() > 0 && b.height() > 0,
            "{} has a degenerate extent {:?}",
            b.name,
            b.extent
        );
    }
}

#[test]
fn full_pipeline_assembles_and_renders() {
    let buildings = load_generated_town(&authoring_root()).expect("load 5 models");
    let layout = layout_town(&buildings, 3, 6);
    let town = assemble_town(buildings, layout).expect("the generated town must have no overlaps");

    assert!(
        town.report.pass,
        "collision check must PASS on the generated town"
    );
    assert_eq!(town.report.overlap_count, 0);
    assert!(town.report.min_gap >= 6, "street margin must be respected");

    let map = town_to_map(&town);
    let (sx, sy, sz) = map.size();
    assert!(sx > 0 && sy > 0 && sz > 1);

    // Both preview frames render non-blank with our raycaster, and the two
    // camera angles show genuinely different silhouettes (a regression guard
    // against a renderer that returns the same frame regardless of pose).
    let mut frames = Vec::new();
    for cam in [street_level_camera(&town), overview_camera(&town)] {
        let frame = render_ascii(&map, &cam, 120, 40);
        let nonspace = frame.chars().filter(|&c| c != ' ' && c != '\n').count();
        assert!(
            nonspace > 100,
            "rendered frame must be substantially non-blank"
        );
        frames.push(frame);
    }
    assert_ne!(
        frames[0], frames[1],
        "street-level and overview cameras must render distinct silhouettes"
    );
}
