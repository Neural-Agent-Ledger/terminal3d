//! `levels/FX-detail.lvl` is an engine demo level and the reveal it proves —
//! `re_render`'s detail-LOD cross-fade over a level `re_level` loads — is
//! entirely engine behaviour, with no game crate involved. `re_runtime`
//! already depends on `re_level` and `re_render` directly.
//!
//! `levels/FX-detail.lvl` — the DETAIL-TIER scripted approach test.
//!
//! `FX-detail.lvl`'s `north_detail_wall` material is the ONLY authored material
//! that opts in with `detail = on`. The acceptance this proves is "approach
//! dumps (distance 8 → 1) show fine detail fading in smoothly" — this test is
//! that proof, scripted rather than eyeballed: it walks the SAME camera the
//! `--dump` CLI path uses straight at the wall through five distances and
//! asserts the `re_render::detail_lod`-gated micro-detail glyph (`:`,
//! `DETAIL_MICRO_GLYPH` in `re_render::ascii_shader`) COUNT rises
//! monotonically, never falls, as the camera approaches — i.e. the reveal is
//! a smooth cross-fade, never a pop, and world-anchored
//! (`re_render::detail_lod::reveal_finer`), never screen-keyed. (A
//! room this size never reaches pure TONE/PATTERN — `Footprint::reference`'s
//! reference config puts even "distance 8" deep in the DETAIL band — so the
//! interesting, testable claim is the MONOTONIC RISE, not an absolute zero at
//! the far end; the `detail_lod` module's own unit tests already cover the
//! TONE/PATTERN boundary in isolation.)
//!
//! Uses the identical render entry point `--dump` and the golden-framebuffer
//! suite both use (`re_render::composite::composite_dump_once`), so this is the
//! REAL live-render path, not a synthetic recipe-only sample.

use std::path::Path;

use re_core::glam::Vec3;
use re_core::Framebuffer;
use re_level::load;
use re_render::{composite::composite_dump_once, registry_for_map, Emitter, Fov};
use re_world::CameraPose;

const COLS: usize = 100;
const ROWS: usize = 32;

/// The glyph `Material::sample_grain` emits for its B39 opt-in micro-detail
/// texel (`DETAIL_MICRO_GLYPH` in `re_render::ascii_shader` — private to that
/// module, so mirrored here by value; a drift between the two would only make
/// this test UNDER-count, never falsely pass).
const DETAIL_MICRO_GLYPH: u32 = b':' as u32;

fn level_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .unwrap()
        .join("levels")
        .join("FX-detail.lvl")
}

/// Render one frame with the camera at `(5.5, cam_y, 1.5)` facing `+y` (straight
/// at `north_detail_wall`), and count how many cells show the micro-detail glyph.
fn micro_detail_count_at(cam_y: f32) -> usize {
    let lv = load(level_path()).expect("FX-detail.lvl must parse");
    let map = lv.build_map();
    let registry = registry_for_map(&map);
    let emitters: Vec<Emitter> = lv.emitters.iter().map(Emitter::from_spec).collect();
    let cam = CameraPose {
        pos: Vec3::new(5.5, cam_y, 1.5),
        yaw: 0.0,
        pitch: 0.0,
    };

    let mut fb = Framebuffer::new(COLS, ROWS);
    composite_dump_once(&map, &cam, &registry, &Fov::default(), &emitters, &mut fb);

    let mut n = 0usize;
    for row in 0..ROWS {
        for col in 0..COLS {
            if fb
                .get(col, row)
                .is_some_and(|c| c.glyph == DETAIL_MICRO_GLYPH)
            {
                n += 1;
            }
        }
    }
    n
}

#[test]
fn fx_detail_level_loads_and_dumps_non_empty() {
    let lv = load(level_path()).expect("FX-detail.lvl must parse");
    let map = lv.build_map();
    let cam = lv.spawn_camera();
    let registry = registry_for_map(&map);
    let emitters: Vec<Emitter> = lv.emitters.iter().map(Emitter::from_spec).collect();
    let mut fb = Framebuffer::new(COLS, ROWS);
    composite_dump_once(&map, &cam, &registry, &Fov::default(), &emitters, &mut fb);
    let non_blank = (0..ROWS)
        .flat_map(|r| (0..COLS).map(move |c| (c, r)))
        .any(|(c, r)| {
            fb.get(c, r)
                .is_some_and(|cell| cell.glyph != u32::from(' '))
        });
    assert!(
        non_blank,
        "FX-detail.lvl must render something, not a blank frame"
    );
}

/// The headline B39 proof: approaching `north_detail_wall` from distance ~8
/// down to ~0.5 must reveal STRICTLY more micro-detail near than far, and the
/// rise across the whole approach must be monotone (never a step back).
#[test]
fn approach_reveals_more_micro_detail_never_less() {
    // cam_y sweeps 1.5 (distance ~7.5 from the y=9 wall face) up to 8.5
    // (distance ~0.5) — five samples spanning the spec's "distance 8 -> 1".
    let samples = [1.5f32, 3.5, 5.5, 7.5, 8.5];
    let counts: Vec<usize> = samples.iter().map(|&y| micro_detail_count_at(y)).collect();

    assert!(
        counts[0] > 0,
        "even the farthest sample must show SOME micro-detail, got counts={counts:?}"
    );
    assert!(
        counts[counts.len() - 1] > counts[0] * 4,
        "up close must reveal MUCH more micro-detail than far away, got counts={counts:?}"
    );
    for w in counts.windows(2) {
        assert!(
            w[1] >= w[0],
            "micro-detail count must never DECREASE as the camera approaches: counts={counts:?}"
        );
    }
}
