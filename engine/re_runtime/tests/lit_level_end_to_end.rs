//! The level under test (`levels/22-lit-cellar.lvl`) is an engine demo level,
//! not campaign, and the chain it exercises — `re_level` parse → `Map` →
//! `re_render::light_env_for_map` bake → raycaster shading — is entirely
//! `re_level` + `re_render` + `re_world`, all of which `re_runtime` already
//! depends on directly.
//!
//! END-TO-END for the per-level LIGHTING grammar: `[light]` / `[atmosphere]`
//! authored in a `.lvl` must survive the parser, reach the `Map`, and actually
//! change shaded pixels.

use re_core::Framebuffer;
use re_level::load_string;
use re_world::CameraPose;

/// Mean foreground luminance (Rec.601) over the whole frame, in `[0,255]`.
/// Measures VISIBLE luminance (see the per-cell rule below), not ink alone.
/// Every cell counts: a lamp that only brightens the cells it reaches still
/// lifts the mean, and using the full frame keeps the measure free of any
/// hand-picked region.
fn mean_luma(fb: &Framebuffer) -> f64 {
    let cells = fb.cells();
    if cells.is_empty() {
        return 0.0;
    }
    let sum: f64 = cells
        .iter()
        .map(|c| {
            // What the eye sees: a blank cell shows only its bg; an inked cell
            // shows the brighter of ink and ground. Ink colour on a blank cell is
            // never drawn, and lamp light lands on the bg of floor cells.
            let luma = |rgb: u32| {
                0.299 * ((rgb >> 16) & 0xFF) as f64
                    + 0.587 * ((rgb >> 8) & 0xFF) as f64
                    + 0.114 * (rgb & 0xFF) as f64
            };
            if c.glyph == u32::from(' ') {
                luma(c.bg)
            } else {
                luma(c.fg).max(luma(c.bg))
            }
        })
        .sum();
    sum / cells.len() as f64
}

/// Cast one 100×32 frame of `text` from `cam`.
fn cast(text: &str, cam: &CameraPose) -> Framebuffer {
    let lv = load_string(text, "<lit>").expect("level must parse");
    let map = lv.build_map();
    let mut fb = Framebuffer::new(100, 32);
    re_render::raycast(&map, cam, &mut fb);
    fb
}

/// The same source with every `[light]` section removed — the level as it would
/// have been authored before the lighting grammar existed.
fn without_lights(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut in_light = false;
    for line in raw.lines() {
        let t = line.trim();
        if t.starts_with('[') {
            in_light = t.eq_ignore_ascii_case("[light]");
        }
        if !in_light {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

fn cellar_source() -> String {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../levels/22-lit-cellar.lvl");
    std::fs::read_to_string(&path).expect("the lighting demo level exists")
}

#[test]
fn the_demo_levels_lighting_reaches_the_map_and_the_runtime_light_set() {
    let lv = load_string(&cellar_source(), "22-lit-cellar").expect("demo level parses");
    let map = lv.build_map();
    let atmo = map.atmosphere();

    assert_eq!(atmo.lights.len(), 2, "the cellar authors two lamps");
    assert!(atmo.has_lighting(), "an authored lamp counts as lighting");
    assert_eq!(atmo.ambient, Some(0.12), "the cellar pulls ambient down");
    assert!(
        atmo.key_dir.is_some() && atmo.key_color.is_some(),
        "a dimmed warm key"
    );
    let h = atmo.fog_height.expect("the cellar authors height fog");
    assert!(h.falloff > 0.0, "height fog must actually thin upward");
    assert_eq!(
        atmo.postfx.as_deref(),
        Some("clean"),
        "the level requests a preset"
    );

    // Both lamps are live in the renderer's baked set (neither was dropped for a
    // bad radius/intensity, and the 4-light cap is not in play).
    let env = re_render::light_env_for_map(&map);
    assert_eq!(env.points.len(), 2, "both lamps reach the shader");
}

#[test]
fn stripping_the_lamps_darkens_the_demo_level() {
    // Identical geometry, identical materials, identical camera — the ONLY
    // difference is the two `[light]` sections. If the lamps did not reach the
    // shader the two frames would be identical.
    let raw = cellar_source();
    let dark = without_lights(&raw);
    assert!(
        !dark
            .lines()
            .any(|l| l.trim().eq_ignore_ascii_case("[light]")),
        "the control level has no lamp SECTIONS (a `[light]` inside a `#` comment is fine)"
    );

    // Standing under the warm lamp at (4, 5), facing along +x down the room.
    let cam = CameraPose::spawn_at(4, 5, 1, 120);
    let lit_luma = mean_luma(&cast(&raw, &cam));
    let dark_luma = mean_luma(&cast(&dark, &cam));

    assert!(
        lit_luma > dark_luma + 1.0,
        "the lamps must visibly light the room (lit {lit_luma:.2} vs unlit {dark_luma:.2})"
    );
}

/// A minimal room with ONE lamp whose position is substituted in, so a near/far
/// pair differs in nothing but the lamp's distance from the camera.
fn one_lamp_room(lamp_pos: &str) -> String {
    format!(
        "[meta]\nname = lamp_falloff\n\
         [room]\nsize = 24 9 4\nambient = solid\n\
         [atmosphere]\nambient = 0.05\n\
         [light]\npos = {lamp_pos}\ncolor = #FFFFFF\nradius = 3.0\nintensity = 2.0\n\
         [node a]\npos = 2 4 2\nis_input = yes\n\
         [node b]\npos = 21 4 2\nis_output = yes\n\
         [pipe a->b]\npath = (2,4,2) (21,4,2)\n\
         [pedestal]\ncenter = 21 4 1\nvolume = 1 1 1\n\
         [door]\npos = 23 7 1\n\
         [player]\nspawn = 4 4 1\n"
    )
}

#[test]
fn a_lamp_beside_the_camera_lights_more_than_the_same_lamp_across_the_room() {
    // The falloff is `1/(1 + d²/radius²)`, so a lamp 2 cells away must beat the
    // same lamp 18 cells away by a wide margin — this is the pooling that makes
    // authoring a light worth doing.
    let cam = CameraPose::spawn_at(4, 4, 1, 120);
    let near = mean_luma(&cast(&one_lamp_room("5.0, 4.0, 2.0"), &cam));
    let far = mean_luma(&cast(&one_lamp_room("21.0, 4.0, 2.0"), &cam));
    assert!(
        near > far,
        "a near lamp must out-light a far one (near {near:.2} vs far {far:.2})"
    );
}

/// The demo level's `[zone]`-stamped IMAGE wall must be a LIT surface, not a
/// self-lit decal: dropping its `relief =` line has to change the frame. This is
/// the end-to-end proof for the relief path — `[texture] height =` decoded,
/// `[material] relief =` parsed, the image material flipped off `self_lit`, and the
/// height field reaching the shader's normal perturbation + glyph ramp.
#[test]
fn the_demo_walls_relief_changes_the_shaded_frame() {
    let raw = cellar_source();
    assert!(
        raw.lines().any(|l| l.trim().starts_with("relief")),
        "the demo level authors a relief wall"
    );
    // Same level, same lamps, same camera — only the `relief =` line differs.
    let flat: String = raw
        .lines()
        .filter(|l| !l.trim().starts_with("relief"))
        .map(|l| format!("{l}\n"))
        .collect();

    // Standing in front of the +x end wall the zone stamps, looking at it.
    let cam = CameraPose::spawn_at(13, 3, 1, 120);
    let lit = cast(&raw, &cam);
    let dec = cast(&flat, &cam);

    let differing = lit
        .cells()
        .iter()
        .zip(dec.cells())
        .filter(|(a, b)| a.glyph != b.glyph || a.fg != b.fg || a.bg != b.bg)
        .count();
    assert!(
        differing > 0,
        "`relief =` must change the wall's shading — a self-lit decal and a lit \
         surface cannot render identically under two point lights"
    );
}
