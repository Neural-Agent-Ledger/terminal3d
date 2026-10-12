//! Both the fixture level (`levels/91-recipe-kinds.lvl`) and the feature under
//! test — `re_level`'s `recipe = KIND` parser matching `re_render`'s catalog
//! `preset = NAME` — are engine-only. `re_runtime` already depends on
//! `re_level` and `re_render` directly, so the comparison needs no new
//! dependency.
//!
//! `recipe = KIND` authored in a `.lvl` must draw the SAME pattern as the
//! catalog `preset = NAME` it mirrors (see `docs/MATERIALS.md` §"Authoring a
//! recipe directly"). This is the one place that sees both halves —
//! `re_level`'s parser and `re_render`'s converter + catalog.
//!
//! `TextureRecipe` deliberately carries no `PartialEq` (it holds floats and is data
//! for the shader, not a key), so the comparison is on its `Debug` form, which
//! prints every knob.

use re_level::parse_recipe;
use re_render::ascii_shader::{ids, MaterialRegistry};
use re_render::recipe_from_def;

fn preset_recipe_debug(id: re_render::ascii_shader::MaterialId) -> String {
    let reg = MaterialRegistry::with_catalog();
    format!("{:?}", reg.get(id).recipe)
}

fn authored_recipe_debug(line: &str) -> String {
    let def = parse_recipe(line).unwrap_or_else(|| panic!("`{line}` must parse"));
    format!("{:?}", recipe_from_def(def))
}

#[test]
fn bare_recipe_kinds_equal_their_catalog_presets() {
    // (`recipe =` line with NO knobs, the preset whose pattern it must reproduce)
    let pairs = [
        ("scanline", ids::MONITOR),
        ("truchet", ids::CONDUIT),
        ("fractal", ids::PORTAL_ZOOM),
        ("substrate", ids::SUBSTRATE),
        ("voice", ids::MNEMOS_HOLD),
        ("voice mode=release", ids::LETHE_RELEASE),
        ("facade", ids::FACADE),
    ];
    for (line, id) in pairs {
        assert_eq!(
            authored_recipe_debug(line),
            preset_recipe_debug(id),
            "`recipe = {line}` must convert to the pattern of its preset ({id:?})"
        );
    }
}

#[test]
fn every_knob_reaches_the_renderer() {
    // A non-default value on every knob must be visible in the converted recipe —
    // a knob the converter silently dropped would print its default instead.
    let cases = [
        (
            "scanline freq=11 duty=0.4 warp=0.2 axis=2",
            &["11", "0.4", "0.2", "2"][..],
        ),
        (
            "truchet seed=-17 density=0.75 scale=6",
            &["-17", "0.75", "6"][..],
        ),
        (
            "fractal arms=5 octave_warp=0.1 twist=0.9",
            &["5", "0.1", "0.9"][..],
        ),
        (
            "substrate scale=2.5 held=0.7 released=0.2",
            &["2.5", "0.7", "0.2"][..],
        ),
        ("voice scale=4 mode=release", &["4", "Release"][..]),
        (
            "facade style=server_rack bay=6 floor=10 lit=0.6 seed=-17 ground_band=on sign_row=on",
            &[
                "ServerRack",
                "6.0",
                "10.0",
                "0.6",
                "-17",
                "ground_band: true",
                "sign_row: true",
            ][..],
        ),
    ];
    for (line, needles) in cases {
        let dbg = authored_recipe_debug(line);
        for n in needles {
            assert!(
                dbg.contains(n),
                "`{line}` → `{dbg}` must carry knob value {n}"
            );
        }
    }
}

#[test]
fn the_fixture_level_renders_headless() {
    // The five-recipe fixture must load through the real level path and cast a
    // frame; a recipe that reached the renderer in a broken state would panic here.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../levels");
    let path = root.join("91-recipe-kinds.lvl");
    let raw = std::fs::read_to_string(&path).expect("fixture exists");
    let level = re_level::load_string(&raw, "91").expect("fixture parses");
    let kinds: std::collections::BTreeSet<String> = level
        .materials
        .iter()
        .filter_map(|m| m.recipe.as_ref())
        .map(|r| {
            format!("{r:?}")
                .split(' ')
                .next()
                .unwrap()
                .trim_end_matches('{')
                .to_string()
        })
        .collect();
    for k in ["Scanline", "Truchet", "Fractal", "Substrate", "Voice"] {
        assert!(kinds.contains(k), "fixture must bind {k}; got {kinds:?}");
    }
}
