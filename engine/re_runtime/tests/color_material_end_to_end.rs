//! The level under test is authored INLINE (no `.lvl` file, campaign or
//! otherwise), and the pipeline it proves — `re_level::load_string` →
//! `build_map` → `re_render::registry_for_map` → `raycast` dispatching a
//! painted cell to its own colour-material — is entirely `re_level` +
//! `re_render` + `re_world`, all regular dependencies of `re_runtime` already.
//!
//! END-TO-END: the colour→material substrate from a `.lvl` all the way to
//! shaded pixels. A level (the exact shape `re_voximport`'s
//! `vox_to_level_full` → `to_lvl_text_with_ids` emits: `[material cm_*]` sections +
//! a `[color_cell_ids]` block) is parsed by `re_level`, `build_map` stamps the
//! 1-based colour ids into the target geometry cells' `side_data` and appends the
//! colour palette to the map's material set, and `re_render::raycast` (which builds
//! the map's registry via `registry_for_map`) dispatches each painted cell to its
//! own colour-material — so two same-surface-kind (`WallSolid`) walls painted with
//! DIFFERENT colour ids render as visibly DIFFERENT colours.

use re_core::Framebuffer;
use re_level::load_string;
use re_world::CameraPose;

/// Build the level once: a square room whose -X wall is painted colour id 1 (a
/// RED brick) and whose +X wall is painted colour id 2 (a GREEN grain). Both are
/// ordinary `WallSolid` perimeter cells — only their `side_data` differs.
const LEVEL: &str = "\
[meta]
name = cm_e2e
[room]
size = 9 9 4
ambient = solid
[material cm_FF4040]
fg = #FF4040
bg = #200000
recipe = brick
[material cm_40FF40]
fg = #40FF40
bg = #002000
recipe = grain
[node a]
pos = 1 1 2
is_input = yes
[node b]
pos = 7 1 2
is_output = yes
[pipe a->b]
path = (1,1,2) (7,1,2)
[pedestal]
center = 7 1 1
volume = 1 1 1
[door]
pos = 8 7 1
[player]
spawn = 4 4 2
[color_cell_ids]
0,2,2 = 1
0,3,2 = 1
0,4,2 = 1
0,5,2 = 1
0,6,2 = 1
8,2,2 = 2
8,3,2 = 2
8,4,2 = 2
8,5,2 = 2
8,6,2 = 2
";

/// Average (R, G) of the central wall band of a cast, ignoring near-black
/// background cells. Sampling the middle rows keeps the eye-level wall (where the
/// painted colour cells sit) and avoids the floor/ceiling bands top and bottom.
fn avg_rg(fb: &Framebuffer) -> (f64, f64) {
    let cols = fb.cols();
    let rows = fb.rows();
    let cells = fb.cells();
    let (r0, r1) = (rows / 3, rows * 2 / 3); // central third (the wall band)
    let mut rs = 0.0f64;
    let mut gs = 0.0f64;
    let mut n = 0.0f64;
    for row in r0..r1 {
        for col in 0..cols {
            let cell = &cells[row * cols + col];
            let r = ((cell.fg >> 16) & 0xFF) as f64;
            let g = ((cell.fg >> 8) & 0xFF) as f64;
            // Skip dim/empty cells so the wall colour dominates the average.
            if r + g < 60.0 {
                continue;
            }
            rs += r;
            gs += g;
            n += 1.0;
        }
    }
    if n == 0.0 {
        (0.0, 0.0)
    } else {
        (rs / n, gs / n)
    }
}

#[test]
fn two_colour_walls_render_as_two_materials() {
    let lv = load_string(LEVEL, "<cm_e2e>").expect("level must parse");
    // The data side reached the map.
    let map = lv.build_map();
    assert_eq!(
        map.side_data_at(0, 4, 2),
        1,
        "the -X wall carries colour id 1"
    );
    assert_eq!(
        map.side_data_at(8, 4, 2),
        2,
        "the +X wall carries colour id 2"
    );
    let mats = map.materials();
    assert_eq!(mats.len(), 7, "5 surface slots + 2 colour palette entries");

    // The registry the renderer builds carries the colour palette (id→slot).
    let reg = re_render::registry_for_map(&map);
    assert_eq!(
        reg.color_material_len(),
        2,
        "two colour-palette entries registered"
    );
    assert!(reg.color_material(1).is_some(), "colour id 1 resolves");
    assert!(reg.color_material(2).is_some(), "colour id 2 resolves");
    assert_ne!(
        reg.color_material(1),
        reg.color_material(2),
        "the two colour ids resolve to DIFFERENT registry slots"
    );

    // Cast facing the RED wall (-X ⇒ facing col 360) and the GREEN wall (+X ⇒ 120).
    // (480 cols == 2π; yaw=0 faces +y, so 360 ⇒ dir(-1,0), 120 ⇒ dir(+1,0).)
    // Stand right next to each wall so it fills the view (minimise floor/ceiling/
    // far-wall dilution in the average).
    let cam_red = CameraPose::spawn_at(1, 4, 2, 360);
    let cam_green = CameraPose::spawn_at(7, 4, 2, 120);

    let mut fb_red = Framebuffer::new(100, 32);
    let mut fb_green = Framebuffer::new(100, 32);
    re_render::raycast(&map, &cam_red, &mut fb_red);
    re_render::raycast(&map, &cam_green, &mut fb_green);

    let (rr, rg) = avg_rg(&fb_red);
    let (gr, gg) = avg_rg(&fb_green);

    // Facing the red wall, the lit wall band leans RED (R > G); facing the green
    // wall it leans GREEN (G > R). This can only happen if the two same-kind walls
    // dispatched to their distinct colour-materials.
    assert!(
        rr > rg,
        "facing the colour-id-1 wall must read RED-dominant (R={rr:.1} G={rg:.1})"
    );
    assert!(
        gg > gr,
        "facing the colour-id-2 wall must read GREEN-dominant (R={gr:.1} G={gg:.1})"
    );
    // And the two views are clearly different hues.
    assert!(
        rr > gr && gg > rg,
        "the two walls render as two different colours \
         (red-view R={rr:.1} G={rg:.1}; green-view R={gr:.1} G={gg:.1})"
    );
}
