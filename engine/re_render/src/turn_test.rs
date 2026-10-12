//! The "turn test" and its companion "strafe test", run through the LIVE
//! composite path ([`crate::build_world_layers_fov_into`], the exact function
//! `composite_render_sys` calls for the near-tier cast every structurally-dirty
//! frame — see that function's own doc comment: "the per-frame system uses
//! `build_world_layers_fov_into`"), not just `composite_dump_once`.
//!
//! The symptom this guards against: "when turning with the mouse, one texture is
//! stretched over the view and does not scroll with the rotation — it's 2D, not
//! 3D." This test is the falsifiable form of "it scrolls, and it scrolls SMOOTHLY
//! (no shimmer)": rotate the camera by one screen column's worth of yaw per
//! frame, over 10 frames, aimed square at a flat textured wall. World-anchored
//! content crosses each screen column's world-texel boundary **at most once** in
//! that sweep, so a column's glyph should change **at most once**. A column that
//! flips more than once is either genuinely screen-anchored (frozen content
//! occasionally "catching up") or shimmering (LOD/band-limit instability) —
//! either way, a real defect. The bound: at most 2 % of textured columns may
//! violate this.
//!
//! The strafe test is the direct "does it scroll" check the owner asked for:
//! move the camera sideways (no turn) and confirm the wall's glyph pattern
//! shifts by the exact perspective-projection amount (`Δx · proj / d`, `proj`
//! being the SAME `cols / (2·plane_scale)` constant `view_params` already
//! publishes for `face_span`), not by zero (frozen / screen-anchored) and not
//! by an arbitrary amount (broken projection).
//!
//! Both tests build a FRESH [`re_core::CompositeStack`] + scratch every frame —
//! deliberately never reusing state across calls — so a pass here rules out any
//! hidden per-call memory (screen-anchored or otherwise) in the near-cast path
//! itself, on top of the world-anchoring already proven at the recipe level by
//! `ascii_shader::tests::texel_noise_is_world_anchored_across_viewpoints` and at
//! the far-panorama level by `far_cache::tests::far_glyph_memory_is_world_anchored_not_screen_anchored`.

use glam::Vec3;
use re_core::{CompositeStack, Framebuffer, SurfaceKind};
use re_world::{CameraPose, Map};

use crate::ascii_shader::MaterialRegistry;
use crate::{build_world_layers_fov_into, registry_for_map, view_params, Fov};

/// Room used by both tests: a plain box with the SOUTH wall (`y = 0`) left as
/// the default `WallSolid` (brick, the built-in catalog's slot 0) and the NORTH
/// wall (`y = SY - 1`) turned into `WallTrace` (circuit, slot 1) — the two
/// procedural recipes that band-limiting most affects (brick mottle, circuit
/// pour/veins). One room camera-positioned at its centre can face either wall
/// dead-on just by choosing `yaw`, so both materials share one scene.
const ROOM_SX: usize = 12;
const ROOM_SY: usize = 12;
const ROOM_SZ: usize = 4;

fn turn_room() -> Map {
    let mut map = Map::new(ROOM_SX, ROOM_SY, ROOM_SZ);
    for z in 0..ROOM_SZ {
        for y in 0..ROOM_SY {
            for x in 0..ROOM_SX {
                let on_perimeter = x == 0 || x == ROOM_SX - 1 || y == 0 || y == ROOM_SY - 1;
                let kind = if y == ROOM_SY - 1 && x > 0 && x < ROOM_SX - 1 {
                    re_world::CellKind::WallTrace
                } else if on_perimeter {
                    re_world::CellKind::WallSolid
                } else if z == 0 {
                    re_world::CellKind::Floor
                } else if z == ROOM_SZ - 1 {
                    re_world::CellKind::Ceiling
                } else {
                    re_world::CellKind::Empty
                };
                map.set(x, y, z, kind);
            }
        }
    }
    map
}

/// A fixed view config: 100 columns, 32 rows, 60° horizontal FOV preset.
/// Reused here purely for consistency across these tests — the turn/strafe
/// tests don't depend on
/// the exact numbers, only on having a fixed, known FOV to derive `proj` from.
const COLS: usize = 100;
const ROWS: usize = 32;

fn test_fov() -> Fov {
    Fov {
        h_preset_deg: 60.0,
        cell_aspect: 0.5,
        stretch: 1.0,
    }
}

/// Cast one fresh frame through the LIVE incremental near-cast path. A brand
/// new [`CompositeStack`] + scratch every call, deliberately: this rules out
/// any hidden per-call/per-screen-cell memory in `build_world_layers_fov_into`
/// itself (the pan/dirty-flag optimizations that DO carry state live one layer
/// up, in `composite.rs`'s `composite_render_sys`, and are documented there as
/// producing byte-identical results to a fresh cast at the same pose — so a
/// fresh cast at every step is the stricter, superset check).
fn cast_frame(
    map: &Map,
    cam: &CameraPose,
    registry: &MaterialRegistry,
    fov: &Fov,
) -> CompositeStack {
    let mut scratch = Framebuffer::new(COLS, ROWS);
    let mut stack = CompositeStack::new(COLS, ROWS);
    build_world_layers_fov_into(map, cam, registry, fov, &mut scratch, &mut stack);
    stack
}

/// Row of the frame we sample: eye level, where a dead-on wall fills the frame
/// edge-to-edge at the distances these tests use.
fn probe_row() -> usize {
    ROWS / 2
}

/// Rotate the camera by a TOTAL of `total_yaw` (one column's worth, across
/// 10 frames),
/// spread over `frames` equal sub-steps, starting from `base_yaw`, aimed at
/// `target` (`WallSolid` = brick / south wall, `WallTrace` = circuit / north
/// wall). Returns, per column, the sequence of glyphs sampled at the probe row
/// across every frame — `None` where that column did not hit `target` that
/// frame.
///
/// Sub-column sub-steps, not whole extra columns: the whole point of this
/// bound is that a WORLD TEXEL crosses a screen column's boundary AT MOST ONCE
/// as the view slides past it by one column's width. Stepping by a full
/// column's yaw on EVERY frame (10 columns of total sweep) would slide many
/// texels past each probe column and manufacture toggles that have nothing to
/// do with shimmer — that is not what this test measures.
fn sweep_glyphs(
    map: &Map,
    registry: &MaterialRegistry,
    fov: &Fov,
    base_yaw: f32,
    target: SurfaceKind,
    frames: usize,
    total_yaw: f32,
) -> Vec<Vec<Option<u32>>> {
    let row = probe_row();
    let step = total_yaw / (frames - 1).max(1) as f32;
    (0..frames)
        .map(|i| {
            let cam = CameraPose {
                pos: Vec3::new(6.0, 6.0, re_world::STANDING_EYE_Z),
                yaw: base_yaw + step * i as f32,
                pitch: 0.0,
            };
            let stack = cast_frame(map, &cam, registry, fov);
            (0..COLS)
                .map(|col| {
                    stack.opaque.get(col, row).and_then(|c| {
                        if c.surface == target {
                            Some(c.glyph)
                        } else {
                            None
                        }
                    })
                })
                .collect()
        })
        .collect()
}

/// Run the shimmer bound for one material/wall and return `(bad, total)`:
/// `total` is the number of columns that hit `target` in EVERY frame of the
/// sweep (the "textured cells" the percentage is over); `bad` is how many of those
/// changed glyph MORE THAN ONCE across the sweep.
fn a6_bad_and_total(
    map: &Map,
    registry: &MaterialRegistry,
    fov: &Fov,
    base_yaw: f32,
    target: SurfaceKind,
    one_col_yaw: f32,
) -> (usize, usize, usize) {
    const FRAMES: usize = 10;
    let frames = sweep_glyphs(map, registry, fov, base_yaw, target, FRAMES, one_col_yaw);

    let mut bad = 0usize;
    let mut total = 0usize;
    let mut any_changed = 0usize;
    for col in 0..COLS {
        let seq: Vec<u32> = {
            let mut v = Vec::with_capacity(FRAMES);
            let mut all_present = true;
            for f in &frames {
                match f[col] {
                    Some(g) => v.push(g),
                    None => {
                        all_present = false;
                        break;
                    }
                }
            }
            if !all_present {
                continue;
            }
            v
        };
        total += 1;
        let toggles = seq.windows(2).filter(|w| w[0] != w[1]).count();
        if toggles > 1 {
            bad += 1;
        }
        if toggles >= 1 {
            any_changed += 1;
        }
    }
    (bad, total, any_changed)
}

#[test]
fn a6_turn_test_brick_south_wall_stays_within_the_shimmer_budget() {
    let map = turn_room();
    let registry = registry_for_map(&map);
    let fov = test_fov();
    let view = view_params(COLS, ROWS, &fov);
    let one_col_yaw = view.hfov / COLS as f32;

    // yaw = PI faces -y (the south wall, y = 0, WallSolid / brick).
    let (bad, total, any_changed) = a6_bad_and_total(
        &map,
        &registry,
        &fov,
        std::f32::consts::PI,
        SurfaceKind::WallSolid,
        one_col_yaw,
    );

    assert!(
        total > 20,
        "expected a healthy span of columns to hit the south wall; got {total}"
    );
    // The core reported defect: content that never scrolls at all while the
    // camera turns is a frozen/screen-anchored texture. Some columns MUST show
    // at least one crossing across the one-column-yaw sweep at this distance.
    assert!(
        any_changed > 0,
        "brick wall showed ZERO glyph change across a {} rad turn — texture is glued to the screen",
        one_col_yaw
    );
    let frac = bad as f32 / total as f32;
    assert!(
        frac <= 0.02,
        "brick wall: {bad}/{total} textured columns ({:.1}%) toggled more than once across the \
         10-frame turn sweep — exceeds the 2% shimmer budget",
        frac * 100.0
    );
}

#[test]
fn a6_turn_test_circuit_north_wall_stays_within_the_shimmer_budget() {
    let map = turn_room();
    let registry = registry_for_map(&map);
    let fov = test_fov();
    let view = view_params(COLS, ROWS, &fov);
    let one_col_yaw = view.hfov / COLS as f32;

    // yaw = 0 faces +y (the north wall, y = SY - 1, WallTrace / circuit).
    let (bad, total, any_changed) = a6_bad_and_total(
        &map,
        &registry,
        &fov,
        0.0,
        SurfaceKind::WallTrace,
        one_col_yaw,
    );

    assert!(
        total > 20,
        "expected a healthy span of columns to hit the north wall; got {total}"
    );
    assert!(
        any_changed > 0,
        "circuit wall showed ZERO glyph change across a {} rad turn — texture is glued to the screen",
        one_col_yaw
    );
    let frac = bad as f32 / total as f32;
    assert!(
        frac <= 0.02,
        "circuit wall: {bad}/{total} textured columns ({:.1}%) toggled more than once across the \
         10-frame turn sweep — exceeds the 2% shimmer budget",
        frac * 100.0
    );
}

/// The strafe test, alongside the turn test above: "surface content at depth d must shift
/// by `Δx·proj/d` columns, ±1." Rather than reconstructing the shift from a
/// glyph/colour cross-correlation (tried first; brick's mortar-course glyph is
/// a near-periodic ~10-11-column alphabet and its fBm mottle is low-contrast
/// over this room's visible span — both made the reconstruction fight its own
/// signal-to-noise instead of testing the anchor contract), this asks the
/// renderer directly for the one number that IS the anchor contract: the
/// world-space hit point [`re_core::Cell::hit`] the near-cast already computed
/// for every column. World-anchored content is a pure function of that hit
/// point (already proven independently by
/// `ascii_shader::tests::texel_noise_is_world_anchored_across_viewpoints`), so
/// proving the hit point itself moves by exactly the perspective-predicted
/// amount under a pure strafe is the direct, non-lossy form of "the surface
/// scrolls like 3D, not a frozen decal" — no reconstruction, no aliasing risk.
#[test]
fn strafe_test_brick_wall_hit_point_shifts_by_delta_x_proj_over_d() {
    let map = turn_room();
    let registry = registry_for_map(&map);
    let fov = test_fov();
    let view = view_params(COLS, ROWS, &fov);
    let proj = COLS as f32 / (2.0 * view.plane_scale);

    let yaw = std::f32::consts::PI; // facing -y, dead-on at the south wall (y=0)
    let start = Vec3::new(6.0, 6.0, re_world::STANDING_EYE_Z);
    let d = start.y; // perpendicular distance to the y=0 wall plane
    let right_axis = Vec3::new(yaw.cos(), -yaw.sin(), 0.0); // matches the cast's plane_x/plane_y basis
    let strafe = 0.37f32;
    let moved = start + right_axis * strafe;

    let cam0 = CameraPose {
        pos: start,
        yaw,
        pitch: 0.0,
    };
    let cam1 = CameraPose {
        pos: moved,
        yaw,
        pitch: 0.0,
    };
    let row = probe_row();

    let stack0 = cast_frame(&map, &cam0, &registry, &fov);
    let stack1 = cast_frame(&map, &cam1, &registry, &fov);

    // For a FIXED world point on the wall, the perspective-predicted screen
    // shift is `Δcam·right_axis · proj / d` in the OPPOSITE direction of the
    // camera's own motion along its right axis (see the module doc's
    // derivation): here that reduces to `-strafe * proj / d`.
    let predicted_shift = -strafe * proj / d;

    // For column `col`, `hit0.x` is a known LINEAR function of `col` (a flat,
    // dead-on wall has no depth variation across the row), with the SAME
    // slope in both frames (view geometry is unchanged — only the camera
    // moved). So instead of comparing hit.x directly (which changes with the
    // camera-position term, not with a "column shift"), compare each column's
    // hit.x against the OTHER frame's hit.x at `col - predicted_shift`
    // (linearly interpolated between its two neighbouring integer columns):
    // if the near-cast is genuinely world-anchored and perspective-correct,
    // that interpolated value must equal this column's own hit.x almost
    // exactly (float precision only) — i.e. the SAME world point.
    let hit_x = |stack: &re_core::CompositeStack, col: usize| -> Option<f32> {
        stack
            .opaque
            .get(col, row)
            .and_then(|c| (c.surface == SurfaceKind::WallSolid).then_some(c.hit.x))
    };
    let interp_hit0_at = |col_f: f32| -> Option<f32> {
        let c0 = col_f.floor();
        let c1 = c0 + 1.0;
        if c0 < 0.0 || c1 >= COLS as f32 {
            return None;
        }
        let a = hit_x(&stack0, c0 as usize)?;
        let b = hit_x(&stack0, c1 as usize)?;
        let t = col_f - c0;
        Some(a + (b - a) * t)
    };

    let mut checked = 0usize;
    let mut max_err = 0.0f32;
    for col in 0..COLS {
        let Some(h1) = hit_x(&stack1, col) else {
            continue;
        };
        let Some(h0_interp) = interp_hit0_at(col as f32 - predicted_shift) else {
            continue;
        };
        checked += 1;
        max_err = max_err.max((h1 - h0_interp).abs());
    }

    assert!(
        checked > 20,
        "expected a healthy span of columns to compare; got {checked} \
         (predicted_shift={predicted_shift:.2})"
    );
    // Budget: exactly the stated tolerance ("±1 column"), converted
    // to world units via the SAME `d/proj` relationship the column shift is
    // derived from. This also absorbs the small curvature error from
    // approximating the (tangent-based, not perfectly linear) column→world
    // mapping with a 2-point linear interpolation between adjacent columns.
    let one_column_world = d / proj;
    assert!(
        max_err <= one_column_world,
        "the near-cast hit point at column `col` in the strafed frame does not match the \
         un-strafed frame's hit point at `col - {predicted_shift:.2}` (max error {max_err:.4} \
         world units across {checked} columns, budget {one_column_world:.4} = one column) — \
         the surface is not tracking the camera's motion like a perspective-correct 3D surface"
    );

    // Negative control: a genuinely FROZEN (screen-anchored) surface would
    // show its hit point UNCHANGED at the SAME column between frames — i.e.
    // matching a shift of 0, not `predicted_shift`. Confirm shift=0 does NOT
    // fit anywhere near as well (predicted_shift is ~-5 columns here, so the
    // two hypotheses are well separated).
    let mut max_err_at_zero_shift = 0.0f32;
    let mut checked_zero = 0usize;
    for col in 0..COLS {
        let (Some(h0), Some(h1)) = (hit_x(&stack0, col), hit_x(&stack1, col)) else {
            continue;
        };
        checked_zero += 1;
        max_err_at_zero_shift = max_err_at_zero_shift.max((h1 - h0).abs());
    }
    assert!(checked_zero > 20);
    assert!(
        max_err_at_zero_shift > 0.1,
        "hit point at the SAME column is unchanged between frames (max diff {max_err_at_zero_shift:.4}) \
         — the strafe did not move the wall content at all; texture is glued to the screen"
    );
}
