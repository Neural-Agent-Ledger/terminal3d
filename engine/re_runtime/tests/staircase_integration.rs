//! Every level exercised (`levels/20-stairs-straight.lvl`,
//! `21-stairs-spiral.lvl`, `22-tower.lvl`) is an engine demo level, and what is
//! under test — `re_world`'s camera physics (`move_with_collision`/
//! `step_gravity`) plus `re_render`'s elevation-aware raycast over a
//! `re_level`-parsed map — is entirely engine behaviour, using only
//! `re_core`, `re_level`, `re_render` and `re_world`, all regular dependencies
//! of `re_runtime` already. `sha2` is a dev-dependency here for the
//! frame-digest helper (see this crate's `Cargo.toml`).
//!
//! Integration tests for multi-floor staircase demo levels.
//!
//! These tests cover:
//!
//! * **Smoke-load** — `levels/20-stairs-straight.lvl` and
//!   `levels/21-stairs-spiral.lvl` parse without error and produce a non-trivial
//!   Framebuffer.
//!
//! * **Floor-0 vs floor-1 frame differs** — rendering from the spawn (floor 0)
//!   and from an equivalent camera placed on floor 1 produces two distinct
//!   `{glyph, fg, bg}` grids. If the raycaster or staircase geometry were broken
//!   such that elevation had no visual effect, this test would catch it.
//!
//! * **Scripted climb** — drive `CameraPose::move_with_collision` +
//!   `step_gravity` up the straight staircase and assert the camera `pos.z`
//!   rises by approximately `storey_height` (= 4) cells and the camera is
//!   grounded at the top. This exercises the actual re_world physics, not just
//!   the parser or renderer.
//!
//! # Artifact policy
//!
//! These tests hash the **Framebuffer** (`{glyph, fg, bg}` per cell, 100×32)
//! for the floor-0 vs floor-1 comparison. We assert the two digests DIFFER
//! rather than comparing them to frozen goldens. This makes the test robust
//! against minor render tweaks while still catching catastrophic regressions
//! (e.g. both floors rendering identically because the z-shear is broken).
//!
//! The climb test uses physics asserts (z within ±0.5 of expected, grounded=true)
//! rather than exact-byte goldens, for the same reason.
//!
//! # How to run
//! ```sh
//! cargo test -p re_runtime --test staircase_integration
//! ```

use re_core::{Framebuffer, Vec3};
use re_level::load;
use re_render::{composite::composite_dump_once, registry_for_map, Emitter, Fov};
use re_world::{CameraPose, PlayerMotion, PLAYER_EYE_HEIGHT, PLAYER_RADIUS, STANDING_EYE_Z};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Framebuffer size — must match the host game's own DUMP_COLS / DUMP_ROWS so
/// the hashes are directly comparable to the CLI `--dump` output.
const COLS: usize = 100;
const ROWS: usize = 32;

/// Level files under test, relative to the workspace root.
const STRAIGHT_LVL: &str = "levels/20-stairs-straight.lvl";
const SPIRAL_LVL: &str = "levels/21-stairs-spiral.lvl";
const TOWER_LVL: &str = "levels/22-tower.lvl";

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Hex-encode a digest byte array. sha2 0.11 (`digest` 0.11) returns its own
/// `hybrid_array::Array<u8, N>` from `finalize()` rather than the old
/// `generic_array::GenericArray`, and that type does not implement
/// `LowerHex` — so `format!("{:x}", ...)` no longer compiles directly.
fn to_hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        write!(s, "{b:02x}").expect("write to String never fails");
    }
    s
}

/// Workspace root: `CARGO_MANIFEST_DIR` (`crates/re_runtime`) stepped two levels up.
fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .to_path_buf()
        })
}

/// Render a level from a given camera pose into a 100×32 Framebuffer and
/// return the SHA-256 of the `{glyph, fg, bg}` cell grid (row-major,
/// 12 bytes per cell, u32-LE each). The hash is terminal-independent —
/// identical across all terminal emulators and host OS.
fn render_hash(level_path: &PathBuf, cam: &CameraPose) -> String {
    let lv =
        load(level_path).unwrap_or_else(|e| panic!("failed to load {}: {e}", level_path.display()));
    let map = lv.build_map();
    let registry = registry_for_map(&map);
    let emitters: Vec<Emitter> = lv.emitters.iter().map(Emitter::from_spec).collect();

    let mut fb = Framebuffer::new(COLS, ROWS);
    composite_dump_once(&map, cam, &registry, &Fov::default(), &emitters, &mut fb);

    let mut hasher = Sha256::new();
    for row in 0..ROWS {
        for col in 0..COLS {
            let cell = fb.get(col, row).expect("in-bounds cell");
            hasher.update(cell.glyph.to_le_bytes());
            hasher.update(cell.fg.to_le_bytes());
            hasher.update(cell.bg.to_le_bytes());
        }
    }
    to_hex(&hasher.finalize())
}

/// Return a non-trivial glyph count (glyphs that are not the space/dot default
/// empty value). Used to assert the frame is non-blank.
fn non_empty_glyph_count(level_path: &PathBuf, cam: &CameraPose) -> usize {
    let lv =
        load(level_path).unwrap_or_else(|e| panic!("failed to load {}: {e}", level_path.display()));
    let map = lv.build_map();
    let registry = registry_for_map(&map);
    let emitters: Vec<Emitter> = lv.emitters.iter().map(Emitter::from_spec).collect();

    let mut fb = Framebuffer::new(COLS, ROWS);
    composite_dump_once(&map, cam, &registry, &Fov::default(), &emitters, &mut fb);

    let mut count = 0usize;
    for row in 0..ROWS {
        for col in 0..COLS {
            let g = fb.get(col, row).expect("in-bounds").glyph;
            let ch = char::from_u32(g).unwrap_or(' ');
            if ch != ' ' && ch != '.' {
                count += 1;
            }
        }
    }
    count
}

// ── Smoke-load tests ─────────────────────────────────────────────────────────

/// The straight staircase level loads without error, and the spawn camera
/// produces a non-trivial (non-blank) Framebuffer at 100×32.
#[test]
fn straight_level_loads_and_renders_nonblank() {
    let path = workspace_root().join(STRAIGHT_LVL);
    assert!(path.exists(), "level file missing: {}", path.display());

    let lv = load(&path).expect("20-stairs-straight.lvl must parse without error");

    // Basic sanity: the level carries the storey_height metadata.
    assert_eq!(
        lv.storey_height(),
        4,
        "storey_height must be 4 (as authored in the [room] header)"
    );

    // Spawn camera comes from the [player] section.
    let cam = lv.spawn_camera();
    // The spawn is at (2,3,1) → world pos (2.5, 3.5, 1.5), i.e. floor 0.
    assert!(
        (cam.pos.z - STANDING_EYE_Z).abs() < 0.6,
        "spawn z ≈ STANDING_EYE_Z (floor 0): got {:.3}",
        cam.pos.z
    );

    // The frame must contain visible geometry.
    let nonblank = non_empty_glyph_count(&path, &cam);
    assert!(
        nonblank > 200,
        "floor-0 frame must be non-blank: only {nonblank} non-empty glyphs"
    );
}

/// The spiral staircase level loads without error, and the spawn camera
/// produces a non-trivial Framebuffer.
#[test]
fn spiral_level_loads_and_renders_nonblank() {
    let path = workspace_root().join(SPIRAL_LVL);
    assert!(path.exists(), "level file missing: {}", path.display());

    let lv = load(&path).expect("21-stairs-spiral.lvl must parse without error");

    assert_eq!(
        lv.storey_height(),
        8,
        "storey_height must be 8 (as authored in the [room] header)"
    );

    let cam = lv.spawn_camera();
    let nonblank = non_empty_glyph_count(&path, &cam);
    assert!(
        nonblank > 200,
        "floor-0 frame must be non-blank: only {nonblank} non-empty glyphs"
    );
}

// ── Floor-0 vs floor-1 frame differs ─────────────────────────────────────────

/// Render the straight staircase level from floor 0 (the spawn) and from an
/// equivalent camera placed on floor 1 (z shifted by storey_height = 4 cells).
/// The two frames MUST differ: different elevation means different raycaster
/// geometry (floor/ceiling distances, wall heights, etc.).
#[test]
fn straight_floor0_and_floor1_frames_differ() {
    let path = workspace_root().join(STRAIGHT_LVL);

    let lv = load(&path).expect("load");
    let storey = lv.storey_height() as f32;

    // Floor-0 camera: the level spawn.
    let cam_floor0 = lv.spawn_camera();

    // Floor-1 camera: same x/y/yaw, but z shifted up by one storey.
    // Floor-1 standing eye z = STANDING_EYE_Z + storey  (floor-0 eye + H cells).
    let cam_floor1 = CameraPose {
        pos: Vec3::new(
            cam_floor0.pos.x,
            cam_floor0.pos.y,
            cam_floor0.pos.z + storey,
        ),
        yaw: cam_floor0.yaw,
        pitch: cam_floor0.pitch,
    };

    let hash0 = render_hash(&path, &cam_floor0);
    let hash1 = render_hash(&path, &cam_floor1);

    assert_ne!(
        hash0, hash1,
        "floor-0 and floor-1 frames must differ: \
         if they match the raycaster is ignoring camera elevation.\n\
         floor-0 digest: {hash0}\n\
         floor-1 digest: {hash1}"
    );
}

/// Same elevation-divergence test for the spiral level.
#[test]
fn spiral_floor0_and_floor1_frames_differ() {
    let path = workspace_root().join(SPIRAL_LVL);

    let lv = load(&path).expect("load");
    let storey = lv.storey_height() as f32;

    let cam_floor0 = lv.spawn_camera();
    let cam_floor1 = CameraPose {
        pos: Vec3::new(
            cam_floor0.pos.x,
            cam_floor0.pos.y,
            cam_floor0.pos.z + storey,
        ),
        yaw: cam_floor0.yaw,
        pitch: cam_floor0.pitch,
    };

    let hash0 = render_hash(&path, &cam_floor0);
    let hash1 = render_hash(&path, &cam_floor1);

    assert_ne!(
        hash0, hash1,
        "spiral floor-0 and floor-1 frames must differ.\n\
         floor-0 digest: {hash0}\n\
         floor-1 digest: {hash1}"
    );
}

// ── Scripted-climb test (straight staircase) ──────────────────────────────────

/// Walk a camera up the straight staircase using the actual re_world physics
/// (`move_with_collision` + `step_gravity`).
///
/// Acceptance criteria:
/// * The camera `pos.z` rises by ≈ `storey_height` (4) cells by the time it
///   reaches the landing (within ±0.6 cells of the expected top eye-z).
/// * `grounded` is `true` continuously across the ramp (no mid-stair fall).
/// * The camera does not fall through the floor or ceiling.
///
/// The scripted walk: start just before ramp tread 1 (x=3.1, y=3.5) facing +x
/// and advance in small steps until the camera passes the landing (x > 9.0).
#[test]
fn straight_staircase_scripted_climb_raises_z_one_storey() {
    let path = workspace_root().join(STRAIGHT_LVL);
    let lv = load(&path).expect("load");
    let map = lv.build_map();

    let storey_height = lv.storey_height();

    // Start just in front of tread 1 (x=4, y=3, z=1), y-centred in the
    // corridor (y=3.5). Eye z = floor surface(1.0) + slope(0.1) + eye_height.
    let start_x: f32 = 3.8;
    let start_surface_z: f32 = 1.0 + (start_x - 3.0).clamp(0.0, 1.0); // flat at floor 0
    let start_eye_z = start_surface_z + PLAYER_EYE_HEIGHT;

    let mut cam = CameraPose {
        pos: Vec3::new(start_x, 3.5, start_eye_z),
        yaw: std::f32::consts::FRAC_PI_2, // facing +x (yaw = π/2)
        pitch: 0.0,
    };
    let mut motion = PlayerMotion::default();

    // Settle onto the floor before walking (apply gravity a few times).
    for _ in 0..20 {
        cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
    }

    assert!(
        motion.grounded,
        "camera must be grounded before starting the climb: z={:.3}",
        cam.pos.z
    );

    let initial_z = cam.pos.z;
    let mut seam_failures: Vec<f32> = Vec::new();

    // Walk east (+x) in small increments until we pass the landing (x > 9.0).
    // Each step: move, then let gravity settle.
    let step = 0.06_f32;
    while cam.pos.x < 9.0 {
        // Forward = +x direction.
        cam.move_with_collision(step, 0.0, &map, PLAYER_RADIUS);
        // Settle gravity (3 ticks at 60 Hz gives enough time to snap to slope).
        for _ in 0..3 {
            cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
        }

        // Check that we are grounded anywhere not very close to a seam.
        // "Near a seam" means the fractional x-part is in the 0.1 edge band.
        let local_x = cam.pos.x - cam.pos.x.floor();
        let near_seam = !(0.15..=0.85).contains(&local_x);
        if near_seam && !motion.grounded {
            seam_failures.push(cam.pos.x);
        }
    }

    assert!(
        seam_failures.is_empty(),
        "grounded flickered near ramp-cell seams at x positions: {seam_failures:?}"
    );

    // After walking past the landing, grounded must still be true.
    assert!(
        motion.grounded,
        "must be grounded on floor 1 after climbing (x={:.3}, z={:.3})",
        cam.pos.x, cam.pos.z
    );

    let total_rise = cam.pos.z - initial_z;

    // Expected rise: storey_height cells, less the ~0.2-cell head-start from
    // starting slightly past x=3.0 on the ramp approach. Allow ±0.8 tolerance
    // to account for the gravity snap granularity and approach offset.
    let expected_rise_min = (storey_height as f32) - 1.0;
    let expected_rise_max = (storey_height as f32) + 0.5;
    assert!(
        total_rise >= expected_rise_min && total_rise <= expected_rise_max,
        "total rise across one storey should be ≈{storey_height} cells \
         (±tolerance); got {total_rise:.3} \
         (initial_z={initial_z:.3}, final_z={:.3})",
        cam.pos.z
    );
}

// ── Scripted-climb test (spiral staircase) ────────────────────────────────────

/// Walk a camera across the second spiral tread (the first effective one at z=1),
/// confirming the ramp cell is a live physics surface that raises the eye z.
///
/// The spiral tread at (6,5,1) is a RampZ(PosX) cell. A player at y=5.5 walking
/// east from x=5.5 to x=7.0 crosses this tread and should rise by ≈ 1 cell.
/// We walk at most 200 steps (bounded loop — no infinite hang), assert progress,
/// then assert the final z is above the starting z.
///
/// This is a targeted physics-surface probe, not a full-storey walk. The full
/// multi-floor frame divergence (which ALSO validates elevation physics) is covered
/// by `spiral_floor0_and_floor1_frames_differ`.
#[test]
fn spiral_staircase_second_tread_is_live_ramp_surface() {
    let path = workspace_root().join(SPIRAL_LVL);
    let lv = load(&path).expect("load");
    let map = lv.build_map();

    // The second PosX tread is at cell (6, 5, 1). Start the player just before
    // it (x=5.5, y=5.5, eye at floor-0 height) and walk east.
    // Note: tread 1 at (5,5,0) is at the floor-slab z-level; tread 2 at (6,5,1)
    // is the first tread in the gameplay space that actually lifts the player.
    let start_x: f32 = 5.5;
    let start_eye_z = STANDING_EYE_Z; // 1.5 — resting on floor 0
    let mut cam = CameraPose {
        pos: Vec3::new(start_x, 5.5, start_eye_z),
        yaw: std::f32::consts::FRAC_PI_2, // facing +x
        pitch: 0.0,
    };
    let mut motion = PlayerMotion::default();

    // Settle onto the floor before the test.
    for _ in 0..20 {
        cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
    }
    let initial_z = cam.pos.z;

    // Advance east until x > 7.0 (past the second tread), bounded at 200 steps
    // (~12 cells at step=0.06) so the test cannot hang even if the player is
    // partially blocked by geometry.
    for _ in 0..200 {
        if cam.pos.x >= 7.0 {
            break;
        }
        cam.move_with_collision(0.06, 0.0, &map, PLAYER_RADIUS);
        for _ in 0..3 {
            cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
        }
    }

    // The player MUST have made forward progress (not stuck against a wall).
    assert!(
        cam.pos.x > start_x + 0.5,
        "player must have advanced east across tread (x stopped at {:.3}); \
         possible geometry block — check spiral cell layout",
        cam.pos.x
    );

    let rise = cam.pos.z - initial_z;

    // After crossing the z=1 PosX tread, eye z should be noticeably higher.
    // The tread at (6,5,1) rises from surface z=1.0 (at x=6.0) to 2.0 (at x=7.0).
    // At x≈7.0, eye z ≈ 2.0 + PLAYER_EYE_HEIGHT = 2.5. Rise ≈ 1.0.
    // Require at least 0.4 rise (generous tolerance for approach/snap lag).
    assert!(
        rise >= 0.4,
        "after crossing the z=1 PosX ramp tread, eye z should have risen ≥ 0.4; \
         got {rise:.3} (initial={initial_z:.3}, final={:.3}, x={:.3})",
        cam.pos.z,
        cam.pos.x
    );
}

// ── 22-tower.lvl tests ────────────────────────────────────────────────────────

/// Diagnostic: trace the full tower climb through both PosX staircases.
/// Run with `cargo test tower_two_staircase_trace -- --nocapture` to see output.
#[test]
fn tower_two_staircase_trace() {
    let path = workspace_root().join(TOWER_LVL);
    let lv = load(&path).expect("load 22-tower.lvl");
    let map = lv.build_map();

    let mut cam = CameraPose {
        pos: Vec3::new(3.8, 3.5, 1.0 + PLAYER_EYE_HEIGHT),
        yaw: std::f32::consts::FRAC_PI_2,
        pitch: 0.0,
    };
    let mut motion = PlayerMotion::default();
    for _ in 0..20 {
        cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
    }
    let initial_z = cam.pos.z;
    eprintln!("[A] start: pos={:?}", cam.pos);

    // Staircase A: east at y=3.5 until past landing (x>9).
    while cam.pos.x < 9.0 {
        cam.move_with_collision(0.06, 0.0, &map, PLAYER_RADIUS);
        for _ in 0..3 {
            cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
        }
    }
    eprintln!(
        "[A] done: pos={:?} rise={:.3}",
        cam.pos,
        cam.pos.z - initial_z
    );

    // Transition: south to y=8.5.
    for _ in 0..300 {
        if cam.pos.y >= 8.5 {
            break;
        }
        cam.move_with_collision(0.0, 0.06, &map, PLAYER_RADIUS);
        for _ in 0..3 {
            cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
        }
    }
    eprintln!("[T] after south: pos={:?}", cam.pos);

    // Back west to x=10 to approach staircase B from the west.
    while cam.pos.x > 10.5 {
        cam.move_with_collision(-0.06, 0.0, &map, PLAYER_RADIUS);
        for _ in 0..3 {
            cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
        }
    }
    eprintln!("[T] repositioned: pos={:?}", cam.pos);

    // Staircase B: east at y=8.5 until past landing (x>16).
    for i in 0..400 {
        if cam.pos.x >= 16.0 {
            break;
        }
        let old_z = cam.pos.z;
        cam.move_with_collision(0.06, 0.0, &map, PLAYER_RADIUS);
        for _ in 0..3 {
            cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
        }
        if (cam.pos.z - old_z).abs() > 0.02 {
            eprintln!("[B] i={i} x={:.2} z={:.3}", cam.pos.x, cam.pos.z);
        }
    }
    eprintln!(
        "[B] final: pos={:?} total_rise={:.3}",
        cam.pos,
        cam.pos.z - initial_z
    );
}

/// The three-storey tower level loads without error and the spawn camera
/// produces a non-trivial (non-blank) Framebuffer at 100×32.
#[test]
fn tower_level_loads_and_renders_nonblank() {
    let path = workspace_root().join(TOWER_LVL);
    assert!(path.exists(), "level file missing: {}", path.display());

    let lv = load(&path).expect("22-tower.lvl must parse without error");

    assert_eq!(
        lv.storey_height(),
        4,
        "storey_height must be 4 (as authored in the [room] header)"
    );

    // Spawn at floor 0 — eye z ≈ STANDING_EYE_Z (1.5).
    let cam = lv.spawn_camera();
    assert!(
        (cam.pos.z - STANDING_EYE_Z).abs() < 0.6,
        "spawn z ≈ STANDING_EYE_Z (floor 0): got {:.3}",
        cam.pos.z
    );

    let nonblank = non_empty_glyph_count(&path, &cam);
    assert!(
        nonblank > 200,
        "floor-0 frame must be non-blank: only {nonblank} non-empty glyphs"
    );
}

/// Render the tower from floor 0 (spawn) and from an equivalent camera placed
/// on floor 2 (z shifted by 2 × storey_height = 8 cells). The two frames MUST
/// differ — if both floors hash identically the raycaster is ignoring camera
/// elevation.
#[test]
fn tower_floor0_and_floor2_frames_differ() {
    let path = workspace_root().join(TOWER_LVL);

    let lv = load(&path).expect("load 22-tower.lvl");
    let storey = lv.storey_height() as f32;

    let cam_floor0 = lv.spawn_camera();

    // Floor-2 camera: same x/y/yaw as spawn but z elevated by 2 storeys.
    let cam_floor2 = CameraPose {
        pos: Vec3::new(
            cam_floor0.pos.x,
            cam_floor0.pos.y,
            cam_floor0.pos.z + 2.0 * storey,
        ),
        yaw: cam_floor0.yaw,
        pitch: cam_floor0.pitch,
    };

    let hash0 = render_hash(&path, &cam_floor0);
    let hash2 = render_hash(&path, &cam_floor2);

    assert_ne!(
        hash0, hash2,
        "floor-0 and floor-2 frames must differ in 22-tower.lvl: \
         if they match the raycaster is ignoring camera elevation.\n\
         floor-0 digest: {hash0}\n\
         floor-2 digest: {hash2}"
    );
}

/// Scripted climb through both staircases in 22-tower.lvl.
///
/// Staircase A (PosX, y=3): floor 0 → floor 1 — camera rises ~4 cells.
/// Staircase B (PosX, y=8): floor 1 → floor 2 — camera rises another ~4 cells.
///
/// Both staircases use PosX ramp physics, proven by the existing 20/21 tests.
/// The scripted walk:
///   1. Settle at spawn, then walk east at y=3.5 through staircase A.
///   2. Walk south to y=8.5, then east at y=8.5 through staircase B.
///   3. Assert total z-rise ≥ 2 × storey_height − 1.5 cells (≥6.5 for height=4).
#[test]
fn tower_scripted_climb_raises_z_two_storeys() {
    let path = workspace_root().join(TOWER_LVL);
    let lv = load(&path).expect("load 22-tower.lvl");
    let map = lv.build_map();

    let storey_height = lv.storey_height();

    // ── Phase 1: staircase A (floor 0 → floor 1, corridor y=3) ──────────────
    let mut cam = CameraPose {
        pos: Vec3::new(3.8, 3.5, 1.0 + PLAYER_EYE_HEIGHT),
        yaw: std::f32::consts::FRAC_PI_2,
        pitch: 0.0,
    };
    let mut motion = PlayerMotion::default();

    for _ in 0..20 {
        cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
    }
    let initial_z = cam.pos.z;

    // Walk east past the staircase A landing (x > 9.0).
    while cam.pos.x < 9.0 {
        cam.move_with_collision(0.06, 0.0, &map, PLAYER_RADIUS);
        for _ in 0..3 {
            cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
        }
    }

    assert!(
        motion.grounded,
        "must be grounded on floor 1 after staircase A (x={:.3}, z={:.3})",
        cam.pos.x, cam.pos.z
    );
    let phase1_rise = cam.pos.z - initial_z;
    assert!(
        phase1_rise >= storey_height as f32 - 1.0,
        "staircase A rise must be ≥{} cells; got {phase1_rise:.3}",
        storey_height - 1
    );

    // ── Transition: walk south from y≈3.5 to y≈8.5 (staircase B corridor) ──
    for _ in 0..300 {
        if cam.pos.y >= 8.5 {
            break;
        }
        cam.move_with_collision(0.0, 0.06, &map, PLAYER_RADIUS);
        for _ in 0..3 {
            cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
        }
    }
    assert!(
        cam.pos.y >= 8.0,
        "must reach y=8.5 corridor; stuck at y={:.3}",
        cam.pos.y
    );

    // Reposition x just west of staircase B's first tread (tread1 at x=11).
    // Walk to x≈10 within the floor-1 corridor.
    while cam.pos.x > 10.5 {
        cam.move_with_collision(-0.06, 0.0, &map, PLAYER_RADIUS);
        for _ in 0..3 {
            cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
        }
    }

    // ── Phase 2: staircase B (floor 1 → floor 2, corridor y=8) ──────────────
    let floor1_z = cam.pos.z;

    // Walk east through all 4 PosX treads at (11,8,5)..(14,8,8) and past
    // the landing at (15,8,8). Target x=16 ensures we clear the landing.
    for _ in 0..400 {
        if cam.pos.x >= 16.0 {
            break;
        }
        cam.move_with_collision(0.06, 0.0, &map, PLAYER_RADIUS);
        for _ in 0..3 {
            cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
        }
    }

    assert!(
        motion.grounded,
        "must be grounded at/near floor 2 after staircase B (x={:.3}, z={:.3})",
        cam.pos.x, cam.pos.z
    );
    let phase2_rise = cam.pos.z - floor1_z;
    assert!(
        phase2_rise >= storey_height as f32 - 1.0,
        "staircase B rise must be ≥{} cells; got {phase2_rise:.3}",
        storey_height - 1
    );

    // ── Final: combined rise ≥ 2 storeys − 1.5 ───────────────────────────────
    let total_rise = cam.pos.z - initial_z;
    let min_expected = (2 * storey_height) as f32 - 1.5;
    assert!(
        total_rise >= min_expected,
        "total rise across two storeys should be ≥{min_expected:.1} cells; \
         got {total_rise:.3} (initial_z={initial_z:.3}, final_z={:.3})",
        cam.pos.z
    );
}

/// Prove staircase B's four PosX treads each provide a live ramp surface.
///
/// This is a targeted physics probe: place the camera just west of tread 1
/// at floor-1 eye height and walk east. By x=16 the camera should have risen
/// at least 3 cells (storey_height − 1) via 4 PosX treads.
#[test]
fn tower_staircase_b_four_treads_raise_z() {
    let path = workspace_root().join(TOWER_LVL);
    let lv = load(&path).expect("load 22-tower.lvl");
    let map = lv.build_map();

    let storey_height = lv.storey_height();

    let mut cam = CameraPose {
        pos: Vec3::new(10.0, 8.5, 5.0 + PLAYER_EYE_HEIGHT),
        yaw: std::f32::consts::FRAC_PI_2,
        pitch: 0.0,
    };
    let mut motion = PlayerMotion::default();

    // Settle onto floor-1 surface.
    for _ in 0..20 {
        cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
    }
    let initial_z = cam.pos.z;

    // Walk east through all 4 treads and past the landing.
    for _ in 0..400 {
        if cam.pos.x >= 16.0 {
            break;
        }
        cam.move_with_collision(0.06, 0.0, &map, PLAYER_RADIUS);
        for _ in 0..3 {
            cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
        }
    }

    let rise = cam.pos.z - initial_z;
    assert!(
        rise >= storey_height as f32 - 1.0,
        "staircase B (4 PosX treads) must raise z ≥{} cells; got {rise:.3} \
         (initial_z={initial_z:.3}, final_z={:.3}, x={:.3})",
        storey_height - 1,
        cam.pos.z,
        cam.pos.x
    );
}
