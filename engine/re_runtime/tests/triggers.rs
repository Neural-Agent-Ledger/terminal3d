//! The pure firing-core tests for `apply_triggers` and `apply_zone_inputs`,
//! alongside `re_runtime::triggers`.
//!
//! `apply_triggers_fires_once_on_entry_not_while_idle` and
//! `apply_triggers_once_vs_refire` load `levels/T0-trigger.lvl`, an engine
//! example/fixture level, not a campaign level — an engine-repo test loading
//! it stays engine-only. `apply_zone_inputs_solves_while_standing_in_zone`
//! exercises the same mechanic (a `[zone_input]` floor-zone gating a
//! `PuzzleState`) against a minimal synthetic fixture defined right here,
//! rather than any campaign level, so the pure helper stays covered with no
//! campaign coupling. A host's own end-to-end app test can pin its own
//! campaign-level content separately.

use re_level::load_string;
use re_puzzle::{set_input_value, PuzzleState};
use re_runtime::triggers::{apply_triggers, apply_zone_inputs, TriggerRuntime};
use re_world::CameraPose;

// ---------------------------------------------------------------------------
// Floor-zone input: player-position-as-input
// ---------------------------------------------------------------------------

/// A minimal room with one `[zone_input]` floor-zone driving the input side of
/// an otherwise permanently-ON pipe: standing in the zone solves the network,
/// stepping out un-solves it.
const ZONE_FIXTURE: &str = "\
[meta]
name = zone-fixture
[room]
size = 8 8 4
[node present]
pos = 4,4,1
is_input = yes
[node output]
pos = 2,4,1
is_output = yes
threshold = 1
[pipe present->output]
state = ON
path = (4,4,1) (2,4,1)
[inputs]
present = 0
[zone_input warm]
node = present
region = (3,3,1)-(5,5,1)
[pedestal]
center = 3 4 1
volume = 2 2 2
[door]
pos = 1 4 1
target_parts = output
[player]
spawn = 6 4 1
facing = -x
";

/// The pure floor-zone helper drives an input from the camera cell: standing
/// in the zone solves the network, stepping out un-solves it.
#[test]
fn apply_zone_inputs_solves_while_standing_in_zone() {
    let level = load_string(ZONE_FIXTURE, "<zone-fixture>").expect("zone fixture must load");
    let mut puzzle = PuzzleState::from_level(&level).expect("zone fixture network builds");
    assert!(
        !level.zone_inputs.is_empty(),
        "the fixture must author a zone_input"
    );

    // OFF the zone (camera at spawn): present must be 0 → not solved.
    let cam = level.spawn_camera();
    set_input_value(&mut puzzle, level.zone_inputs[0].node_index, 0);
    apply_zone_inputs(&level.zone_inputs, &cam, &mut puzzle);
    assert!(
        !puzzle.solved,
        "off the warm zone the level must NOT be solved"
    );

    // Step ONTO the warm zone (centre of the authored zone box). Set the
    // CameraPose position field-by-field so we stay on the same glam Vec3 the
    // component uses (no glam-version juggling).
    let z = &level.zone_inputs[0];
    let mut on_cam = cam;
    on_cam.pos.x = (z.x0 + z.x1) as f32 / 2.0 + 0.5;
    on_cam.pos.y = (z.y0 + z.y1) as f32 / 2.0 + 0.5;
    on_cam.pos.z = z.z0 as f32 + 0.5;
    let changed = apply_zone_inputs(&level.zone_inputs, &on_cam, &mut puzzle);
    assert!(changed, "stepping onto the zone changes the input");
    assert!(
        puzzle.solved,
        "standing in the warm zone solves the fixture"
    );

    // Step OFF again → un-solved.
    let changed = apply_zone_inputs(&level.zone_inputs, &cam, &mut puzzle);
    assert!(changed, "stepping off the zone changes the input back");
    assert!(
        !puzzle.solved,
        "stepping off the warm zone un-solves the fixture"
    );
}

// ---------------------------------------------------------------------------
// Trigger-fields (T0 "Trip-Field"): regions that fire on entry
// ---------------------------------------------------------------------------

/// The workspace `levels/` directory as an ABSOLUTE path, derived from this
/// crate's `CARGO_MANIFEST_DIR` (`crates/re_runtime` → `../../levels`).
/// `cargo test` runs with the crate dir as cwd, so relative level paths would
/// not resolve.
fn levels_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .unwrap()
        .join("levels")
}

fn abs_level(file: &str) -> std::path::PathBuf {
    levels_root().join(file)
}

/// Pure firing core: a trigger fires EXACTLY ONCE on the OUTSIDE→INSIDE entry
/// edge, NOT while merely standing outside, and NOT again while standing still
/// inside. `ActivateNode` drives the named input node so the network solves.
#[test]
fn apply_triggers_fires_once_on_entry_not_while_idle() {
    let level = re_level::load(abs_level("T0-trigger.lvl")).expect("T0 must load");
    let mut puzzle = PuzzleState::from_level(&level).expect("T0 network builds");
    assert_eq!(level.triggers.len(), 1, "T0 authors exactly one trigger");

    // The map carries the resolved trigger (region + resolved ActivateNode index).
    let map = level.build_map();
    let triggers = map.resolved_triggers().to_vec();
    let mut states = vec![TriggerRuntime::default(); triggers.len()];

    // Start the input LOW so the trigger is what raises it (the runtime start;
    // the level's `[inputs] gate = 1` is only the static lvlcheck seed).
    let gate = level.node_index("gate").expect("gate node");
    set_input_value(&mut puzzle, gate, 0);
    assert!(!puzzle.solved, "gate low → T0 not solved at runtime start");

    // The trigger region centre (in cell units) and a point well OUTSIDE it.
    let re_core::Region::Box { min, max } = triggers[0].region else {
        panic!("T0 trigger region is a Box");
    };
    let inside_cam = {
        let mut c = level.spawn_camera();
        c.pos.x = (min.0 + max.0) as f32 / 2.0 + 0.5;
        c.pos.y = (min.1 + max.1) as f32 / 2.0 + 0.5;
        c.pos.z = min.2 as f32 + 0.5;
        c
    };
    let outside_cam = level.spawn_camera(); // spawns OFF the trigger (x=14)

    // FRAME 1 — standing OUTSIDE: nothing fires, puzzle untouched.
    let fired = apply_triggers(&triggers, &outside_cam, &mut states, &mut puzzle);
    assert!(
        fired.is_empty(),
        "no fire while standing outside the region"
    );
    assert!(!puzzle.solved, "outside → gate still low → not solved");

    // FRAME 2 — ENTER the region: fires once, drives `gate` → solves.
    let fired = apply_triggers(&triggers, &inside_cam, &mut states, &mut puzzle);
    assert_eq!(fired.len(), 1, "entering the region fires exactly once");
    assert_eq!(
        fired[0].label, "gate",
        "ActivateNode surfaces the node name"
    );
    assert!(puzzle.solved, "the trigger drove `gate` → network solves");

    // FRAME 3 — STILL INSIDE (no fresh edge): must NOT re-fire.
    let fired = apply_triggers(&triggers, &inside_cam, &mut states, &mut puzzle);
    assert!(fired.is_empty(), "standing still inside never re-fires");
}

/// `once = true` disarms after the first fire even across an EXIT + RE-ENTER;
/// `once = false` re-fires on each fresh entry. Uses synthetic triggers so both
/// flags are covered without authoring two levels.
#[test]
fn apply_triggers_once_vs_refire() {
    let level = re_level::load(abs_level("T0-trigger.lvl")).expect("T0 must load");
    let mut puzzle = PuzzleState::from_level(&level).expect("T0 builds");

    let region = re_core::Region::Box {
        min: (5, 0, 0),
        max: (6, 0, 0),
    };
    let once_t = re_core::Trigger {
        region: region.clone(),
        action: re_core::TriggerAction::event("once_ev"),
        once: true,
    };
    let repeat_t = re_core::Trigger {
        region,
        action: re_core::TriggerAction::event("repeat_ev"),
        once: false,
    };
    let triggers = vec![once_t, repeat_t];
    let mut states = vec![TriggerRuntime::default(); 2];

    let mut cam = level.spawn_camera();
    let inside = |c: &mut CameraPose| {
        c.pos.x = 5.5;
        c.pos.y = 0.5;
        c.pos.z = 0.5;
    };
    let outside = |c: &mut CameraPose| {
        c.pos.x = 0.5;
        c.pos.y = 0.5;
        c.pos.z = 0.5;
    };

    // Enter: BOTH fire.
    inside(&mut cam);
    let f = apply_triggers(&triggers, &cam, &mut states, &mut puzzle);
    assert_eq!(f.len(), 2, "first entry fires both");

    // Exit then re-enter: only the `once: false` trigger fires again.
    outside(&mut cam);
    let f = apply_triggers(&triggers, &cam, &mut states, &mut puzzle);
    assert!(f.is_empty(), "leaving the region fires nothing");
    inside(&mut cam);
    let f = apply_triggers(&triggers, &cam, &mut states, &mut puzzle);
    assert_eq!(f.len(), 1, "re-entry fires only the non-once trigger");
    assert_eq!(
        f[0].label, "repeat_ev",
        "the disarmed `once` trigger stays silent"
    );
}
