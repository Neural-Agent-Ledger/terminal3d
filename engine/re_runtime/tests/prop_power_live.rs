//! Power-reactive props, end to end: a level whose `[prop]`s bind `node:`, `pipe:`,
//! `budget:` and `door` is loaded exactly as a host game loads it (`load_string` →
//! `build_map` → the emit binder → `bind_prop_power`), switches are flipped
//! through the REAL puzzle path (`re_puzzle::puzzle_interact`, and in the ECS test
//! the `InteractAt` message the input layer sends), and the map cells, line palette
//! ids, lamp intensities and emitter slots are asserted to change on the right edge
//! and back — and not at all when unrelated state moves.
//!
//! This fixture's `load()` does not call `re_runtime::notes::build_level_text` before
//! `bind_emit_parts`: the fixture level below authors no `[note]` clause, so
//! `build_level_text`'s `stamp_notes` — the only part that can touch
//! `map.materials()` — would return immediately on an empty `level.notes`,
//! making the two load orders byte-identical for every assertion in this
//! file. A host's own audio-cue integration (asserting a lit prop's lamp
//! actually hums) is out of scope here — audio cueing is host content, not an
//! engine seam this crate owns.

use bevy::prelude::*;
use re_level::props::PowerSource;
use re_level::Level;
use re_puzzle::{puzzle_interact, InteractAt, PuzzleState};
use re_runtime::prop_power::{
    bind_prop_power, prop_power_sys, source_value, sync_prop_power, PropPower, PropPowerEdge,
    PropPowerPlugin,
};
use re_world::{CellKind, Map};

/// A 24×24×10 room: inputs R and G (both `1`), an output that fires on either
/// pipe, both pipes switchable and authored OFF, a budget of 2. The puzzle strip is
/// `y ≥ 19`; props stand in the middle.
const LEVEL: &str = "[meta]
name = prop-power
title = prop power

[room]
size = 24 24 10
ambient = solid
storey_height = 4
power_budget = 2
render_pipes = true

[node R]
pos = 2,22,2
glyph = R
color = #FF4040
is_input = yes

[node G]
pos = 4,22,2
glyph = G
color = #40FF60
is_input = yes

[node output]
pos = 21,22,2
glyph = O
color = #FFFFFF
is_output = yes
threshold = 1

[pipe R->output]
state = OFF
path = (2,22,2) (2,20,2) (21,20,2) (21,22,2)
switch_at = (6,20,2)

[pipe G->output]
state = OFF
path = (4,22,2) (4,21,2) (20,21,2) (20,22,2) (21,22,2)
switch_at = (8,21,2)

[inputs]
R = 1
G = 1

[pedestal]
center = 18 22 2
volume = 1 1 1

[door]
pos = 22 1 1
target_parts = output

[player]
spawn = 12 17 1
facing = -y

[prop rack_out]
kind = rack
at = 11 10 1
facing = +y
size = 1 2 3
power = node:output

[prop rack_r]
kind = rack
at = 14 10 1
facing = +y
size = 1 2 3
power = node:R

[prop rack_tally]
kind = rack
at = 17 10 1
facing = +y
size = 1 2 3
power = door
invert = yes

[prop wire]
kind = cable_run
follow = pipe:R->output

[prop bank]
kind = budget_bank
at = 8 1 4
facing = +y

[prop lamp]
kind = lamp_pendant
at = 6.5 8.5 6.0
light = yes
power = pipe:G->output

[prop holo]
kind = holo_display
shape = pyramid
at = 6 13 2
size = 3 3 1
style = 2
power = node:output
";

const SWITCH_R: (i32, i32, i32) = (6, 20, 2);
const SWITCH_G: (i32, i32, i32) = (8, 21, 2);

/// Load exactly as a host's own level-enter handler does, up to the prop
/// binding (minus the host's note-binding step — see the module docs above
/// for why dropping it is behaviour-preserving for this fixture).
fn load() -> (Level, Map, PuzzleState, PropPower) {
    let level = re_level::load_string(LEVEL, "prop-power.lvl")
        .unwrap_or_else(|e| panic!("fixture loads: {e}"));
    let mut map = level.build_map();
    let _ = re_runtime::emit_parts::bind_emit_parts(&level, &mut map);
    let pp = bind_prop_power(&level, &map);
    let puzzle = PuzzleState::from_level(&level).expect("puzzle builds");
    (level, map, puzzle, pp)
}

fn gate<'a>(level: &'a Level, prop: &str) -> (usize, &'a re_level::props::PowerGate) {
    level
        .prop_power
        .iter()
        .enumerate()
        .find(|(_, g)| g.prop == prop)
        .unwrap_or_else(|| {
            panic!(
                "gate for [prop {prop}]: {:?}",
                level.prop_power.iter().map(|g| &g.prop).collect::<Vec<_>>()
            )
        })
}

/// The `budget:K` gate (the bank has one per slot).
fn budget_gate(level: &Level, k: u32) -> usize {
    level
        .prop_power
        .iter()
        .position(|g| g.source == PowerSource::Budget(k))
        .unwrap_or_else(|| panic!("no budget:{k} gate"))
}

/// Assert gate `gi` shows `lit` in the world: every lit part cell is a `WallSolid`
/// carrying its palette id (else Empty), every gated line carries the matching
/// palette id, every gated lamp is at its authored intensity (else 0).
fn assert_shows(level: &Level, map: &Map, gi: usize, lit: bool) {
    let g = &level.prop_power[gi];
    for &((x, y, z), id) in &g.parts {
        if lit {
            assert_eq!(
                map.cell_at(x, y, z),
                CellKind::WallSolid,
                "[prop {}] part ({x},{y},{z}) lit",
                g.prop
            );
            assert_eq!(
                map.side_data_at(x, y, z),
                id,
                "[prop {}] part ({x},{y},{z}) lit id",
                g.prop
            );
        } else {
            assert_eq!(
                map.cell_at(x, y, z),
                CellKind::Empty,
                "[prop {}] part ({x},{y},{z}) dark",
                g.prop
            );
        }
    }
    for &(i, unlit, lit_id) in &g.lines {
        assert_ne!(
            unlit, lit_id,
            "[prop {}] line {i}: ink and lit ids differ",
            g.prop
        );
        assert_eq!(
            map.line_palette_id(i),
            if lit { lit_id } else { unlit },
            "[prop {}] line {i} palette (lit={lit})",
            g.prop
        );
    }
    for &(i, lamp) in &g.lights {
        let want = if lit { lamp.intensity } else { 0.0 };
        assert!(
            (map.atmosphere().lights[i].intensity - want).abs() < 1e-6,
            "[prop {}] lamp {i} intensity {} want {want}",
            g.prop,
            map.atmosphere().lights[i].intensity
        );
    }
}

fn hologram_cells(map: &Map) -> usize {
    let (sx, sy, sz) = map.size();
    let mut n = 0;
    for z in 0..sz as i32 {
        for y in 0..sy as i32 {
            for x in 0..sx as i32 {
                if map.cell_at(x, y, z) == CellKind::Hologram {
                    n += 1;
                }
            }
        }
    }
    n
}

fn emitter_set(level: &Level) -> Vec<re_render::Emitter> {
    level
        .emitters
        .iter()
        .map(|s| re_render::Emitter::from_spec_lit(s, &level.lights))
        .collect()
}

/// Every binding kind the test exercises is actually present in the fixture.
#[test]
fn the_fixture_covers_every_binding_kind() {
    let (level, _, _, pp) = load();
    let has = |f: &dyn Fn(&PowerSource) -> bool| level.prop_power.iter().any(|g| f(&g.source));
    assert!(has(&|s| matches!(s, PowerSource::Node(_))));
    assert!(has(&|s| matches!(s, PowerSource::Pipe(_))));
    assert!(has(&|s| matches!(s, PowerSource::Budget(_))));
    assert!(has(&|s| matches!(s, PowerSource::Door)));
    assert!(level.prop_power.iter().any(|g| g.invert));
    assert!(level.prop_power.iter().any(|g| !g.parts.is_empty()));
    assert!(level.prop_power.iter().any(|g| !g.lines.is_empty()));
    assert!(level.prop_power.iter().any(|g| !g.lights.is_empty()));
    assert!(level.prop_power.iter().any(|g| !g.holos.is_empty()));
    assert!(level.prop_power.iter().any(|g| !g.emitters.is_empty()));
    assert_eq!(pp.gates.len(), level.prop_power.len());
    // Every part slot was claimed (none silently dropped at bind).
    for (b, g) in pp.gates.iter().zip(&level.prop_power) {
        assert!(
            b.slot_len() >= g.parts.len(),
            "[prop {}] claimed {} of {} parts",
            g.prop,
            b.slot_len(),
            g.parts.len()
        );
    }
}

/// The binding starts in `initial_lit`, and the built map shows exactly that.
#[test]
fn initial_state_matches_initial_lit() {
    let (level, map, _, pp) = load();
    for (gi, g) in level.prop_power.iter().enumerate() {
        assert_eq!(pp.gates[gi].is_lit(), g.initial_lit, "[prop {}]", g.prop);
        let b = &pp.gates[gi];
        assert_eq!(
            b.stamped_len(),
            if g.initial_lit { b.slot_len() } else { 0 },
            "[prop {}] stamped slots at load",
            g.prop
        );
        assert_shows(&level, &map, gi, g.initial_lit);
    }
    // Concretely: R is an input authored 1, so its rack is lit; the output is dark;
    // the door tally (invert) is lit while unsolved; no hologram yet.
    assert!(level.prop_power[gate(&level, "rack_r").0].initial_lit);
    assert!(!level.prop_power[gate(&level, "rack_out").0].initial_lit);
    assert!(level.prop_power[gate(&level, "rack_tally").0].initial_lit);
    assert_eq!(hologram_cells(&map), 0);
}

/// This level's live load state agrees with the static one, so the first frame
/// writes nothing (and would not dirty the render).
#[test]
fn first_sync_after_load_changes_nothing_when_the_static_state_is_right() {
    let (_, mut map, puzzle, mut pp) = load();
    let r = sync_prop_power(&mut pp, &puzzle, &mut map, None);
    assert!(r.edges.is_empty(), "{r:?}");
    assert!(!r.map_changed);
}

/// `node:` + `pipe:` + `door` + `budget:1` + holograms, through the real switch.
#[test]
fn switching_r_lights_its_props_on_the_edge_and_back() {
    let (level, mut map, mut puzzle, mut pp) = load();
    let out = gate(&level, "rack_out").0;
    let tally = gate(&level, "rack_tally").0;
    let wire = gate(&level, "wire").0;
    let holo = gate(&level, "holo").0;
    let b1 = budget_gate(&level, 1);
    let b2 = budget_gate(&level, 2);
    let rack_r = gate(&level, "rack_r").0;
    let lamp = gate(&level, "lamp").0;

    assert!(
        puzzle_interact(&mut puzzle, SWITCH_R.0, SWITCH_R.1, SWITCH_R.2),
        "switch R toggles"
    );
    assert!(
        puzzle.solved,
        "R->output ON fires the output and solves the door"
    );

    let r = sync_prop_power(&mut pp, &puzzle, &mut map, None);
    assert!(r.map_changed);
    let mut flipped: Vec<(usize, bool)> = r.edges;
    flipped.sort_unstable();
    let mut want = vec![
        (out, true),
        (tally, false),
        (wire, true),
        (holo, true),
        (b1, true),
    ];
    want.sort_unstable();
    assert_eq!(flipped, want, "exactly the gates bound to what changed");
    assert_shows(&level, &map, out, true);
    assert_shows(&level, &map, tally, false);
    assert_shows(&level, &map, wire, true);
    assert_shows(&level, &map, b1, true);
    assert!(hologram_cells(&map) > 0, "the hologram is stamped");
    assert_eq!(hologram_cells(&map), pp.gates[holo].stamped_len());
    // Untouched: the R rack (R is an input, still 1), budget:2, the G lamp.
    assert_shows(&level, &map, rack_r, true);
    assert_shows(&level, &map, b2, false);
    assert_shows(&level, &map, lamp, false);

    // Idle: a second sync with nothing changed does nothing.
    let r = sync_prop_power(&mut pp, &puzzle, &mut map, None);
    assert_eq!(r, re_runtime::prop_power::PowerSync::default());

    // And back.
    assert!(puzzle_interact(
        &mut puzzle,
        SWITCH_R.0,
        SWITCH_R.1,
        SWITCH_R.2
    ));
    assert!(!puzzle.solved);
    let r = sync_prop_power(&mut pp, &puzzle, &mut map, None);
    assert_eq!(r.edges.len(), 5, "{r:?}");
    for (gi, g) in level.prop_power.iter().enumerate() {
        assert_shows(&level, &map, gi, g.initial_lit);
    }
    assert_eq!(hologram_cells(&map), 0, "the hologram is cleared");
    // Cell-for-cell identical to a fresh load.
    let fresh = load().1;
    let (sx, sy, sz) = map.size();
    for z in 0..sz as i32 {
        for y in 0..sy as i32 {
            for x in 0..sx as i32 {
                assert_eq!(
                    map.cell_at(x, y, z),
                    fresh.cell_at(x, y, z),
                    "({x},{y},{z}) kind"
                );
                assert_eq!(
                    map.side_data_at(x, y, z),
                    fresh.side_data_at(x, y, z),
                    "({x},{y},{z}) side_data"
                );
            }
        }
    }
}

/// `pipe:` on a lamp (R3) + `budget:2` with its emitter swap.
#[test]
fn switching_g_turns_the_lamp_on_and_both_pipes_fill_the_budget() {
    let (level, mut map, mut puzzle, mut pp) = load();
    let lamp = gate(&level, "lamp").0;
    let b2 = budget_gate(&level, 2);
    let mut emitters = emitter_set(&level);
    let (em_idx, _) = level.prop_power[b2].emitters[0];
    let before = format!("{:?}", emitters[em_idx]);

    assert!(puzzle_interact(
        &mut puzzle,
        SWITCH_G.0,
        SWITCH_G.1,
        SWITCH_G.2
    ));
    let r = sync_prop_power(&mut pp, &puzzle, &mut map, Some(&mut emitters));
    assert!(r.edges.contains(&(lamp, true)));
    assert_shows(&level, &map, lamp, true);
    let (li, lit_lamp) = level.prop_power[lamp].lights[0];
    assert!(map.atmosphere().lights[li].intensity > 0.0);
    // The re-bake: the light environment the renderer derives from the map now
    // differs from the one it derived while the lamp was dark.
    let (_, dark_map, _, _) = load();
    assert_ne!(
        format!("{:?}", re_render::light_env_for_map(&map)),
        format!("{:?}", re_render::light_env_for_map(&dark_map)),
        "the baked LightEnv carries the lamp at {:?}",
        lit_lamp.pos
    );
    assert!(!r.emitters_changed, "one pipe is budget:1, not the cap");
    assert_shows(&level, &map, b2, false);

    assert!(puzzle_interact(
        &mut puzzle,
        SWITCH_R.0,
        SWITCH_R.1,
        SWITCH_R.2
    ));
    let r = sync_prop_power(&mut pp, &puzzle, &mut map, Some(&mut emitters));
    assert!(r.edges.contains(&(b2, true)), "{r:?}");
    assert!(r.emitters_changed, "the cap swaps the spark emitter");
    assert_shows(&level, &map, b2, true);
    assert_ne!(
        format!("{:?}", emitters[em_idx]),
        before,
        "emitter slot swapped to the lit spec"
    );

    assert!(puzzle_interact(
        &mut puzzle,
        SWITCH_R.0,
        SWITCH_R.1,
        SWITCH_R.2
    ));
    let _ = sync_prop_power(&mut pp, &puzzle, &mut map, Some(&mut emitters));
    assert_eq!(
        format!("{:?}", emitters[em_idx]),
        before,
        "and back to the unlit spec"
    );
    assert_shows(&level, &map, b2, false);
    assert_shows(&level, &map, lamp, true);

    assert!(puzzle_interact(
        &mut puzzle,
        SWITCH_G.0,
        SWITCH_G.1,
        SWITCH_G.2
    ));
    let _ = sync_prop_power(&mut pp, &puzzle, &mut map, Some(&mut emitters));
    assert_shows(&level, &map, lamp, false);
}

/// State no gate listens to moves nothing: toggling G while R already powers the
/// output leaves the output / door / R-wire props exactly as they were, and a
/// bookkeeping change (attempts) is not an edge at all.
#[test]
fn unrelated_state_changes_touch_nothing() {
    let (level, mut map, mut puzzle, mut pp) = load();
    assert!(puzzle_interact(
        &mut puzzle,
        SWITCH_R.0,
        SWITCH_R.1,
        SWITCH_R.2
    ));
    let _ = sync_prop_power(&mut pp, &puzzle, &mut map, None);

    puzzle.attempts += 7;
    let r = sync_prop_power(&mut pp, &puzzle, &mut map, None);
    assert_eq!(r, re_runtime::prop_power::PowerSync::default());

    let watched: Vec<usize> = ["rack_out", "rack_tally", "wire", "holo", "rack_r"]
        .iter()
        .map(|p| gate(&level, p).0)
        .collect();
    let holo_before = hologram_cells(&map);
    assert!(puzzle_interact(
        &mut puzzle,
        SWITCH_G.0,
        SWITCH_G.1,
        SWITCH_G.2
    ));
    let r = sync_prop_power(&mut pp, &puzzle, &mut map, None);
    for gi in &watched {
        assert!(
            !r.edges.iter().any(|(g, _)| g == gi),
            "[prop {}] must not flip: {r:?}",
            level.prop_power[*gi].prop
        );
    }
    assert_shows(&level, &map, gate(&level, "rack_out").0, true);
    assert_shows(&level, &map, gate(&level, "rack_tally").0, false);
    assert_shows(&level, &map, gate(&level, "wire").0, true);
    assert_eq!(hologram_cells(&map), holo_before);
}

/// `source_value` per kind.
#[test]
fn source_values_follow_the_spec() {
    let (_, _, mut puzzle, _) = load();
    assert!(source_value(PowerSource::Node(0), &puzzle), "R input = 1");
    assert!(!source_value(PowerSource::Node(2), &puzzle));
    assert!(!source_value(PowerSource::Pipe(0), &puzzle));
    assert!(!source_value(PowerSource::Door, &puzzle));
    assert!(!source_value(PowerSource::Budget(1), &puzzle));
    assert!(!source_value(PowerSource::Seq(1), &puzzle));
    assert!(puzzle_interact(
        &mut puzzle,
        SWITCH_R.0,
        SWITCH_R.1,
        SWITCH_R.2
    ));
    assert!(source_value(PowerSource::Node(2), &puzzle));
    assert!(source_value(PowerSource::Pipe(0), &puzzle));
    assert!(source_value(PowerSource::Door, &puzzle));
    assert!(source_value(PowerSource::Budget(1), &puzzle));
    assert!(!source_value(PowerSource::Budget(2), &puzzle));
    puzzle.sequence.progress = 1;
    assert!(source_value(PowerSource::Seq(1), &puzzle));
    assert!(!source_value(PowerSource::Seq(2), &puzzle));
}

// NB: "a lit prop lamp hums (the `LampHum` loop) and a dark one is silent" is
// NOT asserted here: that audio-cue wiring is host content, and `re_runtime`
// must not depend on a host crate. A small companion test in the host's own
// test tree, importing `re_runtime::prop_power` alongside its own audio-cue
// module, would cover it without creating a reverse dependency.

// ── the live ECS seam ──

#[derive(Resource, Default)]
struct Seen {
    map_dirty: bool,
    emitters_dirty: bool,
    edges: Vec<PropPowerEdge>,
}

#[allow(clippy::needless_pass_by_value)] // Bevy system params are by value.
fn record(
    map: Res<Map>,
    em: Res<re_render::EmitterSet>,
    mut edges: MessageReader<PropPowerEdge>,
    mut seen: ResMut<Seen>,
) {
    seen.map_dirty = map.is_changed();
    seen.emitters_dirty = em.is_changed();
    seen.edges = edges.read().cloned().collect();
}

fn app() -> (App, Level) {
    let (level, map, puzzle, pp) = load();
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(re_puzzle::PuzzlePlugin)
        .add_plugins(PropPowerPlugin);
    app.insert_resource(map);
    app.insert_resource(puzzle);
    app.insert_resource(pp);
    app.insert_resource(re_render::EmitterSet {
        emitters: emitter_set(&level),
    });
    app.init_resource::<Seen>();
    app.add_systems(Update, record.after(prop_power_sys));
    (app, level)
}

/// Through the plugin: an `InteractAt` on switch R flips the props THAT frame, marks
/// the map dirty exactly then, publishes one edge per flipped gate, and idle frames
/// stay quiet.
#[test]
fn the_plugin_follows_interact_messages_and_only_dirties_on_edges() {
    let (mut app, level) = app();
    app.update(); // the insert frame
    app.update();
    assert!(
        !app.world().resource::<Seen>().map_dirty,
        "idle frame is quiet"
    );

    app.world_mut()
        .write_message(InteractAt::cell(SWITCH_R.0, SWITCH_R.1, SWITCH_R.2));
    app.update();
    {
        let seen = app.world().resource::<Seen>();
        assert!(seen.map_dirty, "the flip frame dirties the map");
        assert!(!seen.emitters_dirty, "no emitter gate flipped");
        assert_eq!(seen.edges.len(), 5, "{:?}", seen.edges);
        let tally = seen
            .edges
            .iter()
            .find(|e| e.prop == "rack_tally")
            .expect("tally edge");
        assert!(!tally.lit && tally.at.is_some());
    }
    assert_shows(
        &level,
        app.world().resource::<Map>(),
        gate(&level, "rack_out").0,
        true,
    );

    app.update();
    let seen = app.world().resource::<Seen>();
    assert!(
        !seen.map_dirty && seen.edges.is_empty(),
        "idle after the flip"
    );

    // A press into empty air: the puzzle is untouched, nothing moves.
    app.world_mut().write_message(InteractAt::cell(12, 12, 1));
    app.update();
    let seen = app.world().resource::<Seen>();
    assert!(!seen.map_dirty && seen.edges.is_empty());

    // The cap: both pipes on swaps the spark emitter and marks the set changed.
    app.world_mut()
        .write_message(InteractAt::cell(SWITCH_G.0, SWITCH_G.1, SWITCH_G.2));
    app.update();
    let seen = app.world().resource::<Seen>();
    assert!(seen.emitters_dirty, "budget cap swaps the emitter");
    assert!(seen.edges.iter().any(|e| e.prop == "lamp" && e.lit));
    assert_shows(
        &level,
        app.world().resource::<Map>(),
        gate(&level, "lamp").0,
        true,
    );
}
