//! Unit tests for the level-content linter, on synthetic levels built through
//! the real `re_level` parser (so the fixtures exercise the same path the
//! shipped levels do). Each fixture targets one verdict class.

use super::*;
use re_level::load_string;

/// A small solvable fixture: input R (=1) -> output (threshold 1), one
/// switchable pipe initially OFF. Toggling the one switch wins.
const SOLVABLE: &str = r#"
[meta]
name = fixture-solvable
[room]
size = 12 8 4
ambient = trace-pattern
[node R]
pos = 3,3,2
glyph = R
color = #FF0000
is_input = yes
[node output]
pos = 9,3,2
glyph = O
color = #FFFFFF
is_output = yes
threshold = 1
[pipe R->output]
state = OFF
path = (3,3,2) (9,3,2)
switch_at = (6,3,2)
[inputs]
R = 1
[pedestal]
center = 6 4 2
volume = 2 2 2
[door]
pos = 1 3 1
target_parts = output
[player]
spawn = 10 3 1
facing = -x
"#;

/// Unsolvable: output threshold 2 but only one input pipe can ever feed it, so
/// the sum never reaches 2 — no switch combo wins.
const UNSOLVABLE: &str = r#"
[meta]
name = fixture-unsolvable
[room]
size = 12 8 4
ambient = trace-pattern
[node R]
pos = 3,3,2
glyph = R
color = #FF0000
is_input = yes
[node output]
pos = 9,3,2
glyph = O
color = #FFFFFF
is_output = yes
threshold = 2
[pipe R->output]
state = OFF
path = (3,3,2) (9,3,2)
switch_at = (6,3,2)
[inputs]
R = 1
[pedestal]
center = 6 4 2
volume = 2 2 2
[door]
pos = 11 3 1
target_parts = output
[player]
spawn = 8 3 1
facing = -x
"#;

/// Trivially solved at load: the one pipe is authored ON and the output
/// threshold is met without any toggle.
const TRIVIAL: &str = r#"
[meta]
name = fixture-trivial
[room]
size = 12 8 4
ambient = trace-pattern
[node R]
pos = 3,3,2
glyph = R
color = #FF0000
is_input = yes
[node output]
pos = 9,3,2
glyph = O
color = #FFFFFF
is_output = yes
threshold = 1
[pipe R->output]
state = ON
path = (3,3,2) (9,3,2)
switch_at = (6,3,2)
[inputs]
R = 1
[pedestal]
center = 6 4 2
volume = 2 2 2
[door]
pos = 11 3 1
target_parts = output
[player]
spawn = 8 3 1
facing = -x
"#;

/// An unreachable switch: the room is tall and the switch is buried near the
/// ceiling (z far above any standable floor), so no standing spot can reach it.
const UNREACHABLE_SWITCH: &str = r#"
[meta]
name = fixture-unreachable
[room]
size = 12 8 12
ambient = trace-pattern
[node R]
pos = 3,3,10
glyph = R
color = #FF0000
is_input = yes
[node output]
pos = 9,3,10
glyph = O
color = #FFFFFF
is_output = yes
threshold = 1
[pipe R->output]
state = OFF
path = (3,3,10) (9,3,10)
switch_at = (6,3,10)
[inputs]
R = 1
[pedestal]
center = 6 4 5
volume = 2 2 2
[door]
pos = 11 3 1
target_parts = output
[player]
spawn = 8 3 1
facing = -x
"#;

/// Spawn faces away from the puzzle: everything is to the WEST (-x) of spawn,
/// but the player faces +x (east, away).
const FACES_AWAY: &str = r#"
[meta]
name = fixture-faces-away
[room]
size = 12 8 4
ambient = trace-pattern
[node R]
pos = 2,3,2
glyph = R
color = #FF0000
is_input = yes
[node output]
pos = 4,3,2
glyph = O
color = #FFFFFF
is_output = yes
threshold = 1
[pipe R->output]
state = OFF
path = (2,3,2) (4,3,2)
switch_at = (3,3,2)
[inputs]
R = 1
[pedestal]
center = 6 4 2
volume = 2 2 2
[door]
pos = 1 3 1
target_parts = output
[player]
spawn = 9 3 1
facing = +x
"#;

fn load_fixture(src: &str) -> re_level::Level {
    load_string(src, "fixture").expect("fixture must parse")
}

fn report(src: &str) -> LevelReport {
    check_level(&load_fixture(src), "fixture")
}

fn check_named<'a>(r: &'a LevelReport, name: &str) -> &'a Check {
    r.checks.iter().find(|c| c.name == name).unwrap_or_else(|| {
        panic!(
            "no check named {name}; have {:?}",
            r.checks.iter().map(|c| c.name).collect::<Vec<_>>()
        )
    })
}

#[test]
fn solvable_fixture_is_solvable_one_toggle() {
    let r = report(SOLVABLE);
    assert!(
        matches!(r.solve, Some(SolveVerdict::Solvable { min_toggles: 1, .. })),
        "expected Solvable(min 1), got {:?}",
        r.solve
    );
    assert_eq!(check_named(&r, "solvable").severity, Severity::Pass);
    assert!(!r.failed(), "a clean solvable level must not FAIL");
}

#[test]
fn unsolvable_fixture_fails() {
    let r = report(UNSOLVABLE);
    assert_eq!(r.solve, Some(SolveVerdict::Unsolvable));
    assert_eq!(check_named(&r, "solvable").severity, Severity::Fail);
    assert!(r.failed(), "an unsolvable level must FAIL");
}

#[test]
fn trivial_fixture_warns() {
    let r = report(TRIVIAL);
    assert_eq!(r.solve, Some(SolveVerdict::TriviallySolved));
    assert_eq!(check_named(&r, "solvable").severity, Severity::Warn);
}

#[test]
fn unreachable_switch_is_flagged() {
    let r = report(UNREACHABLE_SWITCH);
    assert_eq!(
        r.reachable,
        Some(false),
        "buried switch must be unreachable"
    );
    assert_eq!(check_named(&r, "reachable").severity, Severity::Fail);
    assert!(
        check_named(&r, "reachable").detail.contains("6, 3, 10")
            || check_named(&r, "reachable").detail.contains("(6, 3, 10)"),
        "the unreachable switch cell should be named: {}",
        check_named(&r, "reachable").detail
    );
}

#[test]
fn spawn_facing_away_is_warned() {
    let r = report(FACES_AWAY);
    assert_eq!(check_named(&r, "spawn-facing").severity, Severity::Warn);
    assert_eq!(
        r.coherent,
        Some(false),
        "faces-away level needs layout work"
    );
    assert!(r.verdict.contains("NEEDS LAYOUT WORK"));
    assert!(r.verdict.to_lowercase().contains("away"));
}

#[test]
fn inventory_counts_elements() {
    let r = report(SOLVABLE);
    let inv = r.inventory.expect("inventory");
    assert_eq!(inv.nodes_total, 2);
    assert_eq!(inv.inputs, 1);
    assert_eq!(inv.outputs, 1);
    assert_eq!(inv.switches, 1);
    assert_eq!(inv.pipes, 1);
    assert!(inv.has_door && inv.has_player && inv.has_pedestal);
}

#[test]
fn dag_check_passes_on_acyclic_fixture() {
    let r = report(SOLVABLE);
    assert_eq!(check_named(&r, "dag").severity, Severity::Pass);
    assert_eq!(check_named(&r, "edges").severity, Severity::Pass);
    assert_eq!(check_named(&r, "door-targets").severity, Severity::Pass);
}

#[test]
fn missing_file_yields_load_fail_not_panic() {
    let r = check_path("definitely/does/not/exist.lvl");
    assert!(r.failed());
    assert_eq!(check_named(&r, "load").severity, Severity::Fail);
}

#[test]
fn solvable_fixture_is_coherent() {
    let r = report(SOLVABLE);
    assert_eq!(r.coherent, Some(true));
    assert_eq!(r.verdict, "COHERENT");
}

// ═══════════════════ `[note]` placard placement (W1 1.7) ════════════════════
//
// The room is 12×8×4 `trace-pattern` ambient, so the perimeter (x=0, x=11, y=0,
// y=7) is masonry at every z and the interior is air. The rule mirrors the
// shipped binder as of commit 63be293: the authored z selects the STOREY and the
// placard is bound at the reader's EYE LAYER (here z=1), sliding along the wall
// run when the exact cell is claimed/tagged/blocked. See lib.rs section 6.

/// `SOLVABLE` plus one `[note]` block, so each case differs only in its placard.
fn with_note(note: &str) -> LevelReport {
    let src = format!("{SOLVABLE}\n{note}\n");
    check_level(&load_fixture(&src), "fixture")
}

/// A bare [`re_level::LevelNote`] for driving the resolver directly.
fn lvl_note(x: i32, y: i32, z: i32, facing: i32) -> re_level::LevelNote {
    re_level::LevelNote {
        kind: re_level::NoteKind::Researcher,
        x,
        y,
        z,
        facing,
        text: "t".to_string(),
    }
}

/// An `sx × sy × sz` map: solid perimeter columns on every layer, floor at z=0,
/// air inside (the same fixture `re_game::level_text`'s tests use).
fn walled_box(sx: usize, sy: usize, sz: usize) -> Map {
    let mut m = Map::new(sx, sy, sz);
    for x in 0..sx {
        for y in 0..sy {
            for z in 0..sz {
                let perim = x == 0 || y == 0 || x == sx - 1 || y == sy - 1;
                if perim {
                    m.set(x, y, z, CellKind::WallSolid);
                } else if z == 0 {
                    m.set(x, y, z, CellKind::Floor);
                }
            }
        }
    }
    m
}

/// The default storey height every `.lvl` without an explicit key gets.
const H: u32 = re_level::DEFAULT_STOREY_HEIGHT;

#[test]
fn a_note_on_a_wall_cell_passes() {
    // (11,3,2) IS the +x perimeter wall; facing -x paints its inward face. The
    // binder z-snaps it to the eye layer (z=1) of the same wall column — still
    // "authored exactly" per the pinned semantics (z selects the storey).
    let r = with_note("[note researcher pos=11,3,2 facing=-x]\n    hand-written\n");
    let c = check_named(&r, "notes");
    assert_eq!(c.severity, Severity::Pass, "{}", c.detail);
    assert!(
        c.detail.contains("name their wall cell directly"),
        "{}",
        c.detail
    );
}

#[test]
fn a_note_in_open_air_warns_but_still_resolves() {
    // (5,3,2) is air; the reader faces +x, so the wall column is (11,3), 6 cells
    // along the gaze, bound at the eye layer z=1.
    let r = with_note("[note system_log pos=5,3,2 facing=-x]\n    > log line\n");
    let c = check_named(&r, "notes");
    assert_eq!(c.severity, Severity::Warn, "{}", c.detail);
    assert!(c.detail.contains("along the reader's gaze"), "{}", c.detail);
    assert!(
        c.detail.contains("(11, 3, 1)"),
        "the resolved eye-layer cell is named: {}",
        c.detail
    );
    assert!(!r.failed(), "a tolerated placard is not a FAILURE");
}

#[test]
fn a_note_that_resolves_to_no_wall_fails() {
    // A closed room's perimeter catches every in-bounds gaze, and the eye-layer
    // fallback rescues a bad z — so the honest binder-unplaceable case is a `pos`
    // OFF THE MAP'S SIDE whose gaze walks further out: the search leaves the map
    // on its first step in the authored plane AND in the eye layer.
    let r = with_note("[note model_thought pos=20,3,2 facing=+x]\n    (nowhere)\n");
    let c = check_named(&r, "notes");
    assert_eq!(c.severity, Severity::Fail, "{}", c.detail);
    assert!(c.detail.contains("CANNOT render"), "{}", c.detail);
    assert!(r.failed(), "an unplaceable placard must FAIL the level");
}

#[test]
fn two_notes_aiming_at_one_cell_slide_apart() {
    // Both look +x down row y=3 and resolve to the (11,3) column at the eye
    // layer; the binder keeps the first and slides the second along the wall run
    // (63be293 semantics — this used to be a false "CANNOT render" FAIL).
    let r = with_note(
        "[note researcher pos=4,3,2 facing=-x]\n    first\n\n[note system_log pos=6,3,2 facing=-x]\n    second\n",
    );
    let c = check_named(&r, "notes");
    assert_eq!(c.severity, Severity::Warn, "{}", c.detail);
    assert!(c.detail.contains("slid"), "{}", c.detail);
    assert!(!r.failed(), "a slid placard renders — not a FAILURE");
}

/// The L9–L12 authoring pattern: placards stacked on z at one (x,y) collapse to
/// ONE eye layer and hang side by side instead of fighting over a cell.
#[test]
fn stacked_placards_collapse_to_the_eye_layer_and_slide_apart() {
    let r = with_note(
        "[note researcher pos=4,3,2 facing=-x]\n    first\n\n[note system_log pos=4,3,3 facing=-x]\n    second\n",
    );
    let c = check_named(&r, "notes");
    assert_eq!(c.severity, Severity::Warn, "{}", c.detail);
    assert!(c.detail.contains("slid"), "{}", c.detail);
    assert!(!r.failed());
}

// ── the ported binder helpers, pinned against `re_game::level_text` ──────────

/// [`reader_eye_layer`] mirrors the shipped binder's storey→layer arithmetic
/// (`re_game::level_text::tests` runs the same cases against the real
/// `re_render::gameplay_eye_ref_z`).
#[test]
fn the_reader_eye_layer_mirrors_the_shipped_binder() {
    // sz=6 ⇒ habitable band [1,4]; every ground-storey authored z reads at 1.
    let map = walled_box(8, 8, 6);
    for z in [1, 2, 3, 4] {
        assert_eq!(reader_eye_layer(&map, z, H), 1, "authored z={z}");
    }
    // A two-storey shell (L1's shape): sz=13, H=4 ⇒ eye layers 1, 5, 9.
    let tall = walled_box(8, 8, 13);
    assert_eq!(reader_eye_layer(&tall, 5, H), 5, "storey 1 is read at z=5");
    assert_eq!(
        reader_eye_layer(&tall, 4, H),
        5,
        "z=4 is storey 1's floor slab"
    );
    assert_eq!(reader_eye_layer(&tall, 9, H), 9, "storey 2 is read at z=9");
    assert_eq!(
        reader_eye_layer(&tall, 3, H),
        1,
        "z=3 is still the ground storey"
    );
    // A storey above the habitable band falls back to the storey below; the
    // degenerate cases never panic and never leave the band.
    let short = walled_box(8, 8, 5); // habitable band [1,3]
    for z in [2, 3, 4, 7, 99] {
        assert_eq!(reader_eye_layer(&short, z, H), 1, "authored z={z}");
    }
    assert_eq!(reader_eye_layer(&short, -5, H), 1);
    let degenerate = reader_eye_layer(&short, 2, 0);
    assert!(
        (1..=3).contains(&degenerate),
        "H=0 must stay in the band, got {degenerate}"
    );
}

/// The wall the author pointed at can be a DOOR at the eye layer even though it
/// was masonry at the authored z (`06.lvl`'s east wall) — the placard slides
/// along the run instead of reporting unplaceable.
#[test]
fn a_placard_slides_past_a_door_cell_at_the_eye_layer() {
    let mut map = walled_box(8, 8, 6);
    map.set_cell(7, 3, 1, CellKind::Object, 0); // the door anchor, at eye level
    let n = lvl_note(3, 3, 2, 360 /* -x */);
    let (placement, cell) = resolve_note_wall(&map, &n, facing_step(360), H, &[]);
    assert_eq!(cell, Some((7, 4, 1)), "the placard hangs BESIDE the door");
    assert!(
        matches!(placement, NotePlacement::Tolerated { slid: 1, .. }),
        "{placement:?}"
    );
}

/// A face with a solid cell in FRONT of it can never be seen (`04.lvl`: the wall
/// cell directly behind the door anchor) — the placard slides to a viewable face.
#[test]
fn a_placard_skips_a_face_blocked_by_the_cell_in_front_of_it() {
    let mut map = walled_box(8, 8, 6);
    map.set_cell(6, 3, 1, CellKind::Object, 0); // object in front of the +x wall
    assert!(!face_is_visible(&map, 7, 3, 1, (-1, 0)));
    assert!(face_is_visible(&map, 7, 4, 1, (-1, 0)));
    let n = lvl_note(3, 3, 2, 360 /* -x */);
    let (_, cell) = resolve_note_wall(&map, &n, facing_step(360), H, &[]);
    assert_eq!(
        cell,
        Some((7, 4, 1)),
        "no placard on a face nobody can stand before"
    );
}

/// The slide is bounded: a wall run fully tagged out within
/// [`NOTE_WALL_SLIDE_MAX`] in both planes is binder-unplaceable.
#[test]
fn a_fully_blocked_wall_run_is_binder_unplaceable() {
    let mut map = walled_box(16, 16, 6);
    for y in 0..16 {
        map.set_cell(15, y, 1, CellKind::WallSolid, 9); // authored colour id ⇒ not paintable
    }
    let n = lvl_note(3, 3, 2, 360 /* -x */);
    let (placement, cell) = resolve_note_wall(&map, &n, facing_step(360), H, &[]);
    assert_eq!(placement, NotePlacement::Unplaced);
    assert_eq!(cell, None);
}

#[test]
fn a_level_with_no_notes_passes_the_note_rule() {
    let r = report(SOLVABLE);
    let c = check_named(&r, "notes");
    assert_eq!(c.severity, Severity::Pass);
    assert_eq!(c.detail, "no [note] placards authored");
}

/// The replicated facing→step mapping MUST match `re_core::FaceDir::from_yaw_col`
/// (`+y = 0`, `+x = 120`, `−y = 240`, `−x = 360`, off-cardinals snapping to the
/// nearest face, out-of-range wrapping). This is the pin between the two copies.
#[test]
fn facing_steps_match_the_engine_yaw_encoding() {
    assert_eq!(facing_step(0), (0, 1));
    assert_eq!(facing_step(120), (1, 0));
    assert_eq!(facing_step(240), (0, -1));
    assert_eq!(facing_step(360), (-1, 0));
    // Off-cardinal snaps to the nearest face; out-of-range wraps.
    assert_eq!(facing_step(130), (1, 0));
    assert_eq!(facing_step(-120), (-1, 0));
    assert_eq!(facing_step(480), (0, 1));
    // Tokens track the steps.
    assert_eq!(facing_token(0), "+y");
    assert_eq!(facing_token(120), "+x");
    assert_eq!(facing_token(240), "-y");
    assert_eq!(facing_token(360), "-x");
}

/// A placard may never claim a TAGGED cell (a pipe tag or an authored colour id)
/// nor a puzzle `Object` cell — the search walks past them. `SOLVABLE`'s switch
/// sits at (6,3,2) on the pipe path, so a note aimed down that row must skip it
/// and land on the masonry beyond.
#[test]
fn the_search_walks_past_tagged_and_object_cells() {
    let r = with_note("[note researcher pos=2,3,2 facing=-x]\n    past the switch\n");
    let c = check_named(&r, "notes");
    assert_ne!(c.severity, Severity::Fail, "{}", c.detail);
    assert!(
        c.detail.contains("(11, 3, 1)"),
        "the placard must land on the far wall at the eye layer, not the pipe/switch cell: {}",
        c.detail
    );
}

// ═══════════════ threshold sanity (dead / always-on nodes) — W2 2.2 ═════════
//
// The rule: for every NON-INPUT node, `0 < threshold <= in_degree`, OR the node is
// DEAD BY DESIGN (`threshold > in_degree` and it is not a door target). See the
// `check_thresholds` section comment in lib.rs for why `threshold == 0` is a WARN
// (it is the parser default) while a NEGATIVE threshold is a FAIL.

/// A three-node fixture whose hidden node's threshold and door wiring the caller
/// controls: `hidden` takes `in_deg` incoming pipes (1 or 2) and the door targets
/// either `output` (the normal case) or `hidden` itself.
fn threshold_fixture(hidden_threshold: i32, in_deg: usize, door_target: &str) -> LevelReport {
    let second_input = if in_deg >= 2 {
        "\n[node S]\npos = 3,5,2\nglyph = S\ncolor = #00FF00\nis_input = yes\n"
    } else {
        ""
    };
    let second_pipe = if in_deg >= 2 {
        "\n[pipe S->hidden]\nstate = OFF\npath = (3,5,2) (6,5,2) (6,3,2)\nswitch_at = (4,5,2)\n"
    } else {
        ""
    };
    let second_binding = if in_deg >= 2 { "\nS = 1" } else { "" };
    let src = format!(
        r#"
[meta]
name = fixture-threshold
[room]
size = 12 8 4
ambient = trace-pattern
[node R]
pos = 3,3,2
glyph = R
color = #FF0000
is_input = yes{second_input}
[node hidden]
pos = 6,3,2
glyph = H
color = #FFD040
threshold = {hidden_threshold}
[node output]
pos = 9,3,2
glyph = O
color = #FFFFFF
is_output = yes
threshold = 1
[pipe R->hidden]
state = OFF
path = (3,3,2) (6,3,2)
switch_at = (4,3,2){second_pipe}
[pipe hidden->output]
state = OFF
path = (6,3,2) (9,3,2)
switch_at = (8,3,2)
[inputs]
R = 1{second_binding}
[pedestal]
center = 6 4 2
volume = 2 2 2
[door]
pos = 11 3 1
target_parts = {door_target}
[player]
spawn = 10 3 1
facing = -x
"#
    );
    check_level(&load_fixture(&src), "fixture")
}

/// The ordinary case: a satisfiable threshold passes and says so.
#[test]
fn a_satisfiable_threshold_passes() {
    let r = threshold_fixture(1, 1, "output");
    let c = check_named(&r, "thresholds");
    assert_eq!(c.severity, Severity::Pass, "{}", c.detail);
    assert!(
        c.detail.contains("every node satisfies"),
        "a clean level must not be reported as dead-by-design: {}",
        c.detail
    );
    // threshold == in_degree is the boundary and must be INSIDE the rule.
    let boundary = threshold_fixture(2, 2, "output");
    let c = check_named(&boundary, "thresholds");
    assert_eq!(c.severity, Severity::Pass, "{}", c.detail);
}

/// `threshold > in_degree` on a node the door does NOT target is the L8 mechanic:
/// PASS, but named in the report so a reviewer sees the scar declared.
#[test]
fn an_unreachable_threshold_off_the_win_path_is_dead_by_design() {
    let r = threshold_fixture(99, 1, "output");
    let c = check_named(&r, "thresholds");
    assert_eq!(
        c.severity,
        Severity::Pass,
        "a permanently-unreachable node the win does not need is route-around content: {}",
        c.detail
    );
    assert!(c.detail.contains("DEAD BY DESIGN"), "{}", c.detail);
    assert!(
        c.detail.contains("hidden"),
        "the node must be named: {}",
        c.detail
    );
    assert!(c.detail.contains("1 dead-by-design"), "{}", c.detail);
}

/// The same unreachable threshold on a node the DOOR TARGETS is unwinnable: FAIL.
#[test]
fn an_unreachable_threshold_on_a_door_target_fails() {
    let r = threshold_fixture(99, 1, "hidden");
    let c = check_named(&r, "thresholds");
    assert_eq!(c.severity, Severity::Fail, "{}", c.detail);
    assert!(c.detail.contains("can never be won"), "{}", c.detail);
    // And the solvability sweep agrees — the two checks corroborate each other.
    assert_eq!(r.solve, Some(SolveVerdict::Unsolvable));
}

/// `threshold == 0` fires on an empty sum. It is also the parser default, so it is
/// a WARN that names the node rather than a FAIL that reddens every demo level.
#[test]
fn a_zero_threshold_warns_because_it_fires_unconditionally() {
    let r = threshold_fixture(0, 1, "output");
    let c = check_named(&r, "thresholds");
    assert_eq!(c.severity, Severity::Warn, "{}", c.detail);
    assert!(c.detail.contains("parser default"), "{}", c.detail);
    assert!(c.detail.contains("hidden"), "{}", c.detail);
}

/// A NEGATIVE threshold is never a default and never intentional: FAIL.
#[test]
fn a_negative_threshold_fails() {
    let r = threshold_fixture(-1, 1, "output");
    let c = check_named(&r, "thresholds");
    assert_eq!(c.severity, Severity::Fail, "{}", c.detail);
    assert!(c.detail.contains("negative"), "{}", c.detail);
}

/// An INPUT node's threshold is unused by the forward pass (its activation comes
/// from `[inputs]`), so the rule reports it and never judges it — even at 0.
#[test]
fn input_node_thresholds_are_reported_not_judged() {
    let r = report(SOLVABLE);
    let c = check_named(&r, "thresholds");
    assert_eq!(c.severity, Severity::Pass, "{}", c.detail);
    assert!(
        c.detail.contains("1 input node(s) skipped"),
        "the skipped inputs must be counted honestly: {}",
        c.detail
    );
}

/// A level with no nodes at all does not panic (the inventory check owns that FAIL).
#[test]
fn a_nodeless_level_does_not_panic_on_the_threshold_rule() {
    let level = re_level::Level::default();
    let c = super::check_thresholds(&level);
    assert_eq!(c.severity, Severity::Pass);
    assert!(c.detail.contains("no nodes"), "{}", c.detail);
}

// ---------------------------------------------------------------------------
// 8. power_budget — the simultaneously-ON pipe cap
// ---------------------------------------------------------------------------

/// A two-input level that genuinely needs BOTH pipes hot (`output` threshold 2).
/// `budget` is injected into `[room]`; `0` means the key is omitted entirely.
fn budget_fixture(budget: u32) -> LevelReport {
    let line = if budget == 0 {
        String::new()
    } else {
        format!(
            "power_budget = {budget}
"
        )
    };
    let src = format!(
        r#"
[meta]
name = fixture-budget
[room]
size = 14 8 4
ambient = trace-pattern
{line}[node A]
pos = 3,3,2
is_input = yes
[node B]
pos = 3,5,2
is_input = yes
[node output]
pos = 11,4,2
is_output = yes
threshold = 2
[pipe A->output]
state = OFF
path = (3,3,2) (11,4,2)
switch_at = (6,3,2)
[pipe B->output]
state = OFF
path = (3,5,2) (11,4,2)
switch_at = (6,5,2)
[inputs]
A = 1
B = 1
[pedestal]
center = 11 4 1
volume = 1 1 1
[door]
pos = 1 4 1
target_parts = output
[player]
spawn = 12 4 1
facing = -x
"#
    );
    report(&src)
}

/// No `power_budget` key ⇒ PASS, explicitly reported as UNLIMITED. This is the
/// state every non-campaign level is in and must stay in.
#[test]
fn an_absent_power_budget_reports_unlimited() {
    let r = budget_fixture(0);
    let c = check_named(&r, "power_budget");
    assert_eq!(c.severity, Severity::Pass, "{}", c.detail);
    assert!(c.detail.contains("UNLIMITED"), "{}", c.detail);
}

/// A budget BELOW the level's true need is the catastrophic case: FAIL, and say
/// the word "UNWINNABLE" so it cannot be skimmed past.
#[test]
fn a_budget_one_too_low_fails_as_unwinnable() {
    let r = budget_fixture(1);
    let c = check_named(&r, "power_budget");
    assert_eq!(c.severity, Severity::Fail, "{}", c.detail);
    assert!(c.detail.contains("UNWINNABLE"), "{}", c.detail);
}

/// A budget at the level's true minimum is a PASS that reports zero slack.
#[test]
fn a_budget_at_the_true_minimum_passes_with_no_slack() {
    // 2 pipes, both required: the cap equals the pipe count, so it cannot bind
    // and the linter says so (WARN) rather than pretending the level is tight.
    let r = budget_fixture(2);
    let c = check_named(&r, "power_budget");
    assert_eq!(c.severity, Severity::Warn, "{}", c.detail);
    assert!(c.detail.contains("can never bind"), "{}", c.detail);
}

// The twelve-campaign-levels budget sweep moved to the host game's own test
// suite (`every_shipped_campaign_level_passes_the_power_budget_rule`): it
// asserts a property of the campaign ("all twelve declare a budget"), not a
// generic engine feature, so it moves with the campaign content it names.

// ---------------------------------------------------------------------------
// unknown_keys — the parser's silent-discard diagnostic, surfaced
// ---------------------------------------------------------------------------

/// An unrecognized key is reported as a WARN naming the section, line and key —
/// never a FAIL, so a stray key can never make a hand-authored level unloadable
/// or block a save. This is the surface that puts `re_level`'s
/// [`re_level::UnknownKey`] diagnostics in front of a human: the editor's
/// validation drawer on every save, and `lvlcheck` in CI.
#[test]
fn unknown_keys_are_reported_as_a_warn_not_a_fail() {
    let mut level = re_level::load_string(SOLVABLE, "<uk>").expect("fixture parses");
    // Clean fixture ⇒ PASS.
    let clean = check_level(&level, "<uk>");
    let c = check_named(&clean, "unknown_keys");
    assert_eq!(
        c.severity,
        Severity::Pass,
        "a clean level passes: {}",
        c.detail
    );

    // Inject what the parser would have collected for `[room] wdith = 6`.
    level.unknown_keys.push(re_level::UnknownKey {
        line: 7,
        section: "room",
        key: "wdith".into(),
    });
    let r = check_level(&level, "<uk>");
    let c = check_named(&r, "unknown_keys");
    assert_eq!(c.severity, Severity::Warn, "visible, but never fatal");
    assert!(
        c.detail.contains("line 7: [room] unknown key 'wdith'"),
        "the detail names section, line and key: {}",
        c.detail
    );
    assert_ne!(
        r.worst(),
        Severity::Fail,
        "an unknown key alone must never sink the report to FAIL"
    );
}

/// End-to-end on the real corpus: a mistyped key in a real level's text shows up
/// in that level's report.
#[test]
fn a_mistyped_key_in_real_level_text_reaches_the_report() {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../levels/01.lvl");
    let raw = std::fs::read_to_string(&path)
        .expect("read 01.lvl")
        .replace("\r\n", "\n")
        .replace("ambient     = trace-pattern", "ambeint     = trace-pattern");
    let level = re_level::load_string(&raw, "01-typo").expect("a typo must not fail the load");
    let r = check_level(&level, "01-typo");
    let c = check_named(&r, "unknown_keys");
    assert_eq!(c.severity, Severity::Warn);
    assert!(
        c.detail.contains("[room] unknown key 'ambeint'"),
        "the typo is named: {}",
        c.detail
    );
}

// ───────────────── 10. `[emit]` reward footprint vs the pipe network ─────────

/// A fixture whose `[emit]` reward sits on the pedestal with a straight pipe
/// running THROUGH it. Pedestal centre (6,4,1) extent 3×3×2 ⇒ the part anchors at
/// x 5..7, y 3..5, z = 1 + `offset z`; the `.#. / ### / .#.` plus authors its
/// centre row at (5,4,2) (6,4,2) (7,4,2), and `R->output` runs the whole y=4 line
/// at z=2. Three of the five cells are eaten.
const EMIT_PIPE_COLLISION: &str = r"
[meta]
name = fixture-emit-collision
[room]
size = 14 10 5
ambient = trace-pattern
render_pipes = true
[node R]
pos = 2,4,2
glyph = R
color = #FF0000
is_input = yes
[node output]
pos = 11,4,2
glyph = O
color = #FFFFFF
is_output = yes
threshold = 1
[pipe R->output]
state = OFF
path = (2,4,2) (11,4,2)
switch_at = (4,4,2)
[part plus]
size = 3 3 1
mask =
.#.
###
.#.
color = #FFD040
offset = 0 0 1
[emit output=1]
parts = plus
[inputs]
R = 1
[pedestal]
center = 6 4 1
volume = 3 3 2
[door]
pos = 12 4 1
target_parts = output
[player]
spawn = 11 4 1
facing = -x
";

/// The same fixture with the pipe routed NORTH around the pedestal — the fix
/// shape this rule asks an author for. Nothing else changes.
fn emit_pipe_routed_clear() -> String {
    EMIT_PIPE_COLLISION.replace(
        "path = (2,4,2) (11,4,2)\nswitch_at = (4,4,2)",
        "path = (2,4,2) (4,4,2) (4,1,2) (9,1,2) (9,4,2) (11,4,2)\nswitch_at = (4,4,2)",
    )
}

#[test]
fn an_emit_part_crossed_by_a_pipe_fails_and_names_every_lost_cell() {
    let level = load_string(EMIT_PIPE_COLLISION, "<emit-collision>").expect("fixture parses");
    let r = check_level(&level, "<emit-collision>");
    let c = check_named(&r, "emit_parts");
    assert_eq!(
        c.severity,
        Severity::Fail,
        "a reward eaten by the wiring is a hard content defect: {}",
        c.detail
    );
    for cell in ["(5, 4, 2)", "(6, 4, 2)", "(7, 4, 2)"] {
        assert!(
            c.detail.contains(cell),
            "the lost cell {cell} must be named: {}",
            c.detail
        );
    }
    assert!(
        c.detail.contains("[pipe R->output]"),
        "the pipe that ate them must be named: {}",
        c.detail
    );
    assert!(
        c.detail.contains("binds 2"),
        "the report states how few cells actually bind: {}",
        c.detail
    );
}

/// `emit_part_losses` is the public payload behind the check — it reports the
/// same three cells, each attributed to the pipe index that owns it.
#[test]
fn emit_part_losses_attributes_each_cell_to_its_pipe() {
    let level = load_string(EMIT_PIPE_COLLISION, "<emit-collision>").expect("fixture parses");
    let losses = emit_part_losses(&level);
    assert_eq!(losses.len(), 3, "three cells lost: {losses:?}");
    for l in &losses {
        assert_eq!(l.clause, 0);
        assert_eq!(l.part, "plus");
        assert_eq!(l.pipe, Some(0), "attributed to the level's only pipe");
    }
    let cells: Vec<_> = losses.iter().map(|l| l.cell).collect();
    assert_eq!(cells, vec![(5, 4, 2), (6, 4, 2), (7, 4, 2)]);
}

/// Routing the pipe around the pedestal clears the rule — and nothing else about
/// the level had to move.
#[test]
fn routing_the_pipe_around_the_pedestal_clears_the_rule() {
    let src = emit_pipe_routed_clear();
    let level = load_string(&src, "<emit-clear>").expect("fixture parses");
    assert!(
        emit_part_losses(&level).is_empty(),
        "no cell is lost once the pipe detours: {:?}",
        emit_part_losses(&level)
    );
    let report = check_level(&level, "<emit-clear>");
    let c = check_named(&report, "emit_parts");
    assert_eq!(c.severity, Severity::Pass, "{}", c.detail);
    assert!(
        c.detail.contains("all 5 cell(s)"),
        "the PASS line states the authored count: {}",
        c.detail
    );
}

/// A part cell that lands on ordinary geometry (not wiring) is a WARN, not a
/// FAIL: authors do legitimately tuck a figure against a wall, and the shipped
/// L9/L10/L11 beacons overlap their own pedestal node. It is still NAMED.
#[test]
fn a_part_cell_buried_in_geometry_warns_rather_than_fails() {
    // Move the part one layer DOWN so its centre lands on the floor slab / node
    // layer instead of the pipe layer.
    let src = emit_pipe_routed_clear().replace("offset = 0 0 1", "offset = 0 0 -1");
    let level = load_string(&src, "<emit-buried>").expect("fixture parses");
    let report = check_level(&level, "<emit-buried>");
    let c = check_named(&report, "emit_parts");
    assert_eq!(c.severity, Severity::Warn, "{}", c.detail);
    assert!(
        c.detail.contains("cannot bind"),
        "the buried cells are named: {}",
        c.detail
    );
}

/// A level with no `[emit]` clause passes trivially — the rule must never invent
/// a finding for the dozens of technique-demo levels that author no reward.
#[test]
fn a_level_without_an_emit_clause_passes_the_rule() {
    let level = load_string(SOLVABLE, "<no-emit>").expect("fixture parses");
    let report = check_level(&level, "<no-emit>");
    let c = check_named(&report, "emit_parts");
    assert_eq!(c.severity, Severity::Pass);
    assert!(c.detail.contains("no [emit] reward authored"));
}

// ---------------------------------------------------------------------------
// assets — every file a level names must actually be there
// ---------------------------------------------------------------------------
//
// The gate that catches a REMOVED asset across the whole corpus at once. Before
// it, `re_level` resolved every `[texture]`/`[sky]`/`[vox]`/`[inset]` path,
// reported a miss with the full candidate list, and no caller anywhere read the
// answer — so deleting a shipped texture produced an untextured level and total
// silence.

/// The shipped `levels/` directory (compile-time anchored, not cwd-relative).
fn levels_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../levels")
}

/// Load a shipped level FROM DISK, so it carries the `source_path` that anchors
/// its asset resolution (a string-parsed level deliberately cannot FAIL — see
/// `a_level_with_no_source_path_warns_instead_of_failing`).
fn shipped(name: &str) -> re_level::Level {
    re_level::load(levels_dir().join(name)).unwrap_or_else(|e| panic!("{name} must load: {e}"))
}

/// THE GATE. A texture the level names but that is not on disk FAILs the level,
/// and the finding names the section, the authored path, and every candidate
/// path that was probed — the message that turns "the wall looks wrong" into
/// "it is looking in these three places and the file is in none of them".
#[test]
fn a_missing_texture_file_fails_the_level() {
    let mut level = shipped("TX5-genplate.lvl");
    let clean = check_level(&level, "TX5-genplate.lvl");
    assert_eq!(
        check_named(&clean, "assets").severity,
        Severity::Pass,
        "precondition: the shipped level's texture is present"
    );

    level.textures[0].file = "assets/texture/genplate_tex_REMOVED.png".into();
    let r = check_level(&level, "TX5-genplate.lvl");
    let c = check_named(&r, "assets");
    assert_eq!(c.severity, Severity::Fail, "{}", c.detail);
    assert!(r.failed(), "a missing asset sinks the level's verdict");
    assert!(c.detail.contains("[texture genplate]"), "{}", c.detail);
    assert!(
        c.detail.contains("genplate_tex_REMOVED.png"),
        "{}",
        c.detail
    );
    assert!(c.detail.contains("looked in"), "{}", c.detail);
    // All three anchors are named: beside the level file, beside the level tree,
    // and the working directory.
    assert!(
        c.detail.matches("genplate_tex_REMOVED.png").count() >= 4,
        "every candidate path is listed: {}",
        c.detail
    );
}

/// EVERY broken reference is reported in one pass. `build_map_checked` — the old
/// (uncalled) surface — `?`-propagates and reports only the first, so an author
/// would fix one texture and be failed again by the next.
///
/// Uses a healthy, asset-free-of-sky/vox level (`TX5-genplate.lvl`) and injects a
/// sky + a vox reference that are not authored in the file at all — the checker
/// reads the in-memory `Level`, not the source text, so this exercises the same
/// multi-miss reporting a deleted fixture level used to (removed along with
/// the unlicensed image sky it authored).
#[test]
fn every_broken_reference_is_listed_not_just_the_first() {
    let mut level = shipped("TX5-genplate.lvl");
    level.sky.cylinder = Some("assets/texture/gone_cylinder.png".into());
    level.sky.cap_top = Some("assets/texture/gone_cap_top.png".into());
    level.voxes.push(re_level::LevelVox {
        name: "gone".into(),
        file: "levels/GoneBuilding.vox".into(),
        ..Default::default()
    });
    let r = check_level(&level, "TX5-genplate.lvl");
    let c = check_named(&r, "assets");
    assert_eq!(c.severity, Severity::Fail, "{}", c.detail);
    assert!(
        c.detail.starts_with("3 of "),
        "the count leads: {}",
        c.detail
    );
    for expected in [
        "[sky] cylinder",
        "gone_cylinder.png",
        "[sky] cap_top",
        "gone_cap_top.png",
        "[vox ",
        "GoneBuilding.vox",
    ] {
        assert!(
            c.detail.contains(expected),
            "{expected} missing from: {}",
            c.detail
        );
    }
    // The one error `build_map_checked` would have reported instead.
    let single = level.build_map_checked().expect_err("it reports something");
    assert!(
        !single.contains("GoneBuilding.vox"),
        "the first-error-only surface never reaches the vox: {single}"
    );
}

/// A file that EXISTS but is not a valid image is caught too — the linter
/// decodes (`audit_assets`), which is the cost it exists to pay. The game's load
/// path deliberately does not (`re_game::asset_notice`).
#[test]
fn a_present_but_undecodable_image_fails() {
    let mut level = shipped("TX5-genplate.lvl");
    // A real file that is certainly not a PNG.
    level.textures[0].file = "levels/TX5-genplate.lvl".into();
    let r = check_level(&level, "TX5-genplate.lvl");
    let c = check_named(&r, "assets");
    assert_eq!(c.severity, Severity::Fail, "{}", c.detail);
    assert!(c.detail.contains("undecodable"), "{}", c.detail);
}

/// A level parsed from a STRING has no file to anchor to, so "not found" means
/// "cannot tell yet". It is reported — and never fatal: a unit-test fixture or a
/// brand-new unsaved editor document must not be sunk to FAIL by it.
#[test]
fn a_level_with_no_source_path_warns_instead_of_failing() {
    let src = SOLVABLE.replace(
        "[room]",
        "[texture pic]\nfile = no/such/texture.png\n[room]",
    );
    let level = load_string(&src, "<no-anchor>").expect("fixture parses");
    assert!(level.source_path.is_none(), "load_string leaves no anchor");
    let r = check_level(&level, "<no-anchor>");
    let c = check_named(&r, "assets");
    assert_eq!(c.severity, Severity::Warn, "{}", c.detail);
    assert!(c.detail.contains("no file behind it"), "{}", c.detail);
    assert_ne!(
        r.worst(),
        Severity::Fail,
        "an unanchored level must never FAIL on assets alone"
    );
}

/// A level that names no external file passes trivially — the rule must not
/// invent a finding for the dozens of levels with no `[texture]`/`[vox]`.
#[test]
fn a_level_with_no_asset_references_passes() {
    let level = load_string(SOLVABLE, "<no-assets>").expect("fixture parses");
    let r = check_level(&level, "<no-assets>");
    let c = check_named(&r, "assets");
    assert_eq!(c.severity, Severity::Pass);
    assert!(
        c.detail.contains("no external asset references"),
        "{}",
        c.detail
    );
}

/// THE CORPUS GATE: every shipped `.lvl` resolves and decodes every file it
/// names. This is the assertion that fires the moment an asset is deleted,
/// moved, or renamed — for all of them at once, which no per-level content test
/// can do.
#[test]
fn every_shipped_level_passes_the_asset_gate() {
    let mut seen = 0usize;
    let mut with_assets = 0usize;
    let mut failures: Vec<String> = Vec::new();
    for entry in std::fs::read_dir(levels_dir()).expect("levels/").flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("lvl") {
            continue;
        }
        let Ok(level) = re_level::load(&path) else {
            continue;
        };
        seen += 1;
        if !level.asset_refs().is_empty() {
            with_assets += 1;
        }
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let r = check_level(&level, &name);
        let c = check_named(&r, "assets");
        if c.severity == Severity::Fail {
            failures.push(format!("{name}: {}", c.detail));
        }
    }
    assert!(seen >= 60, "the whole corpus was walked (saw {seen})");
    // A provenance cleanup removed several levels that referenced third-party
    // assets without a clear licence, and a campaign level (`L8-cold-room`)
    // lives in the separate game repository and is covered by that repo's own
    // asset tests. The remaining engine-repo levels that DO name assets are
    // TX5-genplate, TX6-subcell-ladder, TX7-edge-glyphs, 22-lit-cellar and
    // A5-inset — engine example/showcase levels that still exercise the same
    // asset-reference code path this test checks.
    assert!(
        with_assets >= 5,
        "levels that DO name assets were exercised ({with_assets})"
    );
    assert!(
        failures.is_empty(),
        "{} level(s) reference an asset file that is not on disk:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

// ===========================================================================
// Locations — WHERE each finding is
// ===========================================================================
//
// The rule these tests pin is not "the field exists". It is:
//
//   a check that NAMES an offender in its detail line carries that offender's
//   coordinate, in the same order, and a check that names none carries none.
//
// One test per [`Location`] variant on a level whose broken cell / clause / line
// is known by construction, plus the ordering invariant and the `jump_targets`
// browse list.

/// Every cell a check points at, in report order.
fn cells_of(c: &Check) -> Vec<(i32, i32, i32)> {
    c.locations.iter().filter_map(Location::as_cell).collect()
}

// ---- Location::Cell -------------------------------------------------------

/// THE CELL CASE. `UNREACHABLE_SWITCH` buries its one switch at (6,3,10) in a
/// 12-tall room with no standing surface under it. The `reachable` FAIL must
/// point AT THAT CELL — not at the pipe, not at the level.
#[test]
fn an_unreachable_switch_is_located_at_the_switch_cell() {
    let r = report(UNREACHABLE_SWITCH);
    let c = check_named(&r, "reachable");
    assert_eq!(c.severity, Severity::Fail, "{}", c.detail);
    assert_eq!(
        c.locations,
        vec![Location::Cell { x: 6, y: 3, z: 10 }],
        "the finding is AT the buried switch: {}",
        c.detail
    );
    assert_eq!(c.location(), c.locations.first(), "the Option view agrees");
}

/// The same fixture's `eye-level` coherence WARN is about the same cell and says
/// so, so a browser stepping through the report lands on it twice for two
/// different reasons rather than once with the reason lost.
#[test]
fn a_switch_buried_out_of_initial_reach_is_located_at_its_cell() {
    let r = report(UNREACHABLE_SWITCH);
    let c = check_named(&r, "eye-level");
    assert_eq!(c.severity, Severity::Warn, "{}", c.detail);
    assert_eq!(cells_of(c), vec![(6, 3, 10)]);
}

/// A reward eaten by the wiring reports EVERY lost cell, and the locations line
/// up 1:1 and IN ORDER with the cells the detail names.
#[test]
fn every_pipe_eaten_reward_cell_is_located() {
    let level = load_string(EMIT_PIPE_COLLISION, "<emit-collision>").expect("fixture parses");
    let r = check_level(&level, "<emit-collision>");
    let c = check_named(&r, "emit_parts");
    assert_eq!(c.severity, Severity::Fail, "{}", c.detail);
    assert_eq!(
        cells_of(c),
        vec![(5, 4, 2), (6, 4, 2), (7, 4, 2)],
        "one location per lost cell, in the detail's own order: {}",
        c.detail
    );
}

/// A spawn facing away from the puzzle is fixed AT THE SPAWN, so that is the
/// primary location; the element it should be looking at follows it.
#[test]
fn a_spawn_facing_away_is_located_at_the_spawn_then_its_target() {
    let r = report(FACES_AWAY);
    let c = check_named(&r, "spawn-facing");
    assert_eq!(c.severity, Severity::Warn, "{}", c.detail);
    // spawn = 9,3,1; of R (2,3,2), output (4,3,2) and the switch (3,3,2) the
    // nearest in xy is `output`.
    assert_eq!(cells_of(c), vec![(9, 3, 1), (4, 3, 2)]);
}

/// An orphan node is located at the node's own `pos` — and a node dangling at
/// BOTH ends is listed twice, once per reason, so detail entries and locations
/// stay in step.
#[test]
fn an_orphan_node_is_located_at_the_node_and_listed_once_per_reason() {
    let mut level = load_fixture(SOLVABLE);
    // Drop the only pipe: `output` loses its incoming edge (it is not an input)
    // and, being an output, needs no outgoing one — one reason. `R` is an input
    // so it needs no incoming edge, but it now has no outgoing one — one reason.
    level.pipes.clear();
    let r = check_level(&level, "<orphans>");
    let c = check_named(&r, "orphans");
    assert_eq!(c.severity, Severity::Warn, "{}", c.detail);
    // Node order is parse order: R (3,3,2) then output (9,3,2). R contributes the
    // "no outgoing" reason, output the "no incoming" one.
    assert_eq!(cells_of(c), vec![(3, 3, 2), (9, 3, 2)]);
    assert_eq!(
        c.locations.len(),
        c.detail.matches("no ").count(),
        "one location per named reason: {}",
        c.detail
    );
}

/// The door's two failure modes point at the DOOR, which is the clause an author
/// edits — a broken `target_parts` index names no node, so there is nothing else
/// honest to point at.
#[test]
fn a_door_with_no_targets_is_located_at_the_door() {
    let mut level = load_fixture(SOLVABLE);
    level.door.target_parts.clear();
    let r = check_level(&level, "<no-targets>");
    let c = check_named(&r, "door-targets");
    assert_eq!(c.severity, Severity::Fail, "{}", c.detail);
    assert_eq!(cells_of(c), vec![(1, 3, 1)], "the `[door] pos` of SOLVABLE");
}

/// A DEAD-BY-DESIGN node is a PASS that still carries its node's cell: the line
/// exists to be reviewed, and a reviewer who cannot walk to the node cannot
/// review it. `jump_targets` still leaves it out (below).
#[test]
fn a_dead_by_design_node_is_located_even_though_it_passes() {
    let r = threshold_fixture(99, 1, "output");
    let c = check_named(&r, "thresholds");
    assert_eq!(c.severity, Severity::Pass, "{}", c.detail);
    assert!(c.detail.contains("DEAD BY DESIGN"), "{}", c.detail);
    assert_eq!(
        cells_of(c),
        vec![(6, 3, 2)],
        "the `hidden` node's own pos: {}",
        c.detail
    );
}

/// A placard that leans on the binder's tolerances is located at its AUTHORED
/// `pos` — the value an author retypes — not at the cell the binder slid to.
#[test]
fn a_tolerated_placard_is_located_at_its_authored_pos() {
    // (5,3,2) is open air: the binder walks the gaze to the (11,3) wall column
    // and paints it at the eye layer. The FIX is at the authored pos.
    let r = with_note("[note system_log pos=5,3,2 facing=-x]\n    > log line\n");
    let c = check_named(&r, "notes");
    assert_eq!(c.severity, Severity::Warn, "{}", c.detail);
    assert!(
        c.detail.contains("(11, 3, 1)"),
        "precondition: it slid: {}",
        c.detail
    );
    assert_eq!(
        cells_of(c),
        vec![(5, 3, 2)],
        "the AUTHORED pos, not the cell the binder found: {}",
        c.detail
    );
}

// ---- Location::Element ----------------------------------------------------

/// THE ELEMENT CASE. A `[texture]` whose file is gone has no cell at all — the
/// offender is the CLAUSE, and it is addressed by kind + index + label so an
/// editor can select it in its object tree.
#[test]
fn a_missing_texture_is_located_at_its_texture_clause() {
    let mut level = shipped("TX5-genplate.lvl");
    level.textures[0].file = "assets/texture/genplate_tex_REMOVED.png".into();
    let r = check_level(&level, "TX5-genplate.lvl");
    let c = check_named(&r, "assets");
    assert_eq!(c.severity, Severity::Fail, "{}", c.detail);
    assert_eq!(
        c.locations,
        vec![Location::Element {
            kind: ElementKind::Texture,
            index: 0,
            name: "[texture genplate]".into(),
        }],
        "the clause, not a cell: {}",
        c.detail
    );
    assert_eq!(
        c.locations[0].as_cell(),
        None,
        "an element has no coordinate and must not invent one"
    );
}

/// The `[sky]` slots are indexed by SLOT (0 cylinder, 1 cap_top, 2 cap_bottom),
/// not by position in the reference list — a level that sets only `cap_top` must
/// still report index 1.
#[test]
fn a_broken_sky_slot_is_located_by_its_slot_index() {
    // A healthy level with NO authored `[sky]` at all — the field is set
    // in-memory only, exercising the same slot-indexing logic a deleted
    // fixture level used to.
    let mut level = shipped("TX5-genplate.lvl");
    level.sky.cylinder = None;
    level.sky.cap_bottom = None;
    level.sky.cap_top = Some("assets/texture/GONE_cap.png".into());
    let r = check_level(&level, "TX5-genplate.lvl");
    let c = check_named(&r, "assets");
    assert_eq!(c.severity, Severity::Fail, "{}", c.detail);
    assert_eq!(
        c.locations,
        vec![Location::Element {
            kind: ElementKind::Sky,
            index: 1,
            name: "[sky] cap_top".into(),
        }],
        "cap_top is slot 1 even when it is the only slot set: {}",
        c.detail
    );
}

/// A pipe whose endpoint index is out of range cannot be resolved to node
/// coordinates — by definition, the index names no node — so the `[pipe]` clause
/// is the location. The `.lvl` parser cannot author this (it resolves endpoints
/// by NAME), so the fixture is built by breaking a parsed level directly, which
/// is exactly the state a derived / editor-constructed level can reach.
#[test]
fn a_pipe_with_an_out_of_range_endpoint_is_located_at_the_pipe_clause() {
    let mut level = load_fixture(SOLVABLE);
    level.pipes[0].to_node = 99;
    let r = check_level(&level, "<bad-edge>");
    let want = vec![Location::Element {
        kind: ElementKind::Pipe,
        index: 0,
        name: "R->output".into(),
    }];

    let edges = check_named(&r, "edges");
    assert_eq!(edges.severity, Severity::Fail, "{}", edges.detail);
    assert_eq!(edges.locations, want, "{}", edges.detail);

    // The DAG rule fails on the same defect (`ForwardError::BadEdge`) and names
    // the same pipe rather than shrugging.
    let dag = check_named(&r, "dag");
    assert_eq!(dag.severity, Severity::Fail, "{}", dag.detail);
    assert_eq!(dag.locations, want, "{}", dag.detail);
}

/// A pipe whose `switch_cell` runs off the end of its path has no switch cell to
/// jump to; the clause is the honest address, and `cell`'s `(0,0,0)` filler is
/// never mistaken for one.
#[test]
fn a_switch_index_off_the_end_of_the_path_is_located_at_the_pipe_clause() {
    let mut level = load_fixture(SOLVABLE);
    level.pipes[0].switch_cell = Some(99);
    let r = check_level(&level, "<off-path>");
    let c = check_named(&r, "reachable");
    assert_eq!(c.severity, Severity::Fail, "{}", c.detail);
    assert_eq!(
        c.locations,
        vec![Location::Element {
            kind: ElementKind::Pipe,
            index: 0,
            name: "R->output".into(),
        }]
    );
    assert!(
        cells_of(c).is_empty(),
        "no fabricated (0,0,0) cell reaches the location list"
    );
}

// ---- Location::SourceLine -------------------------------------------------

/// THE SOURCE-LINE CASE. An unrecognized key has no world position even in
/// principle; the parser recorded the line, so that is the location.
#[test]
fn an_unknown_key_is_located_at_its_source_line() {
    let mut level = load_fixture(SOLVABLE);
    level.unknown_keys.push(re_level::UnknownKey {
        line: 7,
        section: "room",
        key: "wdith".into(),
    });
    level.unknown_keys.push(re_level::UnknownKey {
        line: 31,
        section: "pipe",
        key: "swtich_at".into(),
    });
    let r = check_level(&level, "<uk-loc>");
    let c = check_named(&r, "unknown_keys");
    assert_eq!(c.severity, Severity::Warn, "{}", c.detail);
    assert_eq!(
        c.locations,
        vec![Location::SourceLine(7), Location::SourceLine(31)],
        "one per key, in the order the detail lists them: {}",
        c.detail
    );
}

/// A level that does not parse reports the line the parser choked on, straight
/// from `ParseError::line`.
#[test]
fn a_parse_failure_is_located_at_the_offending_line() {
    let dir = std::env::temp_dir().join("ts_lvlcheck_loc_parsefail");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("broken.lvl");
    // Line 5 carries a malformed `[room] size`.
    std::fs::write(
        &path,
        "[meta]\nname = broken\n\n[room]\nsize = 12 8 x\nambient = trace-pattern\n",
    )
    .expect("write");

    let r = check_path(&path);
    let c = check_named(&r, "load");
    assert_eq!(c.severity, Severity::Fail, "{}", c.detail);
    assert_eq!(
        c.locations,
        vec![Location::SourceLine(5)],
        "the report points at the line that failed: {}",
        c.detail
    );
    let _ = std::fs::remove_file(&path);
}

/// A file that cannot be OPENED has no line at all (`ParseError::line == 0` is
/// the reader-error sentinel). It stays unlocated rather than pointing at a
/// fictional line 0.
#[test]
fn an_unopenable_file_reports_no_location_at_all() {
    let r = check_path("levels/definitely_not_here_9f3c.lvl");
    let c = check_named(&r, "load");
    assert_eq!(c.severity, Severity::Fail);
    assert!(
        c.locations.is_empty(),
        "no line to point at: {:?}",
        c.locations
    );
    assert_eq!(c.location(), None);
}

// ---- level-wide findings stay honestly unlocated --------------------------

/// The rules that are about the WHOLE level carry nothing. This is the half of
/// the contract that makes the other half trustworthy: if `inventory` invented a
/// coordinate, an issue browser would send an author to an arbitrary cell and
/// the feature would be worse than absent.
#[test]
fn level_wide_findings_carry_no_location() {
    let r = report(UNSOLVABLE);
    for name in ["inventory", "inputs", "solvable", "power_budget"] {
        let c = check_named(&r, name);
        assert!(
            c.locations.is_empty(),
            "'{name}' is a level-wide rule and must not fabricate a place: {:?} / {}",
            c.locations,
            c.detail
        );
        assert_eq!(c.location(), None, "'{name}' has no primary location");
    }
    let solvable = check_named(&r, "solvable");
    assert_eq!(solvable.severity, Severity::Fail, "precondition");
}

/// A clean level names no offenders anywhere, so nothing anywhere is located.
#[test]
fn a_clean_level_has_no_jump_targets() {
    let r = report(SOLVABLE);
    assert!(
        r.jump_targets().is_empty(),
        "a coherent level offers nothing to jump to: {:?}",
        r.jump_targets()
    );
}

/// `jump_targets` is the browse list: report order, WARN and FAIL only, one
/// entry per located offender, each carrying the index of the check it came
/// from. The dead-by-design PASS is documentation and is left out.
#[test]
fn jump_targets_lists_every_defect_in_report_order_and_skips_passes() {
    let r = report(UNREACHABLE_SWITCH);
    let targets = r.jump_targets();
    assert!(!targets.is_empty(), "this fixture has defects");

    for &(i, _) in &targets {
        assert!(
            r.checks[i].severity > Severity::Pass,
            "a PASS never reaches the browse list ({})",
            r.checks[i].name
        );
    }
    // Indices are non-decreasing: the list walks the report top to bottom.
    assert!(
        targets.windows(2).all(|w| w[0].0 <= w[1].0),
        "report order: {:?}",
        targets.iter().map(|(i, _)| *i).collect::<Vec<_>>()
    );
    // The buried switch appears, addressed by cell.
    assert!(
        targets.iter().any(|(_, l)| l.as_cell() == Some((6, 3, 10))),
        "the buried switch is reachable from the browse list"
    );

    // A PASS that DOES carry locations (dead-by-design) is still excluded.
    let dead = threshold_fixture(99, 1, "output");
    let td = check_named(&dead, "thresholds");
    assert!(!td.locations.is_empty(), "precondition: it is located");
    assert!(
        dead.jump_targets()
            .iter()
            .all(|&(i, _)| dead.checks[i].name != "thresholds"),
        "a PASS is documentation, not a defect to walk"
    );
}

/// Locations render in the report the same way whatever the variant, so the CLI
/// and the editor drawer can print one without matching on the enum.
#[test]
fn every_location_variant_renders_readably() {
    assert_eq!(Location::cell((5, 4, 2)).to_string(), "(5,4,2)");
    assert_eq!(
        Location::element(ElementKind::Pipe, 3, "R->output").to_string(),
        "pipe #3 'R->output'"
    );
    assert_eq!(Location::SourceLine(7).to_string(), "line 7");
}

/// The shipped corpus is the real test of "does this fire at all". A rule that
/// quietly stopped locating itself would show up here.
///
/// Not every firing of a rule has a place: `reachable` also FAILs with "no
/// switches to reach", which is a statement about the whole level and correctly
/// carries nothing. So the assertion is per-FINDING, keyed on the detail actually
/// naming an offender, not per-rule-name.
#[test]
fn the_shipped_corpus_locates_the_findings_that_name_an_offender() {
    let mut located = 0usize;
    let mut unlocated: Vec<String> = Vec::new();
    let mut kinds: Vec<&'static str> = Vec::new();
    for entry in std::fs::read_dir(levels_dir())
        .expect("levels/ must exist")
        .flatten()
    {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("lvl") {
            continue;
        }
        let r = check_path(&path);
        for c in &r.checks {
            if c.severity == Severity::Pass {
                continue;
            }
            // The rules whose every WARN/FAIL names a specific offender.
            let must_locate = matches!(c.name, "notes" | "assets" | "emit_parts" | "unknown_keys")
                || (c.name == "reachable" && c.detail.contains("unreachable"))
                || (c.name == "eye-level" && c.detail.contains("buried"));
            if must_locate {
                assert!(
                    !c.locations.is_empty(),
                    "{}: '{}' named an offender with nowhere to go: {}",
                    path.display(),
                    c.name,
                    c.detail
                );
                if !kinds.contains(&c.name) {
                    kinds.push(c.name);
                }
            }
            if c.locations.is_empty() {
                unlocated.push(format!(
                    "{}::{}",
                    path.file_name().unwrap().to_string_lossy(),
                    c.name
                ));
            } else {
                located += c.locations.len();
            }
        }
    }
    assert!(
        located > 0,
        "the corpus must exercise at least one located finding (unlocated: {unlocated:?})"
    );
    assert!(
        kinds.contains(&"notes"),
        "the corpus's known placard warnings must be located; saw {kinds:?}"
    );
}

// ===========================================================================
// 12. Per-level lighting (`[light]` / `[atmosphere]`)
// ===========================================================================
//
// The fixtures append `[light]` / `[atmosphere]` sections to `SOLVABLE`, a
// 12x8x4 room whose perimeter is `WallTrace` and whose interior (away from the
// pedestal at 5..7 x 3..5) is `Empty`. The rule is WARN-only, so every test also
// pins that the level's overall verdict is not dragged to FAIL by a lamp.

/// `SOLVABLE` plus the given trailing sections, linted.
fn lit(extra: &str) -> LevelReport {
    let src = format!("{SOLVABLE}{extra}");
    let level = load_string(&src, "fixture").expect("lit fixture must parse");
    check_level(&level, "fixture")
}

/// One `[light]` section at `(x, y, z)` with sane radius/intensity.
fn light_at(x: f32, y: f32, z: f32) -> String {
    format!("[light]\npos = {x}, {y}, {z}\ncolor = #FFB060\nradius = 4\nintensity = 1.0\n")
}

#[test]
fn a_level_with_no_lighting_passes_quietly() {
    let r = report(SOLVABLE);
    let c = check_named(&r, "lighting");
    assert_eq!(c.severity, Severity::Pass, "detail: {}", c.detail);
    assert_eq!(
        c.detail, "no authored lighting (engine defaults)",
        "the normal case must be silent, not noisy"
    );
    assert!(c.locations.is_empty());
}

#[test]
fn two_sane_lights_pass_and_name_the_count() {
    let r = lit(&format!(
        "{}{}",
        light_at(2.5, 6.5, 2.0),
        light_at(8.5, 6.5, 2.0)
    ));
    let c = check_named(&r, "lighting");
    assert_eq!(c.severity, Severity::Pass, "detail: {}", c.detail);
    assert!(
        c.detail.contains("2 point light(s)"),
        "the PASS must name the count: {}",
        c.detail
    );
    assert!(
        c.detail.contains("no [atmosphere] key/ambient override"),
        "the PASS must say whether a key/ambient override is authored: {}",
        c.detail
    );
}

#[test]
fn an_authored_atmosphere_override_is_named_in_the_pass() {
    let r = lit(&format!(
        "{}[atmosphere]\nkey_dir = -0.4, -1.0, 0.3\nambient = 0.18\npostfx = clean\n",
        light_at(2.5, 6.5, 2.0)
    ));
    let c = check_named(&r, "lighting");
    assert_eq!(c.severity, Severity::Pass, "detail: {}", c.detail);
    for want in ["key_dir", "ambient 0.18", "postfx 'clean'"] {
        assert!(
            c.detail.contains(want),
            "the PASS must name the {want} override: {}",
            c.detail
        );
    }
}

#[test]
fn an_atmosphere_only_level_passes_without_claiming_point_lights() {
    let r = lit("[atmosphere]\nambient = 0.2\n");
    let c = check_named(&r, "lighting");
    assert_eq!(c.severity, Severity::Pass, "detail: {}", c.detail);
    assert!(
        c.detail.contains("no [light] point lights") && c.detail.contains("ambient 0.2"),
        "detail: {}",
        c.detail
    );
}

#[test]
fn more_than_four_lights_warns_and_names_what_is_dropped() {
    // Six sane lights: the renderer bakes four and drops the last two.
    let mut src = String::new();
    for i in 0..6 {
        src.push_str(&light_at(2.5 + i as f32, 6.5, 2.0));
    }
    let r = lit(&src);
    let c = check_named(&r, "lighting");
    assert_eq!(c.severity, Severity::Warn, "detail: {}", c.detail);
    assert!(
        c.detail.contains("2 of 6") && c.detail.contains("4 of 6 reach"),
        "the WARN must say how many are dropped: {}",
        c.detail
    );
    assert!(c.detail.contains("cap"), "detail: {}", c.detail);

    let issues = light_issues(&load_fixture(&format!("{SOLVABLE}{src}")));
    assert_eq!(issues.len(), 2, "{issues:?}");
    assert_eq!(
        issues.iter().map(|i| i.index).collect::<Vec<_>>(),
        vec![4, 5],
        "the lights past the cap are the 5th and 6th, in parse order"
    );
    assert_eq!(
        c.locations.len(),
        2,
        "both dropped lights must be locatable"
    );
}

#[test]
fn a_light_buried_in_a_wall_warns_and_points_at_the_cell() {
    // (0,3,2) is the fixture room's west perimeter — `WallTrace`, solid.
    let src = light_at(0.5, 3.5, 2.0);
    assert!(
        load_fixture(SOLVABLE)
            .build_map()
            .cell_at(0, 3, 2)
            .is_solid(),
        "fixture precondition: (0,3,2) is a wall"
    );
    let r = lit(&src);
    let c = check_named(&r, "lighting");
    assert_eq!(c.severity, Severity::Warn, "detail: {}", c.detail);
    assert!(
        c.detail.contains("buried inside") && c.detail.contains("WallTrace"),
        "detail: {}",
        c.detail
    );
    assert_eq!(
        c.locations,
        vec![Location::cell((0, 3, 2))],
        "the WARN must send a cursor to the light's own cell"
    );
    // Buried but LIVE: it still costs one of the four slots.
    assert!(c.detail.contains("1 of 1 reach"), "detail: {}", c.detail);
}

#[test]
fn a_light_outside_the_room_footprint_warns() {
    let r = lit(&light_at(20.5, 3.5, 2.0));
    let c = check_named(&r, "lighting");
    assert_eq!(c.severity, Severity::Warn, "detail: {}", c.detail);
    assert!(
        c.detail.contains("outside the room footprint") && c.detail.contains("12x8"),
        "detail: {}",
        c.detail
    );
}

#[test]
fn a_light_above_the_ceiling_or_below_the_floor_warns() {
    // The fixture room is 4 cells tall (one storey of 4).
    let above = lit(&light_at(5.5, 6.5, 9.0));
    let c = check_named(&above, "lighting");
    assert_eq!(c.severity, Severity::Warn, "detail: {}", c.detail);
    assert!(
        c.detail.contains("above the ceiling") && c.detail.contains("storey height 4"),
        "detail: {}",
        c.detail
    );

    let below = lit(&light_at(5.5, 6.5, -2.0));
    let c = check_named(&below, "lighting");
    assert_eq!(c.severity, Severity::Warn, "detail: {}", c.detail);
    assert!(c.detail.contains("below the floor"), "detail: {}", c.detail);
}

#[test]
fn a_zero_intensity_light_is_reported_as_inert() {
    let r = lit("[light]\npos = 5.5, 6.5, 2.0\nradius = 4\nintensity = 0\n");
    let c = check_named(&r, "lighting");
    assert_eq!(c.severity, Severity::Warn, "detail: {}", c.detail);
    assert!(
        c.detail.contains("intensity 0 is not > 0") && c.detail.contains("INERT"),
        "detail: {}",
        c.detail
    );
    // Dropped, so nothing reaches the renderer.
    assert!(c.detail.contains("0 of 1 reach"), "detail: {}", c.detail);
}

/// The parser refuses `radius <= 0` outright, so this class is reachable only
/// through a synthetic / editor-built `Level` — which is exactly what the
/// linter's pure entry point exists to accept.
#[test]
fn a_zero_radius_light_built_in_memory_is_reported_as_inert() {
    let mut level = load_fixture(SOLVABLE);
    level.lights.push(re_world::PointLight {
        pos: re_core::Vec3::new(5.5, 6.5, 2.0),
        color: 0x00FF_FFFF,
        radius: 0.0,
        intensity: 1.0,
    });
    let r = check_level(&level, "fixture");
    let c = check_named(&r, "lighting");
    assert_eq!(c.severity, Severity::Warn, "detail: {}", c.detail);
    assert!(
        c.detail.contains("radius 0 is not > 0"),
        "detail: {}",
        c.detail
    );
}

/// Mirrors `PointLights::from_world`: an inert light is skipped BEFORE the cap
/// counter moves, so it costs no slot. Five sections, one of them dead, means
/// the four live ones all survive — and only the dead one is reported.
#[test]
fn an_inert_light_does_not_consume_a_cap_slot() {
    let mut src = String::from("[light]\npos = 4.5, 6.5, 2.0\nradius = 4\nintensity = 0\n");
    for i in 0..4 {
        src.push_str(&light_at(2.5 + i as f32, 6.5, 2.0));
    }
    let issues = light_issues(&load_fixture(&format!("{SOLVABLE}{src}")));
    assert_eq!(
        issues.len(),
        1,
        "only the inert light is a finding: {issues:?}"
    );
    assert_eq!(issues[0].index, 0);
    let r = lit(&src);
    let c = check_named(&r, "lighting");
    assert!(
        c.detail.contains("4 of 5 reach"),
        "the four live lights must all fit under the cap: {}",
        c.detail
    );
}

/// A lighting mistake is cosmetic. Whatever an author does to `[light]`, the
/// rule must never sink the report to FAIL.
#[test]
fn the_lighting_rule_never_fails() {
    let hostile = format!(
        "{}{}{}{}{}{}{}",
        light_at(0.5, 3.5, 2.0),                                     // buried
        light_at(99.0, 99.0, 9.0),                                   // outside
        light_at(5.5, 6.5, -3.0),                                    // below the floor
        "[light]\npos = 5.5, 6.5, 2.0\nradius = 4\nintensity = 0\n", // inert
        light_at(2.5, 6.5, 2.0),
        light_at(3.5, 6.5, 2.0),
        light_at(4.5, 6.5, 2.0), // past the cap
    );
    let r = lit(&hostile);
    let c = check_named(&r, "lighting");
    assert_eq!(c.severity, Severity::Warn, "detail: {}", c.detail);
    assert!(
        !r.failed(),
        "lighting must not make a solvable level FAIL: {}",
        r.verdict
    );
}

/// Every shipped level must stay quiet under the new rule (none authors
/// `[light]` today, and none may be dragged to WARN by one).
#[test]
fn the_shipped_corpus_is_lighting_clean() {
    let mut checked = 0usize;
    for entry in std::fs::read_dir(levels_dir())
        .expect("levels/ must exist")
        .flatten()
    {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("lvl") {
            continue;
        }
        let r = check_path(&path);
        if r.inventory.is_none() {
            continue;
        }
        checked += 1;
        let c = r
            .checks
            .iter()
            .find(|c| c.name == "lighting")
            .unwrap_or_else(|| panic!("[{}] the lighting rule must run", path.display()));
        assert_ne!(
            c.severity,
            Severity::Fail,
            "[{}] lighting must never FAIL: {}",
            path.display(),
            c.detail
        );
    }
    assert!(checked > 0, "the sweep must find levels");
}

/// The vertical band is the ROOM's, not one storey's. A multi-floor level (the
/// shipped campaign is full of them: L1 is 13 cells tall with `storey_height =
/// 4`) legitimately hangs a lamp on its second floor, and warning "above the
/// ceiling" at every `z > storey_height` would false-flag every upper storey.
/// The storey height is REPORTED in the message, never used as the ceiling.
#[test]
fn a_light_on_an_upper_storey_is_not_flagged_above_the_ceiling() {
    let multi = SOLVABLE.replace(
        "size = 12 8 4",
        "size = 12 8 4
storey_height = 2",
    );
    let src = format!("{multi}{}", light_at(2.5, 6.5, 3.0));
    let level = load_string(&src, "fixture").expect("multi-storey fixture must parse");
    assert_eq!(level.storey_height(), 2);
    let c = check_level(&level, "fixture");
    let c = c
        .checks
        .iter()
        .find(|c| c.name == "lighting")
        .expect("the lighting rule must run");
    assert_eq!(
        c.severity,
        Severity::Pass,
        "z 3 is the second storey of a 4-cell-tall room, not above its ceiling: {}",
        c.detail
    );
}

// ── 12b. lit-surfaces: lamps a level's own surfaces cannot receive ──────────
// A `[light]` falls on nothing when the surface it would light is skinned with a
// SELF-LIT recipe — the shader returns before the lighting pass. This is the
// mistake the lighting demo level itself shipped with for one authoring pass.
mod lit_surfaces {
    use super::*;

    /// A minimal solvable level whose wall_solid material and `[light]` block are
    /// substituted in, so the only thing under test is the recipe on the wall.
    fn level(wall_recipe: &str, lights: &str) -> String {
        format!(
            "[meta]\nname = lit-surf\n[room]\nsize = 10 6 4\nambient = solid\n\
             {lights}\
             [material walls]\nrecipe = {wall_recipe}\nfg = #808080\nbg = #101010\n\
             surface = wall_solid\n\
             [node R]\npos = 2,3,2\nis_input = yes\n\
             [node output]\npos = 6,3,2\nis_output = yes\nthreshold = 1\n\
             [pipe R->output]\nstate = OFF\npath = (2,3,2) (6,3,2)\nswitch_at = (4,3,2)\n\
             [inputs]\nR = 1\n\
             [pedestal]\ncenter = 6 3 1\nvolume = 1 1 1\n\
             [door]\npos = 9 3 1\ntarget_parts = output\n\
             [player]\nspawn = 3 3 1\nfacing = +x\n"
        )
    }

    const LAMP: &str =
        "[light]\npos = 5.0, 3.0, 2.5\ncolor = #FFFFFF\nradius = 4.0\nintensity = 1.0\n";

    #[test]
    fn a_self_lit_wall_under_a_lamp_warns() {
        let r = report(&level("scanline freq=9", LAMP));
        let c = check_named(&r, "lit-surfaces");
        assert_eq!(c.severity, Severity::Warn, "detail: {}", c.detail);
        assert!(
            c.detail.contains("wall_solid"),
            "names the slot: {}",
            c.detail
        );
        assert!(
            c.detail.contains("walls"),
            "names the material: {}",
            c.detail
        );
        assert!(
            c.detail.contains("scanline"),
            "names the recipe: {}",
            c.detail
        );
    }

    #[test]
    fn the_same_wall_on_a_lit_recipe_passes() {
        let r = report(&level("brick aspect=2", LAMP));
        let c = check_named(&r, "lit-surfaces");
        assert_eq!(c.severity, Severity::Pass, "detail: {}", c.detail);
    }

    #[test]
    fn without_lamps_a_self_lit_wall_is_nobodys_business() {
        // Phosphor walls are a legitimate look; the rule only fires when the level
        // authored light that cannot land.
        let r = report(&level("scanline freq=9", ""));
        let c = check_named(&r, "lit-surfaces");
        assert_eq!(c.severity, Severity::Pass, "detail: {}", c.detail);
        assert!(
            c.detail.contains("no [light]"),
            "says why it is quiet: {}",
            c.detail
        );
    }

    #[test]
    fn every_self_lit_recipe_kind_is_caught_and_every_other_is_not() {
        // The pinned table, exercised through the real parser rather than by
        // re-listing the variants in the assertion.
        for kind in [
            "scanline freq=9",
            "truchet seed=1",
            "fractal arms=3",
            "hologram",
            "codewall",
        ] {
            let r = report(&level(kind, LAMP));
            assert_eq!(
                check_named(&r, "lit-surfaces").severity,
                Severity::Warn,
                "`{kind}` is self-lit and must be caught"
            );
        }
        for kind in [
            "brick aspect=2",
            "circuit scale=2",
            "grain scale=3",
            "substrate scale=3",
            "voice scale=3",
        ] {
            let r = report(&level(kind, LAMP));
            assert_eq!(
                check_named(&r, "lit-surfaces").severity,
                Severity::Pass,
                "`{kind}` receives light and must not be flagged"
            );
        }
    }

    #[test]
    fn an_image_wall_is_deaf_until_it_authors_relief() {
        let img = |extra: &str| {
            format!(
                "[meta]\nname = lit-img\n[room]\nsize = 10 6 4\nambient = solid\n\
                 {LAMP}\
                 [texture pic]\nfile = assets/texture/genplate_tex.png\n\
                 [material walls]\nrecipe = image\ntexture = pic\n{extra}surface = wall_solid\n\
                 [node R]\npos = 2,3,2\nis_input = yes\n\
                 [node output]\npos = 6,3,2\nis_output = yes\nthreshold = 1\n\
                 [pipe R->output]\nstate = OFF\npath = (2,3,2) (6,3,2)\nswitch_at = (4,3,2)\n\
                 [inputs]\nR = 1\n\
                 [pedestal]\ncenter = 6 3 1\nvolume = 1 1 1\n\
                 [door]\npos = 9 3 1\ntarget_parts = output\n\
                 [player]\nspawn = 3 3 1\nfacing = +x\n"
            )
        };
        assert_eq!(
            check_named(&report(&img("")), "lit-surfaces").severity,
            Severity::Warn,
            "a bare image is a self-lit decal"
        );
        assert_eq!(
            check_named(&report(&img("relief = height_scale=1.0\n")), "lit-surfaces").severity,
            Severity::Pass,
            "`relief =` is exactly what makes an image a lit surface"
        );
    }

    #[test]
    fn the_shipped_lighting_demo_has_no_deaf_surface() {
        // The regression guard for the mistake this rule was written for.
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../levels/22-lit-cellar.lvl");
        let raw = std::fs::read_to_string(&path).expect("the lighting demo exists");
        let level = load_string(&raw, "22-lit-cellar").expect("it parses");
        assert!(
            crate::light_deaf_surfaces(&level).is_empty(),
            "the lighting demo must not skin a surface with a recipe its lamps cannot reach: {:?}",
            crate::light_deaf_surfaces(&level)
        );
    }
}

// ───────────────────────────── props rule ─────────────────────────────────────

/// A room split by an x = 6 line: spawn in the west half, the switch column in
/// the east half. `{props}` is spliced in.
const PROPS_ROOM: &str = r#"
[meta]
name = fixture-props
[room]
size = 14 10 5
ambient = trace-pattern
[node R]
pos = 10,2,2
glyph = R
color = #FF0000
is_input = yes
[node output]
pos = 10,7,2
glyph = O
color = #FFFFFF
is_output = yes
threshold = 1
[pipe R->output]
state = OFF
path = (10,2,2) (10,7,2)
switch_at = (10,5,2)
[inputs]
R = 1
[pedestal]
center = 12 7 2
volume = 1 1 1
[door]
pos = 12 1 1
target_parts = output
[player]
spawn = 2 5 1
facing = +x
"#;

fn props_report(props: &str) -> LevelReport {
    report(&format!("{PROPS_ROOM}{props}"))
}

#[test]
fn a_level_without_props_passes_the_props_rule() {
    let r = props_report("");
    assert_eq!(check_named(&r, "props").severity, Severity::Pass);
}

#[test]
fn props_leaving_a_gap_pass() {
    let r = props_report("[prop wall_a]\nkind = shelving\nat = 6 8 1\nfacing = +x\nsize = 4 1 3\n");
    let c = check_named(&r, "props");
    assert_eq!(c.severity, Severity::Pass, "{}", c.detail);
}

#[test]
fn props_cutting_off_the_switch_fail() {
    let r = props_report(
        "[prop wall_a]\nkind = shelving\nat = 6 8 1\nfacing = +x\nsize = 4 1 3\n\n\
         [prop wall_b]\nkind = shelving\nat = 6 4 1\nfacing = +x\nsize = 4 1 3\n",
    );
    let c = check_named(&r, "props");
    assert_eq!(c.severity, Severity::Fail, "{}", c.detail);
    assert!(c.detail.contains("cut off the switch"), "{}", c.detail);
    assert!(
        c.detail.contains("door"),
        "the door is behind the wall too: {}",
        c.detail
    );
    assert!(r.failed());
}

/// A 12×8×7 room authored as a dense `[cells]` grid with a raised ledge
/// (x 9..=10, solid z 1..=3, so its top is authored ground at z = 4) that the
/// bare room cannot reach. The puzzle sits on the floor at x = 3.
fn ledge_room(props: &str) -> String {
    let (sx, sy, sz) = (12, 8, 7);
    let mut grid = String::new();
    for z in 0..sz {
        for y in 0..sy {
            for x in 0..sx {
                let border = x == 0 || y == 0 || x == sx - 1 || y == sy - 1;
                let ch = if z == 0 {
                    if border {
                        '#'
                    } else {
                        '_'
                    }
                } else if z == sz - 1 || border || ((9..=10).contains(&x) && (1..=3).contains(&z)) {
                    '#'
                } else if z == 2 && x == 3 && (y == 2 || y == 6) {
                    'O'
                } else {
                    '.'
                };
                grid.push(ch);
            }
            grid.push('\n');
        }
        grid.push('\n');
    }
    format!(
        "[meta]\nname = fixture-ledge\n[room]\nsize = {sx} {sy} {sz}\nambient = solid\n\
         [node R]\npos = 3,2,2\nglyph = R\ncolor = #FF0000\nis_input = yes\n\
         [node output]\npos = 3,6,2\nglyph = O\ncolor = #FFFFFF\nis_output = yes\nthreshold = 1\n\
         [pipe R->output]\nstate = OFF\npath = (3,2,2) (3,6,2)\nswitch_at = (3,4,2)\n\
         [inputs]\nR = 1\n[pedestal]\ncenter = 7 6 2\nvolume = 1 1 1\n[door]\npos = 1 1 1\ntarget_parts = output\n\
         [player]\nspawn = 6 5 1\nfacing = +x\n{props}\n[cells]\nsize = {sx},{sy},{sz}\n{grid}"
    )
}

#[test]
fn an_unreachable_ledge_room_passes_the_props_rule_without_props() {
    let r = report(&ledge_room(""));
    assert_eq!(check_named(&r, "props").severity, Severity::Pass);
}

/// Props stacked into a climbable ladder up to authored ground the bare room
/// cannot reach are a shortcut: FAIL, even though nothing is cut off.
#[test]
fn props_forming_a_ladder_onto_authored_ground_fail() {
    let r = report(&ledge_room(
        "[prop s1]\nkind = column\nat = 6 3 1\nsize = 1 1 1\n\n\
         [prop s2]\nkind = column\nat = 7 3 1\nsize = 1 1 2\n\n\
         [prop s3]\nkind = column\nat = 8 3 1\nsize = 1 1 3\n\n",
    ));
    let c = check_named(&r, "props");
    assert_eq!(c.severity, Severity::Fail, "{}", c.detail);
    assert!(c.detail.contains("ladder"), "{}", c.detail);
}

/// The same climb is fine when it tops out on the props themselves only (the
/// ledge is not reachable from the last step).
#[test]
fn props_climbable_onto_themselves_are_not_a_shortcut() {
    let r = report(&ledge_room(
        "[prop s1]\nkind = column\nat = 5 3 1\nsize = 1 1 1\n\n\
         [prop s2]\nkind = column\nat = 6 3 1\nsize = 1 1 2\n\n",
    ));
    let c = check_named(&r, "props");
    assert_eq!(c.severity, Severity::Pass, "{}", c.detail);
}

/// A spawn with no stand cell (here: inside the ceiling slab) gives the walk
/// flood nothing to start from. Every connectivity rule would then compare two
/// empty floods and pass vacuously ("reach 0 -> 0"), so the rule must FAIL and
/// say why.
#[test]
fn props_rule_fails_when_the_spawn_has_no_reach() {
    let src = PROPS_ROOM.replace("spawn = 2 5 1", "spawn = 2 5 4");
    let r = report(&format!(
        "{src}[prop wall_a]\nkind = shelving\nat = 6 8 1\nfacing = +x\nsize = 4 1 3\n"
    ));
    let c = check_named(&r, "props");
    assert_eq!(c.severity, Severity::Fail, "{}", c.detail);
    assert!(
        c.detail.contains("no stand cell at the spawn"),
        "{}",
        c.detail
    );
    assert!(c.detail.contains("reach 0 -> 0"), "{}", c.detail);
}
