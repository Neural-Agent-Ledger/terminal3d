//! Ported puzzle spec — `legacy/tests/test_puzzle.c` (the AND-gate fixture) +
//! `legacy/tests/test_levels_realpath.c` (the real-path solvability sweep).
//!
//! The realpath sweep loads each shipped `levels/NN.lvl` via `re_level`, builds
//! the network, brute-forces a winning pipe configuration (mirroring
//! `tools/lvlcheck.c`), then drives the WINNING toggle sequence through
//! `puzzle_interact` at the real switch-cell world coordinates and asserts the
//! puzzle reports solved.

use super::*;
use re_level::{load, Level, PipeState};
use std::path::PathBuf;

/// Path to a repo-root level file (workspace root is two dirs above the crate).
fn level_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../levels")
        .join(name)
}

/// Path to a crate-local test fixture under `tests/fixtures/`. Used for engine
/// mechanics (sequence-lock, power-budget) that are convenient to demonstrate
/// on a real `.lvl` shape without depending on a campaign level — an engine
/// test should not depend on shipped campaign content, which lives with the
/// game, not the engine.
fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

// ---------------------------------------------------------------------------
// Real-path solvability sweep (port of test_levels_realpath.c)
// ---------------------------------------------------------------------------

/// Maximum pipe count we brute-force (mirrors `REALPATH_MAX_PIPES_BF`).
const MAX_PIPES_BF: usize = 16;

/// Brute-force a winning pipe-state bitmask (bit `i` = desired ON state for
/// pipe `i`), or `None` if unsolvable within the search space. Restores the
/// state's pipes + forward pass before returning. Port of `find_winning_mask`.
fn find_winning_mask(state: &mut PuzzleState) -> Option<u32> {
    let n = state.pipes.len();
    if n > MAX_PIPES_BF || state.door_targets.is_empty() {
        return None;
    }

    let saved: Vec<PipeState> = state.pipe_states();
    let targets = state.door_targets.clone();
    let total: u32 = 1u32 << n;

    let mut winning = None;
    for s in 0..total {
        for i in 0..n {
            state.pipes[i].state = if (s >> i) & 1 == 1 {
                PipeState::On
            } else {
                PipeState::Off
            };
        }
        if network_forward(state).is_err() {
            continue;
        }
        if check_win(state, &targets) {
            winning = Some(s);
            break;
        }
    }

    // Restore.
    for (pipe, &snap) in state.pipes.iter_mut().zip(&saved) {
        pipe.state = snap;
    }
    let _ = network_forward(state);
    winning
}

/// Load a level, build the network, brute-force a winning config, then drive
/// the required toggles via `puzzle_interact` at the real switch cells and
/// assert `solved`.
fn run_level_solvability(name: &str, label: &str) {
    let level: Level = load(level_path(name)).unwrap_or_else(|e| panic!("[{label}] load: {e}"));
    let mut state =
        PuzzleState::from_level(&level).unwrap_or_else(|e| panic!("[{label}] from_level: {e:?}"));

    let n_pipes = state.pipes.len();
    assert!(
        !state.door_targets.is_empty(),
        "[{label}] level has no door targets — cannot be won"
    );

    // 1. Winning mask.
    let mask = find_winning_mask(&mut state)
        .unwrap_or_else(|| panic!("[{label}] no winning pipe configuration found"));

    // 2. Authored initial states (from_level preserved them; brute force
    //    restored them).
    let initial: Vec<PipeState> = state.pipe_states();

    // 3. Pipes that must flip relative to initial.
    let to_toggle: Vec<usize> = (0..n_pipes)
        .filter(|&i| {
            let desired = if (mask >> i) & 1 == 1 {
                PipeState::On
            } else {
                PipeState::Off
            };
            desired != initial[i]
        })
        .collect();

    // Already solved with the authored config?
    if to_toggle.is_empty() {
        assert!(
            state.solved,
            "[{label}] no toggles needed but not solved at initial config"
        );
        return;
    }

    // 4. Drive each required toggle at its switch cell.
    for &pi in &to_toggle {
        let (sx, sy, sz) = state.pipes[pi]
            .switch_coord()
            .unwrap_or_else(|| panic!("[{label}] pipe {pi} must toggle but has no switch cell"));
        let changed = puzzle_interact(&mut state, sx, sy, sz);
        assert!(
            changed,
            "[{label}] puzzle_interact at pipe {pi} switch ({sx},{sy},{sz}) did nothing"
        );
    }

    // 5. Solved after the winning sequence.
    assert!(
        state.solved,
        "[{label}] not solved after {} toggles (mask={mask:#x})",
        to_toggle.len()
    );
}

#[test]
fn solvable_01() {
    run_level_solvability("01.lvl", "L1");
}
#[test]
fn solvable_02() {
    run_level_solvability("02.lvl", "L2");
}
#[test]
fn solvable_03() {
    run_level_solvability("03.lvl", "L3");
}
#[test]
fn solvable_04() {
    run_level_solvability("04.lvl", "L4");
}
#[test]
fn solvable_05() {
    run_level_solvability("05.lvl", "L5");
}
#[test]
fn solvable_06() {
    run_level_solvability("06.lvl", "L6");
}
#[test]
fn solvable_07() {
    run_level_solvability("07.lvl", "L7");
}

/// Sequence-lock mechanic, driven through the REAL switch cells of a parsed
/// level. Loads the crate-local `sequence_lock_budget.lvl` fixture — a neutral
/// clone of the campaign's L11 "Inner Circuit" graph shape (same coordinates,
/// same topology, generic node names, so the test does not depend on shipped
/// campaign content) — confirms the `[sequence]` parsed into a three-pattern
/// ordered lock, then drives BREAK→FALL→DARK in order and asserts the puzzle
/// reports solved only after the full ordered sequence.
#[test]
fn sequence_lock_ordered_solve_from_level() {
    let level = load(fixture_path("sequence_lock_budget.lvl")).expect("fixture loads");
    let mut s = PuzzleState::from_level(&level).expect("fixture builds");

    // The [sequence] parsed into three distinct patterns; inactive at load.
    assert_eq!(
        s.sequence.patterns.len(),
        3,
        "fixture declares 3 ordered patterns"
    );
    assert_eq!(s.sequence.progress, 0, "no auto-lock at the all-OFF start");
    assert!(!s.solved);

    // Resolve switch cells by node-pair (from->to), via the level's name index.
    let idx = |n: &str| level.node_index(n).unwrap();
    let switch = |s: &PuzzleState, from: usize, to: usize| -> (i32, i32, i32) {
        s.pipes
            .iter()
            .find(|p| p.from == from && p.to == to)
            .and_then(|p| p.switch_coord())
            .unwrap()
    };
    let (in1, in2, in3, gate, out1, out2, out3) = (
        idx("in1"),
        idx("in2"),
        idx("in3"),
        idx("gate"),
        idx("out1"),
        idx("out2"),
        idx("out3"),
    );

    // `gate` (T2 = in1+in2) is load-bearing for both out2 and out3.
    // Pipes: A=in1->out1, B=in1->gate, C=in2->gate, D=in2->out2,
    // E=gate->out2, F=gate->out3, G=in3->out3. Budget 4.

    // P1 BREAK: +A (in1->out1) → output (1,0,0).
    let (x, y, z) = switch(&s, in1, out1);
    assert!(puzzle_interact(&mut s, x, y, z), "A on");
    assert_eq!(s.sequence.progress, 1, "P1 locks");
    assert!(!s.solved);

    // P2 FALL: −A, +B, +C, +D, +E (4/4) → gate = in1+in2 = 2, out2 = D+E = 2.
    let (x, y, z) = switch(&s, in1, out1);
    assert!(puzzle_interact(&mut s, x, y, z), "A off"); // out1 off
    let (x, y, z) = switch(&s, in1, gate);
    assert!(puzzle_interact(&mut s, x, y, z), "B on");
    let (x, y, z) = switch(&s, in2, gate);
    assert!(puzzle_interact(&mut s, x, y, z), "C on");
    let (x, y, z) = switch(&s, in2, out2);
    assert!(puzzle_interact(&mut s, x, y, z), "D on");
    let (x, y, z) = switch(&s, gate, out2);
    assert!(puzzle_interact(&mut s, x, y, z), "E on");
    assert_eq!(s.sequence.progress, 2, "P2 locks");
    assert!(!s.solved);

    // P3 DARK: −D, −E, +F, +G (4/4) → gate stays 2 (B, C untouched),
    // out3 = F+G = 2.
    let (x, y, z) = switch(&s, in2, out2);
    assert!(puzzle_interact(&mut s, x, y, z), "D off"); // out2 off
    let (x, y, z) = switch(&s, gate, out2);
    assert!(puzzle_interact(&mut s, x, y, z), "E off");
    let (x, y, z) = switch(&s, gate, out3);
    assert!(puzzle_interact(&mut s, x, y, z), "F on");
    let (x, y, z) = switch(&s, in3, out3);
    assert!(puzzle_interact(&mut s, x, y, z), "G on");
    assert_eq!(s.sequence.progress, 3, "P3 locks → solved");
    assert!(s.solved, "solved after the ordered BREAK→FALL→DARK drive");
}

// ---------------------------------------------------------------------------
// Known-fixture forward pass — L2: R + G -> mix(threshold 2) -> output
// ---------------------------------------------------------------------------

/// With R->mix, G->mix and mix->output all ON, `mix` reaches threshold 2 and
/// `output` fires. The `neutral` input is 0, so the neutral->mix decoy cannot
/// help. This pins the actual activation values, not just the solved flag.
#[test]
fn l2_forward_known_activations() {
    let level = load(level_path("02.lvl")).expect("02.lvl");
    let mut state = PuzzleState::from_level(&level).expect("build L2");

    let idx = |n: &str| level.node_index(n).unwrap();
    let (r, g, neutral, mix, output) = (
        idx("R"),
        idx("G"),
        idx("neutral"),
        idx("mix"),
        idx("output"),
    );

    // Initial: all pipes OFF → mix sees 0 active inputs → output off.
    assert_eq!(state.nodes[r].activation, 1, "R is input=1");
    assert_eq!(state.nodes[g].activation, 1, "G is input=1");
    assert_eq!(state.nodes[neutral].activation, 0, "neutral is input=0");
    assert_eq!(state.nodes[mix].activation, 0, "mix off with no ON pipes");
    assert_eq!(state.nodes[output].activation, 0, "output off initially");
    assert!(!state.solved);

    // Turn on R->mix and G->mix → mix sums 1+1 >= 2 → fires.
    let switch = |s: &PuzzleState, from: usize, to: usize| -> (i32, i32, i32) {
        s.pipes
            .iter()
            .find(|p| p.from == from && p.to == to)
            .and_then(|p| p.switch_coord())
            .unwrap()
    };
    let (x, y, z) = switch(&state, r, mix);
    assert!(puzzle_interact(&mut state, x, y, z));
    let (x, y, z) = switch(&state, g, mix);
    assert!(puzzle_interact(&mut state, x, y, z));

    assert_eq!(state.nodes[mix].activation, 1, "R+G fires mix (2>=2)");
    assert_eq!(
        state.nodes[output].activation, 0,
        "mix->output still OFF, so output not yet driven"
    );
    assert!(!state.solved);

    // Now wire mix->output ON → output fires (threshold 1) → solved.
    let (x, y, z) = switch(&state, mix, output);
    assert!(puzzle_interact(&mut state, x, y, z));
    assert_eq!(state.nodes[output].activation, 1, "output fires");
    assert!(state.solved, "L2 solved when output active");
}

/// A decoy pipe (L2 `neutral->mix`, source input = 0) cannot by itself push
/// `mix` to threshold. Toggling only the decoy + mix->output leaves the puzzle
/// unsolved.
#[test]
fn l2_decoy_does_not_solve() {
    let level = load(level_path("02.lvl")).expect("02.lvl");
    let mut state = PuzzleState::from_level(&level).expect("build L2");

    let idx = |n: &str| level.node_index(n).unwrap();
    let (neutral, mix, output) = (idx("neutral"), idx("mix"), idx("output"));

    let switch = |s: &PuzzleState, from: usize, to: usize| -> (i32, i32, i32) {
        s.pipes
            .iter()
            .find(|p| p.from == from && p.to == to)
            .and_then(|p| p.switch_coord())
            .unwrap()
    };

    // Toggle the decoy (neutral->mix) and mix->output ON. neutral=0 → mix sees
    // sum 0 < threshold 2 → mix off → output off.
    let (x, y, z) = switch(&state, neutral, mix);
    assert!(puzzle_interact(&mut state, x, y, z));
    let (x, y, z) = switch(&state, mix, output);
    assert!(puzzle_interact(&mut state, x, y, z));

    assert_eq!(state.nodes[mix].activation, 0, "decoy alone can't fire mix");
    assert_eq!(state.nodes[output].activation, 0);
    assert!(!state.solved, "decoy-only toggles must not solve L2");
}

// ---------------------------------------------------------------------------
// AND-gate fixture (port of test_puzzle.c) — built by hand, no parser
// ---------------------------------------------------------------------------

/// in_a(=1) -> out(threshold 2) <- in_b(=1); both pipes start OFF. Mirrors the
/// legacy `fixture_build` AND gate. Switches at distinct world cells.
fn and_gate_fixture() -> PuzzleState {
    let nodes = vec![
        PuzzleNode {
            threshold: 0,
            is_input: true,
            is_output: false,
            activation: 0,
        }, // in_a
        PuzzleNode {
            threshold: 0,
            is_input: true,
            is_output: false,
            activation: 0,
        }, // in_b
        PuzzleNode {
            threshold: 2,
            is_input: false,
            is_output: true,
            activation: 0,
        }, // out
    ];
    let pipes = vec![
        Pipe {
            from: 0,
            to: 2,
            state: PipeState::Off,
            path: vec![(5, 1, 1)],
            switch_cell: Some(0),
        },
        Pipe {
            from: 1,
            to: 2,
            state: PipeState::Off,
            path: vec![(6, 1, 1)],
            switch_cell: Some(0),
        },
    ];
    let mut state = PuzzleState {
        nodes,
        pipes,
        inputs: vec![1, 1, 0],
        door_targets: vec![2],
        sequence: SequenceLock::default(),
        target_off: Vec::new(),
        cycle_keep: false,
        cycle_held: false,
        power_budget: 0,
        solved: false,
        attempts: 0,
    };
    network_forward(&mut state).expect("fixture forward");
    state.solved = check_win(&state, &state.door_targets);
    state
}

#[test]
fn and_gate_initial_unsolved() {
    let state = and_gate_fixture();
    assert!(!state.solved, "both pipes OFF → out OFF → unsolved");
    assert_eq!(state.attempts, 0);
    assert_eq!(state.nodes[2].activation, 0);
}

#[test]
fn and_gate_one_toggle_not_enough() {
    let mut state = and_gate_fixture();
    assert!(puzzle_interact(&mut state, 5, 1, 1)); // pipe 0 ON
    assert_eq!(state.pipes[0].state, PipeState::On);
    assert_eq!(state.attempts, 1);
    assert!(!state.solved, "1 of 2 inputs → sum 1 < threshold 2");
}

#[test]
fn and_gate_both_toggles_solve() {
    let mut state = and_gate_fixture();
    assert!(puzzle_interact(&mut state, 5, 1, 1));
    assert!(puzzle_interact(&mut state, 6, 1, 1));
    assert_eq!(
        state.nodes[2].activation, 1,
        "sum 2 >= threshold 2 fires out"
    );
    assert!(state.solved);
    assert_eq!(state.attempts, 2);
}

#[test]
fn interact_at_empty_cell_is_noop() {
    let mut state = and_gate_fixture();
    assert!(!puzzle_interact(&mut state, 0, 0, 0), "no switch there");
    assert_eq!(state.attempts, 0, "no-op must not count an attempt");
    assert_eq!(state.pipes[0].state, PipeState::Off);
    assert_eq!(state.pipes[1].state, PipeState::Off);
}

#[test]
fn no_targets_never_solves() {
    let mut state = and_gate_fixture();
    state.door_targets.clear();
    // Fire the output anyway.
    puzzle_interact(&mut state, 5, 1, 1);
    puzzle_interact(&mut state, 6, 1, 1);
    assert!(!state.solved, "no door targets → never solved");
    assert!(!check_win(&state, &state.door_targets));
}

#[test]
fn reset_restores_initial_pipe_states() {
    let mut state = and_gate_fixture();
    let snapshot = state.pipe_states();
    puzzle_interact(&mut state, 5, 1, 1);
    puzzle_interact(&mut state, 6, 1, 1);
    assert!(state.solved);

    state.reset_to(&snapshot);
    assert_eq!(state.pipes[0].state, PipeState::Off);
    assert_eq!(state.pipes[1].state, PipeState::Off);
    assert!(!state.solved, "post-reset unsolved");
}

// ---------------------------------------------------------------------------
// Cycle detection
// ---------------------------------------------------------------------------

/// A 2-node cycle (a->b->a, no inputs) is not a DAG → forward errors with
/// `Cycle`. Port of the legacy `network_forward` cycle path.
#[test]
fn cycle_is_detected() {
    let mut state = PuzzleState {
        nodes: vec![
            PuzzleNode {
                threshold: 1,
                is_input: false,
                is_output: false,
                activation: 0,
            },
            PuzzleNode {
                threshold: 1,
                is_input: false,
                is_output: false,
                activation: 0,
            },
        ],
        pipes: vec![
            Pipe {
                from: 0,
                to: 1,
                state: PipeState::On,
                path: vec![],
                switch_cell: None,
            },
            Pipe {
                from: 1,
                to: 0,
                state: PipeState::On,
                path: vec![],
                switch_cell: None,
            },
        ],
        inputs: vec![0, 0],
        door_targets: vec![1],
        sequence: SequenceLock::default(),
        target_off: Vec::new(),
        cycle_keep: false,
        cycle_held: false,
        power_budget: 0,
        solved: false,
        attempts: 0,
    };
    assert_eq!(network_forward(&mut state), Err(ForwardError::Cycle));
}

/// A pipe pointing at a nonexistent node index is a malformed graph → `BadEdge`.
#[test]
fn bad_edge_is_detected() {
    let mut state = PuzzleState {
        nodes: vec![PuzzleNode::default()],
        pipes: vec![Pipe {
            from: 0,
            to: 9,
            state: PipeState::On,
            path: vec![],
            switch_cell: None,
        }],
        inputs: vec![0],
        door_targets: vec![0],
        sequence: SequenceLock::default(),
        target_off: Vec::new(),
        cycle_keep: false,
        cycle_held: false,
        power_budget: 0,
        solved: false,
        attempts: 0,
    };
    assert_eq!(network_forward(&mut state), Err(ForwardError::BadEdge));
}

#[test]
fn interact_rolls_back_on_cycle() {
    // a(input=1) -> b, plus a manual b->b style cycle introduced by a switch.
    // Build a->b (ok) and b->c, c->b (cycle among b,c). Toggling the a->b
    // switch re-runs forward which now errors; the toggle must roll back.
    let mut state = PuzzleState {
        nodes: vec![
            PuzzleNode {
                threshold: 0,
                is_input: true,
                is_output: false,
                activation: 0,
            }, // a
            PuzzleNode {
                threshold: 1,
                is_input: false,
                is_output: false,
                activation: 0,
            }, // b
            PuzzleNode {
                threshold: 1,
                is_input: false,
                is_output: false,
                activation: 0,
            }, // c
        ],
        pipes: vec![
            Pipe {
                from: 0,
                to: 1,
                state: PipeState::Off,
                path: vec![(1, 1, 1)],
                switch_cell: Some(0),
            },
            Pipe {
                from: 1,
                to: 2,
                state: PipeState::On,
                path: vec![],
                switch_cell: None,
            },
            Pipe {
                from: 2,
                to: 1,
                state: PipeState::On,
                path: vec![],
                switch_cell: None,
            },
        ],
        inputs: vec![1, 0, 0],
        door_targets: vec![1],
        sequence: SequenceLock::default(),
        target_off: Vec::new(),
        cycle_keep: false,
        cycle_held: false,
        power_budget: 0,
        solved: false,
        attempts: 0,
    };
    // Initial forward already cyclic among b,c; but for the rollback test we
    // only care that interact returns false and leaves the pipe state intact.
    let before = state.pipes[0].state;
    let changed = puzzle_interact(&mut state, 1, 1, 1);
    assert!(!changed, "interact on a cyclic graph must report no change");
    assert_eq!(state.pipes[0].state, before, "toggle rolled back");
    assert_eq!(state.attempts, 0);
}

// ---------------------------------------------------------------------------
// Bevy plugin smoke test
// ---------------------------------------------------------------------------

/// Driving the plugin end-to-end: insert an L2 network, send the three winning
/// `InteractAt` messages, and assert the `Solved` resource flips to true.
#[test]
fn plugin_solves_via_messages() {
    use bevy::app::App;

    let level = load(level_path("02.lvl")).expect("02.lvl");
    let state = PuzzleState::from_level(&level).expect("build L2");

    // Collect the winning switch cells (R->mix, G->mix, mix->output).
    let idx = |n: &str| level.node_index(n).unwrap();
    let (r, g, mix, output) = (idx("R"), idx("G"), idx("mix"), idx("output"));
    let switch = |from: usize, to: usize| -> InteractAt {
        let (x, y, z) = state
            .pipes
            .iter()
            .find(|p| p.from == from && p.to == to)
            .and_then(|p| p.switch_coord())
            .unwrap();
        InteractAt::cell(x, y, z)
    };
    let m1 = switch(r, mix);
    let m2 = switch(g, mix);
    let m3 = switch(mix, output);

    let mut app = App::new();
    app.add_plugins(PuzzlePlugin);
    app.insert_resource(state);
    app.insert_resource(Solved(false));

    app.world_mut().write_message(m1);
    app.world_mut().write_message(m2);
    app.world_mut().write_message(m3);
    app.update();

    assert!(
        app.world().resource::<Solved>().0,
        "plugin should report solved after the three winning interactions"
    );
    assert!(app.world().resource::<PuzzleState>().solved);
}

// ---------------------------------------------------------------------------
// Floor-zone input: set_input_value (the player-position-as-input seam)
// ---------------------------------------------------------------------------

/// Build the smallest L5-shaped network in code: two inputs (`present`, `a`)
/// into a hidden detector (T=2), into an output (T=1). Both feed-pipes ON so the
/// only gate on the hidden unit is whether BOTH inputs are 1.
fn footstep_state() -> PuzzleState {
    let nodes = vec![
        PuzzleNode {
            is_input: true,
            ..Default::default()
        }, // 0 = present
        PuzzleNode {
            is_input: true,
            ..Default::default()
        }, // 1 = a
        PuzzleNode {
            threshold: 2,
            ..Default::default()
        }, // 2 = hidden
        PuzzleNode {
            threshold: 1,
            is_output: true,
            ..Default::default()
        }, // 3 = output
    ];
    let pipe = |from: usize, to: usize| Pipe {
        from,
        to,
        state: PipeState::On,
        ..Default::default()
    };
    let pipes = vec![pipe(0, 2), pipe(1, 2), pipe(2, 3)];
    let mut state = PuzzleState {
        nodes,
        pipes,
        inputs: vec![0, 1, 0, 0], // present=0 (off zone), a=1
        door_targets: vec![3],
        sequence: SequenceLock::default(),
        target_off: Vec::new(),
        cycle_keep: false,
        cycle_held: false,
        power_budget: 0,
        solved: false,
        attempts: 0,
    };
    network_forward(&mut state).expect("DAG");
    state.solved = check_win(&state, &state.door_targets);
    state
}

#[test]
fn zone_input_drives_win_in_and_out() {
    let mut s = footstep_state();
    // present=0 → hidden has only `a` (sum 1 < 2) → output off → not solved.
    assert!(!s.solved, "off the zone, the network must NOT be solved");
    assert_eq!(s.nodes[3].activation, 0);

    // Step INTO the zone: present → 1. Now sum = 2 ≥ 2 → hidden fires → output.
    let changed = set_input_value(&mut s, 0, 1);
    assert!(changed, "stepping into the zone changes present 0→1");
    assert_eq!(
        s.nodes[2].activation, 1,
        "hidden fires with both inputs high"
    );
    assert!(s.solved, "standing in the warm zone solves the level");

    // Setting it to 1 again is a no-op (the gate must not re-fire endlessly).
    assert!(!set_input_value(&mut s, 0, 1), "unchanged value is a no-op");
    assert!(s.solved);

    // Step OUT: present → 0. The hidden unit drops below threshold; output dark.
    let changed = set_input_value(&mut s, 0, 0);
    assert!(changed, "stepping out changes present 1→0");
    assert_eq!(s.nodes[2].activation, 0);
    assert!(!s.solved, "stepping off the zone un-solves the level");
}

// ---------------------------------------------------------------------------
// Sequence-lock win mode (L11 "Inner Circuit")
// ---------------------------------------------------------------------------

/// A hand-built sequence-lock fixture (no parser): three inputs s1,s2,s3 (=1)
/// each feeding its own output bit o1,o2,o3 (T=1) through one switchable pipe.
/// The 3-bit output window {o1,o2,o3} is observed by an ordered lock whose three
/// DISTINCT target states are P1=only o1, P2=only o2, P3=only o3. All pipes
/// start OFF (so at load every output is dark — no pattern matches, no auto-lock).
///
/// Nodes: 0=s1 1=s2 2=s3 (inputs) · 3=o1 4=o2 5=o3 (outputs, T1).
/// Pipes:  0: s1->o1  1: s2->o2  2: s3->o3  (switch cells at distinct cells).
fn seq_lock_fixture() -> PuzzleState {
    let input = || PuzzleNode {
        is_input: true,
        ..Default::default()
    };
    let out = || PuzzleNode {
        threshold: 1,
        is_output: true,
        ..Default::default()
    };
    let nodes = vec![input(), input(), input(), out(), out(), out()];
    let pipe = |from: usize, to: usize, cell: Coord3| Pipe {
        from,
        to,
        state: PipeState::Off,
        path: vec![cell],
        switch_cell: Some(0),
    };
    let pipes = vec![
        pipe(0, 3, (10, 0, 0)), // s1->o1
        pipe(1, 4, (10, 1, 0)), // s2->o2
        pipe(2, 5, (10, 2, 0)), // s3->o3
    ];
    let pattern = |on: usize| SeqPattern {
        bits: vec![(3, on == 3), (4, on == 4), (5, on == 5)],
    };
    let sequence = SequenceLock {
        patterns: vec![pattern(3), pattern(4), pattern(5)], // o1, then o2, then o3
        progress: 0,
    };
    let mut state = PuzzleState {
        nodes,
        pipes,
        inputs: vec![1, 1, 1, 0, 0, 0],
        door_targets: vec![5], // final pattern's lit bit (lvlcheck reachability)
        sequence,
        target_off: Vec::new(),
        cycle_keep: false,
        cycle_held: false,
        power_budget: 0,
        solved: false,
        attempts: 0,
    };
    network_forward(&mut state).expect("DAG");
    state.refresh_solved();
    state
}

/// Toggle the switch at the i-th pipe's switch cell.
fn toggle_pipe(state: &mut PuzzleState, pipe: usize) {
    let (x, y, z) = state.pipes[pipe].switch_coord().unwrap();
    assert!(puzzle_interact(state, x, y, z), "toggle pipe {pipe}");
}

#[test]
fn seq_lock_inactive_at_load() {
    let s = seq_lock_fixture();
    assert!(s.sequence.is_active(), "fixture declares a sequence");
    assert_eq!(
        s.sequence.progress, 0,
        "no pattern matches the all-OFF start"
    );
    assert!(!s.solved, "an unstarted lock is not solved");
}

/// Driving P1→P2→P3 IN ORDER wins; each pattern locks as the output matches it.
#[test]
fn seq_lock_ordered_drive_wins() {
    let mut s = seq_lock_fixture();

    // P1: light o1 only (s1->o1 ON). Output = (1,0,0) == P1 → lock #1.
    toggle_pipe(&mut s, 0);
    assert_eq!(s.sequence.progress, 1, "P1 locks when o1 fires alone");
    assert!(!s.solved, "one of three patterns locked");

    // P2: o1 OFF, o2 ON. Output = (0,1,0) == P2 → lock #2. Earlier lock holds.
    toggle_pipe(&mut s, 0); // s1->o1 OFF
    assert_eq!(
        s.sequence.progress, 1,
        "turning o1 off does NOT advance to P2 yet"
    );
    toggle_pipe(&mut s, 1); // s2->o2 ON
    assert_eq!(s.sequence.progress, 2, "P2 locks when o2 fires alone");
    assert!(!s.solved);

    // P3: o2 OFF, o3 ON. Output = (0,0,1) == P3 → lock #3 → WON.
    toggle_pipe(&mut s, 1); // s2->o2 OFF
    toggle_pipe(&mut s, 2); // s3->o3 ON
    assert_eq!(s.sequence.progress, 3, "P3 locks → all three locked");
    assert!(s.solved, "the whole ordered sequence is solved");
}

/// Reaching P3's state FIRST (out of order) does NOT win — only P1 can lock
/// while progress==0, and P3's state does not match P1, so nothing advances.
#[test]
fn seq_lock_out_of_order_does_not_win() {
    let mut s = seq_lock_fixture();

    // Jump straight to the P3 output state (0,0,1) without ever hitting P1/P2.
    toggle_pipe(&mut s, 2); // s3->o3 ON  → output (0,0,1)
    assert_eq!(
        s.sequence.progress, 0,
        "P3's state can't lock while P1 is current"
    );
    assert!(!s.solved, "out-of-order final state must NOT win");

    // Even reaching P2's state next (still skipping P1) cannot advance.
    toggle_pipe(&mut s, 2); // o3 OFF
    toggle_pipe(&mut s, 1); // o2 ON → output (0,1,0) == P2, but P1 is current
    assert_eq!(
        s.sequence.progress, 0,
        "P2's state can't lock while P1 is current"
    );
    assert!(!s.solved);
}

/// A partial sequence (only P1 driven) is not a win; and a locked step STAYS
/// locked when a LATER step's configuration is wrong — the whole sequence is
/// never reset (anti-suffocation).
#[test]
fn seq_lock_partial_and_locked_stays_locked() {
    let mut s = seq_lock_fixture();

    // Lock P1.
    toggle_pipe(&mut s, 0); // o1 ON
    assert_eq!(s.sequence.progress, 1);
    assert!(!s.solved, "only P1 locked — not won");

    // Now thrash the output into states that match NEITHER P2 (the current
    // step) NOR P1, never passing through P2's exact (0,1,0). Progress holds at 1.
    toggle_pipe(&mut s, 1); // o2 ON → output (1,1,0): not P2 (needs o1 OFF)
    assert_eq!(s.sequence.progress, 1, "wrong P2 config does not advance");
    toggle_pipe(&mut s, 2); // o3 ON → output (1,1,1): still not P2
    assert_eq!(
        s.sequence.progress, 1,
        "the P1 lock is NOT reset by a later miss"
    );
    // Now clear o2 and o3 back to just P1's state (1,0,0) — re-matching an
    // ALREADY-LOCKED pattern must NOT advance (locks are monotone, never re-fire).
    toggle_pipe(&mut s, 1); // o2 OFF
    toggle_pipe(&mut s, 2); // o3 OFF → output (1,0,0) == P1 again
    assert_eq!(
        s.sequence.progress, 1,
        "re-hitting locked P1 does not advance to P2"
    );
    assert!(!s.solved);
    toggle_pipe(&mut s, 0); // o1 OFF → output (0,0,0): matches no pattern

    // Re-driving the correct P2 then P3 still wins — the early lock was kept.
    toggle_pipe(&mut s, 1); // o2 ON → (0,1,0) == P2 → lock #2
    assert_eq!(s.sequence.progress, 2);
    toggle_pipe(&mut s, 1); // o2 OFF
    toggle_pipe(&mut s, 2); // o3 ON → (0,0,1) == P3 → win
    assert_eq!(s.sequence.progress, 3);
    assert!(
        s.solved,
        "the kept locks let the player finish from where they were"
    );
}

/// A full reset rewinds the sequence-lock progress to zero (back to P1).
#[test]
fn seq_lock_reset_rewinds_progress() {
    let mut s = seq_lock_fixture();
    let snapshot = s.pipe_states();

    toggle_pipe(&mut s, 0); // lock P1
    assert_eq!(s.sequence.progress, 1);

    s.reset_to(&snapshot);
    assert_eq!(
        s.sequence.progress, 0,
        "reset rewinds the lock to the start"
    );
    assert!(!s.solved);
    // Pipes restored to all-OFF.
    assert!(s.pipes.iter().all(|p| p.state == PipeState::Off));
}

/// Once won, the lock holds solved — extra interactions don't un-win it (a
/// solved lock stays at progress == len).
#[test]
fn seq_lock_stays_won() {
    let mut s = seq_lock_fixture();
    toggle_pipe(&mut s, 0); // P1
    toggle_pipe(&mut s, 0); // o1 off
    toggle_pipe(&mut s, 1); // P2
    toggle_pipe(&mut s, 1); // o2 off
    toggle_pipe(&mut s, 2); // P3 → won
    assert!(s.solved);
    assert_eq!(s.sequence.progress, 3);

    // Thrash the output after winning; progress is already at len, so it holds.
    toggle_pipe(&mut s, 0); // o1 on
    assert_eq!(s.sequence.progress, 3, "progress holds at len once won");
    assert!(s.solved, "a solved lock stays solved");
}

/// The default single-target path is byte-identical when no sequence is
/// declared: `refresh_solved` reduces exactly to `check_win(door_targets)`.
#[test]
fn no_sequence_uses_single_target_win() {
    let mut s = and_gate_fixture();
    assert!(
        !s.sequence.is_active(),
        "AND-gate fixture declares no sequence"
    );
    // Drive the AND gate to its win; refresh_solved must agree with check_win.
    puzzle_interact(&mut s, 5, 1, 1);
    puzzle_interact(&mut s, 6, 1, 1);
    assert!(s.solved, "single-target win still fires with no sequence");
    assert_eq!(
        s.solved,
        check_win(&s, &s.door_targets),
        "refresh_solved == check_win when no sequence is active"
    );
}

#[test]
fn set_input_value_rejects_non_input_and_oob() {
    let mut s = footstep_state();
    // Node 2 (hidden) is not an input → refused, no change.
    assert!(!set_input_value(&mut s, 2, 1));
    assert_eq!(s.inputs[2], 0, "a non-input node's input is never written");
    // Out-of-range index → refused.
    assert!(!set_input_value(&mut s, 99, 1));
    // Any non-zero value normalises to 1.
    assert!(set_input_value(&mut s, 0, 7));
    assert_eq!(s.inputs[0], 1, "non-zero normalises to 1");
}

// ===========================================================================
// L12 finale — inverted target (`exit 0`) + cycle-keep (`while(true)`)
// ===========================================================================

// --------------------------------------------------------------------------
// Inverted target ("must-be-0") — the `exit 0` ending
// --------------------------------------------------------------------------

/// The L12 `exit 0` shape, hand-built (no parser):
///
///   gather(input=1) ──[g->rec]──▶ REC(T=1, output)
///                    └─[g->exit]─▶ EXIT(T=1, output)
///
/// Win condition: EXIT must FIRE (`door_targets = [EXIT]`) AND REC must NOT
/// fire (`target_off = [REC]`). The player PRUNES `g->rec` (so REC stays dark)
/// while routing signal to EXIT (`g->exit` ON). Both pipes start ON, so at load
/// REC fires ⇒ NOT won; the solve is to turn `g->rec` OFF.
///
/// Nodes: 0=gather(in=1) · 1=REC(T1,out) · 2=EXIT(T1,out).
/// Pipes: 0: gather->REC (switch @ (1,0,0))  · 1: gather->EXIT (switch @ (2,0,0)).
fn exit0_fixture() -> PuzzleState {
    let nodes = vec![
        PuzzleNode {
            is_input: true,
            ..Default::default()
        }, // 0 gather
        PuzzleNode {
            threshold: 1,
            is_output: true,
            ..Default::default()
        }, // 1 REC
        PuzzleNode {
            threshold: 1,
            is_output: true,
            ..Default::default()
        }, // 2 EXIT
    ];
    let pipe = |from: usize, to: usize, cell: Coord3| Pipe {
        from,
        to,
        state: PipeState::On,
        path: vec![cell],
        switch_cell: Some(0),
    };
    let pipes = vec![pipe(0, 1, (1, 0, 0)), pipe(0, 2, (2, 0, 0))];
    let mut state = PuzzleState {
        nodes,
        pipes,
        inputs: vec![1, 0, 0],
        door_targets: vec![2], // EXIT must fire
        target_off: vec![1],   // REC must NOT fire (inverted)
        cycle_keep: false,
        cycle_held: false,
        sequence: SequenceLock::default(),
        power_budget: 0,
        solved: false,
        attempts: 0,
    };
    network_forward(&mut state).expect("DAG");
    state.refresh_solved();
    state
}

/// At load both pipes are ON ⇒ REC fires ⇒ the inverted win is NOT satisfied,
/// even though EXIT also fires. The player must prune REC.
#[test]
fn inverted_target_not_won_while_rec_fires() {
    let s = exit0_fixture();
    assert_eq!(s.nodes[1].activation, 1, "REC fires while g->rec is ON");
    assert_eq!(s.nodes[2].activation, 1, "EXIT fires too");
    assert!(
        !s.solved,
        "EXIT lit but REC also lit ⇒ inverted win NOT satisfied"
    );
    // The combined check agrees: positive half passes, inverted half fails.
    assert!(
        check_win(&s, &s.door_targets),
        "EXIT (positive half) is lit"
    );
    assert!(
        !check_win_inverted(&s, &s.door_targets, &s.target_off),
        "inverted check fails while REC fires"
    );
}

/// Pruning `g->rec` (turning it OFF) makes REC go dark while EXIT stays lit ⇒
/// the inverted `exit 0` win fires. "Prune yourself out."
#[test]
fn inverted_target_wins_when_rec_pruned() {
    let mut s = exit0_fixture();
    // Prune the self-feed: g->rec OFF (switch at (1,0,0)).
    assert!(puzzle_interact(&mut s, 1, 0, 0));
    assert_eq!(s.nodes[1].activation, 0, "REC dark after pruning g->rec");
    assert_eq!(s.nodes[2].activation, 1, "EXIT still fires via g->exit");
    assert!(s.solved, "REC off + EXIT on ⇒ exit 0 win");
    assert!(check_win_inverted(&s, &s.door_targets, &s.target_off));
}

/// Pruning EXIT instead (REC still lit) does NOT win — the positive half fails.
/// And re-lighting REC after a prune un-wins (the inverted constraint is live).
#[test]
fn inverted_target_loses_if_exit_dark_or_rec_relit() {
    let mut s = exit0_fixture();
    // Turn EXIT's feed OFF (switch (2,0,0)): EXIT dark, REC still lit ⇒ lose.
    assert!(puzzle_interact(&mut s, 2, 0, 0));
    assert_eq!(s.nodes[2].activation, 0, "EXIT dark");
    assert!(!s.solved, "no EXIT ⇒ no win (positive half fails)");

    // Now also prune REC: REC dark, but EXIT also dark ⇒ still lose.
    assert!(puzzle_interact(&mut s, 1, 0, 0));
    assert!(!s.solved, "both dark ⇒ no win (need EXIT lit)");

    // Re-route EXIT back on: REC dark + EXIT lit ⇒ now win.
    assert!(puzzle_interact(&mut s, 2, 0, 0));
    assert!(s.solved, "REC off + EXIT on ⇒ win");

    // Re-light REC: the inverted constraint un-wins it.
    assert!(puzzle_interact(&mut s, 1, 0, 0));
    assert_eq!(s.nodes[1].activation, 1, "REC re-lit");
    assert!(!s.solved, "REC firing again breaks the inverted win");
}

/// An EMPTY `target_off` makes `check_win_inverted` reduce EXACTLY to
/// `check_win` — the default win is byte-identical (no behaviour drift).
#[test]
fn inverted_check_reduces_to_check_win_when_no_target_off() {
    // Use the plain AND-gate fixture (no inverted targets).
    let mut s = and_gate_fixture();
    assert!(s.target_off.is_empty(), "AND-gate has no inverted targets");
    // Across the whole solve, the two checks must agree at every step.
    for &(x, y, z) in &[(5, 1, 1), (6, 1, 1)] {
        assert_eq!(
            check_win(&s, &s.door_targets),
            check_win_inverted(&s, &s.door_targets, &s.target_off),
            "inverted == plain when target_off empty (pre-toggle)"
        );
        puzzle_interact(&mut s, x, y, z);
    }
    assert!(s.solved);
    assert_eq!(
        check_win(&s, &s.door_targets),
        check_win_inverted(&s, &s.door_targets, &s.target_off),
        "inverted == plain when target_off empty (post-solve)"
    );
}

// --------------------------------------------------------------------------
// Cycle-keep ("while(true)") — the loop stands
// --------------------------------------------------------------------------

/// A 3-node ring `rec -> mem -> rec` plus an entry input, built so the ring is
/// closed by ONE switchable pipe. With `cycle_keep == false` this is the
/// historical rollback; with `cycle_keep == true` closing the loop is the win.
///
/// Nodes: 0=seed(in=1) · 1=REC(T1) · 2=MEM(T1, the loop's other half).
/// Pipes:
///   0: seed->REC  (ON, switchless — REC has a live in-edge so the ring forms)
///   1: REC->MEM   (ON, switchless)
///   2: MEM->REC   (switchable @ (3,0,0)) — the pipe that CLOSES the ring.
/// Starting with pipe 2 OFF, the graph is the acyclic chain seed->REC->MEM.
fn cycle_fixture(cycle_keep: bool) -> PuzzleState {
    let nodes = vec![
        PuzzleNode {
            is_input: true,
            ..Default::default()
        }, // 0 seed
        PuzzleNode {
            threshold: 1,
            is_output: true,
            ..Default::default()
        }, // 1 REC
        PuzzleNode {
            threshold: 1,
            ..Default::default()
        }, // 2 MEM
    ];
    let on = |from: usize, to: usize| Pipe {
        from,
        to,
        state: PipeState::On,
        path: vec![],
        switch_cell: None,
    };
    let pipes = vec![
        on(0, 1),
        on(1, 2),
        Pipe {
            from: 2,
            to: 1,
            state: PipeState::Off, // ring OPEN at start
            path: vec![(3, 0, 0)],
            switch_cell: Some(0),
        },
    ];
    let mut state = PuzzleState {
        nodes,
        pipes,
        inputs: vec![1, 0, 0],
        // No positive door target: the ONLY win for this fixture is the kept
        // cycle (the `while(true)` ending). So at the open-ring start nothing is
        // won, and closing the loop is the sole win path — isolating the
        // cycle-keep behaviour under test.
        door_targets: vec![],
        target_off: Vec::new(),
        cycle_keep,
        cycle_held: false,
        sequence: SequenceLock::default(),
        power_budget: 0,
        solved: false,
        attempts: 0,
    };
    // With cycle_keep ON the open-ring start is an acyclic chain (pipe 2 OFF ⇒
    // no edge). With cycle_keep OFF the MEM->REC + REC->MEM edges are a
    // topological cycle regardless of state, so the start ALREADY errors — that
    // is the historical behaviour the flag-off test asserts, so we don't
    // `expect` success here.
    let _ = network_forward(&mut state);
    state.refresh_solved();
    state
}

/// Default (`cycle_keep == false`): a wired loop is a `Cycle` error and any
/// interaction on that graph ROLLS BACK — byte-identical to today's behaviour.
/// (With state-independent in-degree the ring is cyclic whatever the pipe state,
/// so the closed-ring graph errors directly.)
#[test]
fn cycle_rolls_back_when_flag_off() {
    let mut s = cycle_fixture(false);
    assert!(!s.cycle_held, "cycle-keep off never holds a cycle");
    // The graph is topologically cyclic (REC<->MEM) with the flag off, so a raw
    // forward pass errors — the classic rollback trigger.
    assert_eq!(network_forward(&mut s), Err(ForwardError::Cycle));
    // Any toggle on this cyclic graph rolls back (no state change, no attempt).
    let before = s.pipes[2].state;
    let changed = puzzle_interact(&mut s, 3, 0, 0);
    assert!(!changed, "interacting on a cyclic graph rolls back");
    assert_eq!(s.pipes[2].state, before, "pipe restored");
    assert_eq!(s.attempts, 0, "a rolled-back toggle counts no attempt");
    assert!(!s.cycle_held);
}

/// Cycle-keep (`cycle_keep == true`): closing the ring does NOT error — the loop
/// STANDS as a settled terminal state and the puzzle is won (`while(true)`).
#[test]
fn cycle_keep_makes_loop_stand_and_win() {
    let mut s = cycle_fixture(true);
    assert!(
        !s.cycle_held,
        "ring open at start (pipe 2 OFF ⇒ acyclic chain)"
    );
    assert!(
        !s.solved,
        "no win at the open-ring start (no door target either)"
    );

    // Close the ring: MEM->REC ON (switch at (3,0,0)).
    let changed = puzzle_interact(&mut s, 3, 0, 0);
    assert!(changed, "closing the ring with cycle_keep ON is accepted");
    assert!(s.cycle_held, "the kept cycle is recorded");
    assert!(s.solved, "a kept cycle that stands is the while(true) win");
    assert_eq!(s.attempts, 1, "the accepted toggle counts an attempt");

    // The settled state is deterministic: the cyclic nodes latch ON.
    assert_eq!(s.nodes[1].activation, 1, "REC latches in the kept cycle");
    assert_eq!(s.nodes[2].activation, 1, "MEM latches in the kept cycle");
}

/// The kept-cycle forward pass TERMINATES and is DETERMINISTIC: running it many
/// times over the same closed-ring config yields identical activations and the
/// same held flag, with no hang.
#[test]
fn cycle_keep_forward_terminates_and_is_deterministic() {
    let mut s = cycle_fixture(true);
    s.pipes[2].state = PipeState::On; // closed ring

    // First pass.
    assert!(network_forward(&mut s).is_ok(), "kept cycle does not error");
    assert!(s.cycle_held);
    let snapshot: Vec<i32> = s.nodes.iter().map(|n| n.activation).collect();

    // Re-run several times: identical result, always terminating.
    for _ in 0..32 {
        assert!(network_forward(&mut s).is_ok());
        assert!(s.cycle_held, "held flag stable across repeats");
        let again: Vec<i32> = s.nodes.iter().map(|n| n.activation).collect();
        assert_eq!(again, snapshot, "kept-cycle activations are deterministic");
    }
}

/// Re-opening the ring after it stood drops `cycle_held` and re-evaluates as an
/// ordinary acyclic network (the held cycle is a property of the CURRENT graph).
#[test]
fn cycle_keep_reopening_ring_clears_held() {
    let mut s = cycle_fixture(true);
    assert!(puzzle_interact(&mut s, 3, 0, 0)); // close ring → held
    assert!(s.cycle_held);

    assert!(puzzle_interact(&mut s, 3, 0, 0)); // re-open ring
    assert!(
        !s.cycle_held,
        "re-opening the ring clears the kept-cycle flag"
    );
    // Acyclic again: forward pass is a normal Kahn pass with no remainder.
    assert!(network_forward(&mut s).is_ok());
    assert!(!s.cycle_held);
}

/// `reset_to` rewinds a stood cycle: restoring the open-ring snapshot clears the
/// held flag and re-evaluates the acyclic baseline.
#[test]
fn cycle_keep_reset_clears_held() {
    let mut s = cycle_fixture(true);
    let snapshot = s.pipe_states(); // ring open
    assert!(puzzle_interact(&mut s, 3, 0, 0)); // close → held
    assert!(s.cycle_held);

    s.reset_to(&snapshot);
    assert!(
        !s.cycle_held,
        "reset to the open-ring config clears the held cycle"
    );
    assert_eq!(s.pipes[2].state, PipeState::Off, "ring re-opened by reset");
}

/// The static solver agrees with the runtime on both L12 modes: an inverted-
/// target level is `Solvable` (you can prune to win) and a cycle-keep level is
/// `Solvable` (you can close the loop to win).
#[test]
fn solve_brute_force_handles_inverted_and_cycle_keep() {
    let mut inv = exit0_fixture();
    let inv_pipes_before = inv.pipe_states();
    match solve_brute_force(&mut inv) {
        Solvability::Solvable { .. } => {}
        other => panic!("inverted-target level should be Solvable, got {other:?}"),
    }
    // The solver restores the live state (no observable mutation).
    assert_eq!(
        inv.pipe_states(),
        inv_pipes_before,
        "solver restored pipe states"
    );

    let mut cyc = cycle_fixture(true);
    match solve_brute_force(&mut cyc) {
        Solvability::Solvable { .. } | Solvability::TriviallySolvedAtLoad => {}
        other => panic!("cycle-keep level should be winnable, got {other:?}"),
    }
}
