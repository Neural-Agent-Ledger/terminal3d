//! POWER BUDGET spec — `[room] power_budget = N`, an intentional in-scope
//! mechanic (not a later bolt-on).
//!
//! The rule that makes the campaign losable. [`network_forward`] has no negative
//! weights and no inhibition, so a node's activation is MONOTONE in the set of
//! ON pipes: turning a pipe ON can never turn a node off. Theorem: for any level
//! with no `[door] target_off` and no `[sequence]`, the all-switches-ON state
//! wins if any state wins — which is L1 through L10 of the campaign, and is also
//! the natural behaviour of a first-person player flipping every glowing switch
//! they walk past. [`all_pipes_on_wins_when_unbudgeted_the_monotonicity_defect`]
//! MEASURES that premise on a real shipped level; everything after it pins the
//! cap that fixes it.

use super::*;
use re_level::{load, PipeState};
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

/// A fixture with a declared budget: input `a` (=1) drives output `out` (T=1)
/// through TWO independent switchable pipes, whose switch cells are `(1,0,0)`
/// and `(2,0,0)`. Both start OFF, so the draw starts at 0. Either pipe alone
/// wins, so the only thing under test is the CAP, never the network math.
fn budget_fixture(budget: u32) -> PuzzleState {
    let mut state = PuzzleState {
        nodes: vec![
            PuzzleNode {
                threshold: 0,
                is_input: true,
                is_output: false,
                activation: 0,
            },
            PuzzleNode {
                threshold: 1,
                is_input: false,
                is_output: true,
                activation: 0,
            },
        ],
        pipes: vec![
            Pipe {
                from: 0,
                to: 1,
                state: PipeState::Off,
                path: vec![(0, 0, 0), (1, 0, 0)],
                switch_cell: Some(1),
            },
            Pipe {
                from: 0,
                to: 1,
                state: PipeState::Off,
                path: vec![(0, 0, 0), (2, 0, 0)],
                switch_cell: Some(1),
            },
        ],
        inputs: vec![1, 0],
        door_targets: vec![1],
        target_off: Vec::new(),
        sequence: SequenceLock::default(),
        cycle_keep: false,
        cycle_held: false,
        power_budget: budget,
        solved: false,
        attempts: 0,
    };
    network_forward(&mut state).expect("DAG");
    state.refresh_solved();
    state
}

// ---------------------------------------------------------------------------
// The premise
// ---------------------------------------------------------------------------

/// THE DEFECT, MEASURED. On a real shipped level with no `target_off` and no
/// `[sequence]`, switching EVERY pipe ON wins. That is why an unbudgeted
/// campaign cannot be lost, and it is the entire justification for this feature.
#[test]
fn all_pipes_on_wins_when_unbudgeted_the_monotonicity_defect() {
    let level = load(level_path("03.lvl")).expect("03.lvl");
    let mut state = PuzzleState::from_level(&level).expect("build 03");
    assert_eq!(state.power_budget, 0, "an archive level declares no budget");
    assert!(state.target_off.is_empty() && !state.sequence.is_active());
    assert!(!state.solved, "not solved at the authored all-OFF start");

    for pipe in &mut state.pipes {
        pipe.state = PipeState::On;
    }
    network_forward(&mut state).expect("DAG");
    state.refresh_solved();
    assert!(
        state.solved,
        "monotone activation: with every pipe ON the level wins — the defect \
         `power_budget` exists to fix"
    );
}

// ---------------------------------------------------------------------------
// Accounting
// ---------------------------------------------------------------------------

/// `power_used` counts every ON pipe (switchable or not); `power_available`
/// reports the cap honestly; `0` means UNLIMITED, not "a zero cap".
#[test]
fn power_accounting_and_the_zero_means_unlimited_rule() {
    let mut s = budget_fixture(0);
    assert_eq!(s.power_used(), 0);
    assert!(!s.has_power_budget(), "0 is UNLIMITED, not a zero cap");
    assert!(s.power_available());
    s.pipes[0].state = PipeState::On;
    s.pipes[1].state = PipeState::On;
    assert_eq!(s.power_used(), 2);
    assert!(s.power_available(), "unlimited never runs out");

    let mut b = budget_fixture(1);
    assert!(b.has_power_budget());
    assert!(b.power_available());
    b.pipes[0].state = PipeState::On;
    assert_eq!(b.power_used(), 1);
    assert!(!b.power_available(), "1 of 1 drawn ⇒ nothing left");
}

/// A switchless pipe still draws power. This is the L5 shape: `present->hidden`
/// has no switch and is permanently ON, so the room is already spending a unit
/// the player never chose. Counting it is what makes the HUD readout truthful.
#[test]
fn a_switchless_pipe_still_draws_power() {
    let mut s = budget_fixture(2);
    s.pipes[0].switch_cell = None;
    s.pipes[0].state = PipeState::On;
    assert_eq!(s.power_used(), 1, "the unswitchable wire is counted");
    assert!(s.power_available());
    assert!(
        puzzle_interact(&mut s, 2, 0, 0),
        "the one spare unit is spendable"
    );
    assert!(!s.power_available(), "and now the room is fully drawn");
}

// ---------------------------------------------------------------------------
// The rule
// ---------------------------------------------------------------------------

/// A budget of 1: the first ON succeeds, the second is REFUSED with a distinct,
/// self-describing outcome, and the refusal leaves the world byte-for-byte
/// unchanged (no pipe flip, no attempts bump, no forward pass, no solved change).
#[test]
fn budget_refuses_the_second_switch_and_changes_nothing() {
    let mut s = budget_fixture(1);
    let pristine = s.clone();

    // First ON: allowed (and it wins — `out` has threshold 1).
    let o1 = puzzle_interact_ex(&mut s, 1, 0, 0);
    assert_eq!(
        o1,
        InteractOutcome::Toggled {
            pipe: 0,
            now_on: true
        }
    );
    assert_eq!(s.power_used(), 1);
    assert_eq!(s.attempts, 1);

    // Second ON: refused — no power.
    let snapshot = s.clone();
    let o2 = puzzle_interact_ex(&mut s, 2, 0, 0);
    assert_eq!(
        o2,
        InteractOutcome::RefusedNoPower {
            pipe: 1,
            used: 1,
            budget: 1
        },
        "the refusal must be observable and must say WHY"
    );
    assert!(!o2.changed(), "a refusal is not a change");
    assert!(o2.hit_a_switch(), "but it DID land on a switch");
    assert_eq!(
        s.pipe_states(),
        snapshot.pipe_states(),
        "pipe states untouched"
    );
    assert_eq!(s.attempts, snapshot.attempts, "a refusal is not an attempt");
    assert_eq!(s.solved, snapshot.solved);
    assert_eq!(s.power_used(), 1);

    // The boolean seam still reports it exactly as "nothing happened".
    assert!(!puzzle_interact(&mut s, 2, 0, 0));

    // Sanity: the same fixture UNBUDGETED accepts both switches.
    let mut u = pristine;
    u.power_budget = 0;
    assert!(puzzle_interact(&mut u, 1, 0, 0));
    assert!(puzzle_interact(&mut u, 2, 0, 0));
    assert_eq!(u.power_used(), 2);
}

/// Switching a pipe OFF is NEVER refused, even at (or over) the cap. That is
/// what keeps the constrained state space CONNECTED — from any legal
/// configuration the player can reach any other by shutting wires down first —
/// and therefore what guarantees a budgeted level can never dead-end.
#[test]
fn switching_off_always_succeeds_and_frees_a_unit() {
    let mut s = budget_fixture(1);
    assert!(puzzle_interact(&mut s, 1, 0, 0), "pipe 0 ON");
    assert!(
        !puzzle_interact(&mut s, 2, 0, 0),
        "pipe 1 refused at the cap"
    );
    assert!(
        puzzle_interact(&mut s, 1, 0, 0),
        "pipe 0 OFF is always allowed"
    );
    assert_eq!(s.power_used(), 0);
    assert!(
        puzzle_interact(&mut s, 2, 0, 0),
        "the freed unit is now spendable"
    );
    assert_eq!(s.power_used(), 1);

    // Even a state that STARTS over budget can be dug out of: only ON is capped.
    let mut over = budget_fixture(1);
    over.pipes[0].state = PipeState::On;
    over.pipes[1].state = PipeState::On;
    assert_eq!(over.power_used(), 2, "authored draw exceeds the cap");
    assert!(puzzle_interact(&mut over, 1, 0, 0), "OFF is still allowed");
    assert_eq!(over.power_used(), 1);
}

/// Interacting with a cell that has no switch stays `NoTarget` — a full grid
/// never turns "there is nothing there" into "no power".
#[test]
fn budget_does_not_disturb_the_no_target_outcome() {
    let mut s = budget_fixture(1);
    s.pipes[0].state = PipeState::On; // fully drawn
    assert_eq!(
        puzzle_interact_ex(&mut s, 99, 99, 99),
        InteractOutcome::NoTarget
    );
}

/// A refused move must not advance a `[sequence]` lock. The budget check runs
/// BEFORE the toggle and before any forward pass, so a refusal cannot leak
/// progress into a state machine the player was not allowed to drive.
#[test]
fn a_refused_switch_cannot_advance_a_sequence_lock() {
    let level = load(fixture_path("sequence_lock_budget.lvl")).expect("fixture loads");
    let mut s = PuzzleState::from_level(&level).expect("fixture builds");
    s.power_budget = 1; // deliberately tighter than the fixture's shipped 4

    let idx = |n: &str| level.node_index(n).unwrap();
    let switch = |s: &PuzzleState, from: usize, to: usize| -> Coord3 {
        s.pipes
            .iter()
            .find(|p| p.from == from && p.to == to)
            .and_then(Pipe::switch_coord)
            .unwrap()
    };

    // Lock step 1 (out1 alone) with the single unit of power.
    let (x, y, z) = switch(&s, idx("in1"), idx("out1"));
    assert!(puzzle_interact(&mut s, x, y, z));
    assert_eq!(s.sequence.progress, 1);

    // Now every further ON is refused, and progress must not move.
    let (x, y, z) = switch(&s, idx("in2"), idx("out2"));
    assert!(matches!(
        puzzle_interact_ex(&mut s, x, y, z),
        InteractOutcome::RefusedNoPower { .. }
    ));
    assert_eq!(s.sequence.progress, 1, "a refused move advances nothing");
    assert!(!s.solved);
}

/// A `reset` restores the AUTHORED configuration verbatim, budget or no budget —
/// reset is not an interaction and is never refused.
#[test]
fn reset_restores_the_authored_configuration_under_a_budget() {
    let mut s = budget_fixture(1);
    let start = s.pipe_states();
    assert!(puzzle_interact(&mut s, 1, 0, 0));
    assert_eq!(s.power_used(), 1);
    s.reset_to(&start);
    assert_eq!(s.pipe_states(), start);
    assert_eq!(s.power_used(), 0);
    assert!(!s.solved);
}

// ---------------------------------------------------------------------------
// Static analysis
// ---------------------------------------------------------------------------

/// `solve_brute_force` must not report a winning configuration the budget
/// forbids — otherwise `re_lvlcheck` would green-light an unwinnable level.
#[test]
fn solve_brute_force_respects_the_budget() {
    // Two inputs into an output with threshold 2: this level genuinely needs
    // BOTH pipes hot, so a budget of 1 makes it unwinnable.
    let build = |budget: u32| {
        let mut st = PuzzleState {
            nodes: vec![
                PuzzleNode {
                    threshold: 0,
                    is_input: true,
                    is_output: false,
                    activation: 0,
                },
                PuzzleNode {
                    threshold: 0,
                    is_input: true,
                    is_output: false,
                    activation: 0,
                },
                PuzzleNode {
                    threshold: 2,
                    is_input: false,
                    is_output: true,
                    activation: 0,
                },
            ],
            pipes: vec![
                Pipe {
                    from: 0,
                    to: 2,
                    state: PipeState::Off,
                    path: vec![(1, 0, 0)],
                    switch_cell: Some(0),
                },
                Pipe {
                    from: 1,
                    to: 2,
                    state: PipeState::Off,
                    path: vec![(2, 0, 0)],
                    switch_cell: Some(0),
                },
            ],
            inputs: vec![1, 1, 0],
            door_targets: vec![2],
            target_off: Vec::new(),
            sequence: SequenceLock::default(),
            cycle_keep: false,
            cycle_held: false,
            power_budget: budget,
            solved: false,
            attempts: 0,
        };
        network_forward(&mut st).expect("DAG");
        st.refresh_solved();
        st
    };

    let mut ok = build(2);
    assert!(matches!(
        solve_brute_force(&mut ok),
        Solvability::Solvable { min_toggles: 2, .. }
    ));
    let mut starved = build(1);
    assert_eq!(
        solve_brute_force(&mut starved),
        Solvability::Unsolvable,
        "a budget below the level's true need must read as UNSOLVABLE, not as a \
         solution the player could never stand in"
    );

    // …and the stateful search agrees with the static verdict.
    assert_eq!(
        win_reachable_under_budget(&starved, 1),
        BudgetFeasibility::Unwinnable
    );
    assert_eq!(
        win_reachable_under_budget(&ok, 2),
        BudgetFeasibility::Winnable
    );
    assert_eq!(min_power_budget(&ok), Some(2));
}

/// The stateful search must beat the stateless sweep exactly where the sweep
/// COULD be wrong: a `[sequence]` lock. `solve_brute_force`'s own enumeration
/// advances the `SequenceLock` as a side effect, so on a graph where the
/// door-target-only minimum happens to differ from the true ordered-drive
/// minimum, only `win_reachable_under_budget` (real interactions) tells the
/// truth. On this fixture's rebuilt graph — `gate` is T2 (in1+in2) and is
/// load-bearing for both out2 and out3 — the two now agree at 4, so this
/// pins that agreement as well as the stateful path's correctness.
#[test]
fn stateful_search_handles_the_sequence_lock() {
    let level = load(fixture_path("sequence_lock_budget.lvl")).expect("fixture loads");
    let mut s = PuzzleState::from_level(&level).expect("fixture builds");
    assert_eq!(
        s.sequence.patterns.len(),
        3,
        "fixture declares 3 ordered patterns"
    );
    assert_eq!(s.power_budget, 4, "fixture ships with power_budget = 4");

    // The stateless sweep's answer against the door target (out3): 4 toggles
    // ({B, C, F, G}).
    let stateless = solve_brute_force(&mut s);
    assert!(
        matches!(stateless, Solvability::Solvable { min_toggles: 4, .. }),
        "door-target minimum under the rebuilt graph: {stateless:?}"
    );

    // The stateful truth: the whole BREAK->FALL->DARK ordered drive
    // (P1: +A, P2: -A+B+C+D+E, P3: -D-E+F+G) fits inside the 4-unit budget.
    assert_eq!(
        win_reachable_under_budget(&s, 4),
        BudgetFeasibility::Winnable
    );
}

/// **The no-regression proof.** A level with NO `power_budget` behaves EXACTLY
/// as it did before the rule existed: unlimited power, every switch accepted in
/// any order, the same solved state. Checked both as data (every non-campaign
/// `.lvl` still parses to `0`, so no golden can move) and behaviourally.
#[test]
fn a_level_without_a_budget_behaves_exactly_as_before() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../levels");
    let mut unbudgeted = 0usize;
    for entry in std::fs::read_dir(&dir).expect("levels/").flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("lvl") {
            continue;
        }
        let stem = path.file_stem().unwrap().to_string_lossy().to_string();
        // The twelve campaign rooms are `L<digit>…`; everything else is demo,
        // showcase or archive content whose goldens must not move.
        let is_campaign = stem
            .strip_prefix('L')
            .is_some_and(|rest| rest.starts_with(|c: char| c.is_ascii_digit()));
        let Ok(lv) = load(&path) else { continue };
        if is_campaign {
            assert!(
                lv.power_budget > 0,
                "{stem}: a campaign level must declare a budget"
            );
        } else {
            assert_eq!(
                lv.power_budget, 0,
                "{stem}: a non-campaign level must stay UNLIMITED (goldens depend on it)"
            );
            unbudgeted += 1;
        }
    }
    assert!(
        unbudgeted >= 40,
        "expected the shipped non-campaign corpus, saw {unbudgeted}"
    );

    // Behaviourally: `levels/03.lvl` (the un-budgeted archive twin of L3)
    // accepts every single switch, exactly as it always did.
    let level = load(level_path("03.lvl")).expect("03.lvl");
    let mut state = PuzzleState::from_level(&level).expect("build 03");
    assert_eq!(state.power_budget, 0);
    let switches: Vec<Coord3> = state.pipes.iter().filter_map(Pipe::switch_coord).collect();
    for (i, &(x, y, z)) in switches.iter().enumerate() {
        assert!(
            puzzle_interact(&mut state, x, y, z),
            "unbudgeted: switch {i} must always take"
        );
    }
    assert_eq!(
        state.power_used() as usize,
        switches.len(),
        "all wires hot, no refusals"
    );
    assert!(state.solved, "and the historical all-ON win still stands");
}
