//! `faced_switch`, `objective_progress`, `sequence_progress`, `power_readout`.
//! No seams: every function here is a pure read of a [`re_puzzle::PuzzleState`]
//! and/or a [`re_world::CameraPose`], with no Bevy and no terminal. A host's
//! HUD (the in-play overlay, the crosshair) calls these every frame to know
//! what to paint; this module never paints anything itself.

use re_puzzle::PuzzleState;
use re_world::CameraPose;

/// The puzzle objective progress: how many door-target nodes are currently
/// activated, out of the total the level needs lit to win. Computed purely from
/// the live [`PuzzleState`] (the cached `activation` set by the last forward
/// pass), so it tracks every interaction with no extra bookkeeping.
///
/// `(activated, total)`. `total == 0` means the level has no win condition (an
/// empty network / menu state); a HUD can treat that as "nothing to do".
pub fn objective_progress(puzzle: &PuzzleState) -> (usize, usize) {
    let total = puzzle.door_targets.len();
    let activated = puzzle
        .door_targets
        .iter()
        .filter(|&&idx| puzzle.nodes.get(idx).is_some_and(|n| n.activation == 1))
        .count();
    (activated, total)
}

/// Sequence-lock progress for a HUD readout: `Some((locked, total))` when the
/// level declares a `[sequence]` lock (`locked` patterns of `total` are
/// confirmed), or `None` for an ordinary single-target level (a HUD then falls
/// back to [`objective_progress`]). Cheap: a pure read of the live
/// [`re_puzzle::SequenceLock`] the forward pass already maintains — no extra
/// bookkeeping, no re-evaluation.
pub fn sequence_progress(puzzle: &PuzzleState) -> Option<(usize, usize)> {
    if puzzle.sequence.is_active() {
        Some((puzzle.sequence.progress, puzzle.sequence.patterns.len()))
    } else {
        None
    }
}

/// The POWER readout for a HUD: `Some((used, budget))` when the level declares
/// a `[room] power_budget`, or `None` when power is unlimited (the pre-rule
/// default, and the state of every level with no budget declared, whose HUD is
/// therefore byte-identical to before this feature existed).
///
/// The constraint has to be visible BEFORE the player hits it — a cap that only
/// announces itself by silently refusing a switch reads as a broken switch, not
/// as a rule. Pure read of the live [`PuzzleState`]; no bookkeeping.
pub fn power_readout(puzzle: &PuzzleState) -> Option<(u32, u32)> {
    puzzle
        .has_power_budget()
        .then(|| (puzzle.power_used(), puzzle.power_budget))
}

/// Resolve the interact TARGET cell the camera faces, with whether it lands on
/// an interactable pipe switch. A host shares this between its own interact
/// path and its HUD crosshair, so both agree on "is the player looking at
/// something interactable".
///
/// `dir = (sin yaw, cos yaw)` (the `re_world` convention): adding one unit of
/// `dir` to the position and flooring lands on the adjacent cell. The faced
/// `(x, y)` footprint is probed across `[z, z+1, z-1]` (the standing layer first,
/// then one up, then one down) — a switch one layer above/below the standing
/// layer is still reachable. Returns `(x, y, z, lands)`; `lands` is the crosshair
/// "active" predicate.
pub fn faced_switch(cam: &CameraPose, puzzle: &PuzzleState) -> (i32, i32, i32, bool) {
    let (dx, dy) = cam.dir();
    let x = (cam.pos.x + dx).floor() as i32;
    let y = (cam.pos.y + dy).floor() as i32;
    let z = cam.pos.z.floor() as i32;
    for &dz in &[0, 1, -1] {
        let cz = z + dz;
        if puzzle.pipe_at_switch(x, y, cz).is_some() {
            return (x, y, cz, true);
        }
    }
    (x, y, z, false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use re_world::CameraPose;

    /// (#7) The objective progress counts door-target nodes that are currently
    /// lit, out of the total — read straight from the live `PuzzleState`.
    /// Solving the network must drive the lit count up to the total. Uses
    /// `03.lvl` (an archive/example fixture, not a campaign level).
    #[test]
    fn objective_progress_reflects_activation() {
        let level = re_level::load("../../levels/03.lvl").expect("03.lvl loads");
        let mut state = PuzzleState::from_level(&level).expect("03.lvl network builds");

        let (lit0, total) = objective_progress(&state);
        assert!(total >= 1, "03.lvl has at least one door target");
        assert!(
            lit0 < total,
            "03.lvl starts with the objective unmet ({lit0}/{total})"
        );

        // Solve it through the public interaction path.
        let idx = |n: &str| level.node_index(n).unwrap();
        for (from, to) in [("R", "h1"), ("B", "h1"), ("h1", "output")] {
            let (x, y, z) = state
                .pipes
                .iter()
                .find(|p| p.from == idx(from) && p.to == idx(to))
                .and_then(|p| p.switch_coord())
                .unwrap();
            re_puzzle::puzzle_interact(&mut state, x, y, z);
        }
        assert!(state.solved, "03.lvl solved");

        let (lit1, total1) = objective_progress(&state);
        assert_eq!(total1, total, "the total door-target count is stable");
        assert_eq!(
            lit1, total1,
            "every door target is lit once solved ({lit1}/{total1})"
        );
    }

    /// (#3) The crosshair "active" predicate (`faced_switch(..).3`) is TRUE when
    /// the camera faces an interactable switch, and FALSE for empty space —
    /// reusing the exact faced-cell logic an interact path uses. Uses `01.lvl`
    /// (an archive/example fixture, not a campaign level).
    #[test]
    fn crosshair_active_only_when_facing_a_switch() {
        let level = re_level::load("../../levels/01.lvl").expect("01.lvl loads");
        let puzzle = PuzzleState::from_level(&level).expect("01.lvl network builds");
        // 01.lvl's switch is at (6,3,2); the player stands on z=1 (R-M1 fixture).
        assert_eq!(
            puzzle.pipe_at_switch(6, 3, 2),
            Some(0),
            "fixture: switch at (6,3,2)"
        );

        // Face (6,3) standing on z=1 → the probe climbs to z=2 and reports a landing.
        let mut facing = CameraPose::spawn_at(6, 3, 1, 0); // yaw 0 = +y
        facing.pos.y -= 1.0; // stand one cell south so pos+dir lands on (6,3)
        let (_, _, _, active_on) = faced_switch(&facing, &puzzle);
        assert!(active_on, "reticle is ACTIVE when facing the switch");

        // Face an empty corner → idle.
        let mut empty = CameraPose::spawn_at(0, 0, 1, 0);
        empty.pos.y -= 1.0;
        let (_, _, _, active_off) = faced_switch(&empty, &puzzle);
        assert!(!active_off, "reticle is IDLE when facing empty space");
    }
}
