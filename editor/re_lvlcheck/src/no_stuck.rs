//! Check 15 — `no_stuck`: the player can never be trapped.
//!
//! Uses the SAME collision/movement rules the shipped game uses —
//! [`re_world::CameraPose::move_with_collision`] (radius-probed wall-slide) +
//! [`re_world::CameraPose::step_gravity`] (ramp-following, floor-snapping,
//! walk-through-aware) — rather than a hand-rolled grid model. This is
//! deliberately a DIFFERENT, stricter algorithm than check 13 (`props`), whose
//! `flood`/`stand` helpers are an intentionally simplified linter-only walk
//! model (see that module's docs); `no_stuck` instead ports the real-physics
//! flood fill already proven by the host game's own campaign walkability
//! tests — `re_lvlcheck` cannot depend on the host game (layering: it is the
//! top-level binary consumer), so the algorithm is reimplemented here against
//! `re_world` directly, mirroring those tests'
//! constants (`DT`/`H_STEP`/`GRAVITY_SUBSTEPS`/`ADVANCE_ITERS`/`SETTLE_ITERS`/
//! `EXPANSION_BUDGET`) and the same two real physics calls
//! (`move_with_collision` + `step_gravity`). One deliberate deviation:
//! `try_step` steers toward the target neighbour cell's centre every
//! iteration (re-aiming each `H_STEP`) rather than pushing one fixed
//! direction vector for the whole probe — a fixed-direction burst cannot
//! round a tight turn (see `axis_target`'s doc comment; caught on
//! `levels/20-stairs-straight.lvl` and improved further on
//! `levels/21-stairs-spiral.lvl`). This changes only the flood's own search
//! heuristic, never the physics primitives themselves.
//!
//! ## Radius-1 spiral landings — waypoint-chaining fallback
//!
//! `levels/21-stairs-spiral.lvl`'s final tread-to-landing entry used to read as
//! a false-negative trap: a dead-centre, constant-height probe cannot enter a
//! Full-solid neighbour that hands off from a ramp's own max height, because
//! `re_world`'s collision rules make "enter the neighbour's footprint" (feet
//! strictly BELOW its floor) and "catch its top on the next gravity tick" (feet
//! floor-indexing that SAME cell, i.e. `>=` its floor) mutually exclusive along
//! a single, unchanging approach line — provable directly from
//! `re_world::step_gravity`'s own floor-snap window, not merely inferred from a
//! failing search. [`try_step_waypoint_chase`] fixes this WITHOUT weakening the
//! physics: real players clear this seam only by drifting off the tread's
//! centre and/or timing a jump (both change the shape of the approach against
//! a corner that a dead-centre probe never touches), which is exactly what
//! that helper retries — a bounded set of lateral offsets × jump-timings, real
//! `move_with_collision`/`step_gravity`/`try_jump` throughout, engaged ONLY
//! when stepping off a `Shape::RampZ` tread (see its doc comment for why that
//! guard is load-bearing: it is what keeps a flat-floor low-wall pocket, e.g.
//! [`tests::a_sealed_pocket_fails`], from being "solved" by the same widened
//! search). Confirmed genuinely reachable (not merely search-limited) by an
//! exhaustive offset/jump-timing sweep during this fix, landing grounded on the
//! true upper-storey floor (not a partial fall) — the prior doc note here,
//! which leaned on `re_world`'s `spiral_ring_walkable_end_to_end` as proof,
//! was itself over-claiming: that test's `total_rise > 5.0` assertion is loose
//! enough to pass on a PARTIAL fall part-way back down the shaft, not only on
//! a true landing (measured directly while diagnosing this fix; `re_world` is
//! outside this crate's edit surface, so that test's assertion was not
//! tightened here — flagged for whoever next touches that suite).
//!
//! ## What it checks
//!
//! 1. **Spawn is not inside a collider** — [`re_world::point_blocked`] at the
//!    spawn pose's four radius-probe edges, the same predicate
//!    `move_with_collision` itself uses.
//! 2. **Flood the reachable set from spawn**, recording every successful
//!    one-step transition as a directed edge. Reverse those edges and flood
//!    backward from the door's stand cells: every forward-reachable cell must
//!    appear in that reverse flood — i.e. from ANYWHERE the player can reach,
//!    the door must still be reachable. A reachable cell absent from the
//!    reverse flood is a one-way trap (a pit with unclimbable walls, a pocket
//!    sealed by a prop) and FAILs, naming its cell.
//! 3. The same reverse-reachability is required of every pipe's switch stand
//!    cell (the puzzle's only interactable anchor besides the door): a switch
//!    that is reachable but stranded from ever reaching the door again FAILs.
//! 4. **Both power states.** Two maps are built: every power-reactive prop
//!    (`[prop] power=` AbsPart cells AND the legacy `[emit]`/`[part]` reward
//!    figures) forced OFF, and forced ON. Both are walked; a level that traps
//!    the player in only one state still FAILs. Since a power-reactive cell is
//!    now marked walk-through in BOTH states (set-dressing follow-up, the
//!    walk-through prop fix), a level should normally behave identically
//!    either way — this check is the regression guard proving that, and the
//!    net that would catch a FUTURE mechanism that adds genuinely solid
//!    power-reactive geometry (R4 whole-body swaps are explicitly deferred,
//!    not yet delivered).
//!
//! ## Coyote time + jump buffer
//!
//! `re_world::CameraPose::try_jump`/`step_gravity` gained a coyote-time grace
//! window (`re_world::COYOTE_TIME`) and a jump-input buffer
//! (`re_world::JUMP_BUFFER_TIME`) — a press still fires shortly after leaving
//! a ledge, or is remembered and fires on landing. This module's flood is
//! ALREADY "the same envelope" by construction, not by a parallel formula
//! that could drift from it: [`try_step`] and [`try_step_waypoint_chase`]
//! call the real `move_with_collision`/`step_gravity`/`try_jump` directly, so
//! any change to those functions is picked up automatically here with no
//! edit to this file.
//!
//! Concretely, the physics update leaves this flood's *reachable set*
//! unchanged for every shipped level:
//! * [`try_step`] (the flood's default one-cell probe) never calls
//!   `try_jump` at all — it only nudges toward the neighbour cell's centre
//!   and lets gravity settle. Coyote/buffer are inert on this path
//!   regardless of the physics underneath it.
//! * [`try_step_waypoint_chase`] does call `try_jump`, but only as a
//!   RampZ-departure fallback that already sweeps
//!   `WAYPOINT_JUMP_FRACTIONS` × `WAYPOINT_LATERAL_OFFSETS` (18
//!   offset/timing combinations) looking for ONE that lands. Coyote time
//!   widens WHEN a press is accepted, never the jump's trajectory (the
//!   impulse is set unconditionally to `JUMP_IMPULSE`, so the arc is
//!   bit-identical whether it fired on the ground or during the coyote
//!   window) — it cannot make a cell reachable that no offset/timing
//!   combination could already reach, because the search already tries
//!   every timing in that fixed, synchronous sweep. The jump buffer only
//!   affects a press that arrives before a landing that already happens
//!   deterministically in this sweep — it does not extend `ADVANCE_ITERS`.
//!
//! The static, non-physics reachability heuristics elsewhere in this crate
//! (`lib.rs`'s `check_reachability`/`switch_physically_reachable`,
//! `props_rule.rs`'s `flood`/`stand`/`switch_stands`) never call
//! `try_jump`/`step_gravity` at all — they use the fixed scalar
//! `max_jump_height()` (peak jump height, governed only by `JUMP_IMPULSE` and
//! `GRAVITY`, both unchanged by this update) as a vertical reach band. They
//! are mathematically unaffected: coyote time and the jump buffer change
//! WHEN a jump can be triggered, never what the triggered jump does.

use std::collections::{HashMap, HashSet, VecDeque};

use re_core::{Shape, Vec3};
use re_level::props;
use re_level::Level;
use re_world::{CameraPose, Map, PlayerMotion};

use crate::{Check, Location, Severity};

// ── the real-physics flood fill (ported from the host game's own campaign
//    walkability tests — read, not re-guessed; kept in lockstep by reading
//    those tests again on any future change to `CameraPose`) ─────────────────

const DT: f32 = 1.0 / 60.0;
const H_STEP: f32 = 0.05;
const GRAVITY_SUBSTEPS: usize = 3;
const ADVANCE_ITERS: usize = 30;
const SETTLE_ITERS: usize = 90;
const EXPANSION_BUDGET: usize = 40_000;

const DIRS_8: [(f32, f32); 8] = [
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
    (
        std::f32::consts::FRAC_1_SQRT_2,
        std::f32::consts::FRAC_1_SQRT_2,
    ),
    (
        std::f32::consts::FRAC_1_SQRT_2,
        -std::f32::consts::FRAC_1_SQRT_2,
    ),
    (
        -std::f32::consts::FRAC_1_SQRT_2,
        std::f32::consts::FRAC_1_SQRT_2,
    ),
    (
        -std::f32::consts::FRAC_1_SQRT_2,
        -std::f32::consts::FRAC_1_SQRT_2,
    ),
];

type CellKey = (i32, i32, i32);

const fn cell_key(p: Vec3) -> CellKey {
    (p.x.floor() as i32, p.y.floor() as i32, p.z.floor() as i32)
}

fn settle(cam: &mut CameraPose, map: &Map) -> PlayerMotion {
    let mut motion = PlayerMotion::default();
    for _ in 0..SETTLE_ITERS {
        cam.step_gravity(&mut motion, DT, map);
        if motion.grounded {
            break;
        }
    }
    motion
}

/// The neighbour cell centre one step away from `from_key` along `d`'s sign
/// (`d`'s magnitude is irrelevant, only its sign/zero-ness — `dir` is always
/// one of the 8 compass unit vectors). `f32::signum` is wrong for exactly
/// `0.0` (it returns `1.0`), so the zero case is handled explicitly to keep
/// a N/S/E/W probe from silently acquiring a diagonal target.
fn axis_target(from_key: i32, d: f32) -> f32 {
    let step = if d.abs() < 1e-6 { 0.0 } else { d.signum() };
    from_key as f32 + step + 0.5
}

/// Attempt to physically walk from `from` one cell in direction `dir`. `Some`
/// iff the player made real progress (crossed into a different integer cell)
/// and ended up GROUNDED there.
///
/// Steers toward the target NEIGHBOUR CELL'S CENTRE each iteration (re-aiming
/// every `H_STEP`, exactly as a real player continuously corrects heading)
/// rather than pushing a single fixed direction vector for the whole probe.
/// A fixed-direction burst cannot round a tight turn — the exact case a
/// radius-1 spiral staircase tread-to-tread seam requires (verified against
/// `re_world`'s own `spiral_ring_walkable_end_to_end`, which proves the
/// underlying collision genuinely allows the turn; a straight-line burst
/// alone cannot reach it). This changes only the flood's OWN search
/// heuristic — `move_with_collision`/`step_gravity` remain the exact same
/// real physics calls.
fn try_step(map: &Map, from: Vec3, dir: (f32, f32)) -> Option<Vec3> {
    let key = cell_key(from);
    let target_x = axis_target(key.0, dir.0);
    let target_y = axis_target(key.1, dir.1);
    let mut cam = CameraPose {
        pos: from,
        yaw: 0.0,
        pitch: 0.0,
    };
    let mut motion = PlayerMotion::default();
    for _ in 0..ADVANCE_ITERS {
        let dx = (target_x - cam.pos.x).clamp(-H_STEP, H_STEP);
        let dy = (target_y - cam.pos.y).clamp(-H_STEP, H_STEP);
        cam.move_with_collision(dx, dy, map, re_world::PLAYER_RADIUS);
        for _ in 0..GRAVITY_SUBSTEPS {
            cam.step_gravity(&mut motion, DT, map);
        }
    }
    for _ in 0..SETTLE_ITERS {
        if motion.grounded {
            break;
        }
        cam.step_gravity(&mut motion, DT, map);
    }
    if !motion.grounded {
        return None;
    }
    let moved = (cam.pos.x - from.x).abs() > 0.5 || (cam.pos.y - from.y).abs() > 0.5;
    if !moved {
        return None;
    }
    Some(cam.pos)
}

// ── waypoint-chaining fallback for a failed axis-aligned step ────────────────
//
// `try_step` steers from the CURRENT position toward the target neighbour
// cell's centre — a single, dead-centre approach line. That is not enough for
// a genuine 1-cell vertical step onto a Full-solid neighbour (a ramp handing
// off to a flush landing/floor, e.g. `levels/21-stairs-spiral.lvl`'s last
// tread → the upper storey): entering the neighbour's footprint REQUIRES feet
// strictly below its floor (`re_world`'s "step-up" carve-out in
// `move_with_collision`), while catching its top on the very next gravity
// tick requires feet to already floor-index the SAME cell (`>=` its floor) —
// two conditions a dead-centre, constant-height approach can satisfy only in
// the unreachable limit (exactly on the seam). A real player instead drifts
// slightly off-centre and/or times a jump, both of which change the *shape*
// of the approach (a corner is not uniform), and empirically DO clear this
// exact seam (verified by an exhaustive offset/jump-timing search during this
// fix). This helper reproduces that — real `move_with_collision` /
// `step_gravity` / `try_jump`, just with a richer, still-BOUNDED set of
// waypoints/headings than a single dead-centre probe.
//
// Deliberately NOT permissive:
//   * only real `re_world` physics calls are used — nothing here overrides a
//     collision result or teleports the flood past solid geometry;
//   * only accepted if the result is GROUNDED (a fall is never accepted as a
//     "successful" step here — the caller's plain `try_step` already records
//     genuine falls as their own real edges);
//   * only accepted if the landing cell is the ONE cell directly adjacent to
//     the departure cell along `dir` (never a distant fall mistaken for the
//     intended step) and at the same height or higher (a climb, not a drop);
//   * a lateral offset that isn't even standable on the departure cell, or
//     whose very first nudge in `dir` is hard-blocked (still `false`/`false`
//     from `move_with_collision`), is skipped WITHOUT running the expensive
//     multi-jump-timing simulation — a genuine sealed wall blocks every
//     offset identically, so this keeps the fallback cheap on the (common)
//     case where a direction is a plain wall, not a seam;
//   * ONLY engaged when the departure cell is a `Shape::RampZ` tread — i.e.
//     the player is stepping off a climb, the exact class of seam this
//     targets. A flat-floor cell never triggers it. This is the guard that
//     keeps the fallback from ALSO "solving" a low, flat-topped wall by
//     hopping over it: `PLAYER_HEAD_ROOM`/jump reach can genuinely clear a
//     one-cell-tall obstacle with open air above it, and a real player could
//     too — but a level's `[prop]`/pocket geometry that relies on such a wall
//     staying a barrier (proven by the sealed-pocket and prop-seal regression
//     tests below) must not have that barrier silently reinterpreted by a
//     search-heuristic change. Restricting the fallback to a ramp departure
//     keeps it a fix for the staircase-seam class of false negative, not a
//     general "can I jump over this" widening of what counts as reachable.
const WAYPOINT_LATERAL_OFFSETS: [f32; 6] = [-0.45, -0.3, -0.15, 0.15, 0.3, 0.45];
const WAYPOINT_JUMP_FRACTIONS: [f32; 3] = [0.2, 0.4, 0.6];

fn try_step_waypoint_chase(map: &Map, from: Vec3, dir: (f32, f32)) -> Option<Vec3> {
    // Diagonal directions: a lateral offset perpendicular to a single travel
    // axis isn't well-defined, and `try_step` already covers them adequately
    // (no known failure class motivates the extra cost here).
    if dir.0 != 0.0 && dir.1 != 0.0 {
        return None;
    }
    let from_key = cell_key(from);
    // Only engage from a RampZ departure — see the module-level comment above
    // this function for why. The "stand" cell is the one the player's FEET
    // (not eye) are floor-indexed into, mirroring `step_gravity`'s own
    // `feet_z.floor()` convention.
    let feet_z = from.z - re_world::PLAYER_EYE_HEIGHT;
    let stand_z = feet_z.floor() as i32;
    let standing_on_ramp = map.in_bounds(from_key.0, from_key.1, stand_z)
        && matches!(
            map.shape_at(from_key.0, from_key.1, stand_z),
            Shape::RampZ(_)
        );
    if !standing_on_ramp {
        return None;
    }
    // Perpendicular unit vector to `dir` (swap + negate one component) — the
    // axis a lateral offset is applied along.
    let (px, py) = (-dir.1, dir.0);
    let expected_key = (
        from_key.0 + dir.0 as i32,
        from_key.1 + dir.1 as i32,
        from_key.2,
    );

    for &off in &WAYPOINT_LATERAL_OFFSETS {
        let start = Vec3::new(from.x + px * off, from.y + py * off, from.z);
        let mut base_cam = CameraPose {
            pos: start,
            yaw: 0.0,
            pitch: 0.0,
        };
        let base_motion = settle(&mut base_cam, map);
        if !base_motion.grounded || cell_key(base_cam.pos) != from_key {
            continue; // this offset isn't even standable on the departure cell
        }
        // Cheap pre-check: does a single nudge in `dir` make ANY progress from
        // here? If it is hard-blocked outright, no jump timing opens solid rock.
        let mut probe = base_cam;
        let (mx, my) =
            probe.move_with_collision(dir.0 * H_STEP, dir.1 * H_STEP, map, re_world::PLAYER_RADIUS);
        if !mx && !my {
            continue;
        }
        for &jf in &WAYPOINT_JUMP_FRACTIONS {
            let mut cam = base_cam;
            let mut motion = base_motion;
            let target_x = axis_target(from_key.0, dir.0);
            let target_y = axis_target(from_key.1, dir.1);
            let jump_at_iter = (ADVANCE_ITERS as f32 * jf) as usize;
            for i in 0..ADVANCE_ITERS {
                if i == jump_at_iter {
                    CameraPose::try_jump(&mut motion);
                }
                let dx = (target_x - cam.pos.x).clamp(-H_STEP, H_STEP);
                let dy = (target_y - cam.pos.y).clamp(-H_STEP, H_STEP);
                cam.move_with_collision(dx, dy, map, re_world::PLAYER_RADIUS);
                for _ in 0..GRAVITY_SUBSTEPS {
                    cam.step_gravity(&mut motion, DT, map);
                }
            }
            for _ in 0..SETTLE_ITERS {
                if motion.grounded {
                    break;
                }
                cam.step_gravity(&mut motion, DT, map);
            }
            if !motion.grounded {
                continue;
            }
            let landed_key = cell_key(cam.pos);
            if landed_key.0 == expected_key.0
                && landed_key.1 == expected_key.1
                && landed_key.2 >= from_key.2
            {
                return Some(cam.pos);
            }
        }
    }
    None
}

/// One flood-fill result: every reachable cell (with a representative grounded
/// world position) plus every directed edge the flood actually walked.
struct Flood {
    reachable: HashMap<CellKey, Vec3>,
    /// `edges[u]` = every `v` the flood successfully stepped to FROM `u` (may
    /// repeat a `v` seen from another `u`; duplicates are harmless for reverse
    /// reachability).
    edges: HashMap<CellKey, Vec<CellKey>>,
}

/// Real-physics flood fill of every cell reachable from `spawn` (already a
/// settled, grounded position), recording the directed edge graph it walked.
fn flood_fill(map: &Map, spawn: Vec3) -> Flood {
    let mut reachable: HashMap<CellKey, Vec3> = HashMap::new();
    let mut edges: HashMap<CellKey, Vec<CellKey>> = HashMap::new();
    // The queue carries the KEY a position was filed under, not just the
    // position — see below for why the two can diverge.
    let mut queue: VecDeque<(CellKey, Vec3)> = VecDeque::new();
    let start_key = cell_key(spawn);
    reachable.insert(start_key, spawn);
    queue.push_back((start_key, spawn));

    let mut expansions = 0usize;
    while let Some((u_key, pos)) = queue.pop_front() {
        if expansions >= EXPANSION_BUDGET {
            break;
        }
        expansions += 1;
        // NOTE: `u_key` is the key `pos` was FILED under in `reachable`, not
        // `cell_key(pos)` recomputed here. A settled/re-anchored canonical
        // position can land with `pos.z` exactly ON an integer boundary (a
        // ramp tread's top lip is precisely a seam, by construction) — its
        // `cell_key` would then float-floor into the NEXT cell up, even
        // though it was discovered and filed as the seam's lower cell.
        // Recomputing the key here silently records every outgoing edge
        // under that phantom neighbour key instead of the real one, leaving
        // the tread's actual `reachable` entry with zero recorded edges —
        // which `no_stuck` then (wrongly) reports as a one-way trap. Using
        // the filed key keeps `edges` and `reachable` in the same keyspace
        // regardless of where a settle lands on a seam.
        for &dir in &DIRS_8 {
            // The plain dead-centre probe first; if it made no grounded
            // progress at all, retry with the bounded waypoint-chaining
            // fallback (off-centre approach + a well-timed jump) before
            // giving up on this direction — see `try_step_waypoint_chase`'s
            // doc comment for exactly what this does and does not permit.
            let landed = match try_step(map, pos, dir) {
                Some(landed) => landed,
                None => {
                    let Some(landed) = try_step_waypoint_chase(map, pos, dir) else {
                        continue;
                    };
                    landed
                }
            };
            let v_key = cell_key(landed);
            edges.entry(u_key).or_default().push(v_key);
            if reachable.contains_key(&v_key) {
                continue;
            }
            // Re-anchor to the cell's horizontal centre for a clean next probe.
            let anchor = Vec3::new(v_key.0 as f32 + 0.5, v_key.1 as f32 + 0.5, landed.z);
            let mut acam = CameraPose {
                pos: anchor,
                yaw: 0.0,
                pitch: 0.0,
            };
            let am = settle(&mut acam, map);
            let canonical = if am.grounded { acam.pos } else { landed };
            reachable.insert(v_key, canonical);
            queue.push_back((v_key, canonical));
        }
    }
    Flood { reachable, edges }
}

/// Reverse-BFS: every cell that can reach some cell in `targets`, via the
/// reverse of `edges`.
fn reverse_reachable(
    edges: &HashMap<CellKey, Vec<CellKey>>,
    targets: &[CellKey],
) -> HashSet<CellKey> {
    let mut rev: HashMap<CellKey, Vec<CellKey>> = HashMap::new();
    for (&u, vs) in edges {
        for &v in vs {
            rev.entry(v).or_default().push(u);
        }
    }
    let mut seen: HashSet<CellKey> = targets.iter().copied().collect();
    let mut queue: VecDeque<CellKey> = targets.iter().copied().collect();
    while let Some(v) = queue.pop_front() {
        let Some(preds) = rev.get(&v) else { continue };
        for &u in preds {
            if seen.insert(u) {
                queue.push_back(u);
            }
        }
    }
    seen
}

fn max_jump_height() -> f32 {
    (re_world::JUMP_IMPULSE * re_world::JUMP_IMPULSE) / (2.0 * re_world::GRAVITY)
}

/// Stand cells (a subset of `reachable`) from which `switch` is interactable —
/// same reach band `check_reachability`/`campaign_emit_figures.rs` use.
fn switch_stands_reachable(
    reachable: &HashMap<CellKey, Vec3>,
    switch: CellKey,
    jump_reach: f32,
) -> Vec<CellKey> {
    reachable
        .keys()
        .filter(|&&(x, y, z)| {
            (x - switch.0).abs() <= 1
                && (y - switch.1).abs() <= 1
                && (switch.2 as f32) >= z as f32 - 0.5
                && (switch.2 as f32) <= z as f32 + jump_reach + 1.5
        })
        .copied()
        .collect()
}

/// Door stand cells (mirrors `re_world::at_door`'s Chebyshev-1 ball, but only
/// over cells the flood actually proved standable).
fn door_stands_reachable(reachable: &HashMap<CellKey, Vec3>, door: CellKey) -> Vec<CellKey> {
    reachable
        .keys()
        .filter(|&&(x, y, z)| {
            (x - door.0).abs() <= 1 && (y - door.1).abs() <= 1 && (z - door.2).abs() <= 1
        })
        .copied()
        .collect()
}

// ── legacy `[emit]`/`[part]`/`[pedestal]` reward-figure cells ─────────────────
//
// This is a SEPARATE, older mechanism from `re_level::props`'s `PowerGate`
// (AbsPart) — it lives entirely in `re_game::emit_parts`, which `re_lvlcheck`
// cannot depend on (layering). The anchor-rule formula below is ported
// verbatim from `re_game::emit_parts::part_cells`'s documented contract; if
// that formula ever changes, this must change with it.

fn emit_part_cells(level: &Level, part_idx: usize) -> Vec<(i32, i32, i32)> {
    let Some(part) = level.parts.get(part_idx) else {
        return Vec::new();
    };
    let (w, h, d) = (part.w.max(0), part.h.max(0), part.d.max(0));
    if w == 0 || h == 0 || d == 0 {
        return Vec::new();
    }
    let ped = &level.pedestal;
    let base_x = ped.cx - ped.w / 2 + part.off_x;
    let base_y = ped.cy - ped.h / 2 + part.off_y;
    let base_z = ped.cz + part.off_z;
    let mut out = Vec::new();
    for pz in 0..d {
        for py in 0..h {
            for px in 0..w {
                let idx = px as usize + (w as usize) * (py as usize + (h as usize) * pz as usize);
                if part.mask.get(idx).copied().unwrap_or(0) == 0 {
                    continue;
                }
                out.push((base_x + px, base_y + py, base_z + pz));
            }
        }
    }
    out
}

/// Build the two power-state maps: every AbsPart/`[emit]` figure forced OFF,
/// and forced ON. Both start from the SAME base geometry (real furniture,
/// holograms, decals, lines — none of which are power-gated collision today),
/// so a difference between them is purely the power-reactive cells.
fn power_state_maps(level: &Level) -> [(&'static str, Map); 2] {
    let base = level.build_map_base();
    let (exp, _errs) = props::expand_level(level, &base);

    let mut off_exp = exp.clone();
    off_exp.parts.clear(); // no AbsPart cell is stamped -> a clean "all off"
    let mut off_map = base;
    props::stamp_expansion(level, &off_exp, &mut off_map);
    // Legacy `[emit]` figures are never stamped by `build_map`/`stamp_expansion`
    // at all (they are runtime-only, `re_game::emit_parts`), so `off_map`
    // already carries none — nothing further to do for "off".

    let mut on_map = off_map.clone();
    for p in &exp.parts {
        let (x, y, z) = p.pos;
        if x >= 0
            && y >= 0
            && z >= 0
            && on_map.in_bounds(x, y, z)
            && on_map.cell_at(x, y, z) == re_world::CellKind::Empty
        {
            let id = exp.palette_id(level, &p.mat);
            on_map.set_cell(
                x as usize,
                y as usize,
                z as usize,
                re_world::CellKind::WallSolid,
                id,
            );
            on_map.mark_walkthrough(x as usize, y as usize, z as usize, true);
        }
    }
    for emit in &level.emits {
        for (x, y, z) in emit_part_cells(level, emit.part) {
            if x >= 0
                && y >= 0
                && z >= 0
                && on_map.in_bounds(x, y, z)
                && on_map.cell_at(x, y, z) == re_world::CellKind::Empty
            {
                on_map.set_cell(
                    x as usize,
                    y as usize,
                    z as usize,
                    re_world::CellKind::WallSolid,
                    1,
                );
                on_map.mark_walkthrough(x as usize, y as usize, z as usize, true);
            }
        }
    }

    [("power-off", off_map), ("power-on", on_map)]
}

/// One finding: which power state, the offending cell, and why.
struct Trap {
    state: &'static str,
    cell: CellKey,
    reason: String,
}

/// Walk `map` from `level`'s spawn and report every trap. Empty ⇒ this power
/// state is clean.
#[allow(clippy::too_many_lines)]
fn walk_one_state(level: &Level, state: &'static str, map: &Map) -> Vec<Trap> {
    let mut traps = Vec::new();
    let spawn_pose = level.spawn_camera();
    let spawn = spawn_pose.pos;

    // (a) Spawn is not inside a collider: probe the four leading edges
    // `move_with_collision` itself would probe for a zero-length move.
    let r = re_world::PLAYER_RADIUS;
    let spawn_blocked = [(r, 0.0), (-r, 0.0), (0.0, r), (0.0, -r)]
        .iter()
        .any(|&(ox, oy)| {
            re_world::point_blocked(map, Vec3::new(spawn.x + ox, spawn.y + oy, spawn.z))
        });
    if spawn_blocked {
        traps.push(Trap {
            state,
            cell: cell_key(spawn),
            reason: "spawn overlaps a collider".to_string(),
        });
        return traps; // nothing downstream is meaningful from here
    }

    let mut cam = spawn_pose;
    let settled = settle(&mut cam, map);
    if !settled.grounded {
        traps.push(Trap {
            state,
            cell: cell_key(spawn),
            reason: "spawn never settles onto standable ground".to_string(),
        });
        return traps;
    }

    let flood = flood_fill(map, cam.pos);
    let door = (level.door.x, level.door.y, level.door.z);
    let door_targets = door_stands_reachable(&flood.reachable, door);
    if door_targets.is_empty() {
        traps.push(Trap {
            state,
            cell: door,
            reason: "the door is not reachable from spawn at all".to_string(),
        });
        return traps;
    }
    let can_finish = reverse_reachable(&flood.edges, &door_targets);

    let mut stuck: Vec<CellKey> = flood
        .reachable
        .keys()
        .filter(|k| !can_finish.contains(k))
        .copied()
        .collect();
    stuck.sort_unstable();
    for cell in stuck {
        traps.push(Trap {
            state,
            cell,
            reason: "reachable from spawn but cannot reach the door again — a one-way trap"
                .to_string(),
        });
    }

    let jump_reach = max_jump_height();
    for (pi, pipe) in level.pipes.iter().enumerate() {
        let Some(cell) = pipe.switch_cell.and_then(|sc| pipe.path.get(sc).copied()) else {
            continue;
        };
        let stands = switch_stands_reachable(&flood.reachable, cell, jump_reach);
        if stands.is_empty() {
            continue; // not this check's job — `reachable`/`props` already cover it
        }
        if !stands.iter().any(|s| can_finish.contains(s)) {
            traps.push(Trap {
                state,
                cell,
                reason: format!(
                    "switch for pipe {pi} ({}->{}) is reachable but stranded from the door",
                    pipe.from, pipe.to
                ),
            });
        }
    }

    traps
}

/// Check 15 — see the module docs.
pub(crate) fn check_no_stuck(level: &Level) -> Check {
    let mut all_traps: Vec<Trap> = Vec::new();
    for (state, map) in power_state_maps(level) {
        all_traps.extend(walk_one_state(level, state, &map));
    }

    if all_traps.is_empty() {
        return Check::new(
            "no_stuck",
            Severity::Pass,
            "no trap found in either power state",
        );
    }
    let detail = format!(
        "{} trap(s): {}",
        all_traps.len(),
        all_traps
            .iter()
            .map(|t| format!("[{}] {:?}: {}", t.state, t.cell, t.reason))
            .collect::<Vec<_>>()
            .join(" | ")
    );
    Check::new("no_stuck", Severity::Fail, detail)
        .at_all(all_traps.iter().map(|t| Location::cell(t.cell)).collect())
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use re_world::CellKind;

    const HEADER: &str = "\
[meta]
name = no-stuck-test
[room]
size = 12 8 4
ambient = solid
[node in]
pos = 2,4,1
glyph = I
is_input = yes
[node out]
pos = 9,4,1
glyph = O
is_output = yes
threshold = 1
[pipe in->out]
state = OFF
path = (2,4,1) (9,4,1)
switch_at = (4,4,1)
[inputs]
in = 1
[pedestal]
center = 5 4 1
volume = 2 1 1
[door]
pos = 10 4 1
target_parts = out
[player]
spawn = 1 4 1
facing = +x
";

    fn load(src: &str) -> Level {
        re_level::load_string(src, "<no_stuck_test>").expect("test level parses")
    }

    #[test]
    fn a_plain_open_room_passes() {
        let level = load(HEADER);
        let check = check_no_stuck(&level);
        assert_eq!(check.severity, Severity::Pass, "{}", check.detail);
    }

    #[test]
    fn a_sealed_pocket_fails() {
        // Wall a 1x1 pocket on all four horizontal sides at (6,6,1) and put the
        // spawn INSIDE it: the flood can take no horizontal step out (the model
        // moves only 4/8-directionally + step/fall — there is no "jump straight
        // up"), so the door and every switch are simply unreachable — a
        // textbook sealed pocket.
        let mut level = load(HEADER);
        let mut map = level.build_map();
        for (x, y) in [(5, 6), (7, 6), (6, 5), (6, 7)] {
            map.set_cell(x, y, 1, CellKind::WallSolid, 0);
        }
        level.player.x = 6;
        level.player.y = 6;
        level.player.z = 1;
        let traps = walk_one_state(&level, "test", &map);
        assert!(
            !traps.is_empty(),
            "a sealed 1x1 pocket must be reported as a trap"
        );
    }

    #[test]
    fn a_spawn_buried_in_a_wall_fails() {
        let level = load(HEADER);
        let mut map = level.build_map();
        // Bury the spawn cell in solid rock.
        map.set_cell(1, 4, 1, CellKind::WallSolid, 0);
        let traps = walk_one_state(&level, "test", &map);
        assert!(
            traps.iter().any(|t| t.reason.contains("collider")),
            "traps: {:?}",
            traps.iter().map(|t| &t.reason).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_one_way_drop_off_a_ledge_is_reported() {
        // A raised deck (spawn + door) with an open edge dropping 3 cells to the
        // real floor — far past `max_jump_height` (~1 cell), so the pit floor is
        // reachable (the flood walks off the ledge) but can never climb back to
        // the deck: a textbook one-way pit.
        let mut level = load(HEADER);
        level.pipes.clear(); // this test is about spawn/door geometry only

        let spec = re_world::GeometrySpec {
            size: (8, 8, 6), // interior air z=1..=4, floor z=0, ceiling z=5
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[],
        };
        let mut map = re_world::stamp_geometry(&spec);
        // Deck top surface at world z=4 (cell z=3), spanning x=1..=4, y=1..=6.
        // Beyond it (x=5,6) the interior is open air all the way to the real
        // floor at z=0 — nothing to climb back up on.
        for x in 1..=4 {
            for y in 1..=6 {
                map.set_cell(x, y, 3, CellKind::WallSolid, 0);
            }
        }

        level.player.x = 2;
        level.player.y = 3;
        level.player.z = 4;
        level.player.facing_yaw_col = 0;
        level.door = re_level::Door {
            x: 3,
            y: 3,
            z: 4,
            target_parts: Vec::new(),
            target_off: Vec::new(),
            target_model: String::new(),
        };

        let traps = walk_one_state(&level, "test", &map);
        assert!(
            traps.iter().any(|t| t.reason.contains("one-way trap")),
            "the pit floor must be flagged as a one-way trap: {:?}",
            traps
                .iter()
                .map(|t| (&t.cell, &t.reason))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_prop_that_would_seal_the_way_when_powered_is_walk_through_not_a_trap() {
        // The R1 mechanism (`[prop] power=`) stamps `WallSolid` across the ONLY
        // passage between spawn and the door when its gate is lit — exactly the
        // shape the walk-through fix targets. `check_no_stuck` walks BOTH power
        // states; the fix (marking the cell walk-through wherever it is
        // stamped) means the "power-on" state must be exactly as clean as
        // "power-off". This is the regression guard: if a future change ever
        // stamps such a cell WITHOUT marking it walk-through, this test fails.
        // A plain open room (no node/pipe object cells to route around) with a
        // wall at x=5 spanning y=1..=6 and a single 1-cell gap at y=4 — the
        // ONLY passage between spawn (x=1) and the door (x=10).
        let mut level = load(HEADER);
        level.pipes.clear();
        level.player.x = 1;
        level.player.y = 4;
        level.player.z = 1;
        level.player.facing_yaw_col = 0;
        level.door = re_level::Door {
            x: 10,
            y: 4,
            z: 1,
            target_parts: Vec::new(),
            target_off: Vec::new(),
            target_model: String::new(),
        };
        let spec = re_world::GeometrySpec {
            size: (12, 8, 4),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[],
        };
        let mut base = re_world::stamp_geometry(&spec);
        // The wall itself, leaving y=4 open as the one gap.
        for y in 1..=6 {
            if y == 4 {
                continue;
            }
            base.set_cell(5, y, 1, CellKind::WallSolid, 0);
        }
        // Stamp the gap solid, exactly as `prop_power::apply_map` would for a
        // lit R1 slot, INCLUDING the walk-through mark.
        let mut on_map = base.clone();
        on_map.set_cell(5, 4, 1, CellKind::WallSolid, 1);
        on_map.mark_walkthrough(5, 4, 1, true);

        let off_traps = walk_one_state(&level, "power-off", &base);
        let on_traps = walk_one_state(&level, "power-on", &on_map);
        assert!(
            off_traps.is_empty(),
            "power-off must be clean: {:?}",
            off_traps.iter().map(|t| &t.reason).collect::<Vec<_>>()
        );
        assert!(
            on_traps.is_empty(),
            "power-on must be equally clean once the cell is marked walk-through: {:?}",
            on_traps.iter().map(|t| &t.reason).collect::<Vec<_>>()
        );

        // Sanity: WITHOUT the walk-through mark, the same stamp genuinely traps
        // the far side — proving the assertions above are not vacuous.
        let mut sealed_map = base.clone();
        sealed_map.set_cell(5, 4, 1, CellKind::WallSolid, 1);
        let sealed_traps = walk_one_state(&level, "power-on-unmarked", &sealed_map);
        assert!(
            !sealed_traps.is_empty(),
            "an UNMARKED solid cell across the only corridor must be caught"
        );
    }

    #[test]
    fn emit_part_cells_matches_the_documented_anchor_rule() {
        // HEADER's own `[pedestal]` is `center = 5 4 1, volume = 2 1 1`.
        let mut src = String::from(HEADER);
        src.push_str(
            "[part slab]\n\
             size = 2 1 1\n\
             mask =\n\
             ##\n\
             color = #FF4040\n\
             offset = 0 0 0\n\
             [emit out=1]\n\
             parts = slab\n",
        );
        let level = load(&src);
        let cells = emit_part_cells(&level, 0);
        assert_eq!(cells, vec![(4, 4, 1), (5, 4, 1)]);
    }

    /// Regression for the seam-key bug: a settled canonical position landing
    /// EXACTLY on an integer z boundary (a ramp tread's top lip, by
    /// construction) must still record its outgoing edges under the key it
    /// was filed under in `reachable`, not a recomputed `cell_key` that
    /// floors into the next cell up. Ported from the real
    /// `levels/20-stairs-straight.lvl` failure this caught: a straight,
    /// perfectly climbable 4-tread staircase was reported as ~150 one-way
    /// traps until this was fixed.
    #[test]
    fn a_straight_staircase_is_not_a_one_way_trap() {
        use re_core::Shape;
        use re_world::{GeometrySpec, RampDir};

        const H: usize = 4;
        let sz = H + 3;
        let spec = GeometrySpec {
            size: (H + 3, 5, sz),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[],
        };
        let mut map = re_world::stamp_geometry(&spec);
        // Ramp cell n at (n, 2, n) for n=1..=H, rising +X — the exact
        // layout `re_world`'s own `staircase_map(h)` test helper uses.
        for n in 1..=H {
            map.set(n, 2, n, CellKind::WallSolid);
            map.set_shape(n, 2, n, Shape::RampZ(RampDir::PosX));
        }
        // A landing floor immediately past the top tread, at the same
        // z-index as the last tread (its top surface continues at that
        // height) — otherwise there is nothing to stand on past the ramp
        // and the "door" would float in open air over nothing.
        map.set(H + 1, 2, H, CellKind::Floor);

        let mut level = load(HEADER);
        level.pipes.clear();
        level.player.x = 1;
        level.player.y = 2;
        level.player.z = 1;
        level.player.facing_yaw_col = 0;
        level.door = re_level::Door {
            x: (H + 1) as i32,
            y: 2,
            z: (H + 1) as i32,
            target_parts: Vec::new(),
            target_off: Vec::new(),
            target_model: String::new(),
        };

        let traps = walk_one_state(&level, "test", &map);
        assert!(
            traps.is_empty(),
            "a straight climbable staircase must not report one-way traps: {:?}",
            traps
                .iter()
                .map(|t| (t.cell, t.reason.clone()))
                .collect::<Vec<_>>()
        );
    }

    /// End-to-end regression, tied to the real shipped content rather than a
    /// synthetic fixture: `levels/21-stairs-spiral.lvl`'s last tread → landing
    /// transition needs the waypoint-chaining fallback in
    /// `try_step_waypoint_chase` (an off-centre approach + a well-timed jump —
    /// see that function's doc comment for why a dead-centre steer alone can
    /// never satisfy both the ramp's exit condition and the landing's entry
    /// condition). Before that fallback existed, this level's `no_stuck` check
    /// FAILed with a one-way-trap report at the top of the spiral; this test
    /// pins it PASS so a future change to the fallback's guard (e.g. narrowing
    /// it back to "flat floors only") cannot silently regress this level
    /// without a test failure naming it.
    #[test]
    fn the_real_spiral_staircase_level_passes_no_stuck() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../levels/21-stairs-spiral.lvl");
        let raw = std::fs::read_to_string(&path).expect("read levels/21-stairs-spiral.lvl");
        let level = re_level::load_string(&raw, "21-stairs-spiral")
            .expect("the shipped spiral level parses");
        let check = check_no_stuck(&level);
        assert_eq!(
            check.severity,
            Severity::Pass,
            "the spiral's last tread -> landing transition must be provably walkable: {}",
            check.detail
        );
    }
}
