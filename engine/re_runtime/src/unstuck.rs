//! Camera/collision unstuck-recovery helper. No seams: pure over
//! [`re_world::Map`] / [`re_world::CameraPose`].
//!
//! Runtime safety net: if the player ends up embedded in a collider — after a
//! fresh level load or after a prop-power flip — push them to the nearest
//! free standable cell.
//!
//! Marking every power-reactive cell walk-through (`re_world::Map::mark_walkthrough`,
//! wired in `emit_parts.rs`/`prop_power.rs`/`re_level::props::stamp_expansion`) already
//! keeps a power flip from ever solidifying geometry under the player TODAY, and
//! `re_lvlcheck`'s `no_stuck` check gates every shipped level's spawn against being
//! buried at load. This module is the belt to that pair of braces: a defensive net for
//! a FUTURE mechanism that adds genuinely solid power-reactive geometry (whole-body
//! swaps are explicitly deferred, not yet delivered) or any other not-yet-imagined
//! way the player could end up inside a collider.
//!
//! Deterministic: [`re_world::nearest_free_standable_cell`] is a plain, hash-free
//! outward search, so the same map + the same stuck position always recovers to the
//! same cell — never platform- or iteration-order-dependent.

use re_core::Vec3;
use re_world::{CameraPose, Map};

/// How far (in cells) the recovery search looks before giving up and merely
/// logging. Six rings covers any plausible "a wall grew under the player" case
/// in a normal room; further than that the map itself is probably the real
/// defect — the same job `re_lvlcheck::no_stuck`'s spawn check does at load
/// time, this is only the runtime backstop for it.
const UNSTUCK_MAX_RADIUS: i32 = 6;

/// Is `cam` embedded in a collider? Checks the same four eye-height,
/// radius-offset probe points `CameraPose::move_with_collision` itself probes
/// for a zero-length move (and `re_lvlcheck::no_stuck`'s spawn check reuses) —
/// never a re-approximation of the rule.
fn camera_blocked(map: &Map, cam: &CameraPose) -> bool {
    let r = re_world::PLAYER_RADIUS;
    [(r, 0.0), (-r, 0.0), (0.0, r), (0.0, -r)]
        .iter()
        .any(|&(ox, oy)| {
            re_world::point_blocked(map, Vec3::new(cam.pos.x + ox, cam.pos.y + oy, cam.pos.z))
        })
}

/// If `cam` overlaps a collider in `map`, push it to the nearest free
/// standable cell (deterministic). Does nothing if `cam` is already safe.
/// Returns `true` iff `cam` was moved.
pub fn ensure_camera_safe(map: &Map, cam: &mut CameraPose) -> bool {
    if !camera_blocked(map, cam) {
        return false;
    }
    if let Some(safe) = re_world::nearest_free_standable_cell(map, cam.pos, UNSTUCK_MAX_RADIUS) {
        println!(
            "unstuck: player was embedded in a collider at ({:.2},{:.2},{:.2}) — pushed to \
             the nearest free standable cell at ({:.2},{:.2},{:.2})",
            cam.pos.x, cam.pos.y, cam.pos.z, safe.x, safe.y, safe.z,
        );
        cam.pos = safe;
        return true;
    }
    println!(
        "unstuck: player is embedded in a collider at ({:.2},{:.2},{:.2}) and no free cell was \
         found within {UNSTUCK_MAX_RADIUS} cells — leaving unresolved (this indicates a \
         level-data defect `re_lvlcheck::no_stuck` should have caught)",
        cam.pos.x, cam.pos.y, cam.pos.z,
    );
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use re_world::{stamp_geometry, CellKind, GeometrySpec};

    fn open_room() -> Map {
        let spec = GeometrySpec {
            size: (5, 5, 4),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[],
        };
        stamp_geometry(&spec)
    }

    #[test]
    fn a_safe_camera_is_left_alone() {
        let map = open_room();
        let mut cam = CameraPose {
            pos: Vec3::new(2.5, 2.5, 1.5),
            yaw: 0.0,
            pitch: 0.0,
        };
        let before = cam.pos;
        let moved = ensure_camera_safe(&map, &mut cam);
        assert!(!moved);
        assert_eq!(cam.pos, before);
    }

    #[test]
    fn a_camera_embedded_in_a_collider_is_pushed_to_free_ground() {
        let mut map = open_room();
        map.set_cell(2, 2, 1, CellKind::WallSolid, 0);
        let mut cam = CameraPose {
            pos: Vec3::new(2.5, 2.5, 1.5),
            yaw: 0.7,
            pitch: 0.1,
        };
        let moved = ensure_camera_safe(&map, &mut cam);
        assert!(
            moved,
            "a camera embedded in a fresh wall must be pushed out"
        );
        assert!(
            !camera_blocked(&map, &cam),
            "the recovered position must itself be safe"
        );
        // Only `.pos` is corrected — the player's look direction survives the nudge.
        assert_eq!(cam.yaw, 0.7);
        assert_eq!(cam.pitch, 0.1);
    }

    #[test]
    fn recovery_is_deterministic_across_repeated_calls() {
        let mut map = open_room();
        map.set_cell(2, 2, 1, CellKind::WallSolid, 0);
        let mut cam_a = CameraPose {
            pos: Vec3::new(2.5, 2.5, 1.5),
            yaw: 0.0,
            pitch: 0.0,
        };
        let mut cam_b = CameraPose {
            pos: Vec3::new(2.5, 2.5, 1.5),
            yaw: 0.0,
            pitch: 0.0,
        };
        ensure_camera_safe(&map, &mut cam_a);
        ensure_camera_safe(&map, &mut cam_b);
        assert_eq!(cam_a.pos, cam_b.pos);
    }
}
