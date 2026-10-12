//! Preview-camera fly controls: DISCRETE STEP-MOVE, collision-optional.
//!
//! These move `EditorState::cam` (the 3D preview camera) without touching the
//! 2D edit keys. Editor fly is **one fixed step per keypress**, NOT continuous
//! time-based motion. The reason: a terminal delivers key REPEAT as a stream of
//! discrete key messages (not a held-key state we can poll each frame), so
//! scaling each event by the frame `dt` made the real speed
//! `key-repeat-rate × FLY_SPEED × dt` — wildly OS/terminal dependent. A fixed
//! step per event makes "tap to nudge, hold to glide" behave the same on every
//! terminal: N presses always travel N × `FLY_STEP` cells. The `dt` argument is
//! retained for call-site compatibility but is intentionally ignored.
//!
//! With `noclip` off we route horizontal motion through `re_world`'s
//! `move_with_collision` (wall-slide), so the camera respects the same collision
//! the player does; with `noclip` on we fly freely through geometry.
//!
//! This module does not read keys itself: `edit`/`keymap` map the fly/turn keys
//! to a [`Fly`] action and call [`fly`]; this module only executes the action
//! against `state.map` (a `re_world::Map`) and `state.cam`. It knows nothing of
//! `re_render` or `re_level` — those are the caller's concern.
use crate::state::EditorState;
use re_world::Map;

/// World cells travelled per translate keypress on the PER-EVENT FALLBACK path
/// (terminals with no key-release reporting, where `HeldKeys` stays empty). Set to
/// the GAME's per-event fallback magnitude (`re_present`'s private `MOVE_STEP` =
/// 0.11) so that a held key's autorepeat stream GLIDES smoothly like first-person
/// play, instead of the old jarring 0.5-cell jumps. On kitty-capable terminals the
/// continuous `held_preview_mover` (in `main.rs`) drives movement instead, at the
/// game's `MOVE_SPEED`, and this step is not used. Fixed per event — independent of
/// frame rate and key-repeat rate.
const FLY_STEP: f32 = 0.11;
/// Radians turned per turn keypress on the per-event fallback path. Matches the
/// game's per-event fallback (`re_present`'s private `TURN_STEP` = 0.10).
const TURN_STEP: f32 = 0.10;
/// Collision radius used when noclip is off (matches the play camera probe).
const COLLIDE_RADIUS: f32 = 0.2;

/// Direction of a single fly action, resolved from a held/repeated key.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Fly {
    Forward,
    Back,
    StrafeLeft,
    StrafeRight,
    Up,
    Down,
    TurnLeft,
    TurnRight,
}

/// Apply ONE discrete fly step for a single keypress.
///
/// Movement is a fixed step per call ([`FLY_STEP`] / [`TURN_STEP`]) and does NOT
/// scale by `dt`: a terminal sends key-repeat as discrete messages, so a
/// per-event fixed step keeps fly speed identical across terminals (see the
/// module docs). `dt` is accepted for call-site compatibility and intentionally
/// ignored.
pub fn fly(state: &mut EditorState, action: Fly, _dt: f32) {
    let dist = FLY_STEP;
    let turn = TURN_STEP;
    let (sx, sy) = {
        let d = state.cam.dir();
        (d.0, d.1)
    };
    match action {
        Fly::Forward => translate_cam(state, sx * dist, sy * dist),
        Fly::Back => translate_cam(state, -sx * dist, -sy * dist),
        // Strafe: left vector is forward rotated +90° → (cos, -sin) given
        // forward = (sin, cos).
        Fly::StrafeLeft => translate_cam(state, sy * dist, -sx * dist),
        Fly::StrafeRight => translate_cam(state, -sy * dist, sx * dist),
        Fly::Up => vertical(state, dist),
        Fly::Down => vertical(state, -dist),
        Fly::TurnLeft => state.cam.yaw -= turn,
        Fly::TurnRight => state.cam.yaw += turn,
    }
}

/// Horizontal translation of the preview camera by a world-space `(dx, dy)`,
/// collision-aware unless `state.noclip` is on. Shared by the per-event fly steps
/// ([`fly`]) and the continuous game-parity held-key mover (`main::held_preview_mover`),
/// so both honour the SAME collision behaviour and the single `noclip` source of truth.
pub fn translate_cam(state: &mut EditorState, dx: f32, dy: f32) {
    if state.noclip {
        state.cam.pos.x += dx;
        state.cam.pos.y += dy;
    } else {
        // Borrow `map` separately so we can pass it to the camera method.
        let map: &Map = &state.map;
        let mut cam = state.cam;
        cam.move_with_collision(dx, dy, map, COLLIDE_RADIUS);
        state.cam = cam;
    }
}

/// Vertical motion. Noclip moves the raw z; clipped motion uses the world's
/// `apply_height` (which clamps to the interior air band and refuses solids).
fn vertical(state: &mut EditorState, dz: f32) {
    if state.noclip {
        state.cam.pos.z += dz;
    } else {
        let map: &Map = &state.map;
        let mut cam = state.cam;
        cam.apply_height(dz, map);
        state.cam = cam;
    }
}

/// Teleport the preview camera to the cell under the 2D cursor at the active z,
/// facing +y (yaw 0, the level's default forward). Placed at the cell centre so
/// it sits in interior air. Pure: same cursor/z → same pose.
pub fn jump_to_cursor(state: &mut EditorState) {
    let (cx, cy) = state.cursor;
    state.cam.pos.x = cx as f32 + 0.5;
    state.cam.pos.y = cy as f32 + 0.5;
    state.cam.pos.z = state.z as f32 + 0.5;
    state.cam.yaw = 0.0;
    state.cam.pitch = 0.0;
    state.status = format!("jumped to {cx},{cy},{}", state.z);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jump_centres_camera_on_cursor_cell() {
        let mut s = EditorState::new_empty(8, 8, 4);
        s.cursor = (3, 5);
        s.z = 2;
        jump_to_cursor(&mut s);
        assert_eq!(s.cam.pos.x, 3.5);
        assert_eq!(s.cam.pos.y, 5.5);
        assert_eq!(s.cam.pos.z, 2.5);
        assert_eq!(s.cam.yaw, 0.0);
    }

    #[test]
    fn jump_is_deterministic() {
        let mut a = EditorState::new_empty(8, 8, 4);
        let mut b = EditorState::new_empty(8, 8, 4);
        a.cursor = (4, 1);
        b.cursor = (4, 1);
        jump_to_cursor(&mut a);
        jump_to_cursor(&mut b);
        assert_eq!(a.cam.pos, b.cam.pos);
        assert_eq!(a.cam.yaw, b.cam.yaw);
    }

    #[test]
    fn noclip_forward_moves_along_facing() {
        let mut s = EditorState::new_empty(16, 16, 4);
        s.noclip = true;
        s.cam.yaw = 0.0; // faces +y
        let y0 = s.cam.pos.y;
        fly(&mut s, Fly::Forward, 1.0);
        assert!(s.cam.pos.y > y0, "forward at yaw 0 increases y");
    }

    #[test]
    fn back_moves_opposite_facing() {
        // The captured-mode `s`=Back binding maps to `Fly::Back`, which must move
        // the camera BACKWARD along its facing.
        let mut s = EditorState::new_empty(16, 16, 4);
        s.noclip = true;
        s.cam.yaw = 0.0; // faces +y
        let y0 = s.cam.pos.y;
        fly(&mut s, Fly::Back, 1.0);
        assert!(s.cam.pos.y < y0, "back at yaw 0 decreases y");
    }

    #[test]
    fn translate_cam_respects_noclip_for_held_mover() {
        // The continuous held-key mover routes through `translate_cam`; with noclip
        // ON it moves freely, exactly like the per-event fly path.
        let mut s = EditorState::new_empty(16, 16, 4);
        s.noclip = true;
        let x0 = s.cam.pos.x;
        translate_cam(&mut s, 0.5, 0.0);
        assert!(
            (s.cam.pos.x - (x0 + 0.5)).abs() < 1e-6,
            "noclip translate is direct"
        );
    }

    #[test]
    fn turn_steps_by_fixed_amount() {
        let mut s = EditorState::new_empty(8, 8, 4);
        let y0 = s.cam.yaw;
        fly(&mut s, Fly::TurnRight, 0.5);
        assert!((s.cam.yaw - (y0 + TURN_STEP)).abs() < 1e-6);
    }

    #[test]
    fn fly_step_is_constant_regardless_of_dt() {
        // Translate: same displacement no matter what dt is passed (it's ignored).
        let mut a = EditorState::new_empty(16, 16, 4);
        let mut b = EditorState::new_empty(16, 16, 4);
        a.noclip = true;
        b.noclip = true;
        a.cam.yaw = 0.0;
        b.cam.yaw = 0.0;
        let y0 = a.cam.pos.y;
        fly(&mut a, Fly::Forward, 0.001); // tiny dt
        fly(&mut b, Fly::Forward, 1000.0); // huge dt
        assert_eq!(a.cam.pos, b.cam.pos, "translate ignores dt");
        // One forward step advances exactly FLY_STEP cells along +y at yaw 0.
        assert!((a.cam.pos.y - (y0 + FLY_STEP)).abs() < 1e-6);

        // Turn: same yaw delta regardless of dt.
        let mut c = EditorState::new_empty(8, 8, 4);
        let mut d = EditorState::new_empty(8, 8, 4);
        fly(&mut c, Fly::TurnRight, 0.0);
        fly(&mut d, Fly::TurnRight, 999.0);
        assert!((c.cam.yaw - d.cam.yaw).abs() < 1e-6, "turn ignores dt");
        assert!((c.cam.yaw - TURN_STEP).abs() < 1e-6);
    }

    #[test]
    fn clipped_fly_into_wall_is_blocked() {
        let mut s = EditorState::new_empty(8, 8, 4);
        // Build a solid wall column just north of the camera.
        s.cam.pos = re_core::Vec3::new(3.5, 3.5, 1.5);
        s.cam.yaw = 0.0; // faces +y
        s.map.set(3, 4, 1, re_world::CellKind::WallSolid);
        s.noclip = false;
        let y0 = s.cam.pos.y;
        // Many steps toward the wall: collision must keep us short of it.
        for _ in 0..30 {
            fly(&mut s, Fly::Forward, 0.1);
        }
        assert!(s.cam.pos.y < 4.0, "blocked before entering the solid cell");
        assert!(s.cam.pos.y >= y0, "moved toward but not past the wall");
    }
}
