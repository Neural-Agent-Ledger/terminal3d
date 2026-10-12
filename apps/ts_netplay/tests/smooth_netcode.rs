// assert_eq! on f32 yaw values is intentional: reconcile_pose must leave yaw
// byte-identical on a no-op (the exact-equal check is the spec).
#![allow(clippy::float_cmp)]
//! Unit + harness tests for smooth netcode: remote-entity interpolation and
//! local-player prediction reconciliation.
//!
//! All terminal-free and deterministic: no transport, no `re_present`. The
//! interpolation tests exercise the pure [`ts_netplay::interp::sample_at`]
//! helper and its `Interpolatable` impls; the reconciliation tests exercise the
//! pure [`ts_netplay::predict::reconcile_pose`] and the
//! `reconcile_local_player` system on a minimal headless App.

use std::f32::consts::PI;

use bevy::prelude::*;
use bevy::MinimalPlugins;
use glam::Vec3;

use re_world::{CameraPose, Map, PLAYER_RADIUS};
use ts_net::events::MoveIntent;
use ts_net::types::NetPlayer;
use ts_netplay::interp::{sample_at, PlayerSample};
use ts_netplay::predict::{self, reconcile_pose, LocalOwner};

// Server-mirrored move speed for the test mover (matches sim::PLAYER_SPEED).
const PLAYER_SPEED: f32 = 4.0;

// ───────────────────────── interpolation unit tests ─────────────────────────

#[test]
fn sample_at_midpoint_is_true_intermediate() {
    let buf = [
        (
            0.0,
            PlayerSample {
                pos: Vec3::new(0.0, 0.0, 0.0),
                yaw: 0.0,
            },
        ),
        (
            1.0,
            PlayerSample {
                pos: Vec3::new(2.0, 4.0, 0.0),
                yaw: 0.0,
            },
        ),
    ];
    let s = sample_at(&buf, 0.5).expect("midpoint sample");
    // Expect ≈ (1, 2, 0): strictly between the endpoints and not pinned to either.
    assert!((s.pos.x - 1.0).abs() < 1e-4, "x ≈ 1, got {}", s.pos.x);
    assert!((s.pos.y - 2.0).abs() < 1e-4, "y ≈ 2, got {}", s.pos.y);
    assert!((s.pos.z - 0.0).abs() < 1e-4, "z ≈ 0, got {}", s.pos.z);
    // Strictly intermediate (not within 1e-4 of either endpoint).
    assert!(s.pos.x > 1e-4 && (2.0 - s.pos.x) > 1e-4);
    assert!(s.pos.y > 1e-4 && (4.0 - s.pos.y) > 1e-4);
}

#[test]
fn sample_at_clamps_outside_range() {
    let buf = [
        (
            10.0,
            PlayerSample {
                pos: Vec3::new(1.0, 1.0, 1.0),
                yaw: 0.1,
            },
        ),
        (
            20.0,
            PlayerSample {
                pos: Vec3::new(5.0, 5.0, 5.0),
                yaw: 0.9,
            },
        ),
    ];
    // Before the oldest → hold oldest.
    let old = sample_at(&buf, -1.0).expect("clamp-old");
    assert_eq!(old.pos, Vec3::new(1.0, 1.0, 1.0));
    assert!(
        (old.yaw - 0.1).abs() < 1e-6,
        "clamp-old yaw, got {}",
        old.yaw
    );
    // After the newest → hold newest.
    let new = sample_at(&buf, 1000.0).expect("clamp-new");
    assert_eq!(new.pos, Vec3::new(5.0, 5.0, 5.0));
    assert!(
        (new.yaw - 0.9).abs() < 1e-6,
        "clamp-new yaw, got {}",
        new.yaw
    );
}

#[test]
fn sample_at_yaw_shortest_arc_near_pi() {
    // Just under +π to just under −π: the shortest arc crosses the ±π wrap, so
    // the midpoint yaw magnitude should stay NEAR π (~the wrap point), NOT near
    // 0 (which a naive scalar lerp of (π−0.1) and −(π−0.1) would give).
    let buf = [
        (
            0.0,
            PlayerSample {
                pos: Vec3::ZERO,
                yaw: PI - 0.1,
            },
        ),
        (
            1.0,
            PlayerSample {
                pos: Vec3::ZERO,
                yaw: -(PI - 0.1),
            },
        ),
    ];
    let s = sample_at(&buf, 0.5).expect("yaw midpoint");
    assert!(
        s.yaw.abs() > PI - 0.15,
        "shortest-arc yaw should be ≈ ±π, got {}",
        s.yaw
    );
    assert!(
        s.yaw.abs() > 3.0,
        "shortest-arc proof: |yaw| must be large, got {}",
        s.yaw
    );
}

#[test]
fn sample_at_vec3_bullet_midpoint() {
    // Bullet buffers store bare Vec3; verify the Vec3 Interpolatable path.
    let buf = [
        (0.0, Vec3::new(0.0, 0.0, 0.0)),
        (2.0, Vec3::new(4.0, 0.0, -2.0)),
    ];
    let p = sample_at(&buf, 1.0).expect("bullet midpoint");
    assert!((p.x - 2.0).abs() < 1e-4, "x ≈ 2, got {}", p.x);
    assert!((p.z - (-1.0)).abs() < 1e-4, "z ≈ -1, got {}", p.z);
}

// ──────────────────── prediction + reconciliation tests ─────────────────────

#[test]
fn reconcile_snaps_xy_leaves_z() {
    let mut cam = CameraPose {
        pos: Vec3::new(0.0, 0.0, 1.5),
        yaw: 0.0,
        pitch: 0.0,
    };
    let changed = reconcile_pose(&mut cam, Vec3::new(1.0, 1.0, 9.9), 0.5, 0.05, 0.02);
    assert!(changed, "large divergence must reconcile");
    assert!(
        (cam.pos.x - 1.0).abs() < 1e-5,
        "x snapped, got {}",
        cam.pos.x
    );
    assert!(
        (cam.pos.y - 1.0).abs() < 1e-5,
        "y snapped, got {}",
        cam.pos.y
    );
    // z must be UNTOUCHED (local gravity owns it).
    assert!(
        (cam.pos.z - 1.5).abs() < 1e-5,
        "z left alone, got {}",
        cam.pos.z
    );
    assert!((cam.yaw - 0.5).abs() < 1e-5, "yaw snapped, got {}", cam.yaw);
}

#[test]
fn reconcile_noop_when_within_eps() {
    let mut cam = CameraPose {
        pos: Vec3::new(1.0, 1.0, 1.5),
        yaw: 0.30,
        pitch: 0.0,
    };
    let before = cam.pos;
    let before_yaw = cam.yaw;
    let changed = reconcile_pose(
        &mut cam,
        Vec3::new(1.01, 1.01, 7.0),
        0.301,
        predict::RECONCILE_POS_EPS,
        predict::RECONCILE_YAW_EPS,
    );
    assert!(!changed, "tiny divergence must NOT reconcile");
    assert_eq!(cam.pos, before, "pose unchanged on no-op");
    assert_eq!(cam.yaw, before_yaw, "yaw unchanged on no-op");
}

/// Test-only mover mimicking `re_present`'s `held_move_sys` prediction: applies
/// the SAME math as the server's `apply_move` with a fixed dt, so the harness
/// does not depend on the (unavailable-in-test) present backend.
#[derive(Resource, Clone, Copy)]
struct TestIntent(MoveIntent);

#[derive(Resource, Clone, Copy)]
struct TestDt(f32);

fn test_mover(
    intent: Res<TestIntent>,
    dt: Res<TestDt>,
    map: Res<Map>,
    mut cam_q: Query<&mut CameraPose>,
) {
    let Ok(mut cam) = cam_q.single_mut() else {
        return;
    };
    let i = intent.0;
    let yaw = cam.yaw + i.turn;
    let (sy, cy) = (yaw.sin(), yaw.cos());
    let forward = i.forward.clamp(-1.0, 1.0);
    let strafe = i.strafe.clamp(-1.0, 1.0);
    let dx = (sy * forward + cy * strafe) * PLAYER_SPEED * dt.0;
    let dy = (cy * forward - sy * strafe) * PLAYER_SPEED * dt.0;
    cam.yaw = yaw;
    cam.move_with_collision(dx, dy, &map, PLAYER_RADIUS);
}

#[test]
fn prediction_then_reconcile_system_harness() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(re_world::test_room());
    app.insert_resource(TestDt(1.0 / 60.0));
    // Predict forward (+y) before any server entity exists.
    app.insert_resource(TestIntent(MoveIntent {
        forward: 1.0,
        strafe: 0.0,
        turn: 0.0,
        climb: 0.0,
    }));
    app.init_resource::<LocalOwner>();
    app.add_systems(
        Update,
        (test_mover, predict::reconcile_local_player).chain(),
    );

    // Start in an OPEN interior cell: (3,3,1) is empty (the test room has a
    // pillar at cell (2,2,1) and walls on the perimeter). Moving +y toward y≈3.8
    // stays inside the room.
    let cam_e = app
        .world_mut()
        .spawn(CameraPose {
            pos: Vec3::new(3.5, 3.5, 1.5),
            yaw: 0.0,
            pitch: 0.0,
        })
        .id();

    // 1. PREDICTION: local movement happens with NO server entity present.
    let start_y = app.world().get::<CameraPose>(cam_e).unwrap().pos.y;
    for _ in 0..5 {
        app.update();
    }
    let moved_y = app.world().get::<CameraPose>(cam_e).unwrap().pos.y;
    assert!(
        moved_y > start_y + 1e-4,
        "predicted local movement should advance +y with no server entity: {start_y} -> {moved_y}"
    );

    // 2. RECONCILIATION: spawn a DIVERGENT authoritative NetPlayer and stop
    //    predicting (zero intent) so the only change is the reconcile snap.
    app.insert_resource(TestIntent(MoveIntent {
        forward: 0.0,
        strafe: 0.0,
        turn: 0.0,
        climb: 0.0,
    }));
    let server_pos = Vec3::new(4.5, 1.5, 9.9); // clearly != predicted, divergent z
    app.world_mut().spawn(NetPlayer::new(server_pos, 0.0, 0));

    let z_before = app.world().get::<CameraPose>(cam_e).unwrap().pos.z;
    app.update();
    let cam = *app.world().get::<CameraPose>(cam_e).unwrap();
    assert!(
        (cam.pos.x - server_pos.x).abs() < 1e-4,
        "x reconciled to server, got {}",
        cam.pos.x
    );
    assert!(
        (cam.pos.y - server_pos.y).abs() < 1e-4,
        "y reconciled to server, got {}",
        cam.pos.y
    );
    assert!(
        (cam.pos.z - z_before).abs() < 1e-4,
        "z must stay local (not snapped to server z), got {}",
        cam.pos.z
    );
}
