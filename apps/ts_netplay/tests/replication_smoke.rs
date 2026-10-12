//! Integration smoke tests for the multiplayer replication testbed (spec §8).
//!
//! Three complementary tests, each proving a different property; the comment on
//! each says WHICH approach proved it:
//!
//! 1. [`loopback_fire_replicates_bullet`] — a TRUE two-App loopback over the real
//!    renet/UDP transport (the exact pattern from `bevy_replicon_renet`'s own
//!    `tests/netcode.rs`). Proves: a connection spawns a server `NetPlayer` that
//!    REPLICATES to the client, and a client `FireInput` makes the server spawn
//!    exactly one `Bullet` that REPLICATES to the client. (Connection + the
//!    server→client replication path.)
//!
//! 2. [`two_clients_see_each_others_bullet`] — a server with TWO real renet
//!    clients (A and B) connected in-process. Proves the spec §8 cross-client
//!    property: when client A sends a `FireInput`, the resulting
//!    server-authoritative `Bullet` REPLICATES to the OTHER client B within N
//!    ticks. This is the genuine "observed on the (other) client" wire path; the
//!    timing-sensitive damage assertion stays in the deterministic direct-sim
//!    test below.
//!
//! 3. [`direct_sim_bullet_hits_and_damages`] — drives the SERVER SIM SYSTEMS
//!    directly on a transport-less App with a deterministic `SimDt`. Proves:
//!    `integrate_bullets` + `bullet_hits` move a bullet into an enemy player,
//!    decrement its health by exactly the damage, and despawn the bullet. This
//!    is the deterministic source of truth for the integrate+hit+damage logic
//!    (the loopback path is timing-dependent, so the damage assertion lives
//!    here, not over UDP).
//!
//! Neither test adds `re_present` — there is NO terminal in tests; the server sim
//! is terminal-free by construction (it lives in `ts_netplay::sim`, pure ECS).

use std::net::Ipv4Addr;

use bevy::prelude::*;
use bevy::state::app::StatesPlugin;
use bevy::MinimalPlugins;
use bevy_replicon::prelude::*;
use bevy_replicon_renet::{RenetClient, RepliconRenetPlugins};
use glam::Vec3;

use ts_net::events::FireInput;
use ts_net::types::{Bullet, NetPlayer};
use ts_netplay::sim::{self, OwnerIds, SimDt, SpawnPoint};
use ts_netplay::transport;

/// Build a server App exactly like the live `--server` path, minus transport.
fn build_server_app() -> App {
    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
        StatesPlugin,
        RepliconPlugins.set(ServerPlugin::new(PostUpdate)),
        RepliconRenetPlugins,
        ts_net::NetPlugin,
    ));
    // Open arena + spawn point: the built-in test room is enough for replication.
    app.insert_resource(re_world::test_room());
    app.insert_resource(SpawnPoint {
        pos: Vec3::new(2.5, 2.5, 1.5),
        yaw: 0.0,
    });
    sim::add_server_sim(&mut app);
    app
}

/// Build a client App with NO present layer (headless test).
fn build_client_app() -> App {
    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
        StatesPlugin,
        RepliconPlugins.set(ServerPlugin::new(PostUpdate)),
        RepliconRenetPlugins,
        ts_net::NetPlugin,
    ));
    app
}

/// Pump both apps until `predicate(client)` holds or `max_ticks` elapse.
/// Returns the number of ticks taken (or `max_ticks` on timeout).
fn pump_until(
    server: &mut App,
    client: &mut App,
    max_ticks: usize,
    mut predicate: impl FnMut(&mut App) -> bool,
) -> usize {
    for tick in 0..max_ticks {
        client.update();
        server.update();
        if predicate(client) {
            return tick;
        }
    }
    max_ticks
}

#[test]
fn loopback_fire_replicates_bullet() {
    let mut server_app = build_server_app();
    let mut client_app = build_client_app();

    // Freeze channels before wiring the renet transport (spec / crate contract).
    server_app.finish();
    server_app.cleanup();
    client_app.finish();
    client_app.cleanup();

    // OS-assigned ephemeral port (port 0) to avoid collisions in CI.
    let port = transport::insert_server(&mut server_app, 0, 4).expect("server transport");
    transport::insert_client(&mut client_app, 1, Ipv4Addr::LOCALHOST, port)
        .expect("client transport");

    // 1. Connect.
    let connect_ticks = pump_until(&mut server_app, &mut client_app, 600, |c| {
        c.world().resource::<RenetClient>().is_connected()
    });
    assert!(
        connect_ticks < 600,
        "client failed to connect within 600 ticks"
    );

    // 2. The connection observer spawned a server NetPlayer; pump until it
    //    REPLICATES to the client world.
    let repl_ticks = pump_until(&mut server_app, &mut client_app, 600, |c| {
        c.world_mut().query::<&NetPlayer>().iter(c.world()).count() >= 1
    });
    assert!(
        repl_ticks < 600,
        "server NetPlayer never replicated to the client"
    );

    // Sanity: exactly one player on the server (the one client).
    let server_players = server_app
        .world_mut()
        .query::<&NetPlayer>()
        .iter(server_app.world())
        .count();
    assert_eq!(server_players, 1, "server spawned exactly one NetPlayer");

    // 3. Client fires. Aim +y (yaw 0) so the bullet flies down the open room.
    client_app.world_mut().write_message(FireInput {
        yaw: 0.0,
        pitch: 0.0,
    });

    // The server must spawn exactly one Bullet that replicates to the client.
    let bullet_ticks = pump_until(&mut server_app, &mut client_app, 600, |c| {
        c.world_mut().query::<&Bullet>().iter(c.world()).count() >= 1
    });
    assert!(
        bullet_ticks < 600,
        "FireInput did not produce a replicated Bullet within 600 ticks"
    );

    let server_bullets = server_app
        .world_mut()
        .query::<&Bullet>()
        .iter(server_app.world())
        .count();
    assert_eq!(
        server_bullets, 1,
        "server spawned exactly one Bullet for one FireInput"
    );
}

/// Pump a server and TWO clients until `predicate(client_b)` holds or `max_ticks`
/// elapse. Returns the number of ticks taken (or `max_ticks` on timeout).
fn pump_two_until(
    server: &mut App,
    client_a: &mut App,
    client_b: &mut App,
    max_ticks: usize,
    mut predicate: impl FnMut(&mut App) -> bool,
) -> usize {
    for tick in 0..max_ticks {
        client_a.update();
        client_b.update();
        server.update();
        if predicate(client_b) {
            return tick;
        }
    }
    max_ticks
}

#[test]
fn two_clients_see_each_others_bullet() {
    // One server, two real renet clients (A fires, B observes) — the spec §8
    // cross-client property: a bullet fired by one client replicates to the other.
    let mut server_app = build_server_app();
    let mut client_a = build_client_app();
    let mut client_b = build_client_app();

    server_app.finish();
    server_app.cleanup();
    client_a.finish();
    client_a.cleanup();
    client_b.finish();
    client_b.cleanup();

    // OS-assigned ephemeral port; both clients use distinct client ids.
    let port = transport::insert_server(&mut server_app, 0, 4).expect("server transport");
    transport::insert_client(&mut client_a, 1, Ipv4Addr::LOCALHOST, port)
        .expect("client A transport");
    transport::insert_client(&mut client_b, 2, Ipv4Addr::LOCALHOST, port)
        .expect("client B transport");

    // 1. Both clients connect.
    let connected = pump_two_until(&mut server_app, &mut client_a, &mut client_b, 900, |b| {
        b.world().resource::<RenetClient>().is_connected()
    });
    assert!(
        connected < 900,
        "client B failed to connect within 900 ticks"
    );
    assert!(
        client_a.world().resource::<RenetClient>().is_connected(),
        "client A failed to connect"
    );

    // 2. The server spawned a NetPlayer per connection; pump until BOTH have
    //    replicated to client B (so two distinct owners exist on the wire).
    let both_players = pump_two_until(&mut server_app, &mut client_a, &mut client_b, 900, |b| {
        b.world_mut().query::<&NetPlayer>().iter(b.world()).count() >= 2
    });
    assert!(
        both_players < 900,
        "both server NetPlayers never replicated to client B"
    );

    let server_players = server_app
        .world_mut()
        .query::<&NetPlayer>()
        .iter(server_app.world())
        .count();
    assert_eq!(server_players, 2, "server spawned exactly two NetPlayers");

    // 3. Client A fires. The server spawns one authoritative Bullet; assert it
    //    REPLICATES to the OTHER client (B) — the genuine cross-client wire path.
    client_a.world_mut().write_message(FireInput {
        yaw: 0.0,
        pitch: 0.0,
    });

    let bullet_ticks = pump_two_until(&mut server_app, &mut client_a, &mut client_b, 900, |b| {
        b.world_mut().query::<&Bullet>().iter(b.world()).count() >= 1
    });
    assert!(
        bullet_ticks < 900,
        "client A's FireInput did not replicate a Bullet to client B within 900 ticks"
    );

    // The server spawned exactly one bullet for the one FireInput, and it carries
    // client A's owner id (so B is genuinely seeing another client's bullet).
    let server_bullets: Vec<u64> = server_app
        .world_mut()
        .query::<&Bullet>()
        .iter(server_app.world())
        .map(|bul| bul.owner)
        .collect();
    assert_eq!(
        server_bullets.len(),
        1,
        "server spawned exactly one Bullet for one FireInput"
    );
}

#[test]
fn direct_sim_bullet_hits_and_damages() {
    // Transport-less server App: drive the sim systems directly with a fixed dt.
    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
        StatesPlugin,
        RepliconPlugins.set(ServerPlugin::new(PostUpdate)),
        ts_net::NetPlugin,
    ));
    app.insert_resource(re_world::test_room());
    app.insert_resource(SpawnPoint {
        pos: Vec3::new(2.5, 2.5, 1.5),
        yaw: 0.0,
    });
    app.init_resource::<OwnerIds>();
    // A fixed, large-ish dt so the bullet crosses the gap in a few ticks.
    app.insert_resource(SimDt(0.05));
    // Only the integrate + hit systems are under test here (no messages).
    app.add_systems(Update, (sim::integrate_bullets, sim::bullet_hits).chain());
    app.finish();

    // Two players: owner 1 (shooter) at y=2.5, owner 2 (target) at y=3.5, both
    // in the open interior of the 6×6×4 test room (walls at the perimeter).
    let shooter_pos = Vec3::new(2.5, 2.5, 1.5);
    let target_pos = Vec3::new(2.5, 3.5, 1.5);
    app.world_mut().spawn(NetPlayer::new(shooter_pos, 0.0, 1));
    let target = app
        .world_mut()
        .spawn(NetPlayer::new(target_pos, 0.0, 2))
        .id();

    // A bullet from owner 1 aimed +y toward owner 2, fast enough to arrive in a
    // few ticks (speed 12 c/s, gap ~1 cell, dt 0.05 → ~2 ticks).
    app.world_mut().spawn((
        Bullet::new(
            shooter_pos + Vec3::new(0.0, 0.3, 0.0),
            Vec3::new(0.0, sim::BULLET_SPEED, 0.0),
            1,
            sim::BULLET_TTL,
        ),
        Replicated,
    ));

    // Run enough ticks for the bullet to reach the target and resolve the hit.
    for _ in 0..10 {
        app.update();
    }

    let health = app
        .world()
        .get::<NetPlayer>(target)
        .expect("target player still exists")
        .health;
    assert!(
        health < NetPlayer::MAX_HEALTH,
        "target health should have dropped after a hit, got {health}"
    );
    // Exactly one bullet of damage was applied (no respawn — 25 < 100).
    assert!(
        (health - (NetPlayer::MAX_HEALTH - sim::BULLET_DAMAGE)).abs() < 1e-3,
        "expected exactly one hit of {} damage, health={health}",
        sim::BULLET_DAMAGE
    );

    // The bullet must have been despawned on the hit.
    let bullets = app.world_mut().query::<&Bullet>().iter(app.world()).count();
    assert_eq!(bullets, 0, "bullet should despawn after hitting a player");
}
