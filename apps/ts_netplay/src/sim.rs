//! Server-authoritative simulation for the multiplayer testbed.
//!
//! Pure ECS + `re_world` math — NO terminal, NO render, NO transport. These are
//! the systems the SERVER app runs; the client app never adds them. They are
//! deliberately **ungated** (no `run_if(ServerState::Running)`): the direct-sim
//! integration test drives them on a transport-less server App, and the live
//! `--server` run only ever adds them to the server app, so the gate would only
//! cost an archetype scan with no behavioural benefit. The message-reading
//! systems naturally no-op when no `FromClient<…>` messages arrive (a fresh
//! server with no clients), which is the same effect a state gate would give.

use std::collections::HashMap;

use bevy::prelude::*;
use bevy_replicon::prelude::*;
use glam::Vec3;
use re_world::{CameraPose, Map, PLAYER_RADIUS};
use ts_net::events::{FireInput, MoveIntent};
use ts_net::types::{Bullet, NetPlayer};

// ───────────────────────────── tunables ─────────────────────────────────────

/// Bullet muzzle speed in cells/second.
pub const BULLET_SPEED: f32 = 12.0;
/// Bullet lifetime in seconds (despawned at `ttl <= 0`).
pub const BULLET_TTL: f32 = 3.0;
/// Damage applied per bullet hit.
pub const BULLET_DAMAGE: f32 = 25.0;
/// Sphere-overlap radius (cells) for a bullet→player hit.
pub const HIT_RADIUS: f32 = 0.6;
/// Player horizontal move speed in cells/second.
pub const PLAYER_SPEED: f32 = 4.0;
/// Distance ahead of the shooter the bullet spawns (clear of the shooter sphere).
pub const MUZZLE_OFFSET: f32 = 0.5;

/// Fixed simulation timestep (seconds) the server integration systems read.
///
/// Inserted as a resource so both the live server (run at ~60 Hz by
/// `ScheduleRunnerPlugin`) and the deterministic direct-sim test step the
/// bullet integration by a known, constant amount — no dependency on wall-clock
/// `Time` advancing, which `app.update()` in a test would not do.
#[derive(Resource, Clone, Copy, Debug)]
pub struct SimDt(pub f32);

impl Default for SimDt {
    fn default() -> Self {
        SimDt(1.0 / 60.0)
    }
}

/// Where a freshly connected (or respawned) player starts, in the loaded arena.
///
/// Inserted by the server setup so spawning and respawn share one source of
/// truth. `pos` is a cell-centre world position; `yaw` faces down the arena.
#[derive(Resource, Clone, Copy, Debug)]
pub struct SpawnPoint {
    /// Cell-centre world position new/respawning players appear at.
    pub pos: Vec3,
    /// Facing yaw (radians) new/respawning players start with.
    pub yaw: f32,
}

impl Default for SpawnPoint {
    fn default() -> Self {
        // Centre-ish of the level-01 arena floor (z=1 air band), facing +y.
        SpawnPoint {
            pos: Vec3::new(6.5, 4.5, 1.5),
            yaw: 0.0,
        }
    }
}

/// Server-side map from a connected-client Entity to the stable `u64` owner id
/// stamped onto that client's [`NetPlayer`] and every [`Bullet`] it fires.
///
/// `next` is monotonic so owner ids never collide across reconnects within a
/// session (matching the spec's "monotonic owner id" requirement).
#[derive(Resource, Default, Debug)]
pub struct OwnerIds {
    /// Next owner id to hand out.
    pub next: u64,
    /// connected-client Entity → owner id.
    pub map: HashMap<Entity, u64>,
}

impl OwnerIds {
    /// Assign (or reuse) an owner id for a connected-client entity.
    fn assign(&mut self, client: Entity) -> u64 {
        if let Some(&id) = self.map.get(&client) {
            return id;
        }
        let id = self.next;
        self.next += 1;
        self.map.insert(client, id);
        id
    }
}

// ──────────────────────────── connection ────────────────────────────────────

/// Observer: when the server registers a new connected client, assign it an
/// owner id and spawn its server-authoritative, replicated [`NetPlayer`].
///
/// The connected-client Entity is exactly what `ClientId::Client(entity)` wraps
/// and what `FromClient::client_id` reports, so storing `entity → owner` here
/// lets [`apply_move`] / [`apply_fire`] resolve a message back to its player.
pub fn on_client_connected(
    trigger: On<Add, ConnectedClient>,
    spawn: Res<SpawnPoint>,
    mut owners: ResMut<OwnerIds>,
    mut commands: Commands,
) {
    let client = trigger.entity;
    let owner = owners.assign(client);
    commands.spawn((NetPlayer::new(spawn.pos, spawn.yaw, owner), Replicated));
}

// ───────────────────────────── movement ─────────────────────────────────────

/// Drain each client's latest [`MoveIntent`] and apply it to that client's
/// [`NetPlayer`], collision-checked against the arena `Map` via `re_world`.
///
/// Forward scales the facing `(sin yaw, cos yaw)`; strafe scales the
/// right vector `(cos yaw, -sin yaw)`. Turn is added straight to yaw (the
/// client already paced it). A temp [`CameraPose`] borrows the player's pose so
/// the exact same collision routine the single-player game uses runs here.
pub fn apply_move(
    mut reader: MessageReader<FromClient<MoveIntent>>,
    owners: Res<OwnerIds>,
    map: Res<Map>,
    dt: Res<SimDt>,
    mut players: Query<&mut NetPlayer>,
) {
    for from in reader.read() {
        let Some(client) = from.client_id.entity() else {
            continue; // listen-server self; the testbed has no local player
        };
        let Some(&owner) = owners.map.get(&client) else {
            continue;
        };
        let intent: &MoveIntent = &from.message;

        for mut player in &mut players {
            if player.owner != owner {
                continue;
            }
            // Apply the turn first so movement uses the updated facing.
            let yaw = player.yaw + intent.turn;
            let (sy, cy) = (yaw.sin(), yaw.cos());
            let forward = intent.forward.clamp(-1.0, 1.0);
            let strafe = intent.strafe.clamp(-1.0, 1.0);
            // World delta: forward along facing, strafe along facing rotated −90°.
            let dx = (sy * forward + cy * strafe) * PLAYER_SPEED * dt.0;
            let dy = (cy * forward - sy * strafe) * PLAYER_SPEED * dt.0;

            let mut cam = CameraPose {
                pos: player.pos,
                yaw,
                pitch: 0.0,
            };
            cam.move_with_collision(dx, dy, &map, PLAYER_RADIUS);
            player.pos = cam.pos;
            player.yaw = cam.yaw;
        }
    }
}

// ────────────────────────────── firing ──────────────────────────────────────

/// Drain each client's [`FireInput`] and spawn a server-authoritative,
/// replicated [`Bullet`] at the shooting player's muzzle along the supplied aim.
pub fn apply_fire(
    mut reader: MessageReader<FromClient<FireInput>>,
    owners: Res<OwnerIds>,
    players: Query<&NetPlayer>,
    mut commands: Commands,
) {
    for from in reader.read() {
        let Some(client) = from.client_id.entity() else {
            continue;
        };
        let Some(&owner) = owners.map.get(&client) else {
            continue;
        };
        let aim: FireInput = from.message;
        let dir = aim.dir();
        if dir == Vec3::ZERO {
            continue;
        }
        for player in &players {
            if player.owner != owner {
                continue;
            }
            let muzzle = player.pos + dir * MUZZLE_OFFSET;
            commands.spawn((
                Bullet::new(muzzle, dir * BULLET_SPEED, owner, BULLET_TTL),
                Replicated,
            ));
        }
    }
}

// ──────────────────────────── integration ───────────────────────────────────

/// Advance every bullet: `pos += vel * dt`, decay `ttl`, and despawn on TTL
/// expiry or world-solid impact (the cell the bullet now occupies is solid).
pub fn integrate_bullets(
    dt: Res<SimDt>,
    map: Res<Map>,
    mut bullets: Query<(Entity, &mut Bullet)>,
    mut commands: Commands,
) {
    let dt = dt.0;
    for (entity, mut bullet) in &mut bullets {
        let vel = bullet.vel;
        bullet.pos += vel * dt;
        bullet.ttl -= dt;
        let solid = map
            .cell_at(
                bullet.pos.x.floor() as i32,
                bullet.pos.y.floor() as i32,
                bullet.pos.z.floor() as i32,
            )
            .is_solid();
        if bullet.ttl <= 0.0 || solid {
            commands.entity(entity).despawn();
        }
    }
}

/// Resolve bullet→player hits: a bullet that sphere-overlaps a NetPlayer it does
/// NOT own deals [`BULLET_DAMAGE`] and is despawned. A player at/below zero
/// health respawns (pos + health reset to the spawn point).
///
/// Hits are collected first (immutable bullet read), then applied (mutable
/// player write + bullet despawn) to avoid an aliasing borrow of the two
/// queries within the same loop.
pub fn bullet_hits(
    spawn: Res<SpawnPoint>,
    bullets: Query<(Entity, &Bullet)>,
    mut players: Query<(Entity, &mut NetPlayer)>,
    mut commands: Commands,
) {
    // (bullet entity, player entity) pairs to resolve this tick.
    let mut hits: Vec<(Entity, Entity)> = Vec::new();
    // A bullet can only hit ONE player per tick (the first overlap found).
    for (b_entity, bullet) in &bullets {
        for (p_entity, player) in &players {
            if player.owner == bullet.owner {
                continue;
            }
            if (bullet.pos - player.pos).length() < HIT_RADIUS {
                hits.push((b_entity, p_entity));
                break;
            }
        }
    }

    for (b_entity, p_entity) in hits {
        // Each bullet appears at most once in `hits` (the inner loop `break`s on
        // its first overlap), so a bullet is never despawned twice here. The
        // player lookup can still fail if it was despawned elsewhere, hence the
        // guarded `get_mut`; the despawn uses `try_despawn` for the same reason.
        if let Ok((_, mut player)) = players.get_mut(p_entity) {
            player.health -= BULLET_DAMAGE;
            if player.health <= 0.0 {
                player.pos = spawn.pos;
                player.yaw = spawn.yaw;
                player.health = NetPlayer::MAX_HEALTH;
            }
        }
        commands.entity(b_entity).try_despawn();
    }
}

/// Register the server simulation resources + systems on a server App.
///
/// Called by `main`'s `--server` path AND by the loopback integration test (so
/// the wiring under test is the wiring that ships). `Map` + `SpawnPoint` are
/// inserted by the caller (they depend on the chosen arena); this adds the
/// owner-id table, the default sim-dt, the connection observer, and the per-tick
/// systems in a deterministic order (move → fire → integrate → hits).
pub fn add_server_sim(app: &mut App) {
    app.init_resource::<OwnerIds>()
        .init_resource::<SimDt>()
        .add_observer(on_client_connected)
        .add_systems(
            Update,
            (apply_move, apply_fire, integrate_bullets, bullet_hits).chain(),
        );
}
