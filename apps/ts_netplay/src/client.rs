//! Client-side input, rendering, and the terminal present wiring.
//!
//! `ts_netplay` is the ONLY crate that may touch `re_present` (the terminal).
//! The client builds a `MoveIntent` from the present layer's `HeldKeys` and a
//! `FireInput` from `InteractPressed`, renders the replicated world+bullets into
//! the shared `Framebuffer`, and lets `re_present`'s backend present it.
//!
//! ## Local-prediction note (deliberate design)
//!
//! `re_present`'s terminal backend ALSO moves the local `CameraPose` entity from
//! the held keys each frame (responsive local view). We keep that: the local
//! camera is the player's eye and the source of the aim for `FireInput`. The
//! authoritative pose still lives on the server (which integrates our
//! `MoveIntent` against collision and owns hit detection); the client view is a
//! local prediction layered over the replicated bullets. The two are not
//! reconciled in this testbed — that is out of scope (spec §5).

use bevy::prelude::*;
use bevy_replicon::prelude::*;
use glam::Vec3;
use re_core::{Cell, Framebuffer, Layer, LayerBlend};
use re_present::{HeldKeys, InteractPressed, PresentSet};
use re_render::ascii_shader::FogParams;
use re_render::bullet_sprite::{splat_bullets, DEFAULT_FOG};
use re_render::Fov;
use re_world::{CameraPose, Map};
use ts_net::events::{FireInput, MoveIntent};

use crate::interp::{self, INTERP_DELAY};
use crate::predict;

/// Per-tick yaw step (radians) applied when a turn key is held. The present
/// backend also turns the local camera; this value only paces the `turn` field
/// sent to the server so the authoritative yaw tracks the local one.
const TURN_STEP: f32 = 0.06;

/// The arena map + render FOV, held as resources so the render system can read
/// them each frame without reloading.
#[derive(Resource, Clone)]
pub struct RenderFov(pub Fov);

/// Scratch overlay layers reused across frames so the bullet splat does not
/// allocate per frame. Resized to match the framebuffer in [`render_world`].
#[derive(Resource)]
pub struct BulletLayers {
    /// Opaque world copy (for bullet occlusion).
    opaque: Layer,
    /// Bullet glyph overlay (GlyphOnly).
    out: Layer,
    /// Persistent scratch for replicated bullet positions, cleared and refilled
    /// each frame so the per-frame collect does not allocate (matching the
    /// scratch-reuse the rest of the bullet pipeline relies on).
    positions: Vec<Vec3>,
}

impl Default for BulletLayers {
    fn default() -> Self {
        BulletLayers {
            opaque: Layer::new(1, 1, LayerBlend::Opaque),
            out: Layer::new(1, 1, LayerBlend::GlyphOnly),
            positions: Vec::new(),
        }
    }
}

/// Build a [`MoveIntent`] from the present layer's held-key set and send it to
/// the server every frame (the ordered channel coalesces to the latest).
///
/// Forward = W − S, strafe = D − A, turn = (→ − ←) × step, climb = ↑ − ↓.
///
/// The intent is also recorded into [`predict::InputHistory`] (stamped with the
/// client-local clock) so the prediction design is complete — see `predict`.
pub fn send_move_intent(
    held: Res<HeldKeys>,
    time: Res<Time>,
    mut history: ResMut<predict::InputHistory>,
    mut writer: MessageWriter<MoveIntent>,
) {
    let forward = held.forward as i32 as f32 - held.backward as i32 as f32;
    let strafe = held.strafe_right as i32 as f32 - held.strafe_left as i32 as f32;
    let turn = (held.turn_right as i32 as f32 - held.turn_left as i32 as f32) * TURN_STEP;
    let climb = held.pitch_up as i32 as f32 - held.pitch_down as i32 as f32;
    let intent = MoveIntent {
        forward,
        strafe,
        turn,
        climb,
    };
    history.record(time.elapsed_secs(), intent);
    writer.write(intent);
}

/// On an [`InteractPressed`] (E/Enter), read the LOCAL camera's aim and send a
/// [`FireInput`] to the server, which spawns the authoritative bullet.
pub fn send_fire_input(
    mut presses: MessageReader<InteractPressed>,
    cam: Query<&CameraPose>,
    mut writer: MessageWriter<FireInput>,
) {
    // Drain all presses this frame; one shot per press.
    let count = presses.read().count();
    if count == 0 {
        return;
    }
    let Ok(cam) = cam.single() else {
        return; // no local camera yet
    };
    for _ in 0..count {
        writer.write(FireInput {
            yaw: cam.yaw,
            pitch: cam.pitch,
        });
    }
}

/// Render the arena from the local camera plus the replicated bullets into the
/// shared [`Framebuffer`]. Ordered `.before(PresentSet)` so the present backend
/// shows this frame's pixels.
pub fn render_world(
    map: Res<Map>,
    fov: Res<RenderFov>,
    time: Res<Time>,
    bi: Res<interp::BulletInterp>,
    cam: Query<&CameraPose>,
    mut fb: ResMut<Framebuffer>,
    layers: ResMut<BulletLayers>,
) {
    let Ok(cam) = cam.single() else {
        return;
    };
    let fov = &fov.0;

    // 1. Opaque world raycast into the framebuffer.
    re_render::raycast_fov(&map, cam, fov, &mut fb);

    let (cols, rows) = (fb.cols(), fb.rows());
    if cols == 0 || rows == 0 {
        return;
    }

    // Split-borrow the two scratch layers from the single ResMut.
    let layers = layers.into_inner();

    // 2. Bridge the framebuffer into an opaque Layer for bullet occlusion.
    layers.opaque.resize(cols, rows);
    for r in 0..rows {
        for c in 0..cols {
            if let Some(cell) = fb.get(c, r) {
                layers.opaque.set(c, r, *cell);
            }
        }
    }

    // 3. Fill the persistent scratch buffer with INTERPOLATED bullet positions
    //    sampled at `now - INTERP_DELAY` (absorbs network jitter), then splat.
    //    `interpolated_bullets` clears + refills the scratch with no extra alloc.
    let render_t = time.elapsed_secs() - INTERP_DELAY;
    interp::interpolated_bullets(&bi, render_t, &mut layers.positions);
    layers.out.resize(cols, rows);
    let fog: FogParams = DEFAULT_FOG;
    splat_bullets(
        &layers.positions,
        cam,
        fov,
        &fog,
        &layers.opaque,
        &mut layers.out,
    );

    // 4. Composite the GlyphOnly bullet overlay back onto the framebuffer
    //    (depth-tested, keep bg) — reproduces re_core's GlyphOnly fold.
    for r in 0..rows {
        for c in 0..cols {
            let Some(e) = layers.out.get(c, r) else {
                continue;
            };
            let e = *e;
            let Some(below) = fb.get(c, r) else {
                continue;
            };
            if e.depth <= below.depth {
                let mut cc: Cell = *below;
                cc.glyph = e.glyph;
                cc.fg = e.fg;
                if e.bg != 0 {
                    cc.bg = e.bg;
                }
                fb.set(c, r, cc);
            }
        }
    }
}

/// Register the client gameplay systems (move/fire/render) on the client App.
/// `Map`, `RenderFov`, `Framebuffer`, the local `CameraPose` entity, and
/// `re_present::PresentPlugin` are wired by the caller in `main`.
pub fn add_client_systems(app: &mut App) {
    app.init_resource::<BulletLayers>()
        .init_resource::<interp::PlayerInterp>()
        .init_resource::<interp::BulletInterp>()
        .init_resource::<predict::InputHistory>()
        .init_resource::<predict::LocalOwner>()
        .add_systems(
            Update,
            (
                send_move_intent.run_if(in_state(ClientState::Connected)),
                send_fire_input.run_if(in_state(ClientState::Connected)),
                // Capture replicated snapshots, reconcile the predicted local
                // camera against the authoritative pose, THEN render — in that
                // order so the frame shows the reconciled camera + interpolated
                // remote entities.
                interp::capture_snapshots,
                predict::reconcile_local_player.after(interp::capture_snapshots),
                render_world
                    .after(predict::reconcile_local_player)
                    .before(PresentSet),
            ),
        );
}
