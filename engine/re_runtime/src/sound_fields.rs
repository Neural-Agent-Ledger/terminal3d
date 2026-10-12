//! Sound fields runtime: every play frame, read the level's authored
//! [`re_level::fields::ResolvedField`]s (`Level::fields()`), evaluate each
//! against the listener position, occlusion, `re_puzzle::PuzzleState`, the
//! campaign movement phase and a per-field attack/release envelope, select
//! the top 3 (+ any `room`-shaped fields, always additive) with 10%
//! hysteresis so the selection never flickers, and fold the result through
//! `re_audio::combine` into the [`re_audio::SoundFieldMix`] resource that
//! `re_audio::plugin::apply_sound_fields` pushes onto the live kira graph.
//!
//! Pure geometry/signal helpers are free functions so they are unit-testable
//! with no Bevy `World`: falloff curves, capsule distance, cone, the z×1.5
//! storey separation, the real per-level signals. The `Update` system
//! (`sound_fields_sys`) is the only piece that touches ECS resources.
//!
//! This crate knows nothing about a campaign, a cue tracker or a settings
//! screen — three seam resources the host fills instead, so this module
//! never reads a host type directly:
//!
//! * [`FieldContext`] — the current level's campaign placement (movement
//!   phase, fog distance, ambience mood). Written by the host's adapter on
//!   level enter.
//! * [`FieldSignals`] — the player's read-notes bitmask and shard tally.
//!   Written by the host's adapter each frame.
//! * [`FieldSettingsRes`] — a `Resource` wrapper around [`re_audio::FieldSettings`]
//!   (see that type's doc for why it is a wrapper, not the type itself).
//!   Written by the host's adapter on settings change.
//!
//! [`SoundFieldsPlugin`] is generic over the host's `States` enum: the host
//! supplies the state value that means "a level is live", and this crate's
//! own tests use a local `TestState`.

use std::collections::HashMap;

use bevy::prelude::*;
use re_audio::{combine, Archetype, FieldInput, FieldMix, SoundFieldMix};
use re_level::fields::{DriveSpec, Falloff, FieldBind, FieldShape, ResolvedField};
use re_puzzle::{Pipe as PuzzlePipe, PuzzleState};
use re_world::{CameraPose, Map};

use crate::audio_world;
use crate::session::{CurrentLevelData, PlayCamera};

// ─────────────────────────────── seam resources ──────────────────────────────
//
// The host fills these; this crate only reads them. Each is `Option<Res<_>>`
// at the read site and degrades to a neutral default when the host has not
// filled it (no current level, no cue tracker, …).

/// The current level's campaign placement, as plain data: `phase` is the
/// movement-phase index into a field's `phase: [f32; 4]` table (0..=3, clamped
/// defensively at the read site), `fog_far` is the authored fog-escalation
/// distance driving [`rot_floor`] (`None` outside the campaign), and `mood` is
/// the level's base ambience mood, fed through
/// [`re_audio::soundscape::harmony_root`] for the `phantom_hollow`/`live_wire`
/// archetypes' `aux`. Written by the host's adapter on level enter; absent (no
/// campaign level current, e.g. a demo/editor context) reads as phase 0 / no
/// fog floor / a silent harmony root — see [`sound_fields_sys`].
#[derive(Resource, Debug, Clone, Copy, PartialEq)]
pub struct FieldContext {
    /// Movement-phase index (0..=3), indexing a field's `phase` table.
    pub phase: u8,
    /// The campaign fog-escalation distance at the current level, if any.
    pub fog_far: Option<f32>,
    /// The level's base ambience mood.
    pub mood: re_audio::AmbienceMood,
}

/// The player's read-notes bitmask (`bind = read:N` gates bit `N`; 32 bits is
/// well over any authored level's note count) and the recovered colour-shard
/// tally (`drive = shards`, shard count / 7). Written by the host's adapter
/// each frame from its own cue tracker; absent reads as "nothing read yet, no
/// shards" — the same neutral state a fresh tracker starts at.
#[derive(Resource, Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FieldSignals {
    /// Bit `N` set means note `N` has been read.
    pub notes_read: u32,
    /// Recovered colour-shard count.
    pub shards: u32,
}

/// A `Resource` wrapper around [`re_audio::FieldSettings`] (the accessibility
/// intensity + reduced-effects toggle).
///
/// `re_audio::FieldSettings` does not itself derive `bevy::prelude::Resource`,
/// so this thin newtype is the seam instead of the bare type; `Deref` makes
/// every read site (`settings.intensity`, `settings.reduced`) read exactly
/// like the plain value. Once `re_audio` derives `Resource` on `FieldSettings`
/// directly, this wrapper can be deleted and the field renamed away in one
/// pass.
#[derive(Resource, Debug, Clone, Copy, Default, PartialEq)]
pub struct FieldSettingsRes(pub re_audio::FieldSettings);

impl std::ops::Deref for FieldSettingsRes {
    type Target = re_audio::FieldSettings;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// ─────────────────────────────── tuning constants ───────────────────────────

/// At most this many candidates are evaluated per frame.
const SELECT_MAX_CANDIDATES: usize = 24;
/// The top N by `priority × w` are audible.
const SELECT_TOP_N: usize = 3;
/// A challenger must beat the third-place field by 10% to replace it.
const HYSTERESIS_MARGIN: f32 = 0.10;
/// Shape vocabulary: z is scaled by this before the sphere-distance Euclidean
/// norm, so separate storeys do not bleed into each other.
const Z_STOREY_SCALE: f32 = 1.5;
/// Falloff table: `hard`'s 0.5-cell linear edge.
const HARD_EDGE_CELLS: f32 = 0.5;
/// Shape vocabulary: cone's smoothstep over the last 10° of the cone.
const CONE_SMOOTH_DEG: f32 = 10.0;
/// Occlusion: `weight *= 1 - 0.7 * occlusion`.
const OCCLUSION_STRENGTH: f32 = 0.7;
/// State signals: dwell counts seconds with (pre-dwell) weight above this.
const DWELL_WEIGHT_THRESHOLD: f32 = 0.1;
/// `idle_boost`: ×1.5 after 90 s idle.
const IDLE_THRESHOLD_S: f32 = 90.0;
const IDLE_BOOST_MULT: f32 = 1.5;
/// A movement/turn below this (cells / radians per frame comparison against the
/// last sampled pose) does not reset the idle timer — avoids a static camera
/// with floating-point jitter (or a paused frame) reading as "moving".
const IDLE_MOVE_EPS: f32 = 0.02;
const IDLE_YAW_EPS: f32 = 0.02;
/// Rot floor: `floor = clamp((20 - fog_far) / 10, 0, 1) × 0.20`.
const ROT_FLOOR_FOG_MAX: f32 = 20.0;
const ROT_FLOOR_FOG_SPAN: f32 = 10.0;
const ROT_FLOOR_SCALE: f32 = 0.20;
/// A fixed seed for the synthetic room-wide rot-floor field (never authored,
/// so it has no `[field]` name to hash).
const ROT_FLOOR_SEED: u64 = 0x726F_745F_666C_6F6F;

#[inline]
fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ─────────────────────────────── pure geometry ───────────────────────────────

#[inline]
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
fn len3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}
#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Sphere distance: Euclidean with z scaled ×1.5 (the shape vocabulary), so a
/// field one storey up/down does not bleed into the one below/above it. Returns
/// `(distance, nearest_point)`; the nearest point of a sphere is its center —
/// used both as the occlusion sample point and the cone anchor.
fn sphere_nearest(at: [f32; 3], p: [f32; 3]) -> (f32, [f32; 3]) {
    let d = sub3(p, at);
    let scaled = [d[0], d[1], d[2] * Z_STOREY_SCALE];
    (len3(scaled), at)
}

#[inline]
fn axis_outside(p: f32, lo: f32, hi: f32) -> f32 {
    if p < lo {
        lo - p
    } else if p > hi {
        p - hi
    } else {
        0.0
    }
}

/// Box distance: 0 inside, else the distance to the nearest face. The
/// nearest point is `p` clamped into the box — used for occlusion/cone.
fn box_nearest(min: [f32; 3], max: [f32; 3], p: [f32; 3]) -> (f32, [f32; 3]) {
    let dx = axis_outside(p[0], min[0], max[0]);
    let dy = axis_outside(p[1], min[1], max[1]);
    let dz = axis_outside(p[2], min[2], max[2]);
    let nearest = [
        p[0].clamp(min[0], max[0]),
        p[1].clamp(min[1], max[1]),
        p[2].clamp(min[2], max[2]),
    ];
    ((dx * dx + dy * dy + dz * dz).sqrt(), nearest)
}

/// Nearest distance from `p` to a segment `a->b`, plus the nearest point.
fn segment_nearest(a: [f32; 3], b: [f32; 3], p: [f32; 3]) -> (f32, [f32; 3]) {
    let ab = sub3(b, a);
    let ab_len2 = dot3(ab, ab);
    let t = if ab_len2 > 1e-9 {
        (dot3(sub3(p, a), ab) / ab_len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let nearest = [a[0] + ab[0] * t, a[1] + ab[1] * t, a[2] + ab[2] * t];
    (len3(sub3(p, nearest)), nearest)
}

/// Nearest distance from `p` to the polyline `points` — the distance to the
/// nearest segment of the expanded pipe path. A single point (a degenerate
/// path) is treated as a point distance; an empty path returns a very large
/// distance (never selected — defensive, `re_level::fields::resolve` never
/// emits an empty capsule since `follow` requires a resolved pipe path).
fn capsule_nearest(points: &[[f32; 3]], p: [f32; 3]) -> (f32, [f32; 3]) {
    if points.is_empty() {
        return (f32::MAX, p);
    }
    if points.len() == 1 {
        return (len3(sub3(p, points[0])), points[0]);
    }
    let mut best = (f32::MAX, points[0]);
    for w in points.windows(2) {
        let (d, n) = segment_nearest(w[0], w[1], p);
        if d < best.0 {
            best = (d, n);
        }
    }
    best
}

/// `(distance, anchor)` for any shape. `Room` has no meaningful geometry (it
/// reads as 1 everywhere in the level) — the listener itself is returned as the
/// anchor so a caller that still evaluates cone/occlusion on a `room` field
/// (never authored in practice) degrades to "no effect" rather than a panic.
fn shape_geometry(shape: &FieldShape, listener: [f32; 3]) -> (f32, [f32; 3]) {
    match shape {
        FieldShape::Sphere { at } => sphere_nearest(*at, listener),
        FieldShape::BoxShape { min, max } => box_nearest(*min, *max, listener),
        FieldShape::Capsule { points } => capsule_nearest(points, listener),
        FieldShape::Room => (0.0, listener),
    }
}

/// The falloff table. `radius <= inner` degenerates to a hard step at `inner`
/// (division-by-zero guard; not a case any authored field should hit since
/// `re_level::fields` validates `radius > 0` and defaults never coincide).
fn falloff_weight(falloff: Falloff, d: f32, inner: f32, radius: f32) -> f32 {
    match falloff {
        Falloff::Hard => {
            let half = HARD_EDGE_CELLS / 2.0;
            if d <= radius - half {
                1.0
            } else if d >= radius + half {
                0.0
            } else {
                ((radius + half - d) / HARD_EDGE_CELLS).clamp(0.0, 1.0)
            }
        }
        Falloff::Inverse => 1.0 / (1.0 + (d * d) / (radius * radius).max(1e-6)),
        Falloff::Smooth | Falloff::Linear => {
            let span = (radius - inner).max(1e-6);
            if span <= 1e-6 && radius <= inner {
                return if d <= inner { 1.0 } else { 0.0 };
            }
            let t = ((d - inner) / span).clamp(0.0, 1.0);
            if falloff == Falloff::Smooth {
                1.0 - smoothstep(t)
            } else {
                1.0 - t
            }
        }
    }
}

/// The cone factor: the shape distance × a cone factor (smoothstep over the
/// last 10° of the cone). `None` (no `facing`/`cone` authored) is always 1.0.
fn cone_factor(cone: Option<([f32; 3], f32)>, anchor: [f32; 3], listener: [f32; 3]) -> f32 {
    let Some((facing, half_deg)) = cone else {
        return 1.0;
    };
    let v = sub3(listener, anchor);
    let len = len3(v);
    if len < 1e-4 {
        return 1.0; // at the anchor itself: direction is undefined, treat as inside
    }
    let dir = [v[0] / len, v[1] / len, v[2] / len];
    let cos_a = dot3(facing, dir).clamp(-1.0, 1.0);
    let angle_deg = cos_a.acos().to_degrees();
    let ramp_start = (half_deg - CONE_SMOOTH_DEG).max(0.0);
    if angle_deg <= ramp_start {
        1.0
    } else if angle_deg >= half_deg {
        0.0
    } else {
        let span = (half_deg - ramp_start).max(1e-6);
        1.0 - smoothstep((angle_deg - ramp_start) / span)
    }
}

// ─────────────────────────────── state signals ───────────────────────

/// Everything a `bind =` gate or a `drive =` signal needs, gathered once per
/// frame by the system so the pure functions below take no ECS types.
struct SignalCtx<'a> {
    puzzle: Option<&'a PuzzleState>,
    /// [`FieldSignals::notes_read`] bitmask, for `bind = read:N`.
    notes_read: u32,
    /// [`FieldSignals::shards`], for `drive = shards` (shard count / 7).
    shards: u32,
    /// The live `re_render::MirrorClarity` override if set, else `1.0` (no
    /// per-mirror authored clarity is threaded through here yet; a fully
    /// sharp default is the conservative, least-surprising fallback for
    /// `drive = clarity` and the `false_echo` archetype's `aux`).
    clarity: f32,
    /// The `phantom_hollow` and `live_wire` archetypes both want the live
    /// harmony root in Hz (`re_audio::field.rs` falls back to A3/220 Hz when
    /// `aux <= 0`). The nearest thing a host tracks to "the live root" is
    /// the level-index-derived mood fed through
    /// `re_audio::soundscape::harmony_root` — an honest limitation: no
    /// resource records a story-beat-driven mood override at runtime, so a
    /// beat that swaps the bed mid-level is not reflected here until the next
    /// level load. The base level mood is still real data, not a stub.
    harmony_hz: f32,
}

fn node_in_degree(puzzle: &PuzzleState, node: usize) -> usize {
    puzzle.pipes.iter().filter(|p| p.to == node).count()
}

/// Σ over ON pipes into `node` of the source activation (`fed:N`).
fn node_fed(puzzle: &PuzzleState, node: usize) -> i32 {
    puzzle
        .pipes
        .iter()
        .filter(|p| p.to == node && p.is_on())
        .map(|p| puzzle.nodes.get(p.from).map_or(0, |n| n.activation))
        .sum()
}

fn door_ready(puzzle: &PuzzleState) -> bool {
    re_puzzle::check_win_inverted(puzzle, &puzzle.door_targets, &puzzle.target_off)
}

/// The `drive =` grammar, evaluated against live state. `[0, 1]`.
fn drive_signal(drive: &DriveSpec, ctx: &SignalCtx) -> f32 {
    if matches!(drive, DriveSpec::Const) {
        return 1.0;
    }
    let Some(puzzle) = ctx.puzzle else { return 0.0 };
    match drive {
        DriveSpec::Const => unreachable!("handled above"),
        DriveSpec::Lit(n) => puzzle
            .nodes
            .get(*n)
            .map_or(0.0, |nd| f32::from(nd.activation.clamp(0, 1) as i8)),
        DriveSpec::Fed(n) => f32::from(node_fed(puzzle, *n) > 0),
        DriveSpec::Starve(n) => {
            let Some(nd) = puzzle.nodes.get(*n) else {
                return 0.0;
            };
            let fed = node_fed(puzzle, *n);
            let t = nd.threshold;
            let in_deg = node_in_degree(puzzle, *n) as i32;
            let denom = t.min(in_deg);
            if fed > 0 && fed < t && denom > 0 {
                fed as f32 / denom as f32
            } else {
                0.0
            }
        }
        DriveSpec::Live(pipe) => puzzle.pipes.get(*pipe).map_or(0.0, |p| {
            f32::from(p.is_on() && puzzle.nodes.get(p.from).is_some_and(|n| n.activation == 1))
        }),
        DriveSpec::DeadDraw(pipe) => puzzle.pipes.get(*pipe).map_or(0.0, |p| {
            f32::from(p.is_on() && puzzle.nodes.get(p.from).is_some_and(|n| n.activation == 0))
        }),
        DriveSpec::Ring(idxs) => {
            if idxs.is_empty() {
                return 0.0;
            }
            let on = idxs
                .iter()
                .filter(|&&i| puzzle.pipes.get(i).is_some_and(PuzzlePipe::is_on))
                .count();
            on as f32 / idxs.len() as f32
        }
        DriveSpec::DoorReady => f32::from(door_ready(puzzle)),
        DriveSpec::Solved => f32::from(puzzle.solved),
        DriveSpec::CycleHeld => f32::from(puzzle.cycle_held),
        DriveSpec::Budget => {
            if puzzle.power_budget == 0 {
                0.0
            } else {
                puzzle.power_used() as f32 / puzzle.power_budget as f32
            }
        }
        DriveSpec::Seq => {
            let steps = puzzle.sequence.patterns.len();
            if steps == 0 {
                0.0
            } else {
                puzzle.sequence.progress as f32 / steps as f32
            }
        }
        DriveSpec::Clarity => ctx.clarity,
        DriveSpec::Shards => ctx.shards as f32 / 7.0,
    }
}

/// The `bind =` gate, `[0, 1]` (before `invert`).
fn bind_gate(bind: &FieldBind, ctx: &SignalCtx) -> f32 {
    if matches!(bind, FieldBind::None) {
        return 1.0;
    }
    let Some(puzzle) = ctx.puzzle else { return 0.0 };
    match bind {
        FieldBind::None => unreachable!("handled above"),
        FieldBind::Door => f32::from(door_ready(puzzle)),
        FieldBind::Node(n) => f32::from(puzzle.nodes.get(*n).is_some_and(|nd| nd.activation != 0)),
        FieldBind::Pipe(p) => f32::from(puzzle.pipes.get(*p).is_some_and(PuzzlePipe::is_on)),
        FieldBind::Budget(k) => f32::from(puzzle.power_used() >= *k),
        FieldBind::Seq(i) => f32::from(puzzle.sequence.progress as u32 >= *i),
        FieldBind::Fed(n) => f32::from(node_fed(puzzle, *n) > 0),
        FieldBind::Read(idx) => f32::from(*idx < 32 && (ctx.notes_read >> idx) & 1 != 0),
    }
}

// ─────────────────────────────── per-field evaluation ───────────────────────

/// This frame's un-smoothed evaluation of one field, before the attack/release
/// envelope (`FieldRuntime::weight`) and before top-3 selection.
struct FieldEval {
    /// `shape_falloff × occlusion × cone × gate × drive × phase` — everything
    /// in the per-field weight formula except `dwell_ramp` and `idle_boost`,
    /// which need this value's running history (`FieldRuntime::dwell`).
    pre_dwell: f32,
    /// Archetype-specific `aux` (clarity / strain ratio / budget / distance).
    aux: f32,
    /// Archetype-specific `variant` selector.
    variant: u8,
}

/// `phase` is [`FieldContext::phase`] (0..=3), clamped defensively so an
/// out-of-range value degrades to the last phase slot rather than panicking.
fn evaluate_field(
    field: &ResolvedField,
    listener: [f32; 3],
    map: Option<&Map>,
    ctx: &SignalCtx,
    phase: u8,
) -> FieldEval {
    let is_room = matches!(field.shape, FieldShape::Room);
    let (shape_w, anchor) = if is_room {
        (1.0, listener)
    } else {
        let (d, anchor) = shape_geometry(&field.shape, listener);
        let radius = match field.radius {
            re_level::fields::FieldRadius::Value(r) => r,
            re_level::fields::FieldRadius::Room => 1.0, // unreachable: Room radius implies Room shape
        };
        (
            falloff_weight(field.falloff, d, field.inner, radius),
            anchor,
        )
    };

    let cone = if is_room {
        1.0
    } else {
        cone_factor(field.cone, anchor, listener)
    };

    let occl = if !is_room && field.occlude {
        map.map_or(1.0, |m| {
            1.0 - OCCLUSION_STRENGTH * audio_world::occlusion_between(m, anchor, listener)
        })
    } else {
        1.0
    };

    let mut gate = bind_gate(&field.bind, ctx);
    if field.invert {
        gate = 1.0 - gate;
    }

    let drive_raw = drive_signal(&field.drive, ctx).clamp(0.0, 1.0);
    let drive_eff = field.base + (1.0 - field.base) * drive_raw;

    let phase_w = field.phase[(phase as usize).min(field.phase.len() - 1)];

    let pre_dwell = (shape_w * cone * occl * gate * drive_eff * phase_w).clamp(0.0, 1.0);

    let Some(archetype) = Archetype::from_name(&field.archetype) else {
        return FieldEval {
            pre_dwell: 0.0,
            aux: 0.0,
            variant: 0,
        };
    };

    let (aux, variant) = match archetype {
        Archetype::FalseEcho => {
            let memory = field.variant.as_deref() == Some("memory");
            (ctx.clarity, u8::from(memory))
        }
        Archetype::ThresholdStrain | Archetype::Brownout => (drive_raw, 0),
        Archetype::TailF => {
            let (d, _) = if is_room {
                (0.0, listener)
            } else {
                shape_geometry(&field.shape, listener)
            };
            (d, 0)
        }
        Archetype::MnemosHold => (0.0, u8::from(ctx.puzzle.is_some_and(|p| p.cycle_held))),
        Archetype::PhantomHollow | Archetype::LiveWire => (ctx.harmony_hz, 0),
        Archetype::SkyHush => {
            let v = match field.variant.as_deref() {
                Some("datarain") => 1,
                Some("aurora") => 2,
                Some("watcher") => 3,
                Some("storm") => 4,
                _ => 0,
            };
            (0.0, v)
        }
        _ => (0.0, 0),
    };

    FieldEval {
        pre_dwell,
        aux,
        variant,
    }
}

// ─────────────────────────────── mirror voice ────────────────────

/// The mirror voice's speed-of-sound scale: `2·d_plane·1.2 m / 343 m/s`.
const MIRROR_DISTANCE_SCALE: f32 = 1.2;
const SPEED_OF_SOUND_M_S: f32 = 343.0;
/// Plus one sixteenth (167 ms) — a sixteenth note at the 90 BPM default
/// (`transport.rs`), the same value the `POS` bus's fixed `DL` time uses.
const MIRROR_SIXTEENTH_S: f32 = 0.167;
/// Below this the listener is considered to be standing ON the mirror plane;
/// no reflection distance to speak of.
const MIRROR_MIN_PLANE_DIST: f32 = 1e-3;

/// "Mirror voice (native)": when a footstep lands inside a (non-
/// `variant=memory`) `false_echo` field's cone, it is re-emitted as a one-shot
/// at the listener's reflection across the mirror plane. Reuses the same
/// shape/cone geometry `evaluate_field` uses (`shape_geometry`, `cone_factor`)
/// rather than a second implementation, so "inside the field" means the same
/// thing here as it does for the continuous mix.
///
/// The mirror plane is derived from the shape's nearest-point-to-listener
/// (`shape_geometry`'s anchor) together with the authored `facing` direction
/// (the normal points into the room from the wall plane). For the documented
/// wall_decal use (a zero-thickness box authored flush against the wall), that
/// anchor already lies exactly on the wall plane — `box_nearest`'s per-axis
/// clamp degenerates to the shared min==max coordinate on the wall's axis. A
/// `sphere`/`capsule` host degrades to "reflect across the tangent plane at
/// the nearest point", which is a reasonable generalisation but is not a
/// documented use (`false_echo`'s carriers are wall mirrors).
///
/// Returns `(mirror_world_pos, delay_seconds)` for the first field (in
/// authored order) that gates the listener in — at most one mirror voice per
/// field, and this returns at most one across all fields, matching
/// `evaluate_field`'s FieldInput being one entry per field, not per-echo.
pub fn false_echo_mirror(fields: &[ResolvedField], listener: [f32; 3]) -> Option<([f32; 3], f32)> {
    for field in fields {
        if field.archetype != "false_echo" {
            continue;
        }
        if field.variant.as_deref() == Some("memory") {
            continue; // the memory variant has no mirror voice.
        }
        let Some((facing, half_deg)) = field.cone else {
            continue;
        };
        let radius = match field.radius {
            re_level::fields::FieldRadius::Value(r) => r,
            re_level::fields::FieldRadius::Room => continue, // a mirror is never room-shaped
        };
        let (d, plane_point) = shape_geometry(&field.shape, listener);
        if d > radius {
            continue;
        }
        if cone_factor(Some((facing, half_deg)), plane_point, listener) <= 0.0 {
            continue;
        }
        let d_plane = dot3(sub3(listener, plane_point), facing);
        if d_plane <= MIRROR_MIN_PLANE_DIST {
            continue; // at or behind the mirror plane: no reflection distance
        }
        let mirror_pos = [
            listener[0] - 2.0 * d_plane * facing[0],
            listener[1] - 2.0 * d_plane * facing[1],
            listener[2] - 2.0 * d_plane * facing[2],
        ];
        let delay_s =
            2.0 * d_plane * MIRROR_DISTANCE_SCALE / SPEED_OF_SOUND_M_S + MIRROR_SIXTEENTH_S;
        return Some((mirror_pos, delay_s));
    }
    None
}

// ─────────────────────────────── the runtime resource ───────────────────────

#[derive(Default)]
struct FieldMemory {
    /// The attack/release-smoothed `w_i` from last frame.
    weight: f32,
    /// Seconds accumulated with `pre_dwell > DWELL_WEIGHT_THRESHOLD` (resets
    /// the instant the field drops below threshold — the `dwell` signal).
    dwell_s: f32,
}

/// Per-level runtime memory: the attack/release envelope + dwell accumulator
/// per field (keyed by field name, stable across frames within one level) and
/// the idle timer. Re-armed on every level (re-)load so a new level's fields
/// never inherit a stale envelope from the level before it.
#[derive(Resource, Default)]
pub struct FieldRuntime {
    memory: HashMap<String, FieldMemory>,
    /// Last frame's top-3 selection (by field name), for the 10% hysteresis.
    selected: Vec<String>,
    idle_s: f32,
    last_pos: Option<[f32; 3]>,
    last_yaw: Option<f32>,
    armed_for: Option<String>,
}

impl FieldRuntime {
    /// The current attack/release-smoothed weight for the field named `name`
    /// (`FieldMemory::weight`, [`evaluate_mix`]'s per-field envelope state),
    /// or `0.0` if that name has never been evaluated by this runtime (a
    /// fresh [`FieldRuntime::default()`], or a name absent from the level).
    ///
    /// `FieldMemory` itself stays crate-private (it is bookkeeping, not part
    /// of the public per-field contract), but a caller that wants to
    /// *observe* the converged envelope — not just consume the resulting
    /// [`FieldMix`] — needs a read, not a rebuild of the smoothing math. The
    /// game's offline field-chain verification harness is exactly this: it
    /// logs the converged weight at each scripted snapshot for its README.
    #[must_use]
    pub fn weight_of(&self, name: &str) -> f32 {
        self.memory.get(name).map_or(0.0, |m| m.weight)
    }
}

/// The top-3 selection's hysteresis: keep last frame's top-3 unless a
/// challenger beats the weakest incumbent by more than
/// [`HYSTERESIS_MARGIN`]. `candidates` is `(name, priority × w)`, already
/// restricted to fields whose radius contains the listener. Room-shaped
/// fields are never candidates here — the caller adds them unconditionally
/// afterward, always in addition to the three.
fn select_top_n(candidates: &[(String, f32)], previous: &[String]) -> Vec<String> {
    let mut ranked = candidates.to_vec();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));

    let score_of = |name: &str| ranked.iter().find(|(n, _)| n == name).map(|(_, s)| *s);
    let mut incumbents: Vec<(String, f32)> = previous
        .iter()
        .filter_map(|n| score_of(n).map(|s| (n.clone(), s)))
        .collect();
    incumbents.truncate(SELECT_TOP_N);

    if incumbents.len() < SELECT_TOP_N {
        let mut result: Vec<String> = incumbents.into_iter().map(|(n, _)| n).collect();
        for (n, _) in &ranked {
            if result.len() >= SELECT_TOP_N {
                break;
            }
            if !result.contains(n) {
                result.push(n.clone());
            }
        }
        return result;
    }

    // Track the position of the weakest incumbent directly (first occurrence on a
    // tie) instead of re-finding it by float equality later — avoids `float_cmp`
    // without changing which slot gets replaced.
    let mut weakest_idx = 0usize;
    let mut weakest = incumbents[0].1;
    for (i, (_, s)) in incumbents.iter().enumerate().skip(1) {
        if *s < weakest {
            weakest = *s;
            weakest_idx = i;
        }
    }
    let mut result: Vec<String> = incumbents.iter().map(|(n, _)| n.clone()).collect();
    for (n, s) in &ranked {
        if result.contains(n) {
            continue;
        }
        if *s > weakest * (1.0 + HYSTERESIS_MARGIN) {
            // Replace the (first) incumbent holding the weakest score.
            result[weakest_idx] = n.clone();
        }
    }
    result
}

/// One-pole attack/release smoothing, dt-independent (`smooth(w_i, attack,
/// release)`; defaults 0.4 s / 1.2 s).
fn smooth_towards(current: f32, target: f32, attack_s: f32, release_s: f32, dt: f32) -> f32 {
    let tau = if target > current {
        attack_s
    } else {
        release_s
    };
    if tau <= 1e-4 {
        return target;
    }
    let alpha = 1.0 - (-dt / tau).exp();
    current + (target - current) * alpha
}

/// The room-wide `bit_rot` floor that follows the campaign fog curve:
/// `floor = clamp((20 - fog_far) / 10, 0, 1) × 0.20`. `None` for a
/// non-campaign level (no `fog_far` on record).
fn rot_floor(fog_far: Option<f32>) -> f32 {
    let Some(fog_far) = fog_far else { return 0.0 };
    ((ROT_FLOOR_FOG_MAX - fog_far) / ROT_FLOOR_FOG_SPAN).clamp(0.0, 1.0) * ROT_FLOOR_SCALE
}

/// Registers [`FieldRuntime`] and the per-frame evaluation system, running only
/// while the host's state equals `live` (the same generic state-seam idiom
/// `crate::zoom::ZoomPlugin` uses). Add after the host's audio plugin (owns
/// [`SoundFieldMix`]) and whatever fills
/// [`FieldContext`]/[`FieldSignals`]/[`FieldSettingsRes`].
pub struct SoundFieldsPlugin<S: States> {
    /// The state value meaning "a level is live".
    pub live: S,
}

impl<S: States> Plugin for SoundFieldsPlugin<S> {
    fn build(&self, app: &mut App) {
        app.init_resource::<FieldRuntime>()
            .add_systems(Update, sound_fields_sys.run_if(in_state(self.live.clone())));
    }
}

/// Stateless(-ish) core of the per-frame evaluation, taking primitive/`pub`
/// arguments only (no ECS `Res`/`Query`, no private `SignalCtx`, no game type)
/// so it is callable from outside any Bevy `App` — a headless "what would the
/// mix be at this listener position, on this level" query. `sound_fields_sys`
/// below is now a thin ECS-resource-unwrapping wrapper around this.
///
/// Two callers rely on this being real, not a stub: a host's own offline WAV
/// verification harness (which needs campaign `levels/*.lvl` this engine repo
/// does not ship, so it lives in the host's own test tree) drives a scripted
/// walk of listener positions through it and renders each resulting
/// `FieldMix` through the real kira graph via `re_audio::offline_backend`;
/// a future `ts_editor` live-audition tool needs the exact same "evaluate the
/// mix at a point" query with no running game session.
///
/// `runtime` carries the same per-field attack/release envelope and top-3
/// hysteresis memory `sound_fields_sys` threads across frames — a caller
/// stepping through a scripted walk with increasing `dt` gets the same
/// envelope behaviour a real playthrough would, not an idealized snapshot.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_mix(
    fields: &[ResolvedField],
    listener: [f32; 3],
    map: Option<&Map>,
    puzzle: Option<&PuzzleState>,
    notes_read: u32,
    shards: u32,
    clarity: f32,
    harmony_hz: f32,
    phase: u8,
    fog_far: Option<f32>,
    yaw: f32,
    dt: f32,
    runtime: &mut FieldRuntime,
    field_settings: &re_audio::FieldSettings,
) -> FieldMix {
    let ctx = SignalCtx {
        puzzle,
        notes_read,
        shards,
        clarity,
        harmony_hz,
    };

    // Idle timer: reset on any real movement or turn, else accumulate.
    let moved = runtime
        .last_pos
        .is_none_or(|p| len3(sub3(listener, p)) > IDLE_MOVE_EPS)
        || runtime
            .last_yaw
            .is_none_or(|y| (y - yaw).abs() > IDLE_YAW_EPS);
    runtime.idle_s = if moved { 0.0 } else { runtime.idle_s + dt };
    runtime.last_pos = Some(listener);
    runtime.last_yaw = Some(yaw);
    let idle_s = runtime.idle_s;

    let mut room_inputs: Vec<FieldInput> = Vec::new();
    let mut candidates: Vec<(String, f32, FieldInput)> = Vec::new();

    for field in fields {
        let Some(archetype) = Archetype::from_name(&field.archetype) else {
            continue;
        };
        let is_room = matches!(field.shape, FieldShape::Room);

        let eval = evaluate_field(field, listener, map, &ctx, phase);

        let mem = runtime.memory.entry(field.name.clone()).or_default();
        mem.dwell_s = if eval.pre_dwell > DWELL_WEIGHT_THRESHOLD {
            mem.dwell_s + dt
        } else {
            0.0
        };
        let dwell_ramp = if field.dwell <= 0.0 {
            1.0
        } else {
            (mem.dwell_s / field.dwell).clamp(0.0, 1.0)
        };
        let idle_factor = if field.idle_boost && idle_s >= IDLE_THRESHOLD_S {
            IDLE_BOOST_MULT
        } else {
            1.0
        };
        let target = eval.pre_dwell * dwell_ramp * idle_factor;

        mem.weight = smooth_towards(mem.weight, target, field.attack, field.release, dt);
        let w = mem.weight;

        let input = FieldInput {
            archetype,
            weight: w,
            depth: field.depth,
            priority: field.priority,
            aux: eval.aux,
            variant: eval.variant,
            seed: field.seed,
            bearing: Some(match &field.shape {
                FieldShape::Sphere { at } => *at,
                FieldShape::BoxShape { min, max } => [
                    (min[0] + max[0]) * 0.5,
                    (min[1] + max[1]) * 0.5,
                    (min[2] + max[2]) * 0.5,
                ],
                FieldShape::Capsule { points } => points.first().copied().unwrap_or(listener),
                FieldShape::Room => listener,
            }),
        };

        if is_room {
            if w > 0.0 {
                room_inputs.push(input);
            }
        } else {
            candidates.push((field.name.clone(), field.priority as f32 * w, input));
        }
    }

    if candidates.len() > SELECT_MAX_CANDIDATES {
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
        candidates.truncate(SELECT_MAX_CANDIDATES);
    }

    let scored: Vec<(String, f32)> = candidates.iter().map(|(n, s, _)| (n.clone(), *s)).collect();
    let selected_names = select_top_n(&scored, &runtime.selected);
    runtime.selected = selected_names.clone();

    let mut inputs: Vec<FieldInput> = selected_names
        .iter()
        .filter_map(|name| {
            candidates
                .iter()
                .find(|(n, _, _)| n == name)
                .map(|(_, _, i)| *i)
        })
        .filter(|i| i.weight > 0.0)
        .collect();
    inputs.extend(room_inputs);

    let floor = rot_floor(fog_far);
    if floor > 0.0 {
        inputs.push(FieldInput {
            archetype: Archetype::BitRot,
            weight: floor,
            depth: 1.0,
            priority: Archetype::BitRot.default_priority(),
            aux: 0.0,
            variant: 0,
            seed: ROT_FLOOR_SEED,
            bearing: None,
        });
    }

    combine(&inputs, field_settings)
}

/// Every play frame: evaluate every authored field, select the audible top 3 (+
/// room-shaped fields, always additive, + the fog-driven rot floor), smooth,
/// combine, and publish [`SoundFieldMix`]. Reads the three seam resources
/// ([`FieldContext`], [`FieldSignals`], [`FieldSettingsRes`]) the host's adapter
/// fills; each absent seam degrades exactly the way the pre-split code degraded
/// when its game-type source (`CurrentLevel`, `CueTracker`, `Settings`) was
/// itself absent.
#[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
fn sound_fields_sys(
    cam_q: Query<&CameraPose, With<PlayCamera>>,
    level: Option<Res<CurrentLevelData>>,
    puzzle: Option<Res<PuzzleState>>,
    map: Option<Res<Map>>,
    signals: Option<Res<FieldSignals>>,
    ctx: Option<Res<FieldContext>>,
    clarity: Option<Res<re_render::MirrorClarity>>,
    settings: Option<Res<FieldSettingsRes>>,
    time: Res<Time>,
    mut runtime: ResMut<FieldRuntime>,
    mut mix: ResMut<SoundFieldMix>,
) {
    let Ok(cam) = cam_q.single() else { return };
    let Some(lvl) = level.as_ref().and_then(|l| l.0.as_ref()) else {
        mix.0 = FieldMix::NEUTRAL;
        return;
    };

    let dt = time.delta_secs();
    let listener = [cam.pos.x, cam.pos.y, cam.pos.z];

    // Re-arm the per-field envelope on a level change so a new level's fields
    // never inherit a stale attack/release/dwell state from the one before it.
    let level_key = lvl.name.clone();
    if runtime.armed_for.as_deref() != Some(level_key.as_str()) {
        runtime.memory.clear();
        runtime.selected.clear();
        runtime.armed_for = Some(level_key);
    }

    // No `FieldContext` (e.g. a demo/editor context with no campaign index)
    // reads as phase 0 / no fog floor / a silent harmony root — matching the
    // pre-split "no `Res<CurrentLevel>`" fallback exactly.
    let (phase, fog_far, harmony_hz) = match ctx.as_deref() {
        Some(c) => (
            c.phase,
            c.fog_far,
            re_audio::soundscape::harmony_root(c.mood),
        ),
        None => (0u8, None, 0.0),
    };
    let field_settings = settings
        .as_deref()
        .map_or_else(re_audio::FieldSettings::default, |s| s.0);

    mix.0 = evaluate_mix(
        lvl.fields(),
        listener,
        map.as_deref(),
        puzzle.as_deref(),
        signals.as_ref().map_or(0, |s| s.notes_read),
        signals.as_ref().map_or(0, |s| s.shards),
        clarity
            .as_ref()
            .map_or(1.0, |c| c.override_value.unwrap_or(1.0).clamp(0.0, 1.0)),
        harmony_hz,
        phase,
        fog_far,
        cam.yaw,
        dt,
        &mut runtime,
        &field_settings,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── falloff ──────────────────────────────────────────────────────────

    #[test]
    fn smooth_falloff_is_1_at_inner_and_0_at_radius() {
        assert!((falloff_weight(Falloff::Smooth, 1.0, 1.0, 6.0) - 1.0).abs() < 1e-6);
        assert!(falloff_weight(Falloff::Smooth, 6.0, 1.0, 6.0).abs() < 1e-6);
        let mid = falloff_weight(Falloff::Smooth, 3.5, 1.0, 6.0);
        assert!(mid > 0.0 && mid < 1.0, "{mid}");
    }

    #[test]
    fn linear_falloff_is_exactly_linear_in_t() {
        let w = falloff_weight(Falloff::Linear, 3.5, 1.0, 6.0);
        // t = (3.5-1)/(6-1) = 0.5 -> 1-t = 0.5
        assert!((w - 0.5).abs() < 1e-5, "{w}");
    }

    #[test]
    fn inverse_falloff_is_half_at_d_equals_radius() {
        let w = falloff_weight(Falloff::Inverse, 6.0, 1.0, 6.0);
        assert!((w - 0.5).abs() < 1e-5, "{w}");
    }

    #[test]
    fn hard_falloff_steps_with_a_half_cell_edge() {
        assert!((falloff_weight(Falloff::Hard, 5.0, 1.0, 6.0) - 1.0).abs() < 1e-5);
        assert!(falloff_weight(Falloff::Hard, 6.5, 1.0, 6.0).abs() < 1e-5);
        let edge = falloff_weight(Falloff::Hard, 6.0, 1.0, 6.0);
        assert!(edge > 0.0 && edge < 1.0, "{edge}");
    }

    // ── shape geometry ───────────────────────────────────────────────────

    #[test]
    fn sphere_distance_scales_z_by_1_5_so_storeys_do_not_bleed() {
        let (d_flat, _) = sphere_nearest([0.0, 0.0, 0.0], [3.0, 0.0, 0.0]);
        let (d_up, _) = sphere_nearest([0.0, 0.0, 0.0], [0.0, 0.0, 3.0]);
        assert!((d_flat - 3.0).abs() < 1e-5);
        assert!((d_up - 4.5).abs() < 1e-5, "{d_up}"); // 3 * 1.5
    }

    #[test]
    fn box_distance_is_zero_inside_and_positive_outside() {
        let (d_in, _) = box_nearest([0.0, 0.0, 0.0], [4.0, 4.0, 4.0], [2.0, 2.0, 2.0]);
        assert_eq!(d_in, 0.0);
        let (d_out, _) = box_nearest([0.0, 0.0, 0.0], [4.0, 4.0, 4.0], [7.0, 0.0, 0.0]);
        assert!((d_out - 3.0).abs() < 1e-5);
    }

    #[test]
    fn capsule_distance_is_the_nearest_segment() {
        let path = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0]];
        let (d, n) = capsule_nearest(&path, [5.0, 3.0, 0.0]);
        assert!((d - 3.0).abs() < 1e-5, "{d}");
        assert!((n[0] - 5.0).abs() < 1e-5 && n[1].abs() < 1e-5);
    }

    // ── cone ─────────────────────────────────────────────────────────────

    #[test]
    fn cone_factor_is_1_dead_ahead_and_0_well_outside() {
        let facing = [1.0, 0.0, 0.0];
        let anchor = [0.0, 0.0, 0.0];
        let ahead = cone_factor(Some((facing, 35.0)), anchor, [5.0, 0.0, 0.0]);
        assert!((ahead - 1.0).abs() < 1e-5, "{ahead}");
        let behind = cone_factor(Some((facing, 35.0)), anchor, [-5.0, 0.0, 0.0]);
        assert_eq!(behind, 0.0);
    }

    #[test]
    fn cone_factor_ramps_smoothly_over_the_last_10_degrees() {
        let facing = [1.0, 0.0, 0.0];
        let anchor = [0.0, 0.0, 0.0];
        // 30 degrees off-axis, half-angle 35: inside the 10-degree ramp band.
        let angle = 30f32.to_radians();
        let dir = [angle.cos(), angle.sin(), 0.0];
        let p = [dir[0] * 5.0, dir[1] * 5.0, 0.0];
        let w = cone_factor(Some((facing, 35.0)), anchor, p);
        assert!(w > 0.0 && w < 1.0, "{w}");
    }

    // ── drive / bind signals ─────────────────────────────────────────────

    fn tiny_puzzle() -> PuzzleState {
        // n0 --on(1)--> n1 (threshold 2, in-degree 2 via n0 and n2), n2 --off--> n1
        let mut nodes = vec![
            re_puzzle::PuzzleNode {
                threshold: 0,
                is_input: true,
                is_output: false,
                activation: 1,
            },
            re_puzzle::PuzzleNode {
                threshold: 2,
                is_input: false,
                is_output: true,
                activation: 0,
            },
            re_puzzle::PuzzleNode {
                threshold: 0,
                is_input: true,
                is_output: false,
                activation: 0,
            },
        ];
        nodes[1].activation = 0;
        let pipes = vec![
            PuzzlePipe {
                from: 0,
                to: 1,
                state: re_level::PipeState::On,
                path: vec![],
                switch_cell: None,
            },
            PuzzlePipe {
                from: 2,
                to: 1,
                state: re_level::PipeState::Off,
                path: vec![],
                switch_cell: None,
            },
        ];
        PuzzleState {
            nodes,
            pipes,
            inputs: vec![1, 0, 0],
            door_targets: vec![1],
            target_off: vec![],
            sequence: re_puzzle::SequenceLock::default(),
            cycle_keep: false,
            cycle_held: false,
            power_budget: 0,
            solved: false,
            attempts: 0,
        }
    }

    #[test]
    fn starve_signal_matches_the_fed_over_min_t_in_degree_formula() {
        let p = tiny_puzzle();
        // fed(1) = activation(0) via the ON pipe = 1; T=2, in_degree=2 -> 1/2
        let v = drive_signal(&DriveSpec::Starve(1), &ctx_for(&p));
        assert!((v - 0.5).abs() < 1e-5, "{v}");
    }

    #[test]
    fn dead_draw_is_true_only_when_the_pipe_is_on_and_its_source_is_dark() {
        let mut p = tiny_puzzle();
        // pipe 1 (2->1) is OFF -> dead_draw is 0 regardless of source.
        assert_eq!(drive_signal(&DriveSpec::DeadDraw(1), &ctx_for(&p)), 0.0);
        p.pipes[1].state = re_level::PipeState::On;
        // now ON, source (node 2) activation is 0 -> dead_draw fires.
        assert_eq!(drive_signal(&DriveSpec::DeadDraw(1), &ctx_for(&p)), 1.0);
    }

    #[test]
    fn live_is_true_only_when_on_and_source_fires() {
        let p = tiny_puzzle();
        assert_eq!(drive_signal(&DriveSpec::Live(0), &ctx_for(&p)), 1.0);
        assert_eq!(drive_signal(&DriveSpec::Live(1), &ctx_for(&p)), 0.0);
    }

    #[test]
    fn budget_signal_is_zero_when_unlimited() {
        let p = tiny_puzzle();
        assert_eq!(drive_signal(&DriveSpec::Budget, &ctx_for(&p)), 0.0);
    }

    #[test]
    fn door_ready_matches_check_win_inverted() {
        let mut p = tiny_puzzle();
        assert_eq!(drive_signal(&DriveSpec::DoorReady, &ctx_for(&p)), 0.0);
        p.nodes[1].activation = 1;
        assert_eq!(drive_signal(&DriveSpec::DoorReady, &ctx_for(&p)), 1.0);
    }

    fn ctx_for(p: &PuzzleState) -> SignalCtx<'_> {
        SignalCtx {
            puzzle: Some(p),
            notes_read: 0,
            shards: 0,
            clarity: 1.0,
            harmony_hz: 220.0,
        }
    }

    #[test]
    fn bind_fed_gate_matches_the_fed_greater_than_zero_rule() {
        let p = tiny_puzzle();
        assert_eq!(bind_gate(&FieldBind::Fed(1), &ctx_for(&p)), 1.0);
        assert_eq!(bind_gate(&FieldBind::Fed(0), &ctx_for(&p)), 0.0); // node 0 has no incoming pipes
    }

    #[test]
    fn no_puzzle_resource_means_every_signal_and_gate_reads_dark_except_const() {
        let ctx = SignalCtx {
            puzzle: None,
            notes_read: 0,
            shards: 0,
            clarity: 1.0,
            harmony_hz: 220.0,
        };
        assert_eq!(drive_signal(&DriveSpec::Const, &ctx), 1.0);
        assert_eq!(drive_signal(&DriveSpec::Solved, &ctx), 0.0);
        assert_eq!(bind_gate(&FieldBind::Node(0), &ctx), 0.0);
        assert_eq!(bind_gate(&FieldBind::None, &ctx), 1.0);
    }

    // ── rot floor ────────────────────────────────────────────────────────

    #[test]
    fn rot_floor_matches_the_l1_l6_l12_worked_examples() {
        assert!((rot_floor(Some(20.0)) - 0.0).abs() < 1e-5);
        // (20-15)/10 = 0.5 -> 0.5 * 0.20 = 0.10, the doc's worked L6 example.
        assert!((rot_floor(Some(15.0)) - 0.10).abs() < 1e-4);
        assert!((rot_floor(Some(0.0)) - 0.20).abs() < 1e-4);
        assert_eq!(rot_floor(None), 0.0);
    }

    // ── smoothing ────────────────────────────────────────────────────────

    #[test]
    fn smoothing_moves_toward_target_and_reaches_it_over_many_frames() {
        let mut w = 0.0;
        for _ in 0..500 {
            w = smooth_towards(w, 1.0, 0.4, 1.2, 1.0 / 60.0);
        }
        assert!(w > 0.99, "{w}");
        for _ in 0..500 {
            w = smooth_towards(w, 0.0, 0.4, 1.2, 1.0 / 60.0);
        }
        assert!(w < 0.01, "{w}");
    }

    #[test]
    fn smoothing_never_overshoots_or_goes_negative() {
        let mut w = 0.0;
        for _ in 0..10 {
            w = smooth_towards(w, 1.0, 0.4, 1.2, 1.0 / 60.0);
            assert!((0.0..=1.0).contains(&w), "{w}");
        }
    }

    // ── top-3 selection + hysteresis ─────────────────────────────────────

    #[test]
    fn selection_picks_the_top_3_by_score_with_no_history() {
        let candidates = vec![
            ("a".to_string(), 1.0),
            ("b".to_string(), 5.0),
            ("c".to_string(), 3.0),
            ("d".to_string(), 4.0),
        ];
        let sel = select_top_n(&candidates, &[]);
        assert_eq!(sel.len(), 3);
        assert!(sel.contains(&"b".to_string()));
        assert!(sel.contains(&"d".to_string()));
        assert!(sel.contains(&"c".to_string()));
        assert!(!sel.contains(&"a".to_string()));
    }

    #[test]
    fn selection_keeps_the_incumbent_when_the_challenger_beats_it_by_less_than_10_percent() {
        let previous = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        // "d" barely edges out the weakest incumbent "c" (10.0 vs 10.5 = +5%).
        let candidates = vec![
            ("a".to_string(), 30.0),
            ("b".to_string(), 20.0),
            ("c".to_string(), 10.0),
            ("d".to_string(), 10.5),
        ];
        let sel = select_top_n(&candidates, &previous);
        assert!(sel.contains(&"c".to_string()), "{sel:?}");
        assert!(!sel.contains(&"d".to_string()), "{sel:?}");
    }

    #[test]
    fn selection_replaces_the_incumbent_once_the_challenger_clears_the_10_percent_margin() {
        let previous = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let candidates = vec![
            ("a".to_string(), 30.0),
            ("b".to_string(), 20.0),
            ("c".to_string(), 10.0),
            ("d".to_string(), 12.0), // +20%, clears the 10% margin
        ];
        let sel = select_top_n(&candidates, &previous);
        assert!(sel.contains(&"d".to_string()), "{sel:?}");
        assert!(!sel.contains(&"c".to_string()), "{sel:?}");
    }

    // ── level-driven evaluation (real `.lvl` text) ──────────────────────

    fn load_field_level(extra: &str) -> re_level::Level {
        let src = format!(
            "[meta]\nname = fieldtest\n[room]\nsize = 20 20 10\n[pedestal]\ncenter = 1,1,1\nvolume = 1 1 1\n\
             [door]\npos = 5 5 1\n[player]\nspawn = 2 2 1\n{extra}"
        );
        re_level::load_string(&src, "fieldtest.lvl").unwrap()
    }

    #[test]
    fn a_field_far_outside_its_radius_evaluates_to_zero_weight() {
        let lvl = load_field_level("[field river]\nkind = lethe_wash\nat = 1,1,1\nradius = 6\n");
        let field = &lvl.fields()[0];
        let ctx = SignalCtx {
            puzzle: None,
            notes_read: 0,
            shards: 0,
            clarity: 1.0,
            harmony_hz: 220.0,
        };
        let eval = evaluate_field(field, [50.0, 50.0, 1.0], None, &ctx, 0);
        assert_eq!(eval.pre_dwell, 0.0);
    }

    #[test]
    fn a_field_at_its_own_position_is_at_full_weight_when_gated_open() {
        // `phase = 1 1 1 1` overrides lethe_wash's own default phase table
        // (P1 0.3) so this test isolates the shape/gate/drive path.
        let lvl = load_field_level(
            "[field river]\nkind = lethe_wash\nat = 4,4,1\nradius = 6\nphase = 1 1 1 1\n",
        );
        let field = &lvl.fields()[0];
        let ctx = SignalCtx {
            puzzle: None,
            notes_read: 0,
            shards: 0,
            clarity: 1.0,
            harmony_hz: 220.0,
        };
        let eval = evaluate_field(field, [4.0, 4.0, 1.0], None, &ctx, 0);
        assert!((eval.pre_dwell - 1.0).abs() < 1e-4, "{}", eval.pre_dwell);
    }

    #[test]
    fn watcher_gaze_base_keeps_it_faintly_present_even_when_its_drive_node_is_dark() {
        // `phase = 1 1 1 1` overrides watcher_gaze's own default phase table
        // (P1 is 0 — "the gaze isn't a P1 force") so this test isolates `base`.
        let lvl = load_field_level(
            "[node unit]\npos = 4 4 1\n\
             [field gaze]\nkind = watcher_gaze\nat = 4,4,1\nradius = 12\ndrive = lit:unit\nphase = 1 1 1 1\n",
        );
        let field = &lvl.fields()[0];
        assert!((field.base - 0.35).abs() < 1e-4); // archetype default, from re_level::fields
        let mut p = PuzzleState::from_level(&lvl).unwrap();
        p.nodes[0].activation = 0;
        let ctx = SignalCtx {
            puzzle: Some(&p),
            notes_read: 0,
            shards: 0,
            clarity: 1.0,
            harmony_hz: 220.0,
        };
        let dark = evaluate_field(field, [4.0, 4.0, 1.0], None, &ctx, 0);
        assert!(
            dark.pre_dwell > 0.30,
            "gaze should stay faintly present: {}",
            dark.pre_dwell
        );
        p.nodes[0].activation = 1;
        let ctx2 = SignalCtx {
            puzzle: Some(&p),
            notes_read: 0,
            shards: 0,
            clarity: 1.0,
            harmony_hz: 220.0,
        };
        let lit = evaluate_field(field, [4.0, 4.0, 1.0], None, &ctx2, 0);
        assert!(
            lit.pre_dwell > dark.pre_dwell,
            "lit {} should exceed dark {}",
            lit.pre_dwell,
            dark.pre_dwell
        );
    }

    // ── `evaluate_mix` (the pub, ECS-free entry point) ──────────────────────

    #[test]
    fn evaluate_mix_reaches_a_non_neutral_steady_state_and_is_deterministic() {
        // A field parked right on the listener, driven `const` (always on) so
        // the result does not depend on puzzle state — isolates envelope
        // convergence and determinism.
        let lvl = load_field_level(
            "[field wash]\nkind = lethe_wash\nat = 1,1,1\nradius = 6\ndrive = const\nbase = 1.0\n",
        );
        let fs = re_audio::FieldSettings {
            intensity: 1.0,
            reduced: false,
        };
        let mut runtime = FieldRuntime::default();
        let mut mix = FieldMix::NEUTRAL;
        // Step several frames at a plausible dt so the attack envelope has
        // time to converge toward the driven target, exactly as a scripted
        // walk (the offline WAV verification's use case) would.
        for _ in 0..120 {
            mix = evaluate_mix(
                lvl.fields(),
                [1.0, 1.0, 1.0],
                None,
                None,
                0,
                0,
                1.0,
                220.0,
                0,
                None,
                0.0,
                1.0 / 60.0,
                &mut runtime,
                &fs,
            );
        }
        assert_ne!(
            mix,
            FieldMix::NEUTRAL,
            "a fully-driven field must move the mix off neutral"
        );

        // Determinism: same inputs, fresh runtime, same result.
        let mut runtime2 = FieldRuntime::default();
        let mut mix2 = FieldMix::NEUTRAL;
        for _ in 0..120 {
            mix2 = evaluate_mix(
                lvl.fields(),
                [1.0, 1.0, 1.0],
                None,
                None,
                0,
                0,
                1.0,
                220.0,
                0,
                None,
                0.0,
                1.0 / 60.0,
                &mut runtime2,
                &fs,
            );
        }
        assert_eq!(
            mix, mix2,
            "evaluate_mix must be deterministic given the same walk"
        );
    }

    #[test]
    fn invert_flips_the_bind_gate() {
        let lvl = load_field_level(
            "[node face]\npos = 1 1 1\n\
             [field wash]\nkind = lethe_wash\nat = 1,1,1\nradius = 6\nbind = fed:face\ninvert = yes\n",
        );
        let field = &lvl.fields()[0];
        let p = PuzzleState::from_level(&lvl).unwrap();
        // No pipe feeds `face`, so `fed:face` is 0 -> inverted gate is OPEN.
        let ctx = SignalCtx {
            puzzle: Some(&p),
            notes_read: 0,
            shards: 0,
            clarity: 1.0,
            harmony_hz: 220.0,
        };
        let eval = evaluate_field(field, [1.0, 1.0, 1.0], None, &ctx, 0);
        assert!(eval.pre_dwell > 0.0, "{}", eval.pre_dwell);
    }

    // ── F4 mirror voice (`false_echo_mirror`) ───────────────────────────────

    #[test]
    fn mirror_reflects_the_listener_across_the_wall_plane_with_the_speed_of_sound_delay() {
        // A wall_decal-style mirror flush against x=0, facing +x into the room.
        let lvl = load_field_level(
            "[field mirror_left]\nkind = false_echo\nregion = (0,1,1)-(0,6,4)\nradius = 6\nfacing = +x\ncone = 70\n",
        );
        let listener = [3.0, 3.0, 2.0];
        let (pos, delay) =
            false_echo_mirror(lvl.fields(), listener).expect("expected a mirror voice");
        assert!((pos[0] - -3.0).abs() < 1e-4, "{pos:?}"); // reflected across x=0
        assert!(
            (pos[1] - 3.0).abs() < 1e-4 && (pos[2] - 2.0).abs() < 1e-4,
            "{pos:?}"
        );
        // d_plane = 3 -> delay = 2*3*1.2/343 + 0.167
        let expected = 2.0 * 3.0 * MIRROR_DISTANCE_SCALE / SPEED_OF_SOUND_M_S + MIRROR_SIXTEENTH_S;
        assert!((delay - expected).abs() < 1e-5, "{delay} vs {expected}");
    }

    #[test]
    fn memory_variant_has_no_mirror_voice() {
        let lvl = load_field_level(
            "[field mirror_left]\nkind = false_echo\nregion = (0,1,1)-(0,6,4)\nradius = 6\nfacing = +x\ncone = 70\nvariant = memory\n",
        );
        assert!(false_echo_mirror(lvl.fields(), [3.0, 3.0, 2.0]).is_none());
    }

    #[test]
    fn a_field_with_no_facing_never_produces_a_mirror_voice() {
        // false_echo's archetype-default cone only applies once `facing` is
        // authored (re_level::fields' documented rule) — an author who omits
        // it gets no mirror voice rather than a silently-guessed direction.
        let lvl = load_field_level("[field river]\nkind = false_echo\nat = 3,3,2\nradius = 6\n");
        assert!(false_echo_mirror(lvl.fields(), [4.0, 4.0, 2.0]).is_none());
    }

    #[test]
    fn outside_the_radius_or_outside_the_cone_produces_no_mirror_voice() {
        let lvl = load_field_level(
            "[field mirror_left]\nkind = false_echo\nregion = (0,1,1)-(0,6,4)\nradius = 6\nfacing = +x\ncone = 30\n",
        );
        // far outside radius
        assert!(false_echo_mirror(lvl.fields(), [50.0, 3.0, 2.0]).is_none());
        // inside radius (distance to the box ~1.3 cells) but at ~42 degrees
        // off the +x facing normal, outside the 30-degree cone.
        assert!(false_echo_mirror(lvl.fields(), [1.0, 6.9, 2.0]).is_none());
    }

    #[test]
    fn standing_behind_or_on_the_mirror_plane_produces_no_mirror_voice() {
        let lvl = load_field_level(
            "[field mirror_left]\nkind = false_echo\nregion = (0,1,1)-(0,6,4)\nradius = 6\nfacing = +x\ncone = 70\n",
        );
        // Behind the wall (x < 0): excluded by the cone gate first (facing
        // points away from the wall, so a behind-the-plane point is >90 deg
        // off-axis — the same outcome the d_plane<=0 guard exists for, but
        // reached through the cone check for any cone <= 90 deg).
        assert!(false_echo_mirror(lvl.fields(), [-2.0, 3.0, 2.0]).is_none());
        // Standing exactly on the plane: d_plane == 0, no reflection distance.
        assert!(false_echo_mirror(lvl.fields(), [0.0, 3.0, 2.0]).is_none());
    }

    #[test]
    fn phantom_hollow_and_live_wire_aux_carry_the_level_harmony_root() {
        let lvl = load_field_level(
            "[node dormant]\npos = 4 4 1\n\
             [field ghost]\nkind = phantom_hollow\nat = 4,4,1\nradius = 6\nphase = 1 1 1 1\n",
        );
        let field = &lvl.fields()[0];
        let ctx = SignalCtx {
            puzzle: None,
            notes_read: 0,
            shards: 0,
            clarity: 1.0,
            harmony_hz: 164.81,
        };
        let eval = evaluate_field(field, [4.0, 4.0, 1.0], None, &ctx, 0);
        assert!((eval.aux - 164.81).abs() < 1e-4, "{}", eval.aux);
    }

    // ── `SoundFieldsPlugin<S>`: the state seam, end-to-end over real ECS ────

    #[derive(States, Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    enum TestState {
        #[default]
        Idle,
        Live,
    }

    /// A minimal headless app: the real [`SoundFieldsPlugin`] wired to a local
    /// host state (no host game type in sight), a spawned `PlayCamera`, and a
    /// level with one always-on (`drive = const`) field. No `FieldContext` is
    /// inserted — exercises the "no campaign level" default path alongside the
    /// state gate.
    fn seam_test_app() -> App {
        let mut app = App::new();
        app.add_plugins(bevy::state::app::StatesPlugin);
        app.init_state::<TestState>();
        app.init_resource::<Time>();
        app.insert_resource(SoundFieldMix(FieldMix::NEUTRAL));
        app.insert_resource(FieldSettingsRes(re_audio::FieldSettings {
            intensity: 1.0,
            reduced: false,
        }));
        app.add_plugins(SoundFieldsPlugin {
            live: TestState::Live,
        });

        let lvl = load_field_level(
            "[field wash]\nkind = lethe_wash\nat = 1,1,1\nradius = 6\ndrive = const\nbase = 1.0\n",
        );
        app.insert_resource(CurrentLevelData(Some(lvl)));
        app.world_mut().spawn((
            CameraPose {
                pos: re_core::Vec3::new(1.0, 1.0, 1.0),
                yaw: 0.0,
                pitch: 0.0,
            },
            PlayCamera,
        ));
        app
    }

    fn step(app: &mut App, dt_secs: f32) {
        let mut time = app.world_mut().resource_mut::<Time>();
        time.advance_by(std::time::Duration::from_secs_f32(dt_secs));
        app.update();
    }

    #[test]
    fn plugin_only_evaluates_while_the_host_state_matches_live() {
        let mut app = seam_test_app();

        // Idle (the default): the gated system must never run, so the mix the
        // test seeded stays byte-identical.
        for _ in 0..5 {
            step(&mut app, 1.0 / 60.0);
        }
        assert_eq!(
            app.world().resource::<SoundFieldMix>().0,
            FieldMix::NEUTRAL,
            "must not evaluate outside the `live` state"
        );

        // Enter Live: the same const-driven field now moves the mix off neutral.
        app.world_mut()
            .resource_mut::<NextState<TestState>>()
            .set(TestState::Live);
        for _ in 0..120 {
            step(&mut app, 1.0 / 60.0);
        }
        assert_ne!(
            app.world().resource::<SoundFieldMix>().0,
            FieldMix::NEUTRAL,
            "must evaluate once the host state matches `live`"
        );
    }
}
