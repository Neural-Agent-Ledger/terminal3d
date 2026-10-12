//! SOUND-FIELD AUTHORING — the `EntityKind::Field` support module.
//!
//! `re_level::fields` already owns parsing/defaults/validation/round-trip for
//! BOTH authoring syntaxes (standalone `[field NAME]` and the inline
//! `field =` attach on `[node]`/`[pipe]`/`[door]`) — see that module's doc
//! comment. This module is the EDITOR's half: it lets the author place, move,
//! and edit a **standalone** field with the cursor/keyboard/mouse instead of
//! hand-writing the `.lvl` text, plus the visualisation, lint surfacing, and
//! the opt-in live audition.
//!
//! ## Scope note (deliberate, reported honestly)
//!
//! `EntityKind::Field` (this pass) authors:
//! * the **Sphere** shape (`at`, exactly like a Node's position) and the
//!   **Room** shape (no position) — the `FieldShapeKind` row cycles between
//!   them;
//! * `radius`/`inner`/`falloff`/`occlude`, `facing`/`cone`, `bind`, `invert`,
//!   `drive`, `base` ("intensity"), `depth`, `priority`.
//!
//! It does **not** yet author the **Box** (`region`) or **Capsule**
//! (`follow`) standalone shapes, `phase`/`dwell`/`idle_boost`/`buses`/
//! `attack`/`release`/`variant`/`seed`, or a dedicated row on `[node]`/
//! `[pipe]`/`[door]` for their inline `field =` attach — those stay
//! hand-authored in the `.lvl` text. None of this is DATA LOSS: every
//! [`re_level::fields::FieldKey`] this editor does not carry a row for is
//! preserved **verbatim** (`FieldDef::keys` keeps every authored key, in
//! order — the same "verbatim keys" contract `re_level::props::PropDef`
//! already uses), so opening, editing an unrelated field, and saving a level
//! that has a hand-authored Box field or an inline attach never rewrites or
//! drops it. An inline-attached field (`def.host != FieldHost::Standalone`)
//! is simply not offered by the `EntityKind::Field` placement/selection
//! surface (`field_at` only matches `FieldHost::Standalone`), so it is
//! untouched by construction, not merely by care.
//!
//! ## Why `state.level.field_defs` is edited DIRECTLY
//!
//! Every other `EntityKind` in this editor (`Node`, `Trigger`, …) already
//! edits `EditorState::level` in place — there is no separate editor-side
//! mirror synced at save time. Fields follow the same shape: a placed field
//! is a `re_level::fields::FieldDef` pushed straight onto `level.field_defs`,
//! so `re_voximport::to_lvl_text`'s already-tested field writer serializes it
//! with **zero** editor-side writer code, and undo/redo (`history::Snapshot`,
//! which already clones the whole `(Map, Level)`) covers field edits for
//! free.
//!
//! The one piece `re_level` does NOT keep live is `Level::resolved_fields`
//! (the names-resolved, defaults-applied cache `Level::fields()` reads) — it
//! is populated once by `finalize()` at PARSE time. Every mutation to
//! `field_defs` in this module therefore ends with [`re_resolve`], which
//! calls the already-`pub` `re_level::fields::resolve` and mirrors exactly
//! what `finalize` does, so `Level::fields()`, the `fields` lvlcheck rule
//! (`io::save_lvl_text` → `re_lvlcheck::check_level` → `fields_rule::
//! check_fields` → `level.fields()`), and this module's own visualisation/
//! audition all see the edit immediately — never a stale cache.
//!
//! ## Cycle safety
//!
//! `FieldBind`/`FieldDrive`/`FieldFacing` are cycled through `crate::vocab` —
//! the SAME session-sticky "remember, then step" mechanism every other
//! cycling row in this editor uses (`entity::step_node_glyph`, `vocab.rs`'s
//! module doc), seeded from the level the moment it opens
//! (`Vocab::harvest`). A hand-authored value outside the canonical list
//! (`ring:...`, a `budget:`/`seq:`/`read:` target, a non-cardinal facing
//! vector, a Box/Capsule `FieldShapeKind`) is never silently destroyed on the
//! first `c` press AND stays reachable on every later press: a transient
//! "splice only the live current" (this module's [`safe_options`], still
//! used for the fully-closed `FieldArchetype`/`FieldFalloff` domains where no
//! off-list value can exist post-resolve) is not enough on its own — it
//! moves the loss one keypress later instead of removing it, because the
//! next press recomputes the option list from whatever `c` just landed on
//! and the original is gone. `crate::vocab`'s sticky `extras` list is the
//! general fix for that class of bug (cycling an enum row must never replace
//! a hand-authored, off-list value with the list's first entry) — applied
//! here, not reinvented.
use re_level::fields::{
    DriveSpec, Falloff, FieldBind, FieldDef, FieldHost, FieldKey, FieldRadius, FieldShape,
    ResolvedField, FIELD_ARCHETYPES,
};
use re_level::Level;
use re_world::Map;

use crate::state::{EditorState, EntityField};

// ───────────────────────────── key upsert helpers ──────────────────────────

/// The verbatim-authored value of `key` on `def`, if present. Most readers in
/// this module go through the RESOLVED (defaults-applied) value instead, but
/// [`cycle_field_enum`]'s `FieldArchetype` arm needs this raw form to detect
/// "no explicit `radius =` authored" before swapping archetypes (see its
/// doc), and the round-trip test below exercises it directly.
#[must_use]
pub fn get_key<'a>(def: &'a FieldDef, key: &str) -> Option<&'a str> {
    def.keys
        .iter()
        .find(|k| k.key == key)
        .map(|k| k.value.as_str())
}

/// Set (or append, preserving every other key's order) `key = value` on
/// `def`. `line` is always `0` for an editor-written key (there is no source
/// line); every reader treats `line: 0` as "not from disk", exactly like
/// `re_voximport`'s own editor-synthesized entries.
pub fn set_key(def: &mut FieldDef, key: &str, value: String) {
    if let Some(k) = def.keys.iter_mut().find(|k| k.key == key) {
        k.value = value;
    } else {
        def.keys.push(FieldKey {
            key: key.to_string(),
            value,
            line: 0,
        });
    }
}

/// Drop `key` from `def` entirely (used when switching shape kind — e.g. a
/// Room field carries no `at`).
pub fn remove_key(def: &mut FieldDef, key: &str) {
    def.keys.retain(|k| k.key != key);
}

// ─────────────────────────────── re-resolve ────────────────────────────────

/// Re-run `re_level::fields::resolve` and refresh `level.resolved_fields`
/// after any `field_defs` mutation — see the module doc's "why edited
/// directly" section. On a resolve error the cache is CLEARED (not left
/// stale): a stale cache of a different length than `field_defs` could
/// misalign `field_defs[i]` against `resolved_fields[i]` after an insert/delete,
/// so "no visualisation until fixed" is the safe failure mode, and the error
/// is surfaced immediately in the status line (not just deferred to the next
/// save's lvlcheck pass).
pub fn re_resolve(state: &mut EditorState) {
    match re_level::fields::resolve(&state.level) {
        Ok(resolved) => state.level.resolved_fields = resolved,
        Err((line, msg)) => {
            state.level.resolved_fields.clear();
            state.status = format!("field error (line {line}): {msg}");
        }
    }
}

/// A fresh, uniquely-named `field0`/`field1`/… — mirrors `entity.rs`'s
/// `unique_part_name`.
#[must_use]
pub fn unique_field_name(state: &EditorState) -> String {
    let mut n = 0usize;
    loop {
        let candidate = format!("field{n}");
        if !state.level.field_defs.iter().any(|d| d.name == candidate) {
            return candidate;
        }
        n += 1;
    }
}

/// The index of a STANDALONE field whose resolved shape is a Sphere at the
/// integer cell `(x, y, z)`, if any — used by `place_at_cursor`/
/// `delete_at_cursor` to SELECT rather than duplicate, exactly like
/// `entity::node_at`. An inline-attached field (`host != Standalone`) is
/// never matched, by design (see the module doc).
#[must_use]
pub fn field_at(state: &EditorState, x: i32, y: i32, z: i32) -> Option<usize> {
    for (i, def) in state.level.field_defs.iter().enumerate() {
        if def.host != FieldHost::Standalone {
            continue;
        }
        if let Some(resolved) = state.level.resolved_fields.get(i) {
            if let FieldShape::Sphere { at } = resolved.shape {
                let cell = (
                    at[0].round() as i32,
                    at[1].round() as i32,
                    at[2].round() as i32,
                );
                if cell == (x, y, z) {
                    return Some(i);
                }
            }
        }
    }
    None
}

// ──────────────────────────── option vocabularies ──────────────────────────

/// Splice the CURRENT authored value into `canonical` if it is not already
/// present, so cycling through the result can NEVER silently destroy a
/// hand-authored off-list value (module doc's "cycle safety" section).
#[must_use]
pub fn safe_options(mut canonical: Vec<String>, current: &str) -> Vec<String> {
    if !current.is_empty() && !canonical.iter().any(|s| s == current) {
        canonical.push(current.to_string());
    }
    canonical
}

/// The `bind =` picker's canonical targets, by node/pipe/door NAME: `none`,
/// `door`, every node as both `node:NAME` and `fed:NAME`, and every pipe as
/// `pipe:A->B`. Free-form
/// `budget:K`/`seq:K`/`read:I` targets are not offered by the picker (out of
/// the literal node/pipe/door scope) but a hand-authored one round-trips
/// untouched (never destroyed — see `safe_options`, applied by the caller).
#[must_use]
pub fn bind_options(level: &Level) -> Vec<String> {
    let mut v = vec!["none".to_string(), "door".to_string()];
    for n in &level.nodes {
        v.push(format!("node:{}", n.name));
    }
    for n in &level.nodes {
        v.push(format!("fed:{}", n.name));
    }
    for p in &level.pipes {
        v.push(format!("pipe:{}->{}", p.from, p.to));
    }
    v
}

/// The `drive =` picker's canonical targets: the 8 state signals plus
/// `lit:NAME`/`fed:NAME`/`starve:NAME` per node and `live:A->B`/
/// `dead_draw:A->B` per pipe. `ring:...` (a multi-pipe list) is not offered
/// by the picker; a hand-authored one round-trips untouched (see
/// `bind_options`'s doc).
#[must_use]
pub fn drive_options(level: &Level) -> Vec<String> {
    let mut v = vec![
        "const".to_string(),
        "door_ready".to_string(),
        "solved".to_string(),
        "cycle_held".to_string(),
        "budget".to_string(),
        "seq".to_string(),
        "clarity".to_string(),
        "shards".to_string(),
    ];
    for n in &level.nodes {
        v.push(format!("lit:{}", n.name));
    }
    for n in &level.nodes {
        v.push(format!("fed:{}", n.name));
    }
    for n in &level.nodes {
        v.push(format!("starve:{}", n.name));
    }
    for p in &level.pipes {
        v.push(format!("live:{}->{}", p.from, p.to));
    }
    for p in &level.pipes {
        v.push(format!("dead_draw:{}->{}", p.from, p.to));
    }
    v
}

/// `falloff =` cycle order (fixed four-word grammar — no destruction risk,
/// every value the parser accepts is on this list).
pub const FALLOFF_OPTIONS: [&str; 4] = ["smooth", "linear", "inverse", "hard"];

/// `facing =` cardinal cycle: `none` (no cone) plus the six axis directions.
/// A hand-authored NON-cardinal vector is preserved via `safe_options` at the
/// call site (it never parses back to one of these seven labels, so it is
/// always spliced in as an eighth entry rather than lost).
pub const FACING_OPTIONS: [&str; 7] = ["none", "+x", "-x", "+y", "-y", "+z", "-z"];

// ─────────────────────── archetype family colours ──────────────────────────

/// One family tint per archetype: LETHE cold blue, MNEMOS amber, decay
/// sickly green, gaze violet, warmth orange, everything else grey.
#[must_use]
pub fn family_color(archetype: &str) -> u32 {
    match archetype {
        "lethe_wash" => 0x00_4C8CFF,   // LETHE: cold blue
        "mnemos_hold" => 0x00_D9A441,  // MNEMOS: amber
        "bit_rot" => 0x00_6FCB6F,      // decay: sickly green
        "watcher_gaze" => 0x00_9B6FE0, // gaze: violet
        "steam_warmth" => 0x00_E08A3C, // warmth: orange
        _ => 0x00_9AA4B2,              // other: grey
    }
}

// ───────────────────────────── pure geometry/DSP ───────────────────────────

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn len3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Distance from a point to the nearest point of a line segment `a..b`.
fn dist_point_segment(p: [f32; 3], a: [f32; 3], b: [f32; 3]) -> f32 {
    let ab = sub3(b, a);
    let len_sq = dot3(ab, ab);
    if len_sq < 1e-9 {
        return len3(sub3(p, a));
    }
    let t = (dot3(sub3(p, a), ab) / len_sq).clamp(0.0, 1.0);
    let closest = [a[0] + ab[0] * t, a[1] + ab[1] * t, a[2] + ab[2] * t];
    len3(sub3(p, closest))
}

/// The field's own reference point — its centre for a Sphere, the box
/// centroid for a Box, the first waypoint for a Capsule, or the listener
/// itself for a Room (mirrors `re_game::sound_fields::evaluate_mix`'s
/// `bearing` computation, so a field reads the same "where is it" both at
/// runtime and in this editor).
#[must_use]
pub fn shape_reference_point(shape: &FieldShape, listener: [f32; 3]) -> [f32; 3] {
    match shape {
        FieldShape::Sphere { at } => *at,
        FieldShape::BoxShape { min, max } => [
            (min[0] + max[0]) * 0.5,
            (min[1] + max[1]) * 0.5,
            (min[2] + max[2]) * 0.5,
        ],
        FieldShape::Capsule { points } => points.first().copied().unwrap_or(listener),
        FieldShape::Room => listener,
    }
}

/// World-space distance from `p` to a field's shape (`0.0` while inside a Box
/// or for the shapeless Room, which is always in range).
#[must_use]
pub fn distance_to_shape(shape: &FieldShape, p: [f32; 3]) -> f32 {
    match shape {
        FieldShape::Sphere { at } => len3(sub3(p, *at)),
        FieldShape::BoxShape { min, max } => {
            let dx = (min[0] - p[0]).max(0.0).max(p[0] - max[0]);
            let dy = (min[1] - p[1]).max(0.0).max(p[1] - max[1]);
            let dz = (min[2] - p[2]).max(0.0).max(p[2] - max[2]);
            (dx * dx + dy * dy + dz * dz).sqrt()
        }
        FieldShape::Capsule { points } => {
            if points.is_empty() {
                return f32::MAX;
            }
            if points.len() == 1 {
                return len3(sub3(p, points[0]));
            }
            points
                .windows(2)
                .map(|w| dist_point_segment(p, w[0], w[1]))
                .fold(f32::MAX, f32::min)
        }
        FieldShape::Room => 0.0,
    }
}

/// Half the visualisation/falloff "hard edge" band width, in cells: the
/// `Hard` falloff cuts over a `0.5`-cell-wide ramp centred on `radius`.
const HARD_EDGE_HALF_WIDTH: f32 = 0.25;

/// The spatial `[0, 1]` presence factor at distance `d` from a field of the
/// given `falloff`/`radius`/`inner` — pure, no live puzzle state. `Smooth`
/// and `Linear` fall off over the `inner..radius` band (1 inside `inner`, 0
/// at/past `radius`); `Inverse` uses `1 / (1 + (d/radius)^2)` (`0.5` exactly
/// at `d == radius`, continuing to fade slowly beyond it — the "the exit
/// grammar keeps a whisper of pull past its nominal radius" reading `exit_
/// detune` needs); `Hard` is a crisp cutoff with a `0.5`-cell-wide ramp
/// centred on `radius`.
#[must_use]
pub fn falloff_weight(falloff: Falloff, radius: f32, inner: f32, d: f32) -> f32 {
    if !(radius.is_finite() && radius > 0.0) {
        return 0.0;
    }
    match falloff {
        Falloff::Inverse => {
            let t = (d / radius).max(0.0);
            1.0 / (1.0 + t * t)
        }
        Falloff::Hard => {
            if d <= radius - HARD_EDGE_HALF_WIDTH {
                1.0
            } else if d >= radius + HARD_EDGE_HALF_WIDTH {
                0.0
            } else {
                ((radius + HARD_EDGE_HALF_WIDTH) - d) / (2.0 * HARD_EDGE_HALF_WIDTH)
            }
        }
        Falloff::Smooth | Falloff::Linear => {
            let span = (radius - inner).max(f32::EPSILON);
            let t = ((d - inner) / span).clamp(0.0, 1.0);
            let shaped = if falloff == Falloff::Smooth {
                t * t * (3.0 - 2.0 * t)
            } else {
                t
            };
            1.0 - shaped
        }
    }
}

/// `1.0` inside the cone (or when the field carries no cone at all), `0.0`
/// outside it — a hard cutoff, matching the 2D view aid's "cones draw as a
/// wedge" (a wedge has an edge, not a gradient). At the field's own
/// reference point (`from == to`, direction undefined) it reads `1.0` — a
/// field is never inaudible from inside itself.
#[must_use]
pub fn cone_factor(cone: Option<([f32; 3], f32)>, from: [f32; 3], to: [f32; 3]) -> f32 {
    let Some((dir, half_deg)) = cone else {
        return 1.0;
    };
    let v = sub3(to, from);
    let vlen = len3(v);
    if vlen < 1e-6 {
        return 1.0;
    }
    let vn = [v[0] / vlen, v[1] / vlen, v[2] / vlen];
    let dot = dot3(vn, dir).clamp(-1.0, 1.0);
    let angle_deg = dot.acos().to_degrees();
    if angle_deg <= half_deg {
        1.0
    } else {
        0.0
    }
}

/// The editor's `evaluate_mix`-EQUIVALENT (scoped honestly — see the module
/// doc): the PREVIEW weight a field carries when heard from `listener`.
///
/// `re_game::sound_fields::evaluate_mix` (the real per-frame runtime
/// evaluation) additionally reads live puzzle state (which node is lit, which
/// pipe is drawing power, the door's win condition, …) via a running
/// `re_puzzle::PuzzleState` — this editor has no such simulation (adding one
/// is out of this pass's scope; see `ISSUES`). So:
/// * the SPATIAL half — shape distance → falloff → cone → `depth` — is REAL,
///   the same maths the runtime uses, and is what makes moving the cursor/
///   camera through a field audibly/visibly change its weight;
/// * the STATE half is a fixed preview: `drive = const` uses the authored
///   `base` (a real, author-controlled number); every state-driven `drive`
///   (`lit:`/`fed:`/`starve:`/`door_ready`/…) previews at
///   [`PREVIEW_DRIVE_FALLBACK`] — a representative "on" level, not a claim
///   about the puzzle's actual state — so the archetype's character is
///   audible/visible without asserting a game state the editor cannot know.
pub const PREVIEW_DRIVE_FALLBACK: f32 = 0.65;

#[must_use]
fn static_drive_factor(field: &ResolvedField) -> f32 {
    match field.drive {
        DriveSpec::Const => field.base,
        _ => PREVIEW_DRIVE_FALLBACK,
    }
}

/// The full `[0, 1]` preview weight used by the audition and cursor readout
/// — see [`static_drive_factor`]'s doc for the honest state-half scope note.
#[must_use]
pub fn evaluate_weight(field: &ResolvedField, listener: [f32; 3]) -> f32 {
    let drive = static_drive_factor(field);
    let FieldRadius::Value(radius) = field.radius else {
        // `radius = room`: always "in range" — room fields are additive
        // on top of the top-3, not falloff-gated.
        return (drive * field.depth).clamp(0.0, 1.0);
    };
    let center = shape_reference_point(&field.shape, listener);
    let d = distance_to_shape(&field.shape, listener);
    let spatial = falloff_weight(field.falloff, radius, field.inner, d);
    let cone = cone_factor(field.cone, center, listener);
    (spatial * cone * drive * field.depth).clamp(0.0, 1.0)
}

// ─────────────────────────── the audible mix + readout ─────────────────────

/// Build the real `re_audio::field::FieldMix` every resolved field of `level`
/// produces when heard from `listener` — the SAME `combine()` the runtime
/// calls, fed [`evaluate_weight`]'s preview weights. Used by both the cursor
/// readout and the live audition's per-frame tick.
#[must_use]
pub fn build_mix(level: &Level, listener: [f32; 3]) -> re_audio::FieldMix {
    let inputs: Vec<re_audio::FieldInput> = level
        .fields()
        .iter()
        .filter_map(|f| {
            let archetype = re_audio::Archetype::from_name(&f.archetype)?;
            Some(re_audio::FieldInput {
                archetype,
                weight: evaluate_weight(f, listener),
                depth: f.depth,
                priority: f.priority,
                aux: 0.0,
                variant: 0,
                seed: f.seed,
                bearing: Some(shape_reference_point(&f.shape, listener)),
            })
        })
        .collect();
    re_audio::combine(&inputs, &re_audio::FieldSettings::default())
}

/// The CURSOR READOUT: the top-3 fields by `priority × weight` at
/// `listener`, plus the resulting `AMB LP / WD / VOL` from the real
/// `combine()` mix. Plain text, exactly reproducible from `(level,
/// listener)` alone — unit-testable with no ECS/terminal.
#[must_use]
pub fn cursor_readout(level: &Level, listener: [f32; 3]) -> String {
    let mut scored: Vec<(&ResolvedField, f32)> = level
        .fields()
        .iter()
        .map(|f| (f, evaluate_weight(f, listener)))
        .filter(|(_, w)| *w > 0.005)
        .collect();
    if scored.is_empty() {
        return "fields: none audible here".to_string();
    }
    scored.sort_by(|a, b| {
        let sa = a.1 * f32::from(a.0.priority);
        let sb = b.1 * f32::from(b.0.priority);
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
    });
    let top: Vec<String> = scored
        .iter()
        .take(3)
        .map(|(f, w)| format!("{} {w:.2}", f.archetype))
        .collect();
    let mix = build_mix(level, listener);
    let amb = mix.buses[re_audio::field::AMB];
    format!(
        "fields: {}  AMB LP {:.0}Hz WD {:.2} VOL {:.1}dB",
        top.join(" | "),
        amb.lp_cutoff_hz,
        amb.width,
        amb.vol_duck_db,
    )
}

// ─────────────────────────── 2D view aid ───────────────────────────────────

/// What a cell shows under the FIELDS view aid, for ONE field — before the
/// caller picks the strongest across every field covering the cell (see
/// `chrome`'s painter hook). `None` when the cell is outside the field's
/// footprint entirely (including when `facing`/`cone` excludes it).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CellTint {
    /// Inside `radius`, away from either ring — the family wash.
    Interior,
    /// Within half a cell of `inner` (`inner > 0`) — the brighter inner ring.
    InnerRing,
    /// Within half a cell of `radius` — the outer ring outline.
    OuterRing,
}

/// Classify cell centre `p` against ONE resolved field's footprint on the
/// current slice. `Room`-shaped fields have no footprint to ring-paint (they
/// are everywhere at once) and are skipped by design — `None` always.
#[must_use]
pub fn footprint_at(field: &ResolvedField, p: [f32; 3]) -> Option<CellTint> {
    let FieldRadius::Value(radius) = field.radius else {
        return None;
    };
    let center = shape_reference_point(&field.shape, p);
    if cone_factor(field.cone, center, p) <= 0.0 {
        return None;
    }
    let d = distance_to_shape(&field.shape, p);
    if d > radius + HARD_EDGE_HALF_WIDTH {
        return None;
    }
    if (d - radius).abs() <= HARD_EDGE_HALF_WIDTH {
        return Some(CellTint::OuterRing);
    }
    if field.inner > 0.0 && (d - field.inner).abs() <= HARD_EDGE_HALF_WIDTH {
        return Some(CellTint::InnerRing);
    }
    if d <= radius {
        Some(CellTint::Interior)
    } else {
        None
    }
}

/// Per-channel scale of an `0x00RRGGBB` colour (mirrors
/// `viewaids::scale`— kept as a private copy rather than made `pub` there,
/// since the two modules must never couple on an internal helper).
const fn scale_rgb(color: u32, num: u32, den: u32) -> u32 {
    let r = ((color >> 16) & 0xFF) * num / den;
    let g = ((color >> 8) & 0xFF) * num / den;
    let b = (color & 0xFF) * num / den;
    (r << 16) | (g << 8) | b
}

/// A coarse line-of-sight check on the CURRENT z-slice only (the aid is a 2D
/// plan-view painter): `true` when a solid cell lies strictly between `from`
/// and `to` (exclusive of the endpoints), sampled every half-cell. Used for
/// the doc's "occluded cells drawn at half tint" rule. A cheap approximation
/// (no 3D raycast, no portal-awareness) — sufficient for a plan-view aid; the
/// real per-frame audio occlusion is `re_game::sound_fields`'s job.
#[must_use]
pub fn occluded_2d(map: &Map, from: (f32, f32), to: (f32, f32), z: i32) -> bool {
    let dx = to.0 - from.0;
    let dy = to.1 - from.1;
    let dist = (dx * dx + dy * dy).sqrt();
    if dist < 1e-3 {
        return false;
    }
    let steps = (dist / 0.5).ceil().max(1.0) as i32;
    for i in 1..steps {
        let t = i as f32 / steps as f32;
        let x = (from.0 + dx * t).floor() as i32;
        let y = (from.1 + dy * t).floor() as i32;
        if map.cell_at(x, y, z).is_solid() {
            return true;
        }
    }
    false
}

/// The colour the FIELDS view aid paints cell `(x, y, z)`, or `None` when no
/// field's footprint covers it (or the aid is off). Picks the STRONGEST
/// classification across every field covering the cell (`OuterRing` >
/// `InnerRing` > `Interior`, so a ring is never hidden under another field's
/// interior wash), tinted by that field's archetype family, halved when
/// `occlude = yes` and a wall sits between the field and the cell.
#[must_use]
pub fn cell_tint(state: &EditorState, x: i32, y: i32, z: i32) -> Option<u32> {
    if !state.fields_aid {
        return None;
    }
    let p = [x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5];
    let mut best: Option<(u8, u32, &ResolvedField)> = None;
    for f in state.level.fields() {
        let Some(tint) = footprint_at(f, p) else {
            continue;
        };
        let rank = match tint {
            CellTint::OuterRing => 3,
            CellTint::InnerRing => 2,
            CellTint::Interior => 1,
        };
        if best.is_none_or(|(r, ..)| rank > r) {
            best = Some((rank, family_shade(tint, f), f));
        }
    }
    let (_, base, field) = best?;
    if field.occlude {
        let center = shape_reference_point(&field.shape, p);
        if occluded_2d(&state.map, (center[0], center[1]), (p[0], p[1]), z) {
            return Some(scale_rgb(base, 1, 2));
        }
    }
    Some(base)
}

/// The family colour at full/ring brightness for a [`CellTint`].
fn family_shade(tint: CellTint, field: &ResolvedField) -> u32 {
    let base = family_color(&field.archetype);
    match tint {
        CellTint::OuterRing => base,
        CellTint::InnerRing => {
            // Brighter than the base — lift toward white by a quarter.
            let lifted_r = ((base >> 16) & 0xFF).saturating_add(48).min(255);
            let lifted_g = ((base >> 8) & 0xFF).saturating_add(48).min(255);
            let lifted_b = (base & 0xFF).saturating_add(48).min(255);
            (lifted_r << 16) | (lifted_g << 8) | lifted_b
        }
        CellTint::Interior => scale_rgb(base, 1, 3),
    }
}

// ─────────────────────────── live audition ─────────────────────────────────

/// A `.lvl` name hashed to pick a deterministic preview [`re_audio::
/// AmbienceMood`] (splitmix64-style, matching `re_level::fields::hash_name`'s
/// family). The real per-level mood mapping lives in `re_game::flow`
/// (story-driven, level-index keyed) — out of `ts_editor`'s reach by the
/// crate-boundary rules (`re_game` is not a dependency here), so the
/// audition previews A mood deterministically rather than THE campaign mood.
/// Reported honestly, not silently approximated as "the" mood.
fn hash_level_name(name: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in name.as_bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

#[must_use]
fn mood_for_level(level: &Level) -> re_audio::AmbienceMood {
    let key = if level.name.is_empty() {
        "untitled"
    } else {
        level.name.as_str()
    };
    let h = hash_level_name(key);
    let moods = re_audio::AmbienceMood::ALL;
    moods[(h % moods.len() as u64) as usize]
}

/// The armed live audition (`Shift+U`): a real `re_audio::Player` (or,
/// off `TS_AUDIO_SILENT`/no device, `Player`'s own always-safe silent
/// degrade) playing the level's ambience + drone, with field weights
/// re-evaluated from the editor camera every tick.
///
/// `player` is wrapped in a [`std::sync::Mutex`] SOLELY to give `Audition` —
/// and therefore `EditorState`, which is a `#[derive(Resource)]` and so must
/// be `Sync` for Bevy's `ResMut<EditorState>` — a `Sync` marker. `re_audio::
/// Player` carries an `mpsc::Receiver` internally (`Send`, never `Sync`),
/// same as any other channel-backed device handle. Every access here is
/// still single-threaded and already holds `&mut self`, so `Mutex::get_mut`
/// (no actual lock — it borrows the inner value directly, since holding
/// `&mut Mutex<T>` already proves no other thread has it locked) is used
/// throughout: this is a type-system-only fix, not a concurrency mechanism.
pub struct Audition {
    player: std::sync::Mutex<re_audio::Player>,
    mood: re_audio::AmbienceMood,
}

impl Audition {
    /// Arms the audition. Returns `None` — WITHOUT constructing a
    /// [`re_audio::Player`] at all — when [`re_audio::playback::
    /// silence_requested`] is true (`TS_AUDIO_SILENT` set to a non-empty,
    /// non-`"0"` value), so "never open a device" holds a step earlier than
    /// `Player::open`'s own (independently safe) silent-degrade contract.
    /// This is what `TS_AUDIO_SILENT=1 ⇒ no Player created` means and what
    /// the gating test asserts.
    pub fn start(level: &Level) -> Result<Audition, String> {
        if re_audio::playback::silence_requested() {
            return Err("audition disabled: TS_AUDIO_SILENT is set".to_string());
        }
        let mood = mood_for_level(level);
        let mut player = re_audio::Player::open();
        player.set_ambience(re_audio::Ambience::new(mood, 1.0));
        Ok(Audition {
            player: std::sync::Mutex::new(player),
            mood,
        })
    }

    /// Re-evaluate every field from `listener` and push the resulting mix
    /// into the live player. Called once per frame while armed.
    pub fn tick(&mut self, level: &Level, listener: [f32; 3]) {
        let mix = build_mix(level, listener);
        self.player
            .get_mut()
            .expect("single-threaded: never poisoned")
            .set_fields(&mix);
    }

    #[must_use]
    pub fn mood_name(&self) -> &'static str {
        self.mood.name()
    }
}

/// `Shift+U`: arm/disarm the live audition. On arm failure (silent env), the
/// status explains why and `state.audition` stays `None` — never a half-armed
/// state.
pub fn toggle_audition(state: &mut EditorState) {
    if state.audition.take().is_some() {
        state.status = "audition off".to_string();
        return;
    }
    match Audition::start(&state.level) {
        Ok(a) => {
            state.status = format!("AUDITION ON \u{2014} {} (Shift+U to stop)", a.mood_name());
            state.audition = Some(a);
        }
        Err(msg) => state.status = msg,
    }
}

/// `F`: toggle the FIELDS 2D view aid. A view aid never edits the document
/// (mirrors `viewaids::cycle_onion`'s `g_cycles_onion` contract).
pub fn toggle_fields_aid(state: &mut EditorState) {
    state.fields_aid = !state.fields_aid;
    state.status = format!(
        "fields view aid: {}",
        if state.fields_aid { "on" } else { "off" }
    );
}

// ────────────────────── inspector rows: readout + edits ────────────────────

/// `falloff =`'s canonical string, matching one of [`FALLOFF_OPTIONS`].
fn falloff_to_str(f: Falloff) -> &'static str {
    match f {
        Falloff::Smooth => "smooth",
        Falloff::Linear => "linear",
        Falloff::Inverse => "inverse",
        Falloff::Hard => "hard",
    }
}

fn yesno(b: bool) -> &'static str {
    if b {
        "yes"
    } else {
        "no"
    }
}

/// `facing =`'s canonical string for a resolved cone: `"none"` when the field
/// carries no cone, one of the six [`FACING_OPTIONS`] cardinals when the
/// direction matches one exactly, or the literal `x,y,z` for a hand-authored
/// non-cardinal vector (never lost — this is what lets `safe_options` splice
/// it back in rather than silently snapping it to a cardinal).
pub(crate) fn facing_to_str(cone: Option<([f32; 3], f32)>) -> String {
    let Some((dir, _)) = cone else {
        return "none".to_string();
    };
    const EPS: f32 = 1e-3;
    const CARDINALS: [([f32; 3], &str); 6] = [
        ([1.0, 0.0, 0.0], "+x"),
        ([-1.0, 0.0, 0.0], "-x"),
        ([0.0, 1.0, 0.0], "+y"),
        ([0.0, -1.0, 0.0], "-y"),
        ([0.0, 0.0, 1.0], "+z"),
        ([0.0, 0.0, -1.0], "-z"),
    ];
    for (v, label) in CARDINALS {
        if (dir[0] - v[0]).abs() < EPS && (dir[1] - v[1]).abs() < EPS && (dir[2] - v[2]).abs() < EPS
        {
            return label.to_string();
        }
    }
    format!("{:.2},{:.2},{:.2}", dir[0], dir[1], dir[2])
}

fn name_or(level: &Level, i: usize, prefix: &str) -> String {
    level
        .nodes
        .get(i)
        .map_or_else(|| format!("{prefix}:?"), |n| format!("{prefix}:{}", n.name))
}

fn pipe_or(level: &Level, i: usize, prefix: &str) -> String {
    level.pipes.get(i).map_or_else(
        || format!("{prefix}:?"),
        |p| format!("{prefix}:{}->{}", p.from, p.to),
    )
}

/// `bind =`'s canonical string for a resolved [`FieldBind`], resolving a
/// node/pipe index back to its authored NAME (the picker's whole point — see
/// the module doc's "binding picker (targets by node/pipe/door NAME)"). Every
/// variant round-trips to a string, including the ones [`bind_options`]
/// doesn't offer on the picker (`budget:`/`seq:`), so a hand-authored bind is
/// always readable, never shown as `"-"`.
pub(crate) fn bind_to_str(bind: &FieldBind, level: &Level) -> String {
    match bind {
        FieldBind::None => "none".to_string(),
        FieldBind::Door => "door".to_string(),
        FieldBind::Node(i) => name_or(level, *i, "node"),
        FieldBind::Fed(i) => name_or(level, *i, "fed"),
        FieldBind::Pipe(i) => pipe_or(level, *i, "pipe"),
        FieldBind::Budget(k) => format!("budget:{k}"),
        FieldBind::Seq(k) => format!("seq:{k}"),
        FieldBind::Read(i) => format!("read:{i}"),
    }
}

/// `drive =`'s canonical string for a resolved [`DriveSpec`] — see
/// [`bind_to_str`]'s doc; the same name-resolution contract applies.
pub(crate) fn drive_to_str(drive: &DriveSpec, level: &Level) -> String {
    match drive {
        DriveSpec::Const => "const".to_string(),
        DriveSpec::Lit(i) => name_or(level, *i, "lit"),
        DriveSpec::Fed(i) => name_or(level, *i, "fed"),
        DriveSpec::Starve(i) => name_or(level, *i, "starve"),
        DriveSpec::Live(i) => pipe_or(level, *i, "live"),
        DriveSpec::DeadDraw(i) => pipe_or(level, *i, "dead_draw"),
        DriveSpec::Ring(idxs) => {
            let parts: Vec<String> = idxs
                .iter()
                .map(|&i| {
                    level
                        .pipes
                        .get(i)
                        .map_or_else(|| "?".to_string(), |p| format!("{}->{}", p.from, p.to))
                })
                .collect();
            format!("ring:{}", parts.join(","))
        }
        DriveSpec::DoorReady => "door_ready".to_string(),
        DriveSpec::Solved => "solved".to_string(),
        DriveSpec::CycleHeld => "cycle_held".to_string(),
        DriveSpec::Budget => "budget".to_string(),
        DriveSpec::Seq => "seq".to_string(),
        DriveSpec::Clarity => "clarity".to_string(),
        DriveSpec::Shards => "shards".to_string(),
    }
}

/// The selected field's [`ResolvedField`], or `None` when nothing is selected
/// or the level currently fails to resolve (see [`re_resolve`]'s doc — the
/// cache is cleared, not stale, on error).
fn selected_resolved(state: &EditorState) -> Option<&ResolvedField> {
    state.level.resolved_fields.get(state.selected_field?)
}

/// The inspector row text for a `EntityField::Field*` row — ALWAYS reads the
/// RESOLVED (defaults-applied) value, never the raw authored key, so a row
/// left unauthored on a fresh field still shows its true effective value
/// (e.g. `occlude` reading `"yes"` from the archetype default with no
/// authored `occlude =` key at all) rather than a misleading placeholder.
#[must_use]
pub fn value_of(state: &EditorState, field: EntityField) -> String {
    let Some(r) = selected_resolved(state) else {
        return "-".to_string();
    };
    match field {
        EntityField::FieldArchetype => r.archetype.clone(),
        EntityField::FieldShapeKind => match r.shape {
            FieldShape::Sphere { .. } => "sphere".to_string(),
            FieldShape::Room => "room".to_string(),
            FieldShape::BoxShape { .. } => "box".to_string(),
            FieldShape::Capsule { .. } => "capsule".to_string(),
        },
        EntityField::FieldPosX => match r.shape {
            FieldShape::Sphere { at } => (at[0].round() as i32).to_string(),
            _ => "-".to_string(),
        },
        EntityField::FieldPosY => match r.shape {
            FieldShape::Sphere { at } => (at[1].round() as i32).to_string(),
            _ => "-".to_string(),
        },
        EntityField::FieldPosZ => match r.shape {
            FieldShape::Sphere { at } => (at[2].round() as i32).to_string(),
            _ => "-".to_string(),
        },
        EntityField::FieldRadius => match r.radius {
            FieldRadius::Room => "room".to_string(),
            FieldRadius::Value(v) => format!("{v:.1}"),
        },
        EntityField::FieldInner => format!("{:.1}", r.inner),
        EntityField::FieldFalloff => falloff_to_str(r.falloff).to_string(),
        EntityField::FieldOcclude => yesno(r.occlude).to_string(),
        EntityField::FieldFacing => facing_to_str(r.cone),
        EntityField::FieldCone => r
            .cone
            .map_or_else(|| "-".to_string(), |(_, half)| format!("{half:.0}")),
        EntityField::FieldBind => bind_to_str(&r.bind, &state.level),
        EntityField::FieldInvert => yesno(r.invert).to_string(),
        EntityField::FieldDrive => drive_to_str(&r.drive, &state.level),
        EntityField::FieldBase => format!("{:.2}", r.base),
        EntityField::FieldDepth => format!("{:.2}", r.depth),
        EntityField::FieldPriority => r.priority.to_string(),
        _ => "-".to_string(),
    }
}

/// Step `current` to the next/prev entry of a [`safe_options`]-spliced list —
/// the one cycling primitive every enum row below uses, so a hand-authored
/// off-list value is always found at its spliced position (never silently
/// reset to index 0 — see the module doc's "cycle safety").
fn cycle_str(current: &str, canonical: &[&str], delta: i32) -> String {
    let opts = safe_options(
        canonical.iter().map(|s| (*s).to_string()).collect(),
        current,
    );
    let n = opts.len() as i32;
    let pos = opts.iter().position(|s| s == current).unwrap_or(0) as i32;
    opts[(pos + delta).rem_euclid(n) as usize].clone()
}

fn cycle_shape_kind(state: &mut EditorState, idx: usize) {
    // `at`/`region`/`follow` are POSITIONAL — supplied by the host on an
    // inline `field =` attach (`re_level::fields::is_known_inline_knob`
    // explicitly rejects `at` as an inline knob). Writing an `at =` here for
    // a non-Standalone field would desugar to `field = ARCHETYPE at=...`,
    // which `attach_inline`'s own parser then refuses to re-read — a
    // round-trip break. So shape-kind (like position) is Standalone-only;
    // an inline field's shape stays exactly what its host implies.
    if state.level.field_defs[idx].host != FieldHost::Standalone {
        state.status =
            "field shape is set by its host (inline attach) \u{2014} not editable here".to_string();
        return;
    }
    let Some(r) = state.level.resolved_fields.get(idx) else {
        state.status = "field: fix the reported error before editing".to_string();
        return;
    };
    if matches!(
        r.shape,
        FieldShape::BoxShape { .. } | FieldShape::Capsule { .. }
    ) {
        state.status =
            "field shape is box/capsule (hand-authored) \u{2014} edit the .lvl text to change it"
                .to_string();
        return;
    }
    let to_room = !matches!(r.shape, FieldShape::Room);
    let (x, y, z) = (state.cursor.0 as i32, state.cursor.1 as i32, state.z as i32);
    let def = &mut state.level.field_defs[idx];
    if to_room {
        remove_key(def, "at");
        set_key(def, "radius", "room".to_string());
    } else {
        set_key(def, "at", format!("{x}, {y}, {z}"));
        remove_key(def, "radius");
    }
    re_resolve(state);
    state.mark_dirty();
    state.status = format!("field shape: {}", if to_room { "room" } else { "sphere" });
}

fn cycle_facing(state: &mut EditorState, idx: usize) {
    let Some(r) = state.level.resolved_fields.get(idx) else {
        state.status = "field: fix the reported error before editing".to_string();
        return;
    };
    let current = facing_to_str(r.cone);
    // A non-cardinal facing vector is STICKY-remembered before stepping off
    // it (`crate::vocab`'s "remember, then step" contract — matches
    // `entity::step_node_glyph`), so a full lap of `c` returns to it exactly
    // like every other cycling row in this editor.
    let facing_canon: Vec<String> = FACING_OPTIONS.iter().map(|s| (*s).to_string()).collect();
    crate::vocab::remember(&facing_canon, &mut state.vocab.field_facing, &current);
    let next = crate::vocab::step(&facing_canon, &state.vocab.field_facing, &current, 1);
    let def = &mut state.level.field_defs[idx];
    if next == "none" {
        remove_key(def, "facing");
        remove_key(def, "cone");
    } else {
        set_key(def, "facing", next.clone());
    }
    re_resolve(state);
    state.mark_dirty();
    state.status = format!("field facing: {next}");
}

/// `c` on a `Field*` enum row (`FieldArchetype`/`FieldShapeKind`/`FieldFalloff`/
/// `FieldOcclude`/`FieldFacing`/`FieldBind`/`FieldInvert`/`FieldDrive`). Every
/// path reads the CURRENT effective (resolved) value first, so the step is
/// always "away from what's actually in effect" — never a guess from an
/// absent raw key. Refuses (with a status hint, no mutation) when nothing is
/// selected or the field currently fails to resolve.
pub fn cycle_field_enum(state: &mut EditorState) {
    let Some(idx) = state.selected_field else {
        state.status = "no field selected".to_string();
        return;
    };
    let field = state.entity_field;
    match field {
        EntityField::FieldArchetype => {
            let Some(r) = state.level.resolved_fields.get(idx) else {
                state.status = "field: fix the reported error before editing".to_string();
                return;
            };
            let next = cycle_str(&r.archetype.clone(), &FIELD_ARCHETYPES, 1);
            // A POSITIONED shape is anything with an `at`/`region`/`follow`
            // (Sphere/Box/Capsule) — i.e. everything except the shapeless
            // Room. `r.shape` already tells us which.
            let is_positioned = !matches!(r.shape, FieldShape::Room);
            let radius = r.radius;
            let def = &mut state.level.field_defs[idx];
            // FREEZE the current resolved radius as an explicit key before
            // swapping archetypes, but ONLY when this is a POSITIONED field
            // with no `radius =` of its own. Some archetypes default their
            // OWN radius to `room`; silently inheriting that default
            // on an unrelated kind swap would conflict with the field's own
            // `at`/`region`/`follow` and break `resolve()` out from under the
            // author (`radius = room` takes none of them — the same error
            // whether the shape is a Sphere's `at`, a Box's `region`, or a
            // Capsule's `follow`). Freezing the numeric value it already had
            // keeps the field's visible size identical across the swap and
            // makes that conflict impossible for EVERY positioned shape, not
            // just Sphere.
            if is_positioned && get_key(def, "radius").is_none() {
                if let FieldRadius::Value(v) = radius {
                    set_key(def, "radius", format!("{v:.1}"));
                }
            }
            set_key(def, "kind", next.clone());
            re_resolve(state);
            state.mark_dirty();
            state.status = format!("field kind: {next}");
        }
        EntityField::FieldShapeKind => cycle_shape_kind(state, idx),
        EntityField::FieldFalloff => {
            let Some(r) = state.level.resolved_fields.get(idx) else {
                state.status = "field: fix the reported error before editing".to_string();
                return;
            };
            let current = falloff_to_str(r.falloff);
            let next = cycle_str(current, &FALLOFF_OPTIONS, 1);
            set_key(&mut state.level.field_defs[idx], "falloff", next.clone());
            re_resolve(state);
            state.mark_dirty();
            state.status = format!("field falloff: {next}");
        }
        EntityField::FieldOcclude => {
            let Some(r) = state.level.resolved_fields.get(idx) else {
                state.status = "field: fix the reported error before editing".to_string();
                return;
            };
            let next = !r.occlude;
            set_key(
                &mut state.level.field_defs[idx],
                "occlude",
                yesno(next).to_string(),
            );
            re_resolve(state);
            state.mark_dirty();
            state.status = format!("field occlude: {}", yesno(next));
        }
        EntityField::FieldInvert => {
            let Some(r) = state.level.resolved_fields.get(idx) else {
                state.status = "field: fix the reported error before editing".to_string();
                return;
            };
            let next = !r.invert;
            set_key(
                &mut state.level.field_defs[idx],
                "invert",
                yesno(next).to_string(),
            );
            re_resolve(state);
            state.mark_dirty();
            state.status = format!("field invert: {}", yesno(next));
        }
        EntityField::FieldFacing => cycle_facing(state, idx),
        EntityField::FieldBind => {
            let Some(r) = state.level.resolved_fields.get(idx) else {
                state.status = "field: fix the reported error before editing".to_string();
                return;
            };
            let current = bind_to_str(&r.bind, &state.level);
            // Sticky-remember an off-picker `budget:`/`seq:`/`read:` target
            // before stepping off it — see `cycle_facing`'s doc; same
            // `crate::vocab` contract.
            let canon = bind_options(&state.level);
            crate::vocab::remember(&canon, &mut state.vocab.field_bind, &current);
            let next = crate::vocab::step(&canon, &state.vocab.field_bind, &current, 1);
            set_key(&mut state.level.field_defs[idx], "bind", next.clone());
            re_resolve(state);
            state.mark_dirty();
            state.status = format!("field bind: {next}");
        }
        EntityField::FieldDrive => {
            let Some(r) = state.level.resolved_fields.get(idx) else {
                state.status = "field: fix the reported error before editing".to_string();
                return;
            };
            let current = drive_to_str(&r.drive, &state.level);
            // Sticky-remember an off-picker `ring:` target before stepping
            // off it — see `cycle_facing`'s doc; same `crate::vocab` contract.
            let canon = drive_options(&state.level);
            crate::vocab::remember(&canon, &mut state.vocab.field_drive, &current);
            let next = crate::vocab::step(&canon, &state.vocab.field_drive, &current, 1);
            set_key(&mut state.level.field_defs[idx], "drive", next.clone());
            re_resolve(state);
            state.mark_dirty();
            state.status = format!("field drive: {next}");
        }
        _ => state.status = format!("'{}' is numeric \u{2014} use +/-", field.label()),
    }
}

fn clamp_axis(v: i32, extent: usize) -> i32 {
    v.clamp(0, extent.saturating_sub(1) as i32)
}

/// `+`/`-` on a `Field*` numeric row (`FieldPosX/Y/Z`, `FieldRadius`,
/// `FieldInner`, `FieldCone`, `FieldBase`, `FieldDepth`, `FieldPriority`).
/// Every step is computed from the CURRENT resolved value (same "away from
/// what's actually in effect" contract as [`cycle_field_enum`]) and written
/// back as an explicit authored key. Refuses gracefully (status hint, no
/// mutation) for a row that is not meaningful in the field's current shape
/// (e.g. position while `shape = room`, cone while `facing = none`).
pub fn adjust_field_numeric(state: &mut EditorState, delta: i32) {
    let Some(idx) = state.selected_field else {
        state.status = "no field selected".to_string();
        return;
    };
    let Some(r) = state.level.resolved_fields.get(idx).cloned() else {
        state.status = "field: fix the reported error before editing".to_string();
        return;
    };
    let field = state.entity_field;
    let (sx, sy, sz) = state.map.size();
    match field {
        EntityField::FieldPosX | EntityField::FieldPosY | EntityField::FieldPosZ => {
            // `at` is a POSITIONAL key — inline attaches don't carry one at
            // all (the host supplies it); see `cycle_shape_kind`'s doc for
            // why writing it here for a non-Standalone field would break
            // round-trip.
            if state.level.field_defs[idx].host != FieldHost::Standalone {
                state.status = "field position is set by its host (inline attach) \u{2014} not \
                                 editable here"
                    .to_string();
                return;
            }
            let FieldShape::Sphere { at } = r.shape else {
                state.status = "field shape is room: no position to move".to_string();
                return;
            };
            let mut p = at;
            let (axis, extent, label) = match field {
                EntityField::FieldPosX => (0usize, sx, "x"),
                EntityField::FieldPosY => (1usize, sy, "y"),
                _ => (2usize, sz, "z"),
            };
            p[axis] = clamp_axis(p[axis].round() as i32 + delta, extent) as f32;
            set_key(
                &mut state.level.field_defs[idx],
                "at",
                format!("{}, {}, {}", p[0], p[1], p[2]),
            );
            re_resolve(state);
            state.mark_dirty();
            state.status = format!("field {label} {}", p[axis]);
        }
        EntityField::FieldRadius => {
            if matches!(r.radius, FieldRadius::Room) {
                state.status = "radius is 'room' \u{2014} cycle shape to sphere first".to_string();
                return;
            }
            let cur = match r.radius {
                FieldRadius::Value(v) => v,
                FieldRadius::Room => 1.0,
            };
            let next = (cur + delta as f32 * 0.5).max(0.5);
            set_key(
                &mut state.level.field_defs[idx],
                "radius",
                format!("{next:.1}"),
            );
            re_resolve(state);
            state.mark_dirty();
            state.status = format!("field radius {next:.1}");
        }
        EntityField::FieldInner => {
            let next = (r.inner + delta as f32 * 0.5).max(0.0);
            set_key(
                &mut state.level.field_defs[idx],
                "inner",
                format!("{next:.1}"),
            );
            re_resolve(state);
            state.mark_dirty();
            state.status = format!("field inner {next:.1}");
        }
        EntityField::FieldCone => {
            let Some((_, half)) = r.cone else {
                state.status = "cone needs a facing \u{2014} cycle 'facing' first".to_string();
                return;
            };
            let next = (half + delta as f32 * 5.0).clamp(1.0, 180.0);
            set_key(
                &mut state.level.field_defs[idx],
                "cone",
                format!("{next:.0}"),
            );
            re_resolve(state);
            state.mark_dirty();
            state.status = format!("field cone {next:.0}\u{b0}");
        }
        EntityField::FieldBase => {
            let next = (r.base + delta as f32 * 0.05).clamp(0.0, 1.0);
            set_key(
                &mut state.level.field_defs[idx],
                "base",
                format!("{next:.2}"),
            );
            re_resolve(state);
            state.mark_dirty();
            state.status = format!("field intensity {next:.2}");
        }
        EntityField::FieldDepth => {
            let next = (r.depth + delta as f32 * 0.05).clamp(0.0, 1.0);
            set_key(
                &mut state.level.field_defs[idx],
                "depth",
                format!("{next:.2}"),
            );
            re_resolve(state);
            state.mark_dirty();
            state.status = format!("field depth {next:.2}");
        }
        EntityField::FieldPriority => {
            let next = (i32::from(r.priority) + delta).clamp(0, 9);
            set_key(
                &mut state.level.field_defs[idx],
                "priority",
                next.to_string(),
            );
            re_resolve(state);
            state.mark_dirty();
            state.status = format!("field priority {next}");
        }
        _ => state.status = format!("'{}' is an enum \u{2014} use c", field.label()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use re_level::fields::FieldRadius as FR;

    fn sphere(archetype: &str, at: [f32; 3], radius: f32, inner: f32) -> ResolvedField {
        ResolvedField {
            name: "t".into(),
            archetype: archetype.into(),
            shape: FieldShape::Sphere { at },
            radius: FR::Value(radius),
            inner,
            falloff: Falloff::Smooth,
            occlude: false,
            cone: None,
            depth: 1.0,
            bind: FieldBind::None,
            invert: false,
            drive: DriveSpec::Const,
            base: 1.0,
            phase: [1.0; 4],
            dwell: 0.0,
            idle_boost: false,
            buses: re_level::fields::ALL_BUSES,
            priority: 5,
            attack: 0.4,
            release: 1.2,
            variant: None,
            seed: 1,
            line: 0,
        }
    }

    // ── key upsert ──
    #[test]
    fn set_key_updates_in_place_and_appends_new_keys_at_the_end() {
        let mut def = FieldDef {
            name: "f".into(),
            line: 0,
            host: FieldHost::Standalone,
            keys: vec![FieldKey {
                key: "kind".into(),
                value: "lethe_wash".into(),
                line: 1,
            }],
        };
        set_key(&mut def, "kind", "bit_rot".into());
        assert_eq!(get_key(&def, "kind"), Some("bit_rot"));
        assert_eq!(
            def.keys.len(),
            1,
            "updating an existing key does not duplicate it"
        );
        set_key(&mut def, "radius", "5".into());
        assert_eq!(def.keys.len(), 2);
        assert_eq!(
            def.keys[1].key, "radius",
            "a new key is appended, not inserted"
        );
        remove_key(&mut def, "radius");
        assert_eq!(get_key(&def, "radius"), None);
    }

    // ── distance / falloff / cone (pure-logic acceptance) ──
    #[test]
    fn sphere_distance_is_euclidean() {
        let s = FieldShape::Sphere {
            at: [0.0, 0.0, 0.0],
        };
        assert!((distance_to_shape(&s, [3.0, 4.0, 0.0]) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn box_distance_is_zero_inside_and_correct_outside() {
        let b = FieldShape::BoxShape {
            min: [0.0, 0.0, 0.0],
            max: [2.0, 2.0, 2.0],
        };
        assert_eq!(distance_to_shape(&b, [1.0, 1.0, 1.0]), 0.0);
        assert!((distance_to_shape(&b, [5.0, 0.0, 0.0]) - 3.0).abs() < 1e-5);
    }

    #[test]
    fn capsule_distance_follows_the_path() {
        let c = FieldShape::Capsule {
            points: vec![[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        };
        assert!((distance_to_shape(&c, [5.0, 3.0, 0.0]) - 3.0).abs() < 1e-5);
        assert!((distance_to_shape(&c, [-2.0, 0.0, 0.0]) - 2.0).abs() < 1e-5);
    }

    #[test]
    fn room_shape_is_always_in_range() {
        assert_eq!(distance_to_shape(&FieldShape::Room, [999.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn falloff_smooth_is_half_at_the_band_midpoint() {
        // inner=0, radius=10: midpoint d=5 -> t=0.5 -> smoothstep(0.5)=0.5.
        let w = falloff_weight(Falloff::Smooth, 10.0, 0.0, 5.0);
        assert!((w - 0.5).abs() < 1e-5, "{w}");
    }

    #[test]
    fn falloff_linear_is_exactly_proportional() {
        let w = falloff_weight(Falloff::Linear, 10.0, 0.0, 5.0);
        assert!((w - 0.5).abs() < 1e-5, "{w}");
    }

    #[test]
    fn falloff_inverse_is_half_exactly_at_radius() {
        let w = falloff_weight(Falloff::Inverse, 8.0, 0.0, 8.0);
        assert!((w - 0.5).abs() < 1e-5, "{w}");
    }

    #[test]
    fn falloff_hard_has_a_half_cell_edge() {
        assert_eq!(falloff_weight(Falloff::Hard, 10.0, 0.0, 9.0), 1.0);
        assert_eq!(falloff_weight(Falloff::Hard, 10.0, 0.0, 11.0), 0.0);
        let mid = falloff_weight(Falloff::Hard, 10.0, 0.0, 10.0);
        assert!((mid - 0.5).abs() < 1e-5, "{mid}");
    }

    #[test]
    fn falloff_is_full_strength_inside_inner() {
        assert_eq!(falloff_weight(Falloff::Smooth, 10.0, 3.0, 1.0), 1.0);
    }

    #[test]
    fn cone_factor_admits_the_forward_direction_and_rejects_behind() {
        let cone = Some(([0.0, 1.0, 0.0], 30.0));
        assert_eq!(cone_factor(cone, [0.0, 0.0, 0.0], [0.0, 5.0, 0.0]), 1.0);
        assert_eq!(cone_factor(cone, [0.0, 0.0, 0.0], [0.0, -5.0, 0.0]), 0.0);
    }

    #[test]
    fn cone_factor_is_one_with_no_cone_authored() {
        assert_eq!(cone_factor(None, [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]), 1.0);
    }

    // ── evaluate_weight / build_mix / cursor_readout ──
    #[test]
    fn evaluate_weight_decays_with_distance() {
        let f = sphere("lethe_wash", [0.0, 0.0, 0.0], 10.0, 0.0);
        let near = evaluate_weight(&f, [1.0, 0.0, 0.0]);
        let far = evaluate_weight(&f, [9.0, 0.0, 0.0]);
        assert!(near > far, "near={near} far={far}");
    }

    #[test]
    fn evaluate_weight_room_ignores_distance() {
        let mut f = sphere("sky_hush", [0.0, 0.0, 0.0], 1.0, 0.0);
        f.radius = FR::Room;
        f.shape = FieldShape::Room;
        let a = evaluate_weight(&f, [0.0, 0.0, 0.0]);
        let b = evaluate_weight(&f, [500.0, 500.0, 0.0]);
        assert_eq!(a, b);
    }

    #[test]
    fn const_drive_uses_base_state_driven_drive_uses_the_preview_fallback() {
        let mut f = sphere("phantom_hollow", [0.0, 0.0, 0.0], 10.0, 0.0);
        f.base = 0.4;
        f.drive = DriveSpec::Const;
        assert!((evaluate_weight(&f, [0.0, 0.0, 0.0]) - 0.4).abs() < 1e-5);
        f.drive = DriveSpec::Lit(0);
        assert!((evaluate_weight(&f, [0.0, 0.0, 0.0]) - PREVIEW_DRIVE_FALLBACK).abs() < 1e-5);
    }

    #[test]
    fn cursor_readout_reports_none_audible_when_nothing_is_in_range() {
        let lvl = re_level::load_string(
            // `lethe_wash` defaults `base = 0.0` (silent until a `bind`/`drive`
            // wakes it, per the archetype table) — an authored
            // `base = 1` is required or `Const` drive * base(0.0) reads zero
            // EVERYWHERE, near included, and this fixture would test nothing.
            "[meta]\nname = t\n[room]\nsize = 20 20 4\n[pedestal]\ncenter=1,1,1\nvolume=1 1 1\n\
             [door]\npos=5 5 1\n[player]\nspawn=2 2 1\n\
             [field river]\nkind = lethe_wash\nat = 1,1,1\nradius=2\nbase=1\n",
            "t.lvl",
        )
        .unwrap();
        let far = cursor_readout(&lvl, [19.0, 19.0, 1.0]);
        assert_eq!(far, "fields: none audible here");
        let near = cursor_readout(&lvl, [1.0, 1.0, 1.0]);
        assert!(near.starts_with("fields: lethe_wash"), "{near}");
        assert!(near.contains("AMB LP"), "{near}");
    }

    #[test]
    fn build_mix_of_no_fields_is_neutral() {
        let lvl = re_level::load_string(
            "[meta]\nname = t\n[room]\nsize = 8 8 4\n[pedestal]\ncenter=1,1,1\nvolume=1 1 1\n\
             [door]\npos=5 5 1\n[player]\nspawn=2 2 1\n",
            "t.lvl",
        )
        .unwrap();
        let mix = build_mix(&lvl, [0.0, 0.0, 0.0]);
        assert_eq!(mix, re_audio::FieldMix::NEUTRAL);
    }

    // ── 2D view aid ──
    #[test]
    fn footprint_classifies_interior_and_both_rings() {
        let f = sphere("bit_rot", [5.0, 5.0, 1.0], 4.0, 1.0);
        assert_eq!(footprint_at(&f, [5.0, 5.0, 1.0]), Some(CellTint::Interior));
        assert_eq!(footprint_at(&f, [5.9, 5.0, 1.0]), Some(CellTint::InnerRing));
        assert_eq!(footprint_at(&f, [9.0, 5.0, 1.0]), Some(CellTint::OuterRing));
        assert_eq!(footprint_at(&f, [20.0, 5.0, 1.0]), None);
    }

    #[test]
    fn footprint_room_shape_has_no_footprint() {
        let mut f = sphere("sky_hush", [0.0, 0.0, 0.0], 1.0, 0.0);
        f.radius = FR::Room;
        assert_eq!(footprint_at(&f, [3.0, 3.0, 1.0]), None);
    }

    #[test]
    fn family_colors_match_the_five_named_buckets_plus_grey_fallback() {
        assert_eq!(family_color("lethe_wash"), 0x00_4C8CFF);
        assert_eq!(family_color("mnemos_hold"), 0x00_D9A441);
        assert_eq!(family_color("bit_rot"), 0x00_6FCB6F);
        assert_eq!(family_color("watcher_gaze"), 0x00_9B6FE0);
        assert_eq!(family_color("steam_warmth"), 0x00_E08A3C);
        assert_eq!(family_color("exit_detune"), 0x00_9AA4B2);
        assert_eq!(family_color("sky_hush"), 0x00_9AA4B2);
    }

    #[test]
    fn cell_tint_is_none_with_the_aid_off() {
        let mut s = EditorState::new_empty(16, 16, 4);
        s.level.field_defs.push(FieldDef {
            name: "f".into(),
            line: 0,
            host: FieldHost::Standalone,
            keys: vec![
                FieldKey {
                    key: "kind".into(),
                    value: "lethe_wash".into(),
                    line: 0,
                },
                FieldKey {
                    key: "at".into(),
                    value: "5, 5, 0".into(),
                    line: 0,
                },
            ],
        });
        re_resolve(&mut s);
        assert!(!s.fields_aid);
        assert_eq!(
            cell_tint(&s, 5, 5, 0),
            None,
            "the aid must be OFF by default"
        );
        s.fields_aid = true;
        assert!(
            cell_tint(&s, 5, 5, 0).is_some(),
            "on, the field's own cell must tint"
        );
    }

    // ── re_resolve / field_at ──
    #[test]
    fn re_resolve_populates_and_field_at_finds_a_sphere_field() {
        let mut s = EditorState::new_empty(16, 16, 4);
        s.level.field_defs.push(FieldDef {
            name: "river".into(),
            line: 0,
            host: FieldHost::Standalone,
            keys: vec![
                FieldKey {
                    key: "kind".into(),
                    value: "lethe_wash".into(),
                    line: 0,
                },
                FieldKey {
                    key: "at".into(),
                    value: "3, 4, 0".into(),
                    line: 0,
                },
            ],
        });
        re_resolve(&mut s);
        assert_eq!(s.level.resolved_fields.len(), 1);
        assert_eq!(field_at(&s, 3, 4, 0), Some(0));
        assert_eq!(field_at(&s, 0, 0, 0), None);
    }

    #[test]
    fn re_resolve_on_error_clears_the_cache_and_reports_the_line() {
        let mut s = EditorState::new_empty(16, 16, 4);
        s.level.field_defs.push(FieldDef {
            name: "bad".into(),
            line: 7,
            host: FieldHost::Standalone,
            keys: vec![FieldKey {
                key: "kind".into(),
                value: "lethe_wash".into(),
                line: 7,
            }],
        });
        re_resolve(&mut s);
        assert!(s.level.resolved_fields.is_empty());
        assert!(s.status.contains("field error"), "{}", s.status);
    }

    // ── option lists / cycle safety ──
    #[test]
    fn bind_and_drive_options_name_every_node_and_pipe() {
        let mut lvl = Level::default();
        lvl.push_node(1, 1, 0);
        lvl.push_node(2, 2, 0);
        assert!(bind_options(&lvl).contains(&"node:n0".to_string()));
        assert!(bind_options(&lvl).contains(&"fed:n1".to_string()));
        assert!(drive_options(&lvl).contains(&"lit:n0".to_string()));
        assert!(drive_options(&lvl).contains(&"starve:n1".to_string()));
    }

    #[test]
    fn safe_options_splices_an_off_list_current_value_never_loses_it() {
        let canonical = vec!["const".to_string(), "door_ready".to_string()];
        let with_current = safe_options(canonical.clone(), "ring:a->b,c->d");
        assert_eq!(with_current.last(), Some(&"ring:a->b,c->d".to_string()));
        // Already-present values are not duplicated.
        let unchanged = safe_options(canonical.clone(), "const");
        assert_eq!(unchanged, canonical);
    }

    // ── audition gating (the load-bearing test: TS_AUDIO_SILENT => no Player) ──
    #[test]
    fn audition_start_refuses_and_opens_no_player_when_silent_env_is_set() {
        std::env::set_var("TS_AUDIO_SILENT", "1");
        assert!(re_audio::playback::silence_requested());
        let lvl = Level::default();
        // `.unwrap_err()` needs `T: Debug`; `Audition` cannot derive it (its
        // `re_audio::Player` field carries an `mpsc::Receiver`, which has no
        // `Debug` impl), so match explicitly instead.
        let err = match Audition::start(&lvl) {
            Ok(_) => panic!("expected an error under TS_AUDIO_SILENT"),
            Err(e) => e,
        };
        assert!(err.contains("TS_AUDIO_SILENT"), "{err}");
    }

    #[test]
    fn toggle_audition_leaves_state_none_when_silent() {
        std::env::set_var("TS_AUDIO_SILENT", "1");
        let mut s = EditorState::new_empty(8, 8, 2);
        toggle_audition(&mut s);
        assert!(s.audition.is_none());
        assert!(s.status.contains("TS_AUDIO_SILENT"), "{}", s.status);
    }

    // ── keymap wiring (mirrors `viewaids`'s
    // `every_view_aid_chord_in_the_keymap_is_a_chord_the_editor_answers`) ──
    #[test]
    fn the_fields_aid_chord_in_the_keymap_is_a_chord_the_editor_answers() {
        use re_present::EditorKeyCode as E;
        let b = crate::keymap::KEYMAP
            .iter()
            .find(|b| b.chord == "F" && b.group == "view / files")
            .expect("chord F must be in the keymap table");
        assert!(b.palette, "F must be palette-reachable");
        assert_eq!(b.key, Some(E::Char('F')), "F must be executable");
        let mut s = EditorState::new_empty(8, 8, 2);
        crate::edit::apply_key(
            &mut s,
            re_present::EditorKey {
                code: E::Char('F'),
                shift: true,
            },
        );
        assert!(s.fields_aid);
    }

    #[test]
    fn the_audition_chord_in_the_keymap_is_a_chord_the_editor_answers() {
        use re_present::EditorKeyCode as E;
        let b = crate::keymap::KEYMAP
            .iter()
            .find(|b| b.chord == "Shift+U" && b.group == "view / files")
            .expect("chord Shift+U must be in the keymap table");
        assert!(b.palette, "Shift+U must be palette-reachable");
        assert_eq!(b.key, Some(E::Char('U')), "U must be executable");
        std::env::set_var("TS_AUDIO_SILENT", "1");
        let mut s = EditorState::new_empty(8, 8, 2);
        crate::edit::apply_key(
            &mut s,
            re_present::EditorKey {
                code: E::Char('U'),
                shift: true,
            },
        );
        // Silent env: the chord is answered (status changes) but no Player opens.
        assert!(s.audition.is_none());
        assert!(s.status.contains("TS_AUDIO_SILENT"), "{}", s.status);
    }

    #[test]
    fn toggle_fields_aid_flips_and_never_marks_dirty() {
        let mut s = EditorState::new_empty(8, 8, 2);
        assert!(!s.dirty);
        toggle_fields_aid(&mut s);
        assert!(s.fields_aid);
        assert!(!s.dirty, "a view aid never edits the document");
        toggle_fields_aid(&mut s);
        assert!(!s.fields_aid);
    }
}
