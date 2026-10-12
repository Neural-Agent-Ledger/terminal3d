//! **Sound fields** — `[field NAME]` sections.
//!
//! A sound field is a spatial region, carried by an authored `[field NAME]` section,
//! that later bends the mix (`re_audio::field::combine`) as the player gets close.
//! `re_level` is engine-agnostic and knows nothing about `re_audio` — this module owns
//! only the LEVEL-FORMAT side: parsing, defaults, validation and name/index resolution.
//! `re_runtime::sound_fields` turns a [`ResolvedField`] plus live puzzle state into an
//! `re_audio::field::FieldInput` every frame.
//!
//! # Scope note (deliberate)
//!
//! The design doc's §4.1 "one-line attach to an existing entity" (`field = archetype
//! k=v` inline inside a host section) is implemented for **`[node]`, `[pipe]` and
//! `[door]`** — the three host kinds whose position/shape is already a plain field on
//! the parsed struct (a node's `x,y,z`, a pipe's expanded `path`, the level's single
//! `door.{x,y,z}`), so no extra geometry-resolution machinery is needed to supply the
//! implied `at`/`follow` shape. [`FieldHost::Node`]/[`FieldHost::Pipe`]/
//! [`FieldHost::Door`] additionally supply the host-implied **default `drive =`**
//! (§4.1's host table): `threshold_strain` on a node defaults to `starve:<node>`, any
//! other archetype on a node defaults to `lit:<node>`; `bit_rot` on a pipe defaults to
//! `dead_draw:<pipe>`, `live_wire` defaults to `live:<pipe>`; a door always defaults to
//! `door_ready`. An explicit `drive =` in the inline text overrides the implied one.
//!
//! `[prop]`, `[wall_decal]`, `[holo]`, `[portal]`, `[zone_input]`, `[trigger]`,
//! `[emitter]` and `[light]` hosts are **not implemented**: several of them (`[prop]`
//! facing/power, `[wall_decal]`'s "the normal points into the room from the wall
//! plane") need a geometry/spec resolution step (`props::parse_spec`, or — for
//! `wall_decal` — inferring which side of a flush-authored region is "the room",
//! which is not determinable from the region alone without walking the built map)
//! this pass did not implement. Every `[field]` example in the design doc that
//! attaches to one of those host kinds is instead authored as an equivalent
//! standalone `[field NAME]` section (§4.2) — functionally identical, since
//! `Level::fields()` is the one surface the runtime reads regardless of which syntax
//! produced a [`ResolvedField`]; only the authoring shortcut is missing for those 8
//! kinds. Flagged as follow-up work.

use std::collections::HashSet;

use crate::{parse_region, parse_vec3, split_arrow, Level};

/// Mirror of `re_audio::field::Archetype::ALL`'s `.name()`s (§7.0's frozen contract:
/// `re_level` must not depend on `re_audio`, so the 15 names are duplicated here; a
/// cross-crate test — `FIELD_ARCHETYPES` equals `Archetype::ALL.map(name)` — is the
/// single point of truth that keeps the two tables from drifting).
pub const FIELD_ARCHETYPES: [&str; 15] = [
    "lethe_wash",
    "mnemos_hold",
    "bit_rot",
    "false_echo",
    "watcher_gaze",
    "phantom_hollow",
    "lure_wobble",
    "exit_detune",
    "live_wire",
    "threshold_strain",
    "brownout",
    "tail_f",
    "recognition_narrow",
    "steam_warmth",
    "sky_hush",
];

/// Every key a `[field]` section may carry (§4.2). Unknown keys are a parse error,
/// same contract as `props::is_known_key`.
const FIELD_KEYS: [&str; 24] = [
    "kind",
    "at",
    "region",
    "follow",
    "radius",
    "inner",
    "falloff",
    "facing",
    "cone",
    "occlude",
    "depth",
    "bind",
    "invert",
    "drive",
    "base",
    "phase",
    "dwell",
    "idle_boost",
    "buses",
    "priority",
    "attack",
    "release",
    "variant",
    "seed",
];

/// True iff `k` is one of the §4.2 `[field]` section keys (`FIELD_KEYS`).
/// An unknown key is a parse error, same contract as `props::is_known_key`.
#[must_use]
pub fn is_known_key(k: &str) -> bool {
    FIELD_KEYS.contains(&k)
}

/// §4.1: an inline knob is any §4.2 key EXCEPT the positional ones (`at`,
/// `region`, `follow` — supplied by the host) and `kind` itself (the
/// archetype is the inline text's first bare token, never a `kind=` pair).
#[must_use]
fn is_known_inline_knob(k: &str) -> bool {
    is_known_key(k) && !matches!(k, "kind" | "at" | "region" | "follow")
}

/// The deterministic, host-derived name synthesized for an inline attach
/// (never authored, so it can never collide with a real `[field NAME]` —
/// every real name matches `props`' `[a-z0-9_]+` convention, which excludes
/// `$` and `:`). Stable across re-parses of the same file: it depends only on
/// the host, not on insertion order, which is what makes the round-trip
/// identity test (`resolve → to_lvl_text → resolve`) exact.
fn host_inline_name(host: &FieldHost) -> String {
    match host {
        FieldHost::Node(i) => format!("$inline:node:{i}"),
        FieldHost::Pipe(i) => format!("$inline:pipe:{i}"),
        FieldHost::Door => "$inline:door".to_string(),
        FieldHost::Prop(name) => format!("$inline:prop:{name}"),
        FieldHost::Decal(i) => format!("$inline:decal:{i}"),
        FieldHost::Holo(i) => format!("$inline:holo:{i}"),
        FieldHost::Portal(i) => format!("$inline:portal:{i}"),
        FieldHost::ZoneInput(i) => format!("$inline:zone_input:{i}"),
        FieldHost::Trigger(i) => format!("$inline:trigger:{i}"),
        FieldHost::Emitter(i) => format!("$inline:emitter:{i}"),
        FieldHost::Light(i) => format!("$inline:light:{i}"),
        FieldHost::Sky => "$inline:sky".to_string(),
        FieldHost::Standalone => unreachable!("attach_inline is never called with Standalone"),
    }
}

/// Parses the §4.1 inline value `ARCHETYPE k=v k=v…` and appends the
/// resulting synthetic [`FieldDef`] to `level.field_defs`. The caller (one of
/// `lib.rs`'s `[node]`/`[pipe]`/`[door]` key-match arms) supplies the
/// already-resolved `host` and the 1-based source line; the returned error
/// (never wrapped — the caller adds the `[section] field:` prefix its own
/// diagnostics use) names exactly what is wrong: an unknown archetype, a
/// malformed `k=v` pair, an unknown/positional knob, or a second `field =` on
/// the same host (§4.1: "At most **one** `field =` per host").
pub fn attach_inline(
    level: &mut Level,
    host: FieldHost,
    v: &str,
    line: usize,
) -> Result<(), String> {
    if level.field_defs.iter().any(|f| f.host == host) {
        return Err("at most one 'field =' per host".to_string());
    }
    let mut tokens = v.split_whitespace();
    let Some(archetype) = tokens.next() else {
        return Err("expected 'ARCHETYPE k=v …'".to_string());
    };
    if !FIELD_ARCHETYPES.contains(&archetype) {
        return Err(format!("unknown archetype '{archetype}'"));
    }
    let mut keys = vec![FieldKey {
        key: "kind".to_string(),
        value: archetype.to_string(),
        line,
    }];
    for tok in tokens {
        let Some((k, val)) = tok.split_once('=') else {
            return Err(format!("malformed knob '{tok}' (expected 'key=value')"));
        };
        if !is_known_inline_knob(k) {
            return Err(format!(
                "unknown knob '{k}' (positional keys are supplied by the host; \
                 did you mean a `[field]` section instead?)"
            ));
        }
        keys.push(FieldKey {
            key: k.to_string(),
            value: val.to_string(),
            line,
        });
    }
    let name = host_inline_name(&host);
    level.field_defs.push(FieldDef {
        name,
        line,
        host,
        keys,
    });
    Ok(())
}

/// The inverse of [`attach_inline`], for the writer (`re_voximport`):
/// reconstructs the `ARCHETYPE k=v k=v…` text for a host-attached
/// [`FieldDef`]. `None` for [`FieldHost::Standalone`] (written as its own
/// `[field NAME]` section instead, unchanged from before this pass).
#[must_use]
pub fn to_inline_line(def: &FieldDef) -> Option<String> {
    if def.host == FieldHost::Standalone {
        return None;
    }
    // `kind` is always `keys[0]` by construction (`attach_inline` pushes it
    // first and the parser only ever appends), but this does not rely on
    // that order holding forever: it explicitly finds and fronts it.
    let kind_pos = def.keys.iter().position(|k| k.key == "kind")?;
    let mut parts = Vec::with_capacity(def.keys.len());
    parts.push(def.keys[kind_pos].value.clone());
    for (i, k) in def.keys.iter().enumerate() {
        if i != kind_pos {
            parts.push(format!("{}={}", k.key, k.value));
        }
    }
    Some(parts.join(" "))
}

/// Where a field's position/shape/implied defaults came from (§7.0's frozen
/// cross-crate API). `Standalone` is an authored `[field NAME]` section (§4.2);
/// every other variant is a §4.1 inline `field = ARCHETYPE k=v…` attach. Only
/// `Node`/`Pipe`/`Door` are produced by this pass's parser — see the module
/// doc's scope note for the rest, kept here so the type is the one the doc
/// specifies even where a variant has no producer yet.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum FieldHost {
    /// An authored `[field NAME]` section — not attached to any other entity.
    #[default]
    Standalone,
    /// §4.1 inline attach on `[node]` index `usize`.
    Node(usize),
    /// §4.1 inline attach on `[pipe]` index `usize`.
    Pipe(usize),
    /// §4.1 inline attach on the level's single `[door]`.
    Door,
    /// Inline attach on a `[prop NAME]` section (parser not yet implemented —
    /// see the module doc's scope note).
    Prop(String),
    /// Inline attach on a `[wall_decal]` index (parser not yet implemented).
    Decal(usize),
    /// Inline attach on a `[holo]` index (parser not yet implemented).
    Holo(usize),
    /// Inline attach on a `[portal]` index (parser not yet implemented).
    Portal(usize),
    /// Inline attach on a `[zone_input]` index (parser not yet implemented).
    ZoneInput(usize),
    /// Inline attach on a `[trigger]` index (parser not yet implemented).
    Trigger(usize),
    /// Inline attach on an `[emitter]` index (parser not yet implemented).
    Emitter(usize),
    /// Inline attach on a `[light]` index (parser not yet implemented).
    Light(usize),
    /// Inline attach on the level's single `[sky]` (parser not yet implemented).
    Sky,
}

/// One authored `[field NAME]` section, or one §4.1 inline `field =` attach
/// desugared to the same shape. Keys are kept VERBATIM (order + values) so the
/// writer re-emits them byte-for-byte, mirroring `props::PropDef`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FieldDef {
    /// The field's name — authored for [`FieldHost::Standalone`], or the
    /// synthesized `$inline:…` name (`host_inline_name`) for an attach.
    pub name: String,
    /// 1-based source line (the `[field NAME]` header, or the `field =` line
    /// for an inline attach).
    pub line: usize,
    /// Where this field's position/shape/defaults come from.
    pub host: FieldHost,
    /// The verbatim `key=value` pairs, in authored order.
    pub keys: Vec<FieldKey>,
}

/// One verbatim `key=value` pair from a `[field]` section or inline attach.
#[derive(Clone, Debug, PartialEq)]
pub struct FieldKey {
    /// The key name (validated against `FIELD_KEYS`).
    pub key: String,
    /// The raw, unparsed value text.
    pub value: String,
    /// 1-based source line.
    pub line: usize,
}

/// §2.0's shape vocabulary, resolved to plain numbers (world-cell units, the same
/// frame `[light] pos` and `CameraPose.pos` use).
#[derive(Clone, Debug, PartialEq)]
pub enum FieldShape {
    /// `at = x,y,z`: a point source.
    Sphere {
        /// World-cell-unit center.
        at: [f32; 3],
    },
    /// `region = …`: an axis-aligned box.
    BoxShape {
        /// World-cell-unit min corner.
        min: [f32; 3],
        /// World-cell-unit max corner.
        max: [f32; 3],
    },
    /// `follow = pipe:A->B`: the pipe's expanded path, as world-cell centers.
    Capsule {
        /// The path's cell centers, in traversal order.
        points: Vec<[f32; 3]>,
    },
    /// `radius = room`: no position — always in addition to the top-3 (§3.1).
    Room,
}

/// How a field's influence fades with distance from its shape (`falloff =`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Falloff {
    /// A smoothstep-shaped fade (the default).
    Smooth,
    /// A linear fade.
    Linear,
    /// An inverse-distance fade.
    Inverse,
    /// A hard cutoff — full strength inside, none outside.
    Hard,
}

impl Falloff {
    fn parse(s: &str) -> Option<Falloff> {
        match s.trim() {
            "smooth" => Some(Falloff::Smooth),
            "linear" => Some(Falloff::Linear),
            "inverse" => Some(Falloff::Inverse),
            "hard" => Some(Falloff::Hard),
            _ => None,
        }
    }
}

/// `radius =`: a plain outer radius/fade distance, or the `room` sentinel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FieldRadius {
    /// An explicit outer radius/fade distance, in world-cell units.
    Value(f32),
    /// `radius = room`: the field's outer edge is the room boundary.
    Room,
}

/// `bind =` (§4.2): the `PowerSpec` grammar plus `fed:N` and `read:INDEX`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FieldBind {
    /// No binding — the field is always active (subject only to `invert`).
    None,
    /// Bound to the level's single door being ready.
    Door,
    /// Bound to a node's activation, by node index.
    Node(usize),
    /// Bound to a pipe's on/off state, by pipe index.
    Pipe(usize),
    /// Bound to the power-budget draw hitting a threshold.
    Budget(u32),
    /// Bound to a sequence-lock progress step.
    Seq(u32),
    /// `fed:N` — true (before `invert`) when `fed:N > 0` (§2.0's state-signal table).
    Fed(usize),
    /// `read:INDEX` — index into `Level::notes`.
    Read(usize),
}

/// `drive =` (§2.0): names which live puzzle/level signal scales the field's
/// intensity. `re_level` only resolves the name to an index (never computes
/// the `[0,1]` scale itself — that is `re_audio::field::combine`'s job each
/// frame, from the live `re_puzzle::PuzzleState`/`re_runtime` values these
/// variants name).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DriveSpec {
    /// `const`: a fixed, level-independent scale (always `base`).
    Const,
    /// `lit:NAME` — a node's activation, by node index.
    Lit(usize),
    /// `fed:NAME` — a node's `fed:N` state signal, by node index.
    Fed(usize),
    /// `starve:NAME` — a node's threshold-strain signal, by node index.
    Starve(usize),
    /// `live:A->B` — a pipe's on/off state, by pipe index.
    Live(usize),
    /// `dead_draw:A->B` — a pipe's dead-draw signal, by pipe index.
    DeadDraw(usize),
    /// `ring:A->B,C->D,…` — pipe indices forming a ring.
    Ring(Vec<usize>),
    /// `door_ready` — whether the level's door is ready.
    DoorReady,
    /// `solved` — whether the level is solved.
    Solved,
    /// `cycle_held` — whether a kept cycle (L12 `while(true)`) is currently held.
    CycleHeld,
    /// `budget` — the power-budget draw fraction.
    Budget,
    /// `seq` — the sequence-lock progress fraction.
    Seq,
    /// `clarity` — the level's authored "clarity" signal.
    Clarity,
    /// `shards` — the level's authored "shards" signal.
    Shards,
}

/// A fully resolved, authored `[field]` — every name resolved to an index, every
/// default applied. `re_runtime::sound_fields` reads these through [`Level::fields`] each level load
/// and evaluates them fresh every frame.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedField {
    /// Authored name for a [`FieldHost::Standalone`], or the synthesized
    /// `$inline:…` name for an attach.
    pub name: String,
    /// Validated against [`FIELD_ARCHETYPES`] at parse time.
    pub archetype: String,
    /// The field's position/extent, resolved from `at` / `region` / `follow`
    /// / `radius = room`.
    pub shape: FieldShape,
    /// Outer radius or fade distance (an explicit value, or the `room` sentinel).
    pub radius: FieldRadius,
    /// Full-strength core radius: inside this distance the field is at its
    /// authored strength; the falloff only applies between `inner` and `radius`.
    pub inner: f32,
    /// How strength fades between `inner` and `radius`.
    pub falloff: Falloff,
    /// Whether world geometry blocks (occludes) this field's contribution.
    pub occlude: bool,
    /// `(direction, half_angle_degrees)`, world-space unit direction. Present
    /// only when `facing =` was authored, narrowing the field to a cone.
    pub cone: Option<([f32; 3], f32)>,
    /// Authored strength, `0..=1` (multiplies the archetype's targets).
    pub depth: f32,
    /// The gate: the field contributes only while `bind` reads true (subject
    /// to `invert`).
    pub bind: FieldBind,
    /// Inverts the `bind` gate's true/false reading.
    pub invert: bool,
    /// The continuous `[0,1]` scale driving the field's intensity each frame.
    pub drive: DriveSpec,
    /// The `drive` floor — the minimum scale even when the driving signal
    /// reads zero.
    pub base: f32,
    /// P1..P4 phase multipliers (per-archetype narrative beats).
    pub phase: [f32; 4],
    /// Seconds the player must remain in-field before the dwell ramp reaches
    /// full strength (time-in-field ramp).
    pub dwell: f32,
    /// `true` ⇒ the field's weight gets an extra ×1.5 after 90s of player
    /// idle (the PROGRESSION §3 unstick rule); author opt-in only, default `false`.
    pub idle_boost: bool,
    /// Bitmask `1=amb 2=mus 4=sfx 8=pos` ([`ALL_BUSES`] = all four). INERT
    /// today: `re_audio::field::combine`'s per-archetype recipes hardcode
    /// their own bus routing (the frozen §7.0 API has no per-`FieldInput` bus
    /// mask), so this narrows nothing yet. Preserved for round-trip fidelity
    /// and as a documented future hook, the same "authored but currently
    /// inert" contract as `Level::room_floor`/`target_model`.
    pub buses: u8,
    /// `0..=9`: which of the top-weighted fields at a point stay audible
    /// (§3.2) when more fields overlap than the mix can carry.
    pub priority: u8,
    /// Seconds to smooth a rising weight change.
    pub attack: f32,
    /// Seconds to smooth a falling weight change.
    pub release: f32,
    /// Archetype-specific variant word (e.g. F4's `memory`, F15's `void`), if authored.
    pub variant: Option<String>,
    /// Seed for this field's deterministic dropout/flicker noise. Defaults to
    /// a hash of `name` when not authored.
    pub seed: u64,
    /// 1-based header line, for diagnostics (lvlcheck locations).
    pub line: usize,
}

/// All four mix buses set in [`ResolvedField::buses`]: `amb | mus | sfx | pos`.
pub const ALL_BUSES: u8 = 0b1111;

/// §4.3 defaults per archetype. `drive`/`base` default to `Const`/`0.0` for every
/// archetype here (see the module doc's scope note: the doc's host-implied default
/// drives, e.g. bit_rot's "dead_draw:host", need an inline-attach host this module
/// does not implement); the three archetypes the doc gives an explicit non-zero
/// standalone `base` for (watcher_gaze, phantom_hollow, lure_wobble) keep it.
struct ArchetypeDefaults {
    radius: FieldRadius,
    inner: f32,
    falloff: Falloff,
    occlude: bool,
    /// Half-angle degrees, applied only when `facing` is authored.
    cone: Option<f32>,
    base: f32,
    priority: u8,
    /// P1..P4, only the archetypes the doc states explicitly; else `[1,1,1,1]`.
    phase: [f32; 4],
    dwell: f32,
}

fn archetype_defaults(name: &str) -> ArchetypeDefaults {
    let v = FieldRadius::Value;
    match name {
        "lethe_wash" => ArchetypeDefaults {
            radius: v(6.0),
            inner: 1.5,
            falloff: Falloff::Smooth,
            occlude: true,
            cone: None,
            base: 0.0,
            priority: 3,
            phase: [0.3, 0.6, 1.0, 1.0],
            dwell: 3.0,
        },
        "mnemos_hold" => ArchetypeDefaults {
            radius: v(5.0),
            inner: 2.0,
            falloff: Falloff::Smooth,
            occlude: false,
            cone: None,
            base: 0.0,
            priority: 5,
            phase: [0.0, 0.3, 0.7, 1.0],
            dwell: 0.0,
        },
        "bit_rot" => ArchetypeDefaults {
            radius: v(4.0),
            inner: 0.5,
            falloff: Falloff::Linear,
            occlude: true,
            cone: None,
            base: 0.0,
            priority: 5,
            phase: [1.0, 1.0, 1.0, 1.0],
            dwell: 0.0,
        },
        "false_echo" => ArchetypeDefaults {
            radius: v(6.0),
            inner: 1.0,
            falloff: Falloff::Smooth,
            occlude: true,
            cone: Some(70.0),
            base: 0.0,
            priority: 5,
            phase: [0.5, 1.0, 1.0, 1.0],
            dwell: 0.0,
        },
        "watcher_gaze" => ArchetypeDefaults {
            radius: v(12.0),
            inner: 2.0,
            falloff: Falloff::Linear,
            occlude: true,
            cone: Some(35.0),
            base: 0.35,
            priority: 5,
            phase: [0.0, 1.0, 0.6, 0.8],
            dwell: 0.0,
        },
        "phantom_hollow" => ArchetypeDefaults {
            radius: v(6.0),
            inner: 1.5,
            falloff: Falloff::Smooth,
            occlude: false,
            cone: None,
            base: 0.25,
            priority: 7,
            phase: [0.0, 0.3, 1.0, 0.6],
            dwell: 0.0,
        },
        "lure_wobble" => ArchetypeDefaults {
            radius: v(4.0),
            inner: 1.0,
            falloff: Falloff::Smooth,
            occlude: true,
            cone: None,
            base: 0.4,
            priority: 5,
            phase: [0.0, 0.6, 0.8, 1.0],
            dwell: 0.0,
        },
        "exit_detune" => ArchetypeDefaults {
            radius: v(7.0),
            inner: 1.0,
            falloff: Falloff::Inverse,
            occlude: true,
            cone: None,
            base: 0.0,
            priority: 9,
            phase: [1.0, 1.0, 1.0, 1.0],
            dwell: 0.0,
        },
        "live_wire" => ArchetypeDefaults {
            radius: v(1.5),
            inner: 0.0,
            falloff: Falloff::Linear,
            occlude: false,
            cone: None,
            base: 0.0,
            priority: 3,
            phase: [1.0, 1.0, 1.0, 1.0],
            dwell: 0.0,
        },
        "threshold_strain" => ArchetypeDefaults {
            radius: v(5.0),
            inner: 1.0,
            falloff: Falloff::Smooth,
            occlude: true,
            cone: None,
            base: 0.0,
            priority: 7,
            phase: [0.6, 1.0, 1.0, 1.0],
            dwell: 0.0,
        },
        "brownout" => ArchetypeDefaults {
            radius: v(4.0),
            inner: 1.0,
            falloff: Falloff::Smooth,
            occlude: false,
            cone: None,
            base: 0.0,
            priority: 5,
            phase: [1.0, 1.0, 1.0, 1.0],
            dwell: 0.0,
        },
        "tail_f" => ArchetypeDefaults {
            radius: v(10.0),
            inner: 0.5,
            falloff: Falloff::Linear,
            occlude: true,
            cone: Some(25.0),
            base: 0.0,
            priority: 1,
            phase: [1.0, 1.0, 1.0, 1.0],
            dwell: 0.0,
        },
        "recognition_narrow" => ArchetypeDefaults {
            radius: FieldRadius::Room,
            inner: 0.0,
            falloff: Falloff::Smooth,
            occlude: false,
            cone: None,
            base: 0.0,
            priority: 7,
            phase: [0.0, 0.0, 0.6, 1.0],
            dwell: 0.0,
        },
        "steam_warmth" => ArchetypeDefaults {
            radius: v(4.0),
            inner: 1.0,
            falloff: Falloff::Smooth,
            occlude: true,
            cone: None,
            base: 0.0,
            priority: 3,
            phase: [1.0, 1.0, 1.0, 1.0],
            dwell: 0.0,
        },
        _ => ArchetypeDefaults {
            // "sky_hush" (the validated fallback; callers only reach this arm after
            // checking membership in FIELD_ARCHETYPES).
            radius: FieldRadius::Room,
            inner: 0.0,
            falloff: Falloff::Smooth,
            occlude: false,
            cone: None,
            base: 0.0,
            priority: 1,
            phase: [1.0, 1.0, 1.0, 1.0],
            dwell: 0.0,
        },
    }
}

const DEFAULT_ATTACK_S: f32 = 0.4;
const DEFAULT_RELEASE_S: f32 = 1.2;
const DEFAULT_DEPTH: f32 = 1.0;

/// splitmix64-style hash for the deterministic default `seed =` (matches the family
/// `boot_console::hash64` and `re_audio::synth::Lcg` already standardize on).
fn hash_name(name: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in name.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

fn parse_facing(s: &str) -> Option<[f32; 3]> {
    match s.trim() {
        "+x" => return Some([1.0, 0.0, 0.0]),
        "-x" => return Some([-1.0, 0.0, 0.0]),
        "+y" => return Some([0.0, 1.0, 0.0]),
        "-y" => return Some([0.0, -1.0, 0.0]),
        "+z" => return Some([0.0, 0.0, 1.0]),
        "-z" => return Some([0.0, 0.0, -1.0]),
        _ => {}
    }
    let (x, y, z) = parse_vec3(s)?;
    let len = (x * x + y * y + z * z).sqrt();
    (len > 1e-6).then_some([x / len, y / len, z / len])
}

fn parse_phase(s: &str) -> Option<[f32; 4]> {
    let parts: Vec<f32> = s
        .split_whitespace()
        .filter_map(|t| t.parse::<f32>().ok())
        .collect();
    if s.split_whitespace().count() != 4 || parts.len() != 4 {
        return None;
    }
    Some([parts[0], parts[1], parts[2], parts[3]])
}

fn parse_buses(s: &str) -> Option<u8> {
    let mut mask = 0u8;
    for tok in s.split_whitespace() {
        mask |= match tok {
            "amb" => 1,
            "mus" => 2,
            "sfx" => 4,
            "pos" => 8,
            _ => return None,
        };
    }
    (mask != 0).then_some(mask)
}

fn node_idx(level: &Level, name: &str) -> Option<usize> {
    level.node_index(name)
}

fn pipe_idx(level: &Level, a: &str, b: &str) -> Option<usize> {
    let ai = node_idx(level, a)?;
    let bi = node_idx(level, b)?;
    level.pipe_between(ai, bi)
}

/// Parses the `PowerSpec` grammar plus `fed:` / `read:` (§4.2 `bind`).
fn parse_bind(level: &Level, s: &str) -> Option<FieldBind> {
    let s = s.trim();
    if s == "none" {
        return Some(FieldBind::None);
    }
    if s == "door" {
        return Some(FieldBind::Door);
    }
    let (head, tail) = s.split_once(':')?;
    let tail = tail.trim();
    match head.trim() {
        "node" => node_idx(level, tail).map(FieldBind::Node),
        "pipe" => {
            let (a, b) = tail.split_once("->")?;
            pipe_idx(level, a.trim(), b.trim()).map(FieldBind::Pipe)
        }
        "budget" => tail
            .parse::<u32>()
            .ok()
            .filter(|&k| k >= 1)
            .map(FieldBind::Budget),
        "seq" => tail
            .parse::<u32>()
            .ok()
            .filter(|&k| k >= 1)
            .map(FieldBind::Seq),
        "fed" => node_idx(level, tail).map(FieldBind::Fed),
        "read" => tail
            .parse::<usize>()
            .ok()
            .filter(|&i| i < level.notes.len())
            .map(FieldBind::Read),
        _ => None,
    }
}

/// Parses the §2.0 `drive` grammar.
fn parse_drive(level: &Level, s: &str) -> Option<DriveSpec> {
    let s = s.trim();
    match s {
        "const" => return Some(DriveSpec::Const),
        "door_ready" => return Some(DriveSpec::DoorReady),
        "solved" => return Some(DriveSpec::Solved),
        "cycle_held" => return Some(DriveSpec::CycleHeld),
        "budget" => return Some(DriveSpec::Budget),
        "seq" => return Some(DriveSpec::Seq),
        "clarity" => return Some(DriveSpec::Clarity),
        "shards" => return Some(DriveSpec::Shards),
        _ => {}
    }
    let (head, tail) = s.split_once(':')?;
    let tail = tail.trim();
    match head.trim() {
        "lit" => node_idx(level, tail).map(DriveSpec::Lit),
        "fed" => node_idx(level, tail).map(DriveSpec::Fed),
        "starve" => node_idx(level, tail).map(DriveSpec::Starve),
        "live" => {
            let (a, b) = tail.split_once("->")?;
            pipe_idx(level, a.trim(), b.trim()).map(DriveSpec::Live)
        }
        "dead_draw" => {
            let (a, b) = tail.split_once("->")?;
            pipe_idx(level, a.trim(), b.trim()).map(DriveSpec::DeadDraw)
        }
        "ring" => {
            let mut idxs = Vec::new();
            for clause in tail.split(',').map(str::trim).filter(|c| !c.is_empty()) {
                let (a, b) = split_arrow(clause)?;
                idxs.push(pipe_idx(level, a.trim(), &b)?);
            }
            (!idxs.is_empty()).then_some(DriveSpec::Ring(idxs))
        }
        _ => None,
    }
}

/// Resolve every `Level::field_defs` entry into a [`ResolvedField`], in declared
/// order. Errors carry the offending key's line and name the field + key.
pub fn resolve(level: &Level) -> Result<Vec<ResolvedField>, (usize, String)> {
    let mut seen: HashSet<&str> = HashSet::new();
    let mut out = Vec::with_capacity(level.field_defs.len());
    for def in &level.field_defs {
        if !seen.insert(def.name.as_str()) {
            return Err((
                def.line,
                format!("[field {}] duplicate field name", def.name),
            ));
        }
        out.push(resolve_one(level, def)?);
    }
    Ok(out)
}

#[allow(clippy::too_many_lines)]
fn resolve_one(level: &Level, def: &FieldDef) -> Result<ResolvedField, (usize, String)> {
    let name = def.name.clone();
    let mut archetype: Option<String> = None;
    let mut at: Option<[f32; 3]> = None;
    let mut region: Option<([f32; 3], [f32; 3])> = None;
    let mut follow: Option<(String, String)> = None;
    let mut radius: Option<FieldRadius> = None;
    let mut inner: Option<f32> = None;
    let mut falloff: Option<Falloff> = None;
    let mut occlude: Option<bool> = None;
    let mut facing: Option<[f32; 3]> = None;
    let mut cone_deg: Option<f32> = None;
    let mut depth = DEFAULT_DEPTH;
    let mut bind = FieldBind::None;
    let mut invert = false;
    let mut drive = DriveSpec::Const;
    let mut base: Option<f32> = None;
    let mut phase: Option<[f32; 4]> = None;
    let mut dwell: Option<f32> = None;
    let mut idle_boost = false;
    let mut buses = ALL_BUSES;
    let mut priority: Option<u8> = None;
    let mut attack = DEFAULT_ATTACK_S;
    let mut release = DEFAULT_RELEASE_S;
    let mut variant: Option<String> = None;
    let mut seed: Option<u64> = None;
    // §4.1 host-implied default drive; overridden by an authored `drive =`.
    let mut drive_authored = false;
    // §4.1 host-implied shape for `Node`/`Pipe`/`Door` (the module doc's
    // scope note): set BEFORE the key loop so an authored `at`/`region`/
    // `follow` — impossible for these three today since `attach_inline`
    // rejects positional inline knobs, but kept general — would still win.
    let mut shape_override: Option<FieldShape> = None;
    match &def.host {
        FieldHost::Node(idx) => {
            if let Some(n) = level.nodes.get(*idx) {
                at = Some([n.x as f32, n.y as f32, n.z as f32]);
            }
        }
        FieldHost::Pipe(idx) => {
            if let Some(p) = level.pipes.get(*idx) {
                let points: Vec<[f32; 3]> = p
                    .path
                    .iter()
                    .map(|&(x, y, z)| [x as f32, y as f32, z as f32])
                    .collect();
                shape_override = Some(FieldShape::Capsule { points });
            }
        }
        FieldHost::Door => {
            at = Some([
                level.door.x as f32,
                level.door.y as f32,
                level.door.z as f32,
            ]);
        }
        _ => {}
    }

    let herr = |line: usize, msg: String| (line, format!("[field {name}] {msg}"));

    for key in &def.keys {
        let (k, v, line) = (key.key.as_str(), key.value.as_str(), key.line);
        match k {
            "kind" => {
                let kv = v.trim();
                if !FIELD_ARCHETYPES.contains(&kv) {
                    return Err(herr(line, format!("kind: unknown archetype '{kv}'")));
                }
                archetype = Some(kv.to_string());
            }
            "at" => {
                let (x, y, z) =
                    parse_vec3(v).ok_or_else(|| herr(line, "at: expected 'x, y, z'".into()))?;
                at = Some([x, y, z]);
            }
            "region" => {
                let (a, b) =
                    parse_region(v).ok_or_else(|| herr(line, "region: bad format".into()))?;
                let min = [
                    a.0.min(b.0) as f32,
                    a.1.min(b.1) as f32,
                    a.2.min(b.2) as f32,
                ];
                let max = [
                    a.0.max(b.0) as f32,
                    a.1.max(b.1) as f32,
                    a.2.max(b.2) as f32,
                ];
                region = Some((min, max));
            }
            "follow" => {
                let rest = v
                    .trim()
                    .strip_prefix("pipe:")
                    .ok_or_else(|| herr(line, "follow: expected 'pipe:A->B'".into()))?;
                let (a, b) = split_arrow(rest)
                    .ok_or_else(|| herr(line, "follow: expected 'pipe:A->B'".into()))?;
                follow = Some((a, b));
            }
            "radius" => {
                let vt = v.trim();
                if vt == "room" {
                    radius = Some(FieldRadius::Room);
                } else {
                    let r: f32 = vt
                        .parse()
                        .map_err(|_| herr(line, "radius: not a number or 'room'".into()))?;
                    if !(r.is_finite() && r > 0.0) {
                        return Err(herr(line, "radius: must be > 0".into()));
                    }
                    radius = Some(FieldRadius::Value(r));
                }
            }
            "inner" => {
                let r: f32 = v
                    .trim()
                    .parse()
                    .map_err(|_| herr(line, "inner: not a number".into()))?;
                if !(r.is_finite() && r >= 0.0) {
                    return Err(herr(line, "inner: must be >= 0".into()));
                }
                inner = Some(r);
            }
            "falloff" => {
                falloff = Some(Falloff::parse(v).ok_or_else(|| {
                    herr(
                        line,
                        format!(
                            "falloff: unknown '{}' (want smooth|linear|inverse|hard)",
                            v.trim()
                        ),
                    )
                })?);
            }
            "occlude" => {
                occlude = Some(match v.trim() {
                    "yes" => true,
                    "no" => false,
                    other => {
                        return Err(herr(
                            line,
                            format!("occlude: unknown '{other}' (want yes|no)"),
                        ))
                    }
                });
            }
            "facing" => {
                facing = Some(
                    parse_facing(v).ok_or_else(|| herr(line, "facing: bad direction".into()))?,
                );
            }
            "cone" => {
                let c: f32 = v
                    .trim()
                    .parse()
                    .map_err(|_| herr(line, "cone: not a number".into()))?;
                if !(c.is_finite() && (1.0..=180.0).contains(&c)) {
                    return Err(herr(line, "cone: half-angle must be 1..=180".into()));
                }
                cone_deg = Some(c);
            }
            "depth" => {
                let d: f32 = v
                    .trim()
                    .parse()
                    .map_err(|_| herr(line, "depth: not a number".into()))?;
                if !(0.0..=1.0).contains(&d) {
                    return Err(herr(line, "depth: must be 0..=1".into()));
                }
                depth = d;
            }
            "bind" => {
                bind = parse_bind(level, v)
                    .ok_or_else(|| herr(line, format!("bind: '{}' does not resolve", v.trim())))?;
            }
            "invert" => {
                invert = match v.trim() {
                    "yes" => true,
                    "no" => false,
                    other => {
                        return Err(herr(
                            line,
                            format!("invert: unknown '{other}' (want yes|no)"),
                        ))
                    }
                };
            }
            "drive" => {
                drive = parse_drive(level, v)
                    .ok_or_else(|| herr(line, format!("drive: '{}' does not resolve", v.trim())))?;
                drive_authored = true;
            }
            "base" => {
                let b: f32 = v
                    .trim()
                    .parse()
                    .map_err(|_| herr(line, "base: not a number".into()))?;
                if !(0.0..=1.0).contains(&b) {
                    return Err(herr(line, "base: must be 0..=1".into()));
                }
                base = Some(b);
            }
            "phase" => {
                phase = Some(
                    parse_phase(v).ok_or_else(|| herr(line, "phase: expected 4 numbers".into()))?,
                );
            }
            "dwell" => {
                let d: f32 = v
                    .trim()
                    .parse()
                    .map_err(|_| herr(line, "dwell: not a number".into()))?;
                if !(d.is_finite() && d >= 0.0) {
                    return Err(herr(line, "dwell: must be >= 0".into()));
                }
                dwell = Some(d);
            }
            "idle_boost" => {
                idle_boost = match v.trim() {
                    "yes" => true,
                    "no" => false,
                    other => {
                        return Err(herr(
                            line,
                            format!("idle_boost: unknown '{other}' (want yes|no)"),
                        ))
                    }
                };
            }
            "buses" => {
                buses = parse_buses(v).ok_or_else(|| {
                    herr(line, "buses: expected a subset of 'amb mus sfx pos'".into())
                })?;
            }
            "priority" => {
                let p: u32 = v
                    .trim()
                    .parse()
                    .map_err(|_| herr(line, "priority: not a number".into()))?;
                if p > 9 {
                    return Err(herr(line, "priority: must be 0..=9".into()));
                }
                priority = Some(p as u8);
            }
            "attack" => {
                attack = v
                    .trim()
                    .parse()
                    .map_err(|_| herr(line, "attack: not a number".into()))?;
                if !(attack.is_finite() && attack >= 0.0) {
                    return Err(herr(line, "attack: must be >= 0".into()));
                }
            }
            "release" => {
                release = v
                    .trim()
                    .parse()
                    .map_err(|_| herr(line, "release: not a number".into()))?;
                if !(release.is_finite() && release >= 0.0) {
                    return Err(herr(line, "release: must be >= 0".into()));
                }
            }
            "variant" => variant = Some(v.trim().to_string()),
            "seed" => {
                seed = Some(
                    v.trim()
                        .parse()
                        .map_err(|_| herr(line, "seed: not a u64".into()))?,
                );
            }
            _ => unreachable!("is_known_key already gated this key"),
        }
    }

    let archetype = archetype.ok_or_else(|| herr(def.line, "missing 'kind ='".to_string()))?;
    let defaults = archetype_defaults(&archetype);

    // §4.1 host table's "implied default drive" column, for the three host
    // kinds this pass resolves inline shape for. An authored `drive =`
    // (`drive_authored`) always wins.
    if !drive_authored {
        let implied = match (&def.host, archetype.as_str()) {
            (FieldHost::Node(idx), "threshold_strain") => Some(DriveSpec::Starve(*idx)),
            (FieldHost::Node(idx), _) => Some(DriveSpec::Lit(*idx)),
            (FieldHost::Pipe(idx), "bit_rot") => Some(DriveSpec::DeadDraw(*idx)),
            (FieldHost::Pipe(idx), "live_wire") => Some(DriveSpec::Live(*idx)),
            (FieldHost::Door, _) => Some(DriveSpec::DoorReady),
            _ => None,
        };
        if let Some(d) = implied {
            drive = d;
        }
    }

    let radius = radius.unwrap_or(defaults.radius);
    let is_room = matches!(radius, FieldRadius::Room);

    // `re_runtime::sound_fields` assumes "Room radius implies Room shape" (it branches
    // on `field.shape == FieldShape::Room`, not on `field.radius`). An inline-attached
    // host (Node/Pipe/Door) pre-seeds `shape_override` to a point shape *before* this
    // function runs, independently of `radius`. Some archetypes (e.g.
    // `recognition_narrow`) default to `radius = FieldRadius::Room`, so an inline
    // attach with no explicit `radius =` knob would silently produce shape=Sphere,
    // radius=Room — violating that invariant. Caught here at authoring time: the
    // knob must override radius to a concrete value (e.g. `radius=5`, matching the
    // archetype doc's "sphere r5 on the host") before it can be hosted inline.
    if is_room && shape_override.is_some() {
        return Err(herr(
            def.line,
            "'radius = room' (the default for this archetype, or authored explicitly) \
             cannot be combined with an inline-attached host; the host implies a point \
             shape. Author an explicit 'radius = <value>' knob, or use a standalone \
             '[field]' section instead"
                .to_string(),
        ));
    }

    let host_count =
        usize::from(at.is_some()) + usize::from(region.is_some()) + usize::from(follow.is_some());
    if host_count > 1 {
        return Err(herr(
            def.line,
            "at most one of 'at', 'region' or 'follow' may be set".to_string(),
        ));
    }
    let shape = if let Some(s) = shape_override {
        s
    } else if is_room {
        if host_count != 0 {
            return Err(herr(
                def.line,
                "'radius = room' takes no 'at'/'region'/'follow'".to_string(),
            ));
        }
        FieldShape::Room
    } else if let Some(p) = at {
        FieldShape::Sphere { at: p }
    } else if let Some((min, max)) = region {
        FieldShape::BoxShape { min, max }
    } else if let Some((a, b)) = follow {
        let idx = pipe_idx(level, &a, &b)
            .ok_or_else(|| herr(def.line, format!("follow: pipe '{a}->{b}' does not exist")))?;
        let points: Vec<[f32; 3]> = level.pipes[idx]
            .path
            .iter()
            .map(|&(x, y, z)| [x as f32, y as f32, z as f32])
            .collect();
        FieldShape::Capsule { points }
    } else {
        return Err(herr(
            def.line,
            "exactly one of 'at', 'region' or 'follow' is required (unless 'radius = room')"
                .to_string(),
        ));
    };

    let cone = match (facing, cone_deg.or(defaults.cone)) {
        (Some(dir), Some(half)) => Some((dir, half)),
        (None, Some(_)) if cone_deg.is_some() => {
            return Err(herr(def.line, "'cone' requires 'facing'".to_string()));
        }
        (Some(_), None) => None,
        _ => None,
    };
    if cone_deg.is_some() && facing.is_none() {
        return Err(herr(def.line, "'cone' requires 'facing'".to_string()));
    }

    Ok(ResolvedField {
        name,
        archetype,
        shape,
        radius,
        inner: inner.unwrap_or(defaults.inner),
        falloff: falloff.unwrap_or(defaults.falloff),
        occlude: occlude.unwrap_or(defaults.occlude),
        cone,
        depth,
        bind,
        invert,
        drive,
        base: base.unwrap_or(defaults.base),
        phase: phase.unwrap_or(defaults.phase),
        dwell: dwell.unwrap_or(defaults.dwell),
        idle_boost,
        buses,
        priority: priority.unwrap_or(defaults.priority),
        attack,
        release,
        variant,
        seed: seed.unwrap_or_else(|| hash_name(&def.name)),
        line: def.line,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{load_string, ParseError};

    const HEADER: &str = "\
[meta]
name = test
[room]
size = 20 20 10
[pedestal]
center = 1,1,1
volume = 1 1 1
[door]
pos = 5 5 1
[player]
spawn = 2 2 1
";

    fn load(extra: &str) -> Result<Level, ParseError> {
        load_string(&format!("{HEADER}{extra}"), "test.lvl")
    }

    #[test]
    fn a_level_with_no_field_section_has_no_fields() {
        let lvl = load("").unwrap();
        assert!(lvl.fields().is_empty());
    }

    #[test]
    fn minimal_sphere_field_parses_with_archetype_defaults() {
        let lvl = load("[field river]\nkind = lethe_wash\nat = 4, 9, 1\n").unwrap();
        assert_eq!(lvl.fields().len(), 1);
        let f = &lvl.fields()[0];
        assert_eq!(f.archetype, "lethe_wash");
        assert_eq!(
            f.shape,
            FieldShape::Sphere {
                at: [4.0, 9.0, 1.0]
            }
        );
        assert_eq!(f.radius, FieldRadius::Value(6.0));
        assert!((f.inner - 1.5).abs() < 1e-6);
        assert_eq!(f.falloff, Falloff::Smooth);
        assert!(f.occlude);
        assert_eq!(f.priority, 3);
        assert!((f.depth - 1.0).abs() < 1e-6);
        assert_eq!(f.bind, FieldBind::None);
        assert_eq!(f.drive, DriveSpec::Const);
    }

    #[test]
    fn box_region_field_parses() {
        let lvl = load("[field river]\nkind = lethe_wash\nregion = (4,9,1)-(14,11,1)\n").unwrap();
        let f = &lvl.fields()[0];
        assert_eq!(
            f.shape,
            FieldShape::BoxShape {
                min: [4.0, 9.0, 1.0],
                max: [14.0, 11.0, 1.0]
            }
        );
    }

    #[test]
    fn radius_room_field_needs_no_position() {
        let lvl = load("[field hush]\nkind = sky_hush\nradius = room\n").unwrap();
        let f = &lvl.fields()[0];
        assert_eq!(f.shape, FieldShape::Room);
        assert_eq!(f.radius, FieldRadius::Room);
    }

    #[test]
    fn follow_pipe_resolves_to_the_expanded_path() {
        let lvl = load(
            "[node a]\npos = 2 12 2\n[node b]\npos = 9 6 2\n\
             [pipe a->b]\npath = (2,12,2) (9,12,2) (9,6,2)\n\
             [field rack_hold]\nkind = mnemos_hold\nfollow = pipe:a->b\n",
        )
        .unwrap();
        let f = &lvl.fields()[0];
        match &f.shape {
            FieldShape::Capsule { points } => assert!(!points.is_empty()),
            other => panic!("expected Capsule, got {other:?}"),
        }
    }

    #[test]
    fn every_key_of_the_full_control_form_parses() {
        let lvl = load(
            "[node growth]\npos = 4 4 2\n\
             [field strain]\n\
             kind = threshold_strain\n\
             at = 4, 4, 2\n\
             radius = 5.0\n\
             inner = 1.0\n\
             falloff = smooth\n\
             facing = +x\n\
             cone = 30\n\
             occlude = yes\n\
             depth = 0.7\n\
             bind = node:growth\n\
             invert = yes\n\
             drive = starve:growth\n\
             base = 0.2\n\
             phase = 0.6 1 1 1\n\
             dwell = 1.5\n\
             idle_boost = yes\n\
             buses = amb mus\n\
             priority = 8\n\
             attack = 0.3\n\
             release = 1.0\n\
             variant = memory\n\
             seed = 42\n",
        )
        .unwrap();
        let f = &lvl.fields()[0];
        assert_eq!(f.radius, FieldRadius::Value(5.0));
        assert!((f.inner - 1.0).abs() < 1e-6);
        assert_eq!(f.falloff, Falloff::Smooth);
        assert_eq!(f.cone, Some(([1.0, 0.0, 0.0], 30.0)));
        assert!(f.occlude);
        assert!((f.depth - 0.7).abs() < 1e-6);
        assert_eq!(f.bind, FieldBind::Node(0));
        assert!(f.invert);
        assert_eq!(f.drive, DriveSpec::Starve(0));
        assert!((f.base - 0.2).abs() < 1e-6);
        assert_eq!(f.phase, [0.6, 1.0, 1.0, 1.0]);
        assert!((f.dwell - 1.5).abs() < 1e-6);
        assert!(f.idle_boost);
        assert_eq!(f.buses, 0b0011);
        assert_eq!(f.priority, 8);
        assert!((f.attack - 0.3).abs() < 1e-6);
        assert!((f.release - 1.0).abs() < 1e-6);
        assert_eq!(f.variant.as_deref(), Some("memory"));
        assert_eq!(f.seed, 42);
    }

    #[test]
    fn bind_fed_and_read_parse() {
        let lvl = load(
            "[node face_07]\npos = 1 1 2\n\
             [note system_log n1]\nAn ordinary note.\n\
             [field wash]\nkind = lethe_wash\nat = 1,1,2\nbind = fed:face_07\ninvert = yes\n\
             [field tail]\nkind = tail_f\nat = 2,2,2\nbind = read:0\n",
        )
        .unwrap();
        assert_eq!(lvl.fields()[0].bind, FieldBind::Fed(0));
        assert!(lvl.fields()[0].invert);
        assert_eq!(lvl.fields()[1].bind, FieldBind::Read(0));
    }

    #[test]
    fn drive_ring_resolves_multiple_pipes() {
        let lvl = load(
            "[node rec]\npos = 1 1 2\n[node mem]\npos = 2 1 2\n\
             [pipe rec->mem]\n[pipe mem->rec]\n\
             [field loop]\nkind = mnemos_hold\nat = 1,1,2\ndrive = ring:rec->mem,mem->rec\n",
        )
        .unwrap();
        match &lvl.fields()[0].drive {
            DriveSpec::Ring(idxs) => assert_eq!(idxs.len(), 2),
            other => panic!("expected Ring, got {other:?}"),
        }
    }

    #[test]
    fn unknown_archetype_is_a_line_numbered_error() {
        let err = load("[field x]\nkind = not_a_real_one\nat = 1,1,1\n").unwrap_err();
        assert_eq!(err.line, 13);
        assert!(err.msg.contains("unknown archetype"), "{}", err.msg);
    }

    #[test]
    fn unknown_knob_is_a_line_numbered_error() {
        let err = load("[field x]\nkind = lethe_wash\nat = 1,1,1\nnot_a_key = 3\n").unwrap_err();
        assert_eq!(err.line, 15);
    }

    #[test]
    fn duplicate_field_name_is_an_error() {
        let err = load(
            "[field x]\nkind = lethe_wash\nat = 1,1,1\n[field x]\nkind = bit_rot\nat = 2,2,1\n",
        )
        .unwrap_err();
        assert!(err.msg.contains("duplicate"), "{}", err.msg);
    }

    #[test]
    fn follow_to_a_missing_pipe_is_an_error() {
        let err = load("[field x]\nkind = mnemos_hold\nfollow = pipe:a->b\n").unwrap_err();
        assert!(err.msg.contains("does not exist"), "{}", err.msg);
    }

    #[test]
    fn bind_to_a_missing_node_is_an_error() {
        let err =
            load("[field x]\nkind = lethe_wash\nat = 1,1,1\nbind = node:ghost\n").unwrap_err();
        assert!(err.msg.contains("does not resolve"), "{}", err.msg);
    }

    #[test]
    fn more_than_one_of_at_region_follow_is_an_error() {
        let err = load("[field x]\nkind = lethe_wash\nat = 1,1,1\nregion = (0,0,0)-(1,1,1)\n")
            .unwrap_err();
        assert!(err.msg.contains("at most one"), "{}", err.msg);
    }

    #[test]
    fn missing_position_without_radius_room_is_an_error() {
        let err = load("[field x]\nkind = lethe_wash\n").unwrap_err();
        assert!(err.msg.contains("exactly one"), "{}", err.msg);
    }

    #[test]
    fn radius_out_of_range_is_an_error() {
        let err = load("[field x]\nkind = lethe_wash\nat = 1,1,1\nradius = -3\n").unwrap_err();
        assert!(err.msg.contains("radius"), "{}", err.msg);
    }

    #[test]
    fn round_trip_load_write_load_every_key_of_the_full_form() {
        let lvl = load(
            "[node growth]\npos = 4 4 2\n\
             [field strain]\n\
             kind = threshold_strain\n\
             at = 4, 4, 2\n\
             radius = 5.0\n\
             inner = 1.0\n\
             falloff = smooth\n\
             occlude = yes\n\
             depth = 0.7\n\
             bind = node:growth\n\
             invert = yes\n\
             drive = starve:growth\n\
             base = 0.2\n\
             phase = 0.6 1 1 1\n\
             dwell = 1.5\n\
             idle_boost = yes\n\
             priority = 8\n\
             variant = memory\n\
             seed = 42\n",
        )
        .unwrap();
        let text = re_voximport_reexport_to_lvl_text(&lvl);
        let lvl2 = load_string(&text, "roundtrip.lvl").unwrap();
        assert_eq!(lvl.fields(), lvl2.fields());
    }

    // The round-trip test above needs the writer, which lives in `re_voximport` (a
    // downstream crate) to avoid a `re_level -> re_voximport` dependency; this local
    // helper re-implements ONLY the `[field]` emission (byte-for-byte what
    // `re_voximport::to_lvl_text` does) so the test can run inside `re_level` without
    // that dependency. `re_voximport`'s own suite has the real round-trip test against
    // its actual writer (§7(b)).
    fn re_voximport_reexport_to_lvl_text(level: &Level) -> String {
        let mut s = String::new();
        s.push_str("[meta]\nname = roundtrip\n[room]\nsize = 20 20 10\n[pedestal]\ncenter = 1,1,1\nvolume = 1 1 1\n[door]\npos = 5 5 1\n[player]\nspawn = 2 2 1\n");
        for n in &level.nodes {
            let _ = std::fmt::Write::write_fmt(
                &mut s,
                format_args!("[node {}]\npos = {} {} {}\n", n.name, n.x, n.y, n.z),
            );
        }
        for def in &level.field_defs {
            let _ = std::fmt::Write::write_fmt(&mut s, format_args!("[field {}]\n", def.name));
            for k in &def.keys {
                let _ =
                    std::fmt::Write::write_fmt(&mut s, format_args!("{} = {}\n", k.key, k.value));
            }
        }
        s
    }

    // ── §4.1 inline host-attach: [node]/[pipe]/[door] ──────────────────────

    #[test]
    fn inline_field_on_a_node_takes_its_position_and_implied_lit_drive() {
        let lvl = load(
            "[node dormant]\npos = 4 4 2\nis_input = yes\n\
             field = phantom_hollow depth=0.5\n",
        )
        .unwrap();
        assert_eq!(lvl.fields().len(), 1);
        let f = &lvl.fields()[0];
        assert_eq!(f.archetype, "phantom_hollow");
        assert_eq!(
            f.shape,
            FieldShape::Sphere {
                at: [4.0, 4.0, 2.0]
            }
        );
        assert!((f.depth - 0.5).abs() < 1e-6);
        assert_eq!(f.drive, DriveSpec::Lit(0)); // not threshold_strain -> implied lit:N
        assert_eq!(lvl.field_defs[0].host, FieldHost::Node(0));
    }

    #[test]
    fn inline_threshold_strain_on_a_node_implies_starve_drive() {
        let lvl = load("[node growth]\npos = 4 4 2\nfield = threshold_strain\n").unwrap();
        assert_eq!(lvl.fields()[0].drive, DriveSpec::Starve(0));
    }

    #[test]
    fn an_explicit_inline_drive_overrides_the_implied_one() {
        let lvl =
            load("[node growth]\npos = 4 4 2\nfield = threshold_strain drive=const\n").unwrap();
        assert_eq!(lvl.fields()[0].drive, DriveSpec::Const);
    }

    #[test]
    fn inline_field_on_a_pipe_is_a_capsule_with_implied_dead_draw_drive() {
        let lvl = load(
            "[node a]\npos = 2 12 2\n[node b]\npos = 9 6 2\n\
             [pipe a->b]\npath = (2,12,2) (5,12,2) (9,12,2) (9,6,2)\n\
             field = bit_rot depth=0.6\n",
        )
        .unwrap();
        let f = &lvl.fields()[0];
        assert_eq!(f.archetype, "bit_rot");
        assert!((f.depth - 0.6).abs() < 1e-6);
        match &f.shape {
            FieldShape::Capsule { points } => assert!(!points.is_empty()),
            other => panic!("expected Capsule, got {other:?}"),
        }
        assert_eq!(f.drive, DriveSpec::DeadDraw(0));
        assert_eq!(lvl.field_defs[0].host, FieldHost::Pipe(0));
    }

    #[test]
    fn inline_live_wire_on_a_pipe_implies_live_drive() {
        let lvl = load(
            "[node a]\npos = 2 12 2\n[node b]\npos = 9 6 2\n\
             [pipe a->b]\npath = (2,12,2) (9,12,2) (9,6,2)\n\
             field = live_wire\n",
        )
        .unwrap();
        assert_eq!(lvl.fields()[0].drive, DriveSpec::Live(0));
    }

    #[test]
    fn inline_field_on_the_door_is_a_sphere_at_the_door_with_door_ready_drive() {
        let lvl = load_string(
            "[meta]\nname = test\n[room]\nsize = 20 20 10\n\
             [pedestal]\ncenter = 1,1,1\nvolume = 1 1 1\n\
             [door]\npos = 17 6 1\nfield = exit_detune radius=7\n\
             [player]\nspawn = 2 2 1\n",
            "test.lvl",
        )
        .unwrap();
        let f = &lvl.fields()[0];
        assert_eq!(f.archetype, "exit_detune");
        assert_eq!(
            f.shape,
            FieldShape::Sphere {
                at: [17.0, 6.0, 1.0]
            }
        );
        assert_eq!(f.drive, DriveSpec::DoorReady);
        assert_eq!(lvl.field_defs[0].host, FieldHost::Door);
    }

    #[test]
    fn inline_field_unknown_archetype_is_a_line_numbered_error() {
        let err = load("[node growth]\npos = 4 4 2\nfield = not_a_real_one\n").unwrap_err();
        assert!(err.msg.contains("unknown archetype"), "{}", err.msg);
        assert!(err.msg.contains("[node]"), "{}", err.msg);
    }

    #[test]
    fn inline_field_positional_knob_is_rejected() {
        let err =
            load("[node growth]\npos = 4 4 2\nfield = threshold_strain at=1,1,1\n").unwrap_err();
        assert!(err.msg.contains("unknown knob 'at'"), "{}", err.msg);
    }

    #[test]
    fn inline_field_malformed_knob_is_a_parse_error() {
        let err =
            load("[node growth]\npos = 4 4 2\nfield = threshold_strain not_kv\n").unwrap_err();
        assert!(err.msg.contains("malformed knob"), "{}", err.msg);
    }

    #[test]
    fn a_second_inline_field_on_the_same_host_is_an_error() {
        let err = load("[node growth]\npos = 4 4 2\nfield = threshold_strain\nfield = bit_rot\n")
            .unwrap_err();
        assert!(err.msg.contains("at most one"), "{}", err.msg);
    }

    #[test]
    fn inline_attach_of_a_room_default_radius_archetype_is_an_error() {
        // `recognition_narrow` defaults to `radius = room`; a Node host pre-seeds a
        // point shape (Sphere at the node's position). Combining them would violate
        // `re_runtime::sound_fields`'s "Room radius implies Room shape" assumption.
        let err = load("[node REC]\npos = 4 4 2\nfield = recognition_narrow\n").unwrap_err();
        assert!(err.msg.contains("radius = room"), "{}", err.msg);
    }

    #[test]
    fn inline_attach_of_a_room_default_archetype_with_explicit_radius_is_allowed() {
        // Overriding `radius` to a concrete value resolves the conflict: shape
        // becomes the host's point shape (matching the archetype doc's "sphere r5
        // on the host") and radius is no longer Room.
        let lvl = load("[node REC]\npos = 4 4 2\nfield = recognition_narrow radius=5\n").unwrap();
        let f = &lvl.fields()[0];
        assert_eq!(
            f.shape,
            FieldShape::Sphere {
                at: [4.0, 4.0, 2.0]
            }
        );
        assert_eq!(f.radius, FieldRadius::Value(5.0));
        assert_eq!(f.drive, DriveSpec::Lit(0));
    }

    #[test]
    fn inline_and_standalone_fields_coexist_in_one_level() {
        let lvl = load(
            "[node growth]\npos = 4 4 2\nfield = threshold_strain\n\
             [field river]\nkind = lethe_wash\nat = 1,1,1\n",
        )
        .unwrap();
        assert_eq!(lvl.fields().len(), 2);
    }

    #[test]
    fn to_inline_line_reconstructs_the_authored_text_kind_first() {
        let lvl = load("[node growth]\npos = 4 4 2\nfield = threshold_strain depth=0.6 base=0.2\n")
            .unwrap();
        let def = &lvl.field_defs[0];
        let line = to_inline_line(def).expect("inline host has a reconstructable line");
        assert_eq!(line, "threshold_strain depth=0.6 base=0.2");
    }

    #[test]
    fn to_inline_line_is_none_for_a_standalone_field() {
        let lvl = load("[field river]\nkind = lethe_wash\nat = 1,1,1\n").unwrap();
        assert_eq!(to_inline_line(&lvl.field_defs[0]), None);
    }
}
