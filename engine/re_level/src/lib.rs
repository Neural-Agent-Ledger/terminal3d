// Test-module assert_eq! on f32 values are intentional exact-equal checks.
// Allow float_cmp only in test compilation units; lib code remains denied.
#![cfg_attr(test, allow(clippy::float_cmp))]
#![warn(missing_docs)]
//! `re_level` — the `.lvl` text-format parser.
//!
//! This is the **entry point for anyone working on the level format**. The
//! human-readable tour of the format (section-by-section, with examples) is
//! `docs/LEVEL_FORMAT.md` at the repo root; this crate doc is the other half —
//! how the parser itself is put together, so you can find your way around the
//! code and add to it. Where the two disagree, the parser is right.
//!
//! A Rust port of legacy `src/level.c`. Reads a level file (or an in-memory
//! string — the test seam, [`load_string`]) into a [`Level`] struct: room
//! dims, the node / pipe / part / emit puzzle network, pedestal + door +
//! player spawn, wall decals, and the optional effects / palette / note data.
//!
//! Engine-agnostic: no terminal, no rendering, no Bevy. The only dependency is
//! [`re_world`] for the shared [`CellKind`] surface registry and the
//! geometry-stamping seam ([`GeometrySpec`]). The parser captures the full
//! level data and can build the [`Map`] + camera spawn ([`Level::build_map`]);
//! the puzzle forward-pass (activation propagation) lives in `re_puzzle`,
//! consuming [`Level::nodes`]/[`Level::pipes`] as plain data.
//!
//! Unlike the C arena-of-pointers model, the Rust port uses owned `String`s
//! and `Vec`s — ownership is just the type system.
//!
//! # How a `.lvl` file becomes a [`Level`]
//!
//! [`load`] reads a file and calls [`load_string`], the crate's real entry
//! point (and the one every test uses directly, via an in-memory fixture
//! string). `load_string` is a two-pass design:
//!
//! 1. **`Parser::run`** walks the text ONE LINE AT A TIME. A `[section]` or
//!    `[section name]` header opens a new section via `open_section`, which
//!    matches the keyword to an internal `Section` enum and does any
//!    once-per-section setup (e.g. pushing a new [`Node`] onto
//!    [`Level::nodes`] when a `[node NAME]` header is seen). Every following
//!    `key = value` line is routed by section through `handle_kv`, which
//!    parses and validates that one key and either writes straight into the
//!    in-progress [`Level`] or into scratch fields on `Parser` for names that
//!    cannot be resolved yet (a `[door] target = A B` names nodes that may be
//!    declared *later* in the file).
//! 2. **`Parser::finalize`** runs once the whole file is read: it resolves
//!    every name→index reference the first pass deferred (door targets,
//!    pipe endpoints, sequence-lock observed nodes, hierarchy links, …),
//!    applies section-level defaults (e.g. an absent `storey_height`
//!    normalises to [`DEFAULT_STOREY_HEIGHT`]), and returns the finished
//!    [`Level`] or a [`ParseError`] naming the first thing that does not
//!    resolve.
//!
//! An unknown key inside a known section is collected onto
//! [`Level::unknown_keys`] rather than treated as an error — see
//! [`UnknownKey`]'s doc for why. An unknown *section* keyword, or a malformed
//! value in a section that does not tolerate one, is a line-numbered
//! [`ParseError`] via `Parser::err`.
//!
//! [`Level::build_map`] is a separate, later step: it turns the parsed data
//! into a [`re_world::Map`] (the raycaster's world), stamping node/pipe/prop
//! geometry, materials, lights and effects. Parsing never touches the map;
//! a [`Level`] is plain data that can be inspected, round-tripped by
//! `re_voximport`'s writer, or turned into a map, all without the others.
//!
//! # Adding a new `[section]`
//!
//! 1. Add data fields to [`Level`] (or a new sub-struct, following
//!    [`fields`]/[`props`] as examples of a section big enough to deserve its
//!    own module).
//! 2. Add the section keyword to `Section` and to `open_section`'s match arm
//!    (push any once-per-header setup here — see the `"node"` arm).
//! 3. Add the section's keys to `handle_kv`'s match on `Section`, parsing and
//!    validating each with a line-numbered [`ParseError`] on a bad value.
//! 4. If any key names another object by string (not yet resolvable during
//!    the first pass), stash the raw name on `Parser` and resolve it in
//!    `finalize`.
//! 5. Extend `docs/LEVEL_FORMAT.md`'s section table with the new section.
//! 6. If the section should round-trip, extend `re_voximport`'s writer to
//!    emit it — a section this crate can parse but the writer cannot emit
//!    is lost on every editor save.

use std::collections::HashSet;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use glam::Vec3;
use re_core::{
    DiagXY, Inset, InsetContent, InsetId, Line, LineMaterialId, LinePath, MaterialId, Overlay,
    RampDir, Region, ResolvedInset, ScaleRule, SelfSimilarParams, Shape, Zone,
};
use re_world::{
    ambient_to_wall_kind, stamp_geometry, stamp_holos, Atmosphere, BoldMode, CameraPose, CellKind,
    ColorJitter, ColorMode, DecalRegion, Distribution, DustVisibility, EmitterBlend, EmitterKind,
    EmitterOrigin, EmitterSpec, GeometrySpec, GlyphPick, GlyphSource, HeightFog, HoloObject,
    HoloShape, Map, MaterialAnim, MaterialAnimKind, MaterialDef, MaterialRecipe, MotionKind,
    ObjectCell, PointLight, TrailSpec,
};

// Re-export the Scalable Material Overlay primitives that ride on `Level.overlay`,
// plus the geometry-shape type used by `[cells]`. `re_level` owns the `.lvl`
// parse/serialize contract for these, so downstream tooling (`re_voximport`'s
// writer, `ts_editor`) names them through this crate — ONE source of truth — rather
// than reaching into `re_core` directly. `HoloShape` (the `[holo]` shape) is already
// re-exported below via `pub use re_world::*` patterns at use sites; it is named
// here for the same reason.
pub use re_core::{
    Inset as OverlayInset, InsetContent as OverlayInsetContent, Line as OverlayLine,
    Region as OverlayRegion, ScaleRule as OverlayScaleRule,
    SelfSimilarParams as OverlaySelfSimilarParams, Zone as OverlayZone,
};
// Re-export the pipe-cell `side_data` tag contract (R2 reactive materials). The
// `Map`'s `side_data` channel is OVERLOADED (`from_map`'s B2 doc explains how), and
// these helpers are how tooling (`re_voximport`'s writer + round-trip tests) tells a
// procedural pipe tag from an authored colour id. Re-exported through `re_level` so
// `.lvl` tooling has one source of truth without re-depending on `re_core`.
pub use re_core::{decode_pipe_tag, encode_pipe_side_data, PIPE_TAG_FLAG, PIPE_TAG_MASK};
// Re-export the trigger-field primitives (`Level.triggers`). `re_level` owns the
// `[trigger]` `.lvl` parse/serialize contract, so `re_voximport`'s writer names
// them through this crate (one source of truth) without re-depending on `re_core`.
pub use re_core::{Trigger, TriggerAction};

/// Pure staircase PREFAB GENERATOR — emits cell stamps for climbable straight /
/// spiral staircases (the multi-floor stairs design, §3a). The
/// single source of truth for "what a walkable staircase looks like"; its layout
/// matches the climbable run proven by `re_world`'s staircase unit tests.
pub mod assets;
/// **Sound fields** (`[field NAME]` sections): parsing, per-archetype defaults,
/// and name/index resolution. `re_runtime::sound_fields` reads the resolved
/// list via [`Level::fields`].
pub mod fields;
/// The PROP KIT (`[prop NAME]` sections, set-dressing §3): a pure generator that
/// expands 24 prefab kinds into cells, lines, holograms, emitters and lights.
pub mod props;
pub mod staircase;

pub use fields::{
    DriveSpec, Falloff, FieldBind, FieldDef, FieldHost, FieldKey, FieldRadius, FieldShape,
    ResolvedField, FIELD_ARCHETYPES,
};

/// Which external files a level names, and whether they are actually on disk —
/// the reporting surface behind `lvlcheck`'s asset gate and the game's load-time
/// "rendering without X" notice.
pub mod asset_audit;

pub use asset_audit::{AssetFault, AssetIssue, AssetKind, AssetRef};

/// One zoom-octave in log-radius — the period of the SelfSimilar infinite-zoom loop.
/// Duplicated from `re_render::fractal::LN2` (== `std::f32::consts::LN_2`): `re_level`
/// is engine-agnostic and must NOT depend on `re_render`, so the phase-band the
/// resolver computes is expressed in this crate's own copy of the constant. The two
/// are bit-identical (both `std::f32::consts::LN_2`), so the band `re_level` writes is
/// exactly the band `re_render` tests against.
const LN2: f32 = std::f32::consts::LN_2;

/// Documented NEUTRAL fallback colour for a resolved inset whose child scene yields no
/// representative colour (no `[material]`, no `[palette]` fog tint) OR whose child load
/// was SKIPPED by the cycle guard. A mid-grey `0x00_8888_88` — visible but unobtrusive,
/// so a broken/cyclic inset reference fails LOUD-ish (a flat grey patch) rather than
/// silently vanishing or painting garbage. Deterministic (a constant).
const INSET_FALLBACK_RGB: u32 = 0x0088_8888;

/// Default storey height (cells per storey) for a `.lvl` with no
/// `[room] storey_height` key — the legacy single-storey value (the
/// multi-floor stairs design, §4). [`load_string`] normalises an
/// absent key to this in `finalize`, and [`Level::storey_height`] reports it for
/// the unset (`0`) sentinel.
pub const DEFAULT_STOREY_HEIGHT: u32 = 4;

/// `side_data` tag for a `[cells]`-painted `O` that no puzzle entity backs.
///
/// `build_map` gives it to an `O` ([`CellKind::Object`]) cell with no `[node]` anchor
/// that is not the `[door]` cell (set-dressing F2). Without it such a cell carries tag `0`, which the renderer
/// cannot tell apart from node 0, so it rendered as a self-lit PUZZLE NODE and read
/// as a false neuron (L1's upstairs benches). `re_render::INERT_OBJECT_TAG` mirrors
/// this value (a cross-crate test pins the two) and shades such a cell as plain solid
/// geometry. The value is owned by `re_core` ([`re_core::INERT_OBJECT_TAG`]): it lives
/// in the reserved sentinel band, disjoint from the pipe namespace and from the
/// 1-based colour-palette / node-index range, and is never persisted by
/// [`CellGrid::from_map`] (it is recomputed on every load).
pub const INERT_OBJECT_TAG: u16 = re_core::INERT_OBJECT_TAG;

/// `side_data` tag for the exit-door `Object` anchor cell.
///
/// `build_map` used to stamp the door anchor with `tag = 0`, which is exactly node
/// 0's tag — so the door rendered as a duplicate of node 0, animated by node 0's
/// power state instead of shading as [`re_core::SurfaceKind::WallOther`]/`PUZZLE_DOOR` via
/// `re_render::DOOR_TAG_SENTINEL`. Mirrors [`re_core::DOOR_OBJECT_TAG`] (a
/// cross-crate test pins the two), lives in the reserved sentinel band disjoint from
/// the pipe namespace and the 1-based node-index range, and is never persisted by
/// [`CellGrid::from_map`] (recomputed on every load).
pub const DOOR_OBJECT_TAG: u16 = re_core::DOOR_OBJECT_TAG;

/// How far along the reader's gaze the note binder searches for a placard wall —
/// mirrors `re_runtime::notes::NOTE_WALL_SEARCH_MAX` (`re_level` cannot depend on
/// `re_runtime`). Used only to keep a note's host wall out of a `[zone]` skin.
const NOTE_HOST_GAZE_MAX: i32 = 24;

/// An `(x, y, z)` integer cell coordinate.
type Coord3 = (i32, i32, i32);

/// Which OBJECT KIND a [`HierRef`] refers to. The multi-instance, movable kinds
/// (`Node`/`Emitter`/`Vox`/`Trigger`) carry a meaningful index into the matching
/// `Level` `Vec`; the one-per-level singletons (`Door`/`Spawn`/`Pedestal`) ignore
/// the index (it is always `0`). This is the persisted half of the editor's
/// object-nesting feature — the kind+index pair is STABLE within a saved file's
/// deterministic object order, so a `[hierarchy]` link survives save→reload.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum HierKind {
    /// A `[node NAME]`, indexed into `Level::nodes`.
    Node,
    /// An `[emitter NAME]`, indexed into `Level::emitters`.
    Emitter,
    /// A `[vox NAME]` stamp, indexed into `Level::voxes`.
    Vox,
    /// A `[trigger NAME]`, indexed into `Level::triggers`.
    Trigger,
    /// The level's single `[door]` (singleton — index always `0`).
    Door,
    /// The level's single `[player] spawn` (singleton — index always `0`).
    Spawn,
    /// The level's single `[pedestal]` (singleton — index always `0`).
    Pedestal,
}

impl HierKind {
    /// The canonical `[hierarchy]` token for this kind (the exact inverse of
    /// [`from_token`](HierKind::from_token)). Shared by the writer + parser so a
    /// link round-trips losslessly.
    pub fn token(self) -> &'static str {
        match self {
            HierKind::Node => "node",
            HierKind::Emitter => "emitter",
            HierKind::Vox => "vox",
            HierKind::Trigger => "trigger",
            HierKind::Door => "door",
            HierKind::Spawn => "spawn",
            HierKind::Pedestal => "pedestal",
        }
    }

    /// Parse a `[hierarchy]` kind token; unknown ⇒ `None`.
    pub fn from_token(s: &str) -> Option<HierKind> {
        Some(match s {
            "node" => HierKind::Node,
            "emitter" => HierKind::Emitter,
            "vox" => HierKind::Vox,
            "trigger" => HierKind::Trigger,
            "door" => HierKind::Door,
            "spawn" => HierKind::Spawn,
            "pedestal" => HierKind::Pedestal,
            _ => return None,
        })
    }

    /// True for the one-per-level singleton kinds, whose [`HierRef`] index is
    /// always `0` (and not index-shifted on delete/duplicate).
    pub fn is_singleton(self) -> bool {
        matches!(self, HierKind::Door | HierKind::Spawn | HierKind::Pedestal)
    }
}

/// A uniform reference to one level object — `(kind, index)`. The persisted +
/// in-editor object-nesting primitive: a [`Level::hierarchy`] entry is a
/// `(child, parent)` pair of these. For singleton kinds the `index` is `0`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct HierRef {
    /// Which object kind this reference names.
    pub kind: HierKind,
    /// Index into the matching `Level` `Vec` (always `0` for a singleton kind).
    pub index: usize,
}

impl HierRef {
    /// Construct a reference (the index is forced to `0` for singleton kinds so
    /// every singleton ref is canonical).
    pub fn new(kind: HierKind, index: usize) -> HierRef {
        let index = if kind.is_singleton() { 0 } else { index };
        HierRef { kind, index }
    }
}

/// A parse error: the source name, 1-based line number, and a message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError {
    /// The source name (usually the file path, or `<string>` for the test seam).
    pub src: String,
    /// 1-based source line the error is reported against.
    pub line: usize,
    /// Human-readable message naming the section and the key.
    pub msg: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LEVEL: {}:{}: {}", self.src, self.line, self.msg)
    }
}

impl std::error::Error for ParseError {}

/// One `[node NAME]` — a neuron in the puzzle network.
#[derive(Clone, Debug, Default)]
pub struct Node {
    /// The section name (unique per level; what pipes/emit clauses/door
    /// targets refer to it by).
    pub name: String,
    /// World X coordinate.
    pub x: i32,
    /// World Y coordinate.
    pub y: i32,
    /// World Z coordinate.
    pub z: i32,
    /// The glyph the renderer draws for this node's cell.
    pub glyph: char,
    /// The primary (single-language) word pool this node's readout draws from
    /// while the network is in its ordinary, non-collapsed state.
    pub pool: Vec<String>,
    /// The corrupted/multilingual-glitch word pool this node's readout draws
    /// from instead, once the network reaches a collapsed/denial ending.
    pub pool_alt: Vec<String>,
    /// Index of the **anchor word** in [`Node::pool`] — the word this node contributes
    /// to the level's assembled "memory readout" phrase when it fires. Optional
    /// `[node] anchor = N`; defaults to `0` (the pool's dominant first word).
    /// Additive + back-compatible: levels without an `anchor` key keep `0`.
    /// Out-of-range indices fall back to `0` (or the node glyph) at
    /// phrase-assembly time, never panic.
    pub anchor: usize,
    /// The node's colour-law hue.
    pub color_rgb: u32,
    /// Driven by the level's `[inputs]` pattern, not by pipes.
    pub is_input: bool,
    /// Editor flag; emit clauses / door targets may name any node.
    pub is_output: bool,
    /// A non-input node fires when the sum of active incoming ON-pipe source
    /// activations is `>= threshold`.
    pub threshold: i32,
}

/// Pipe on/off state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PipeState {
    /// Not carrying activation (the default).
    #[default]
    Off,
    /// Carrying the source node's activation.
    On,
}

/// One `[pipe A->B]` — a wired connection. `path` holds the **expanded**
/// per-cell trace (waypoints joined x-then-y-then-z, like legacy `expand_path`).
#[derive(Clone, Debug, Default)]
pub struct Pipe {
    /// Source node name, as authored (`A` in `[pipe A->B]`).
    pub from: String,
    /// Destination node name, as authored (`B` in `[pipe A->B]`).
    pub to: String,
    /// Source node index, resolved from `from`.
    pub from_node: usize,
    /// Destination node index, resolved from `to`.
    pub to_node: usize,
    /// The pipe's authored initial on/off state.
    pub state: PipeState,
    /// `[pipe] density =` (default `1.0`): authored visual-density hint for
    /// how the renderer draws this pipe's trail.
    pub density: f32,
    /// The pipe's colour-law hue.
    pub color_rgb: u32,
    /// Expanded contiguous cell path (each step is one cell, 6-connected).
    pub path: Vec<Coord3>,
    /// Index into `path` of the toggle switch cell, or `None`.
    pub switch_cell: Option<usize>,
    /// Raw `[pipe] switch_at` coordinate as authored, before path resolution.
    /// Resolved against the expanded `path` in `finalize` (a second pass, like
    /// legacy `level.c`), which hard-errors if the cell is off-path. `None`
    /// means no `switch_at` was declared. Derived/serialized levels set
    /// `switch_cell` directly and leave this `None`.
    pub switch_at_raw: Option<Coord3>,
}

/// One `[part NAME]` — a voxel stamp for the pedestal output volume.
#[derive(Clone, Debug, Default)]
pub struct Part {
    /// The section name (unique per level; what emit clauses refer to it by).
    pub name: String,
    /// Mask width, in cells.
    pub w: i32,
    /// Mask height, in cells.
    pub h: i32,
    /// Mask depth, in cells.
    pub d: i32,
    /// `w*h*d` mask bytes (`1` = solid), index `x + w*(y + h*z)`.
    pub mask: Vec<u8>,
    /// X offset of the mask's origin from the pedestal's anchor.
    pub off_x: i32,
    /// Y offset of the mask's origin from the pedestal's anchor.
    pub off_y: i32,
    /// Z offset of the mask's origin from the pedestal's anchor.
    pub off_z: i32,
    /// The part's colour-law hue when lit.
    pub color_rgb: u32,
}

/// How a `[part]` LOOKS when its `[emit]` clause lights it, beyond its mask and colour:
/// the height each part cell fills and an optional two-tone top (a lit figure used to
/// be a stack of storey-tall columns, which this replaces with a shorter band).
///
/// Kept beside [`Part`] in [`Level::part_looks`] (same index; a missing entry is the
/// default) so `Part`'s shape — built field-by-field by the editor — is unchanged.
///
/// ```text
/// height   = full | half | F    # fraction of each part cell that is solid, from its
///                               # floor; default full (unchanged)
/// top      = #RRGGBB            # the top of each cell's solid band in a second colour
/// top_from = F                  # where that top starts, as a fraction of the band
///                               # (default 0.6 — the upper 40 %)
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PartLook {
    /// Solid fraction of each part cell, from its floor (`1.0` = the whole cell).
    pub height: f32,
    /// The two-tone top: `(rgb, from)` — the band from `from·height` up is `rgb`.
    pub top: Option<(u32, f32)>,
}

impl Default for PartLook {
    fn default() -> Self {
        PartLook {
            height: 1.0,
            top: None,
        }
    }
}

/// Default `top_from` of a two-tone part: the upper 40 % of the band is the top colour.
pub const PART_TOP_FROM_DEFAULT: f32 = 0.6;

/// Smallest `[part] height` accepted (a thinner band would be a sub-row sliver).
pub const PART_MIN_HEIGHT: f32 = 0.1;

/// One expanded emit clause: `source_node` lights `part`.
#[derive(Clone, Debug)]
pub struct EmitClause {
    /// Index into `Level::nodes` — the node whose firing lights `part`.
    pub source_node: usize,
    /// Index into `Level::parts` — the voxel stamp shown while `source_node` fires.
    pub part: usize,
}

/// `[pedestal]` bounding box (center cell + full extent).
#[derive(Clone, Copy, Debug, Default)]
pub struct Pedestal {
    /// Center X coordinate.
    pub cx: i32,
    /// Center Y coordinate.
    pub cy: i32,
    /// Center Z coordinate.
    pub cz: i32,
    /// Full width, in cells.
    pub w: i32,
    /// Full height, in cells.
    pub h: i32,
    /// Full depth, in cells.
    pub d: i32,
}

/// `[door]` win-condition anchor + the node indices that must all fire.
#[derive(Clone, Debug, Default)]
pub struct Door {
    /// World X coordinate of the door's anchor cell.
    pub x: i32,
    /// World Y coordinate of the door's anchor cell.
    pub y: i32,
    /// World Z coordinate of the door's anchor cell.
    pub z: i32,
    /// Node indices that must all activate to win (`[door] target = NAME...`).
    pub target_parts: Vec<usize>,
    /// Opt-in **inverted** targets (L12 `exit 0`): node indices that must read
    /// **0** (NOT fire) for the win, in addition to every `target_parts` node
    /// firing. Parsed from `[door] target_off = NAME...`. Empty ⇒ the ordinary
    /// win (a level with no `target_off` is byte-identical). `re_puzzle` consumes
    /// this as `re_puzzle::PuzzleState::target_off`.
    pub target_off: Vec<usize>,
    /// `[door] target_model` — an authored tag naming the model the door leads
    /// to, kept VERBATIM. INERT: nothing in the engine reads it (only
    /// `levels/00-test-load.lvl` carries one). Preserved so a load→save cannot
    /// delete it. Empty ⇒ the key was absent and the writer emits nothing.
    pub target_model: String,
}

/// One ordered target output-state in a [`SequenceSpec`] (the L11 sequence-lock
/// mechanic). `bits` is the resolved `(node_index, want_on)` constraint set: the
/// observed output nodes and the value each must hold for this step to match.
///
/// `re_puzzle` consumes this directly (it mirrors `re_puzzle::SeqPattern`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SequencePattern {
    /// `(node_index, want_on)` constraints. ALL must hold for a match.
    pub bits: Vec<(usize, bool)>,
}

/// The opt-in `[sequence]` win mode (L11 "Inner Circuit").
///
/// A level declares a fixed set of **observed** output nodes (`observe`) plus an
/// ordered list of `step` lines, each naming the subset of observed nodes that
/// must be ON at that step (every other observed node must be OFF). The player
/// must drive the network output to match each step **in order**; a missed step
/// resets only the current step, never the whole sequence (`re_puzzle`
/// `SequenceLock`). Empty `patterns` ⇒ inactive: the level uses the ordinary
/// single-target `[door]` win and is byte-identical to a level with no
/// `[sequence]` section.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SequenceSpec {
    /// The resolved ordered patterns (each step's `(node, want_on)` constraints
    /// over the observed window). Empty ⇒ the sequence-lock is inactive. This is
    /// what `re_puzzle` consumes; the win behaviour is driven entirely by this.
    pub patterns: Vec<SequencePattern>,
    /// The AUTHORED `observe = NAME...` window — the output node NAMES, in their
    /// declared order. Kept verbatim from the `.lvl` so the `[sequence]` section
    /// is losslessly re-emittable by `re_voximport`'s writer: the resolved
    /// `patterns` store only `(node_index, want_on)` and have discarded the
    /// names + the observe-window structure. Populated in
    /// [`finalize`](struct.Parser.html) alongside `patterns`; never read by the
    /// win check. Empty ⇒ the sequence-lock is inactive (no `[sequence]`).
    pub observe_names: Vec<String>,
    /// The AUTHORED `step = NAME...` rows — each a list of the observed node
    /// NAMES that must be ON at that step (every other observed node OFF), in
    /// declared order. The lossless authored counterpart to `patterns`: a step
    /// with an empty Vec is the valid "all observed nodes OFF" step. Used only by
    /// the writer to reproduce the exact `[sequence]` block; never by the win
    /// check. Length matches `patterns`.
    pub step_names: Vec<Vec<String>>,
}

/// `[player]` spawn pose.
#[derive(Clone, Copy, Debug, Default)]
pub struct PlayerSpawn {
    /// World X coordinate.
    pub x: i32,
    /// World Y coordinate.
    pub y: i32,
    /// World Z coordinate.
    pub z: i32,
    /// The `[player] facing` yaw column (`+y` 0, `+x` 120, `-y` 240, `-x` 360).
    pub facing_yaw_col: i32,
}

/// One `[wall_decal NAME]` region.
#[derive(Clone, Debug, Default)]
pub struct WallDecal {
    /// The section name (unique per level).
    pub name: String,
    /// Which surface kind (e.g. matrix-rain, circuit-trace) this decal renders as.
    pub surface_kind: CellKind,
    /// Region min X.
    pub x0: i32,
    /// Region min Y.
    pub y0: i32,
    /// Region min Z.
    pub z0: i32,
    /// Region max X.
    pub x1: i32,
    /// Region max Y.
    pub y1: i32,
    /// Region max Z.
    pub z1: i32,
    /// Authored visual-density hint for how thick the decal's effect renders.
    pub density: f32,
    /// The decal's leading-edge/head colour.
    pub color_head: u32,
    /// The decal's trailing-edge/tail colour.
    pub color_tail: u32,
}

/// One `[holo NAME]` transparent HOLOGRAM object: a bounding-box region + a
/// shape that selects which interior cells fill, stamped as `CellKind::Hologram`.
/// The renderer draws the resulting silhouette as a glowing wireframe; the body
/// stays see-through. Additive: a level with no `[holo]` section has none.
#[derive(Clone, Debug, Default)]
pub struct LevelHolo {
    /// The section name (unique per level).
    pub name: String,
    /// Which interior cells of the bounding box fill (box / pyramid / sphere).
    pub shape: HoloShape,
    /// Bounding box min X.
    pub x0: i32,
    /// Bounding box min Y.
    pub y0: i32,
    /// Bounding box min Z.
    pub z0: i32,
    /// Bounding box max X.
    pub x1: i32,
    /// Bounding box max Y.
    pub y1: i32,
    /// Bounding box max Z.
    pub z1: i32,
    /// Colour/look variant tag (0 = default blue).
    pub style: u16,
}

/// One `[portal NAME]` non-euclidean warp-portal PAIR: two wall-face openings
/// (`region_a` / `region_b`) linked by a rigid transform. On
/// [`build_map`](Level::build_map) each face is stamped [`CellKind::Portal`]
/// (solid to the raycaster, walkable for the player) and the reciprocal
/// `delta`/`delta_inv` pair is registered via [`Map::add_portal_pair`], so the
/// geometry math is shared with the engine and cannot diverge. Additive: a level
/// with no `[portal]` section has none and renders exactly as before.
///
/// Fields (spec §5):
/// * `region_a` / `region_b` — inclusive grid-cell boxes describing each
///   opening; each MUST be a flat wall face on a grid boundary.
/// * `rotate` — extra yaw (degrees) applied crossing A→B; B→A is the inverse.
///   `0` ⇒ the plain "walk straight, the world turned" warp.
/// * `skin` — optional material name tinting the opening (carried for the
///   renderer; `re_world` ignores it).
#[derive(Clone, Debug, Default)]
pub struct LevelPortal {
    /// The section name (unique per level).
    pub name: String,
    /// A-side region min X.
    pub ax0: i32,
    /// A-side region min Y.
    pub ay0: i32,
    /// A-side region min Z.
    pub az0: i32,
    /// A-side region max X.
    pub ax1: i32,
    /// A-side region max Y.
    pub ay1: i32,
    /// A-side region max Z.
    pub az1: i32,
    /// B-side region min X.
    pub bx0: i32,
    /// B-side region min Y.
    pub by0: i32,
    /// B-side region min Z.
    pub bz0: i32,
    /// B-side region max X.
    pub bx1: i32,
    /// B-side region max Y.
    pub by1: i32,
    /// B-side region max Z.
    pub bz1: i32,
    /// Extra yaw (degrees) applied crossing A→B.
    pub rotate: f32,
    /// Optional material name tinting the opening (empty ⇒ none).
    pub skin: String,
}

impl LevelPortal {
    /// This portal's A-side region as an inclusive cell box (for
    /// [`Map::add_portal_pair`]).
    pub fn region_a(&self) -> ((i32, i32, i32), (i32, i32, i32)) {
        (
            (self.ax0, self.ay0, self.az0),
            (self.ax1, self.ay1, self.az1),
        )
    }

    /// This portal's B-side region as an inclusive cell box.
    pub fn region_b(&self) -> ((i32, i32, i32), (i32, i32, i32)) {
        (
            (self.bx0, self.by0, self.bz0),
            (self.bx1, self.by1, self.bz1),
        )
    }
}

/// Which surface kind a stamped `[vox]` model renders as. The voxels are stamped
/// as that solid [`CellKind`]; the renderer then shades them with the bound
/// material / built-in for that surface. Default [`VoxMaterial::Solid`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum VoxMaterial {
    /// A plain opaque [`CellKind::WallSolid`] block (the default).
    #[default]
    Solid,
    /// A [`CellKind::WallTrace`] (circuit-trace) surface.
    Trace,
    /// A [`CellKind::WallMatrix`] (matrix-rain) surface.
    Matrix,
}

impl VoxMaterial {
    /// Parse an authored `material =` value; unknown ⇒ `None`.
    fn from_str_opt(s: &str) -> Option<VoxMaterial> {
        match s.trim() {
            "solid" | "wall_solid" | "brick" => Some(VoxMaterial::Solid),
            "trace" | "wall_trace" | "circuit" => Some(VoxMaterial::Trace),
            "matrix" | "wall_matrix" => Some(VoxMaterial::Matrix),
            _ => None,
        }
    }

    /// The canonical `material =` token for this variant — the EXACT inverse of
    /// the keyword `VoxMaterial::from_str_opt` parses, so a
    /// `[vox]` section written with this string re-parses to the same variant
    /// (the round-trip the editor save depends on). Public so `re_voximport`'s
    /// `write_vox` and the editor panel share one source of truth.
    pub fn as_str(self) -> &'static str {
        match self {
            VoxMaterial::Solid => "solid",
            VoxMaterial::Trace => "trace",
            VoxMaterial::Matrix => "matrix",
        }
    }

    /// The [`CellKind`] a stamped voxel becomes for this material.
    fn cell_kind(self) -> CellKind {
        match self {
            VoxMaterial::Solid => CellKind::WallSolid,
            VoxMaterial::Trace => CellKind::WallTrace,
            VoxMaterial::Matrix => CellKind::WallMatrix,
        }
    }
}

/// One `[zone_input NAME]` — a FLOOR-ZONE that drives an input node.
///
/// While the player's cell is inside the inclusive `(x0,y0,z0)-(x1,y1,z1)` box,
/// the named input node is held at `1`; the instant the player steps out it
/// returns to `0`. This is the "player-position-as-input" runtime feature (L5
/// "Soft-Footstep"): standing in the warm floor-zone fires `present = 1`.
///
/// Engine-agnostic: `re_level` only PARSES + validates the zone. The host runtime
/// (`re_runtime`) owns the per-frame "is the camera inside?" test and pushes the
/// resulting `0/1` into the live `re_puzzle::PuzzleState` (re-solving on a
/// change). The named node MUST exist and MUST be an input node — both are
/// enforced in `finalize`, so a malformed zone fails the load rather than
/// silently driving nothing.
///
/// Additive: a level with no `[zone_input]` section has none and behaves exactly
/// as before.
#[derive(Clone, Debug, Default)]
pub struct ZoneInput {
    /// The section name (unique per level).
    pub name: String,
    /// The input node (by name) this zone drives to `1` while occupied.
    pub node: String,
    /// Resolved index of `node` into [`Level::nodes`] (set in `finalize`).
    pub node_index: usize,
    /// Inclusive cell box (min/max corners, normalised in `finalize`).
    pub x0: i32,
    /// Box min Y.
    pub y0: i32,
    /// Box min Z.
    pub z0: i32,
    /// Box max X.
    pub x1: i32,
    /// Box max Y.
    pub y1: i32,
    /// Box max Z.
    pub z1: i32,
}

impl ZoneInput {
    /// True iff the integer cell `(x, y, z)` lies inside this zone's inclusive
    /// box. The corners are normalised in `finalize`, so a min/max swap in the
    /// authored `region` is harmless.
    #[inline]
    pub fn contains(&self, x: i32, y: i32, z: i32) -> bool {
        (self.x0..=self.x1).contains(&x)
            && (self.y0..=self.y1).contains(&y)
            && (self.z0..=self.z1).contains(&z)
    }
}

/// One `[vox NAME]` — a `.vox` voxel model stamped into the room as solid cells.
///
/// On [`build_map`](Level::build_map) the file is loaded via `dot_vox` (the SAME
/// loader the importer uses — no `.vox` parsing is reimplemented), each SOLID
/// voxel is downscaled by `scale` and offset by `offset`, and stamped as the
/// `material`'s solid [`CellKind`] (tagged so the renderer shades it). Additive:
/// a level with no `[vox]` section never loads anything. Out-of-room cells are
/// CLIPPED (skipped), so an over-large model never escapes the room.
#[derive(Clone, Debug, Default)]
pub struct LevelVox {
    /// The section name (unique per level).
    pub name: String,
    /// Path to the `.vox` file (relative to the process cwd, like the level path).
    pub file: String,
    /// Where the model's local `(0,0,0)` lands in the room (z-up).
    pub off_x: i32,
    /// Y offset (see [`off_x`](Self::off_x)).
    pub off_y: i32,
    /// Z offset (see [`off_x`](Self::off_x)).
    pub off_z: i32,
    /// Surface the stamped voxels render as.
    pub material: VoxMaterial,
    /// Integer downscale (`1` = full res; `2` = every other voxel; …). Min 1.
    pub scale: i32,
}

/// One `[texture NAME]` — a decoded image (PNG) the level surfaces a wall with.
///
/// On [`build_map`](Level::build_map) the `file` is decoded via the `image` crate
/// into a [`re_core::TextureImage`] (RGBA8, top-left origin) and pushed onto
/// [`Map::textures`], yielding a `tex_id` an `[material] recipe = image texture = NAME`
/// binds to. Only the FILE PATH is serialized (`re_voximport`'s writer) — never the
/// decoded buffer. A decode failure is non-fatal: the texture is skipped (no entry in
/// `Map::textures`) and reported by [`build_map_checked`](Level::build_map_checked),
/// exactly like a missing `[vox]` model. Additive: a level with no `[texture]` section
/// decodes nothing and renders byte-identically to today.
#[derive(Clone, Debug, Default)]
pub struct LevelTexture {
    /// Author-facing name an `[material] texture = NAME` references.
    pub name: String,
    /// Path to the image file (relative to the level dir, like `[vox] file`).
    pub file: String,
    /// `max_size = N` — the longest edge, in pixels, the decoded image is allowed
    /// to keep. A larger source is box-downsampled ON LOAD, preserving its aspect
    /// ratio, before it ever reaches [`re_core::TextureImage`].
    ///
    /// This is a MEMORY lever, not a look lever: an ASCII cell samples a handful
    /// of texels, so a 9428x3548 photo (~134 MB of RGBA in RAM) carries no more
    /// legible detail than its 1024-wide reduction (~1.6 MB) — see the engine
    /// temporal-stability spec §1.4. `None` (every level authored so far) keeps
    /// the image at its native size, byte-identical to before.
    pub max_size: Option<u32>,
    /// `height = <path>` — an OPTIONAL 8-bit RELIEF map for this texture, decoded
    /// through the SAME asset resolver and the SAME [`max_size`](Self::max_size) cap
    /// as `file`, converted to luminance, and (when its pixel dimensions differ from
    /// the colour image's) RESAMPLED to the colour image's dimensions so the two
    /// channels address the same texel. It rides into the renderer on the colour
    /// image, as [`re_core::TextureImage::height_map`].
    ///
    /// A height map alone changes NOTHING: the surface only becomes lit relief when a
    /// `[material]` bound to this texture also authors `relief = …`. `None` (every
    /// level authored before this key) is the historical flat, self-lit image.
    pub height: Option<String>,
    /// `frames = <path.gif>` — an ANIMATED texture: EVERY frame of the named GIF is
    /// decoded at load into a [`re_core::TextureSet`], and the shader indexes that set
    /// by the cast's `time`, so a wall plays a looping clip in ASCII.
    ///
    /// When set, the frame set's pixels come from THIS path and [`file`](Self::file)
    /// is not decoded (it stays the authored still/poster reference and round-trips
    /// unchanged). A `frames` file that holds only ONE frame is not an error — it
    /// yields a one-frame set, i.e. exactly a still texture.
    ///
    /// [`max_size`](Self::max_size) applies to EVERY frame: 68 frames of 500x213 RGBA
    /// is ~29 MB uncapped, which is the whole reason the cap landed before this key.
    /// `None` (every level authored before this key) decodes nothing extra.
    pub frames: Option<String>,
    /// `fps = <f32 > 0>` — the playback rate of the [`frames`](Self::frames) set,
    /// default [`re_core::TEXTURE_SET_DEFAULT_FPS`] (10). A non-finite or
    /// non-positive value is a LINE-NUMBERED parse error, never a silent default: a
    /// wall that quietly played at 10 fps when the author wrote `fps = 0` would be
    /// debugged in the renderer instead of in the level.
    ///
    /// Ignored (but preserved and round-tripped) when `frames` is unset — a still
    /// texture has nothing to play.
    pub fps: Option<f32>,
}

/// Optional `[sky]` image cubemap paths (cylinder + zenith/nadir caps).
///
/// Only FILE PATHS are stored here — the decoded [`re_core::TextureImage`] buffers are
/// produced at [`build_map`](Level::build_map) time via `decode_texture_file` (the same
/// helper `[texture]` uses) and handed to [`Map`] via its sky setters. A level with no
/// `[sky]` section keeps all three `None` and the procedural sky is unchanged
/// (byte-identical). Decode failures are non-fatal in `build_map` (skip + continue) and
/// surfaced as an error by [`build_map_checked`](Level::build_map_checked).
///
/// Keys (all optional):
/// * `cylinder = <path>` — 360° wraparound horizon strip.
/// * `cap_top = <path>` — zenith cap (above horizon).
/// * `cap_bottom = <path>` — nadir cap (below horizon, rarely needed).
#[derive(Clone, Debug, Default)]
pub struct SkyDef {
    /// Path to the cylinder image (360° horizon band).
    pub cylinder: Option<String>,
    /// Path to the zenith cap image.
    pub cap_top: Option<String>,
    /// Path to the nadir cap image (optional).
    pub cap_bottom: Option<String>,
    /// The parsed procedural sky PROGRAM — `base = ` / `layer = ` (repeatable) / `kind = ` / `seed = ` /
    /// `hz = ` keys, accumulated across the section's lines and assembled into a
    /// [`re_world::sky_spec::SkySpec`] by [`SkyDef::to_spec`]. `None` when no
    /// such key was authored (the section may still carry only image paths).
    pub base: Option<re_world::sky_spec::SkyBase>,
    /// Accumulated `layer = ` lines (repeatable), in authored order.
    pub layers: Vec<re_world::sky_spec::SkyLayerSpec>,
    /// `seed = ` for the procedural sky program.
    pub seed: Option<u32>,
    /// `hz = ` update rate for the procedural sky program.
    pub hz: Option<u32>,
}

impl SkyDef {
    /// Returns `true` when no sky path is set — the level uses procedural sky only.
    pub fn is_empty(&self) -> bool {
        self.cylinder.is_none() && self.cap_top.is_none() && self.cap_bottom.is_none()
    }

    /// Assemble the authored `base`/`layer`/`seed`/`hz` keys into a
    /// [`re_world::sky_spec::SkySpec`], or `None` when the section authored no
    /// procedural-sky key at all (image-only `[sky]`, or no `[sky]` section).
    pub fn to_spec(&self) -> Option<re_world::sky_spec::SkySpec> {
        if self.base.is_none() && self.layers.is_empty() {
            return None;
        }
        Some(re_world::sky_spec::SkySpec {
            base: self.base.unwrap_or_default(),
            layers: self.layers.clone(),
            seed: self.seed.unwrap_or(0),
            hz: self.hz.unwrap_or(12),
        })
    }
}

/// Optional `[effects]` ambient knobs (all-off default).
#[derive(Clone, Copy, Debug, Default)]
pub struct LevelEffects {
    /// `[effects] snow` — authored particle density (0..1) of the falling-snow
    /// emitter lattice. `0` = no snow (default).
    pub snow: f32,
    /// `[effects] steam` — whether the level's steam/haze emitters are active.
    pub steam: bool,
    /// `[effects] matrix` — authored density (0..1) of the matrix-rain effect.
    /// `0` = no matrix rain (default).
    pub matrix: f32,
    /// `[effects] glitch` — authored per-level visual-corruption density (0..1),
    /// the game's narrative decay timer. `0` = no glitch (default).
    pub glitch: f32,
}

/// Optional `[palette]` distance-fog override.
#[derive(Clone, Copy, Debug, Default)]
pub struct LevelPalette {
    /// Whether `[palette] fog_color` was authored.
    pub has_fog: bool,
    /// The distance-fog colour.
    pub fog_color: u32,
    /// Whether `[palette] fog_near`/`fog_far` were authored.
    pub has_fog_dist: bool,
    /// Distance at which fog begins.
    pub fog_near: f32,
    /// Distance at which fog reaches full strength.
    pub fog_far: f32,
}

/// The `[atmosphere]` section's LIGHTING + height-fog knobs, as authored.
///
/// The fog *distance* keys (`fog_color`/`fog_near`/`fog_far`) are deliberately
/// NOT here: `[atmosphere]` accepts them as aliases and writes them straight into
/// [`LevelPalette`], so there is exactly one place fog distance lives and old
/// `[palette]`-authored levels keep working untouched.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LevelAtmosphere {
    /// `key_dir = x, y, z` — direction TOWARD the key light (normalised later).
    pub key_dir: Option<Vec3>,
    /// `key_color = #RRGGBB`.
    pub key_color: Option<u32>,
    /// `ambient = f` in `[0,1]`.
    pub ambient: Option<f32>,
    /// `fog_height = base falloff`.
    pub fog_height: Option<HeightFog>,
    /// `postfx = NAME` — validated against [`re_world::POSTFX_PRESETS`] at parse.
    pub postfx: Option<String>,
}

/// `[note KIND]` placard kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoteKind {
    /// A human researcher's note (a person's voice).
    Researcher,
    /// A machine/system log entry.
    SystemLog,
    /// A fragment of the model's own "thought" (the network's own voice).
    ModelThought,
    /// A STENCILLED wall plate — signage the building carries (a lift call panel, an
    /// exit marker, a switch legend), not a voice. Its body is laid out VERBATIM by
    /// the renderer, so hand-drawn ASCII frames survive onto the wall.
    Stencil,
}

/// One `[material NAME]` — an authored material the level binds to its surfaces.
///
/// This is the parsed form of the optional `[material …]` section: a name, flat
/// `fg`/`bg` colours, an optional fixed glyph, an optional procedural recipe, an
/// optional animation, and a LOD rank bias. `build_map` forwards these (and the
/// per-surface bindings) to the [`Map`] as [`MaterialDef`]s; the renderer turns
/// them into its `MaterialRegistry`. Levels without any `[material]` section
/// leave `materials` empty and render exactly as before (built-in fallback).
#[derive(Clone, Debug)]
pub struct LevelMaterial {
    /// The section name, bound to surfaces by `[material NAME]`.
    pub name: String,
    /// Flat foreground colour.
    pub fg: u32,
    /// Flat background colour.
    pub bg: u32,
    /// Optional fixed glyph override (unset ⇒ the renderer's per-surface default).
    pub glyph: Option<u32>,
    /// Optional procedural pattern recipe.
    pub recipe: Option<MaterialRecipe>,
    /// Optional animation applied to this material.
    pub anim: MaterialAnim,
    /// LOD rank bias — how eagerly this material keeps detail at distance.
    pub rank: u8,
    /// `reflective = true` turns the bound surface into a MIRROR (the raycaster
    /// reflects the scene into it). Default `false` ⇒ an ordinary opaque surface.
    pub reflective: bool,
    /// Mirror CLARITY `[0,1]` (only meaningful when `reflective`): `1` = crisp,
    /// `0` = glitch/empty. Default [`MaterialDef::DEFAULT_CLARITY`]. The game can
    /// override it live via the renderer's `MirrorClarity` resource.
    pub clarity: f32,
    /// Optional `preset = NAME` — selects a renderer CATALOG material by name (e.g.
    /// `metal`, `rust`, `monitor`). When set, the bound surface starts from that
    /// catalog material's full config (palette, relief, animation, emissive,
    /// reflective, stability, underline); any explicit `fg`/`bg`/`glyph`/`recipe`/
    /// `animation`/`reflective`/`clarity` keys on the SAME `[material]` still
    /// override on top. `re_level` is engine-agnostic so it cannot look the catalog
    /// up itself — it forwards the name to the renderer encoded in the emitted
    /// [`MaterialDef::name`] (see `Level::material_defs`). `None` ⇒ no preset
    /// (the historical `recipe=`/`fg=` path, fully back-compatible).
    pub preset: Option<String>,
    /// Optional `texture = NAME` — binds this material to a decoded `[texture NAME]`
    /// image. Only meaningful with `recipe = image`: at [`build_map`](Level::build_map)
    /// time the NAME is resolved to the image's `tex_id` (its index in
    /// [`Map::textures`]) and the material's recipe becomes
    /// [`MaterialRecipe::Image`]`{ tex_id }`. `None` ⇒ no image binding (the historical
    /// procedural/preset path, fully back-compatible). An unknown NAME, or a missing
    /// `recipe = image`, leaves the material non-image (reported by `build_map_checked`).
    pub texture: Option<String>,
    /// Optional `span = ALONG_MIN,ALONG_SIZE UP_TOP,UP_SIZE` (M3a) — the WORLD-ANCHORED
    /// UV span for a `recipe = image` material. When set, ONE copy of the figure is
    /// stretched across the surface's full world extent (so an N-cell-wide textured
    /// wall shows ONE figure, not N tiled copies — walk up to a big one and see
    /// detail). `None` ⇒ the legacy per-cell `[0,1]` tiling. Folded into the resolved
    /// [`MaterialRecipe::Image`]`{ span }` at [`build_map`](Level::build_map) time.
    pub image_span: Option<re_world::ImageSpan>,
    /// Optional `subcell = off | quad | braille | ascii | contour | lines` (M3b) — the
    /// SUB-CELL detail mode for a `recipe = image` material. `quad` renders the image at
    /// 2×2 (two colours/cell); `braille` at 2×4 (sharp monochrome silhouette via the
    /// Braille block); `ascii` applies a luminance ramp + Sobel-direction edge glyphs;
    /// `contour` uses a DoG hybrid for clean contour lines (hero mode); `lines` renders
    /// line-only / ink with non-edge cells inked near their background. Default
    /// [`Off`](re_world::SubcellMode::Off) ⇒ the legacy per-cell solid block, byte-
    /// identical. Folded into [`MaterialRecipe::Image`]`{ subcell }` at build time.
    pub image_subcell: re_world::SubcellMode,
    /// Optional `fit = stretch | contain | cover` — the fit mode for a `recipe = image`
    /// material (Component 1 of the quality-textures spec). Controls how the image is
    /// mapped onto the wall surface when no explicit `span` override is given.
    /// `stretch` fills the wall exactly (aspect ignored); `contain` preserves aspect,
    /// whole image fits inside the wall (default); `cover` preserves aspect, fills the
    /// wall completely cropping the overflow. An explicit `span` key overrides `fit`.
    /// Folded into [`MaterialRecipe::Image`]`{ fit }` at build time.
    pub image_fit: re_world::Fit,
    /// Optional `tile = ALONG UP` — how many times the image REPEATS across the
    /// mapped span (`ALONG` across the surface, `UP` up it). `None` ⇒ ONE copy
    /// ([`re_world::IMAGE_TILE_NONE`]), which is what every level authored before this
    /// key gets and what the renderer short-circuits on, so the un-tiled path stays
    /// byte-identical. Both factors are validated finite and `> 0` at parse: a `0`
    /// would divide the figure into nothing and a negative would mirror it, neither
    /// of which is a repeat count. Folded into [`MaterialRecipe::Image`]`{ tile }`.
    pub image_tile: Option<(f32, f32)>,
    /// Optional `relief = height_scale=… normal_strength=…` — turns a `recipe = image`
    /// material from a SELF-LIT decal into a LIT SURFACE: the renderer stops passing
    /// the sampled colour straight through, reads the bound texture's 8-bit height
    /// channel (`[texture] height = …`) as the material's height field, and runs the
    /// same normal-perturbation + glyph-ramp relief pass every procedural material
    /// uses. `None` (the default) ⇒ the historical self-lit block, byte-identical.
    /// Folded into [`MaterialRecipe::Image`]`{ relief }` at build time.
    pub image_relief: Option<re_world::ImageRelief>,
    /// Optional `detail = on | off` — opt-in
    /// world-anchored detail-tier cross-fade (tone/pattern/detail + underline
    /// micro-detail) for this material's procedural recipe. Default `false` ⇒ the
    /// historical single-tier shading, byte-identical. `re_render` is the only
    /// consumer that gates live shading behaviour on this flag; `re_level` only
    /// parses and forwards it via [`MaterialDef::detail`](re_world::MaterialDef).
    pub detail: bool,
}

impl Default for LevelMaterial {
    fn default() -> Self {
        LevelMaterial {
            name: String::new(),
            fg: 0,
            bg: 0,
            glyph: None,
            recipe: None,
            anim: MaterialAnim::default(),
            rank: 0,
            reflective: false,
            clarity: MaterialDef::DEFAULT_CLARITY,
            preset: None,
            texture: None,
            image_span: None,
            image_subcell: re_world::SubcellMode::Off,
            image_fit: re_world::Fit::Contain,
            image_tile: None,
            image_relief: None,
            detail: false,
        }
    }
}

/// The sentinel prefix `re_level` stamps on an emitted [`MaterialDef::name`] to
/// carry a `preset = NAME` selection to the renderer through the engine-agnostic
/// [`MaterialDef`] (whose only string field is `name`). Unit-separator framed so it
/// can never collide with an authored `[material NAME]` section name. The renderer
/// strips this prefix to recover the catalog material name. Keep in sync with
/// `re_render::material_from_def`'s `preset_name`.
pub const PRESET_NAME_PREFIX: &str = "\u{1f}preset\u{1f}";

/// Which authored material (by name) a level binds to each surface kind. An
/// unset binding (`None`) means "use the renderer's built-in for that surface".
/// Additive: a level with no `[material]` section leaves every binding `None`.
#[derive(Clone, Debug, Default)]
pub struct SurfaceMaterials {
    /// Material bound to plain solid walls.
    pub wall_solid: Option<String>,
    /// Material bound to circuit-trace walls.
    pub wall_trace: Option<String>,
    /// Material bound to matrix-rain walls.
    pub wall_matrix: Option<String>,
    /// Material bound to floors.
    pub floor: Option<String>,
    /// Material bound to ceilings.
    pub ceiling: Option<String>,
}

/// One `[note ...]` easter-egg placard (verbatim body).
#[derive(Clone, Debug)]
pub struct LevelNote {
    /// Which voice this placard reads in.
    pub kind: NoteKind,
    /// World X coordinate of the reader cell.
    pub x: i32,
    /// World Y coordinate of the reader cell.
    pub y: i32,
    /// World Z coordinate of the reader cell.
    pub z: i32,
    /// The `[note] facing` yaw column.
    pub facing: i32,
    /// The placard's verbatim body text.
    pub text: String,
}

/// Per-cell geometry SHAPE for the `[cells]` `shapes =` sub-table (M3c 45°
/// geometry).
///
/// This is an alias for [`re_core::Shape`] — the SAME enum the world
/// (`re_world` collision) and renderer (`re_render` DDA) use, so the saved
/// `shapes =` table carries the real orientations (`DiagXY` for wedges,
/// `RampDir` for ramps) with no lossy `u8` indirection. A cell with no entry
/// is [`Shape::Full`] (the default solid cube); only non-`Full` shapes are
/// written, so an all-`Full` level emits no `shapes =` sub-table and stays
/// byte-identical to a pre-shape save.
///
/// The on-disk grammar uses readable direction names (see [`shape_to_tokens`]
/// / [`tokens_to_shape`]):
///
/// ```text
/// x y z wedgexy ne|nw|se|sw
/// x y z rampz   posx|negx|posy|negy
/// ```
pub type CellShape = Shape;

/// Readable `[cells]` `shapes =` tokens for a non-`Full` [`Shape`].
///
/// Returns `(shape_keyword, direction_keyword)`, e.g.
/// `(`"wedgexy"`, `"ne"`)` or `(`"rampz"`, `"posx"`)`. Returns `None` for
/// [`Shape::Full`] (which is never written — the absence of an entry IS
/// `Full`). Inverse of [`tokens_to_shape`].
pub fn shape_to_tokens(shape: Shape) -> Option<(&'static str, &'static str)> {
    Some(match shape {
        Shape::Full => return None,
        Shape::WedgeXY(d) => (
            "wedgexy",
            match d {
                DiagXY::NE => "ne",
                DiagXY::NW => "nw",
                DiagXY::SE => "se",
                DiagXY::SW => "sw",
            },
        ),
        Shape::RampZ(d) => (
            "rampz",
            match d {
                RampDir::PosX => "posx",
                RampDir::NegX => "negx",
                RampDir::PosY => "posy",
                RampDir::NegY => "negy",
            },
        ),
    })
}

/// Parse a readable `[cells]` `shapes =` `(shape, dir)` token pair into a
/// [`Shape`]. Case-insensitive on the caller's side (pass lowercased tokens).
/// Inverse of [`shape_to_tokens`]. Returns `None` on an unknown keyword.
pub fn tokens_to_shape(shape_kw: &str, dir_kw: &str) -> Option<Shape> {
    match shape_kw {
        "full" => Some(Shape::Full),
        "wedgexy" => Some(Shape::WedgeXY(match dir_kw {
            "ne" => DiagXY::NE,
            "nw" => DiagXY::NW,
            "se" => DiagXY::SE,
            "sw" => DiagXY::SW,
            _ => return None,
        })),
        "rampz" => Some(Shape::RampZ(match dir_kw {
            "posx" => RampDir::PosX,
            "negx" => RampDir::NegX,
            "posy" => RampDir::PosY,
            "negy" => RampDir::NegY,
            _ => return None,
        })),
        _ => None,
    }
}

/// A dense, explicit per-cell geometry grid — the data backing the `[cells]`
/// section. This is what makes freely-painted geometry persist (the procedural
/// `[room]`/`[part]`/`[vox]` channels cannot represent arbitrary cells).
///
/// Layout: `kinds[x + sx*(y + sy*z)]`, matching [`re_world`]'s row-major cell
/// indexing. `material_ids` is a sparse map keyed by `(x,y,z)` of the **1-based**
/// colour-material id stored in a cell's `side_data` (the SAME id space as
/// `[color_cell_ids]`; `0` ⇒ default, omitted). `shapes` is sparse: only cells
/// whose [`Shape`] is non-`Full` are recorded (all-`Full` ⇒ empty ⇒ no
/// `shapes =` sub-section, byte-identical to a shape-less save).
///
/// PRECEDENCE: when a level carries a [`CellGrid`], [`Level::build_map`] stamps
/// it LAST — it is authoritative over the procedural shell/parts/vox.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CellGrid {
    /// Grid dimensions `(sx, sy, sz)`.
    pub size: (usize, usize, usize),
    /// Dense cell kinds, `kinds[x + sx*(y + sy*z)]`.
    pub kinds: Vec<CellKind>,
    /// Sparse per-cell colour-material ids (1-based; same space as
    /// `[color_cell_ids]`). Sorted by `(z,y,x)` on write for determinism.
    pub material_ids: Vec<(i32, i32, i32, u16)>,
    /// Sparse per-cell shapes (M3c). Only non-[`Shape::Full`] cells are
    /// recorded. Sorted by `(z,y,x)` on write for determinism.
    pub shapes: Vec<(i32, i32, i32, CellShape)>,
}

impl CellGrid {
    /// A new all-[`Empty`](CellKind::Empty) grid of the given size.
    pub fn new(size: (usize, usize, usize)) -> Self {
        let n = size.0 * size.1 * size.2;
        CellGrid {
            size,
            kinds: vec![CellKind::Empty; n],
            material_ids: Vec::new(),
            shapes: Vec::new(),
        }
    }

    #[inline]
    fn index(&self, x: usize, y: usize, z: usize) -> Option<usize> {
        let (sx, sy, sz) = self.size;
        if x >= sx || y >= sy || z >= sz {
            return None;
        }
        Some(x + sx * (y + sy * z))
    }

    /// Cell kind at `(x,y,z)`; [`CellKind::Empty`] out of bounds.
    pub fn kind_at(&self, x: usize, y: usize, z: usize) -> CellKind {
        match self.index(x, y, z) {
            Some(i) => self.kinds.get(i).copied().unwrap_or(CellKind::Empty),
            None => CellKind::Empty,
        }
    }

    /// Capture an existing [`Map`] into a [`CellGrid`] — the editor's
    /// save-time helper. Every cell's kind is recorded densely; a cell with a
    /// non-zero `side_data` records its id sparsely so it round-trips, and a
    /// cell with a non-[`Shape::Full`] geometry shape records its
    /// [`Shape`] sparsely (M3c).
    ///
    /// ## `side_data` is OVERLOADED — capture all, arbitrate on LOAD (B2)
    ///
    /// `side_data` is a single `u16` channel that [`build_map`](Level::build_map)
    /// stamps PROCEDURALLY for several disjoint roles, NOT just authored colour ids:
    ///
    /// * **node index** on [`CellKind::Object`] cells (`tag = node_index`),
    /// * **pipe live-tag** (incl. the bit-15 R2 reactive flag) on
    ///   [`CellKind::WallTrace`] pipe cells ([`re_core::PIPE_TAG_FLAG`]),
    /// * **hologram style** on [`CellKind::Hologram`] cells,
    /// * **overlay zone id** on solid cells inside a `[zone]`.
    ///
    /// Re-stamping a CAPTURED procedural tag over a freshly-recomputed one on the
    /// next load would CORRUPT it (pipes/nodes/holos mis-tag). The original B2 fix
    /// tried to prevent this at CAPTURE time by EXCLUDING the procedural-detection
    /// kinds (`Object`/`Hologram`/pipe-flagged) here — but that also dropped a
    /// GENUINELY authored tag that merely SHARES one of those kinds yet has no
    /// procedural source (a bare painted `Object` with no backing `[node]`), losing
    /// the author's data on the round-trip.
    ///
    /// So capture is now total: EVERY non-zero `side_data` is recorded. The
    /// procedural-vs-authored arbitration happens entirely on the LOAD side —
    /// [`build_map`](Level::build_map) runs every procedural pass FIRST, then
    /// re-applies the captured `material_ids` only onto cells that no procedural pass
    /// claimed (whose `side_data` is still `0`) via
    /// [`Self::reapply_authored_ids_where_empty`]. A pipe/zone/node/holo cell keeps
    /// its recomputed tag (the procedural pass wrote it, so the re-apply skips it);
    /// an authored tag on a cell nobody recomputes survives. Invariant:
    /// procedurally-derived tags are never persisted-and-restamped, yet no authored
    /// value is ever lost.
    pub fn from_map(map: &Map) -> Self {
        let (sx, sy, sz) = map.size();
        let mut g = CellGrid::new((sx, sy, sz));
        for z in 0..sz {
            for y in 0..sy {
                for x in 0..sx {
                    let (xi, yi, zi) = (x as i32, y as i32, z as i32);
                    let kind = map.cell_at(xi, yi, zi);
                    g.kinds[x + sx * (y + sy * z)] = kind;
                    let sd = map.side_data_at(xi, yi, zi);
                    // Capture EVERY non-zero `side_data` so no authored value is ever
                    // lost across a save→load (B2-fix). The OLD B2 attempt excluded
                    // `Object`/`Hologram`/pipe-flagged cells here, on the theory that
                    // those tags are always RECOMPUTED procedurally on reload — but a
                    // BARE `Object` painted directly on the map with no backing
                    // `[node]` (the editor's inspect-`n` tag, or any author-painted
                    // Object/Hologram) then has NO procedural source and its tag was
                    // silently dropped. Capturing every tag here is safe because the
                    // procedural-vs-authored decision is now made on the LOAD side
                    // ([`build_map`]): captured ids are re-applied AFTER all procedural
                    // passes and ONLY onto cells no procedural pass claimed (its
                    // `side_data` is still `0`). So a pipe/zone/node/holo cell keeps
                    // its freshly-recomputed tag, while a genuinely authored tag (incl.
                    // a bare Object) survives. See [`Self::reapply_authored_ids_where_empty`].
                    // Every RESERVED sentinel on an `Object` cell — `INERT_OBJECT_TAG`
                    // (an unbacked painted `O`) AND `DOOR_OBJECT_TAG` (the door anchor,
                    // added 2026-09-27 alongside the door alongside fix RC5) — is
                    // DERIVED (recomputed by `build_map`/`objects.push` on every load),
                    // never authored directly. Persisting one would write a meaningless
                    // `x,y,z = <sentinel>` entry into every save; for the door tag it
                    // was WORSE — `stamp_onto`/`reapply_authored_ids_where_empty` would
                    // then try to re-apply the captured sentinel as a colour-material id
                    // on reload, corrupting the door's freshly-recomputed tag. Checking
                    // the whole reserved BAND (`>= RESERVED_TAG_BASE`), not one sentinel
                    // at a time, means a future addition to `re_core::RESERVED_SENTINELS`
                    // is covered here for free.
                    let derived_reserved =
                        kind == CellKind::Object && sd >= re_core::RESERVED_TAG_BASE;
                    if sd != 0 && !derived_reserved {
                        g.material_ids.push((xi, yi, zi, sd));
                    }
                    let shape = map.shape_at(xi, yi, zi);
                    if shape != Shape::Full {
                        g.shapes.push((xi, yi, zi, shape));
                    }
                }
            }
        }
        g
    }

    /// Stamp this grid onto `map` (the load-time / build-time rebuild helper).
    /// Authoritative: every in-bounds cell is overwritten with this grid's
    /// kind, and its `side_data` set to the recorded material id (or `0`).
    /// Out-of-bounds cells are ignored by [`Map::set_cell`]'s bounds guard.
    pub fn stamp_onto(&self, map: &Map) -> Map {
        let mut map = map.clone();
        self.stamp_into(&mut map);
        map
    }

    /// In-place variant of [`stamp_onto`](Self::stamp_onto).
    ///
    /// This is the STANDALONE editor capture→apply path (capture a `Map` with
    /// [`from_map`](Self::from_map), apply onto a fresh `Map`): it stamps geometry
    /// AND authored material ids together, so a `from_map` → `stamp_into` round-trip
    /// reproduces the captured cells exactly. [`build_map`](Level::build_map) does
    /// NOT call this — it splits the two phases (material ids EARLY, geometry LATE)
    /// so the procedural `[zone]`/pipe passes win over a captured id on their cells
    /// (B2). Calling both phases here, in order, is equivalent for the standalone
    /// path (no procedural passes run between).
    pub fn stamp_into(&self, map: &mut Map) {
        self.stamp_material_ids_into(map);
        self.stamp_geometry_into(map);
    }

    /// Write the captured `material_ids` into each target cell's `side_data`
    /// AUTHORITATIVELY (overwriting whatever is there), leaving the cell's KIND
    /// untouched. This is the STANDALONE capture→apply path's material phase
    /// ([`stamp_into`](Self::stamp_into)) — no procedural passes run between capture
    /// and apply there, so an unconditional write reproduces the captured cells
    /// exactly. [`build_map`](Level::build_map) does NOT use this; it re-applies the
    /// authored ids AFTER its procedural passes and only where no pass claimed the
    /// cell — see [`reapply_authored_ids_where_empty`](Self::reapply_authored_ids_where_empty).
    /// Out-of-bounds cells are ignored by [`Map::set_cell`]'s bounds guard.
    pub fn stamp_material_ids_into(&self, map: &mut Map) {
        for &(x, y, z, id) in &self.material_ids {
            if x < 0 || y < 0 || z < 0 {
                continue;
            }
            // Keep the cell's CURRENT kind; only set its colour-material id. The
            // kind is written authoritatively in `stamp_geometry_into` (phase 2).
            let kind = map.cell_at(x, y, z);
            map.set_cell(x as usize, y as usize, z as usize, kind, id);
        }
    }

    /// Re-apply the captured authored `material_ids` onto `map`, but ONLY where the
    /// cell's current `side_data` is `0` — i.e. where NO procedural pass
    /// (node/door, hologram, zone, `[color_cell_ids]`, pipe-trace) already claimed
    /// the cell. This is the B2-fix arbitration: [`build_map`](Level::build_map)
    /// runs every procedural pass FIRST (each writing a non-zero tag onto the cells
    /// it owns), then calls this LAST — so a procedural tag always wins on its own
    /// cell (we skip it), while a genuinely authored id on a cell nobody recomputed
    /// (a plain painted wall, a bare `Object`/`Hologram` with no backing entity)
    /// is restored. The cell's KIND is preserved (already written by
    /// [`stamp_geometry_into`](Self::stamp_geometry_into)); only `side_data` is set.
    /// Out-of-bounds / negative coords are skipped.
    pub fn reapply_authored_ids_where_empty(&self, map: &mut Map) {
        for &(x, y, z, id) in &self.material_ids {
            if x < 0 || y < 0 || z < 0 {
                continue;
            }
            if map.side_data_at(x, y, z) != 0 {
                // A procedural pass owns this cell — its recomputed tag wins.
                continue;
            }
            let kind = map.cell_at(x, y, z);
            map.set_cell(x as usize, y as usize, z as usize, kind, id);
        }
    }

    /// Phase 2 of the split stamp (B2): write each cell's KIND + geometry SHAPE
    /// authoritatively, PRESERVING whatever `side_data` is already on the cell
    /// (procedural zone/pipe/holo/node tags or a phase-1 authored id). This is the
    /// "cells stamp last" rule for geometry WITHOUT the namespace-clobbering bug —
    /// the old `stamp_into` re-set `side_data` to the captured id here, which
    /// re-stamped stale procedural tags over the freshly-computed ones. Now the
    /// side_data is read back and re-passed unchanged, so the procedural passes that
    /// ran between phase 1 and phase 2 keep their tags.
    ///
    /// Unrecorded shapes reset to [`Shape::Full`] (authoritative over any prior
    /// shape on the map — a level without a `shapes =` sub-table flattens any stray
    /// 45° geometry, matching the pre-split behaviour).
    pub fn stamp_geometry_into(&self, map: &mut Map) {
        let (sx, sy, sz) = self.size;
        // Sparse shape lookup; default Full (so unrecorded cells reset to Full
        // — stamping is authoritative over any prior shape on the map).
        let mut shape_of: std::collections::BTreeMap<(i32, i32, i32), Shape> =
            std::collections::BTreeMap::new();
        for &(x, y, z, shape) in &self.shapes {
            shape_of.insert((x, y, z), shape);
        }
        for z in 0..sz {
            for y in 0..sy {
                for x in 0..sx {
                    let kind = self.kinds[x + sx * (y + sy * z)];
                    // PRESERVE the existing side_data (procedural tag or phase-1
                    // authored id); never clobber it with a stale captured value.
                    let sd = map.side_data_at(x as i32, y as i32, z as i32);
                    map.set_cell(x, y, z, kind, sd);
                    let shape = shape_of
                        .get(&(x as i32, y as i32, z as i32))
                        .copied()
                        .unwrap_or(Shape::Full);
                    map.set_shape(x, y, z, shape);
                }
            }
        }
    }
}

/// Glyph ⇒ [`CellKind`] for the `[cells]` ASCII rows. Matches `ts_editor`'s
/// `chrome.rs` table so the saved file is the same alphabet the editor paints:
/// `.`=Empty, `#`=WallSolid, `/`=WallTrace, `%`=WallMatrix, `_`=Floor,
/// `^`=Ceiling, `O`=Object, `H`=Hologram, `@`=Portal. Unknown glyphs ⇒ `None`.
pub fn glyph_to_cell_kind(c: char) -> Option<CellKind> {
    Some(match c {
        '.' => CellKind::Empty,
        '#' => CellKind::WallSolid,
        '/' => CellKind::WallTrace,
        '%' => CellKind::WallMatrix,
        '_' => CellKind::Floor,
        '^' => CellKind::Ceiling,
        'O' => CellKind::Object,
        'H' => CellKind::Hologram,
        '@' => CellKind::Portal,
        _ => return None,
    })
}

/// [`CellKind`] ⇒ glyph for the `[cells]` ASCII rows (inverse of
/// [`glyph_to_cell_kind`]). Total — every `CellKind` has a glyph.
pub fn cell_kind_to_glyph(k: CellKind) -> char {
    match k {
        CellKind::Empty => '.',
        CellKind::WallSolid => '#',
        CellKind::WallTrace => '/',
        CellKind::WallMatrix => '%',
        CellKind::Floor => '_',
        CellKind::Ceiling => '^',
        CellKind::Object => 'O',
        CellKind::Hologram => 'H',
        CellKind::Portal => '@',
    }
}

/// One `key = value` line whose KEY the parser did not recognize.
///
/// ## Why this type exists
///
/// Every section handler in the parser used to end in `_ => {}`: an unrecognized
/// key was discarded in complete silence. That one pattern is the ENABLER for
/// this project's signature defect — "authored, parsed, tested, never on screen".
/// A mistyped key vanishes with no message, and so does a correctly-typed key
/// nobody wired up; six instances were found and fixed one at a time, and every
/// one of them would have been findable in seconds if the parser had simply said
/// something.
///
/// It is deliberately **NOT an error**. The corpus is 63 hand-authored files and
/// a stray key must never make a level unloadable. These are collected onto
/// [`Level::unknown_keys`] and surfaced by `re_lvlcheck` (so `ts_editor` reports
/// them on save and CI reports them in the linter), where a human can triage.
///
/// Sections whose keys are DATA are exempt, because there is no such thing as an
/// unknown key there: `[inputs]` (`NODE = 0/1`), `[color_cell_ids]` and
/// `[cells]`'s id sub-table (`x,y,z = id`), `[map]`/`[win_when]` (read-and-skip
/// blocks), and `[note]` bodies (verbatim text).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnknownKey {
    /// 1-based line number in the source file.
    pub line: usize,
    /// The section the key sat in, as it is written (`"room"`, `"node"`, …).
    pub section: &'static str,
    /// The key exactly as authored.
    pub key: String,
}

impl std::fmt::Display for UnknownKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "line {}: [{}] unknown key '{}'",
            self.line, self.section, self.key
        )
    }
}

/// A fully-parsed level.
#[derive(Clone, Debug, Default)]
pub struct Level {
    /// `[meta] name` — the level's file-system-safe identifier.
    pub name: String,
    /// `[meta] title` — the level's human-readable display title.
    pub title: String,
    /// `[meta] version` — the authored file-format revision tag, kept VERBATIM.
    ///
    /// INERT: nothing in the engine branches on it. It is retained purely so a
    /// load→save round-trip does not delete a line the author wrote. Empty ⇒ the
    /// key was absent and the writer emits nothing (byte-identical back-compat).
    ///
    /// This field exists because `re_level` used to discard every `[meta]`/`[room]`
    /// key it had no runtime use for, so `re_voximport`'s writer had nothing to
    /// emit and `ts_editor` silently erased the line on save. Preserving is
    /// strictly safer than deleting, and costs one `String`.
    pub version: String,
    /// `[meta] spec_rev` — the design-document revision date this level was
    /// authored against (e.g. `2026-07-30`), kept VERBATIM.
    ///
    /// INERT at runtime, but NOT unused: a downstream `canon_consistency`
    /// integration test (`spec_rev_stays_in_the_real_world`) reads these lines
    /// straight off disk and asserts every one names a real, pre-fiction calendar
    /// date. An editor that erased the field would quietly disarm that guard —
    /// which is exactly why it is preserved here rather than dropped.
    /// Empty ⇒ the key was absent.
    pub spec_rev: String,
    /// Room dimensions (sx, sy, sz).
    pub size: (usize, usize, usize),
    /// `[room] ambient` string (selects the perimeter wall kind).
    pub ambient: String,
    /// `[room] floor` — the authored floor-style tag (e.g. `grid`), kept VERBATIM.
    ///
    /// INERT: `build_map` does not read it and no renderer consumes it; the
    /// floor's actual look comes from `[material] surface = floor`. It is authored
    /// on 17 shipped levels and is preserved so a save cannot delete it. Empty ⇒
    /// the key was absent.
    pub room_floor: String,
    /// `[room] ceiling` — the authored ceiling-style tag (e.g. `dim`), kept
    /// VERBATIM. INERT for the same reason as [`Level::room_floor`]; preserved for
    /// the same reason. Empty ⇒ the key was absent.
    pub room_ceiling: String,
    /// `[room] render_pipes` opt-in (default `false`). When `true`, [`build_map`](Level::build_map)
    /// stamps each pipe's expanded `path` cells as visible [`CellKind::WallTrace`]
    /// traces tagged with a pipe id ([`re_core::encode_pipe_side_data`]), so the
    /// reactive flow material (R2) can address them and the bridge can drive their
    /// powered/flow state. Default OFF ⇒ pipes stay invisible wiring (geometry
    /// byte-identical to today); only an opt-in level (or the R5 showcase) grows
    /// visible flowing traces. Spec: every layer defaults to a no-op.
    pub render_pipes: bool,
    /// `[room] cycle_keep` opt-in (default `false`). The L12 `while(true)`
    /// finale flag: when `true`, `re_puzzle`'s forward pass treats a wired graph
    /// CYCLE not as a rollback error but as a stable, accepted terminal state
    /// (the loop "stands"). Default OFF ⇒ today's cycle-rollback behaviour,
    /// byte-identical. `re_puzzle` consumes this as
    /// `re_puzzle::PuzzleState::cycle_keep`.
    pub cycle_keep: bool,
    /// `[room] power_budget = N` — the per-level cap on how many pipes may be in
    /// the `On` state **simultaneously** (a deliberate, user-approved v0.2 puzzle
    /// rule addition).
    ///
    /// **Why the rule exists.** `re_puzzle`'s forward pass has no negative weights
    /// and no inhibition, so a node's activation is MONOTONE in the set of ON
    /// pipes: turning a pipe ON can never turn a node off. Consequently, for any
    /// level with no `[door] target_off` and no `[sequence]`, the all-switches-ON
    /// state wins whenever *any* state wins — i.e. the campaign could be beaten by
    /// flipping every switch you walk past, which is also the natural behaviour of
    /// a first-person player. `power_budget` is what makes a wrong wiring cost
    /// something, and therefore what makes a level losable.
    ///
    /// `0` (and an ABSENT key) means **UNLIMITED** — byte-identical to the
    /// behaviour before the rule existed. Only levels that opt in change, so the
    /// ~51 demo/showcase/archive levels and their goldens are untouched.
    ///
    /// Consumed by `re_puzzle` as `re_puzzle::PuzzleState::power_budget`, which
    /// owns the enforcement (an ON-toggle that would exceed the cap is REFUSED and
    /// the puzzle state is left unchanged; an OFF-toggle always succeeds).
    pub power_budget: u32,
    /// `[room] storey_height = H` — cells per storey for the multi-floor vertical
    /// model (the multi-floor stairs design, §4). Floor planes
    /// live at `z = 0, H, 2H, …`; a 45° staircase needs `H` consecutive `RampZ`
    /// cells to climb one storey. This is METADATA ONLY — it does NOT change grid
    /// storage or dimensions (the `.lvl` already carries explicit `[room] size`);
    /// it is the contract the editor stamps staircases against and a validation
    /// hint, NOT a second source of truth for the grid extent.
    ///
    /// Optional in the `.lvl`; absent ⇒ the default
    /// [`DEFAULT_STOREY_HEIGHT`] (4), i.e. legacy single-storey behaviour and a
    /// byte-identical [`build_map`](Level::build_map). `H < 1` is a parse error. Stored as the
    /// effective value: [`load_string`]/[`load`] normalise an absent key to `4` in
    /// `finalize`, so a loaded `Level` always reports a concrete height. A
    /// programmatically-built [`Level::default`] leaves this `0` (the unset
    /// sentinel); prefer the [`Level::storey_height`] accessor, which maps `0` to
    /// the default.
    pub storey_height: u32,
    /// Every `[node NAME]` — the neurons of the puzzle network.
    pub nodes: Vec<Node>,
    /// Every `[pipe A->B]` — the wired connections between nodes.
    pub pipes: Vec<Pipe>,
    /// Every `[part NAME]` — voxel stamps for the pedestal output volume.
    pub parts: Vec<Part>,
    /// [`PartLook`] per [`Level::parts`] entry (same index). Shorter than `parts` when a
    /// caller built parts directly; [`Level::part_look`] reads it with the default.
    pub part_looks: Vec<PartLook>,
    /// Every expanded `[emit]` clause — which node lights which part.
    pub emits: Vec<EmitClause>,
    /// `[inputs]` — initial activation per node name.
    pub inputs: Vec<(String, i32)>,
    /// The `[pedestal]` bounding box.
    pub pedestal: Pedestal,
    /// The `[door]` win-condition anchor + target nodes.
    pub door: Door,
    /// Optional `[sequence]` ordered-pattern win mode (L11 sequence-lock). Empty
    /// `patterns` ⇒ inactive: the level uses the ordinary single-target `[door]`
    /// win (additive; a level with no `[sequence]` section is unchanged).
    pub sequence: SequenceSpec,
    /// The `[player]` spawn pose.
    pub player: PlayerSpawn,
    /// Every `[wall_decal NAME]` region.
    pub decals: Vec<WallDecal>,
    /// Optional `[holo NAME]` transparent hologram objects. Empty ⇒ none
    /// (additive; the shipped levels are unchanged).
    pub holos: Vec<LevelHolo>,
    /// Optional `[portal NAME]` non-euclidean warp-portal pairs. Empty ⇒ none
    /// (additive; the shipped levels are unchanged). Each pair stamps two
    /// wall-face openings and registers a reciprocal `delta` in `build_map`.
    pub portals: Vec<LevelPortal>,
    /// Optional `[vox NAME]` `.vox` model stamps. Empty ⇒ none (additive; a
    /// level with no `[vox]` section is byte-for-byte unchanged).
    pub voxes: Vec<LevelVox>,
    /// Optional `[texture NAME]` image textures (decoded at `build_map` time). Empty
    /// ⇒ none (additive; a level with no `[texture]` section decodes nothing and is
    /// byte-for-byte unchanged). An `[material] recipe = image texture = NAME` binds
    /// to one of these by name.
    pub textures: Vec<LevelTexture>,
    /// Optional `[sky]` image-cubemap paths (cylinder + zenith/nadir caps). All
    /// `None` ⇒ procedural sky only (additive; a level with no `[sky]` section is
    /// byte-for-byte unchanged). Decoded at [`build_map`](Level::build_map) time and
    /// forwarded to the [`Map`] via its sky setters.
    pub sky: SkyDef,
    /// Optional `[zone_input NAME]` floor-zones (player-position-as-input).
    /// Empty ⇒ none (additive; a level with no `[zone_input]` section behaves
    /// exactly as before). Each zone names an input node it holds at `1` while
    /// the player stands inside it; the host runtime drives the live state.
    pub zone_inputs: Vec<ZoneInput>,
    /// Optional `[trigger NAME]` trigger-fields: a box region that fires an
    /// arbitrary action when the player walks into it. Empty ⇒ none (additive; a
    /// level with no `[trigger]` section is byte-for-byte unchanged). Each
    /// trigger's `ActivateNode` node name is resolved to an index in `finalize`;
    /// `build_map` copies the resolved list onto the [`Map`] for the game layer's
    /// firing system.
    pub triggers: Vec<re_core::Trigger>,
    /// Optional `[effects]` ambient knobs (snow/steam/matrix). All-off default.
    pub effects: LevelEffects,
    /// Optional `[palette]` distance-fog override.
    pub palette: LevelPalette,
    /// Optional `[atmosphere]` lighting/height-fog/postfx knobs. All-`None` ⇒ the
    /// renderer keeps its built-ins (additive; a level with no `[atmosphere]`
    /// section is byte-for-byte unchanged).
    pub atmos: LevelAtmosphere,
    /// Optional repeatable `[light]` point lights. Empty ⇒ no punctual lights
    /// and the shading path is byte-identical to the pre-lighting build.
    pub lights: Vec<PointLight>,
    /// Every `[note ...]` easter-egg placard.
    pub notes: Vec<LevelNote>,
    /// Optional authored materials (`[material NAME]` sections). Empty ⇒ the
    /// renderer uses its built-ins (additive; back-compatible with 01-07).
    pub materials: Vec<LevelMaterial>,
    /// Optional `[color_cell_ids]` per-cell colour-material assignments — the
    /// data side of the colour→material substrate (`re_voximport` rule 9). Each
    /// tuple is `(x, y, z, id)` where `id` is the **1-based** colour-material id
    /// (index into `materials` + 1; `1` = `materials[0]`). `build_map` stamps the
    /// id into the target geometry cell's `side_data`, where the renderer reads
    /// it back and dispatches that cell's material from the colour palette
    /// instead of the surface-kind default. Empty ⇒ none (additive; a level with
    /// no `[color_cell_ids]` is byte-for-byte unchanged).
    pub color_cell_ids: Vec<(i32, i32, i32, u16)>,
    /// Which authored material name (if any) each surface kind binds to. Names
    /// are resolved against `materials` in `finalize`.
    pub surface_materials: SurfaceMaterials,
    /// Optional `[emitter NAME]` particle emitters. Each `[emitter]` section
    /// produces one [`EmitterSpec`] (validated/clamped at parse time). Empty ⇒
    /// none (additive; a level with no `[emitter]` section is unchanged). The
    /// `NAME` after `[emitter ...]` is optional and purely for readability — it
    /// does not appear on the stored spec.
    ///
    /// May also contain SYNTHESIZED entries (`EmitterSpec::synthesized == true`)
    /// that `Parser::finalize` expands from a non-emitter knob (today:
    /// `[effects] snow`) — mixed in anywhere in the list, not just a trailing/leading
    /// run. The renderer treats every entry the same; a writer or editor that must
    /// not persist/display a self-regenerating entry filters on
    /// [`EmitterSpec::synthesized`] instead of relying on list position.
    pub emitters: Vec<EmitterSpec>,
    /// Optional **Scalable Material Overlay** (`[zone]`/`[line]`/`[inset]`
    /// sections). [`Overlay::default`] (all-empty) ⇒ none (additive; a level with
    /// no overlay sections is byte-for-byte unchanged — the bottom rung of the
    /// degradation ladder).
    /// P4: the renderer-facing carrier; insets are RESOLVED here too — see [`build_map`](Level::build_map).
    /// P0 carrier + parse only: NO resolve, NO render (the scalable material overlay
    /// design's staged rollout).
    pub overlay: Overlay,
    /// The `.lvl` file this level was [`load`]ed from, if any. Set by [`load`]; a
    /// string-only [`load_string`] leaves it `None`. Used by [`build_map`](Level::build_map) (P4) to
    /// resolve an `[inset]`'s child-scene path RELATIVE to this file's directory and to
    /// seed the cycle guard with the parent path. `None` ⇒ child paths resolve relative
    /// to the process CWD and the cycle guard starts empty.
    pub source_path: Option<PathBuf>,
    /// Optional explicit, dense per-cell geometry (`[cells]` section). When
    /// `Some`, this is the AUTHORITATIVE geometry: [`build_map`](Level::build_map)
    /// stamps the procedural shell/parts/vox/overlay FIRST and then overwrites
    /// every in-bounds cell from this grid LAST (so painted edits win). `None` ⇒
    /// the level has no `[cells]` section and parses/serializes exactly as before
    /// (byte-identical back-compat). The editor populates this from the live
    /// `Map` via [`CellGrid::from_map`] before saving.
    pub cells: Option<CellGrid>,
    /// Optional OBJECT-NESTING hierarchy (`[hierarchy]` section). Each entry is a
    /// `(child, parent)` link; a child has at most ONE parent. Empty ⇒ no nesting
    /// (additive; a level with no `[hierarchy]` section is byte-for-byte
    /// unchanged). The editor uses this so moving a parent object group-moves its
    /// descendants. The links are validated (cycle-free, in-range) on load by
    /// [`prune_hierarchy`](Level::prune_hierarchy); a stale/cyclic link is dropped
    /// rather than rejecting the whole level.
    pub hierarchy: Vec<(HierRef, HierRef)>,
    /// Keys the parser did not recognize, in file order — see [`UnknownKey`].
    /// Empty for every shipped level (a corpus test enforces it). Diagnostics
    /// only: never an error, never consulted by the runtime, and never written
    /// back out — an unknown key IS lost by a save, which is exactly what the
    /// report is for.
    pub unknown_keys: Vec<UnknownKey>,
    /// `[prop NAME]` sections, VERBATIM (set-dressing §3.2). Expanded by
    /// [`build_map`](Level::build_map) into the built map (never into `[cells]` or
    /// any other section, so the writer re-emits exactly these).
    pub props: Vec<props::PropDef>,
    /// The resolved power bindings of every prop (`power =`), in the state the
    /// built map shows. Consumed by the game to flip lit parts; see
    /// [`props::PowerGate`].
    pub prop_power: Vec<props::PowerGate>,
    /// Notes a prop re-homed (`note =`) and where they were AUTHORED. `notes`
    /// holds the moved position the game renders; the writer emits this one.
    pub prop_note_origins: Vec<props::NoteOrigin>,
    /// The `style` a prop gets when it names none: `[meta] prop_style`, parsed
    /// by `props::PropStyle::parse`, else `lab` (the level is self-describing —
    /// see `props::PropStyle::for_movement` for the movement→pack convention the
    /// twelve campaign levels follow).
    pub prop_default_style: props::PropStyle,
    /// Whether `[meta] prop_style` was actually authored, as opposed to
    /// [`Level::prop_default_style`] simply holding its struct default (`Lab`).
    ///
    /// `PropStyle::Lab` is both the default AND a legitimately-authored value
    /// (three of the twelve campaign levels genuinely want the lab pack), so a
    /// plain enum cannot tell "the key was never written" from "the key was
    /// written as `lab`" — and the writer must not collapse those, or an
    /// authored `prop_style = lab` line would silently vanish on the next save.
    /// Same failure class `version`/`spec_rev` guard against for strings,
    /// adapted for an enum whose default is itself a valid authored value.
    /// `false` ⇒ the key was absent and the writer emits nothing (byte-identical
    /// back-compat for every level that predates this key).
    pub prop_style_authored: bool,
    /// `[field NAME]` sections, VERBATIM.
    /// Empty ⇒ none (additive; a level with no `[field]` section is byte-for-byte
    /// unchanged). Resolved into [`Level::resolved_fields`] by `finalize`; the
    /// writer (`re_voximport`) re-emits this list verbatim, mirroring `props`.
    pub field_defs: Vec<fields::FieldDef>,
    /// Every `field_defs` entry resolved (names → indices, defaults applied).
    /// Populated once at load time by `finalize`. `pub` (like every other `Level`
    /// field) only so external crates can use `Level { .. ..Default::default() }`
    /// struct-update syntax; prefer the [`Level::fields`] accessor over reading
    /// this directly.
    pub resolved_fields: Vec<fields::ResolvedField>,
}

impl Level {
    /// The [`PartLook`] of `parts[i]` — the default (full height, one colour) when the
    /// part authored none. A `top_from` with no `top` colour has no top.
    pub fn part_look(&self, i: usize) -> PartLook {
        let mut look = self.part_looks.get(i).copied().unwrap_or_default();
        if look.top.is_some_and(|(rgb, _)| rgb > 0xFF_FFFF) {
            look.top = None;
        }
        look
    }

    /// The EFFECTIVE storey height (cells per storey) for this level. Maps the
    /// unset sentinel `0` (what a bare [`Level::default`] leaves and what the
    /// parser starts with) to [`DEFAULT_STOREY_HEIGHT`] (4), so callers always
    /// get a concrete, ≥1 value regardless of whether the level came from
    /// [`load`] (where `finalize` already normalised an absent key) or was built
    /// in code. Prefer this over reading the raw [`Level::storey_height`] field.
    pub fn storey_height(&self) -> u32 {
        if self.storey_height == 0 {
            DEFAULT_STOREY_HEIGHT
        } else {
            self.storey_height
        }
    }

    /// Find a node index by name.
    pub fn node_index(&self, name: &str) -> Option<usize> {
        self.nodes.iter().position(|n| n.name == name)
    }

    /// Every resolved `[field NAME]`, host
    /// defaults already applied, names already resolved to indices. Empty for a
    /// level with no `[field]` section.
    #[must_use]
    pub fn fields(&self) -> &[fields::ResolvedField] {
        &self.resolved_fields
    }

    /// Find the index of a node positioned exactly at `(x, y, z)`, if any. The
    /// editor uses this to SELECT (rather than duplicate) when a node already
    /// occupies the placement cell.
    pub fn node_at(&self, x: i32, y: i32, z: i32) -> Option<usize> {
        self.nodes
            .iter()
            .position(|n| n.x == x && n.y == y && n.z == z)
    }

    /// Append a new puzzle [`Node`] at `(x, y, z)` with an auto-assigned UNIQUE
    /// name (`n<k>` for the smallest `k` not already taken) and a default glyph.
    /// Returns the new node's index. The editor's object-placement path uses this
    /// so it never reaches into the private `nodes` Vec and never produces a
    /// duplicate name (which would break node-by-name wiring on reload).
    pub fn push_node(&mut self, x: i32, y: i32, z: i32) -> usize {
        let mut k = 0usize;
        let name = loop {
            let candidate = format!("n{k}");
            if self.node_index(&candidate).is_none() {
                break candidate;
            }
            k += 1;
        };
        self.nodes.push(Node {
            name,
            x,
            y,
            z,
            glyph: 'N',
            ..Node::default()
        });
        self.nodes.len() - 1
    }

    /// Find the index of an existing pipe wiring `from_idx -> to_idx` (by the two
    /// node indices), if any. The editor uses this to REFUSE a duplicate A->B
    /// (a self-pipe is rejected separately). Order matters: `a->b` and `b->a` are
    /// distinct wires.
    pub fn pipe_between(&self, from_idx: usize, to_idx: usize) -> Option<usize> {
        self.pipes
            .iter()
            .position(|p| p.from_node == from_idx && p.to_node == to_idx)
    }

    /// Auto-route an orthogonal (L-shaped / Manhattan) 6-connected cell path
    /// between two nodes' cells, advancing x-then-y-then-z (the SAME canonical
    /// expansion the `[pipe] path` parser uses via `expand_path`, so a routed pipe
    /// round-trips byte-identically through save→reload). Returns the empty path if
    /// either index is out of range. Public so the editor never reimplements the
    /// expansion (one source of truth with the `.lvl` parser).
    pub fn auto_route_path(&self, from_idx: usize, to_idx: usize) -> Vec<Coord3> {
        let (Some(a), Some(b)) = (self.nodes.get(from_idx), self.nodes.get(to_idx)) else {
            return Vec::new();
        };
        expand_path(&[(a.x, a.y, a.z), (b.x, b.y, b.z)])
    }

    /// Append a new [`Pipe`] wiring node `from_idx -> to_idx`, AUTO-ROUTED with an
    /// orthogonal (L-shaped) [`auto_route_path`](Self::auto_route_path), `Off`
    /// state, default density/colour, and no switch. Returns the new pipe's index.
    ///
    /// VALIDATE-BEFORE-MUTATE: returns `None` (leaving the level UNCHANGED) for a
    /// self-pipe (`from_idx == to_idx`), an out-of-range endpoint, or a duplicate
    /// `from->to` wire ([`pipe_between`](Self::pipe_between) already exists). The
    /// `from`/`to` NAME fields are filled from the nodes so the pipe serializes as
    /// `[pipe A->B]` and re-resolves on reload. The editor's pipe-create flow uses
    /// this so it never reaches into the `pipes` Vec or reinvents the routing.
    pub fn push_pipe_between(&mut self, from_idx: usize, to_idx: usize) -> Option<usize> {
        if from_idx == to_idx {
            return None;
        }
        if from_idx >= self.nodes.len() || to_idx >= self.nodes.len() {
            return None;
        }
        if self.pipe_between(from_idx, to_idx).is_some() {
            return None;
        }
        let path = self.auto_route_path(from_idx, to_idx);
        let from = self.nodes[from_idx].name.clone();
        let to = self.nodes[to_idx].name.clone();
        let color_rgb = self.nodes[from_idx].color_rgb;
        self.pipes.push(Pipe {
            from,
            to,
            from_node: from_idx,
            to_node: to_idx,
            state: PipeState::Off,
            density: 1.0,
            color_rgb,
            path,
            switch_cell: None,
            switch_at_raw: None,
        });
        Some(self.pipes.len() - 1)
    }

    /// Find a material index by name.
    pub fn material_index(&self, name: &str) -> Option<usize> {
        self.materials.iter().position(|m| m.name == name)
    }

    /// Append a new authored [`LevelMaterial`] with an auto-assigned UNIQUE name
    /// (`mat<k>` for the smallest `k` not already taken) and a sensible default
    /// appearance (a mid-grey fg on a near-black bg, no glyph/recipe). Returns the
    /// new material's index. The editor's material-authoring path uses this so it
    /// never reaches into `materials` to invent a (possibly colliding) name — a
    /// duplicate name would break the name→palette-id resolution shared by
    /// `[color_cell_ids]`, zones, and surface bindings.
    ///
    /// The new material participates in the colour palette immediately: its
    /// 1-based palette id is `material_index + 1`, so a freshly-pushed material
    /// (index `i`) can be stamped into a cell's `side_data` as `i + 1`.
    pub fn push_material(&mut self) -> usize {
        let mut k = 0usize;
        let name = loop {
            let candidate = format!("mat{k}");
            if self.material_index(&candidate).is_none() {
                break candidate;
            }
            k += 1;
        };
        self.materials.push(LevelMaterial {
            name,
            fg: 0x00_B0B0B0,
            bg: 0x00_101010,
            ..LevelMaterial::default()
        });
        self.materials.len() - 1
    }

    // ───────────────────────────── object nesting ─────────────────────────────
    //
    // The hierarchy is `Vec<(child, parent)>`: a child has at most one parent,
    // cycles are forbidden, and a deleted/duplicated object shifts the indices of
    // its later siblings (the `Vec` kinds). The invariants are kept by the set /
    // clear / fixup helpers below so the map never points at a gone object.

    /// True iff `r` names an object that actually exists in this level (in range
    /// for a `Vec` kind; a singleton kind is always in range — its presence is the
    /// editor's concern, not the link's). Used to PRUNE stale links on load.
    fn hier_ref_in_range(&self, r: HierRef) -> bool {
        match r.kind {
            HierKind::Node => r.index < self.nodes.len(),
            HierKind::Emitter => r.index < self.emitters.len(),
            HierKind::Vox => r.index < self.voxes.len(),
            HierKind::Trigger => r.index < self.triggers.len(),
            HierKind::Door | HierKind::Spawn | HierKind::Pedestal => true,
        }
    }

    /// The direct parent of `child`, if any.
    pub fn hier_parent_of(&self, child: HierRef) -> Option<HierRef> {
        self.hierarchy
            .iter()
            .find(|(c, _)| *c == child)
            .map(|(_, p)| *p)
    }

    /// Would making `parent` the parent of `child` create a cycle? True when
    /// `parent == child` (self-parent) or `child` is already an ancestor of
    /// `parent` (so the new edge would close a loop).
    pub fn hier_would_cycle(&self, child: HierRef, parent: HierRef) -> bool {
        if child == parent {
            return true;
        }
        // child is reachable from parent by walking UP ⇒ a cycle.
        let mut cur = parent;
        for _ in 0..=self.hierarchy.len() {
            if cur == child {
                return true;
            }
            match self.hier_parent_of(cur) {
                Some(p) => cur = p,
                None => break,
            }
        }
        false
    }

    /// SET `child`'s parent to `parent`, replacing any existing parent link.
    /// Refuses (returns `false`, leaving the map unchanged) a self-parent or any
    /// reparent that would form a cycle. Validate-before-mutate.
    pub fn set_hier_parent(&mut self, child: HierRef, parent: HierRef) -> bool {
        if self.hier_would_cycle(child, parent) {
            return false;
        }
        self.clear_hier_parent(child);
        self.hierarchy.push((child, parent));
        true
    }

    /// CLEAR `child`'s parent link (un-parent). Returns `true` if a link was
    /// removed. Children OF `child` keep their own links (only the one edge above
    /// `child` is cut).
    pub fn clear_hier_parent(&mut self, child: HierRef) -> bool {
        let before = self.hierarchy.len();
        self.hierarchy.retain(|(c, _)| *c != child);
        self.hierarchy.len() != before
    }

    /// Every transitive DESCENDANT of `root` (children, grandchildren, …), as a
    /// de-duplicated list NOT including `root`. Bounded so a cyclic map can't loop.
    pub fn hier_descendants(&self, root: HierRef) -> Vec<HierRef> {
        let mut out: Vec<HierRef> = Vec::new();
        let mut frontier = vec![root];
        // Bound: at most one push per link, so total work is O(links²) worst case.
        while let Some(cur) = frontier.pop() {
            for (c, p) in &self.hierarchy {
                if *p == cur && *c != root && !out.contains(c) {
                    out.push(*c);
                    frontier.push(*c);
                }
            }
            if out.len() > self.hierarchy.len() {
                break; // cycle guard
            }
        }
        out
    }

    /// Drop every hierarchy link whose child or parent is out of range, plus any
    /// self-parent or cyclic edge. Called on load so a stale/hand-broken
    /// `[hierarchy]` never points at a gone object or loops. Order-preserving.
    pub fn prune_hierarchy(&mut self) {
        // First drop out-of-range / self links. Take the Vec out so the in-range
        // test can borrow `self` immutably while we filter.
        let raw = std::mem::take(&mut self.hierarchy);
        self.hierarchy = raw
            .into_iter()
            .filter(|(c, p)| c != p && self.hier_ref_in_range(*c) && self.hier_ref_in_range(*p))
            .collect();
        // Drop duplicate child links (a child may have only one parent) keeping the
        // first occurrence.
        let mut seen: Vec<HierRef> = Vec::new();
        self.hierarchy.retain(|(c, _)| {
            if seen.contains(c) {
                false
            } else {
                seen.push(*c);
                true
            }
        });
        // Finally break cycles: walk and drop any edge that closes a loop. Rebuild
        // incrementally so the cycle test sees only already-accepted edges.
        let candidates = std::mem::take(&mut self.hierarchy);
        for (c, p) in candidates {
            if !self.hier_would_cycle(c, p) {
                self.hierarchy.push((c, p));
            }
        }
    }

    /// Fix up the hierarchy after an object of `kind` at `removed` index was
    /// DELETED from its `Vec`: drop every link touching the removed object, then
    /// shift every later sibling's index down by one (so a link to what was index
    /// `removed+1` still names the right object after the `Vec::remove`). Singleton
    /// kinds never shift. Call AFTER the `Vec::remove`.
    pub fn hier_fixup_after_remove(&mut self, kind: HierKind, removed: usize) {
        if kind.is_singleton() {
            // A cleared singleton: just drop links touching it.
            let r = HierRef::new(kind, 0);
            self.hierarchy.retain(|(c, p)| *c != r && *p != r);
            return;
        }
        let shift = |r: &mut HierRef| {
            if r.kind == kind && r.index > removed {
                r.index -= 1;
            }
        };
        // Drop links touching the removed object, then re-index the survivors.
        let removed_ref = HierRef::new(kind, removed);
        self.hierarchy
            .retain(|(c, p)| *c != removed_ref && *p != removed_ref);
        for (c, p) in self.hierarchy.iter_mut() {
            shift(c);
            shift(p);
        }
    }

    /// REMOVE the node at `idx` and fix up EVERY index-based reference to nodes so
    /// no surviving reference dangles or silently re-points at the wrong node.
    ///
    /// Node indices are stored in five places besides `nodes` itself; a raw
    /// `nodes.remove(idx)` corrupts all of them (B2). For each reference type the
    /// rule after removing index `idx` is: a ref **== idx** is gone (DROP it), a
    /// ref **> idx** is shifted down one (DECREMENT it), a ref **< idx** is
    /// unchanged. The per-reference policy:
    ///
    /// - `door.target_parts` / `door.target_off` (`Vec<usize>` win targets): drop the
    ///   deleted node's entry, decrement higher entries. A win target on a deleted
    ///   node is simply removed (the door re-derives a default target on load only
    ///   if `target_parts` ends up empty — matching the parser's behaviour).
    /// - `zone_inputs[].node_index` + the `node` NAME string: a zone that drove the
    ///   deleted node has no valid target, so the whole `ZoneInput` is DROPPED.
    ///   Survivors with a higher `node_index` are decremented (their `node` name is
    ///   already correct — names don't shift).
    /// - `pipes[].from_node` / `.to_node`: a pipe with EITHER endpoint on the deleted
    ///   node is DROPPED (a half-connected pipe is meaningless). Survivors with a
    ///   higher endpoint index are decremented. The `from`/`to` NAME strings on a
    ///   surviving pipe are unaffected (names don't shift).
    /// - `[hierarchy]` node links: delegated to [`Level::hier_fixup_after_remove`].
    ///
    /// Out-of-range `idx` is a no-op (validate-before-mutate). Call this instead of
    /// `nodes.remove(idx)` anywhere a node is deleted.
    pub fn remove_node(&mut self, idx: usize) {
        if idx >= self.nodes.len() {
            return; // out of range ⇒ nothing to remove, no fixup.
        }
        self.nodes.remove(idx);

        // Helper: drop == idx, decrement > idx for a Vec of node indices.
        let fixup_indices = |v: &mut Vec<usize>| {
            v.retain(|&n| n != idx);
            for n in v.iter_mut() {
                if *n > idx {
                    *n -= 1;
                }
            }
        };
        fixup_indices(&mut self.door.target_parts);
        fixup_indices(&mut self.door.target_off);

        // Zone inputs: drop any whose target node was deleted, decrement the rest.
        self.zone_inputs.retain(|z| z.node_index != idx);
        for z in self.zone_inputs.iter_mut() {
            if z.node_index > idx {
                z.node_index -= 1;
            }
        }

        // Pipes: drop any with either endpoint on the deleted node, decrement the
        // rest. A half-connected pipe is meaningless, so the whole pipe goes.
        self.pipes
            .retain(|p| p.from_node != idx && p.to_node != idx);
        for p in self.pipes.iter_mut() {
            if p.from_node > idx {
                p.from_node -= 1;
            }
            if p.to_node > idx {
                p.to_node -= 1;
            }
        }

        // `[hierarchy]` node links: drop links touching the removed node, shift the
        // rest down one (same drop-==/decrement->-rule, applied to the hier Vec).
        self.hier_fixup_after_remove(HierKind::Node, idx);
    }

    /// Build the [`re_core::PoolTable`] for this level: `pipe tag → owning node's
    /// word pool`.
    ///
    /// The renderer streams readable words along a powered pipe, but a pipe cell
    /// carries only its pipe tag ([`re_core::encode_pipe_side_data`], stamped by
    /// `Self::stamp_pipes` with `tag == the pipe's index`) — there is no
    /// path from that tag back to the node whose words it should carry. This is that
    /// path, distilled to plain data the render layer can index.
    ///
    /// **The owning node is the pipe's SOURCE** (`from_node`). The words a pipe carries
    /// are the words the upstream node is sending — the flow direction and the reading
    /// direction agree — so a pipe fed by node `A` streams `A`'s pool, and `A`'s several
    /// outgoing pipes all read as its voice.
    ///
    /// Pure + allocation-once (at load): the pools are cloned ONCE per node, not per
    /// pipe. A level whose nodes author no pools yields a table whose every lookup is
    /// `None`, and the render side falls back to its built-in word list.
    pub fn pool_table(&self) -> re_core::PoolTable {
        let nodes: Vec<re_core::NodePool> = self
            .nodes
            .iter()
            .map(|n| re_core::NodePool {
                pool: n.pool.clone(),
                pool_alt: n.pool_alt.clone(),
                anchor: n.anchor,
            })
            .collect();
        // Pipe tag == pipe index (see `stamp_pipes`), so the owner list is positional.
        let owners: Vec<usize> = self.pipes.iter().map(|p| p.from_node).collect();
        re_core::PoolTable::from_parts(nodes, &owners)
    }

    /// Build a [`GeometrySpec`]-backed [`Map`] by stamping this level's static
    /// geometry (shell + decals + objects). The game's node cells are stamped as
    /// [`re_world::CellKind::Object`] with `tag = node_index`; the door anchor
    /// is also stamped as an `Object` with `tag = 0`. Pipes and pedestal parts
    /// are not stamped (transparent to the raycaster).
    pub fn build_map(&self) -> Map {
        let mut map = self.build_map_base();
        // `[prop]` sections stamp LAST, onto the finished prop-free map (Empty-only,
        // so they never overwrite `[cells]` or any puzzle cell). No props => skipped
        // => byte-identical to the pre-kit map.
        if !self.props.is_empty() {
            let (exp, _) = props::expand_level(self, &map);
            props::stamp_expansion(self, &exp, &mut map);
        }
        map
    }

    /// [`build_map`](Self::build_map) WITHOUT the `[prop]` expansion: the room as
    /// the puzzle authored it. The prop kit expands against this map, and
    /// `re_lvlcheck`'s `props` rule floods it to prove props cut nothing off.
    pub fn build_map_base(&self) -> Map {
        let wall_kind = ambient_to_wall_kind(&self.ambient);
        let decals: Vec<DecalRegion> = self
            .decals
            .iter()
            .map(|d| DecalRegion {
                kind: d.surface_kind,
                x0: d.x0,
                y0: d.y0,
                z0: d.z0,
                x1: d.x1,
                y1: d.y1,
                z1: d.z1,
            })
            .collect();
        // Collect all game objects: node cells (tag = node index) followed by
        // the door anchor (tag = DOOR_OBJECT_TAG). The game layer identifies
        // specific objects via `side_data_at`; the engine treats them all as
        // generic Object cells.
        let mut objects: Vec<ObjectCell> = self
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| ObjectCell {
                x: n.x,
                y: n.y,
                z: n.z,
                tag: i as u16,
            })
            .collect();
        // Door anchor as a generic object, tagged DOOR_OBJECT_TAG so it renders as
        // the exit rather than a duplicate of node 0 (the game still tracks the
        // door position in its own Level.door struct, not by scanning the grid).
        objects.push(ObjectCell {
            x: self.door.x,
            y: self.door.y,
            z: self.door.z,
            tag: DOOR_OBJECT_TAG,
        });
        let spec = GeometrySpec {
            size: self.size,
            wall_kind,
            decals: &decals,
            objects: &objects,
        };
        let mut map = stamp_geometry(&spec);
        // HOLOGRAM objects: transparent wireframe voxels, stamped after the static
        // geometry (only into Empty air, so they never clobber walls/objects).
        if !self.holos.is_empty() {
            stamp_holos(&mut map, &self.holo_objects());
        }
        // [vox] MODEL STAMPS: load each `.vox`, downscale, offset, and stamp every
        // solid voxel as the chosen solid surface. Stamped AFTER holos so a solid
        // building wins any cell. On a load/IO error we skip that stamp (the dump
        // path + lvlcheck stay usable); `build_map_checked` surfaces the error.
        if !self.voxes.is_empty() {
            let _ = self.stamp_voxes(&mut map);
        }
        // [portal] PAIRS: stamp each declared warp-portal pair via the shared
        // engine math (`Map::add_portal_pair`) — the SAME implementation the
        // Stage-1 demo scaffold uses, so render + teleport stay consistent.
        // Regions were validated in `finalize` (flat wall faces, matching area),
        // so this never produces a degenerate link. The optional `skin` is a
        // render concern and is not consumed here.
        for pt in &self.portals {
            map.add_portal_pair(pt.region_a(), pt.region_b(), pt.rotate);
        }
        // OVERLAY ZONES (Scalable Material Overlay §6 P1, DP1-A/DP3-A): stamp each
        // `[zone]`'s resolved 1-based palette id into the SOLID cells of its region.
        // Done BEFORE `[color_cell_ids]` so an explicit per-cell id (more specific)
        // overwrites a zone region-fill; among overlapping zones the later-listed
        // one wins (see `stamp_zones`). A level with no `[zone]` sections has an
        // empty `overlay.zones`, so this is skipped and the map is byte-identical.
        if !self.overlay.zones.is_empty() {
            self.stamp_zones(&mut map);
        }
        // NB: the `[cells]` authored material ids are NOT stamped here. They are
        // re-applied AFTER every procedural pass AND after the `[cells]` geometry —
        // see `reapply_authored_ids_where_empty` at the end of `build_map`. That
        // ordering lets each procedural pass (zone/pipe/node/holo/`[color_cell_ids]`)
        // win on the cells it claims, while a genuinely authored id on an unclaimed
        // cell still round-trips (B2-fix).
        // COLOUR→MATERIAL substrate: stamp each `[color_cell_ids]` assignment into
        // the target GEOMETRY cell's `side_data` (the 1-based colour-material id).
        // The cell keeps its existing kind; only `side_data` is set, so the
        // renderer's surface-kind dispatch is unchanged and it reads the id back to
        // pick the colour palette. A level with no `[color_cell_ids]` skips this
        // loop entirely (byte-identical). Out-of-bounds coords / negatives are
        // ignored by `set_cell`'s bounds guard.
        for &(x, y, z, id) in &self.color_cell_ids {
            if x < 0 || y < 0 || z < 0 {
                continue;
            }
            let kind = map.cell_at(x, y, z);
            map.set_cell(x as usize, y as usize, z as usize, kind, id);
        }
        // PIPE TRACES (R2 reactive-materials §3): when the level opts in via
        // `[room] render_pipes`, stamp each pipe's expanded path as visible tagged
        // `WallTrace` cells so the flow material can address them and the bridge can
        // drive their powered/flow state. Default OFF ⇒ this loop is skipped and the
        // geometry is byte-identical to today (pipes stay invisible wiring).
        if self.render_pipes {
            self.stamp_pipes(&mut map);
        }
        // IMAGE TEXTURES: decode each `[texture NAME]` file into a
        // `re_core::TextureImage` and push it onto `Map::textures`, BEFORE the
        // material set is built (an `[material] recipe = image texture = NAME` then
        // resolves NAME → the pushed `tex_id`). A decode failure is non-fatal: the
        // texture is skipped (no entry, no name→id binding) and surfaced by
        // `build_map_checked`. An empty `[texture]` set leaves `Map::textures` empty
        // and the name→id map empty ⇒ byte-identical to today.
        let tex_ids = self.decode_textures(&mut map);
        // SKY IMAGES: decode each authored `[sky]` path into a
        // `re_core::TextureImage` and forward it to the Map via the sky setters.
        // Best-effort: a missing/bad file is skipped (the slot stays `None`, the
        // procedural sky covers it) and surfaced by `build_map_checked`. A level
        // with no `[sky]` section leaves all sky slots at their default `None` ⇒
        // byte-identical to today.
        self.apply_sky_images(&mut map);
        // PROCEDURAL SKY PROGRAM: assemble the accumulated `[sky] base=/layer=/
        // kind=/seed=/hz=` keys into a `SkySpec` and hand it to the Map. A level
        // that authored none of those keys (image-only `[sky]`, or no `[sky]`
        // section at all) gets `None` here, leaving `Map::sky_spec` at its
        // default and the render byte-identical to today (no golden churn on
        // existing levels).
        if let Some(spec) = self.sky.to_spec() {
            map.set_sky_spec(spec);
        }
        // Renderer-facing material set: the five surface slots, followed by the
        // per-cell colour palette (id N ⇒ palette[N-1]) when any `[color_cell_ids]`
        // are present. `registry_for_map` reads the first five as the surface
        // dispatch and the remainder as the colour-material palette. Empty ⇒ the
        // renderer keeps every built-in (back-compat with 01-07).
        let mats = self.renderer_material_defs(&tex_ids);
        if !mats.is_empty() {
            map.set_materials(mats);
        }
        let atmo = self.atmosphere();
        if !atmo.is_none() {
            map.set_atmosphere(atmo);
        }
        // SCALABLE MATERIAL OVERLAY carrier: hand the authored overlay (zones already
        // stamped into `side_data` above; LINES + insets carried for the renderer) to
        // the map so `re_render`'s P2 line post-pass can draw `[line]` strokes. An empty
        // overlay (no `[zone]`/`[line]`/`[inset]` sections) sets `Overlay::default()`,
        // which is inert ⇒ the map renders byte-identically to today. Cheap clone (a
        // few small Vecs); `build_map` runs once per level load, never per frame.
        if !self.overlay.is_empty() {
            map.set_overlay(self.overlay.clone());
        }
        // `[line]` colours (set-dressing E0): hand the renderer each line's resolved
        // palette id, resolved HERE by the authored material name (the renderer only
        // sees preset-encoded `MaterialDef` names, so it cannot resolve them itself).
        if let Some(ids) = self.line_palette_ids() {
            map.set_line_palette_ids(ids);
        }
        // SCALABLE MATERIAL OVERLAY P4 — INSETS ("врезки"), bounded phase-band reveal:
        // resolve each authored `[inset]` (which references a child `.lvl`) into a
        // `re_core::ResolvedInset` (region + phase-band + representative colour) so the
        // renderer can apply the bounded reveal over a SelfSimilar zone WITHOUT loading a
        // `.lvl`. Empty `overlay.insets` ⇒ no resolution ⇒ empty `resolved_insets` ⇒ the
        // renderer's override loop never runs (byte-identical to today). The cycle guard
        // is seeded with this level's own canonical path so a self-referential inset is
        // caught immediately.
        if !self.overlay.insets.is_empty() {
            let mut in_flight: HashSet<PathBuf> = HashSet::new();
            if let Some(p) = &self.source_path {
                in_flight.insert(canonical_or_raw(p));
            }
            let resolved = self.resolve_insets(&mut in_flight);
            if !resolved.is_empty() {
                map.set_resolved_insets(resolved);
            }
        }
        // TRIGGER-FIELDS: copy the resolved `[trigger]`s onto the map so the game
        // layer's firing system can query them. Names are already resolved to
        // node indices in `finalize`. Empty ⇒ no triggers (byte-identical).
        if !self.triggers.is_empty() {
            map.set_resolved_triggers(self.triggers.clone());
        }
        // EXPLICIT CELL GEOMETRY (`[cells]`, editor save format): AUTHORITATIVE.
        // GEOMETRY — stamped LAST so freely-painted cells overwrite the procedural
        // shell / parts / vox / overlay-tagged cells (spec §5/§11: "cells stamp
        // last"). This writes KIND + 45° SHAPE authoritatively but PRESERVES each
        // cell's existing `side_data` (the procedural zone/pipe/holo/node tags
        // computed above) — so it never re-stamps a stale captured tag over a
        // freshly-computed one. A level with no `[cells]` (`cells == None`) skips
        // this entirely, so the map is byte-identical to today (back-compat).
        //
        // Then RE-APPLY the captured authored material ids, but ONLY onto cells no
        // procedural pass claimed (their `side_data` is still `0`). This is the
        // B2-fix arbitration: a pipe/zone/node/holo/`[color_cell_ids]` cell keeps
        // its recomputed tag (skipped here), while a genuinely authored id — incl. a
        // bare painted `Object`/`Hologram` with no backing entity — round-trips.
        if let Some(cells) = &self.cells {
            cells.stamp_geometry_into(&mut map);
            // DOOR ANCHOR STALE-TAG GUARD (RC5 follow-up, 2026-09-27).
            // `stamp_geometry_into` PRESERVES whatever `side_data` was already on a
            // cell — correct for the zone/pipe/node/holo tags it is designed to
            // protect, but the door anchor is a special case: it is deliberately
            // NOT re-stamped below (see "NODES ONLY" a few lines down), because in
            // every `[cells]`-format level the door lives baked into the room shell
            // (a `#`/`_` glyph), not as a distinct `Object` cell. So when `[cells]`
            // repaints the door's position to a non-`Object` kind, the `DOOR_OBJECT_TAG`
            // written by the earlier `stamp_geometry` pass (line ~2058) survives on
            // the WRONG kind of cell as orphaned `side_data`. Before RC5 the door's
            // tag was `0`, so this exact leak was silently harmless (indistinguishable
            // from "no tag"); RC5 gave the door a distinct nonzero tag, turning the
            // same leak into a real, observable mismatch — a Floor/WallSolid cell
            // reporting `side_data = DOOR_OBJECT_TAG`, which `CellGrid::from_map`
            // then captures as a spurious `material_ids` entry on save. Clear it
            // whenever `[cells]` moved the door's own cell away from `Object`.
            if self.door.x >= 0 && self.door.y >= 0 && self.door.z >= 0 {
                let kind = map.cell_at(self.door.x, self.door.y, self.door.z);
                if kind != CellKind::Object
                    && map.side_data_at(self.door.x, self.door.y, self.door.z) == DOOR_OBJECT_TAG
                {
                    map.set_cell(
                        self.door.x as usize,
                        self.door.y as usize,
                        self.door.z as usize,
                        kind,
                        0,
                    );
                }
            }
            // PUZZLE ANCHORS RE-STAMP (visibility fix). `[cells]` is authoritative for
            // GEOMETRY, but a node/door cell is NOT geometry — it is derived from the
            // `[node]`/`[door]` ENTITIES above, and the editor's save grid does not paint
            // entity cells. So an editor-format level (every `L*.lvl` tower level) had
            // its node + door anchors CLEARED by the stamp above: L1's (2,5,2), (13,4,6)
            // and door (14,4,5) all came out `Empty`, so the renderer had no `Object`
            // cell to draw and the entire puzzle network was invisible in the frame.
            // Re-stamping the anchors restores them at their AUTHORED z, so collision
            // (which probes `floor(eye_z)` only) is unchanged.
            //
            // ANCHORS ARE AUTHORITATIVE (set-dressing loose ends, 2026-09-24). The anchor
            // is restored whatever `[cells]` painted there — not only when it left air.
            // `stamp_geometry_into` keeps each cell's `side_data`, so a glyph painted
            // over a node (L10's `J` at (8,7,2), every node of L11/L12 — `/` traces an
            // editor save captured) left a WallTrace cell still carrying the NODE INDEX,
            // which the renderer then read as a colour id: the node drew as an unrelated
            // palette material (L10's `J` came out as the researcher placard) and could
            // not light. A node is an entity; its cell is derived from `[node] pos`, so
            // `[cells]` has no say over it. Consequence for the editor: a save after this
            // load captures an `O` at the anchor, which is exactly what the cell is.
            //
            // NODES ONLY — the DOOR anchor is deliberately NOT restored. Two reasons:
            // `Door` has no "declared" flag, so a level with no `[door]` section carries
            // the `Door::default()` (0,0,0) sentinel, and restoring that would stamp a
            // phantom object into the corner of every editor-authored level; and in the
            // `[cells]` campaign the door is authored INTO the room's shell (L3/L8/L9/L10/
            // L11/L12 carry a `#` there, the stair levels a `_`) — the game tracks it in
            // `Level.door` (`DoorCell`), not by scanning the grid, so turning that wall
            // into a tag-0 `Object` would only make the exit draw as a lit node 0.
            //
            // ORDER: before `reapply_authored_ids_where_empty`, so an explicitly authored
            // `[cells] material_ids` entry on node 0's anchor (tag 0) still wins its
            // `side_data` — the arbitration that held before this re-stamp existed.
            // Levels with no `[cells]` never reach this and are byte-identical.
            for o in &objects[..self.nodes.len()] {
                if o.x >= 0 && o.y >= 0 && o.z >= 0 {
                    map.set_cell(
                        o.x as usize,
                        o.y as usize,
                        o.z as usize,
                        CellKind::Object,
                        o.tag,
                    );
                    map.set_shape(o.x as usize, o.y as usize, o.z as usize, Shape::Full);
                }
            }
            // HOLOGRAMS OVER `[cells]` (set-dressing loose ends). The holo pass above ran
            // before `[cells]`, whose dense grid then wrote `.` (Empty) over every holo
            // voxel — a `[holo]` section in any `[cells]` level had no effect at all.
            // Stamp them again, with the same Empty-air-only rule, so `[cells]` still
            // wins every cell it actually fills and a painted `H` keeps its own tag.
            if !self.holos.is_empty() {
                stamp_holos(&mut map, &self.holo_objects());
            }
            // ZONES OVER `[cells]` GEOMETRY (set-dressing E1 / F1). The zone pass above
            // ran before `[cells]`, and it skips air — so every interior block that
            // exists ONLY in `[cells]` (L8's shelves, L11's monolith + cabinets, L12's
            // colonnade + loop column) was still `Empty` then and never received its
            // zone material. Stamp the zones again onto exactly those cells: solid
            // geometry the `[cells]` pass created that no procedural pass has tagged.
            // BEFORE the authored-id re-apply, for the same B2 reason the first zone
            // pass precedes it: a zone is recomputed on every load and must win over an
            // id a previous editor save captured from it.
            if !self.overlay.zones.is_empty() {
                self.stamp_zones_onto_cells_geometry(&mut map);
            }
            cells.reapply_authored_ids_where_empty(&mut map);
            // FALSE-NEURON GUARD (set-dressing F2): a painted `O` that no puzzle entity
            // backs still carries tag 0 here — indistinguishable from node 0, so the
            // renderer drew it as a lit puzzle node. Mark it inert. Runs after the
            // authored-id re-apply so an explicitly authored tag (the editor's
            // inspect-`n`) still wins.
            Self::tag_unbacked_objects(&mut map, &objects);
        }
        map
    }

    /// The authored `[holo]` sections as the engine's [`HoloObject`] stamps.
    fn holo_objects(&self) -> Vec<HoloObject> {
        self.holos
            .iter()
            .map(|h| HoloObject {
                shape: h.shape,
                x0: h.x0,
                y0: h.y0,
                z0: h.z0,
                x1: h.x1,
                y1: h.y1,
                z1: h.z1,
                style: h.style,
            })
            .collect()
    }

    /// Second zone pass for `[cells]`-authored geometry (see the call site in
    /// [`build_map`](Self::build_map)). A cell is eligible when it is plain GEOMETRY
    /// (`WallSolid`/`WallTrace`/`WallMatrix`/`Floor`/`Ceiling` — never a node/door
    /// `Object`, a `Hologram` or a `Portal`, whose `side_data` means something else)
    /// and its `side_data` is still `0`, i.e. no earlier pass (zone, pipe, node,
    /// `[color_cell_ids]`) claimed it. Eligibility is decided BEFORE any write, and the
    /// zones are applied in authored order, so among overlapping zones the LATER one
    /// wins — the same precedence as [`stamp_zones`](Self::stamp_zones). Cells the
    /// first pass already tagged are untouched, so a level whose zones only cover the
    /// procedural shell renders exactly as before.
    ///
    /// NOTE HOSTS are exempt: the wall cell a `[note]` placard will mount on keeps tag
    /// `0`, because the placard binder (`re_runtime::notes`, mirrored by
    /// `re_lvlcheck`'s `notes` rule) only paints UNTAGGED masonry. A placard is a point
    /// element authored ON that wall, so it is more specific than a region skin — the
    /// same precedence `[color_cell_ids]` has over a zone. Without the exemption, making
    /// zones reach `[cells]` geometry left L11's two monolith notes and L12's loop-column
    /// note with no paintable wall at all (lvlcheck FAIL: "this placard CANNOT render").
    fn stamp_zones_onto_cells_geometry(&self, map: &mut Map) {
        let (sx, sy, sz) = map.size();
        let note_hosts = self.note_host_cells(map);
        let eligible = |map: &Map, x: i32, y: i32, z: i32| {
            !note_hosts.contains(&(x, y, z))
                && matches!(
                    map.cell_at(x, y, z),
                    CellKind::WallSolid
                        | CellKind::WallTrace
                        | CellKind::WallMatrix
                        | CellKind::Floor
                        | CellKind::Ceiling
                )
                && map.side_data_at(x, y, z) == 0
        };
        // (cell, id) writes, collected first so eligibility is judged on the map as
        // the `[cells]` pass left it, not as earlier zones in this loop rewrote it.
        let mut writes: std::collections::BTreeMap<Coord3, u16> = std::collections::BTreeMap::new();
        for zone in &self.overlay.zones {
            let Some(id) = self.zone_material_id(zone) else {
                continue;
            };
            let Region::Box { min, max } = zone.region else {
                continue;
            };
            let xa = min.0.min(max.0).max(0);
            let ya = min.1.min(max.1).max(0);
            let za = min.2.min(max.2).max(0);
            let xb = min.0.max(max.0).min(sx as i32 - 1);
            let yb = min.1.max(max.1).min(sy as i32 - 1);
            let zb = min.2.max(max.2).min(sz as i32 - 1);
            for z in za..=zb {
                for y in ya..=yb {
                    for x in xa..=xb {
                        if writes.contains_key(&(x, y, z)) || eligible(map, x, y, z) {
                            writes.insert((x, y, z), id); // later zone overwrites
                        }
                    }
                }
            }
        }
        for ((x, y, z), id) in writes {
            let kind = map.cell_at(x, y, z);
            map.set_cell(x as usize, y as usize, z as usize, kind, id);
        }
    }

    /// The wall cells the note binder will mount this level's `[note]` placards on (at
    /// slide offset 0) — see [`stamp_zones_onto_cells_geometry`](Self::stamp_zones_onto_cells_geometry).
    ///
    /// Reproduces the binder's two pinned steps: the placard sits at the reader's EYE
    /// LAYER (`k·storey_height + 1` for the note's storey `k`, stepping down a storey
    /// while that layer is above the habitable top `sz-2`), on the first masonry cell
    /// along the reader's gaze (`-facing`) from the note's `(x, y)` — searched in the
    /// authored plane and, as the binder's fallback, in the eye-layer plane. Every
    /// candidate is returned; exempting one extra wall cell from a skin is harmless,
    /// missing the real host is not.
    pub(crate) fn note_host_cells(&self, map: &Map) -> HashSet<Coord3> {
        let mut hosts = HashSet::new();
        if self.notes.is_empty() {
            return hosts;
        }
        let (_, _, sz) = map.size();
        let h = self.storey_height().max(1) as i32;
        let top = (sz as i32 - 2).max(1);
        let is_masonry = |x: i32, y: i32, z: i32| {
            map.in_bounds(x, y, z)
                && matches!(
                    map.cell_at(x, y, z),
                    CellKind::WallSolid | CellKind::WallTrace | CellKind::WallMatrix
                )
        };
        for note in &self.notes {
            // `facing` is a yaw column (480 per turn); same quadrants as the binder.
            let (dx, dy) = match note.facing.rem_euclid(480) {
                60..=179 => (1, 0),
                180..=299 => (0, -1),
                300..=419 => (-1, 0),
                _ => (0, 1),
            };
            let gaze_hit = |z: i32| -> Option<(i32, i32)> {
                (0..=NOTE_HOST_GAZE_MAX)
                    .map(|step| (note.x - dx * step, note.y - dy * step))
                    .take_while(|&(x, y)| map.in_bounds(x, y, z))
                    .find(|&(x, y)| is_masonry(x, y, z))
            };
            let mut storey = note.z.max(0) / h;
            while storey > 0 && storey * h + 1 > top {
                storey -= 1;
            }
            let eye = (storey * h + 1).clamp(1, top);
            for plane in [note.z, eye] {
                if let Some((x, y)) = gaze_hit(plane) {
                    hosts.insert((x, y, eye));
                }
            }
        }
        hosts
    }

    /// Give every `Object` cell that no puzzle entity backs the
    /// [`INERT_OBJECT_TAG`] (see its docs). `objects` is the node + door anchor list
    /// `build_map` stamped; a cell at one of those positions is a real puzzle anchor
    /// and keeps its tag (node 0 legitimately carries `0`; the door now carries
    /// [`DOOR_OBJECT_TAG`], not `0`). Only cells whose tag is still `0` are touched,
    /// so an authored non-zero tag is never overridden.
    fn tag_unbacked_objects(map: &mut Map, objects: &[ObjectCell]) {
        let anchors: HashSet<Coord3> = objects.iter().map(|o| (o.x, o.y, o.z)).collect();
        let (sx, sy, sz) = map.size();
        for z in 0..sz {
            for y in 0..sy {
                for x in 0..sx {
                    let c = (x as i32, y as i32, z as i32);
                    if map.cell_at(c.0, c.1, c.2) == CellKind::Object
                        && map.side_data_at(c.0, c.1, c.2) == 0
                        && !anchors.contains(&c)
                    {
                        map.set_cell(x, y, z, CellKind::Object, INERT_OBJECT_TAG);
                    }
                }
            }
        }
    }

    /// The anchor every relative asset path in this level resolves against: the
    /// `.lvl`'s own path.
    ///
    /// This is the whole fix for the CWD problem in one accessor. A level knows
    /// where it lives; the process working directory does not — it is set by whoever
    /// launched the game (Explorer, a shortcut, a PATH install, `cargo test`). So
    /// `[texture] file`, `[sky] cylinder`, `[vox] file` and `[inset] child` are all
    /// resolved through [`assets::resolve_level_asset`] against THIS, not against the
    /// CWD. `None` (a [`load_string`] level with no file behind it) leaves the CWD as
    /// the only anchor, which is the historical behaviour for that case.
    fn asset_base(&self) -> Option<&Path> {
        self.source_path.as_deref()
    }

    /// Resolve this level's authored `[inset]`s into render-ready
    /// [`ResolvedInset`]s (Scalable Material Overlay P4). For each inset:
    ///
    /// 1. **Load the child scene** named by [`InsetContent::ChildScene`], resolved
    ///    RELATIVE to this level's [`source_path`](Level::source_path) directory (or the
    ///    CWD when loaded from a string), guarded by `in_flight` — if the child's
    ///    canonical path is already being loaded (a self- or mutually-referential
    ///    chain) OR the child fails to load, the inset falls back to
    ///    [`INSET_FALLBACK_RGB`] and is logged, NEVER recursing. **One level deep only:**
    ///    we load the child to read its representative colour but DO NOT recurse into the
    ///    child's own `[inset]`s in P4 (documented deferral — full child-scene
    ///    compositing + nested insets are a later phase).
    /// 2. **Representative colour:** the child's FIRST `[material]` base `fg`; else its
    ///    `[palette]` fog tint (when `has_fog`); else [`INSET_FALLBACK_RGB`].
    ///    Deterministic.
    /// 3. **Phase-band** from `at_octave` — see [`phase_band_for_octave`] for the chosen
    ///    deterministic mapping (octaves are indistinguishable in the seamless loop, so
    ///    the inset reveals during a distinct PHASE-BAND of `[0, LN2)`, not an octave
    ///    NUMBER).
    ///
    /// DETERMINISM (spec DP7): every step is a pure function of the authored inset + the
    /// (deterministically-loaded) child scene, so the same parent always yields the same
    /// `Vec<ResolvedInset>`.
    fn resolve_insets(&self, in_flight: &mut HashSet<PathBuf>) -> Vec<ResolvedInset> {
        // Inset children resolve exactly like every other level asset: the parent
        // file's own directory FIRST (which is what every shipped `[inset] child =`
        // relies on), then the level tree's parent, then the CWD — see
        // `assets::resolve_level_asset`.
        let base = self.asset_base();

        let mut out = Vec::with_capacity(self.overlay.insets.len());
        for inset in &self.overlay.insets {
            let InsetContent::ChildScene(child_ref) = &inset.content;
            let rgb = self.resolve_inset_colour(child_ref, base, in_flight);
            let phase_band = phase_band_for_octave(inset.at_octave);
            out.push(ResolvedInset {
                region: inset.region.clone(),
                phase_band,
                rgb,
            });
        }
        out
    }

    /// Load the inset's child scene (cycle-guarded, one level deep) and distil its
    /// representative colour. Returns [`INSET_FALLBACK_RGB`] on an empty reference, a
    /// cycle, a load error, or a colourless child — every path is DEFINED, never a panic
    /// or infinite recursion.
    fn resolve_inset_colour(
        &self,
        child_ref: &str,
        base: Option<&Path>,
        in_flight: &mut HashSet<PathBuf>,
    ) -> u32 {
        if child_ref.is_empty() {
            // An inert/default inset reference — nothing to load.
            return INSET_FALLBACK_RGB;
        }
        let child_path = match assets::resolve_level_asset(base, child_ref) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("re_level: inset child {e} (fallback colour)");
                return INSET_FALLBACK_RGB;
            }
        };
        let key = canonical_or_raw(&child_path);
        // CYCLE GUARD: if this child is already in-flight (self- or mutually-referential
        // chain), do NOT load it again — that is the infinite-recursion case. Skip with
        // the fallback colour and a note.
        if in_flight.contains(&key) {
            eprintln!(
                "re_level: inset child '{child_ref}' forms a load cycle — skipping (fallback colour)"
            );
            return INSET_FALLBACK_RGB;
        }
        let child = match load(&child_path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!(
                    "re_level: inset child '{child_ref}' failed to load: {e} (fallback colour)"
                );
                return INSET_FALLBACK_RGB;
            }
        };
        // Mark in-flight for the duration of THIS child's resolution. We do not recurse
        // into the child's own insets in P4 (one level deep), so the guard is mainly a
        // belt-and-braces against a child whose own resolution would re-enter (it won't,
        // here) and a precise record for a future nested-compositing phase. Insert/remove
        // keeps the guard accurate without leaking the key to sibling insets.
        in_flight.insert(key.clone());
        let rgb = representative_colour(&child);
        in_flight.remove(&key);
        rgb
    }

    /// Stamp every pipe's expanded `path` into `map` as visible, tagged
    /// [`CellKind::WallTrace`] cells (called by [`build_map`] only when
    /// [`render_pipes`](Self::render_pipes) is set). Each cell of pipe `i` is tagged
    /// `side_data = re_core::encode_pipe_side_data(i)` (the high-bit pipe namespace,
    /// disjoint from R1 colour ids), so the renderer routes it to the state bus and
    /// `re_runtime::state_bridge` can drive its powered/flow word keyed by the pipe index.
    ///
    /// SAFETY / non-destructive: a path cell is stamped ONLY if it is currently
    /// [`CellKind::Empty`] — a pipe never clobbers a wall, node, door, or another
    /// pipe (the first pipe to claim a shared cell keeps it). Out-of-bounds cells
    /// are skipped by `set_cell`'s bounds guard. The pipe index is masked to the
    /// 15-bit tag space; v0.1 levels have far fewer pipes than that.
    fn stamp_pipes(&self, map: &mut Map) {
        for (i, pipe) in self.pipes.iter().enumerate() {
            let tag = re_core::encode_pipe_side_data(i as u16);
            for &(x, y, z) in &pipe.path {
                if x < 0 || y < 0 || z < 0 {
                    continue;
                }
                // Only fill genuine air, so the trace never replaces existing
                // geometry (walls/nodes/door/an earlier pipe).
                if map.cell_at(x, y, z) != CellKind::Empty {
                    continue;
                }
                map.set_cell(x as usize, y as usize, z as usize, CellKind::WallTrace, tag);
            }
        }
    }

    /// Stamp every `[vox]` model into `map` (called by [`build_map`]). Loads each
    /// `.vox` via `dot_vox`, downscales by `scale`, offsets, and writes each solid
    /// voxel as the model's [`VoxMaterial`] solid cell (tagged `side_data = 0`).
    /// Cells outside the room are CLIPPED. Returns the first load error
    /// encountered (so [`build_map_checked`](Self::build_map_checked) can surface
    /// it); [`build_map`](Self::build_map) ignores the result and stamps best-effort.
    fn stamp_voxes(&self, map: &mut Map) -> Result<(), String> {
        let (sx, sy, sz) = map.size();
        let mut first_err: Result<(), String> = Ok(());
        let base = self.asset_base();
        for v in &self.voxes {
            let scale = v.scale.max(1) as u32;
            // Resolved against the LEVEL's own location, not the process CWD — see
            // `assets::resolve_level_asset`. A shipped `file = levels/Small….vox`
            // therefore loads from an installed copy or from `cargo test` alike.
            let bytes = match assets::resolve_level_asset(base, &v.file)
                .map_err(|e| e.to_string())
                .and_then(|p| fs::read(&p).map_err(|e| format!("{}: {e}", p.display())))
            {
                Ok(b) => b,
                Err(e) => {
                    if first_err.is_ok() {
                        first_err = Err(format!("[vox {}] cannot read '{}': {e}", v.name, v.file));
                    }
                    continue;
                }
            };
            let cells = match load_vox_solids(&bytes, scale) {
                Ok(c) => c,
                Err(e) => {
                    if first_err.is_ok() {
                        first_err = Err(format!("[vox {}] {e}", v.name));
                    }
                    continue;
                }
            };
            let kind = v.material.cell_kind();
            for (cx, cy, cz) in cells {
                let (x, y, z) = (cx + v.off_x, cy + v.off_y, cz + v.off_z);
                // CLIP: skip any cell that would land outside the room.
                if x < 0 || y < 0 || z < 0 {
                    continue;
                }
                let (ux, uy, uz) = (x as usize, y as usize, z as usize);
                if ux < sx && uy < sy && uz < sz {
                    map.set(ux, uy, uz, kind);
                }
            }
        }
        first_err
    }

    /// Decode each `[texture NAME]` image file into a [`re_core::TextureImage`]
    /// (RGBA8, top-left origin) and push it onto [`Map::textures`], returning the
    /// `name → tex_id` map an image-recipe material binds against.
    ///
    /// Each decode is best-effort: a missing file, an unreadable file, or a bad image
    /// is SKIPPED (no texture pushed, no name binding) so the map still builds and
    /// renders (exactly like a missing `[vox]` model). The error is returned by the
    /// fallible sibling [`decode_textures_checked`](Self::decode_textures_checked),
    /// which [`build_map_checked`](Self::build_map_checked) surfaces. The decoded
    /// pixel buffer is NEVER serialized — only the file path round-trips.
    fn decode_textures(&self, map: &mut Map) -> std::collections::HashMap<String, u32> {
        let mut ids = std::collections::HashMap::new();
        let base = self.asset_base();
        for t in &self.textures {
            // `frames = clip.gif` ⇒ decode the WHOLE animation; otherwise the single
            // `file`. Both land in a `TextureSet`; a still is a one-frame set, so the
            // downstream shape is identical and the still path is byte-identical.
            let decoded = match &t.frames {
                Some(fp) => decode_texture_frames(base, fp, t.max_size),
                None => decode_texture_file(base, &t.file, t.max_size).map(|img| vec![img]),
            };
            match decoded {
                Ok(frames) => {
                    // RELIEF channel (`height = <path>`), best-effort like the colour
                    // decode: a missing/bad relief file leaves the image flat and is
                    // reported by `decode_textures_checked`, rather than losing the
                    // whole texture over a secondary channel.
                    //
                    // ONE relief map serves EVERY frame (the relief belongs to the
                    // surface, not to the moment), so it is decoded once, resampled to
                    // frame 0's dimensions, and `Arc`-shared across the set — never
                    // copied per frame.
                    let shared_height = match (&t.height, frames.first()) {
                        (Some(hp), Some(first)) => {
                            decode_height_map(base, hp, t.max_size, first.width, first.height)
                                .ok()
                                .map(|b| std::sync::Arc::<[u8]>::from(b.into_boxed_slice()))
                        }
                        _ => None,
                    };
                    let frames: Vec<re_core::TextureImage> = match shared_height {
                        Some(h) => frames
                            .into_iter()
                            .map(|img| img.with_shared_height_map(h.clone()))
                            .collect(),
                        None => frames,
                    };
                    let fps = t.fps.unwrap_or(re_core::TEXTURE_SET_DEFAULT_FPS);
                    let id = map.push_texture_set(re_core::TextureSet::new(frames, fps));
                    ids.insert(t.name.clone(), id);
                }
                Err(_) => {
                    // Non-fatal: skip this texture; `decode_textures_checked` reports it.
                }
            }
        }
        ids
    }

    /// The fallible mirror of [`decode_textures`](Self::decode_textures): returns the
    /// FIRST decode error (so [`build_map_checked`](Self::build_map_checked) can
    /// surface a broken `[texture]` file) without mutating a map. A level with no
    /// `[texture]` section never errors.
    fn decode_textures_checked(&self) -> Result<(), String> {
        let base = self.asset_base();
        for t in &self.textures {
            let img = match &t.frames {
                // The animated path decodes every frame; report frame 0's dimensions
                // to the relief check, exactly as `decode_textures` does.
                Some(fp) => decode_texture_frames(base, fp, t.max_size)
                    .map_err(|e| format!("[texture {}] cannot load '{fp}': {e}", t.name))?
                    .into_iter()
                    .next()
                    .ok_or_else(|| format!("[texture {}] '{fp}' decoded no frames", t.name))?,
                None => decode_texture_file(base, &t.file, t.max_size)
                    .map_err(|e| format!("[texture {}] cannot load '{}': {e}", t.name, t.file))?,
            };
            if let Some(hp) = &t.height {
                decode_height_map(base, hp, t.max_size, img.width, img.height)
                    .map_err(|e| format!("[texture {}] cannot load height '{hp}': {e}", t.name))?;
            }
        }
        Ok(())
    }

    /// Best-effort: decode each authored `[sky]` path and call the matching
    /// [`Map`] setter. A missing/bad file is silently skipped — the slot stays
    /// `None` (procedural sky) and the error is surfaced by
    /// [`apply_sky_images_checked`](Self::apply_sky_images_checked). A level
    /// with no `[sky]` section leaves all sky slots at `None` (byte-identical).
    fn apply_sky_images(&self, map: &mut Map) {
        use std::sync::Arc;
        let base = self.asset_base();
        if let Some(p) = &self.sky.cylinder {
            if let Ok(img) = decode_texture_file(base, p, None) {
                map.set_sky_cylinder(Arc::new(img));
            }
        }
        if let Some(p) = &self.sky.cap_top {
            if let Ok(img) = decode_texture_file(base, p, None) {
                map.set_sky_cap_top(Arc::new(img));
            }
        }
        if let Some(p) = &self.sky.cap_bottom {
            if let Ok(img) = decode_texture_file(base, p, None) {
                map.set_sky_cap_bottom(Arc::new(img));
            }
        }
    }

    /// The fallible mirror of [`apply_sky_images`](Self::apply_sky_images):
    /// returns the FIRST decode error so [`build_map_checked`](Level::build_map_checked)
    /// can surface a broken `[sky]` file. A level with no `[sky]` section never errors.
    fn apply_sky_images_checked(&self) -> Result<(), String> {
        let base = self.asset_base();
        if let Some(p) = &self.sky.cylinder {
            decode_texture_file(base, p, None)
                .map_err(|e| format!("[sky] cylinder cannot load '{p}': {e}"))?;
        }
        if let Some(p) = &self.sky.cap_top {
            decode_texture_file(base, p, None)
                .map_err(|e| format!("[sky] cap_top cannot load '{p}': {e}"))?;
        }
        if let Some(p) = &self.sky.cap_bottom {
            decode_texture_file(base, p, None)
                .map_err(|e| format!("[sky] cap_bottom cannot load '{p}': {e}"))?;
        }
        Ok(())
    }

    /// Like [`build_map`](Self::build_map), but returns an error if any `[vox]`
    /// model failed to load (missing file / bad `.vox`). The map is still built
    /// (the failed stamp is simply absent), so the caller can both render and
    /// report. A level with no `[vox]` section never errors.
    pub fn build_map_checked(&self) -> Result<Map, String> {
        // Re-run the same pipeline as build_map, but capture the vox load error.
        // build_map already stamps best-effort; to surface the error we re-derive
        // it cheaply by attempting the loads here (the stamp itself is idempotent
        // on the returned map).
        // Validate-before-accept: an overlay `[zone]` that names a material with no
        // matching `[material NAME]` section is an authoring error. `build_map`
        // skips such a zone with no panic; here we surface it. Checked before the
        // vox loads so a bad zone is reported regardless of `[vox]` presence.
        self.validate_zones()?;
        // Surface a broken `[texture]` file (missing / undecodable) before the vox
        // loads — a bad image is an authoring error worth reporting even though
        // `build_map` itself skips it best-effort.
        self.decode_textures_checked()?;
        // Surface a broken `[sky]` file (missing / undecodable) before the vox
        // loads — same policy as `[texture]`.
        self.apply_sky_images_checked()?;
        let mut map = self.build_map();
        if self.voxes.is_empty() {
            return Ok(map);
        }
        self.stamp_voxes(&mut map)?;
        Ok(map)
    }

    /// Build the engine-agnostic [`Atmosphere`] for this level from its parsed
    /// `[palette]` fog params and `[effects] matrix` knob.
    ///
    /// Each field is forwarded only when the level actually authored it:
    /// * `fog_color` ⇐ `[palette] fog_color` (when `has_fog`),
    /// * `fog_near`/`fog_far` ⇐ `[palette] fog_near`/`fog_far` (when `has_fog_dist`),
    /// * `matrix` ⇐ `[effects] matrix` (when `> 0`).
    ///
    /// A level with no fog and no matrix knob yields [`Atmosphere::default`]
    /// (`is_none()`), so `build_map` leaves the map carrying the engine defaults
    /// and behaviour is unchanged (back-compat with 01-07).
    pub fn atmosphere(&self) -> Atmosphere {
        let mut atmo = Atmosphere::default();
        if self.palette.has_fog {
            atmo.fog_color = Some(self.palette.fog_color);
        }
        if self.palette.has_fog_dist {
            atmo.fog_near = Some(self.palette.fog_near);
            atmo.fog_far = Some(self.palette.fog_far);
        }
        if self.effects.matrix > 0.0 {
            atmo.matrix = Some(self.effects.matrix);
        }
        // §1.3 lighting/atmosphere: forwarded verbatim. All-absent ⇒ the
        // Atmosphere stays `is_none()` and every shipped level is unchanged.
        atmo.lights.clone_from(&self.lights);
        atmo.key_dir = self.atmos.key_dir;
        atmo.key_color = self.atmos.key_color;
        atmo.ambient = self.atmos.ambient;
        atmo.fog_height = self.atmos.fog_height;
        atmo.postfx.clone_from(&self.atmos.postfx);
        atmo
    }

    /// Build the renderer-facing [`MaterialDef`] list for this level.
    ///
    /// Returns either an **empty** list (no `[material]` section ⇒ the renderer
    /// keeps every built-in) OR a **fixed-length, slot-ordered** list with one
    /// entry per surface, in the renderer's built-in slot order:
    /// `wall_solid, wall_trace, wall_matrix, floor, ceiling`. A surface bound to
    /// an authored material gets that material's data; an unbound surface (or one
    /// naming a missing material) gets a [`MaterialDef::fallback`] marker so the
    /// renderer keeps its own built-in for that exact slot. This keeps the
    /// surface→slot dispatch the renderer already uses valid, while letting any
    /// subset of surfaces be re-skinned per level.
    fn material_defs(&self, tex_ids: &std::collections::HashMap<String, u32>) -> Vec<MaterialDef> {
        if self.materials.is_empty() {
            return Vec::new();
        }
        let resolve = |name: &Option<String>| -> MaterialDef {
            match name
                .as_ref()
                .and_then(|n| self.materials.iter().find(|m| &m.name == n))
            {
                Some(m) => Self::material_to_def(m, tex_ids),
                None => MaterialDef::fallback(),
            }
        };
        vec![
            resolve(&self.surface_materials.wall_solid),
            resolve(&self.surface_materials.wall_trace),
            resolve(&self.surface_materials.wall_matrix),
            resolve(&self.surface_materials.floor),
            resolve(&self.surface_materials.ceiling),
        ]
    }

    /// Convert one authored [`LevelMaterial`] into the renderer-facing
    /// [`MaterialDef`], encoding any `preset = NAME` into the name (the only string
    /// a `MaterialDef` carries) exactly like the surface-slot path.
    fn material_to_def(
        m: &LevelMaterial,
        tex_ids: &std::collections::HashMap<String, u32>,
    ) -> MaterialDef {
        // RESOLVE an image recipe's `tex_id` from the material's bound `texture = NAME`
        // against the decoded-texture name→id map. A `recipe = image` whose `texture`
        // name did not decode (missing / failed file, or no `texture =` key) keeps the
        // placeholder `tex_id` — the renderer's `attach_image_textures` then finds no
        // texture at that id and the `Image` recipe self-renders a neutral block (no
        // panic). Any non-image recipe is passed through untouched.
        let recipe = match m.recipe {
            Some(MaterialRecipe::Image { .. }) => {
                let tex_id = m
                    .texture
                    .as_ref()
                    .and_then(|n| tex_ids.get(n))
                    .copied()
                    .unwrap_or(0);
                // Carry the authored WORLD-ANCHORED span (M3a) + SUB-CELL mode (M3b) +
                // FIT mode through to the renderer recipe; `None`/`Off`/`Contain` ⇒
                // the legacy per-cell tiling / solid block / aspect-preserving default.
                Some(MaterialRecipe::Image {
                    tex_id,
                    span: m.image_span,
                    subcell: m.image_subcell,
                    fit: m.image_fit,
                    // TILE (span-relative repeat) + RELIEF (height-mapped lighting):
                    // unauthored ⇒ one copy and no relief, i.e. exactly the recipe
                    // this arm produced before the two keys existed.
                    tile: m.image_tile.unwrap_or(re_world::IMAGE_TILE_NONE),
                    relief: m.image_relief,
                })
            }
            other => other,
        };
        MaterialDef {
            name: match &m.preset {
                Some(p) => format!("{PRESET_NAME_PREFIX}{p}"),
                None => m.name.clone(),
            },
            fg: m.fg,
            bg: m.bg,
            glyph: m.glyph,
            recipe,
            anim: m.anim,
            rank: m.rank,
            reflective: m.reflective,
            clarity: m.clarity,
            fallback: false,
            detail: m.detail,
        }
    }

    /// The full renderer-facing material set: the five surface slots followed by
    /// the per-cell **colour palette** (every authored material, in `materials`
    /// order, so the 1-based `[color_cell_ids]` id `N` resolves to palette entry
    /// `N - 1`). The renderer ([`re_render::registry_for_map`]) reads the first
    /// five entries as the surface-kind dispatch and the remainder as the colour
    /// palette indexed by a geometry cell's `side_data`.
    ///
    /// The palette is appended when the level has `[color_cell_ids]` OR an overlay
    /// `[zone]` (both stamp 1-based ids into `side_data` and SHARE this one palette
    /// / id space — spec DP1-A/DP3-A); a level that authors materials purely for
    /// surface binding (no per-cell ids, no zones) keeps the historical five-slot
    /// list → byte-identical. A level with neither materials nor colour ids nor
    /// zones returns an empty list (renderer keeps its built-ins).
    fn renderer_material_defs(
        &self,
        tex_ids: &std::collections::HashMap<String, u32>,
    ) -> Vec<MaterialDef> {
        let mut defs = self.material_defs(tex_ids);
        // The colour palette (slots 5..) is appended when EITHER `[color_cell_ids]`
        // OR an overlay `[zone]` is present — both stamp 1-based palette ids into
        // `side_data` and SHARE this one palette (DP1-A/DP3-A). A zone-only level
        // (no `[color_cell_ids]`) still needs the palette so its resolved zone-
        // material ids have somewhere to dispatch from. A level with neither keeps
        // the historical five-slot (or empty) list → byte-identical.
        // A `[line]` binding a level-authored material also needs the palette: the
        // line post-pass tints its stroke from that palette slot (set-dressing E0).
        // Props stamp palette ids into cells and lines, so a level with any
        // `[prop]` always carries the palette (their synthesized materials follow it).
        if self.color_cell_ids.is_empty()
            && self.overlay.zones.is_empty()
            && !self.any_line_binds_authored_material()
            && self.props.is_empty()
        {
            return defs;
        }
        // Colour ids reference geometry cells; ensure the surface slots exist so the
        // palette is always appended at index 5.. (the renderer's fixed offset).
        if defs.is_empty() {
            defs = vec![
                MaterialDef::fallback(),
                MaterialDef::fallback(),
                MaterialDef::fallback(),
                MaterialDef::fallback(),
                MaterialDef::fallback(),
            ];
        }
        for m in &self.materials {
            defs.push(Self::material_to_def(m, tex_ids));
        }
        defs
    }

    /// Resolve an overlay [`Zone`]'s material **name** → the **1-based colour
    /// palette id** it stamps into `side_data` — the SAME id space and palette
    /// `[color_cell_ids]` uses (spec DP1-A/DP3-A, P1).
    ///
    /// The renderer-facing palette (`renderer_material_defs`, slots 5..) appends
    /// every authored `[material NAME]` in `self.materials` order, so a material at
    /// index `i` is palette id `i + 1`. A zone naming that material therefore
    /// resolves to `i + 1` — one slot per material, deduplicated by name, shared
    /// with `[color_cell_ids]`. An empty name, or a name with no matching
    /// `[material NAME]` section, returns `None` (an authoring error surfaced by
    /// [`build_map_checked`](Self::build_map_checked); [`build_map`](Self::build_map)
    /// skips that zone with no panic).
    fn zone_material_id(&self, zone: &Zone) -> Option<u16> {
        let name = zone.material.0.trim();
        if name.is_empty() {
            return None;
        }
        let idx = self.materials.iter().position(|m| m.name == name)?;
        // 1-based id; index 0 ⇒ id 1. `materials` length is far below u16::MAX.
        Some((idx + 1) as u16)
    }

    /// Resolve a `[line]`'s material **name** → the 1-based colour-palette id of the
    /// level-authored `[material NAME]` it binds (the same id space as
    /// [`zone_material_id`](Self::zone_material_id)). `None` for an empty name or a
    /// name that is not a level-authored material — e.g. a catalog name such as
    /// `rust`, which the renderer resolves by itself.
    fn line_material_id(&self, line: &Line) -> Option<u16> {
        let name = line.material.0.trim();
        if name.is_empty() {
            return None;
        }
        let idx = self.materials.iter().position(|m| m.name == name)?;
        Some((idx + 1) as u16)
    }

    /// Whether any `[line]` binds a level-authored material (so the palette must be
    /// appended for the renderer to resolve its colour).
    fn any_line_binds_authored_material(&self) -> bool {
        self.overlay
            .lines
            .iter()
            .any(|l| self.line_material_id(l).is_some())
    }

    /// Per-line resolved palette ids, parallel to `overlay.lines` (`0` = unresolved),
    /// or `None` when no line binds a level-authored material.
    fn line_palette_ids(&self) -> Option<Vec<u16>> {
        if !self.any_line_binds_authored_material() {
            return None;
        }
        Some(
            self.overlay
                .lines
                .iter()
                .map(|l| self.line_material_id(l).unwrap_or(0))
                .collect(),
        )
    }

    /// Stamp every overlay [`Zone`]'s resolved colour-palette id into the
    /// `side_data` of the SOLID cells inside its [`Region::Box`] (clamped to map
    /// bounds) — the P1 region-fill (spec §6 P1, DP1-A/DP3-A).
    ///
    /// PRECEDENCE: this is called BEFORE the `[color_cell_ids]` loop, so an explicit
    /// per-cell id (more specific) OVERWRITES a zone region-fill. Among overlapping
    /// zones, the LATER-listed zone wins (last-writer; the iteration order is the
    /// authored `[zone]` order). Both rules are pinned by unit tests.
    ///
    /// SOLID-ONLY: only a non-[`Empty`](CellKind::Empty) cell is stamped — a zone
    /// material on air is inert, so air is preserved (a floor-zone tints the floor
    /// voxels, never the empty space above them). Out-of-bounds / negative cells are
    /// skipped (mirrors the `[color_cell_ids]` guard via `set_cell`'s bounds guard).
    ///
    /// A zone whose material name does not resolve (missing `[material NAME]`) is
    /// skipped here with no panic; [`build_map_checked`](Self::build_map_checked)
    /// surfaces it as an authoring error.
    fn stamp_zones(&self, map: &mut Map) {
        let (sx, sy, sz) = map.size();
        for zone in &self.overlay.zones {
            let Some(id) = self.zone_material_id(zone) else {
                continue; // unbound / unresolved material → inert (validated elsewhere)
            };
            // `Region` is `#[non_exhaustive]` (a future analytic variant is P1+);
            // P1 fills only the discrete voxel box. A non-Box region is skipped.
            let Region::Box { min, max } = zone.region else {
                continue;
            };
            let (x0, x1) = (min.0.min(max.0), min.0.max(max.0));
            let (y0, y1) = (min.1.min(max.1), min.1.max(max.1));
            let (z0, z1) = (min.2.min(max.2), min.2.max(max.2));
            // Clamp the inclusive box to [0, size) on each axis; an entirely
            // out-of-bounds box yields an empty loop (nothing stamped).
            let xa = x0.max(0);
            let ya = y0.max(0);
            let za = z0.max(0);
            let xb = x1.min(sx as i32 - 1);
            let yb = y1.min(sy as i32 - 1);
            let zb = z1.min(sz as i32 - 1);
            for z in za..=zb {
                for y in ya..=yb {
                    for x in xa..=xb {
                        let kind = map.cell_at(x, y, z);
                        // SOLID-ONLY: never tint air (a zone material on Empty is
                        // inert; preserve the air cell exactly).
                        if kind == CellKind::Empty {
                            continue;
                        }
                        map.set_cell(x as usize, y as usize, z as usize, kind, id);
                    }
                }
            }
        }
    }

    /// Validate the overlay's zones against this level's authored materials: every
    /// zone with a non-empty `material` name MUST name an existing `[material NAME]`
    /// section. Returns the first offending zone as an error string; `Ok(())` if all
    /// zones resolve (or the level has none). Used by
    /// [`build_map_checked`](Self::build_map_checked) — validate-before-accept.
    fn validate_zones(&self) -> Result<(), String> {
        for zone in &self.overlay.zones {
            let name = zone.material.0.trim();
            if name.is_empty() {
                continue; // an unbound zone is inert, not an error
            }
            if !self.materials.iter().any(|m| m.name == name) {
                return Err(format!(
                    "[zone {}] references unknown material '{name}' (no matching [material {name}] section)",
                    zone.name
                ));
            }
        }
        Ok(())
    }

    /// The camera spawn pose from `[player]`.
    pub fn spawn_camera(&self) -> CameraPose {
        CameraPose::spawn_at(
            self.player.x,
            self.player.y,
            self.player.z,
            self.player.facing_yaw_col,
        )
    }
}

/// Load and parse a `.lvl` file from disk.
/// Canonicalise `p` for the cycle guard, falling back to the raw path when the file
/// does not yet exist / cannot be canonicalised. Two references to the same file thus
/// compare equal regardless of `./` or `..` spelling; a missing file still gets a stable
/// key (its raw path) so a self-reference by the same literal path is still caught.
fn canonical_or_raw(p: &Path) -> PathBuf {
    std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf())
}

/// The inset's **representative colour** (packed `0x00RR_GGBB`) — the DETERMINISTIC
/// colour the bounded phase-band reveal paints. The fallback chain (spec P4):
///
/// 1. the child's FIRST `[material]` base `fg`;
/// 2. else the child's `[palette]` fog tint (when `has_fog`);
/// 3. else [`INSET_FALLBACK_RGB`] (a documented neutral grey).
///
/// Masked to 24 bits so a stray alpha/high byte in an authored colour can never leak
/// into the packed RGB the renderer unpacks.
fn representative_colour(child: &Level) -> u32 {
    if let Some(m) = child.materials.first() {
        return m.fg & 0x00FF_FFFF;
    }
    if child.palette.has_fog {
        return child.palette.fog_color & 0x00FF_FFFF;
    }
    INSET_FALLBACK_RGB
}

/// Map an authored inset `at_octave = (min, max)` to a DETERMINISTIC reveal window
/// `[lo, hi)` in reduced zoom phase `[0, LN2)` (Scalable Material Overlay P4).
///
/// ## Why a phase-band (the approved honesty)
///
/// The infinite-zoom field is PERIODIC over one octave (`LN2`) — that periodicity IS the
/// seamless loop, so octave N is byte-identical to octave N+1 and there is no "current
/// octave" counter. An inset therefore cannot key off a distinguishable octave NUMBER;
/// it reveals during a PHASE-BAND of the `[0, LN2)` loop. We map `at_octave` to a
/// distinct band so different authored octaves land at distinct, repeatable phases.
///
/// ## The mapping (deterministic, documented)
///
/// * `lo = (at_octave.0 * 0.382 * LN2) mod LN2` — the golden-ratio fraction `0.382`
///   (`1 - 1/φ`) spreads successive octave indices to well-separated phases, so several
///   insets at different octaves rarely overlap.
/// * `width = clamp((max - min + 1) * 0.2 * LN2, _, LN2)` — a wider authored octave
///   range opens a wider reveal window; a single octave (`min == max`) gives a `0.2·LN2`
///   window (~20% of the loop visible). `(max - min + 1)` is at least 1.
/// * `hi = (lo + width) mod LN2` — the band may wrap past the `LN2` seam; the membership
///   test ([`ResolvedInset::phase_in_band`]) handles `lo > hi`.
fn phase_band_for_octave(at_octave: (i32, i32)) -> (f32, f32) {
    let (min, max) = at_octave;
    let lo = (min as f32 * 0.382 * LN2).rem_euclid(LN2);
    let span = (max - min + 1).max(1) as f32;
    let width = span * 0.2 * LN2;
    // A band at/over a full octave covers the WHOLE loop — represent it as the explicit
    // `[0, LN2)` always-on window so the wraparound test never collapses to an
    // empty `lo == hi` band (which `(lo + LN2).rem_euclid(LN2) == lo` would produce).
    if width >= LN2 {
        return (0.0, LN2);
    }
    let hi = (lo + width).rem_euclid(LN2);
    (lo, hi)
}

/// Parse a `.lvl` file from disk into a [`Level`].
///
/// The main entry point of the crate: reads `path`, parses it with
/// [`load_string`], and records the source path on the returned [`Level`] (so
/// [`Level::build_map`] can resolve `[inset]` child-scene paths relative to
/// this file's directory). A missing or unreadable file is a [`ParseError`]
/// at line `0`; a malformed section is a line-numbered [`ParseError`] from
/// the parser itself.
pub fn load(path: impl AsRef<Path>) -> Result<Level, ParseError> {
    let path = path.as_ref();
    let src = path.display().to_string();
    let contents = fs::read_to_string(path).map_err(|e| ParseError {
        src: src.clone(),
        line: 0,
        msg: format!("cannot read file: {e}"),
    })?;
    let mut level = load_string(&contents, &src)?;
    // Record the source path so `build_map` can resolve `[inset]` child-scene paths
    // relative to this file's directory and seed the cycle guard with the parent.
    level.source_path = Some(path.to_path_buf());
    Ok(level)
}

// ---- internal parsing ------------------------------------------------------

/// Which section the parser is currently inside.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Section {
    None,
    Meta,
    Room,
    Node,
    Pipe,
    Part,
    Emit,
    Inputs,
    Pedestal,
    Door,
    Sequence,
    Player,
    WallDecal,
    Holo,
    Portal,
    Vox,
    /// `[texture NAME]` — an image texture (`file = …`) decoded at build time.
    Texture,
    /// `[sky]` — image-cubemap paths (`cylinder`/`cap_top`/`cap_bottom` keys).
    /// Decoded at `build_map` time; procedural sky is unchanged when absent.
    Sky,
    ZoneInput,
    /// `[trigger NAME]` — a trigger-field (region + action + once).
    Trigger,
    Effects,
    Palette,
    /// `[atmosphere]` — per-level key light, ambient, height fog, postfx preset
    /// (and fog-distance aliases for the `[palette]` keys).
    Atmosphere,
    /// `[light]` — one repeatable punctual light.
    Light,
    Note,
    Material,
    Emitter,
    /// `[color_cell_ids]` — per-cell colour-material id assignments
    /// (`x,y,z = id`). The data side of the colour→material substrate.
    ColorCellIds,
    /// `[cells]` — explicit dense per-cell geometry (the editor save format).
    /// ASCII rows (one block per z-layer, blank line between layers) plus an
    /// optional `material_ids =` sub-table and a reserved `shapes =` sub-table.
    Cells,
    /// `[zone NAME]` — a Scalable Material Overlay material zone (P0).
    Zone,
    /// `[line NAME]` — a Scalable Material Overlay crisp line (P0).
    Line,
    /// `[inset NAME]` — a Scalable Material Overlay inset/"врезка" (P0).
    Inset,
    /// `[hierarchy]` — object-nesting `(child) -> (parent)` links. Each line is
    /// `child_kind child_index -> parent_kind parent_index`.
    Hierarchy,
    /// `[prop NAME]` — one set-dressing prop (set-dressing §3.2).
    Prop,
    /// `[field NAME]` — one sound field.
    Field,
    /// `[map ...]` / `[win_when]` — read-and-skip block sections.
    Skip,
}

impl Section {
    /// The section name as it is written in the `.lvl`, for diagnostics
    /// ([`UnknownKey::section`]). Total — every variant has one, so adding a
    /// section forces a decision here rather than defaulting to a blank label.
    const fn label(&self) -> &'static str {
        match self {
            Section::None => "?",
            Section::Meta => "meta",
            Section::Room => "room",
            Section::Node => "node",
            Section::Pipe => "pipe",
            Section::Part => "part",
            Section::Emit => "emit",
            Section::Inputs => "inputs",
            Section::Pedestal => "pedestal",
            Section::Door => "door",
            Section::Sequence => "sequence",
            Section::Player => "player",
            Section::WallDecal => "wall_decal",
            Section::Holo => "holo",
            Section::Portal => "portal",
            Section::Vox => "vox",
            Section::Texture => "texture",
            Section::Sky => "sky",
            Section::ZoneInput => "zone_input",
            Section::Trigger => "trigger",
            Section::Effects => "effects",
            Section::Palette => "palette",
            Section::Atmosphere => "atmosphere",
            Section::Light => "light",
            Section::Note => "note",
            Section::Material => "material",
            Section::Emitter => "emitter",
            Section::ColorCellIds => "color_cell_ids",
            Section::Cells => "cells",
            Section::Zone => "zone",
            Section::Line => "line",
            Section::Inset => "inset",
            Section::Hierarchy => "hierarchy",
            Section::Prop => "prop",
            Section::Field => "field",
            Section::Skip => "skip",
        }
    }
}

/// Parse a `.lvl` document from a string. `src` is used only in error messages.
pub fn load_string(contents: &str, src: &str) -> Result<Level, ParseError> {
    let mut p = Parser::new(src);
    p.run(contents)?;
    p.finalize()
}

/// A `[emit]` clause awaiting node/part name resolution in `finalize`.
struct PendingEmit {
    source: String,
    parts: Vec<String>,
}

/// Mutable parser state assembled while walking the file once.
struct Parser<'a> {
    src: &'a str,
    level: Level,
    room_set: bool,
    meta_seen: bool,
    pedestal_set: bool,
    door_set: bool,
    player_set: bool,
    /// raw `[door] target_parts` node names (resolved to indices in finalize).
    door_target_names: Vec<String>,
    /// raw `[door] target_off` node names — the opt-in INVERTED targets (L12
    /// `exit 0`), resolved to indices in finalize. Empty ⇒ no inverted targets
    /// (byte-identical default).
    door_target_off_names: Vec<String>,
    /// raw `[sequence] observe` node names — the fixed observation window
    /// (resolved to indices in finalize). Empty ⇒ no `[sequence]` declared.
    seq_observe_names: Vec<String>,
    /// raw `[sequence] step` rows, each a list of node names that must be ON at
    /// that step (every other observed node must be OFF). Order = step order.
    seq_step_names: Vec<Vec<String>>,
    /// True once a `[sequence]` header was opened (so finalize can distinguish a
    /// declared-but-empty sequence from no sequence at all).
    seq_declared: bool,
    /// Per-`[trigger]` raw `ActivateNode` node name awaiting resolution in
    /// `finalize` (parallel to `level.triggers`). `None` ⇒ the trigger is an
    /// `Event` action (no node to resolve). Index `i` matches `triggers[i]`.
    trigger_node_names: Vec<Option<String>>,
    /// raw `[emit NAME=val]` source + part-name list (resolved in finalize).
    pending_emits: Vec<PendingEmit>,
    /// Part mask collection cursor for the current `[part]`.
    mask_y: i32,
    mask_z: i32,
    /// note-body collection for the current `[note]`.
    note_body: String,
    note_started: bool,
    /// `[cells]` scratch: the grid being assembled (size + dense kinds), the
    /// current ASCII-row cursor `(y, z)`, and which sub-block we are in.
    cells_grid: Option<CellGrid>,
    cells_y: usize,
    cells_z: usize,
    cells_sub: CellsSub,
}

/// Which sub-block of a `[cells]` section the parser is currently reading.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CellsSub {
    /// ASCII glyph rows (the default after `size =`).
    Rows,
    /// `material_ids =` table: `x,y,z = id` lines.
    MaterialIds,
    /// `shapes =` table (reserved M3): `x y z shape dir` lines.
    Shapes,
}

impl<'a> Parser<'a> {
    fn new(src: &'a str) -> Self {
        Parser {
            src,
            level: Level::default(),
            room_set: false,
            meta_seen: false,
            pedestal_set: false,
            door_set: false,
            player_set: false,
            door_target_names: Vec::new(),
            door_target_off_names: Vec::new(),
            seq_observe_names: Vec::new(),
            seq_step_names: Vec::new(),
            trigger_node_names: Vec::new(),
            seq_declared: false,
            pending_emits: Vec::new(),
            mask_y: 0,
            mask_z: 0,
            note_body: String::new(),
            note_started: false,
            cells_grid: None,
            cells_y: 0,
            cells_z: 0,
            cells_sub: CellsSub::Rows,
        }
    }

    fn err(&self, line: usize, msg: impl Into<String>) -> ParseError {
        err_at(self.src, line, msg)
    }

    /// Walk every line, dispatching by section.
    fn run(&mut self, contents: &str) -> Result<(), ParseError> {
        let mut sec = Section::None;
        for (i, raw) in contents.lines().enumerate() {
            let line_no = i + 1;
            // strip trailing CR (CRLF tolerance)
            let raw = raw.strip_suffix('\r').unwrap_or(raw);

            // --- [note] body capture: verbatim until next header ---
            //
            // ONE exception to "verbatim": a line whose FIRST COLUMN is `#` is a
            // level-file COMMENT, not placard text. The body runs until the next
            // section header, so every dev comment an author wrote *after* the note
            // (the usual `# ---- nodes ----` banner before `[node …]`) used to be
            // swallowed into the note and PRINTED ON THE WALL — including `{time}`
            // and friends, which the placard interpolator then expanded. Column 0 is
            // the discriminator on purpose: authored body lines are indented (that is
            // how `[note]` bodies are written throughout `levels/`), and a MID-LINE
            // `#` ("MINISTRY CORPUS #9") must survive untouched — which it does, since
            // note bodies are never run through `strip_comment`. `re_voximport::to_lvl_text`
            // escapes any AUTHORED line that would otherwise collide with this rule —
            // see [`escape_note_body_line`] — so this discriminator never eats real text.
            //
            // The SAME column-0 rule governs `[`: a bracketed line ends the body only
            // when it starts in column 0. An INDENTED bracketed line is placard text.
            //
            // This used to be `!raw.trim_start().starts_with('[')`, i.e. ANY bracketed
            // line ended the note — which ate `levels/L7-hidden-depths.lvl:369`:
            //
            // ```text
            //     [ STEWARDSHIP - DO NOT POWER DOWN ]
            // ```
            //
            // an authored placard line that terminated its note, was dropped from the
            // note text, and opened a bogus section called `STEWARDSHIP`. No player has
            // ever seen it. Found by the unknown-key diagnostic; it is the same
            // "authored, parsed, never on screen" defect as the other six, except this
            // one was eaten by the grammar rather than by the renderer.
            //
            // Column 0 is not a new concept here — it is exactly the discriminator the
            // body already uses for `#`. A grammar that says "column 0 means structure,
            // indented means content" for one character should say it for the other.
            // The runaway-note protection is untouched: every real section header in
            // every shipped level (and every header `re_voximport` writes) is at column
            // 0, so an unterminated note still cannot swallow the rest of the file.
            // [`escape_note_body_line`] escapes BOTH markers on the way out, so an
            // authored column-0 `[` round-trips instead of being lost on the next load.
            if sec == Section::Note && !raw.starts_with('[') {
                if note_body_line_is_comment(raw) {
                    continue;
                }
                if self.note_started {
                    self.note_body.push('\n');
                }
                self.note_body.push_str(raw);
                self.note_started = true;
                continue;
            }

            // --- [part] mask rows: keep '#' intact (it is the solid glyph) ---
            if sec == Section::Part && self.is_mask_row(raw) {
                self.consume_mask_row(line_no, raw)?;
                continue;
            }

            // --- [cells] ASCII geometry rows: keep glyphs intact (no trim) ---
            if sec == Section::Cells && self.cells_sub == CellsSub::Rows && self.is_cells_row(raw) {
                self.consume_cells_row(line_no, raw)?;
                continue;
            }

            let line = strip_comment(raw).trim();
            if line.is_empty() {
                // a blank line inside a part mask advances to the next z layer
                if sec == Section::Part && self.mask_y > 0 {
                    self.mask_y = 0;
                    self.mask_z += 1;
                }
                // a blank line inside a [cells] row block advances the z layer
                if sec == Section::Cells && self.cells_sub == CellsSub::Rows && self.cells_y > 0 {
                    self.cells_y = 0;
                    self.cells_z += 1;
                }
                continue;
            }

            // --- section header? ---
            if let Some(inside) = section_header(line) {
                self.flush_section(&sec);
                sec = self.open_section(line_no, inside)?;
                continue;
            }

            // --- [cells] shapes sub-table: `x y z shape dir` (no `=`) ---
            // A line WITH `=` is never a shape row: it is a sub-table header
            // (`material_ids =` / `shapes =`) and falls through to the key handler,
            // which switches sub-table. Without this a `material_ids =` placed after
            // `shapes =` was swallowed as a malformed shape row — a parse error — so
            // the two sub-tables only worked in one order (set-dressing §0.1).
            if sec == Section::Cells && self.cells_sub == CellsSub::Shapes && !line.contains('=') {
                self.consume_cells_shape(line_no, line)?;
                continue;
            }

            // --- [hierarchy] link: `child_kind idx -> parent_kind idx` (no `=`) ---
            if sec == Section::Hierarchy {
                self.consume_hierarchy_line(line_no, line)?;
                continue;
            }

            // --- key = value? ---
            if let Some((k, v)) = split_kv(line) {
                self.handle_kv(&sec, line_no, k, v)?;
                continue;
            }
            // other block lines (map/win_when) are ignored
        }
        self.flush_section(&sec);
        Ok(())
    }

    /// Heuristic: is `raw` a `[part]` mask row (matches the part's width and
    /// consists only of `#`/`.`/`0`/`1`)? Mirrors legacy pass-2 mask detection.
    fn is_mask_row(&self, raw: &str) -> bool {
        let Some(part) = self.level.parts.last() else {
            return false;
        };
        if part.w <= 0 {
            return false;
        }
        if raw.len() != part.w as usize {
            return false;
        }
        // NB: we intentionally do NOT gate on `mask_z < part.d` here. A
        // width-matching glyph row that arrives after `d` layers are full is an
        // OVERFLOW — it must still route to `consume_mask_row`, which raises the
        // "[part] mask too many rows for d" error (R-M8). Gating it out here
        // silently dropped the extra row instead.
        raw.chars().all(|c| matches!(c, '#' | '.' | '0' | '1'))
    }

    fn consume_mask_row(&mut self, line_no: usize, raw: &str) -> Result<(), ParseError> {
        let (w, h) = {
            let part = self.level.parts.last().unwrap();
            (part.w, part.h)
        };
        if self.mask_z >= self.level.parts.last().unwrap().d {
            return Err(self.err(line_no, "[part] mask too many rows for d"));
        }
        let my = self.mask_y;
        let mz = self.mask_z;
        let part = self.level.parts.last_mut().unwrap();
        for (x, c) in raw.chars().enumerate() {
            let b = u8::from(c == '#' || c == '1');
            let idx = x + (w as usize) * (my as usize + (h as usize) * mz as usize);
            part.mask[idx] = b;
        }
        self.mask_y += 1;
        if self.mask_y >= h {
            self.mask_y = 0;
            self.mask_z += 1;
        }
        Ok(())
    }

    /// Is `raw` a `[cells]` ASCII glyph row? It must have a grid allocated
    /// (`size =` seen), match the grid width exactly, and consist only of known
    /// cell glyphs (so `material_ids =`/`shapes =` sub-headers and the `size =`
    /// line — all of which contain `=` — never match).
    fn is_cells_row(&self, raw: &str) -> bool {
        let Some(g) = &self.cells_grid else {
            return false;
        };
        let sx = g.size.0;
        if sx == 0 || raw.chars().count() != sx {
            return false;
        }
        raw.chars().all(|c| glyph_to_cell_kind(c).is_some())
    }

    /// Consume one `[cells]` ASCII row into the grid at the current `(y, z)`.
    fn consume_cells_row(&mut self, line_no: usize, raw: &str) -> Result<(), ParseError> {
        let (sx, sy, sz) = self.cells_grid.as_ref().unwrap().size;
        let (y, z) = (self.cells_y, self.cells_z);
        if z >= sz {
            return Err(self.err(line_no, "[cells] too many z-layers for size.z"));
        }
        if y >= sy {
            return Err(self.err(line_no, "[cells] too many rows for size.y"));
        }
        let g = self.cells_grid.as_mut().unwrap();
        for (x, c) in raw.chars().enumerate() {
            // is_cells_row guaranteed every glyph maps and the row width == sx.
            let kind = glyph_to_cell_kind(c).unwrap_or(CellKind::Empty);
            g.kinds[x + sx * (y + sy * z)] = kind;
        }
        self.cells_y += 1;
        if self.cells_y >= sy {
            self.cells_y = 0;
            self.cells_z += 1;
        }
        Ok(())
    }

    /// Consume one `[cells]` `shapes =` table line: `x y z shape dir` (M3c).
    /// `shape` ∈ `{full, wedgexy, rampz}` and `dir` is a READABLE direction
    /// name — `ne|nw|se|sw` for `wedgexy`, `posx|negx|posy|negy` for `rampz`
    /// (omitted for `full`). See [`tokens_to_shape`]. Only non-`Full` shapes are
    /// recorded.
    fn consume_cells_shape(&mut self, line_no: usize, line: &str) -> Result<(), ParseError> {
        let mut it = line.split([' ', '\t', ',']).filter(|t| !t.is_empty());
        let parse_i = |s: Option<&str>| s.and_then(|t| t.parse::<i32>().ok());
        let (x, y, z) = match (parse_i(it.next()), parse_i(it.next()), parse_i(it.next())) {
            (Some(x), Some(y), Some(z)) => (x, y, z),
            _ => return Err(self.err(line_no, "[cells] shapes: expected 'x y z shape [dir]'")),
        };
        let shape_kw = it
            .next()
            .ok_or_else(|| self.err(line_no, "[cells] shapes: missing shape keyword"))?
            .to_ascii_lowercase();
        let dir_kw = it.next().unwrap_or("").to_ascii_lowercase();
        let shape = tokens_to_shape(&shape_kw, &dir_kw).ok_or_else(|| {
            self.err(
                line_no,
                "[cells] shapes: expected 'wedgexy ne|nw|se|sw' or 'rampz posx|negx|posy|negy'",
            )
        })?;
        if let Some(g) = self.cells_grid.as_mut() {
            if shape != CellShape::Full {
                g.shapes.push((x, y, z, shape));
            }
        }
        Ok(())
    }

    /// Parse one `[hierarchy]` link line:
    /// `child_kind child_index -> parent_kind parent_index`.
    /// Tolerant of whitespace/commas around the tokens; the `->` arrow separates
    /// the child and parent halves. Cyclic / out-of-range links are NOT rejected
    /// here (the indices may forward-reference an object declared later); they are
    /// pruned in `finalize` via [`Level::prune_hierarchy`].
    fn consume_hierarchy_line(&mut self, line_no: usize, line: &str) -> Result<(), ParseError> {
        let (lhs, rhs) = line
            .split_once("->")
            .ok_or_else(|| self.err(line_no, "[hierarchy] expected 'child -> parent'"))?;
        let parse_ref = |half: &str| -> Option<HierRef> {
            let mut it = half.split([' ', '\t', ',']).filter(|t| !t.is_empty());
            let kind = HierKind::from_token(it.next()?)?;
            let index = it.next().and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
            Some(HierRef::new(kind, index))
        };
        let child = parse_ref(lhs)
            .ok_or_else(|| self.err(line_no, "[hierarchy] bad child 'kind index'"))?;
        let parent = parse_ref(rhs)
            .ok_or_else(|| self.err(line_no, "[hierarchy] bad parent 'kind index'"))?;
        self.level.hierarchy.push((child, parent));
        Ok(())
    }

    /// Open a new section from its `[...]` interior; sets up scratch entries.
    fn open_section(&mut self, line_no: usize, inside: &str) -> Result<Section, ParseError> {
        // reset per-section scratch
        self.mask_y = 0;
        self.mask_z = 0;
        self.note_body.clear();
        self.note_started = false;
        self.cells_y = 0;
        self.cells_z = 0;
        self.cells_sub = CellsSub::Rows;

        let (kw, rest) = match inside.split_once(char::is_whitespace) {
            Some((a, b)) => (a, b.trim()),
            None => (inside, ""),
        };
        Ok(match kw {
            "meta" => Section::Meta,
            "room" => Section::Room,
            "inputs" => Section::Inputs,
            "pedestal" => Section::Pedestal,
            "door" => Section::Door,
            "sequence" | "lock" => {
                // Opt-in L11 sequence-lock. Just opening the header marks it
                // declared (so finalize can resolve / validate it); the keys are
                // collected in handle_kv.
                self.seq_declared = true;
                Section::Sequence
            }
            "player" => Section::Player,
            "effects" => Section::Effects,
            "palette" => Section::Palette,
            "atmosphere" => Section::Atmosphere,
            "light" => {
                // Repeatable: each header pushes a default light the keys fill in.
                self.level.lights.push(PointLight::default());
                Section::Light
            }
            "win_when" | "map" => Section::Skip,
            "node" => {
                self.level.nodes.push(Node {
                    name: rest.to_string(),
                    color_rgb: 0xFF_FFFF,
                    ..Default::default()
                });
                Section::Node
            }
            "pipe" => {
                let (from, to) =
                    split_arrow(rest).ok_or_else(|| self.err(line_no, "[pipe] expected 'A->B'"))?;
                self.level.pipes.push(Pipe {
                    from,
                    to,
                    density: 1.0,
                    color_rgb: 0xFF_FFFF,
                    state: PipeState::Off,
                    ..Default::default()
                });
                Section::Pipe
            }
            "part" => {
                self.level.parts.push(Part {
                    name: rest.to_string(),
                    color_rgb: 0xFF_FFFF,
                    ..Default::default()
                });
                let n = self.level.parts.len();
                self.level.part_looks.resize(n, PartLook::default());
                Section::Part
            }
            "emit" => {
                // clause "NAME=val" — keep the source node name; parts come next.
                let name = rest.split('=').next().unwrap_or(rest).trim().to_string();
                self.pending_emits.push(PendingEmit {
                    source: name,
                    parts: Vec::new(),
                });
                Section::Emit
            }
            "wall_decal" => {
                self.level.decals.push(WallDecal {
                    name: rest.to_string(),
                    surface_kind: CellKind::WallTrace,
                    ..Default::default()
                });
                Section::WallDecal
            }
            "holo" => {
                self.level.holos.push(LevelHolo {
                    name: rest.to_string(),
                    shape: HoloShape::Box,
                    ..Default::default()
                });
                Section::Holo
            }
            "portal" => {
                self.level.portals.push(LevelPortal {
                    name: rest.to_string(),
                    ..Default::default()
                });
                Section::Portal
            }
            "vox" => {
                self.level.voxes.push(LevelVox {
                    name: rest.to_string(),
                    material: VoxMaterial::Solid,
                    scale: 1,
                    ..Default::default()
                });
                Section::Vox
            }
            "texture" => {
                self.level.textures.push(LevelTexture {
                    name: rest.to_string(),
                    ..Default::default()
                });
                Section::Texture
            }
            "sky" => {
                // `[sky]` has no NAME suffix; keys fill in `level.sky` below.
                Section::Sky
            }
            "zone_input" => {
                self.level.zone_inputs.push(ZoneInput {
                    name: rest.to_string(),
                    ..Default::default()
                });
                Section::ZoneInput
            }
            "trigger" => {
                // The NAME after `[trigger ...]` is optional + purely cosmetic
                // (no stored field). Default = inert empty-region fire-once Event.
                self.level.triggers.push(re_core::Trigger::default());
                self.trigger_node_names.push(None);
                Section::Trigger
            }
            "note" => {
                if let Some(note) = parse_note_header(rest) {
                    self.level.notes.push(note);
                    Section::Note
                } else {
                    // unknown KIND → skip its body, like legacy
                    Section::Skip
                }
            }
            "material" => {
                self.level.materials.push(LevelMaterial {
                    name: rest.to_string(),
                    ..Default::default()
                });
                Section::Material
            }
            "color_cell_ids" => Section::ColorCellIds,
            "hierarchy" => Section::Hierarchy,
            "prop" => {
                // `[prop NAME]`: the keys follow one per line (§3.2); a header that
                // carries more than the name is the doc's one-line shorthand, which
                // is not the file grammar — refuse it rather than guess a split.
                if rest.is_empty() || rest.contains(char::is_whitespace) {
                    return Err(self.err(
                        line_no,
                        format!(
                            "[prop {rest}] expected '[prop NAME]' with each key = value on its own line"
                        ),
                    ));
                }
                self.level.props.push(props::PropDef {
                    name: rest.to_string(),
                    line: line_no,
                    keys: Vec::new(),
                });
                Section::Prop
            }
            "field" => {
                // `[field NAME]`: same "verbatim keys, interpreted in finalize"
                // contract as `[prop]` (see `fields.rs`'s module doc for the scope
                // note on which §4.1 inline-attach host kinds are also implemented).
                if rest.is_empty() || rest.contains(char::is_whitespace) {
                    return Err(self.err(
                        line_no,
                        format!(
                            "[field {rest}] expected '[field NAME]' with each key = value on its own line"
                        ),
                    ));
                }
                self.level.field_defs.push(fields::FieldDef {
                    name: rest.to_string(),
                    line: line_no,
                    host: fields::FieldHost::Standalone,
                    keys: Vec::new(),
                });
                Section::Field
            }
            "cells" => {
                // The grid is allocated when `size =` is read; until then no row
                // matches `is_cells_row`. Start each `[cells]` fresh.
                self.cells_grid = None;
                self.cells_y = 0;
                self.cells_z = 0;
                self.cells_sub = CellsSub::Rows;
                Section::Cells
            }
            // Scalable Material Overlay sections (P0). Each pushes one inert
            // default primitive; keys fill it in. A level with none of these
            // sections leaves `level.overlay` at its empty default (back-compat).
            "zone" => {
                self.level.overlay.zones.push(Zone {
                    name: rest.to_string(),
                    ..Default::default()
                });
                Section::Zone
            }
            "line" => {
                self.level.overlay.lines.push(Line {
                    name: rest.to_string(),
                    ..Default::default()
                });
                Section::Line
            }
            "inset" => {
                self.level.overlay.insets.push(Inset {
                    name: rest.to_string(),
                    ..Default::default()
                });
                Section::Inset
            }
            "emitter" => {
                // The optional `NAME` (`rest`) is purely for readability — the
                // EmitterSpec carries no name. Start from the sane default (a
                // gentle upward haze) and let keys override fields one at a time.
                self.level.emitters.push(EmitterSpec::default());
                Section::Emitter
            }
            // An UNRECOGNIZED section header. Reported ONCE here — not once per key
            // inside it, which would bury the actual finding (a mistyped `[trigger]`
            // would emit one diagnostic per line of the block). Recorded with the
            // section keyword as the "key" and `?` as the section, so a reader sees
            // `[?] unknown key 'triger'` and looks at the header, not the body.
            _ => {
                self.level.unknown_keys.push(UnknownKey {
                    line: line_no,
                    section: Section::None.label(),
                    key: kw.to_string(),
                });
                Section::None
            }
        })
    }

    /// Commit accumulated block data when a section closes.
    fn flush_section(&mut self, sec: &Section) {
        if *sec == Section::Note {
            if let Some(note) = self.level.notes.last_mut() {
                // CANONICALIZE the captured body by trimming trailing whitespace.
                //
                // A `[note]` body is captured VERBATIM up to the next header, so it
                // absorbs whatever blank lines happen to separate the section from
                // the next one. That made the parsed body depend on the file's
                // cosmetic spacing: a note ending at EOF parsed to `"text"`, while
                // the SAME note followed by a blank line parsed to `"text\n"`.
                // `re_voximport`'s writer already trims on the way out and always
                // emits one blank separator, so a load→save→load shifted the first
                // shape into the second — a one-shot, harmless-but-real structural
                // diff that made "the level survived the save unchanged" untestable.
                //
                // Trimming HERE makes the parsed body the canonical form, so the
                // round-trip is an identity instead of a convergence. Render-neutral:
                // the host game's text-surface builder already `trim_end`s every
                // note body before publishing it as a text surface, so no frame moves.
                let body = std::mem::take(&mut self.note_body);
                note.text = body.trim_end().to_string();
            }
            self.note_started = false;
        }
        if *sec == Section::Cells {
            if let Some(grid) = self.cells_grid.take() {
                self.level.cells = Some(grid);
            }
        }
    }

    fn handle_kv(
        &mut self,
        sec: &Section,
        line_no: usize,
        k: &str,
        v: &str,
    ) -> Result<(), ParseError> {
        let src = self.src;
        // UNKNOWN-KEY DIAGNOSTIC (see `UnknownKey`). Every section handler below
        // used to end in a bare `_ => {}`, discarding an unrecognized key in total
        // silence — the enabler for the whole "authored, parsed, never on screen"
        // defect class. Each of those arms now records the key here instead.
        //
        // A plain local (not a `self` method) so the arms that hold a live `&mut`
        // borrow into `self.level` — `let n = self.level.nodes.last_mut()` and
        // friends — can still set it. Drained into `self.level.unknown_keys` once
        // the match, and every borrow it took, has ended.
        //
        // NOT an error: a stray key must never make one of 63 hand-authored levels
        // unloadable. `re_lvlcheck` surfaces these; a corpus test keeps the shipped
        // count at zero.
        let mut unknown: Option<&str> = None;
        match sec {
            Section::Meta => match k {
                "name" => {
                    self.level.name = v.to_string();
                    self.meta_seen = true;
                }
                "title" => self.level.title = v.to_string(),
                // AUTHORING METADATA — parsed but never acted on. Retained verbatim
                // so a load→save (`re_voximport::to_lvl_text`, and therefore every
                // `ts_editor` save) re-emits the line instead of deleting it. A key
                // the parser reads and the writer never emits is invisible to a
                // text-vs-text round-trip test, which is how these survived unnoticed.
                "version" => self.level.version = v.to_string(),
                "spec_rev" => self.level.spec_rev = v.to_string(),
                // `[meta] prop_style` (two-repo split A1, replaces the
                // `campaign.toml`-derived default): the material pack a `[prop]`
                // gets when it names no `style` of its own. A level that omits the
                // key keeps the struct default (`PropStyle::Lab`), so every
                // pre-existing level not yet carrying the key is unaffected. An
                // unrecognized value is a load error (same discipline as
                // `[material] surface`), not a silently-ignored key, because a
                // typo here would silently repaint every undressed prop.
                "prop_style" => {
                    self.level.prop_default_style = props::PropStyle::parse(v)
                        .ok_or_else(|| self.err(line_no, "[meta] prop_style: unknown style"))?;
                    self.level.prop_style_authored = true;
                }
                _ => unknown = Some(k),
            },
            Section::Room => match k {
                "size" => {
                    let (sx, sy, sz) = parse_coord3(v)
                        .ok_or_else(|| self.err(line_no, "[room] size: bad coord triple"))?;
                    if sx <= 0 || sy <= 0 || sz <= 0 {
                        return Err(self.err(line_no, "[room] size: non-positive dim"));
                    }
                    self.level.size = (sx as usize, sy as usize, sz as usize);
                    self.room_set = true;
                }
                "ambient" => self.level.ambient = v.to_string(),
                // INERT authoring metadata (see `Level::room_floor`): authored on
                // 17 shipped levels, consumed by nothing. Retained verbatim so the
                // writer can re-emit it rather than silently deleting it on save.
                "floor" => self.level.room_floor = v.to_string(),
                "ceiling" => self.level.room_ceiling = v.to_string(),
                // R2 opt-in: stamp pipe paths as visible tagged WallTrace traces.
                // Accepts true/1/on/yes (case-insensitive); anything else ⇒ false.
                "render_pipes" => {
                    let t = v.trim().to_ascii_lowercase();
                    self.level.render_pipes = matches!(t.as_str(), "true" | "1" | "on" | "yes");
                }
                // L12 `while(true)` finale flag: let a wired graph cycle STAND as
                // a terminal state instead of rolling back. Accepts
                // true/1/on/yes (case-insensitive); anything else ⇒ false (the
                // historical cycle-rollback default).
                "cycle_keep" => {
                    let t = v.trim().to_ascii_lowercase();
                    self.level.cycle_keep = matches!(t.as_str(), "true" | "1" | "on" | "yes");
                }
                // POWER BUDGET (v0.2 re-scope): the cap on simultaneously-ON pipes.
                // Absent OR `0` ⇒ UNLIMITED (the historical behaviour, byte-identical),
                // so only levels that opt in change. A negative or non-numeric value is
                // a hard parse error rather than a silent fall-back to unlimited: a
                // typo'd budget would otherwise quietly delete the level's only real
                // constraint, which is precisely the "authored → parsed → never took
                // effect" class this project keeps re-hitting.
                "power_budget" => {
                    let n: i64 = v.trim().parse().map_err(|_| {
                        self.err(line_no, "[room] power_budget: expected an integer")
                    })?;
                    if n < 0 {
                        return Err(self.err(line_no, "[room] power_budget: must be >= 0"));
                    }
                    self.level.power_budget = n as u32;
                }
                // Multi-floor vertical model (design §4): cells per storey. Optional
                // metadata — does NOT change grid storage (the grid size comes from
                // `[room] size`). Absent ⇒ default 4 (applied in `finalize`); `H < 1`
                // is rejected so a staircase can never be authored against a degenerate
                // height. A non-numeric value is a hard parse error (not silently
                // ignored) because, unlike the boolean flags above, a malformed height
                // would otherwise leave the level on the default with no signal.
                "storey_height" => {
                    let h: i64 = v.trim().parse().map_err(|_| {
                        self.err(line_no, "[room] storey_height: expected an integer")
                    })?;
                    if h < 1 {
                        return Err(self.err(line_no, "[room] storey_height: must be >= 1"));
                    }
                    self.level.storey_height = h as u32;
                }
                _ => unknown = Some(k),
            },
            Section::Node => {
                // An inline `[field]` attach: handled
                // FIRST and returns early, so `attach_inline`'s `&mut self.level`
                // never overlaps the field-specific `n = self.level.nodes.last_mut()`
                // borrow every other arm below uses.
                if k == "field" {
                    let idx = self.level.nodes.len() - 1;
                    fields::attach_inline(
                        &mut self.level,
                        fields::FieldHost::Node(idx),
                        v,
                        line_no,
                    )
                    .map_err(|msg| err_at(src, line_no, format!("[node] field: {msg}")))?;
                    return Ok(());
                }
                let (sx, sy, sz) = self.level.size;
                let room_set = self.room_set;
                let n = self.level.nodes.last_mut().unwrap();
                match k {
                    "pos" => {
                        let (x, y, z) = parse_coord3(v)
                            .ok_or_else(|| err_at(src, line_no, "[node] pos: bad coord triple"))?;
                        if !room_set {
                            return Err(err_at(src, line_no, "[node] pos before [room] size"));
                        }
                        if x < 0
                            || x >= sx as i32
                            || y < 0
                            || y >= sy as i32
                            || z < 0
                            || z >= sz as i32
                        {
                            return Err(err_at(src, line_no, "[node] pos out of room bounds"));
                        }
                        n.x = x;
                        n.y = y;
                        n.z = z;
                    }
                    "glyph" => n.glyph = v.chars().next().unwrap_or(' '),
                    "pool" => n.pool = parse_words(v),
                    "pool_alt" => n.pool_alt = parse_words(v),
                    // Optional anchor word index into `pool` (default 0). A
                    // non-numeric / negative value falls back to 0 — additive
                    // and never an error, so old levels and typos stay valid.
                    "anchor" => n.anchor = v.trim().parse::<usize>().unwrap_or(0),
                    "color" => {
                        n.color_rgb = parse_color(v)
                            .ok_or_else(|| err_at(src, line_no, "[node] color: bad hex"))?
                    }
                    "is_input" => n.is_input = is_truthy(v),
                    "is_output" => n.is_output = is_truthy(v),
                    "threshold" => n.threshold = v.trim().parse().unwrap_or(0),
                    _ => unknown = Some(k),
                }
            }
            Section::Pipe => {
                // §4.1 inline attach — see the matching comment in `Section::Node`.
                if k == "field" {
                    let idx = self.level.pipes.len() - 1;
                    fields::attach_inline(
                        &mut self.level,
                        fields::FieldHost::Pipe(idx),
                        v,
                        line_no,
                    )
                    .map_err(|msg| err_at(src, line_no, format!("[pipe] field: {msg}")))?;
                    return Ok(());
                }
                let (sx, sy, sz) = self.level.size;
                let q = self.level.pipes.last_mut().unwrap();
                match k {
                    "state" => {
                        q.state = if matches!(v, "ON" | "on" | "1") {
                            PipeState::On
                        } else {
                            PipeState::Off
                        }
                    }
                    "path" => {
                        let waypoints = parse_waypoints(v)
                            .ok_or_else(|| err_at(src, line_no, "[pipe] path: bad waypoint"))?;
                        for &(wx, wy, wz) in &waypoints {
                            if wx < 0
                                || wx >= sx as i32
                                || wy < 0
                                || wy >= sy as i32
                                || wz < 0
                                || wz >= sz as i32
                            {
                                return Err(err_at(
                                    src,
                                    line_no,
                                    "[pipe] waypoint out of room bounds",
                                ));
                            }
                        }
                        q.path = expand_path(&waypoints);
                    }
                    // ADDITIVE alternative to `path`: a smooth Catmull-Rom curve
                    // through ≥2 control points, sampled and voxelized into a
                    // GUARANTEED 6-connected contiguous cell path (no diagonal
                    // gaps). The result drops into `path` exactly like
                    // `expand_path`'s output, so switch_at resolution, the puzzle,
                    // and rendering all work unchanged. A pipe uses `path` OR
                    // `spline`, never both.
                    "spline" => {
                        let ctrl = parse_waypoints(v).ok_or_else(|| {
                            err_at(src, line_no, "[pipe] spline: bad control point")
                        })?;
                        if ctrl.len() < 2 {
                            return Err(err_at(
                                src,
                                line_no,
                                "[pipe] spline: need ≥2 control points",
                            ));
                        }
                        for &(cx, cy, cz) in &ctrl {
                            if cx < 0
                                || cx >= sx as i32
                                || cy < 0
                                || cy >= sy as i32
                                || cz < 0
                                || cz >= sz as i32
                            {
                                return Err(err_at(
                                    src,
                                    line_no,
                                    "[pipe] spline control point out of room bounds",
                                ));
                            }
                        }
                        let path = spline_path(&ctrl);
                        // The voxelized curve must also stay inside the room: a
                        // Catmull-Rom curve can bow OUTSIDE the convex hull of its
                        // control points, so an in-bounds control set can still
                        // generate an out-of-bounds sample. Reject clearly.
                        for &(px, py, pz) in &path {
                            if px < 0
                                || px >= sx as i32
                                || py < 0
                                || py >= sy as i32
                                || pz < 0
                                || pz >= sz as i32
                            {
                                return Err(err_at(
                                    src,
                                    line_no,
                                    "[pipe] spline curve leaves room bounds (bow the control points inward)",
                                ));
                            }
                        }
                        q.path = path;
                    }
                    "switch_at" => {
                        let (sx2, sy2, sz2) = parse_coord3(v).ok_or_else(|| {
                            err_at(src, line_no, "[pipe] switch_at: bad coord triple")
                        })?;
                        // Store raw; resolved against the expanded path in a
                        // SECOND pass in finalize (order-independent, and a hard
                        // error if the cell is off-path — matches legacy).
                        q.switch_at_raw = Some((sx2, sy2, sz2));
                    }
                    "density" => q.density = v.trim().parse().unwrap_or(1.0),
                    "color" => {
                        q.color_rgb = parse_color(v)
                            .ok_or_else(|| err_at(src, line_no, "[pipe] color: bad hex"))?
                    }
                    _ => unknown = Some(k),
                }
            }
            Section::Part => {
                let t = self.level.parts.last_mut().unwrap();
                match k {
                    "size" => {
                        let (w, h, d) = parse_coord3(v)
                            .ok_or_else(|| err_at(src, line_no, "[part] size: bad triple"))?;
                        if w <= 0 || h <= 0 || d <= 0 {
                            return Err(err_at(src, line_no, "[part] size: non-positive dim"));
                        }
                        t.w = w;
                        t.h = h;
                        t.d = d;
                        t.mask = vec![0u8; (w * h * d) as usize];
                    }
                    "offset" => {
                        let (ox, oy, oz) = parse_coord3(v)
                            .ok_or_else(|| err_at(src, line_no, "[part] offset: bad triple"))?;
                        t.off_x = ox;
                        t.off_y = oy;
                        t.off_z = oz;
                    }
                    "color" => {
                        t.color_rgb = parse_color(v)
                            .ok_or_else(|| err_at(src, line_no, "[part] color: bad hex"))?
                    }
                    "mask" => { /* opens the indented block; rows handled above */ }
                    "height" | "top" | "top_from" => {
                        let n = self.level.parts.len();
                        self.level.part_looks.resize(n, PartLook::default());
                        let look = self.level.part_looks.last_mut().unwrap();
                        match k {
                            "height" => {
                                look.height = match v.trim() {
                                    "full" => 1.0,
                                    "half" => 0.5,
                                    x => x
                                        .parse::<f32>()
                                        .ok()
                                        .filter(|h| (PART_MIN_HEIGHT..=1.0).contains(h))
                                        .ok_or_else(|| {
                                            err_at(
                                                src,
                                                line_no,
                                                "[part] height: full | half | a fraction 0.1..1",
                                            )
                                        })?,
                                };
                            }
                            "top" => {
                                let rgb = parse_color(v)
                                    .ok_or_else(|| err_at(src, line_no, "[part] top: bad hex"))?;
                                let from = look.top.map_or(PART_TOP_FROM_DEFAULT, |(_, f)| f);
                                look.top = Some((rgb, from));
                            }
                            _ => {
                                let f = v
                                    .trim()
                                    .parse::<f32>()
                                    .ok()
                                    .filter(|f| (0.0..=1.0).contains(f))
                                    .ok_or_else(|| {
                                        err_at(src, line_no, "[part] top_from: a fraction 0..1")
                                    })?;
                                // Order-independent: `top_from` before `top` is kept.
                                look.top = Some(look.top.map_or((u32::MAX, f), |(c, _)| (c, f)));
                            }
                        }
                    }
                    _ => unknown = Some(k),
                }
            }
            Section::Emit => {
                if k == "parts" {
                    let names = parse_words(v);
                    if let Some(last) = self.pending_emits.last_mut() {
                        last.parts = names;
                    }
                } else {
                    unknown = Some(k);
                }
            }
            // `[inputs]` keys are DATA — a key IS a node name (`R = 1`), so there
            // is no such thing as an unknown one here. An unresolvable name is
            // caught in `finalize` as a real error.
            Section::Inputs => {
                // store raw; resolved/validated in finalize
                self.level
                    .inputs
                    .push((k.to_string(), v.trim().parse().unwrap_or(0)));
            }
            Section::Pedestal => match k {
                "center" => {
                    let (cx, cy, cz) = parse_coord3(v)
                        .ok_or_else(|| self.err(line_no, "[pedestal] center: bad triple"))?;
                    self.level.pedestal.cx = cx;
                    self.level.pedestal.cy = cy;
                    self.level.pedestal.cz = cz;
                }
                "volume" => {
                    let (w, h, d) = parse_coord3(v)
                        .ok_or_else(|| self.err(line_no, "[pedestal] volume: bad triple"))?;
                    self.level.pedestal.w = w;
                    self.level.pedestal.h = h;
                    self.level.pedestal.d = d;
                    self.pedestal_set = true;
                }
                _ => unknown = Some(k),
            },
            Section::Door => match k {
                "pos" => {
                    let (x, y, z) = parse_coord3(v)
                        .ok_or_else(|| self.err(line_no, "[door] pos: bad triple"))?;
                    self.level.door.x = x;
                    self.level.door.y = y;
                    self.level.door.z = z;
                    self.door_set = true;
                }
                "target_parts" => {
                    self.door_target_names = parse_words(v);
                }
                // Opt-in INVERTED targets (L12 `exit 0`): nodes that must read 0
                // (NOT fire) for the win. Resolved + validated in finalize.
                "target_off" => {
                    self.door_target_off_names = parse_words(v);
                }
                // INERT authoring metadata (see `Door::target_model`) — retained
                // verbatim so the writer re-emits it instead of dropping it.
                "target_model" => self.level.door.target_model = v.to_string(),
                // §4.1 inline attach — see the matching comment in `Section::Node`.
                // The door has no per-instance index (`Level` owns exactly one).
                "field" => {
                    fields::attach_inline(&mut self.level, fields::FieldHost::Door, v, line_no)
                        .map_err(|msg| self.err(line_no, format!("[door] field: {msg}")))?;
                }
                _ => unknown = Some(k),
            },
            Section::Sequence => match k {
                // The fixed observation window: the output nodes the lock reads
                // each step. Re-declaring replaces (last wins) — but a level
                // should declare it once.
                "observe" => {
                    self.seq_observe_names = parse_words(v);
                }
                // One ordered step: the observed nodes that must be ON at this
                // step (every other observed node must be OFF). An empty value
                // is a valid "all observed nodes OFF" step.
                "step" => {
                    self.seq_step_names.push(parse_words(v));
                }
                _ => unknown = Some(k),
            },
            Section::Player => match k {
                "spawn" => {
                    let (x, y, z) = parse_coord3(v)
                        .ok_or_else(|| self.err(line_no, "[player] spawn: bad triple"))?;
                    self.level.player.x = x;
                    self.level.player.y = y;
                    self.level.player.z = z;
                    self.player_set = true;
                }
                "facing" => self.level.player.facing_yaw_col = facing_to_yaw_col(v),
                _ => unknown = Some(k),
            },
            Section::WallDecal => {
                let d = self.level.decals.last_mut().unwrap();
                match k {
                    // The decal's surface kind. Load-bearing, not cosmetic:
                    // `build_map` paints the whole decal region with this
                    // `CellKind`, so `hologram` yields transparent hologram
                    // cells and `portal` yields portal faces.
                    //
                    // All SIX spellings `re_voximport::to_lvl_text` can emit
                    // are accepted, so a level that came in through the `.vox`
                    // import path (tag 13→Hologram, 14→Portal, 15→Object)
                    // survives save→reload. Accepting only the first three used
                    // to hard-error on exactly the levels the importer produced.
                    // An unknown spelling still fails loudly — this widens the
                    // accepted set, it does not make the parser lenient.
                    "surface" => {
                        d.surface_kind = match v {
                            "matrix" => CellKind::WallMatrix,
                            "trace" => CellKind::WallTrace,
                            "solid" => CellKind::WallSolid,
                            "hologram" => CellKind::Hologram,
                            "portal" => CellKind::Portal,
                            "object" => CellKind::Object,
                            _ => {
                                return Err(err_at(
                                    src,
                                    line_no,
                                    "[wall_decal] surface: unknown kind",
                                ))
                            }
                        }
                    }
                    "region" => {
                        let (a, b) = parse_region(v).ok_or_else(|| {
                            err_at(src, line_no, "[wall_decal] region: bad format")
                        })?;
                        d.x0 = a.0;
                        d.y0 = a.1;
                        d.z0 = a.2;
                        d.x1 = b.0;
                        d.y1 = b.1;
                        d.z1 = b.2;
                    }
                    "density" => d.density = v.trim().parse().unwrap_or(0.0),
                    "color_head" => {
                        d.color_head = parse_color(v).ok_or_else(|| {
                            err_at(src, line_no, "[wall_decal] color_head bad hex")
                        })?
                    }
                    "color_tail" => {
                        d.color_tail = parse_color(v).ok_or_else(|| {
                            err_at(src, line_no, "[wall_decal] color_tail bad hex")
                        })?
                    }
                    _ => unknown = Some(k),
                }
            }
            Section::Holo => {
                let h = self.level.holos.last_mut().unwrap();
                match k {
                    "shape" => {
                        h.shape = HoloShape::from_str_opt(v)
                            .ok_or_else(|| err_at(src, line_no, "[holo] shape: unknown kind"))?;
                    }
                    "region" => {
                        let (a, b) = parse_region(v)
                            .ok_or_else(|| err_at(src, line_no, "[holo] region: bad format"))?;
                        h.x0 = a.0;
                        h.y0 = a.1;
                        h.z0 = a.2;
                        h.x1 = b.0;
                        h.y1 = b.1;
                        h.z1 = b.2;
                    }
                    "style" => h.style = v.trim().parse::<u16>().unwrap_or(0),
                    _ => unknown = Some(k),
                }
            }
            Section::Portal => {
                let pt = self.level.portals.last_mut().unwrap();
                match k {
                    "region_a" => {
                        let (a, b) = parse_region(v)
                            .ok_or_else(|| err_at(src, line_no, "[portal] region_a: bad format"))?;
                        pt.ax0 = a.0;
                        pt.ay0 = a.1;
                        pt.az0 = a.2;
                        pt.ax1 = b.0;
                        pt.ay1 = b.1;
                        pt.az1 = b.2;
                    }
                    "region_b" => {
                        let (a, b) = parse_region(v)
                            .ok_or_else(|| err_at(src, line_no, "[portal] region_b: bad format"))?;
                        pt.bx0 = a.0;
                        pt.by0 = a.1;
                        pt.bz0 = a.2;
                        pt.bx1 = b.0;
                        pt.by1 = b.1;
                        pt.bz1 = b.2;
                    }
                    "rotate" => {
                        pt.rotate = parse_f32_req(v)
                            .ok_or_else(|| err_at(src, line_no, "[portal] rotate: not a number"))?;
                    }
                    "skin" => pt.skin = v.trim().to_string(),
                    _ => unknown = Some(k),
                }
            }
            Section::Vox => {
                let vx = self.level.voxes.last_mut().unwrap();
                match k {
                    "file" => vx.file = v.trim().to_string(),
                    "offset" => {
                        let (ox, oy, oz) = parse_coord3(v).ok_or_else(|| {
                            err_at(src, line_no, "[vox] offset: bad coord triple")
                        })?;
                        vx.off_x = ox;
                        vx.off_y = oy;
                        vx.off_z = oz;
                    }
                    "material" => {
                        vx.material = VoxMaterial::from_str_opt(v)
                            .ok_or_else(|| err_at(src, line_no, "[vox] material: unknown kind"))?;
                    }
                    // Integer downscale ≥ 1 (a 0/negative/garbage value clamps to 1).
                    "scale" => vx.scale = v.trim().parse::<i32>().unwrap_or(1).max(1),
                    _ => unknown = Some(k),
                }
            }
            Section::Texture => {
                let tx = self.level.textures.last_mut().unwrap();
                match k {
                    "file" => tx.file = v.trim().to_string(),
                    "max_size" => {
                        let n: u32 = v.trim().parse().map_err(|_| {
                            err_at(src, line_no, "[texture] max_size: not a whole number")
                        })?;
                        if n == 0 {
                            return Err(err_at(
                                src,
                                line_no,
                                "[texture] max_size: must be >= 1 (0 would erase the image)",
                            ));
                        }
                        tx.max_size = Some(n);
                    }
                    // `height = <path>` — the 8-bit relief channel that makes this
                    // image a lit surface once a `[material] relief =` asks for it.
                    // Resolved, capped and resampled exactly like `file` at decode.
                    "height" => tx.height = Some(v.trim().to_string()),
                    // `frames = <path.gif>` — decode EVERY frame of the named GIF at
                    // load into an animated texture set. A single-frame file yields a
                    // one-frame set (a still), which is not an error.
                    "frames" => tx.frames = Some(v.trim().to_string()),
                    // `fps = <f32 > 0>` — playback rate of the `frames` set. A bad
                    // value stops the load with a line number rather than silently
                    // becoming the default.
                    "fps" => {
                        let f = parse_f32_req(v)
                            .ok_or_else(|| err_at(src, line_no, "[texture] fps: not a number"))?;
                        if !f.is_finite() || f <= 0.0 {
                            return Err(err_at(
                                src,
                                line_no,
                                "[texture] fps: must be finite and > 0                                  (a still frame set would never advance)",
                            ));
                        }
                        tx.fps = Some(f);
                    }
                    // Unknown `[texture]` keys used to be swallowed here in silence,
                    // so a typo'd key looked accepted and then did nothing. Route
                    // them through the same diagnostic every other section uses.
                    _ => unknown = Some(k),
                }
            }
            Section::Sky => {
                let sky = &mut self.level.sky;
                match k {
                    "cylinder" => sky.cylinder = Some(v.trim().to_string()),
                    "cap_top" => sky.cap_top = Some(v.trim().to_string()),
                    "cap_bottom" => sky.cap_bottom = Some(v.trim().to_string()),
                    // PROCEDURAL SKY PROGRAM keys (DSL owned by `re_world::sky_spec`).
                    // `kind = ` is a
                    // one-line preset (expands to a whole `SkySpec` via `from_kind`,
                    // so it must be the ONLY procedural key on the section — mixing
                    // it with `base`/`layer` is rejected rather than silently
                    // resolving an ordering). `base = ` sets the gradient once;
                    // `layer = ` is repeatable, one call per authored layer, appended
                    // in file order (later layers composite over earlier ones, same
                    // as `SkyProgram::sample`'s fold order).
                    "kind" => {
                        if sky.base.is_some() || !sky.layers.is_empty() {
                            return Err(err_at(
                                src,
                                line_no,
                                "[sky] kind: cannot be combined with base/layer on the same section",
                            ));
                        }
                        let spec = re_world::sky_spec::SkySpec::from_kind(v.trim())
                            .map_err(|e| err_at(src, line_no, format!("[sky] kind: {e}")))?;
                        sky.base = Some(spec.base);
                        sky.layers = spec.layers;
                        sky.seed = Some(spec.seed);
                        sky.hz = Some(spec.hz);
                    }
                    "base" => {
                        let base = re_world::sky_spec::SkySpec::parse_base(v)
                            .map_err(|e| err_at(src, line_no, format!("[sky] base: {e}")))?;
                        sky.base = Some(base);
                    }
                    "layer" => {
                        let layer = re_world::sky_spec::SkySpec::parse_layer(v)
                            .map_err(|e| err_at(src, line_no, format!("[sky] layer: {e}")))?;
                        sky.layers.push(layer);
                    }
                    "seed" => {
                        let n = parse_u32_req(v).ok_or_else(|| {
                            err_at(src, line_no, "[sky] seed: expected an integer")
                        })?;
                        sky.seed = Some(n);
                    }
                    "hz" => {
                        let n = parse_u32_req(v)
                            .ok_or_else(|| err_at(src, line_no, "[sky] hz: expected an integer"))?;
                        if n == 0 {
                            return Err(err_at(src, line_no, "[sky] hz: must be > 0"));
                        }
                        sky.hz = Some(n);
                    }
                    _ => unknown = Some(k),
                }
            }
            Section::ZoneInput => {
                let (sx, sy, sz) = self.level.size;
                let room_set = self.room_set;
                let z = self.level.zone_inputs.last_mut().unwrap();
                match k {
                    "node" => z.node = v.trim().to_string(),
                    "region" => {
                        let (a, b) = parse_region(v).ok_or_else(|| {
                            err_at(src, line_no, "[zone_input] region: bad format")
                        })?;
                        if !room_set {
                            return Err(err_at(
                                src,
                                line_no,
                                "[zone_input] region before [room] size",
                            ));
                        }
                        // Normalise + bounds-check both corners.
                        let (x0, x1) = (a.0.min(b.0), a.0.max(b.0));
                        let (y0, y1) = (a.1.min(b.1), a.1.max(b.1));
                        let (z0, z1) = (a.2.min(b.2), a.2.max(b.2));
                        if x0 < 0
                            || x1 >= sx as i32
                            || y0 < 0
                            || y1 >= sy as i32
                            || z0 < 0
                            || z1 >= sz as i32
                        {
                            return Err(err_at(
                                src,
                                line_no,
                                "[zone_input] region out of room bounds",
                            ));
                        }
                        z.x0 = x0;
                        z.y0 = y0;
                        z.z0 = z0;
                        z.x1 = x1;
                        z.y1 = y1;
                        z.z1 = z1;
                    }
                    _ => unknown = Some(k),
                }
            }
            // ── Trigger-fields: region + action + once ──────────────────────
            Section::Trigger => {
                let (sx, sy, sz) = self.level.size;
                let room_set = self.room_set;
                let idx = self.level.triggers.len() - 1;
                let t = self.level.triggers.last_mut().unwrap();
                match k {
                    "region" => {
                        let (a, b) = parse_region(v)
                            .ok_or_else(|| err_at(src, line_no, "[trigger] region: bad format"))?;
                        if !room_set {
                            return Err(err_at(
                                src,
                                line_no,
                                "[trigger] region before [room] size",
                            ));
                        }
                        let (x0, x1) = (a.0.min(b.0), a.0.max(b.0));
                        let (y0, y1) = (a.1.min(b.1), a.1.max(b.1));
                        let (z0, z1) = (a.2.min(b.2), a.2.max(b.2));
                        if x0 < 0
                            || x1 >= sx as i32
                            || y0 < 0
                            || y1 >= sy as i32
                            || z0 < 0
                            || z1 >= sz as i32
                        {
                            return Err(err_at(
                                src,
                                line_no,
                                "[trigger] region out of room bounds",
                            ));
                        }
                        t.region = Region::Box {
                            min: (x0, y0, z0),
                            max: (x1, y1, z1),
                        };
                    }
                    // `action = activate NODE` ⇒ ActivateNode (node resolved in
                    // finalize). `action = event LABEL` ⇒ a generic named event.
                    "action" => {
                        let v = v.trim();
                        let (kind, arg) = match v.split_once(char::is_whitespace) {
                            Some((kind, arg)) => (kind, arg.trim()),
                            None => (v, ""),
                        };
                        match kind {
                            "activate" => {
                                if arg.is_empty() {
                                    return Err(err_at(
                                        src,
                                        line_no,
                                        "[trigger] action activate: missing node name",
                                    ));
                                }
                                t.action = re_core::TriggerAction::activate_node(arg);
                                self.trigger_node_names[idx] = Some(arg.to_string());
                            }
                            "event" => {
                                t.action = re_core::TriggerAction::event(arg);
                                self.trigger_node_names[idx] = None;
                            }
                            other => {
                                return Err(err_at(
                                    src,
                                    line_no,
                                    format!("[trigger] unknown action '{other}'"),
                                ));
                            }
                        }
                    }
                    "once" => {
                        t.once = matches!(v.trim(), "true" | "1" | "yes");
                    }
                    _ => unknown = Some(k),
                }
            }
            // ── Scalable Material Overlay (P0 parse — carrier only, no resolve) ──
            Section::Zone => {
                let z = self.level.overlay.zones.last_mut().unwrap();
                match k {
                    "region" => {
                        let (a, b) = parse_region(v)
                            .ok_or_else(|| err_at(src, line_no, "[zone] region: bad format"))?;
                        z.region = Region::Box { min: a, max: b };
                    }
                    "material" => z.material = MaterialId(v.trim().to_string()),
                    "scale" => {
                        z.scale = parse_scale_rule(v)
                            .ok_or_else(|| err_at(src, line_no, "[zone] scale: unknown rule"))?;
                    }
                    // Comma/space-separated inset names this zone references.
                    "insets" => {
                        z.insets = v
                            .split([',', ' ', '\t'])
                            .map(str::trim)
                            .filter(|s| !s.is_empty())
                            .map(|s| InsetId(s.to_string()))
                            .collect();
                    }
                    _ => unknown = Some(k),
                }
            }
            Section::Line => {
                let ln = self.level.overlay.lines.last_mut().unwrap();
                match k {
                    // `path = x0,y0,z0; x1,y1,z1; …` — a world-space polyline. Each
                    // `;`-separated group is one `Vec3` vertex (DP2-B authoring).
                    "path" => {
                        let pts = parse_polyline(v)
                            .ok_or_else(|| err_at(src, line_no, "[line] path: bad polyline"))?;
                        ln.path = LinePath { points: pts };
                    }
                    "material" => ln.material = LineMaterialId(v.trim().to_string()),
                    "scale" => {
                        ln.scale = parse_scale_rule(v)
                            .ok_or_else(|| err_at(src, line_no, "[line] scale: unknown rule"))?;
                    }
                    _ => unknown = Some(k),
                }
            }
            Section::Inset => {
                let ins = self.level.overlay.insets.last_mut().unwrap();
                match k {
                    // `at_octave = N` (single octave) or `N-M` (inclusive range).
                    "at_octave" => {
                        ins.at_octave = parse_octave_range(v)
                            .ok_or_else(|| err_at(src, line_no, "[inset] at_octave: bad range"))?;
                    }
                    "region" => {
                        let (a, b) = parse_region(v)
                            .ok_or_else(|| err_at(src, line_no, "[inset] region: bad format"))?;
                        ins.region = Region::Box { min: a, max: b };
                    }
                    // DP6-B: a referenced child scene (id/path). No loading in P0.
                    "child" | "content" | "scene" => {
                        ins.content = InsetContent::ChildScene(v.trim().to_string());
                    }
                    _ => unknown = Some(k),
                }
            }
            Section::Effects => match k {
                "snow" => self.level.effects.snow = parse_intensity(v),
                "steam" => self.level.effects.steam = is_truthy(v),
                "matrix" => self.level.effects.matrix = parse_intensity(v),
                "glitch" => self.level.effects.glitch = parse_intensity(v),
                _ => unknown = Some(k),
            },
            // `[light]` — one punctual light. Every key is validated with a line
            // number; a malformed light is a parse error, never a silent default
            // (an invisible or NaN light is far worse than a refused level).
            Section::Light => {
                let l = self.level.lights.last_mut().unwrap();
                match k {
                    "pos" | "at" => {
                        let (x, y, z) = parse_vec3(v).ok_or_else(|| {
                            err_at(src, line_no, "[light] pos: expected 'x, y, z'")
                        })?;
                        if !(x.is_finite() && y.is_finite() && z.is_finite()) {
                            return Err(err_at(src, line_no, "[light] pos: non-finite"));
                        }
                        l.pos = Vec3::new(x, y, z);
                    }
                    "color" => {
                        l.color = parse_color(v).ok_or_else(|| {
                            err_at(src, line_no, "[light] color: bad hex (#RRGGBB)")
                        })?;
                    }
                    "radius" => {
                        let r: f32 = v
                            .trim()
                            .parse()
                            .map_err(|_| err_at(src, line_no, "[light] radius: not a number"))?;
                        if !(r.is_finite() && r > 0.0) {
                            return Err(err_at(src, line_no, "[light] radius: must be > 0"));
                        }
                        l.radius = r;
                    }
                    "intensity" => {
                        let i: f32 = v
                            .trim()
                            .parse()
                            .map_err(|_| err_at(src, line_no, "[light] intensity: not a number"))?;
                        if !(i.is_finite() && i >= 0.0) {
                            return Err(err_at(src, line_no, "[light] intensity: must be >= 0"));
                        }
                        l.intensity = i;
                    }
                    _ => unknown = Some(k),
                }
            }
            // `[atmosphere]` — key light + ambient + height fog + postfx preset.
            // The three fog-DISTANCE keys are accepted here as aliases of the
            // `[palette]` ones and land in the same `LevelPalette` fields, so a
            // level may author fog in either section (never in both meanings).
            Section::Atmosphere => {
                match k {
                    "fog_color" => {
                        self.level.palette.fog_color = parse_color(v).ok_or_else(|| {
                            err_at(src, line_no, "[atmosphere] fog_color: bad hex (#RRGGBB)")
                        })?;
                        self.level.palette.has_fog = true;
                    }
                    "fog_near" => {
                        self.level.palette.fog_near = v.trim().parse().map_err(|_| {
                            err_at(src, line_no, "[atmosphere] fog_near: not a number")
                        })?;
                        self.level.palette.has_fog_dist = true;
                    }
                    "fog_far" => {
                        self.level.palette.fog_far = v.trim().parse().map_err(|_| {
                            err_at(src, line_no, "[atmosphere] fog_far: not a number")
                        })?;
                        self.level.palette.has_fog_dist = true;
                    }
                    "key_dir" => {
                        let (x, y, z) = parse_vec3(v).ok_or_else(|| {
                            err_at(src, line_no, "[atmosphere] key_dir: expected 'x, y, z'")
                        })?;
                        let d = Vec3::new(x, y, z);
                        if !d.is_finite() || d.length_squared() <= 0.0 {
                            return Err(err_at(
                                src,
                                line_no,
                                "[atmosphere] key_dir: zero/non-finite",
                            ));
                        }
                        self.level.atmos.key_dir = Some(d);
                    }
                    "key_color" => {
                        self.level.atmos.key_color = Some(parse_color(v).ok_or_else(|| {
                            err_at(src, line_no, "[atmosphere] key_color: bad hex (#RRGGBB)")
                        })?);
                    }
                    "ambient" => {
                        let a: f32 = v.trim().parse().map_err(|_| {
                            err_at(src, line_no, "[atmosphere] ambient: not a number")
                        })?;
                        if !(a.is_finite() && (0.0..=1.0).contains(&a)) {
                            return Err(err_at(
                                src,
                                line_no,
                                "[atmosphere] ambient: must be in [0,1]",
                            ));
                        }
                        self.level.atmos.ambient = Some(a);
                    }
                    "fog_height" => {
                        let (base, falloff) = parse_pair(v).ok_or_else(|| {
                            err_at(
                                src,
                                line_no,
                                "[atmosphere] fog_height: expected 'base falloff'",
                            )
                        })?;
                        if !(base.is_finite() && falloff.is_finite() && falloff >= 0.0) {
                            return Err(err_at(
                                src,
                                line_no,
                                "[atmosphere] fog_height: falloff must be >= 0 and finite",
                            ));
                        }
                        self.level.atmos.fog_height = Some(HeightFog { base, falloff });
                    }
                    "postfx" => {
                        let name = v.trim().to_ascii_lowercase();
                        if !re_world::is_postfx_preset(&name) {
                            return Err(err_at(
                                src,
                                line_no,
                                format!(
                                    "[atmosphere] postfx: unknown preset '{}' (known: {})",
                                    name,
                                    re_world::POSTFX_PRESETS.join(", ")
                                ),
                            ));
                        }
                        self.level.atmos.postfx = Some(name);
                    }
                    _ => unknown = Some(k),
                }
            }
            Section::Palette => match k {
                "fog_color" => {
                    if let Some(c) = parse_color(v) {
                        self.level.palette.fog_color = c;
                        self.level.palette.has_fog = true;
                    }
                }
                "fog_near" => {
                    self.level.palette.fog_near = v.trim().parse().unwrap_or(0.0);
                    self.level.palette.has_fog_dist = true;
                }
                "fog_far" => {
                    self.level.palette.fog_far = v.trim().parse().unwrap_or(0.0);
                    self.level.palette.has_fog_dist = true;
                }
                _ => unknown = Some(k),
            },
            Section::Material => {
                // Capture the per-material `surface = …` binding separately (it
                // lives on the level, not the material), then mutate the material.
                if k == "surface" {
                    let name = self.level.materials.last().map(|m| m.name.clone());
                    self.bind_surface_material(line_no, v, name)?;
                    return Ok(());
                }
                let m = self.level.materials.last_mut().unwrap();
                match k {
                    "fg" => {
                        m.fg = parse_color(v)
                            .ok_or_else(|| err_at(src, line_no, "[material] fg: bad hex"))?
                    }
                    "bg" => {
                        m.bg = parse_color(v)
                            .ok_or_else(|| err_at(src, line_no, "[material] bg: bad hex"))?
                    }
                    "glyph" => m.glyph = v.chars().next().map(u32::from),
                    // Select a renderer CATALOG material by name (e.g. `metal`,
                    // `rust`, `monitor`). Forwarded to the renderer in the emitted
                    // MaterialDef name; explicit keys on this material still override.
                    "preset" => m.preset = Some(v.trim().to_string()),
                    "rank" => m.rank = v.trim().parse::<u8>().unwrap_or(0),
                    // MIRROR authoring: `reflective = true/yes/1` marks the bound
                    // surface a mirror; `clarity = 0..1` sets how crisply it resolves.
                    "reflective" | "mirror" => m.reflective = is_truthy(v.trim()),
                    "clarity" => {
                        m.clarity = v
                            .trim()
                            .parse::<f32>()
                            .map(|c| c.clamp(0.0, 1.0))
                            .map_err(|_| err_at(src, line_no, "[material] clarity: not a number"))?
                    }
                    "recipe" => {
                        // `recipe = image` is RESOLVED LATE (in `build_map`): the
                        // `tex_id` is not known until the `[texture]` files are decoded
                        // and pushed onto `Map::textures`. We store a placeholder
                        // `Image { tex_id: 0 }` here and patch the real id in
                        // `renderer_material_defs` once the name→id map exists. The
                        // bound `texture = NAME` rides on `m.texture`.
                        m.recipe = Some(if v.trim() == "image" {
                            // Placeholder; the real `tex_id` AND the authored `span` are
                            // folded in late (`renderer_material_defs`) once the texture
                            // name→id map + the `span` key are both parsed.
                            MaterialRecipe::Image {
                                tex_id: 0,
                                span: None,
                                subcell: re_world::SubcellMode::Off,
                                fit: re_world::Fit::Contain,
                                tile: re_world::IMAGE_TILE_NONE,
                                relief: None,
                            }
                        } else {
                            parse_recipe(v)
                                .ok_or_else(|| err_at(src, line_no, "[material] recipe: unknown"))?
                        });
                    }
                    // `texture = NAME` binds this material (with `recipe = image`) to a
                    // decoded `[texture NAME]` image. Resolved to a `tex_id` in build_map.
                    "texture" => m.texture = Some(v.trim().to_string()),
                    // `span = ALONG_MIN,ALONG_SIZE UP_TOP,UP_SIZE` (M3a): the WORLD-ANCHORED
                    // UV span so ONE figure spans the whole textured surface. Two
                    // comma-pairs separated by whitespace, all in world units.
                    "span" => {
                        m.image_span = Some(
                            parse_image_span(v)
                                .ok_or_else(|| err_at(src, line_no, "[material] span: bad format (want ALONG_MIN,ALONG_SIZE UP_TOP,UP_SIZE)"))?,
                        )
                    }
                    // `subcell = off | quad | braille | ascii | contour | lines` (M3b):
                    // the SUB-CELL detail mode for a `recipe = image` material.
                    // Default `off` ⇒ legacy solid block.
                    "subcell" => {
                        m.image_subcell = parse_subcell(v).ok_or_else(|| {
                            err_at(src, line_no, "[material] subcell: want off | quad | braille | ascii | contour | lines")
                        })?
                    }
                    // `fit = stretch | contain | cover` — the fit mode for a
                    // `recipe = image` material (Component 1, quality-textures spec).
                    // Default `contain` ⇒ aspect-preserving, whole image inside the wall.
                    // An explicit `span` key overrides `fit`.
                    "fit" => {
                        m.image_fit = parse_fit(v).ok_or_else(|| {
                            err_at(src, line_no, "[material] fit: want stretch | contain | cover")
                        })?
                    }
                    // `tile = ALONG UP` — the span-relative repeat count for a
                    // `recipe = image` material. Validated here (finite, > 0) so a
                    // typo is a LINE-NUMBERED parse error, never a silent default:
                    // `tile = 0 4` renders nothing and would otherwise look like a
                    // renderer bug.
                    "tile" => {
                        m.image_tile = Some(parse_image_tile(v).ok_or_else(|| {
                            err_at(
                                src,
                                line_no,
                                "[material] tile: want 'ALONG UP', both finite and > 0",
                            )
                        })?)
                    }
                    // `relief = height_scale=… normal_strength=…` — flips a
                    // `recipe = image` material off self-lit and routes its height
                    // field through the bound texture's `height =` channel. Same
                    // `k=v` token style as `recipe = KIND k=v …`; an unknown knob is
                    // an error rather than a silently ignored token, so a
                    // misremembered name cannot look accepted and do nothing.
                    "relief" => {
                        m.image_relief = Some(parse_image_relief(v).map_err(|e| {
                            err_at(src, line_no, format!("[material] relief: {e}"))
                        })?)
                    }
                    "animation" | "anim" => {
                        m.anim = parse_anim(v)
                            .ok_or_else(|| err_at(src, line_no, "[material] animation: bad spec"))?
                    }
                    // `detail = on | off` — opt-in
                    // world-anchored detail-tier cross-fade for this material's
                    // procedural recipe. Default `off` ⇒ historical single-tier shading.
                    "detail" => m.detail = is_truthy(v.trim()),
                    _ => unknown = Some(k),
                }
            }
            Section::Emitter => {
                let e = self.level.emitters.last_mut().unwrap();
                match k {
                    // ── KIND ────────────────────────────────────────────────────
                    "kind" => {
                        e.kind = parse_emitter_kind(v).ok_or_else(|| {
                            err_at(src, line_no, "[emitter] kind: Standard|GodRays")
                        })?;
                    }
                    // ── ORIGIN (mutually exclusive; last one wins) ──────────────
                    "origin_world" => {
                        let (x, y, z) = parse_vec3(v).ok_or_else(|| {
                            err_at(src, line_no, "[emitter] origin_world: bad 'X Y Z'")
                        })?;
                        e.origin = EmitterOrigin::World(Vec3::new(x, y, z));
                    }
                    "origin_screen" => {
                        let (col, row) = parse_f32_pair(v).ok_or_else(|| {
                            err_at(src, line_no, "[emitter] origin_screen: bad 'COL ROW'")
                        })?;
                        e.origin = EmitterOrigin::Screen { col, row };
                    }
                    // ── COLOUR ──────────────────────────────────────────────────
                    "colors" => {
                        e.colors = parse_color_triple(v).ok_or_else(|| {
                            err_at(
                                src,
                                line_no,
                                "[emitter] colors: expected three 0xRRGGBB stops",
                            )
                        })?;
                    }
                    "color_mode" => {
                        e.color_mode = parse_color_mode(v).ok_or_else(|| {
                            err_at(
                                src,
                                line_no,
                                "[emitter] color_mode: RampOverAge|RandomPick|Fixed",
                            )
                        })?;
                    }
                    // ── GLYPHS ──────────────────────────────────────────────────
                    "family" => e.family = v.trim().to_string(),
                    "glyph_set" => {
                        let glyphs = parse_glyph_set(v).ok_or_else(|| {
                            err_at(src, line_no, "[emitter] glyph_set: need ≥1 glyph")
                        })?;
                        e.glyph_set = Some(glyphs);
                    }
                    // ── BOLD ────────────────────────────────────────────────────
                    "bold" => {
                        e.bold = parse_bold_mode(v).ok_or_else(|| {
                            err_at(src, line_no, "[emitter] bold: Off|On|ByProximity|ByAge")
                        })?;
                    }
                    // ── DIRECTION & SPREAD ──────────────────────────────────────
                    "direction" => {
                        let (x, y, z) = parse_vec3(v).ok_or_else(|| {
                            err_at(src, line_no, "[emitter] direction: bad 'X Y Z'")
                        })?;
                        e.direction = Vec3::new(x, y, z);
                    }
                    "spread_deg" => {
                        e.spread_deg = parse_f32_req(v).ok_or_else(|| {
                            err_at(src, line_no, "[emitter] spread_deg: not a number")
                        })?;
                    }
                    // ── SIZE & GROWTH ───────────────────────────────────────────
                    "size" => {
                        e.size = parse_f32_req(v)
                            .ok_or_else(|| err_at(src, line_no, "[emitter] size: not a number"))?;
                    }
                    "growth" => {
                        e.growth = parse_f32_req(v).ok_or_else(|| {
                            err_at(src, line_no, "[emitter] growth: not a number")
                        })?;
                    }
                    // ── FREQUENCY ───────────────────────────────────────────────
                    "rate" => {
                        e.rate = parse_f32_req(v)
                            .ok_or_else(|| err_at(src, line_no, "[emitter] rate: not a number"))?;
                    }
                    "burst" => {
                        e.burst = Some(parse_u32_req(v).ok_or_else(|| {
                            err_at(src, line_no, "[emitter] burst: not a non-negative integer")
                        })?);
                    }
                    // ── SPEED ───────────────────────────────────────────────────
                    "speed" => {
                        e.speed = parse_f32_req(v)
                            .ok_or_else(|| err_at(src, line_no, "[emitter] speed: not a number"))?;
                    }
                    "speed_spread" => {
                        e.speed_spread = parse_f32_req(v).ok_or_else(|| {
                            err_at(src, line_no, "[emitter] speed_spread: not a number")
                        })?;
                    }
                    // ── PATH RANDOMISATION ──────────────────────────────────────
                    "path_jitter" => {
                        e.path_jitter = parse_f32_req(v).ok_or_else(|| {
                            err_at(src, line_no, "[emitter] path_jitter: not a number")
                        })?;
                    }
                    "seed" => {
                        e.seed = parse_u64_req(v).ok_or_else(|| {
                            err_at(src, line_no, "[emitter] seed: not a non-negative integer")
                        })?;
                    }
                    "swirl" => {
                        e.swirl = parse_f32_req(v)
                            .ok_or_else(|| err_at(src, line_no, "[emitter] swirl: not a number"))?;
                    }
                    // ── LIFETIME ────────────────────────────────────────────────
                    "lifetime" => {
                        e.lifetime = parse_f32_req(v).ok_or_else(|| {
                            err_at(src, line_no, "[emitter] lifetime: not a number")
                        })?;
                    }
                    "lifetime_spread" => {
                        e.lifetime_spread = parse_f32_req(v).ok_or_else(|| {
                            err_at(src, line_no, "[emitter] lifetime_spread: not a number")
                        })?;
                    }
                    // ── FORCES ──────────────────────────────────────────────────
                    "accel" => {
                        let (x, y, z) = parse_vec3(v)
                            .ok_or_else(|| err_at(src, line_no, "[emitter] accel: bad 'X Y Z'"))?;
                        e.accel = Vec3::new(x, y, z);
                    }
                    // ── CAPACITY ────────────────────────────────────────────────
                    "max_particles" => {
                        e.max_particles = parse_u32_req(v).ok_or_else(|| {
                            err_at(
                                src,
                                line_no,
                                "[emitter] max_particles: not a non-negative integer",
                            )
                        })?;
                    }
                    // ── BLEND ───────────────────────────────────────────────────
                    "blend" => {
                        e.blend = parse_emitter_blend(v).ok_or_else(|| {
                            err_at(src, line_no, "[emitter] blend: KeepBg|Opaque")
                        })?;
                    }
                    // ── VARIETY & RANDOMISATION ──────────────────────────────
                    // Every key below is opt-in; a level that never sets one
                    // leaves the corresponding `EmitterSpec` field at its
                    // neutral default (today's behaviour, byte-identical).
                    "motion" => {
                        e.motion = parse_motion(v).ok_or_else(|| {
                            err_at(
                                src,
                                line_no,
                                "[emitter] motion: ballistic|curl|attract|lorenz|flock|ripple (bad knobs)",
                            )
                        })?;
                    }
                    "lifetime_dist" => {
                        e.lifetime_dist = Some(parse_distribution(v).ok_or_else(|| {
                            err_at(
                                src,
                                line_no,
                                "[emitter] lifetime_dist: 'a..b' | 'm~s' | 'a..b^k'",
                            )
                        })?);
                    }
                    "speed_dist" => {
                        e.speed_dist = Some(parse_distribution(v).ok_or_else(|| {
                            err_at(
                                src,
                                line_no,
                                "[emitter] speed_dist: 'a..b' | 'm~s' | 'a..b^k'",
                            )
                        })?);
                    }
                    "size_dist" => {
                        e.size_dist = Some(parse_distribution(v).ok_or_else(|| {
                            err_at(
                                src,
                                line_no,
                                "[emitter] size_dist: 'a..b' | 'm~s' | 'a..b^k'",
                            )
                        })?);
                    }
                    "randomize" => {
                        e.randomize = parse_f32_req(v).ok_or_else(|| {
                            err_at(src, line_no, "[emitter] randomize: not a number")
                        })?;
                    }
                    "color_jitter" => {
                        e.color_jitter = Some(parse_color_jitter(v).ok_or_else(|| {
                            err_at(
                                src,
                                line_no,
                                "[emitter] color_jitter: 'h=<deg> s=<0..1> l=<0..1> dance=<0..1>'",
                            )
                        })?);
                    }
                    "color_ramp" => {
                        e.color_ramp = Some(parse_color_ramp(v).ok_or_else(|| {
                            err_at(src, line_no, "[emitter] color_ramp: 2-8 0xRRGGBB stops")
                        })?);
                    }
                    "glyph_pick" => {
                        e.glyph_pick = parse_glyph_pick(v).ok_or_else(|| {
                            err_at(src, line_no, "[emitter] glyph_pick: Sequential|Random")
                        })?;
                    }
                    "glyph_source" => {
                        e.glyph_source = parse_glyph_source(v).ok_or_else(|| {
                            err_at(
                                src,
                                line_no,
                                "[emitter] glyph_source: family | text \"<string>\"",
                            )
                        })?;
                    }
                    "trail" => {
                        e.trail = Some(parse_trail(v).ok_or_else(|| {
                            err_at(
                                src,
                                line_no,
                                "[emitter] trail: '<count> fade=<0..1> dir_glyphs=<on|off>'",
                            )
                        })?);
                    }
                    "bounce" => {
                        e.bounce = Some(parse_f32_req(v).ok_or_else(|| {
                            err_at(src, line_no, "[emitter] bounce: not a number")
                        })?);
                    }
                    "visibility" => {
                        e.visibility = parse_visibility(v).ok_or_else(|| {
                            err_at(
                                src,
                                line_no,
                                "[emitter] visibility: always | lit gain=<f32>",
                            )
                        })?;
                    }
                    _ => return Err(err_at(src, line_no, format!("[emitter] unknown key '{k}'"))),
                }
            }
            Section::ColorCellIds => {
                // `x,y,z = id` — the key is the cell coord triple, the value the
                // 1-based colour-material id. `build_map` stamps `id` into that
                // cell's `side_data`; the renderer dispatches the colour palette.
                let (x, y, z) = parse_coord3(k).ok_or_else(|| {
                    err_at(src, line_no, "[color_cell_ids] expected 'x,y,z = id'")
                })?;
                let id: u16 = v
                    .trim()
                    .parse()
                    .map_err(|_| err_at(src, line_no, "[color_cell_ids] id must be a u16"))?;
                self.level.color_cell_ids.push((x, y, z, id));
            }
            Section::Cells => self.handle_cells_kv(line_no, k, v)?,
            // `[prop]` keys are kept verbatim (the writer re-emits them) and
            // interpreted in `finalize`, where every error names its line and key.
            Section::Prop => {
                if props::is_known_key(k) {
                    if let Some(p) = self.level.props.last_mut() {
                        p.keys.push(props::PropKey {
                            key: k.to_string(),
                            value: v.to_string(),
                            line: line_no,
                        });
                    }
                } else {
                    unknown = Some(k);
                }
            }
            // `[field]` keys are kept verbatim (the writer re-emits them) and
            // interpreted in `finalize`, where every error names its line and key.
            // Unlike `[prop]`, an unknown key here is a HARD parse error (§7(b)'s
            // acceptance test), not a soft `unknown_keys` diagnostic — a field
            // controls audio mixing, not set dressing, and a silently-dropped knob
            // would be a gameplay/accessibility regression nobody sees.
            Section::Field => {
                if fields::is_known_key(k) {
                    if let Some(f) = self.level.field_defs.last_mut() {
                        f.keys.push(fields::FieldKey {
                            key: k.to_string(),
                            value: v.to_string(),
                            line: line_no,
                        });
                    }
                } else {
                    return Err(err_at(src, line_no, format!("[field] unknown key '{k}'")));
                }
            }
            // The remaining sections have NO notion of an unknown key and are
            // deliberately silent: `Skip` is `[map]`/`[win_when]` (read-and-skip
            // blocks whose contents the parser never interprets); `None` is an
            // unrecognized SECTION, reported once at its header rather than once
            // per key inside it; `Note` and `Hierarchy` never reach here (their
            // lines are consumed earlier in `run`).
            _ => {}
        }
        if let Some(key) = unknown {
            self.level.unknown_keys.push(UnknownKey {
                line: line_no,
                section: sec.label(),
                key: key.to_string(),
            });
        }
        Ok(())
    }

    /// Handle a `key = value` line inside `[cells]`:
    /// - `size = sx,sy,sz` allocates the grid (required, first).
    /// - `material_ids =` / `materials =` switches to the id sub-table.
    /// - `shapes =` switches to the 45° shapes sub-table (M3c).
    /// - `x,y,z = id` (while in the id sub-table) records a sparse material id.
    fn handle_cells_kv(&mut self, line_no: usize, k: &str, v: &str) -> Result<(), ParseError> {
        match k {
            "size" => {
                let (sx, sy, sz) = parse_coord3(v)
                    .ok_or_else(|| self.err(line_no, "[cells] size: expected 'sx,sy,sz'"))?;
                if sx < 0 || sy < 0 || sz < 0 {
                    return Err(self.err(line_no, "[cells] size: dims must be non-negative"));
                }
                self.cells_grid = Some(CellGrid::new((sx as usize, sy as usize, sz as usize)));
                self.cells_y = 0;
                self.cells_z = 0;
                self.cells_sub = CellsSub::Rows;
            }
            "material_ids" | "materials" => {
                if self.cells_grid.is_none() {
                    return Err(self.err(line_no, "[cells] material_ids before size"));
                }
                self.cells_sub = CellsSub::MaterialIds;
            }
            "shapes" => {
                if self.cells_grid.is_none() {
                    return Err(self.err(line_no, "[cells] shapes before size"));
                }
                self.cells_sub = CellsSub::Shapes;
            }
            _ => {
                // Inside the material_ids sub-table: `x,y,z = id`.
                if self.cells_sub == CellsSub::MaterialIds {
                    let (x, y, z) = parse_coord3(k).ok_or_else(|| {
                        self.err(line_no, "[cells] material_ids: expected 'x,y,z = id'")
                    })?;
                    let id: u16 = v
                        .trim()
                        .parse()
                        .map_err(|_| self.err(line_no, "[cells] material id must be a u16"))?;
                    if let Some(g) = self.cells_grid.as_mut() {
                        g.material_ids.push((x, y, z, id));
                    }
                } else {
                    return Err(self.err(line_no, "[cells] unexpected key (need size first)"));
                }
            }
        }
        Ok(())
    }

    /// Bind a `[material]`'s `surface = KIND` to the level's per-surface table.
    fn bind_surface_material(
        &mut self,
        line_no: usize,
        v: &str,
        name: Option<String>,
    ) -> Result<(), ParseError> {
        let slot = match v.trim() {
            "wall_solid" | "solid" => &mut self.level.surface_materials.wall_solid,
            "wall_trace" | "trace" => &mut self.level.surface_materials.wall_trace,
            "wall_matrix" | "matrix" => &mut self.level.surface_materials.wall_matrix,
            "floor" => &mut self.level.surface_materials.floor,
            "ceiling" => &mut self.level.surface_materials.ceiling,
            _ => return Err(self.err(line_no, "[material] surface: unknown kind")),
        };
        *slot = name;
        Ok(())
    }

    /// Resolve cross-references and validate structure (port of `finalize_level`).
    fn finalize(mut self) -> Result<Level, ParseError> {
        if !self.room_set {
            return Err(self.err(0, "missing [room] size"));
        }
        if !self.meta_seen || self.level.name.is_empty() {
            return Err(self.err(0, "missing [meta] section or 'name' key"));
        }
        if !self.pedestal_set {
            return Err(self.err(0, "missing [pedestal] section"));
        }
        if !self.door_set {
            return Err(self.err(0, "missing [door] pos"));
        }
        if !self.player_set {
            return Err(self.err(0, "missing [player] spawn"));
        }

        // Normalise an absent `[room] storey_height` (the `0` sentinel) to the
        // default so a LOADED level always reports a concrete, ≥1 height. The
        // parse path already rejected `H < 1`, so the only `0` here is "key
        // absent". Metadata only — never touches the grid or `build_map`.
        if self.level.storey_height == 0 {
            self.level.storey_height = DEFAULT_STOREY_HEIGHT;
        }

        // A [vox] stamp must name a file (the model to stamp). An empty `file`
        // is an authoring error — fail clearly rather than silently stamping
        // nothing at build_map time.
        for v in &self.level.voxes {
            if v.file.trim().is_empty() {
                return Err(self.err(0, "[vox] missing 'file' key"));
            }
        }

        // Resolve pipe endpoints to node indices.
        for i in 0..self.level.pipes.len() {
            let (from, to) = {
                let p = &self.level.pipes[i];
                (p.from.clone(), p.to.clone())
            };
            let from_idx = self
                .level
                .node_index(&from)
                .ok_or_else(|| self.err(0, "[pipe] references unknown node"))?;
            let to_idx = self
                .level
                .node_index(&to)
                .ok_or_else(|| self.err(0, "[pipe] references unknown node"))?;
            let from_color = self.level.nodes[from_idx].color_rgb;
            let p = &mut self.level.pipes[i];
            p.from_node = from_idx;
            p.to_node = to_idx;
            if p.color_rgb == 0xFF_FFFF {
                p.color_rgb = from_color;
            }
            // Second pass: resolve switch_at against the now-final expanded path.
            // Off-path is a hard error (legacy level.c ~1707-1714).
            if let Some(cell) = p.switch_at_raw {
                match p.path.iter().position(|&c| c == cell) {
                    Some(idx) => p.switch_cell = Some(idx),
                    None => {
                        return Err(self.err(0, "[pipe] switch_at cell not on expanded path"));
                    }
                }
            }
        }

        // Resolve emit clauses (one clause per (source, part)).
        let pending = std::mem::take(&mut self.pending_emits);
        for emit in pending {
            let src_idx = self
                .level
                .node_index(&emit.source)
                .ok_or_else(|| self.err(0, "[emit] references unknown node"))?;
            for pn in emit.parts {
                let part_idx = self
                    .level
                    .parts
                    .iter()
                    .position(|p| p.name == pn)
                    .ok_or_else(|| self.err(0, "[emit] unknown part name"))?;
                self.level.emits.push(EmitClause {
                    source_node: src_idx,
                    part: part_idx,
                });
            }
        }

        // Validate [inputs] reference real nodes.
        for (name, _) in &self.level.inputs {
            if self.level.node_index(name).is_none() {
                return Err(self.err(0, "[inputs] references unknown node"));
            }
        }

        // Resolve [zone_input] node names → indices; the node MUST exist and
        // MUST be an input node (a zone can only drive an input — driving a
        // hidden/output node would fight the forward pass). An empty `node`
        // key is an authoring error.
        for i in 0..self.level.zone_inputs.len() {
            let (zname, node) = {
                let z = &self.level.zone_inputs[i];
                (z.name.clone(), z.node.clone())
            };
            if node.trim().is_empty() {
                return Err(self.err(0, "[zone_input] missing 'node' key"));
            }
            let idx = self.level.node_index(&node).ok_or_else(|| {
                self.err(
                    0,
                    format!("[zone_input {zname}] references unknown node '{node}'"),
                )
            })?;
            if !self.level.nodes[idx].is_input {
                return Err(self.err(
                    0,
                    format!("[zone_input {zname}] node '{node}' is not an input node"),
                ));
            }
            self.level.zone_inputs[i].node_index = idx;
        }

        // Resolve [trigger] ActivateNode node names → indices. As with
        // `[zone_input]`, the target MUST exist and MUST be an input node (a
        // trigger drives a node like an input → forward pass). `Event` triggers
        // carry no node (their `trigger_node_names` entry is `None`) and are left
        // untouched.
        for i in 0..self.level.triggers.len() {
            let Some(node) = self.trigger_node_names[i].clone() else {
                continue;
            };
            let idx = self.level.node_index(&node).ok_or_else(|| {
                self.err(
                    0,
                    format!("[trigger] action activate references unknown node '{node}'"),
                )
            })?;
            if !self.level.nodes[idx].is_input {
                return Err(self.err(
                    0,
                    format!("[trigger] action activate node '{node}' is not an input node"),
                ));
            }
            if let re_core::TriggerAction::ActivateNode { node_index, .. } =
                &mut self.level.triggers[i].action
            {
                *node_index = idx;
            }
        }

        // Resolve door target_parts (explicit names → indices; else default to
        // every output node — matches legacy's auto-from-is_output).
        if self.door_target_names.is_empty() {
            self.level.door.target_parts = self
                .level
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, n)| n.is_output)
                .map(|(i, _)| i)
                .collect();
        } else {
            let names = std::mem::take(&mut self.door_target_names);
            let mut idxs = Vec::with_capacity(names.len());
            for name in names {
                let idx = self
                    .level
                    .node_index(&name)
                    .ok_or_else(|| self.err(0, "[door] target_parts: unknown node"))?;
                idxs.push(idx);
            }
            self.level.door.target_parts = idxs;
        }

        // Resolve the opt-in [door] target_off INVERTED targets (L12 `exit 0`):
        // each must name a real node (the node that must read 0 to win). Empty ⇒
        // no inverted targets (byte-identical default). Validated like
        // target_parts so a typo'd inverted target fails the load rather than
        // silently never being satisfiable.
        if !self.door_target_off_names.is_empty() {
            let names = std::mem::take(&mut self.door_target_off_names);
            let mut idxs = Vec::with_capacity(names.len());
            for name in names {
                let idx = self
                    .level
                    .node_index(&name)
                    .ok_or_else(|| self.err(0, "[door] target_off: unknown node"))?;
                idxs.push(idx);
            }
            self.level.door.target_off = idxs;
        }

        // Resolve the opt-in [sequence] lock (L11). A declared `[sequence]`
        // section must name a non-empty `observe` window and at least one
        // `step`; each `step` lists the observed nodes ON at that step (the rest
        // OFF). Names resolve to node indices; an unknown name, or a step that
        // names a node outside the observe window, is an authoring error. A
        // level with NO `[sequence]` leaves `level.sequence` default-empty
        // (inactive) — byte-identical to before this feature existed.
        if self.seq_declared {
            if self.seq_observe_names.is_empty() {
                return Err(self.err(0, "[sequence] missing 'observe' (the output node window)"));
            }
            if self.seq_step_names.is_empty() {
                return Err(self.err(0, "[sequence] needs at least one 'step'"));
            }
            // Resolve the observe window to indices (preserve declared order).
            let mut observe: Vec<usize> = Vec::with_capacity(self.seq_observe_names.len());
            for name in &self.seq_observe_names {
                let idx = self.level.node_index(name).ok_or_else(|| {
                    self.err(0, format!("[sequence] observe: unknown node '{name}'"))
                })?;
                if observe.contains(&idx) {
                    return Err(self.err(0, format!("[sequence] observe: duplicate node '{name}'")));
                }
                observe.push(idx);
            }
            // Build each step's full constraint vector over the observe window.
            let steps = std::mem::take(&mut self.seq_step_names);
            let mut patterns: Vec<SequencePattern> = Vec::with_capacity(steps.len());
            for (si, on_names) in steps.iter().enumerate() {
                // Resolve the ON set for this step; every name must be in observe.
                let mut on_idx: Vec<usize> = Vec::with_capacity(on_names.len());
                for name in on_names {
                    let idx = self.level.node_index(name).ok_or_else(|| {
                        self.err(
                            0,
                            format!("[sequence] step {}: unknown node '{name}'", si + 1),
                        )
                    })?;
                    if !observe.contains(&idx) {
                        return Err(self.err(
                            0,
                            format!(
                                "[sequence] step {}: node '{name}' is not in the observe window",
                                si + 1
                            ),
                        ));
                    }
                    on_idx.push(idx);
                }
                // Full pattern: every observed node, ON iff it is in the step set.
                let bits: Vec<(usize, bool)> = observe
                    .iter()
                    .map(|&idx| (idx, on_idx.contains(&idx)))
                    .collect();
                patterns.push(SequencePattern { bits });
            }
            // Distinctness guard: ordered targets must be DISTINCT output states
            // (a repeated pattern would auto-advance two steps at once, defeating
            // the "hold the path through N turns" design). Adjacent or any
            // duplicate is rejected.
            for i in 0..patterns.len() {
                for j in (i + 1)..patterns.len() {
                    if patterns[i] == patterns[j] {
                        return Err(self.err(
                            0,
                            format!(
                                "[sequence] steps {} and {} are identical output states \
                                 (the targets must be distinct)",
                                i + 1,
                                j + 1
                            ),
                        ));
                    }
                }
            }
            // Preserve the AUTHORED form for lossless re-emission. The resolved
            // `patterns` discard the node names and the observe-window order, so
            // without this the `[sequence]` block could not be rewritten (the
            // round-trip gap this fix closes). The observe names are kept in
            // declared order; each step keeps exactly the names it listed (an
            // empty step Vec ⇒ the "all OFF" step). These drive NOTHING in the win
            // check — `patterns` above is unchanged.
            let observe_names = self.seq_observe_names.clone();
            self.level.sequence = SequenceSpec {
                patterns,
                observe_names,
                step_names: steps,
            };
        }

        // Clamp/fix any out-of-range authored emitter values so a hand-written
        // `.lvl` can never hand the renderer a degenerate spec (negative rate,
        // spread > 180, zero size, …). Mirrors the contract: parse keys then
        // `validate()`. This is in-place and never errors — clamping is silent.
        for e in &mut self.level.emitters {
            e.validate();
        }

        // `[prop]` EXPANSION (set-dressing §3.4). Run once here to VALIDATE every
        // prop against the finished level (bad values, keep-out violations and
        // generator refusals are load errors naming the line, key and puzzle
        // element) and to fold the parts the game reads straight off `Level` into
        // it: synthesized emitters, re-homed notes and the power bindings. The
        // cells / lines / holos / lights live only in the built map
        // (`build_map` re-expands, purely), so nothing is ever baked back into a
        // section. `prop_default_style` is set (or left at its `Lab` default) by
        // the `[meta] prop_style` key handler in `handle_kv`, above — no
        // finalize-time lookup needed now that levels are self-describing.
        if !self.level.props.is_empty() {
            let base = self.level.build_map_base();
            let (exp, errs) = props::expand_level(&self.level, &base);
            if let Some(e) = errs.into_iter().next() {
                return Err(self.err(e.line, e.msg));
            }
            for (line, key) in &exp.unused_keys {
                self.level.unknown_keys.push(UnknownKey {
                    line: *line,
                    section: Section::Prop.label(),
                    key: key.clone(),
                });
            }
            for m in &exp.note_moves {
                if let Some(n) = self.level.notes.get_mut(m.note) {
                    self.level.prop_note_origins.push(props::NoteOrigin {
                        note: m.note,
                        x: n.x,
                        y: n.y,
                        z: n.z,
                        facing: n.facing,
                    });
                    (n.x, n.y, n.z) = m.pos;
                    n.facing = m.facing;
                }
            }
            let emitter_base = self.level.emitters.len();
            self.level.emitters.extend(exp.emitters.iter().cloned());
            self.level.prop_power = exp.power_gates(
                &self.level,
                self.level.overlay.lines.len(),
                emitter_base,
                self.level.lights.len(),
            );
        }

        // `[effects] snow` → SYNTHESIZED emitters. The knob was parsed, round-tripped
        // and exposed in the editor but read by nothing, so a "COLD" level rendered
        // exactly like a warm one. The renderer already has one honest surface for
        // ambient particles — the emitter pool — so the knob is resolved into it here,
        // at load, where the room's dimensions are known. `snow_emitters` tags each
        // spec `synthesized = true` so a writer never serializes it back and an editor
        // never lists it as an editable object (see `EmitterSpec::synthesized`).
        //
        // Builds the level's geometry here (`build_map`, pure/idempotent — the real
        // caller builds it again from the returned `Level`) so `snow_emitters` can
        // place its lattice against the ACTUAL built map (shell + `[cells]` + vox +
        // zones) instead of a blind bounding box: a room with interior solids (e.g.
        // `L8-cold-room.lvl`'s full-height shelving) previously buried some lattice
        // points inside them, where the particle pool's screen-depth occlusion
        // (`re_render::emitter::splat_particles_kinds`) discards them every frame.
        if self.level.effects.snow > 0.0 {
            let map_for_snow = self.level.build_map();
            let snow = snow_emitters(&map_for_snow, self.level.effects.snow);
            self.level.emitters.extend(snow);
        }

        // Validate [portal] pairs: each side must be a flat wall face on a grid
        // boundary, and the two openings must have the same in-plane area (a
        // mismatch yields a visible tear, so reject it now with a clear error).
        let (sx, sy, sz) = self.level.size;
        for pt in &self.level.portals {
            validate_portal_face(pt.region_a(), (sx, sy, sz), &pt.name, "region_a")
                .map_err(|m| self.err(0, m))?;
            validate_portal_face(pt.region_b(), (sx, sy, sz), &pt.name, "region_b")
                .map_err(|m| self.err(0, m))?;
            let area_a = face_area(pt.region_a());
            let area_b = face_area(pt.region_b());
            if area_a != area_b {
                return Err(self.err(
                    0,
                    format!(
                        "[portal {}] region_a area {area_a:?} != region_b area {area_b:?} \
                         (the two openings must match)",
                        pt.name
                    ),
                ));
            }
        }

        // OBJECT-NESTING: drop any `[hierarchy]` link that is out of range, a
        // self-parent, a duplicate child, or forms a cycle — so a stale or
        // hand-broken hierarchy never points at a gone object or loops. A clean
        // hierarchy is unaffected; an empty one stays empty (byte-identical).
        self.level.prune_hierarchy();

        // Resolve `[field]` sections — every
        // `bind =`/`drive =`/`follow =` name against the now-finished level (nodes,
        // pipes and notes are all populated by this point). Empty ⇒ no fields
        // (byte-identical to before this feature existed).
        self.level.resolved_fields =
            fields::resolve(&self.level).map_err(|(line, msg)| self.err(line, msg))?;

        Ok(self.level)
    }
}

/// The in-plane (non-flat-axis) cell counts of a portal face region, as the
/// SORTED pair `(min_span, max_span)` so two faces with the same opening shape
/// compare equal regardless of which grid axes they lie in (e.g. a north wall's
/// x×z opening matches a west wall's y×z opening of the same dimensions).
fn face_area(region: ((i32, i32, i32), (i32, i32, i32))) -> (i32, i32) {
    let ((x0, y0, z0), (x1, y1, z1)) = region;
    let dx = (x1 - x0).abs();
    let dy = (y1 - y0).abs();
    let dz = (z1 - z0).abs();
    // Exactly one of dx/dy/dz is 0 (the flat axis) for a valid face; collect the
    // other two as the opening's in-plane spans.
    let mut spans = [dx, dy, dz];
    spans.sort_unstable();
    // spans[0] is the smallest; for a wall face it is the flat axis (0). The
    // opening shape is the other two.
    (spans[1], spans[2])
}

/// Validate that `region` is a single flat wall face sitting on a grid boundary
/// of a `size` room. Returns a clear `[portal NAME] FIELD: …` error otherwise.
fn validate_portal_face(
    region: ((i32, i32, i32), (i32, i32, i32)),
    size: (usize, usize, usize),
    name: &str,
    field: &str,
) -> Result<(), String> {
    let ((x0, y0, z0), (x1, y1, z1)) = region;
    let (sx, sy, sz) = (size.0 as i32, size.1 as i32, size.2 as i32);
    let flat_x = x0 == x1;
    let flat_y = y0 == y1;
    let flat_z = z0 == z1;
    let flat_count = [flat_x, flat_y, flat_z].iter().filter(|&&f| f).count();
    if flat_count != 1 {
        return Err(format!(
            "[portal {name}] {field}: must be a flat wall face (exactly one axis \
             with equal endpoints), got {flat_count} flat axes"
        ));
    }
    // The flat axis must lie on a grid boundary (0 or size-1) so it is a wall.
    let on_boundary = if flat_x {
        x0 == 0 || x0 == sx - 1
    } else if flat_y {
        y0 == 0 || y0 == sy - 1
    } else {
        z0 == 0 || z0 == sz - 1
    };
    if !on_boundary {
        return Err(format!(
            "[portal {name}] {field}: the flat axis must sit on a room boundary \
             (a wall), not in mid-air"
        ));
    }
    // Every cell must be in bounds.
    let in_range = |c: i32, hi: i32| (0..hi).contains(&c);
    if !in_range(x0.min(x1), sx)
        || !in_range(x0.max(x1), sx)
        || !in_range(y0.min(y1), sy)
        || !in_range(y0.max(y1), sy)
        || !in_range(z0.min(z1), sz)
        || !in_range(z0.max(z1), sz)
    {
        return Err(format!(
            "[portal {name}] {field}: region out of room bounds"
        ));
    }
    Ok(())
}

/// Build a [`ParseError`] without borrowing the parser — usable inside
/// closures that already hold a `&mut` into the level.
fn err_at(src: &str, line: usize, msg: impl Into<String>) -> ParseError {
    ParseError {
        src: src.to_string(),
        line,
        msg: msg.into(),
    }
}

// ---- free helper functions -------------------------------------------------

/// Read the SOLID voxel cells of a `.vox` byte blob with an optional integer
/// `scale` downscale, deduplicated, as `(x, y, z)` cells (z-up — `.vox` is
/// already right-handed Z-up, matching our world).
///
/// Flattens EVERY model's voxels (a `.vox` voxel is solid by definition — it
/// occupies its cell). When `scale > 1`, keeps voxels on the `scale` lattice and
/// divides the coordinate, so a large model packs contiguously at a smaller
/// extent. Uses `dot_vox::load_bytes` — the SAME loader the importer uses — so no
/// `.vox` parsing is reimplemented here.
/// Decode the image file an authored `[texture]`/`[sky]` path names into a
/// [`re_core::TextureImage`] (RGBA8, top-left origin) via the `image` crate.
///
/// `base` is the `.lvl`'s own path ([`Level::source_path`]); the authored `path` is
/// resolved against it by [`assets::resolve_level_asset`] — the level's own
/// directory, then the level TREE's parent, then the CWD. It used to be read
/// straight off the process CWD, which is why `assets/texture/…` silently failed to
/// load whenever the game was not started from the repo root (including under
/// `cargo test`, where the golden frames then pinned the untextured fallback).
///
/// Returns an error string on an unresolvable/unreadable file or a decode failure;
/// the caller decides whether that is fatal (`decode_textures_checked`) or skippable
/// (`decode_textures`). An unresolvable path reports the whole search.
///
/// The decoder is forced to `RGBA8` so the buffer matches
/// [`re_core::TextureImage`]'s `width*height*4` row-major layout exactly. No
/// terminal, no GPU — a pure byte transform.
fn decode_texture_file(
    base: Option<&Path>,
    path: &str,
    max_size: Option<u32>,
) -> Result<re_core::TextureImage, String> {
    let resolved = assets::resolve_level_asset(base, path).map_err(|e| e.to_string())?;
    let bytes = fs::read(&resolved).map_err(|e| format!("{}: {e}", resolved.display()))?;
    let dynimg = image::load_from_memory(&bytes).map_err(|e| e.to_string())?;
    // `max_size` (engine temporal-stability spec §1.4): cap the longest edge BEFORE
    // the RGBA8 buffer is materialised, so an oversized source never costs its full
    // `w*h*4` bytes for the whole session. Aspect ratio is preserved and the filter
    // is a box average (`Triangle`), which is what a minifying sampler wants — a
    // nearest-neighbour reduction would alias the very detail the cap is meant to
    // fold away. `None`, or an image already within the cap, is left untouched.
    let dynimg = cap_longest_edge(dynimg, max_size);
    let rgba = dynimg.to_rgba8();
    let (w, h) = (rgba.width(), rgba.height());
    Ok(re_core::TextureImage::new(w, h, rgba.into_raw()))
}

/// Decode EVERY frame of the animated image file an authored `[texture] frames = …`
/// names, into a [`re_core::TextureSet`] (spec §1.4).
///
/// Resolution, the `max_size` cap and the RGBA8 layout are IDENTICAL to
/// [`decode_texture_file`]'s — the cap is applied to each frame as it is produced, so
/// the peak cost is one native-size frame plus the capped set, never the whole
/// uncapped animation. (`assets/texture/gen_anim.gif` — the committed, generated
/// fixture, `re_voximport::texfixture::anim_gif` — is 8 frames of 48x24: a few KB
/// uncapped; the cap matters for a real download-sized clip, which is what this
/// generator stands in for.)
///
/// A file the GIF decoder will not open (e.g. an author pointed `frames =` at a PNG)
/// is NOT an error: it falls back to the single-image decode and yields a ONE-frame
/// set — the same thing a one-frame GIF yields, and exactly a still texture. A file
/// that opens but yields no frames at all likewise degrades to the single-image
/// decode rather than producing an empty set.
///
/// Pure byte transform: no terminal, no GPU, no clock. Called ONCE per texture at
/// load — never per cast, per cell, or per frame of gameplay.
fn decode_texture_frames(
    base: Option<&Path>,
    path: &str,
    max_size: Option<u32>,
) -> Result<Vec<re_core::TextureImage>, String> {
    use image::AnimationDecoder;

    let resolved = assets::resolve_level_asset(base, path).map_err(|e| e.to_string())?;
    let bytes = fs::read(&resolved).map_err(|e| format!("{}: {e}", resolved.display()))?;

    let decoded: Option<Vec<re_core::TextureImage>> =
        match image::codecs::gif::GifDecoder::new(std::io::Cursor::new(&bytes)) {
            Ok(dec) => {
                let mut out = Vec::new();
                // `into_frames` is LAZY: convert + cap each frame before pulling the
                // next, so the uncapped pixels of at most one frame are ever live.
                for frame in dec.into_frames() {
                    let frame = frame.map_err(|e| e.to_string())?;
                    let img = image::DynamicImage::ImageRgba8(frame.into_buffer());
                    let img = cap_longest_edge(img, max_size);
                    let rgba = img.to_rgba8();
                    let (w, h) = (rgba.width(), rgba.height());
                    out.push(re_core::TextureImage::new(w, h, rgba.into_raw()));
                }
                if out.is_empty() {
                    None
                } else {
                    Some(out)
                }
            }
            // Not a GIF (or a malformed header): fall through to the still decode so
            // `frames = photo.png` is a one-frame set rather than a load failure.
            Err(_) => None,
        };
    match decoded {
        Some(frames) => Ok(frames),
        None => decode_texture_file(base, path, max_size).map(|img| vec![img]),
    }
}

/// Box-downsample `img` so its longest edge is at most `max_size`, preserving aspect.
/// `None`, a `0` cap, or an image already within the cap is returned untouched.
///
/// Shared by the colour decode and the relief (`height =`) decode so BOTH channels
/// of one `[texture]` obey the same memory cap — a 4096-px height map beside a
/// capped 1024-px colour image would otherwise cost 4× the pixels for a channel that
/// is immediately resampled down to the colour image's size anyway.
fn cap_longest_edge(img: image::DynamicImage, max_size: Option<u32>) -> image::DynamicImage {
    match max_size {
        Some(cap) if cap > 0 && img.width().max(img.height()) > cap => {
            let (w, h) = (img.width() as f64, img.height() as f64);
            let scale = cap as f64 / w.max(h);
            let nw = ((w * scale).round() as u32).max(1);
            let nh = ((h * scale).round() as u32).max(1);
            img.resize_exact(nw, nh, image::imageops::FilterType::Triangle)
        }
        _ => img,
    }
}

/// Decode a `[texture NAME] height = <path>` RELIEF map into `target_w * target_h`
/// bytes, one per texel, matching the COLOUR image's dimensions exactly.
///
/// Resolved through the SAME [`assets::resolve_level_asset`] search and capped by the
/// SAME `max_size` as the colour image, then converted to 8-bit luminance (a height
/// map is a greyscale field; a colour source is read by its luma).
///
/// # Dimension mismatch is expected, not an error
///
/// An author will not always export the relief at the colour image's exact size (and
/// `max_size` can round the two differently anyway). When the dimensions differ the
/// relief is RESAMPLED with the same box/`Triangle` filter used for minification, so
/// texel `(x, y)` of the colour image and of the height map always describe the same
/// point on the surface. Nothing here can panic on a mismatch:
/// * a zero-area TARGET yields an empty channel (the colour image is zero-area too,
///   and `TextureImage::with_height_map` normalizes to the same zero length);
/// * a zero-area SOURCE yields an all-`0` (flat, lowest) channel rather than being
///   handed to `imageops::resize`, which has no meaningful answer for it.
fn decode_height_map(
    base: Option<&Path>,
    path: &str,
    max_size: Option<u32>,
    target_w: u32,
    target_h: u32,
) -> Result<Vec<u8>, String> {
    let resolved = assets::resolve_level_asset(base, path).map_err(|e| e.to_string())?;
    let bytes = fs::read(&resolved).map_err(|e| format!("{}: {e}", resolved.display()))?;
    let dynimg = image::load_from_memory(&bytes).map_err(|e| e.to_string())?;
    let dynimg = cap_longest_edge(dynimg, max_size);
    let luma = dynimg.to_luma8();
    let needed = (target_w as usize).saturating_mul(target_h as usize);
    if needed == 0 {
        return Ok(Vec::new());
    }
    if luma.width() == 0 || luma.height() == 0 {
        return Ok(vec![0u8; needed]);
    }
    if luma.width() == target_w && luma.height() == target_h {
        return Ok(luma.into_raw());
    }
    Ok(image::imageops::resize(
        &luma,
        target_w,
        target_h,
        image::imageops::FilterType::Triangle,
    )
    .into_raw())
}

fn load_vox_solids(bytes: &[u8], scale: u32) -> Result<Vec<Coord3>, String> {
    let data = dot_vox::load_bytes(bytes).map_err(|e| format!("vox parse failed: {e}"))?;
    let scale = scale.max(1);
    let mut seen = std::collections::HashSet::new();
    let mut cells = Vec::new();
    for model in &data.models {
        for v in &model.voxels {
            if scale > 1
                && (!(v.x as u32).is_multiple_of(scale)
                    || !(v.y as u32).is_multiple_of(scale)
                    || !(v.z as u32).is_multiple_of(scale))
            {
                continue;
            }
            let cell = (
                (v.x as i32) / scale as i32,
                (v.y as i32) / scale as i32,
                (v.z as i32) / scale as i32,
            );
            if seen.insert(cell) {
                cells.push(cell);
            }
        }
    }
    Ok(cells)
}

// ───────────────────────── `[effects] snow` → emitters ──────────────────────
//
// `[effects] snow` is the game's COLD narrative axis (levels L3 / L6 / L8). It
// parsed, round-tripped and showed up in the editor — and then NOTHING read it,
// so a level that declares itself cold rendered identically to a warm one.
//
// It is resolved HERE, at load, into the renderer's existing ambient-particle
// surface (`re_world::EmitterSpec` → `re_render::Emitter`), rather than as a new
// screen-space overlay pass: the emitter path is already deterministic (fixed seed),
// already depth-tested against geometry (so flakes go BEHIND the pipes and walls
// instead of pasting over them), and already composited in its own `GlyphOnly` layer.
// A bespoke overlay would have duplicated all three.
//
// The one design decision worth stating: snowfall is a VOLUME, but an emitter births
// at a POINT. So the knob expands into a small 3-D LATTICE of emitters filling the
// room's interior, which gives an approximately uniform flake field from the first
// frame — a single ceiling point would read as a leaking pipe, and (because a still
// `--dump` only integrates a fraction of a second) would not have fallen anywhere yet.

/// Nominal spacing, in world cells, between synthesized snow emitters on the two
/// horizontal axes. The lattice is what gives the flake field its SPATIAL spread —
/// particles cluster near their own emitter for the first second of life, so a coarse
/// lattice reads as a handful of dripping points rather than as weather.
const SNOW_LATTICE_STEP_XY: f32 = 2.5;

/// Nominal vertical spacing between snow-emitter layers. Smaller than the horizontal
/// step because a room is only a few cells tall and the flake field must reach the
/// floor band, not just hang under the ceiling.
const SNOW_LATTICE_STEP_Z: f32 = 1.2;

/// Hard cap on synthesized snow emitters, whatever the room size. Bounds both the
/// per-frame integration cost and the emitter list a `--dump` resolves.
const SNOW_EMITTER_MAX: usize = 48;

/// TOTAL particles per second across the whole lattice at `snow = 1.0`, scaled
/// linearly by the authored density and then split evenly between lattice points.
///
/// Budgeted as a TOTAL, not per-emitter, so refining the lattice buys spatial spread
/// without multiplying the particle count: the live field is
/// `SNOW_TOTAL_RATE · density · SNOW_LIFETIME` flakes (≈ 460 at the campaign's 0.4,
/// ≈ 1150 at a full whiteout), well inside `re_render`'s 4096-particle pool.
const SNOW_TOTAL_RATE: f32 = 520.0;

/// Flake lifetime (seconds) and its symmetric spread — long enough to fall a couple
/// of cells, short enough that the field refreshes.
const SNOW_LIFETIME: f32 = 2.2;
const SNOW_LIFETIME_SPREAD: f32 = 0.7;

/// Mean fall speed (cells/second) and its spread. Snow drifts; it does not rain.
const SNOW_SPEED: f32 = 0.85;
const SNOW_SPEED_SPREAD: f32 = 0.35;

/// Downward cone half-angle (degrees), lateral turbulence, and swirl — the three
/// terms that turn a straight drop into a drift.
const SNOW_SPREAD_DEG: f32 = 55.0;
const SNOW_PATH_JITTER: f32 = 0.30;
const SNOW_SWIRL: f32 = 0.12;

/// Gentle downward acceleration: enough to bias the drift toward the floor without
/// turning flakes into sleet.
const SNOW_ACCEL_Z: f32 = -0.18;

/// Cold colour ramp over a flake's age: near-white → pale ice blue → slate.
const SNOW_COLORS: [u32; 3] = [0x00E8_F2FF, 0x00AF_C8E4, 0x004A_5C74];

/// The flake glyph cycle, indexed by normalised age (born faint → full → fading).
const SNOW_GLYPHS: [char; 4] = ['·', '*', '·', '.'];

/// Base RNG seed for synthesized snow. Each lattice point mixes its index in, so the
/// emitters differ from one another but the whole field is reproducible frame to
/// frame and run to run (golden-frame safe).
const SNOW_SEED_BASE: u64 = 0x5C01_D5E1_0F41_1EE5;

/// Odd multiplier that decorrelates the per-lattice-point seeds.
const SNOW_SEED_STRIDE: u64 = 0x9E37_79B9_7F4A_7C15;

/// Deterministic integer-cell nudge offsets tried (in this priority order) when a
/// lattice point's own cell is solid. Horizontal moves come first: the occluding
/// geometry this fix targets (e.g. `L8-cold-room.lvl`'s full-height shelving rows)
/// is a wall that runs the room's full z-extent, so a vertical nudge alone almost
/// never escapes it, while the aisle beside it is one cell away horizontally.
/// `(0,0,0)` (the original cell) is NOT included — the caller only consults this
/// table once the original cell has already tested solid.
const SNOW_NUDGE_OFFSETS: [(i32, i32, i32); 12] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (1, 1, 0),
    (-1, -1, 0),
    (1, -1, 0),
    (-1, 1, 0),
    (0, 0, 1),
    (0, 0, -1),
    (2, 0, 0),
    (-2, 0, 0),
];

/// True if `(x, y, z)` (an interior world cell) is solid enough to bury a snow
/// emitter: an ordinary raycaster solid ([`CellKind::is_solid`] — walls/objects),
/// OR a [`CellKind::Ceiling`]/[`CellKind::Floor`] slab. The lattice never targets
/// the room's floor/ceiling layer itself ([`snow_emitters`] confines it to
/// `1..size-1`), so in practice this only ever fires on interior solids (shelving,
/// pillars, objects) — but a decal/vox stamp can still place a Floor/Ceiling cell
/// one layer in from the shell, so both are covered.
#[inline]
fn snow_cell_blocks(map: &Map, x: i32, y: i32, z: i32) -> bool {
    let kind = map.cell_at(x, y, z);
    kind.is_solid() || matches!(kind, CellKind::Ceiling | CellKind::Floor)
}

/// Resolve one lattice candidate at continuous position `(x, y, z)` against the
/// BUILT map: if its own cell is clear, keep it verbatim (unaffected points are
/// byte-identical to before — same position, same seed). If it is buried in a
/// solid cell (`L8-cold-room.lvl`'s shelving is the motivating case), try
/// [`SNOW_NUDGE_OFFSETS`] in order and relocate to the CENTRE of the first clear
/// neighbour still inside the room's interior (`1..size-1` on every axis, so a
/// nudge never births a flake inside the perimeter shell). `None` if the original
/// cell AND every offset are blocked or out of the interior — the point is
/// dropped (its share of the total particle budget is redistributed among the
/// survivors by the caller, so a handful of drops never dims the whole field).
fn snow_resolve_lattice_point(
    map: &Map,
    x: f32,
    y: f32,
    z: f32,
    interior_max: (i32, i32, i32),
) -> Option<(f32, f32, f32)> {
    let (bx, by, bz) = (x.floor() as i32, y.floor() as i32, z.floor() as i32);
    if !snow_cell_blocks(map, bx, by, bz) {
        return Some((x, y, z));
    }
    let (max_x, max_y, max_z) = interior_max;
    for (dx, dy, dz) in SNOW_NUDGE_OFFSETS {
        let (cx, cy, cz) = (bx + dx, by + dy, bz + dz);
        if cx < 1 || cy < 1 || cz < 1 || cx >= max_x || cy >= max_y || cz >= max_z {
            continue;
        }
        if !snow_cell_blocks(map, cx, cy, cz) {
            return Some((cx as f32 + 0.5, cy as f32 + 0.5, cz as f32 + 0.5));
        }
    }
    None
}

/// Expand an `[effects] snow` density into the lattice of emitters that renders it.
///
/// `map` is the level's BUILT geometry (its `size()` gives the room's `(sx, sy,
/// sz)` in cells); the lattice fills the INTERIOR (`1 .. size-1` on each axis), so
/// no flake is born inside the perimeter wall. Each candidate point is then
/// resolved against `map` ([`snow_resolve_lattice_point`]): a point whose own cell
/// is solid interior geometry (e.g. `L8-cold-room.lvl`'s full-height shelving
/// rows) is nudged to the nearest clear neighbour, or dropped if none is found —
/// a blind bounding-box lattice previously placed emitters inside such walls,
/// where the particle pool's occlusion splat (`re_render::emitter`) discards them
/// before they ever reach a visible cell, undercounting the visible flake field.
/// `density` is the authored `0..=1` knob. A non-positive or non-finite density,
/// or a room with no interior, yields no emitters.
fn snow_emitters(map: &Map, density: f32) -> Vec<EmitterSpec> {
    let (sx, sy, sz) = map.size();
    if !density.is_finite() || density <= 0.0 || sx < 3 || sy < 3 || sz < 3 {
        return Vec::new();
    }
    let density = density.clamp(0.0, 1.0);
    // Interior extents (world units) on each axis.
    let (ex, ey, ez) = ((sx - 2) as f32, (sy - 2) as f32, (sz - 2) as f32);
    // Lattice counts: at least one per axis, then trimmed together to the cap.
    let axis_n = |extent: f32, step: f32| ((extent / step).round() as usize).max(1);
    let mut nx = axis_n(ex, SNOW_LATTICE_STEP_XY);
    let mut ny = axis_n(ey, SNOW_LATTICE_STEP_XY);
    let mut nz = axis_n(ez, SNOW_LATTICE_STEP_Z);
    // Trim the largest axis until the product fits the cap — deterministic and
    // bounded (each pass strictly shrinks a count ≥ 2).
    while nx * ny * nz > SNOW_EMITTER_MAX {
        let m = nx.max(ny).max(nz);
        if m <= 1 {
            break;
        }
        if nx == m {
            nx -= 1;
        } else if ny == m {
            ny -= 1;
        } else {
            nz -= 1;
        }
    }
    let interior_max = (sx as i32 - 1, sy as i32 - 1, sz as i32 - 1);
    // PASS 1 — resolve every lattice candidate against the built geometry, keeping
    // the ORIGINAL lattice `idx` (drives the seed) alongside its (possibly
    // nudged) position. Unaffected points are untouched: same position, same idx,
    // same seed as before this fix — only an occluded point's position (never its
    // idx) changes, and only a truly buried one drops out of `resolved` entirely.
    let at = |i: usize, n: usize, extent: f32| 1.0 + extent * (i as f32 + 0.5) / n as f32;
    let mut resolved: Vec<(usize, f32, f32, f32)> = Vec::with_capacity(nx * ny * nz);
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let idx = (iz * ny + iy) * nx + ix;
                let (x, y, z) = (at(ix, nx, ex), at(iy, ny, ey), at(iz, nz, ez));
                if let Some(p) = snow_resolve_lattice_point(map, x, y, z, interior_max) {
                    resolved.push((idx, p.0, p.1, p.2));
                }
            }
        }
    }
    if resolved.is_empty() {
        return Vec::new();
    }
    // PASS 2 — redistribute the TOTAL particle budget over the SURVIVING points
    // only, so a few drops (deep inside a wall, no clear neighbour) never dim the
    // rest of the field: the total system-wide particle rate stays
    // `SNOW_TOTAL_RATE · density` regardless of how many candidates the built
    // geometry rejected.
    let rate = SNOW_TOTAL_RATE * density / resolved.len() as f32;
    let max_particles = ((rate * SNOW_LIFETIME).ceil() as u32).clamp(4, 64);
    let mut out = Vec::with_capacity(resolved.len());
    for (idx, x, y, z) in resolved {
        let spec = EmitterSpec {
            kind: re_world::EmitterKind::Standard,
            origin: re_world::EmitterOrigin::World(re_core::Vec3::new(x, y, z)),
            colors: SNOW_COLORS,
            color_mode: re_world::ColorMode::RampOverAge,
            family: String::from("haze_dot"),
            glyph_set: Some(SNOW_GLYPHS.to_vec()),
            bold: re_world::BoldMode::ByProximity,
            direction: re_core::Vec3::new(0.0, 0.0, -1.0),
            spread_deg: SNOW_SPREAD_DEG,
            size: 1.0,
            growth: 1.0,
            rate,
            burst: None,
            speed: SNOW_SPEED,
            speed_spread: SNOW_SPEED_SPREAD,
            path_jitter: SNOW_PATH_JITTER,
            seed: SNOW_SEED_BASE ^ (idx as u64).wrapping_add(1).wrapping_mul(SNOW_SEED_STRIDE),
            swirl: SNOW_SWIRL,
            lifetime: SNOW_LIFETIME,
            lifetime_spread: SNOW_LIFETIME_SPREAD,
            accel: re_core::Vec3::new(0.0, 0.0, SNOW_ACCEL_Z),
            max_particles,
            blend: re_world::EmitterBlend::KeepBg,
            // SYNTHESIZED: regenerated from `[effects] snow` on every load, so
            // a writer/editor must never persist/display this as an object of
            // its own (see `EmitterSpec::synthesized`).
            synthesized: true,
            // `..Default::default()` (not a new field with an explicit value): this
            // literal predates most of `EmitterSpec`'s variety/randomisation fields,
            // so any newly-added opt-in field falls through to its neutral default
            // here (no snow behaviour change) instead of failing to compile.
            ..Default::default()
        };
        out.push(spec.validated());
    }
    out
}

/// The one character that, **in column 0**, marks a level-file comment inside an
/// otherwise-verbatim `[note]` body. Named so the placard capture rule is a
/// constant, not a scattered literal; see the `[note]` branch of `Parser::run`.
const NOTE_COMMENT_MARK: char = '#';

/// The one character that, **in column 0**, ends a `[note]` body by opening the
/// next section. The structural twin of [`NOTE_COMMENT_MARK`]: inside a note body
/// the grammar reads column 0 as STRUCTURE and any indent as CONTENT, and it now
/// says so for both markers.
const NOTE_SECTION_MARK: char = '[';

/// True if `line`, taken as ONE line of a `[note]` body, would be misread by the
/// parser as a level-file comment — column 0 == [`NOTE_COMMENT_MARK`]. The single
/// discriminator shared by `Parser::run`'s capture and [`escape_note_body_line`]'s
/// writer-side escape, so the two can never drift apart.
fn note_body_line_is_comment(line: &str) -> bool {
    line.starts_with(NOTE_COMMENT_MARK)
}

/// True if `line`, taken as ONE line of a `[note]` body, would be misread by the
/// parser as a SECTION HEADER and end the note — column 0 == [`NOTE_SECTION_MARK`].
/// The `[` counterpart of [`note_body_line_is_comment`], shared by the same two
/// sides so the capture rule and the writer's escape cannot drift.
///
/// This is what makes `    [ STEWARDSHIP - DO NOT POWER DOWN ]` (indented, in
/// `levels/L7-hidden-depths.lvl`) placard TEXT instead of a bogus section header
/// that silently truncated its note.
fn note_body_line_is_header(line: &str) -> bool {
    line.starts_with(NOTE_SECTION_MARK)
}

/// Escape ONE `[note]` body line for writing, so neither of the parser's column-0
/// rules — `note_body_line_is_comment` (`#`) nor `note_body_line_is_header`
/// (`[`) — can ever fire on AUTHORED content.
///
/// `[note]` bodies are written verbatim and unindented by
/// `re_voximport::to_lvl_text`. The editor's caret text field deliberately
/// allows a leading `#` in placard text (it is legitimate placard prose, e.g.
/// "#9 MINISTRY"), and the parser deliberately treats a column-0 `#` as a
/// level-file comment (so a dev banner like `# ---- nodes ----` typed after a
/// note is never printed on the wall). Written naively, an authored line
/// starting with `#` collides with that comment rule and is SILENTLY DROPPED
/// on the next load. `re_voximport::to_lvl_text` calls this per body line
/// before emitting a `[note]` section to close that gap.
///
/// The same collision exists for `[`: a column-0 bracketed line ends the note, so
/// an authored line like `[ STEWARDSHIP - DO NOT POWER DOWN ]` written at column 0
/// would truncate its own placard on the next load. Both markers get the same
/// escape.
///
/// The escape is a single leading space: it moves the line's first character
/// off column 0, so
/// - the parser no longer misreads it as a comment or a header (the bugs this
///   closes), and
/// - a SECOND save/reparse cycle is a no-op — the escaped line no longer
///   starts with `#` or `[`, so it needs no further escaping and round-trips
///   byte-identically from here on (idempotent).
///
/// Every shipped `[note]` body line is already indented (never `#` or `[` in
/// column 0), so this is a no-op — same bytes out as in — for all existing level
/// content; it only ever changes a line that would otherwise be corrupted.
pub fn escape_note_body_line(line: &str) -> std::borrow::Cow<'_, str> {
    if note_body_line_is_comment(line) || note_body_line_is_header(line) {
        std::borrow::Cow::Owned(format!(" {line}"))
    } else {
        std::borrow::Cow::Borrowed(line)
    }
}

/// Strip a `#`-style comment. A `#` starts a comment only as a full-line lead
/// (optional leading whitespace) or as ` #` where what follows is **not** a
/// 6-hex-digit color (so `color = #FF0000` survives). Port of legacy
/// `str_strip_comment`.
///
/// ## The lone-`#` value exception
///
/// One further case is NOT a comment: a `#` that is the last non-whitespace
/// character of the line, on a line whose value would otherwise be left EMPTY.
/// `[material green_matrix] glyph = #` (`levels/12-mirror.lvl`) is someone
/// specifying the `#` glyph, and the old rule silently reduced it to
/// `glyph       =` — an empty value, which is never what anybody wrote. An
/// EMPTY comment carries no information, so reading it as one could only ever
/// destroy data.
///
/// Both conditions are required, which keeps the blast radius at zero:
/// * `pos = 1,2,3 #` still strips (the value is non-empty, so the `#` is a
///   stray trailing comment, and treating it as data would make `1,2,3 #` fail
///   to parse as a coord triple).
/// * `speed = 2 # fast` still strips (the comment has a body).
/// * A line with no `=` at all (a `[hierarchy]` link, a shape row) still
///   strips, exactly as before.
fn strip_comment(s: &str) -> &str {
    let trimmed = s.trim_start();
    if trimmed.starts_with('#') {
        return "";
    }
    let bytes = s.as_bytes();
    let mut i = 0;
    while i + 1 < bytes.len() {
        if (bytes[i] == b' ' || bytes[i] == b'\t') && bytes[i + 1] == b'#' {
            // is what follows a 6-hex color?
            let after = &s[i + 2..];
            let hex_run = after
                .chars()
                .take(6)
                .take_while(|c| c.is_ascii_hexdigit())
                .count();
            if hex_run < 6 {
                if hash_at_is_value(s, i + 1) {
                    return s;
                }
                return &s[..i];
            }
            // a #RRGGBB color; keep scanning for a later " # "
        }
        i += 1;
    }
    s
}

/// True when the `#` at byte index `hash` of `line` is a VALUE rather than the
/// start of a comment: nothing but whitespace follows it, AND the text before it
/// is a `key =` whose value is empty. See `strip_comment`'s "lone-`#` value
/// exception" — the only case where an empty comment body would otherwise delete
/// authored data (`[material] glyph = #`).
///
/// **Public because it must not be reimplemented.** `ts_editor`'s save-time
/// `split_trailing_comment` mirrors `strip_comment`'s comment rule so an
/// open→save neither invents nor loses a trailing comment; when this exception
/// was added and only `re_level` knew about it, the editor captured a phantom
/// ` #` suffix and re-emitted `glyph = # #` on five shipped levels. One
/// discriminator, one source of truth — the same reason
/// [`escape_note_body_line`] is public.
///
/// `hash` must index a `#` byte; anything else returns `false`.
#[must_use]
pub fn hash_at_is_value(line: &str, hash: usize) -> bool {
    if line.as_bytes().get(hash) != Some(&b'#') {
        return false;
    }
    line[hash + 1..].trim().is_empty()
        && line[..hash]
            .split_once('=')
            .is_some_and(|(_, v)| v.trim().is_empty())
}

/// Return the interior of a `[section ...]` header line, or `None`.
fn section_header(line: &str) -> Option<&str> {
    let line = line.trim();
    if line.starts_with('[') && line.ends_with(']') {
        Some(line[1..line.len() - 1].trim())
    } else {
        None
    }
}

/// Split `key = value` on the first `=`. Returns trimmed `(k, v)`.
fn split_kv(line: &str) -> Option<(&str, &str)> {
    let (k, v) = line.split_once('=')?;
    Some((k.trim(), v.trim()))
}

/// Parse `"X,Y,Z"` / `"X Y Z"` / `"(X,Y,Z)"` into three ints.
fn parse_coord3(s: &str) -> Option<Coord3> {
    let s = s.trim().trim_start_matches('(').trim_end_matches(')');
    let nums: Vec<i32> = s
        .split([',', ' ', '\t'])
        .filter(|t| !t.is_empty())
        .map(|t| t.trim().parse::<i32>())
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    if nums.len() == 3 {
        Some((nums[0], nums[1], nums[2]))
    } else {
        None
    }
}

/// Parse `"(x0,y0,z0)-(x1,y1,z1)"` into two corners.
fn parse_region(s: &str) -> Option<(Coord3, Coord3)> {
    let lp1 = s.find('(')?;
    let rp1 = s[lp1..].find(')')? + lp1;
    let dash = s[rp1..].find('-')? + rp1;
    let lp2 = s[dash..].find('(')? + dash;
    let rp2 = s[lp2..].find(')')? + lp2;
    let a = parse_coord3(&s[lp1 + 1..rp1])?;
    let b = parse_coord3(&s[lp2 + 1..rp2])?;
    Some((a, b))
}

/// Parse a Scalable Material Overlay `scale = …` value (P0).
///
/// `none` ⇒ [`ScaleRule::None`], `crisp` ⇒ [`ScaleRule::Crisp`], and
/// `self_similar` / `selfsimilar` / `fractal` ⇒ [`ScaleRule::SelfSimilar`] with
/// optional `arms:N twist:F warp:F` (any order, space/comma separated) mapping to
/// the [`SelfSimilarParams`] POD mirror of `re_render::fractal::FractalParams`.
/// Missing fractal sub-params default to the inert [`SelfSimilarParams::default`].
fn parse_scale_rule(s: &str) -> Option<ScaleRule> {
    let s = s.trim();
    let mut it = s.split([' ', '\t', ',']).filter(|t| !t.is_empty());
    let head = it.next()?.to_ascii_lowercase();
    match head.as_str() {
        "none" => Some(ScaleRule::None),
        "crisp" => Some(ScaleRule::Crisp),
        "self_similar" | "selfsimilar" | "fractal" => {
            let mut p = SelfSimilarParams::default();
            for tok in it {
                let (key, val) = tok.split_once(':')?;
                match key.to_ascii_lowercase().as_str() {
                    "arms" => p.arms = val.parse::<u8>().ok()?,
                    "warp" | "octave_warp" => p.octave_warp = val.parse::<f32>().ok()?,
                    "twist" => p.twist = val.parse::<f32>().ok()?,
                    _ => return None,
                }
            }
            Some(ScaleRule::SelfSimilar(p))
        }
        _ => None,
    }
}

/// Parse a `[line] path` polyline: `;`-separated `x,y,z` float vertices, e.g.
/// `"1,2,0; 4,2,0; 4,5,0"`. Returns `None` if any vertex is malformed.
fn parse_polyline(s: &str) -> Option<Vec<Vec3>> {
    let mut pts = Vec::new();
    for group in s.split(';').map(str::trim).filter(|g| !g.is_empty()) {
        let nums: Vec<f32> = group
            .split([',', ' ', '\t'])
            .filter(|t| !t.is_empty())
            .map(|t| t.trim().parse::<f32>())
            .collect::<Result<Vec<_>, _>>()
            .ok()?;
        if nums.len() != 3 || nums.iter().any(|n| !n.is_finite()) {
            return None;
        }
        pts.push(Vec3::new(nums[0], nums[1], nums[2]));
    }
    Some(pts)
}

/// Parse an `[inset] at_octave` value: `"N"` (single) or `"N-M"` (inclusive
/// range). Returns the normalised `(min, max)` octave pair.
fn parse_octave_range(s: &str) -> Option<(i32, i32)> {
    let s = s.trim();
    if let Some((a, b)) = s.split_once('-') {
        // Handle a leading negative on the first bound (`-1-3`): try the simple
        // split first, fall back to parsing as a single negative number.
        if let (Ok(a), Ok(b)) = (a.trim().parse::<i32>(), b.trim().parse::<i32>()) {
            return Some((a.min(b), a.max(b)));
        }
    }
    let n = s.parse::<i32>().ok()?;
    Some((n, n))
}

/// Parse `#RRGGBB` into `0x00RRGGBB`.
fn parse_color(s: &str) -> Option<u32> {
    let s = s.trim();
    let hex = s.strip_prefix('#')?;
    if hex.len() < 6 {
        return None;
    }
    u32::from_str_radix(&hex[..6], 16).ok()
}

/// Parse a float triple `"X Y Z"` / `"X,Y,Z"` / `"(X,Y,Z)"` (the `[emitter]`
/// `origin_world` / `direction` / `accel` value). Returns `None` unless exactly
/// three finite-parseable floats are present.
fn parse_vec3(s: &str) -> Option<(f32, f32, f32)> {
    let s = s.trim().trim_start_matches('(').trim_end_matches(')');
    let nums: Vec<f32> = s
        .split([',', ' ', '\t'])
        .filter(|t| !t.is_empty())
        .map(|t| t.trim().parse::<f32>())
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    if nums.len() == 3 {
        Some((nums[0], nums[1], nums[2]))
    } else {
        None
    }
}

/// Parse a float PAIR `"A B"` / `"A, B"` (the `[atmosphere] fog_height` value:
/// base height then falloff). Accepts the same separators as [`parse_vec3`].
fn parse_pair(s: &str) -> Option<(f32, f32)> {
    let s = s.trim().trim_start_matches('(').trim_end_matches(')');
    let nums: Vec<f32> = s
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|t| !t.is_empty())
        .map(|t| t.trim().parse::<f32>())
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    if nums.len() == 2 {
        Some((nums[0], nums[1]))
    } else {
        None
    }
}

/// Parse a `[material] span` value (M3a): `"ALONG_MIN,ALONG_SIZE UP_TOP,UP_SIZE"` —
/// two comma-pairs separated by whitespace, all in world units. Returns `None` unless
/// exactly four finite floats parse AND both sizes are `> 0` (a degenerate span is a
/// level error, not silently dropped — keeps an authored figure from collapsing).
/// Parse `tile = ALONG UP` — the span-relative repeat count of a `recipe = image`
/// material. Two whitespace- or comma-separated numbers; both must be FINITE and
/// strictly POSITIVE.
///
/// Returns `None` on anything else, which the caller turns into a line-numbered
/// parse error. `0` is rejected because it collapses the whole figure onto one texel
/// row and a negative factor mirrors rather than repeats — neither is a "repeat
/// count", and both would render as a renderer bug rather than as an authoring
/// mistake if they were quietly accepted.
fn parse_image_tile(s: &str) -> Option<(f32, f32)> {
    let mut it = s.split([' ', '\t', ',']).filter(|t| !t.is_empty());
    let a: f32 = it.next()?.parse().ok()?;
    let b: f32 = it.next()?.parse().ok()?;
    if it.next().is_some() {
        return None; // a third number is a typo, not a tile.
    }
    (a.is_finite() && b.is_finite() && a > 0.0 && b > 0.0).then_some((a, b))
}

/// Parse `relief = height_scale=1.2 normal_strength=0.9` — the two knobs that turn a
/// `recipe = image` material into a lit relief surface.
///
/// Uses the SAME `k=v` whitespace-separated token convention as
/// [`parse_recipe`] (`recipe = KIND k=v …`), so authors learn one syntax. Both knobs
/// are optional and fall back to [`re_world::ImageRelief::DEFAULT`]; a bare
/// `relief =` therefore means "relief on, at full strength".
///
/// UNLIKE `parse_recipe`'s permissive knob table, an UNKNOWN knob name is an ERROR.
/// A recipe knob table is fixed and documented; `relief` has exactly two knobs, and
/// a misspelled one (`height=`, `strength=`) would otherwise parse as "relief on
/// with defaults" and look like the knob had no effect. Returns the message body for
/// the caller to wrap with a line number.
fn parse_image_relief(s: &str) -> Result<re_world::ImageRelief, String> {
    let mut out = re_world::ImageRelief::DEFAULT;
    for tok in s.split_whitespace() {
        let Some((k, val)) = tok.split_once('=') else {
            return Err(format!("'{tok}' is not a KNOB=VALUE token"));
        };
        let f: f32 = val
            .parse()
            .map_err(|_| format!("{k}: '{val}' is not a number"))?;
        if !f.is_finite() || f < 0.0 {
            return Err(format!("{k}: must be finite and >= 0"));
        }
        match k {
            "height_scale" => out.height_scale = f,
            "normal_strength" => out.normal_strength = f,
            other => {
                return Err(format!(
                    "unknown knob '{other}' (want height_scale, normal_strength)"
                ))
            }
        }
    }
    Ok(out)
}

fn parse_image_span(s: &str) -> Option<re_world::ImageSpan> {
    let mut groups = s.split_whitespace();
    let along = groups.next()?;
    let up = groups.next()?;
    if groups.next().is_some() {
        return None; // more than two groups ⇒ malformed.
    }
    let pair = |g: &str| -> Option<(f32, f32)> {
        let mut it = g.split(',');
        let a = it.next()?.trim().parse::<f32>().ok()?;
        let b = it.next()?.trim().parse::<f32>().ok()?;
        if it.next().is_some() {
            return None;
        }
        Some((a, b))
    };
    let (along_min, along_size) = pair(along)?;
    let (up_top, up_size) = pair(up)?;
    // Reject a degenerate or NaN size (a collapsed figure is a level error). The
    // explicit `> 0.0` test is false for NaN and for any non-positive value.
    if along_size <= 0.0 || up_size <= 0.0 || along_size.is_nan() || up_size.is_nan() {
        return None;
    }
    Some(re_world::ImageSpan {
        along_min,
        along_size,
        up_top,
        up_size,
    })
}

/// Parse a `[material] subcell` value (M3b):
/// `off | quad | braille | ascii | contour | lines` (case-insensitive).
/// `None` for any other token (a level error reported with a clear message). `Off` is
/// also the default when the key is absent.
fn parse_subcell(s: &str) -> Option<re_world::SubcellMode> {
    match s.trim().to_ascii_lowercase().as_str() {
        "off" | "none" => Some(re_world::SubcellMode::Off),
        "half" | "2x1" => Some(re_world::SubcellMode::Half),
        "quad" | "2x2" => Some(re_world::SubcellMode::Quad),
        "sextant" | "3x2" => Some(re_world::SubcellMode::Sextant),
        "braille" | "2x4" => Some(re_world::SubcellMode::Braille),
        "ascii" => Some(re_world::SubcellMode::Ascii),
        "contour" => Some(re_world::SubcellMode::Contour),
        "lines" | "line" => Some(re_world::SubcellMode::Lines),
        _ => None,
    }
}

/// Parse a `[material] fit` value: `stretch | contain | cover` (case-insensitive).
/// `None` for any other token (a level error reported with a clear message).
/// [`re_world::Fit::Contain`] is also the default when the key is absent.
fn parse_fit(s: &str) -> Option<re_world::Fit> {
    match s.trim().to_ascii_lowercase().as_str() {
        "stretch" => Some(re_world::Fit::Stretch),
        "contain" => Some(re_world::Fit::Contain),
        "cover" => Some(re_world::Fit::Cover),
        _ => None,
    }
}

/// Parse a float pair `"A B"` / `"A,B"` (the `[emitter] origin_screen` value).
fn parse_f32_pair(s: &str) -> Option<(f32, f32)> {
    let nums: Vec<f32> = s
        .split([',', ' ', '\t'])
        .filter(|t| !t.is_empty())
        .map(|t| t.trim().parse::<f32>())
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    if nums.len() == 2 {
        Some((nums[0], nums[1]))
    } else {
        None
    }
}

/// Parse a single REQUIRED float (errors on garbage, unlike the `.unwrap_or`
/// best-effort keys). Used by `[emitter]` numeric keys so a typo is a clear
/// parse error with line context, not a silent default.
fn parse_f32_req(s: &str) -> Option<f32> {
    s.trim().parse::<f32>().ok()
}

/// Parse a single REQUIRED non-negative `u32` (errors on garbage / negatives).
fn parse_u32_req(s: &str) -> Option<u32> {
    s.trim().parse::<u32>().ok()
}

/// Parse a single REQUIRED non-negative `u64` (errors on garbage / negatives).
fn parse_u64_req(s: &str) -> Option<u64> {
    s.trim().parse::<u64>().ok()
}

/// Parse a `0xRRGGBB` / `#RRGGBB` / bare-hex colour token into `0x00RRGGBB`.
/// More permissive than [`parse_color`] (which requires a leading `#`) because
/// `[emitter] colors` authors `0xRRGGBB` per the contract.
fn parse_hex_color(tok: &str) -> Option<u32> {
    let t = tok.trim();
    let hex = t
        .strip_prefix("0x")
        .or_else(|| t.strip_prefix("0X"))
        .or_else(|| t.strip_prefix('#'))
        .unwrap_or(t);
    if hex.len() < 6 || !hex.chars().take(6).all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    u32::from_str_radix(&hex[..6], 16).ok()
}

/// Parse the `[emitter] colors = C0 C1 C2` value: exactly three colour stops.
fn parse_color_triple(s: &str) -> Option<[u32; 3]> {
    let stops: Vec<u32> = s
        .split([',', ' ', '\t'])
        .filter(|t| !t.is_empty())
        .map(parse_hex_color)
        .collect::<Option<Vec<_>>>()?;
    if stops.len() == 3 {
        Some([stops[0], stops[1], stops[2]])
    } else {
        None
    }
}

/// Parse the `[emitter] glyph_set` value: space-separated Unicode scalars. Each
/// whitespace/comma-separated token contributes its FIRST character. `None` when
/// no glyphs are present.
fn parse_glyph_set(s: &str) -> Option<Vec<char>> {
    let glyphs: Vec<char> = s
        .split([' ', '\t', ',', '\r', '\n'])
        .filter(|t| !t.is_empty())
        .filter_map(|t| t.chars().next())
        .collect();
    if glyphs.is_empty() {
        None
    } else {
        Some(glyphs)
    }
}

/// Parse a `[emitter] color_mode` keyword.
fn parse_color_mode(v: &str) -> Option<ColorMode> {
    match v.trim() {
        "RampOverAge" | "ramp_over_age" | "ramp" => Some(ColorMode::RampOverAge),
        "RandomPick" | "random_pick" | "random" => Some(ColorMode::RandomPick),
        "Fixed" | "fixed" => Some(ColorMode::Fixed),
        _ => None,
    }
}

/// Parse a `[emitter] bold` keyword.
fn parse_bold_mode(v: &str) -> Option<BoldMode> {
    match v.trim() {
        "Off" | "off" => Some(BoldMode::Off),
        "On" | "on" => Some(BoldMode::On),
        "ByProximity" | "by_proximity" | "proximity" => Some(BoldMode::ByProximity),
        "ByAge" | "by_age" | "age" => Some(BoldMode::ByAge),
        _ => None,
    }
}

/// Parse a `[emitter] blend` keyword.
fn parse_emitter_blend(v: &str) -> Option<EmitterBlend> {
    match v.trim() {
        "KeepBg" | "keep_bg" | "keepbg" => Some(EmitterBlend::KeepBg),
        "Opaque" | "opaque" => Some(EmitterBlend::Opaque),
        _ => None,
    }
}

/// Parse a `[emitter] kind` keyword (CASE-INSENSITIVE): `Standard` | `GodRays`.
/// Accepts the snake/spaced/joined variants too (`god_rays`, `godrays`). Defaults
/// to `Standard` when the key is absent (handled by `EmitterSpec::default`).
fn parse_emitter_kind(v: &str) -> Option<EmitterKind> {
    match v.trim().to_ascii_lowercase().as_str() {
        "standard" | "std" | "normal" => Some(EmitterKind::Standard),
        "godrays" | "god_rays" | "god-rays" | "rays" => Some(EmitterKind::GodRays),
        _ => None,
    }
}

// ── `[emitter]` variety & randomisation keys ────────────────────────────────
// Every parser below follows the SAME `keyword k1=v1 k2=v2 …` grammar already
// established by `parse_image_relief`/`recipe = KIND k=v …` elsewhere in this
// file, so an author who already knows one knows all of them.

/// Parse a `.lvl` distribution value: `a..b` (uniform), `m~s` (Gaussian), or
/// `a..b^k` (power law). Used by `lifetime_dist`/`speed_dist`/`size_dist`.
fn parse_distribution(s: &str) -> Option<Distribution> {
    let s = s.trim();
    if let Some(idx) = s.find('~') {
        let mean: f32 = s[..idx].trim().parse().ok()?;
        let std: f32 = s[idx + 1..].trim().parse().ok()?;
        return Some(Distribution::Gaussian { mean, std });
    }
    if let Some(idx) = s.find("..") {
        let lo: f32 = s[..idx].trim().parse().ok()?;
        let rest = s[idx + 2..].trim();
        if let Some(kidx) = rest.find('^') {
            let hi: f32 = rest[..kidx].trim().parse().ok()?;
            let k: f32 = rest[kidx + 1..].trim().parse().ok()?;
            return Some(Distribution::PowerLaw { lo, hi, k });
        }
        let hi: f32 = rest.parse().ok()?;
        return Some(Distribution::Uniform { lo, hi });
    }
    None
}

/// Parse `[emitter] motion = ballistic | curl scale=… strength=… evolve=… |
/// attract X Y Z strength=… swirl=… | lorenz scale=… | flock n=… sep=… align=…
/// coh=… seek_light=… | ripple speed=…` (case-insensitive keyword, `k=v` knobs
/// in any order; an unrecognised knob or keyword is `None` — a line-numbered
/// error at the call site, never a silently-ignored typo).
fn motion_knob_f32(tok: &str) -> Option<(&str, f32)> {
    let (k, v) = tok.split_once('=')?;
    Some((k, v.trim().parse().ok()?))
}

fn parse_motion(s: &str) -> Option<MotionKind> {
    let mut parts = s.split_whitespace();
    let kind = parts.next()?.to_ascii_lowercase();
    let knob_f32 = motion_knob_f32;
    match kind.as_str() {
        "ballistic" => Some(MotionKind::Ballistic),
        "curl" => {
            let (mut scale, mut strength, mut evolve) = (0.6, 0.8, 0.2);
            for tok in parts {
                let (k, f) = knob_f32(tok)?;
                match k {
                    "scale" => scale = f,
                    "strength" => strength = f,
                    "evolve" => evolve = f,
                    _ => return None,
                }
            }
            Some(MotionKind::Curl {
                scale,
                strength,
                evolve,
            })
        }
        "attract" => {
            let x: f32 = parts.next()?.parse().ok()?;
            let y: f32 = parts.next()?.parse().ok()?;
            let z: f32 = parts.next()?.parse().ok()?;
            let (mut strength, mut swirl) = (2.0, 0.0);
            for tok in parts {
                let (k, f) = knob_f32(tok)?;
                match k {
                    "strength" => strength = f,
                    "swirl" => swirl = f,
                    _ => return None,
                }
            }
            Some(MotionKind::Attract {
                target: Vec3::new(x, y, z),
                strength,
                swirl,
            })
        }
        "lorenz" => {
            let mut scale = 0.08;
            for tok in parts {
                let (k, f) = knob_f32(tok)?;
                match k {
                    "scale" => scale = f,
                    _ => return None,
                }
            }
            Some(MotionKind::Lorenz { scale })
        }
        "flock" => {
            let (mut n, mut sep, mut align, mut coh, mut seek_light) = (40.0, 0.4, 0.3, 0.2, 0.0);
            for tok in parts {
                let (k, f) = knob_f32(tok)?;
                match k {
                    "n" => n = f,
                    "sep" => sep = f,
                    "align" => align = f,
                    "coh" => coh = f,
                    "seek_light" => seek_light = f,
                    _ => return None,
                }
            }
            Some(MotionKind::Flock {
                n: n.max(1.0) as u32,
                sep,
                align,
                coh,
                seek_light,
            })
        }
        "ripple" => {
            let mut speed = 1.2;
            for tok in parts {
                let (k, f) = knob_f32(tok)?;
                match k {
                    "speed" => speed = f,
                    _ => return None,
                }
            }
            Some(MotionKind::Ripple { speed })
        }
        _ => None,
    }
}

/// Parse `[emitter] color_jitter = h=<deg> s=<0..1> l=<0..1> dance=<0..1>`
/// (every knob optional, defaulting to `0.0`).
fn parse_color_jitter(s: &str) -> Option<ColorJitter> {
    let mut j = ColorJitter {
        h_deg: 0.0,
        s: 0.0,
        l: 0.0,
        dance: 0.0,
    };
    for tok in s.split_whitespace() {
        let (k, v) = tok.split_once('=')?;
        let f: f32 = v.trim().parse().ok()?;
        match k {
            "h" => j.h_deg = f,
            "s" => j.s = f,
            "l" => j.l = f,
            "dance" => j.dance = f,
            _ => return None,
        }
    }
    Some(j)
}

/// Parse `[emitter] color_ramp = 0xRRGGBB 0xRRGGBB …` — 2 to 8 stops.
fn parse_color_ramp(s: &str) -> Option<Vec<u32>> {
    let stops: Vec<u32> = s
        .split([',', ' ', '\t'])
        .filter(|t| !t.is_empty())
        .map(parse_hex_color)
        .collect::<Option<Vec<_>>>()?;
    if (2..=8).contains(&stops.len()) {
        Some(stops)
    } else {
        None
    }
}

/// Parse `[emitter] glyph_pick = Sequential | Random`.
fn parse_glyph_pick(s: &str) -> Option<GlyphPick> {
    match s.trim().to_ascii_lowercase().as_str() {
        "sequential" | "seq" => Some(GlyphPick::Sequential),
        "random" | "rand" => Some(GlyphPick::Random),
        _ => None,
    }
}

/// Parse `[emitter] glyph_source = family | text "<string>"`.
fn parse_glyph_source(s: &str) -> Option<GlyphSource> {
    let s = s.trim();
    if s.eq_ignore_ascii_case("family") {
        return Some(GlyphSource::Family);
    }
    let rest = s
        .strip_prefix("text")
        .or_else(|| s.strip_prefix("Text"))?
        .trim();
    if rest.len() >= 2 && rest.starts_with('"') && rest.ends_with('"') {
        return Some(GlyphSource::Text(rest[1..rest.len() - 1].to_string()));
    }
    None
}

/// Parse `[emitter] trail = <count> fade=<0..1> dir_glyphs=<on|off>`.
fn parse_trail(s: &str) -> Option<TrailSpec> {
    let mut parts = s.split_whitespace();
    let count: u8 = parts.next()?.trim().parse().ok()?;
    let (mut fade, mut dir_glyphs) = (0.6, true);
    for tok in parts {
        let (k, v) = tok.split_once('=')?;
        match k {
            "fade" => fade = v.trim().parse().ok()?,
            "dir_glyphs" => {
                dir_glyphs = matches!(
                    v.trim().to_ascii_lowercase().as_str(),
                    "on" | "true" | "1" | "yes"
                )
            }
            _ => return None,
        }
    }
    Some(TrailSpec {
        count,
        fade,
        dir_glyphs,
    })
}

/// Parse `[emitter] visibility = always | lit gain=<f32>`.
fn parse_visibility(s: &str) -> Option<DustVisibility> {
    let mut parts = s.split_whitespace();
    match parts.next()?.to_ascii_lowercase().as_str() {
        "always" => Some(DustVisibility::Always),
        "lit" => {
            let mut gain = 3.0;
            for tok in parts {
                let (k, v) = tok.split_once('=')?;
                if k == "gain" {
                    gain = v.trim().parse().ok()?;
                } else {
                    return None;
                }
            }
            Some(DustVisibility::Lit { gain })
        }
        _ => None,
    }
}

/// Whitespace/comma-separated words → owned strings.
fn parse_words(s: &str) -> Vec<String> {
    s.split([' ', '\t', ',', '\r', '\n'])
        .filter(|t| !t.is_empty())
        .map(|t| t.to_string())
        .collect()
}

/// `"ON"/"yes"/"1"/"y"` → true.
/// Parse an `[effects]` INTENSITY knob (`snow`/`matrix`/`glitch`) in `0.0..=1.0`.
///
/// These read as a float, but the sibling `steam` knob is a plain boolean parsed by
/// [`is_truthy`], so authors reasonably write `snow = on` — and a bare
/// `parse::<f32>().unwrap_or(0.0)` turned that into `0.0`, silently killing the very
/// effect the line asks for (`levels/L9-stereopsis.lvl` shipped a dead `snow = on`
/// this way). A keyword therefore falls back to the boolean reading: truthy words
/// mean FULL intensity, everything else stays off.
fn parse_intensity(v: &str) -> f32 {
    let v = v.trim();
    match v.parse::<f32>() {
        Ok(n) if n.is_finite() => n.clamp(0.0, 1.0),
        // NaN/inf and every non-numeric word land here; `is_truthy` promotes the
        // affirmative ones to 1.0 and leaves `off`/`false`/garbage at 0.0.
        _ => f32::from(u8::from(is_truthy(v))),
    }
}

fn is_truthy(v: &str) -> bool {
    matches!(
        v,
        "1" | "on" | "ON" | "yes" | "YES" | "true" | "TRUE" | "True"
    ) || v.starts_with('y')
        || v.starts_with('Y')
}

/// Split `"A->B"` / `"A→B"` into `(from, to)`. Handles ASCII and UTF-8 arrows.
fn split_arrow(spec: &str) -> Option<(String, String)> {
    if let Some(idx) = spec.find("->") {
        let from = spec[..idx].trim().to_string();
        let to = spec[idx + 2..].trim().to_string();
        return Some((from, to));
    }
    if let Some(idx) = spec.find('\u{2192}') {
        let from = spec[..idx].trim().to_string();
        let to = spec[idx + '\u{2192}'.len_utf8()..].trim().to_string();
        return Some((from, to));
    }
    None
}

/// Map a `facing` shortcut to legacy yaw_col (`+y=0, +x=120, -y=240, -x=360`);
/// a bare integer is taken as a numeric override.
fn facing_to_yaw_col(v: &str) -> i32 {
    match v.trim() {
        "+y" => 0,
        "+x" => 120,
        "-y" => 240,
        "-x" => 360,
        other => other.parse().unwrap_or(0),
    }
}

/// Parse a `path` value's `(x,y,z) (x,y,z) ...` waypoint list.
fn parse_waypoints(v: &str) -> Option<Vec<Coord3>> {
    let mut out = Vec::new();
    let mut rest = v;
    while let Some(lp) = rest.find('(') {
        let rp = rest[lp..].find(')')? + lp;
        let wp = parse_coord3(&rest[lp + 1..rp])?;
        out.push(wp);
        rest = &rest[rp + 1..];
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// Expand a waypoint list into a contiguous 6-connected cell path, advancing
/// x then y then z between consecutive waypoints. Port of legacy `expand_path`.
fn expand_path(wp: &[Coord3]) -> Vec<Coord3> {
    if wp.is_empty() {
        return Vec::new();
    }
    let mut path = Vec::new();
    let (mut cx, mut cy, mut cz) = wp[0];
    path.push((cx, cy, cz));
    for &(dx, dy, dz) in &wp[1..] {
        while cx != dx {
            cx += if dx > cx { 1 } else { -1 };
            path.push((cx, cy, cz));
        }
        while cy != dy {
            cy += if dy > cy { 1 } else { -1 };
            path.push((cx, cy, cz));
        }
        while cz != dz {
            cz += if dz > cz { 1 } else { -1 };
            path.push((cx, cy, cz));
        }
    }
    path
}

/// Generate a contiguous 6-connected cell path from a list of spline control
/// points (the `[pipe] spline = ...` value).
///
/// The route is a **centripetal-free, uniform Catmull-Rom** curve: a C1 spline
/// that passes through every control point and bends smoothly between them.
/// We:
///   1. sample each segment densely (small `t`-step) and round each sample to a
///      voxel cell,
///   2. between consecutive *distinct* cells insert the **3D supercover /
///      Bresenham line** so the result is GUARANTEED 6-connected (no diagonal
///      gaps — the raycaster/render needs face-adjacency),
///   3. dedup consecutive duplicates.
///
/// With exactly 2 control points the curve degenerates to a near-straight routed
/// line (the endpoint tangents reduce to the chord direction). The output is a
/// `Vec<Coord3>` shaped identically to [`expand_path`]'s, so everything
/// downstream consumes it unchanged.
///
/// Deterministic: the sample count is a fixed function of the control-point
/// chord lengths, and all math is plain `f64` — no RNG, no platform floats.
fn spline_path(ctrl: &[Coord3]) -> Vec<Coord3> {
    if ctrl.is_empty() {
        return Vec::new();
    }
    if ctrl.len() == 1 {
        return vec![ctrl[0]];
    }

    // Phantom endpoints so every interior segment has 4 control points. We
    // reflect the first/last chord (P0 - (P1 - P0), Pn + (Pn - Pn-1)), which makes
    // the end tangents follow the chord — a 2-point spline becomes a straight line.
    let n = ctrl.len();
    let pt = |i: i32| -> (f64, f64, f64) {
        let c = ctrl[i.clamp(0, n as i32 - 1) as usize];
        (c.0 as f64, c.1 as f64, c.2 as f64)
    };
    let reflect = |a: (f64, f64, f64), b: (f64, f64, f64)| {
        (2.0 * a.0 - b.0, 2.0 * a.1 - b.1, 2.0 * a.2 - b.2)
    };

    // Collect raw rounded samples along the whole curve.
    let mut samples: Vec<Coord3> = Vec::new();
    samples.push(ctrl[0]);

    for seg in 0..(n - 1) {
        let p1 = pt(seg as i32);
        let p2 = pt(seg as i32 + 1);
        let p0 = if seg == 0 {
            reflect(p1, p2)
        } else {
            pt(seg as i32 - 1)
        };
        let p3 = if seg + 1 == n - 1 {
            reflect(p2, p1)
        } else {
            pt(seg as i32 + 2)
        };

        // Steps proportional to the chord length so longer spans stay smooth;
        // a small fixed floor keeps short spans sampled. Deterministic.
        let chord = {
            let dx = p2.0 - p1.0;
            let dy = p2.1 - p1.1;
            let dz = p2.2 - p1.2;
            (dx * dx + dy * dy + dz * dz).sqrt()
        };
        let steps = ((chord * 8.0).ceil() as i32).max(8);

        for s in 1..=steps {
            let t = s as f64 / steps as f64;
            let p = catmull_rom(p0, p1, p2, p3, t);
            samples.push((p.0.round() as i32, p.1.round() as i32, p.2.round() as i32));
        }
    }

    // Stitch the rounded samples into a 6-connected path: between each pair of
    // consecutive distinct cells lay a supercover line, dedup as we go.
    let mut path: Vec<Coord3> = Vec::new();
    for &cell in &samples {
        match path.last() {
            None => path.push(cell),
            Some(&prev) if prev == cell => {}
            Some(&prev) => {
                let seg = supercover_line(prev, cell);
                // seg[0] == prev (already in path); push the rest.
                for &c in &seg[1..] {
                    if path.last() != Some(&c) {
                        path.push(c);
                    }
                }
            }
        }
    }
    path
}

/// Evaluate a uniform Catmull-Rom spline at parameter `t ∈ [0,1]` for the
/// segment between `p1` and `p2` (with neighbours `p0`, `p3`).
fn catmull_rom(
    p0: (f64, f64, f64),
    p1: (f64, f64, f64),
    p2: (f64, f64, f64),
    p3: (f64, f64, f64),
    t: f64,
) -> (f64, f64, f64) {
    let t2 = t * t;
    let t3 = t2 * t;
    // Standard Catmull-Rom basis (tension 0.5).
    let f = |a: f64, b: f64, c: f64, d: f64| -> f64 {
        0.5 * ((2.0 * b)
            + (-a + c) * t
            + (2.0 * a - 5.0 * b + 4.0 * c - d) * t2
            + (-a + 3.0 * b - 3.0 * c + d) * t3)
    };
    (
        f(p0.0, p1.0, p2.0, p3.0),
        f(p0.1, p1.1, p2.1, p3.1),
        f(p0.2, p1.2, p2.2, p3.2),
    )
}

/// 3D supercover (Bresenham-style) line between two integer cells. Returns a
/// **6-connected** sequence: each step changes exactly one of x/y/z by ±1, so
/// there are NO diagonal gaps. Includes both endpoints; `a` first, `b` last.
fn supercover_line(a: Coord3, b: Coord3) -> Vec<Coord3> {
    let (mut x, mut y, mut z) = a;
    let (bx, by, bz) = b;
    let dx = (bx - x).abs();
    let dy = (by - y).abs();
    let dz = (bz - z).abs();
    let sx = if bx > x { 1 } else { -1 };
    let sy = if by > y { 1 } else { -1 };
    let sz = if bz > z { 1 } else { -1 };

    let mut out = vec![(x, y, z)];
    // Drive the dominant axis; step the others when their accumulated error
    // crosses a half-cell. Each loop iteration emits exactly one 6-connected
    // step (one axis at a time) so the path is face-adjacent throughout.
    if dx >= dy && dx >= dz {
        let mut ey = 2 * dy - dx;
        let mut ez = 2 * dz - dx;
        while x != bx {
            if ey >= 0 {
                y += sy;
                ey -= 2 * dx;
                out.push((x, y, z));
            }
            if ez >= 0 {
                z += sz;
                ez -= 2 * dx;
                out.push((x, y, z));
            }
            ey += 2 * dy;
            ez += 2 * dz;
            x += sx;
            out.push((x, y, z));
        }
    } else if dy >= dx && dy >= dz {
        let mut ex = 2 * dx - dy;
        let mut ez = 2 * dz - dy;
        while y != by {
            if ex >= 0 {
                x += sx;
                ex -= 2 * dy;
                out.push((x, y, z));
            }
            if ez >= 0 {
                z += sz;
                ez -= 2 * dy;
                out.push((x, y, z));
            }
            ex += 2 * dx;
            ez += 2 * dz;
            y += sy;
            out.push((x, y, z));
        }
    } else {
        let mut ex = 2 * dx - dz;
        let mut ey = 2 * dy - dz;
        while z != bz {
            if ex >= 0 {
                x += sx;
                ex -= 2 * dz;
                out.push((x, y, z));
            }
            if ey >= 0 {
                y += sy;
                ey -= 2 * dz;
                out.push((x, y, z));
            }
            ex += 2 * dx;
            ey += 2 * dy;
            z += sz;
            out.push((x, y, z));
        }
    }
    out
}

/// Parse a `[material] recipe = KIND k=v k=v …` value into a [`MaterialRecipe`].
///
/// The first token is the recipe NAME; the rest are `key=value` knob overrides.
/// Unknown keys are ignored; a missing key keeps that recipe's default. An unknown
/// recipe name returns `None` (the caller turns that into a `[material] recipe:
/// unknown` parse error) — a level never silently gets a different surface than it
/// asked for.
///
/// ## Knob value syntax
///
/// | kind of knob | written as | example |
/// |---|---|---|
/// | float | a decimal number | `duty=0.55` |
/// | integer | a bare integer | `axis=0`, `seed=7`, `panel=2` |
/// | keyword | one of a named set | `mode=hold`, `mode=release` |
///
/// (There is no boolean knob on any recipe today; were one added it would follow
/// the same keyword form, `on` / `off`.) Integer knobs are read as integers, not
/// as floats, so a large `seed` keeps every bit; out-of-range enum-ish integers
/// (`axis`) are CLAMPED to the set the renderer names, which keeps the clamp
/// idempotent and the save→load round-trip exact.
///
/// ## Defaults are the catalog presets
///
/// Every kind's defaults are the values the matching `preset = NAME` catalog
/// material carries, so `recipe = scanline` and `preset = monitor` produce the
/// same `TextureRecipe` (the palette/animation/lighting still come from the
/// preset — the recipe key authors the PATTERN alone):
///
/// | `recipe =` | default knobs | same pattern as `preset =` |
/// |---|---|---|
/// | `brick` | `aspect=2 mortar_w=0.16 grain=0.6` | the wall_solid built-in |
/// | `circuit` | `scale=1.6 trace_w=0.12` | — |
/// | `grain` | `scale=3 speck=0.3 tile_eps=0.06` | — |
/// | `sky` | `scale=6 density=0.1` | — |
/// | `hologram` | `line_freq=6 scan_w=0.5` | — |
/// | `codewall` | `panel=0` | — |
/// | `scanline` | `freq=9 duty=0.55 warp=0.05 axis=0` | `monitor` |
/// | `truchet` | `seed=0 density=0.5 scale=4` | `conduit` |
/// | `fractal` | `arms=0 octave_warp=0.3 twist=0` | `portal_zoom` |
/// | `substrate` | `scale=3 held=0.55 released=0.3` | `substrate` |
/// | `voice` | `scale=3 mode=hold` | `mnemos_hold` / `lethe_release` |
/// | `facade` | `style=office bay=3 floor=4 lit=0.35 seed=0 ground_band=off sign_row=off` | `facade` |
///
/// `image` is NOT handled here: its `tex_id` is not known until the `[texture]`
/// files are decoded, so the section parser makes the placeholder itself and
/// `build_map` patches the id late.
///
/// PUBLIC because `re_voximport` parses the SAME string from a `.toml` sidecar
/// (`[[material]] recipe = "…"`). One parser, one vocabulary — the two surfaces
/// cannot drift apart.
pub fn parse_recipe(v: &str) -> Option<MaterialRecipe> {
    let mut it = v.split_whitespace();
    let kind = it.next()?;
    // Raw token values, so a knob can be a float, an integer, or a keyword. The
    // float map is derived from the same tokens, keeping the old behaviour exact.
    let mut raw: std::collections::BTreeMap<&str, &str> = std::collections::BTreeMap::new();
    let mut params: std::collections::BTreeMap<&str, f32> = std::collections::BTreeMap::new();
    for tok in it {
        if let Some((k, val)) = tok.split_once('=') {
            raw.insert(k, val);
            if let Ok(f) = val.parse::<f32>() {
                params.insert(k, f);
            }
        }
    }
    let g = |k: &str, d: f32| params.get(k).copied().unwrap_or(d);
    // An INTEGER knob: read from the raw token so no bits are lost through f32.
    let gi = |k: &str, d: i64| raw.get(k).and_then(|s| s.parse::<i64>().ok()).unwrap_or(d);
    Some(match kind {
        "brick" => MaterialRecipe::Brick {
            aspect: g("aspect", 2.0),
            mortar_w: g("mortar_w", 0.16),
            grain: g("grain", 0.6),
        },
        "circuit" => MaterialRecipe::Circuit {
            scale: g("scale", 1.6),
            trace_w: g("trace_w", 0.12),
        },
        "grain" => MaterialRecipe::Grain {
            scale: g("scale", 3.0),
            speck: g("speck", 0.30),
            tile_eps: g("tile_eps", 0.06),
        },
        "sky" => MaterialRecipe::Sky {
            scale: g("scale", 6.0),
            density: g("density", 0.10),
        },
        // The glyph-only transparent HOLOGRAM: fg-coloured scanline glyphs on a void
        // bg, scrolling vertically (the renderer defaults a `shift` animation unless
        // the author sets one). `line_freq` = scan lines per world-cell; `scan_w` =
        // line thickness `[0,1]`. Bind a surface's `[material]` to `recipe = hologram`
        // and give it an `fg` colour to author a holographic panel.
        "hologram" | "holo" => MaterialRecipe::Hologram {
            line_freq: g("line_freq", 6.0),
            scan_w: g("scan_w", 0.5),
        },
        // The strange loop: `recipe = codewall panel=N` binds a surface to render
        // the engine's own annotated source. `panel` selects which file (0 = the
        // code_wall renderer reflecting itself; see re_render::code_wall). Default
        // panel 0 — the safe, always-present self-reference.
        "codewall" | "code" => MaterialRecipe::CodeWall {
            panel: gi("panel", 0).max(0) as u32,
        },
        // FILLED directional bands — CRT raster / brushed metal / grate / weave.
        // `axis` is clamped to the three the renderer names (0 h, 1 v, 2 woven) so
        // the value that comes back out of a save is always one of them.
        "scanline" => MaterialRecipe::Scanline {
            freq: g("freq", 9.0),
            duty: g("duty", 0.55),
            warp: g("warp", 0.05),
            axis: gi("axis", 0).clamp(0, 2) as u8,
        },
        // Edge-matched conduit lattice: runs that continue across cell boundaries.
        "truchet" => MaterialRecipe::Truchet {
            seed: gi("seed", 0).clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32,
            density: g("density", 0.5),
            scale: g("scale", 4.0),
        },
        // Log-polar self-similar portal. `arms` is a small count (0 = pure rings).
        "fractal" => MaterialRecipe::Fractal {
            arms: gi("arms", 0).clamp(0, 255) as u8,
            octave_warp: g("octave_warp", 0.30),
            twist: g("twist", 0.0),
        },
        // The cell alphabet as a static wall field.
        "substrate" => MaterialRecipe::Substrate {
            scale: g("scale", 3.0),
            held: g("held", 0.55),
            released: g("released", 0.30),
        },
        // The same alphabet driven by the level's solve-progress mark. The CHANNEL
        // is a keyword, not a number: `hold` fills, `release` erodes. An unknown
        // keyword is rejected outright rather than silently read as `hold` — the two
        // walls say opposite things, so guessing is worse than erroring.
        "voice" => MaterialRecipe::Voice {
            scale: g("scale", 3.0),
            mode: match raw.get("mode").copied().unwrap_or("hold") {
                "hold" => re_world::VoiceMode::Hold,
                "release" => re_world::VoiceMode::Release,
                _ => return None,
            },
        },
        // The structured bay×floor window-grid recipe.
        // `style` is a keyword, not a number; an unknown style is
        // rejected outright (same discipline as `voice`'s `mode`) rather than
        // silently falling back to `office`. `ground_band`/`sign_row` are `on`/`off`
        // flags, matching the rest of the level format's boolean keywords.
        "facade" => MaterialRecipe::Facade {
            style: match raw.get("style").copied().unwrap_or("office") {
                "office" => re_world::FacadeStyle::Office,
                "residential" => re_world::FacadeStyle::Residential,
                "industrial" => re_world::FacadeStyle::Industrial,
                "server_rack" => re_world::FacadeStyle::ServerRack,
                "lockers" => re_world::FacadeStyle::Lockers,
                _ => return None,
            },
            bay: g("bay", 3.0),
            floor_h: g("floor", 4.0),
            lit: g("lit", 0.35),
            seed: gi("seed", 0).clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32,
            ground_band: match raw.get("ground_band").copied().unwrap_or("off") {
                "on" => true,
                "off" => false,
                _ => return None,
            },
            sign_row: match raw.get("sign_row").copied().unwrap_or("off") {
                "on" => true,
                "off" => false,
                _ => return None,
            },
        },
        _ => return None,
    })
}

/// Parse a `[material] animation = KIND speed=… amplitude=…` value into a
/// [`MaterialAnim`]. `none`/`static` → a static block.
fn parse_anim(v: &str) -> Option<MaterialAnim> {
    let mut it = v.split_whitespace();
    let kind = match it.next()? {
        "none" | "static" => MaterialAnimKind::None,
        "pulse" => MaterialAnimKind::Pulse,
        "shift" => MaterialAnimKind::Shift,
        "flicker" => MaterialAnimKind::Flicker,
        _ => return None,
    };
    let mut speed = 1.0f32;
    let mut amplitude = 0.2f32;
    for tok in it {
        if let Some((k, val)) = tok.split_once('=') {
            if let Ok(f) = val.parse::<f32>() {
                match k {
                    "speed" => speed = f,
                    "amplitude" | "amp" => amplitude = f,
                    _ => {}
                }
            }
        }
    }
    Some(MaterialAnim {
        kind,
        speed,
        amplitude,
    })
}

/// Parse a `[note KIND pos=x,y,z facing=DIR]` header. `None` for an unknown
/// kind (the body is then skipped, like legacy).
///
/// `"banner"` is a KEYWORD ALIAS for [`NoteKind::Stencil`], not a distinct
/// `NoteKind` variant: `re_render`'s `note_style`/`scene_text` (the ONLY place a
/// pixel-font banner is actually rendered — see `re_render/src/banner.rs`) already
/// receives every stencilled placard's raw body verbatim and recognises a leading
/// `--- banner ---` marker there, so no new render dispatch is needed at this layer.
/// Keeping it an alias (rather than a new `NoteKind`) matters beyond style: `NoteKind`
/// is matched EXHAUSTIVELY outside this crate (`re_voximport`'s `.lvl` writer, among
/// others), so a new variant would be a breaking cross-crate change this parser has no
/// business making unilaterally. `[note banner pos=… facing=…]` therefore reads and
/// writes exactly like `[note stencil …]` at this layer; only its BODY differs.
///
/// # `[note stencil]` body syntax (parsed by `re_render::stencil_variants` /
/// `re_render::banner`, NOT by this crate — this parser only captures `note.text`
/// verbatim; everything below is a reference for level authors)
///
/// A stencil body is either a plain ASCII-art block (legacy, unchanged) or one that
/// opens with zero or more directive lines, then zero or more `--- MARKER ---`
/// sections:
///
/// ```text
/// [note stencil pos=3,1,2 facing=+x]
/// bind = 4
/// colors = [:#e0a040 E:#40ff80
/// --- 0 ---
///     [ SWITCH: OFF ]
/// --- 1 ---
///     [ SWITCH: ON  ]
/// --- far ---
///     OFF/ON
/// --- glyph S ---
/// [/note]
/// ```
///
/// - `bind = N` (optional, first section only) — `N` is a puzzle node's 0-based
///   declaration index in `[node ...]`. Each frame the sampler reads that node's live
///   state via `re_render`'s `state_bus` and picks the `--- V ---` section whose marker
///   parses as the integer `V`. No `bind` line ⇒ the stencil is state-blind and always
///   shows its first (or only) section — byte-identical to a pre-existing stencil that
///   never used any of this syntax.
/// - `colors = [glyph:#rrggbb ...]` (optional) — a glyph→colour key. Each entry is
///   `GLYPH:#RRGGBB` (no separator required between entries beyond whitespace); every
///   occurrence of `GLYPH` in every section's body is drawn in an accent colour instead
///   of the stencil's plain ink. NOTE: the renderer has exactly one spare colour channel
///   for this (the `Comment` class, shared by every keyed glyph), so a body with more
///   than one distinct keyed glyph gets a single shared accent colour, not one hue per
///   glyph — a genuine two-tone approximation, not a full-palette key.
/// - `--- V ---` where `V` parses as `u32` — the FULL (near) body for state value `V`.
///   Unmatched live state falls back to the LAST declared section, so an author who
///   only writes `--- 0 ---` / `--- 1 ---` for a boolean switch is covered no matter
///   what the puzzle's raw state integer turns out to be.
/// - `--- far ---` (optional, attaches to the state section that precedes it, or to an
///   implicit single state if there is none) — a compact body shown once the wall face
///   projects to 8 rows of screen space or fewer (but more than the glyph threshold).
/// - `--- glyph X ---` (optional, same attachment rule) — `X` is a single character
///   rendered alone, shown once the face projects to 2 rows or fewer. Tiers degrade
///   full → far → glyph as the face shrinks; a tier that was never authored is skipped
///   (falls through to the next coarser one that was).
/// - `--- banner ---` (only meaningful on a `[note banner]`/aliased header) — marks the
///   body as pixel-font banner directives, not ASCII art. Recognised `key = value`
///   lines inside: `word = TEXT` (rendered upper-cased, ≤16 glyphs, unknown characters
///   render blank), `scale = N` (1-5, font pixels per block, default 2), `frame = on|off`
///   (draws a `+`/`=`/`|` border), `cycle = A, B, C` (parsed but NOT currently sampled
///   over time — only the first `word` renders today; cycling through the list is a
///   known gap, tracked as future editor work).
///
/// A body with none of `bind`/`colors`/any `--- MARKER ---` line is the plain legacy
/// stencil path and renders exactly as before this syntax existed.
fn parse_note_header(spec: &str) -> Option<LevelNote> {
    let mut it = spec.split_whitespace();
    let kind = match it.next()? {
        "researcher" => NoteKind::Researcher,
        "system_log" => NoteKind::SystemLog,
        "model_thought" => NoteKind::ModelThought,
        "stencil" | "banner" => NoteKind::Stencil,
        _ => return None,
    };
    let mut note = LevelNote {
        kind,
        x: 0,
        y: 0,
        z: 0,
        facing: 0,
        text: String::new(),
    };
    for tok in it {
        if let Some((key, val)) = tok.split_once('=') {
            match key {
                "pos" => {
                    if let Some((x, y, z)) = parse_coord3(val) {
                        note.x = x;
                        note.y = y;
                        note.z = z;
                    }
                }
                "facing" => note.facing = facing_to_yaw_col(val),
                _ => {}
            }
        }
    }
    Some(note)
}

#[cfg(test)]
mod tests;
