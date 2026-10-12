//! The **PROP KIT** — `[prop NAME]` sections and their pure stamp generator
//! (set-dressing).
//!
//! A `[prop]` names one of 24 prefab KINDS plus a handful of parameters. At load
//! time it EXPANDS into the structures levels already carry — solid cells (with a
//! shape and a per-cell palette material), `[line]` strokes, holograms, particle
//! emitters, point lights and floor decals — so the renderer needs no changes.
//!
//! # Contract (mirrors `staircase.rs`)
//!
//! * [`crate::props::expand_level`] is PURE: `(level, base map) -> PropExpansion`. It never
//!   mutates a map; [`crate::Level::build_map`] applies the result. Every scatter or
//!   variation comes from [`crate::props::hash3`] over the prop's `seed` — no RNG state, so the
//!   same file always yields the same room.
//! * **Empty only.** A `build` cell is stamped only where the base map (everything
//!   except props) is `Empty` and no earlier prop took it. A skipped cell is
//!   REPORTED ([`crate::props::PropIssue`]), never silent.
//! * **Keep-out.** A solid prop cell may never enter a puzzle cell (node, pipe path,
//!   switch column, pedestal, door, spawn, emit-part target): that is a LOAD ERROR
//!   naming both. The wider §3.7 keep-out rings are `lvlcheck`'s `props` FAIL.
//! * **Round trip.** Prop output is never written back as its own section: the
//!   emitters are `synthesized`, the lines/holos/lights/materials live only in the
//!   built map, and [`crate::props::strip_prop_cells`] removes prop-stamped cells from an editor
//!   capture so a save keeps `[prop]` and never bakes it into `[cells]`.
//!
//! # Coordinates
//!
//! Solid kinds take integer cell coordinates and the §3.3 local frame:
//! `world(u,v,w) = at + u·r − v·f + w·ẑ`, `f` = facing, `r = (f.y, −f.x)` (the
//! viewer's right), so `at` is the front-left-bottom cell as seen from the front.
//! Line points are WORLD coordinates (`[line]` z is world z since set-dressing F4:
//! the walking floor's top is `1.0`). Trace kinds (`furniture_trace`) take a float
//! `at` that is the MIN corner of an axis-aligned footprint; `size = W D` runs along
//! the facing's right/depth axes (so ±x facings swap the world extents).
//!
//! # Power
//!
//! `power =` bindings are parsed, resolved and published as [`crate::props::PowerGate`]s on
//! [`crate::Level::prop_power`]. The map is built in the INITIAL state (sources
//! evaluated statically: input activations from `[inputs]`, pipes authored `On`,
//! budgets from those); flipping it at runtime is the host runtime's job (`re_runtime::prop_power`) (see the gate
//! docs for exactly which structures to rewrite).

use std::collections::{BTreeMap, HashMap, HashSet};

use glam::{IVec3, Vec3};
use re_core::{DiagXY, RampDir, Shape};
use re_world::{CellKind, EmitterSpec, HoloShape, Map, MaterialRecipe, PointLight};

use crate::{parse_recipe, Level, LevelMaterial};

mod kinds;
#[cfg(test)]
mod tests;

/// An `(x, y, z)` integer cell coordinate.
pub type Cell3 = (i32, i32, i32);

/// Prefix of every material the kit synthesizes (never written by the writer,
/// because it never enters `Level::materials`). A level-authored `[material]` of
/// the SAME name wins over the synthesized one, so a pack can be re-tuned per level.
pub const PROP_MATERIAL_PREFIX: &str = "__prop_";

/// Ceiling on `[light]`s the renderer honours (`re_lvlcheck::MAX_POINT_LIGHTS`).
pub const PROP_MAX_LIGHTS: usize = 4;

/// The renderer CATALOG material names a `mat.SLOT = NAME` / `face =` value may
/// select (`re_render::ascii_shader::catalog_specs`). Mirrored here because
/// `re_level` is engine-agnostic and cannot look the catalog up; an unknown name is
/// a parse error rather than a silent built-in.
pub const CATALOG_PRESETS: [&str; 45] = [
    "metal",
    "gold",
    "glass",
    "stone",
    "rust",
    "ice",
    "wood",
    "fabric",
    "water",
    "lava",
    "neon",
    "monitor",
    "flesh",
    "void",
    "holo_amber",
    "circuit_red",
    "haze",
    "flow",
    "spark",
    "vitrine",
    "screen",
    "veil",
    "moss",
    "ember_field",
    "static_noise",
    "lattice",
    "beat_grid",
    "auto_tile",
    "boil",
    "portal_zoom",
    "portal_spiral",
    "facet",
    "pane",
    "relief_vol",
    "frost",
    "wet_stone",
    "redacted",
    "watcher",
    "worn_floor",
    "decay",
    "node_breathing",
    "substrate",
    "mnemos_hold",
    "lethe_release",
    "conduit",
];

// ─────────────────────────────── authored form ───────────────────────────────

/// One authored `[prop NAME]` section, kept VERBATIM (key order, values, source
/// lines) so the writer re-emits it byte-for-byte and every error can name a line.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PropDef {
    /// The section name (`[a-z0-9_]+`, unique per level).
    pub name: String,
    /// 1-based line of the `[prop NAME]` header.
    pub line: usize,
    /// Every `key = value` in authored order.
    pub keys: Vec<PropKey>,
}

/// One `key = value` line of a `[prop]` section.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PropKey {
    /// The key name (validated against [`is_known_key`]).
    pub key: String,
    /// The raw, unparsed value text.
    pub value: String,
    /// 1-based source line.
    pub line: usize,
}

/// A prop parse / expansion error: the source line and a message that names the
/// prop and the key (and, for a keep-out refusal, the puzzle element).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PropError {
    /// 1-based source line the error is reported against.
    pub line: usize,
    /// Human-readable message naming the prop, the key, and (for a keep-out
    /// refusal) the puzzle element it collided with.
    pub msg: String,
}

impl PropError {
    fn new(line: usize, msg: impl Into<String>) -> Self {
        PropError {
            line,
            msg: msg.into(),
        }
    }
}

/// `gallery` `posts =`: which support posts stand under the deck edge.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum GalleryPosts {
    /// Every `post_every` cells along each open edge, plus both ends of each edge.
    #[default]
    Auto,
    /// The `post_every` cadence only — no extra post at the end of an edge (an end
    /// post landing on furniture or in a doorway is the usual reason).
    NoEnds,
    /// No posts: a cantilevered or wall-hung deck.
    None,
    /// Exactly these deck cells, in the deck's local `(U, V)` (`U` along the width
    /// to the right, `V` back along the depth), each on an open edge or not.
    At(Vec<(i32, i32)>),
}

impl GalleryPosts {
    fn parse(v: &str) -> Option<Self> {
        match v.trim() {
            "auto" => return Some(Self::Auto),
            "no_ends" | "noends" => return Some(Self::NoEnds),
            "none" => return Some(Self::None),
            _ => {}
        }
        let mut out = Vec::new();
        for pair in v.split(';').map(str::trim).filter(|p| !p.is_empty()) {
            let t: Vec<&str> = pair
                .split([' ', '\t', ','])
                .filter(|s| !s.is_empty())
                .collect();
            let [u, w] = t.as_slice() else { return None };
            out.push((u.parse().ok()?, w.parse().ok()?));
        }
        (!out.is_empty()).then_some(Self::At(out))
    }
}

/// Keys every kind accepts (§3.2), plus `mat.SLOT`.
const GENERIC_KEYS: [&str; 18] = [
    "kind", "at", "facing", "size", "to", "follow", "lift", "extend", "mode", "style", "power",
    "invert", "seed", "light", "emitter", "note", "shape", "count",
];

/// Every key a `[prop]` section may carry at all (generic + kind-specific). Used
/// by the parser's unknown-key diagnostic; applicability per kind is checked when
/// the section is interpreted.
pub fn is_known_key(k: &str) -> bool {
    GENERIC_KEYS.contains(&k)
        || k.strip_prefix("mat.")
            .is_some_and(|s| MatSlot::parse(s).is_some())
        || PropKind::ALL
            .iter()
            .any(|kind| kind.extra_keys().contains(&k))
}

// ──────────────────────────────── enums ──────────────────────────────────────

/// The 24 prefab kinds of §3.8.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PropKind {
    /// Server rack — the physical body of a node.
    Rack,
    /// A row of racks, giving the aisle and the perspective.
    RackRow,
    /// Operator desk, marking where an operator stood (switches sit beside it).
    Console,
    /// Archive stack or reagent shelving, giving aisles.
    Shelving,
    /// Personnel lockers — one per former technician.
    LockerBank,
    /// Coolant or water tank — the machine's thermal body.
    Tank,
    /// The power-budget meter built into the world (`power n/N` readout).
    BudgetBank,
    /// A pillar with base and capital — rhythm and the vanishing point.
    Column,
    /// Office divider or screen wall, with an optional door gap.
    Partition,
    /// Fallen panels, broken cabinets, dead monitors.
    Debris,
    /// Surveillance camera, lit while the level is unsolved.
    Cctv,
    /// Monitor bank, acoustic panel or vent grille on a wall.
    WallPanel,
    /// A post or board that `[note]`s hang on.
    NoticeBoard,
    /// A wrapper over `re_level::staircase` — straight, switchback or spiral.
    Stair,
    /// Mezzanine, catwalk or balcony deck — a vantage and vertical layering.
    Gallery,
    /// Overhead wiring made visible, following a pipe or a straight run.
    CableTray,
    /// Cooling or water pipe, with optional risers and a vent.
    Conduit,
    /// A broken ceiling that reveals the room's height.
    HangingPanels,
    /// The pipe graph drawn on the floor, following a pipe's route.
    CableRun,
    /// A hanging cable bundle from the ceiling to a point.
    CableDrop,
    /// A lamp hanging from a cord, binding to or declaring a `[light]`.
    LampPendant,
    /// Low, non-colliding furniture outlines (table, chair, cot, etc.).
    FurnitureTrace,
    /// A node's thought projected into the room as a hologram.
    HoloDisplay,
    /// Vines and moss taking the machine back.
    Overgrowth,
}

impl PropKind {
    /// Every kind, in §3.8 order.
    pub const ALL: [PropKind; 24] = [
        PropKind::Rack,
        PropKind::RackRow,
        PropKind::Console,
        PropKind::Shelving,
        PropKind::LockerBank,
        PropKind::Tank,
        PropKind::BudgetBank,
        PropKind::Column,
        PropKind::Partition,
        PropKind::Debris,
        PropKind::Cctv,
        PropKind::WallPanel,
        PropKind::NoticeBoard,
        PropKind::Stair,
        PropKind::Gallery,
        PropKind::CableTray,
        PropKind::Conduit,
        PropKind::HangingPanels,
        PropKind::CableRun,
        PropKind::CableDrop,
        PropKind::LampPendant,
        PropKind::FurnitureTrace,
        PropKind::HoloDisplay,
        PropKind::Overgrowth,
    ];

    /// The `kind =` token.
    pub fn as_str(self) -> &'static str {
        match self {
            PropKind::Rack => "rack",
            PropKind::RackRow => "rack_row",
            PropKind::Console => "console",
            PropKind::Shelving => "shelving",
            PropKind::LockerBank => "locker_bank",
            PropKind::Tank => "tank",
            PropKind::BudgetBank => "budget_bank",
            PropKind::Column => "column",
            PropKind::Partition => "partition",
            PropKind::Debris => "debris",
            PropKind::Cctv => "cctv",
            PropKind::WallPanel => "wall_panel",
            PropKind::NoticeBoard => "notice_board",
            PropKind::Stair => "stair",
            PropKind::Gallery => "gallery",
            PropKind::CableTray => "cable_tray",
            PropKind::Conduit => "conduit",
            PropKind::HangingPanels => "hanging_panels",
            PropKind::CableRun => "cable_run",
            PropKind::CableDrop => "cable_drop",
            PropKind::LampPendant => "lamp_pendant",
            PropKind::FurnitureTrace => "furniture_trace",
            PropKind::HoloDisplay => "holo_display",
            PropKind::Overgrowth => "overgrowth",
        }
    }

    /// Parse a `kind =` token (exact, lower-case).
    pub fn parse(s: &str) -> Option<PropKind> {
        PropKind::ALL.iter().copied().find(|k| k.as_str() == s)
    }

    /// SOLID kinds stamp cells and take integer coordinates. (`holo_display` stamps
    /// hologram cells too, but they are walk-through, so it is a trace kind.)
    pub fn is_solid(self) -> bool {
        !matches!(
            self,
            PropKind::CableRun
                | PropKind::CableDrop
                | PropKind::LampPendant
                | PropKind::FurnitureTrace
                | PropKind::HoloDisplay
                | PropKind::Overgrowth
        )
    }

    /// Kinds whose front matters (`facing` is meaningful). Others ignore it.
    fn is_oriented(self) -> bool {
        !matches!(
            self,
            PropKind::Cctv
                | PropKind::CableTray
                | PropKind::Conduit
                | PropKind::HangingPanels
                | PropKind::CableRun
                | PropKind::CableDrop
                | PropKind::LampPendant
        )
    }

    /// Kinds that are drawn along a route (`to` / `follow`) instead of a box.
    fn is_linear(self) -> bool {
        matches!(
            self,
            PropKind::CableTray | PropKind::Conduit | PropKind::CableRun | PropKind::CableDrop
        )
    }

    /// Kind-specific keys (beyond [`GENERIC_KEYS`]).
    fn extra_keys(self) -> &'static [&'static str] {
        match self {
            PropKind::RackRow => &["gap", "dead", "tray", "power_each"],
            PropKind::Console => &["monitor", "chair"],
            PropKind::Shelving => &["hatch", "faces"],
            PropKind::LockerBank => &["open"],
            PropKind::Column => &["ribs"],
            PropKind::Partition => &["door_gap"],
            PropKind::Cctv => &["look"],
            PropKind::WallPanel => &["screen", "face"],
            PropKind::Stair => &["form", "rise", "radius"],
            PropKind::Gallery => &["post_every", "posts", "stair"],
            PropKind::Conduit => &["riser", "end"],
            PropKind::CableDrop => &["strands"],
            PropKind::CableRun => &["bundle"],
            PropKind::HoloDisplay => &["base"],
            PropKind::Overgrowth => &["floor", "height"],
            PropKind::FurnitureTrace => &["solid"],
            _ => &[],
        }
    }

    /// Generic keys that mean nothing for this kind (reported as unknown).
    fn rejects_generic(self, k: &str) -> bool {
        match k {
            "to" => !self.is_linear(),
            "follow" | "lift" => !matches!(
                self,
                PropKind::CableTray | PropKind::CableRun | PropKind::Conduit
            ),
            "extend" => !matches!(self, PropKind::CableRun | PropKind::CableTray),
            "light" => self != PropKind::LampPendant,
            "emitter" => !matches!(
                self,
                PropKind::Tank
                    | PropKind::Conduit
                    | PropKind::WallPanel
                    | PropKind::Overgrowth
                    | PropKind::HangingPanels
            ),
            "note" => !matches!(self, PropKind::NoticeBoard | PropKind::LockerBank),
            "shape" => !matches!(
                self,
                PropKind::FurnitureTrace
                    | PropKind::HoloDisplay
                    | PropKind::LampPendant
                    | PropKind::Stair
            ),
            "count" => self != PropKind::Debris,
            "facing" => !self.is_oriented() && self != PropKind::Stair,
            _ => false,
        }
    }

    /// `(default, min, max)` of `size = W D H`. `None` for kinds without a box.
    fn size_rule(self) -> Option<([f32; 3], [f32; 3], [f32; 3])> {
        Some(match self {
            PropKind::Rack => ([1.0, 1.0, 2.0], [1.0, 1.0, 1.0], [1.0, 2.0, 4.0]),
            PropKind::RackRow => ([3.0, 1.0, 2.0], [2.0, 1.0, 1.0], [12.0, 1.0, 4.0]),
            PropKind::Console => ([2.0, 2.0, 1.0], [1.0, 1.0, 1.0], [8.0, 2.0, 1.0]),
            PropKind::Shelving => ([4.0, 1.0, 3.0], [1.0, 1.0, 2.0], [16.0, 2.0, 4.0]),
            PropKind::LockerBank => ([3.0, 1.0, 2.0], [1.0, 1.0, 2.0], [12.0, 1.0, 2.0]),
            PropKind::Tank => ([2.0, 2.0, 2.0], [1.0, 1.0, 1.0], [6.0, 6.0, 6.0]),
            PropKind::BudgetBank => ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [8.0, 1.0, 1.0]),
            PropKind::Column => ([1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [4.0, 4.0, 16.0]),
            PropKind::Partition => ([2.0, 1.0, 2.0], [1.0, 1.0, 1.0], [16.0, 1.0, 6.0]),
            PropKind::Debris => ([3.0, 3.0, 1.0], [1.0, 1.0, 1.0], [16.0, 16.0, 1.0]),
            PropKind::Cctv => ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]),
            PropKind::WallPanel => ([2.0, 1.0, 2.0], [1.0, 1.0, 1.0], [16.0, 1.0, 8.0]),
            PropKind::NoticeBoard => ([1.0, 1.0, 2.0], [1.0, 1.0, 2.0], [2.0, 1.0, 2.0]),
            PropKind::Gallery => ([3.0, 2.0, 1.0], [1.0, 1.0, 1.0], [32.0, 32.0, 1.0]),
            PropKind::HangingPanels => ([4.0, 3.0, 1.0], [1.0, 1.0, 1.0], [32.0, 32.0, 1.0]),
            PropKind::HoloDisplay => ([2.0, 2.0, 2.0], [1.0, 1.0, 1.0], [16.0, 16.0, 16.0]),
            PropKind::Overgrowth => ([2.0, 1.0, 3.0], [0.2, 0.0, 0.3], [16.0, 16.0, 8.0]),
            PropKind::FurnitureTrace => ([1.0, 1.0, 0.0], [0.05, 0.0, 0.0], [16.0, 16.0, 8.0]),
            _ => return None,
        })
    }
}

/// `facing =` — the prop's FRONT normal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Facing {
    /// `+x`.
    PosX,
    /// `-x`.
    NegX,
    /// `+y`.
    PosY,
    /// `-y`.
    NegY,
}

impl Facing {
    /// Every direction, in `+x -x +y -y` order.
    pub const ALL: [Facing; 4] = [Facing::PosX, Facing::NegX, Facing::PosY, Facing::NegY];

    /// Parse a `facing =` value (`+x`, `-x`, `+y` or `-y`). `None` for anything else.
    pub fn parse(s: &str) -> Option<Facing> {
        Some(match s {
            "+x" => Facing::PosX,
            "-x" => Facing::NegX,
            "+y" => Facing::PosY,
            "-y" => Facing::NegY,
            _ => return None,
        })
    }

    /// The authored text this direction round-trips to.
    pub fn as_str(self) -> &'static str {
        match self {
            Facing::PosX => "+x",
            Facing::NegX => "-x",
            Facing::PosY => "+y",
            Facing::NegY => "-y",
        }
    }

    /// The unit vector `f`.
    pub fn vec(self) -> (i32, i32) {
        match self {
            Facing::PosX => (1, 0),
            Facing::NegX => (-1, 0),
            Facing::PosY => (0, 1),
            Facing::NegY => (0, -1),
        }
    }

    /// The viewer's right `r = (f.y, −f.x)`.
    pub fn right(self) -> (i32, i32) {
        let (fx, fy) = self.vec();
        (fy, -fx)
    }

    /// The `[note] facing` yaw column for this direction (`+y` 0, `+x` 120, …).
    pub fn yaw_col(self) -> i32 {
        match self {
            Facing::PosY => 0,
            Facing::PosX => 120,
            Facing::NegY => 240,
            Facing::NegX => 360,
        }
    }
}

/// The `RampDir` that rises toward the horizontal unit vector `d`.
pub fn ramp_dir_of(d: (i32, i32)) -> RampDir {
    match d {
        (1, _) => RampDir::PosX,
        (-1, _) => RampDir::NegX,
        (_, 1) => RampDir::PosY,
        _ => RampDir::NegY,
    }
}

/// The `WedgeXY` whose SOLID half sits toward the corner vector `c` (§3.3: the
/// sign pair maps to compass with y growing south).
pub fn wedge_toward(c: (i32, i32)) -> Shape {
    Shape::WedgeXY(match (c.0 >= 0, c.1 >= 0) {
        (true, false) => DiagXY::NE,
        (false, false) => DiagXY::NW,
        (true, true) => DiagXY::SE,
        (false, true) => DiagXY::SW,
    })
}

/// `mode =`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PropMode {
    /// Stamp cells into Empty (the default).
    #[default]
    Build,
    /// Re-material EXISTING non-Empty cells in the footprint.
    Skin,
    /// Add only rails / posts / lines to existing geometry.
    Dress,
    /// `wall_panel` only: one layer built in front of the wall.
    Bank,
}

impl PropMode {
    fn parse(s: &str) -> Option<PropMode> {
        Some(match s {
            "build" => PropMode::Build,
            "skin" => PropMode::Skin,
            "dress" => PropMode::Dress,
            "bank" => PropMode::Bank,
            _ => return None,
        })
    }
}

/// `style =` — the material pack (§3.5).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum PropStyle {
    /// The P1 lab pack.
    #[default]
    Lab,
    /// The P2 firing-range pack.
    Range,
    /// The P3 flooded-vault pack.
    Vault,
    /// The P4 archive pack.
    Archive,
    /// The decayed/dead variant (bit-rot).
    Dead,
    /// The empty/inert variant (marks "outside the model").
    Void,
}

impl PropStyle {
    /// Every style, in `[Self::as_str]` order.
    pub const ALL: [PropStyle; 6] = [
        PropStyle::Lab,
        PropStyle::Range,
        PropStyle::Vault,
        PropStyle::Archive,
        PropStyle::Dead,
        PropStyle::Void,
    ];

    /// Parse a `style =` value. `None` for anything not in [`PropStyle::ALL`].
    pub fn parse(s: &str) -> Option<PropStyle> {
        PropStyle::ALL.iter().copied().find(|p| p.as_str() == s)
    }

    /// The authored text this style round-trips to.
    pub fn as_str(self) -> &'static str {
        match self {
            PropStyle::Lab => "lab",
            PropStyle::Range => "range",
            PropStyle::Vault => "vault",
            PropStyle::Archive => "archive",
            PropStyle::Dead => "dead",
            PropStyle::Void => "void",
        }
    }

    /// The movement → pack convention the twelve campaign levels follow (`P1`
    /// lab, `P2` range, `P3` vault, `P4` archive; see `levels/campaign.toml`).
    /// `re_level` no longer derives a level's `prop_style` from this at load
    /// time — every campaign `.lvl` authors its `[meta] prop_style` explicitly
    /// — but the mapping is kept, and public, so a downstream test can assert
    /// the two never drift apart.
    pub fn for_movement(m: &str) -> Option<PropStyle> {
        Some(match m {
            "P1" => PropStyle::Lab,
            "P2" => PropStyle::Range,
            "P3" => PropStyle::Vault,
            "P4" => PropStyle::Archive,
            _ => return None,
        })
    }
}

/// One of the seven pack slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MatSlot {
    /// The main volume material.
    Body,
    /// The front-facing surface material.
    Face,
    /// Frame/trim/edge material.
    Trim,
    /// The powered-on (lit) variant.
    Lit,
    /// The powered-off (dead) variant.
    Dead,
    /// Transparent/glass surfaces.
    Glass,
    /// Ink/stroke colour for lines.
    Ink,
}

impl MatSlot {
    /// Every slot, in `mat.SLOT` authoring order.
    pub const ALL: [MatSlot; 7] = [
        MatSlot::Body,
        MatSlot::Face,
        MatSlot::Trim,
        MatSlot::Lit,
        MatSlot::Dead,
        MatSlot::Glass,
        MatSlot::Ink,
    ];

    /// Parse a `mat.SLOT` suffix. `None` for anything not in [`MatSlot::ALL`].
    pub fn parse(s: &str) -> Option<MatSlot> {
        MatSlot::ALL.iter().copied().find(|m| m.as_str() == s)
    }

    /// The authored `mat.SLOT` suffix this slot round-trips to.
    pub fn as_str(self) -> &'static str {
        match self {
            MatSlot::Body => "body",
            MatSlot::Face => "face",
            MatSlot::Trim => "trim",
            MatSlot::Lit => "lit",
            MatSlot::Dead => "dead",
            MatSlot::Glass => "glass",
            MatSlot::Ink => "ink",
        }
    }
}

/// `light =` (lamp_pendant).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LightMode {
    /// No lamp declared (the default).
    #[default]
    No,
    /// Declare a new `[light]`, counting toward the level's light cap.
    Yes,
    /// Bind to the existing `[light]` at this prop's position; the cap is unchanged.
    Existing,
}

/// An authored `power =` source, names unresolved.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum PowerSpec {
    /// No power source — never reacts.
    #[default]
    None,
    /// Bound to a node's activation, by node name.
    Node(String),
    /// Bound to a pipe's on/off state, by `(from, to)` node names.
    Pipe(String, String),
    /// Bound to the level's door being ready.
    Door,
    /// Bound to the power-budget draw hitting this threshold.
    Budget(u32),
    /// Bound to a sequence-lock progress step.
    Seq(u32),
}

impl PowerSpec {
    fn parse(s: &str) -> Option<PowerSpec> {
        let s = s.trim();
        if s == "none" {
            return Some(PowerSpec::None);
        }
        if s == "door" {
            return Some(PowerSpec::Door);
        }
        let (head, tail) = s.split_once(':')?;
        let tail = tail.trim();
        match head.trim() {
            "node" if !tail.is_empty() => Some(PowerSpec::Node(tail.to_string())),
            "pipe" => {
                let (a, b) = tail.split_once("->")?;
                let (a, b) = (a.trim(), b.trim());
                (!a.is_empty() && !b.is_empty()).then(|| PowerSpec::Pipe(a.into(), b.into()))
            }
            "budget" => tail
                .parse::<u32>()
                .ok()
                .filter(|&k| k >= 1)
                .map(PowerSpec::Budget),
            "seq" => tail
                .parse::<u32>()
                .ok()
                .filter(|&k| k >= 1)
                .map(PowerSpec::Seq),
            _ => None,
        }
    }

    fn is_none(&self) -> bool {
        *self == PowerSpec::None
    }
}

/// A RESOLVED power source (§3.6), the value `re_runtime::prop_power` evaluates
/// each frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PowerSource {
    /// Node index into `Level::nodes`: its activation.
    Node(usize),
    /// Pipe index into `Level::pipes`: that pipe is ON.
    Pipe(usize),
    /// The level is won.
    Door,
    /// At least `K` pipes are ON.
    Budget(u32),
    /// `[sequence]` step `I` (1-based) has been observed.
    Seq(u32),
}

/// What a prop does with its power source `s` — everything `re_runtime::prop_power`
/// needs to flip the prop between its unlit and lit states. Positions are world
/// cells; line indices address `Map::overlay().lines` (and
/// `Map::set_line_palette_ids`); palette ids are the 1-based colour ids the
/// renderer reads from `side_data`.
#[derive(Clone, Debug)]
pub struct PowerGate {
    /// The owning `[prop NAME]`.
    pub prop: String,
    /// The resolved source this gate reacts to.
    pub source: PowerSource,
    /// `invert = yes`: lit when the source reads 0.
    pub invert: bool,
    /// The state the built map shows (after `invert`).
    pub initial_lit: bool,
    /// The colour-law hue the prop lights in (§3.6).
    pub lit_rgb: u32,
    /// R1 lit cells: `(cell, palette id)` — an Empty slot that holds a
    /// `WallSolid` of that palette id while lit and is Empty while unlit (the
    /// `emit_parts` contract with an absolute anchor). Filled at load iff
    /// `initial_lit`.
    pub parts: Vec<(Cell3, u16)>,
    /// R2 lit lines: `(map line index, unlit palette id, lit palette id)` — write
    /// the matching id with `Map::set_line_palette_ids`. The built map carries the
    /// `initial_lit` one.
    pub lines: Vec<(usize, u16, u16)>,
    /// R2 holograms shown only while lit (re-stamp with `re_world::stamp_holos`;
    /// clear their cells back to Empty when unlit). Stamped at load iff
    /// `initial_lit`.
    pub holos: Vec<re_world::HoloObject>,
    /// R2 emitter swaps: `(index into Level::emitters, the spec for the OTHER
    /// state)` — the level carries the initial-state spec at that index.
    pub emitters: Vec<(usize, EmitterSpec)>,
    /// R3 lamps: `(index into the built map's atmosphere lights, the LIT light)`.
    /// An unlit lamp sits in its slot at `intensity = 0` (so it still counts
    /// against the 4-light cap it will need when lit).
    pub lights: Vec<(usize, PointLight)>,
}

// ───────────────────────────── parsed spec ───────────────────────────────────

/// `stair form =`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum StairForm {
    /// One straight run the height of `rise` (default `storey_height`).
    #[default]
    Straight,
    /// Two straight runs joined by a mid landing, fitting a storey into half
    /// the length.
    Switchback,
    /// A spiral run; only valid when `rise = 8 * radius * turns`.
    Spiral,
}

/// `wall_panel screen =`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Screen {
    /// Every cell lit.
    Live,
    /// Every cell dark.
    Dead,
    /// A seeded mix of lit/dead/static per cell.
    Mixed,
    /// No screen material — a plain panel.
    #[default]
    None,
}

/// `conduit riser =`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Riser {
    /// No vertical run (the default).
    #[default]
    None,
    /// A vertical run down to the floor at the start.
    Start,
    /// A vertical run up to the ceiling at the end.
    End,
    /// A vertical run at both ends.
    Both,
}

/// A `[prop]` interpreted: every key validated, defaults applied.
#[derive(Clone, Debug)]
pub struct PropSpec {
    /// The section name (`[a-z0-9_]+`, unique per level).
    pub name: String,
    /// 1-based line of the `[prop NAME]` header.
    pub line: usize,
    /// Which of the 24 prefabs this is (`kind =`).
    pub kind: PropKind,
    /// `at =`: the anchor cell (the front-left-bottom cell as seen from the
    /// front, or the float min corner for a trace kind).
    pub at: Vec3,
    /// `facing =`: the prop's front normal.
    pub facing: Facing,
    /// `size = W D H`, defaulted/clamped per kind (`H == 0` ⇒ "to the ceiling").
    pub size: [f32; 3],
    /// Whether `size` was authored.
    pub size_set: bool,
    /// `to =`: the far endpoint of a linear kind (`cable_tray`/`conduit`/`cable_run`/`cable_drop`).
    pub to: Option<Vec3>,
    /// `follow = pipe:A->B`: route the prop along that pipe's expanded path
    /// instead of a straight `at`→`to` run.
    pub follow: Option<(String, String)>,
    /// `lift =`: cells above the followed pipe (only meaningful with `follow`).
    pub lift: i32,
    /// `extend =`: continue a `cable_run` past the node to this point.
    pub extend: Option<Vec3>,
    /// `mode =`: build / skin / dress / bank.
    pub mode: PropMode,
    /// `style =`: the material pack.
    pub style: PropStyle,
    /// `holo_display` only: a numeric hologram style tag.
    pub holo_style: u16,
    /// `power =`: the authored power source, names unresolved.
    pub power: PowerSpec,
    /// `invert = yes`: the prop reads lit when its power source reads 0.
    pub invert: bool,
    /// `seed =`, defaulting to `fnv1a(level name + NAME)` when not authored.
    pub seed: u32,
    /// `mat.SLOT = NAME` overrides, as `(slot, material name, source line)`.
    pub mats: Vec<(MatSlot, String, usize)>,
    /// `light =`: no lamp / declare a new one / bind to an existing one.
    pub light: LightMode,
    /// `emitter = yes`: this kind's optional particle emitter is present.
    pub emitter: bool,
    /// `note = NAME`: bind that note's position to this prop (locker_bank, notice_board).
    pub note: Option<String>,
    /// `shape =`: kind-specific shape selector (furniture_trace, holo_display).
    pub shape: Option<String>,
    /// `count =`: how many pieces to scatter (`debris` only).
    pub count: Option<u32>,
    // kind-specific
    /// `rack_row gap = K`: every K-th slot is left Empty.
    pub gap: Option<u32>,
    /// `rack_row dead =`: seeded fraction of racks rendered dead (`0.0..=1.0`).
    pub dead: f32,
    /// `rack_row tray = yes`: add a `cable_tray` along the top if it fits under the ceiling.
    pub tray: bool,
    /// `rack_row power_each =`: per-rack power sources, one-to-one with the row's slots.
    pub power_each: Vec<PowerSpec>,
    /// `console monitor =`: adds a CRT above the desk (default `yes`).
    pub monitor: bool,
    /// `console chair = yes`: adds a `furniture_trace chair` behind the desk.
    pub chair: bool,
    /// `shelving hatch =`: which face-material stroke direction (`/ | \`) to use.
    pub hatch: Option<char>,
    /// `shelving faces = both`: put drive-LED lines on both sides.
    pub faces_both: bool,
    /// `locker_bank open = i`: leave unit `i` Empty (with a coat/hook trace inside).
    pub open: Option<i32>,
    /// `column ribs = yes`: add 4 vertical flutes per visible face.
    pub ribs: bool,
    /// `partition door_gap = u0..u1`: the Empty gap cells in the divider.
    pub door_gap: Option<(i32, i32)>,
    /// `cctv look =`: the world point the camera's lens stroke points at (default: spawn).
    pub look: Option<Vec3>,
    /// `wall_panel screen =`: the per-cell lit/dead/static mix.
    pub screen: Screen,
    /// `wall_panel face =`: which material covers the panel.
    pub face: Option<String>,
    /// `stair form =`: straight / switchback / spiral.
    pub form: StairForm,
    /// `stair rise =`: the storey height climbed (defaults to `storey_height`).
    pub rise: Option<u32>,
    /// `stair radius =`: turn radius (spiral form only).
    pub radius: i32,
    /// `gallery post_every =`: cadence of support posts along an open edge (default 4).
    pub post_every: i32,
    /// `gallery` only: which support posts to stand (`posts =`).
    pub posts: GalleryPosts,
    /// `gallery stair = X Y Z FACING RISE`: emit an attached `stair` prop whose
    /// landing must touch the deck.
    pub stair: Option<(IVec3, Facing, u32)>,
    /// `conduit riser =`: whether a vertical run is added at the start/end/both.
    pub riser: Riser,
    /// `conduit end = vent`: the last cell becomes a grille with an emitter.
    pub end_vent: bool,
    /// `cable_drop strands =`: how many hanging strands (1–3).
    pub strands: u32,
    /// `cable_run bundle =`: how many parallel strands, 0.12 apart (1–3).
    pub bundle: u32,
    /// `holo_display base = yes`: add a 1-cell plinth if it fits outside the keep-out ring.
    pub base: bool,
    /// `furniture_trace solid = yes`: the piece is a LOW SOLID BLOCK (a colliding
    /// z-band cell per footprint cell, [`solid_furniture_band`]) instead of strokes
    /// over a floor decal. Only the [`SOLID_FURNITURE_SHAPES`].
    pub solid: bool,
    /// `overgrowth floor = yes`: add a `moss` `[zone]` under it.
    pub floor: bool,
    /// `overgrowth height =`: how tall the climbing vines reach.
    pub height: Option<f32>,
    /// Source line of every authored key (for located errors).
    key_lines: HashMap<String, usize>,
    /// Non-fatal parse findings (size clamps).
    warnings: Vec<String>,
}

impl PropSpec {
    /// The line an authored key sat on, or the header line.
    pub fn line_of(&self, key: &str) -> usize {
        self.key_lines.get(key).copied().unwrap_or(self.line)
    }

    /// The prop's anchor as an integer cell.
    pub fn at_cell(&self) -> IVec3 {
        IVec3::new(
            self.at.x.floor() as i32,
            self.at.y.floor() as i32,
            self.at.z.floor() as i32,
        )
    }
}

/// FNV-1a over bytes — the default-seed hash (`seed = fnv1a(level name + NAME)`).
pub fn fnv1a(bytes: &[u8]) -> u32 {
    let mut h: u32 = 0x811C_9DC5;
    for b in bytes {
        h ^= u32::from(*b);
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

/// Stateless per-index hash (`hash(seed, index, param_id)`).
pub fn hash3(seed: u32, a: u32, b: u32) -> u32 {
    let mut x = u64::from(seed)
        ^ u64::from(a).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ u64::from(b).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    (x & 0xFFFF_FFFF) as u32
}

/// [`hash3`] as a unit float in `[0, 1)`.
pub fn unit3(seed: u32, a: u32, b: u32) -> f32 {
    (hash3(seed, a, b) >> 8) as f32 / (1u32 << 24) as f32
}

fn parse_f3(v: &str) -> Option<Vec3> {
    let n: Vec<f32> = v
        .split([',', ' ', '\t'])
        .filter(|t| !t.is_empty())
        .map(str::parse::<f32>)
        .collect::<Result<_, _>>()
        .ok()?;
    (n.len() == 3 && n.iter().all(|x| x.is_finite())).then(|| Vec3::new(n[0], n[1], n[2]))
}

fn parse_f_list(v: &str) -> Option<Vec<f32>> {
    let n: Vec<f32> = v
        .split([',', ' ', '\t'])
        .filter(|t| !t.is_empty())
        .map(str::parse::<f32>)
        .collect::<Result<_, _>>()
        .ok()?;
    n.iter().all(|x| x.is_finite()).then_some(n)
}

fn parse_yes_no(v: &str) -> Option<bool> {
    match v.trim() {
        "yes" | "true" | "on" | "1" => Some(true),
        "no" | "false" | "off" | "0" => Some(false),
        _ => None,
    }
}

fn is_int(x: f32) -> bool {
    x.fract() == 0.0
}

/// The valid `shape =` values per kind.
fn shapes_for(kind: PropKind) -> &'static [&'static str] {
    match kind {
        PropKind::FurnitureTrace => &[
            "table", "bench", "chair", "cot", "planter", "rail", "frame", "pier", "coat", "hook",
            "stack",
        ],
        PropKind::HoloDisplay => &["box", "pyramid", "sphere"],
        PropKind::LampPendant => &["cone", "square"],
        PropKind::Stair => &["straight", "switchback", "spiral"],
        _ => &[],
    }
}

/// Interpret one authored `[prop]` against the level's defaults. Returns the spec
/// and the `(line, key)` pairs of keys this KIND does not use (surfaced as unknown
/// keys, never an error).
pub fn parse_spec(
    def: &PropDef,
    level_name: &str,
    default_style: PropStyle,
) -> Result<(PropSpec, Vec<(usize, String)>), PropError> {
    let name = &def.name;
    let bad = |line: usize, key: &str, msg: String| {
        PropError::new(line, format!("[prop {name}] {key}: {msg}"))
    };
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
    {
        return Err(PropError::new(
            def.line,
            format!("[prop {name}] name must match [a-z0-9_]+"),
        ));
    }
    let mut key_lines: HashMap<String, usize> = HashMap::new();
    let mut vals: BTreeMap<&str, (&str, usize)> = BTreeMap::new();
    for k in &def.keys {
        if key_lines.insert(k.key.clone(), k.line).is_some() {
            return Err(bad(k.line, &k.key, "duplicate key".into()));
        }
        vals.insert(k.key.as_str(), (k.value.trim(), k.line));
    }
    let (kind_v, kind_line) = vals.get("kind").copied().ok_or_else(|| {
        PropError::new(
            def.line,
            format!("[prop {name}] missing required key 'kind'"),
        )
    })?;
    let kind = PropKind::parse(kind_v).ok_or_else(|| {
        bad(
            kind_line,
            "kind",
            format!(
                "unknown kind '{kind_v}' (known: {})",
                PropKind::ALL.map(PropKind::as_str).join(", ")
            ),
        )
    })?;

    // Keys this kind does not use: reported, not fatal.
    let mut inapplicable = Vec::new();
    for k in &def.keys {
        let key = k.key.as_str();
        let generic = GENERIC_KEYS.contains(&key) || key.starts_with("mat.");
        let applies = if generic {
            !kind.rejects_generic(key)
        } else {
            kind.extra_keys().contains(&key)
        };
        if !applies {
            inapplicable.push((k.line, k.key.clone()));
        }
    }
    let get = |k: &str| -> Option<(&str, usize)> {
        let hit = vals.get(k).copied()?;
        let generic = GENERIC_KEYS.contains(&k);
        let applies = if generic {
            !kind.rejects_generic(k)
        } else {
            kind.extra_keys().contains(&k)
        };
        applies.then_some(hit)
    };

    // at — required except for `follow`-routed kinds.
    let at = match get("at") {
        Some((v, l)) => {
            let p =
                parse_f3(v).ok_or_else(|| bad(l, "at", format!("expected 'X Y Z', got '{v}'")))?;
            if kind.is_solid() && !(is_int(p.x) && is_int(p.y) && is_int(p.z)) {
                return Err(bad(
                    l,
                    "at",
                    format!(
                        "a solid kind ({}) needs integer cell coordinates",
                        kind.as_str()
                    ),
                ));
            }
            Some(p)
        }
        None => None,
    };
    let follow = match get("follow") {
        Some((v, l)) => {
            let rest = v
                .strip_prefix("pipe:")
                .ok_or_else(|| bad(l, "follow", format!("expected 'pipe:FROM->TO', got '{v}'")))?;
            let (a, b) = rest
                .split_once("->")
                .ok_or_else(|| bad(l, "follow", format!("expected 'pipe:FROM->TO', got '{v}'")))?;
            Some((a.trim().to_string(), b.trim().to_string()))
        }
        None => None,
    };
    let at = match (at, &follow) {
        (Some(p), _) => p,
        (None, Some(_)) => Vec3::ZERO,
        (None, None) => {
            return Err(PropError::new(
                def.line,
                format!("[prop {name}] missing required key 'at'"),
            ));
        }
    };

    let facing = match get("facing") {
        Some((v, l)) => Facing::parse(v).ok_or_else(|| {
            bad(
                l,
                "facing",
                format!("expected +x | -x | +y | -y, got '{v}'"),
            )
        })?,
        None => Facing::PosY,
    };

    let mut warnings = Vec::new();
    let mut size_set = false;
    let size = match kind.size_rule() {
        Some((def_s, min, max)) => {
            let mut s = def_s;
            if let Some((v, l)) = get("size") {
                let n = parse_f_list(v)
                    .ok_or_else(|| bad(l, "size", format!("expected 'W D H', got '{v}'")))?;
                if n.is_empty() || n.len() > 3 {
                    return Err(bad(l, "size", format!("expected 'W D H', got '{v}'")));
                }
                if kind.is_solid() && n.iter().any(|x| !is_int(*x)) {
                    return Err(bad(
                        l,
                        "size",
                        "a solid kind needs integer dimensions".into(),
                    ));
                }
                size_set = true;
                for (i, x) in n.iter().enumerate() {
                    s[i] = *x;
                }
                for i in 0..3 {
                    if i >= n.len() {
                        continue;
                    }
                    if s[i] < min[i] {
                        return Err(bad(
                            l,
                            "size",
                            format!(
                                "{} below the {} minimum {}",
                                ["W", "D", "H"][i],
                                kind.as_str(),
                                min[i]
                            ),
                        ));
                    }
                    if s[i] > max[i] {
                        warnings.push(format!(
                            "size {} = {} above the {} maximum {}; clamped",
                            ["W", "D", "H"][i],
                            s[i],
                            kind.as_str(),
                            max[i]
                        ));
                        s[i] = max[i];
                    }
                }
            }
            s
        }
        None => {
            if let Some((_, l)) = get("size") {
                inapplicable.push((l, "size".to_string()));
            }
            [1.0, 1.0, 1.0]
        }
    };

    let to = match get("to") {
        Some((v, l)) => {
            Some(parse_f3(v).ok_or_else(|| bad(l, "to", format!("expected 'X Y Z', got '{v}'")))?)
        }
        None => None,
    };
    let lift = match get("lift") {
        Some((v, l)) => v
            .parse::<i32>()
            .map_err(|_| bad(l, "lift", format!("expected an integer, got '{v}'")))?,
        None => {
            if kind == PropKind::CableTray {
                2
            } else {
                0
            }
        }
    };
    let extend = match get("extend") {
        Some((v, l)) => Some(
            parse_f3(v).ok_or_else(|| bad(l, "extend", format!("expected 'X Y Z', got '{v}'")))?,
        ),
        None => None,
    };
    if kind.is_linear() && kind != PropKind::CableDrop && follow.is_none() && to.is_none() {
        return Err(PropError::new(
            def.line,
            format!(
                "[prop {name}] {} needs 'to = X Y Z' or 'follow = pipe:A->B'",
                kind.as_str()
            ),
        ));
    }
    if kind == PropKind::CableDrop && to.is_none() {
        return Err(PropError::new(
            def.line,
            format!("[prop {name}] cable_drop needs 'to = X Y Z'"),
        ));
    }
    if follow.is_some() && vals.contains_key("at") && kind != PropKind::CableRun {
        // `at` is ignored when a route is followed; not an error.
    }

    let mode = match get("mode") {
        Some((v, l)) => PropMode::parse(v).ok_or_else(|| {
            bad(
                l,
                "mode",
                format!("expected build | skin | dress | bank, got '{v}'"),
            )
        })?,
        None => PropMode::Build,
    };
    let mode_ok = match mode {
        PropMode::Build => true,
        PropMode::Skin => matches!(
            kind,
            PropKind::Rack | PropKind::Shelving | PropKind::Column | PropKind::WallPanel
        ),
        PropMode::Dress => kind == PropKind::Gallery,
        PropMode::Bank => kind == PropKind::WallPanel,
    };
    if !mode_ok {
        let (_, l) = vals["mode"];
        return Err(bad(
            l,
            "mode",
            format!("mode is not supported by kind {}", kind.as_str()),
        ));
    }

    let mut holo_style = 0u16;
    let style = match vals.get("style").copied() {
        Some((v, l)) => match PropStyle::parse(v) {
            Some(s) => s,
            None if kind == PropKind::HoloDisplay => {
                // A hologram tint: a numeric style, a preset hue name, or `#RRGGBB`
                // (see `re_core::decode_holo_style` for the layout).
                let hex = v.strip_prefix('#').and_then(|h| {
                    (h.len() == 6)
                        .then(|| u32::from_str_radix(h, 16).ok())
                        .flatten()
                });
                holo_style = match (v.parse::<u16>(), re_core::holo_style_named(v), hex) {
                    (Ok(n), _, _) => n,
                    (_, Some(n), _) => n,
                    (_, _, Some(rgb)) => re_core::encode_holo_rgb(rgb),
                    _ => {
                        let names: Vec<&str> =
                            re_core::HOLO_PRESETS.iter().map(|(n, _)| *n).collect();
                        return Err(bad(
                            l,
                            "style",
                            format!(
                                "expected a pack name, a hologram hue ({}), #RRGGBB or a numeric \
                                 hologram style, got '{v}'",
                                names.join(" | ")
                            ),
                        ));
                    }
                };
                default_style
            }
            None => {
                return Err(bad(
                    l,
                    "style",
                    format!("expected lab | range | vault | archive | dead | void, got '{v}'"),
                ))
            }
        },
        None => default_style,
    };

    let power = match get("power") {
        Some((v, l)) => PowerSpec::parse(v).ok_or_else(|| {
            bad(
                l,
                "power",
                format!("expected none | node:N | pipe:A->B | door | budget:K | seq:I, got '{v}'"),
            )
        })?,
        None => PowerSpec::None,
    };
    let invert = match get("invert") {
        Some((v, l)) => parse_yes_no(v)
            .ok_or_else(|| bad(l, "invert", format!("expected yes | no, got '{v}'")))?,
        None => false,
    };
    let seed = match get("seed") {
        Some((v, l)) => v
            .parse::<u32>()
            .map_err(|_| bad(l, "seed", format!("expected a u32, got '{v}'")))?,
        None => fnv1a(format!("{level_name}{name}").as_bytes()),
    };
    let mut mats = Vec::new();
    for k in &def.keys {
        if let Some(slot) = k.key.strip_prefix("mat.") {
            let slot = MatSlot::parse(slot).ok_or_else(|| {
                bad(
                    k.line,
                    &k.key,
                    "unknown slot (body | face | trim | lit | dead | glass | ink)".into(),
                )
            })?;
            let v = k.value.trim();
            if v.is_empty() {
                return Err(bad(k.line, &k.key, "empty material".into()));
            }
            mats.push((slot, v.to_string(), k.line));
        }
    }
    let light = match get("light") {
        Some((v, l)) => match v {
            "no" => LightMode::No,
            "yes" => LightMode::Yes,
            "existing" => LightMode::Existing,
            _ => {
                return Err(bad(
                    l,
                    "light",
                    format!("expected no | yes | existing, got '{v}'"),
                ))
            }
        },
        None => LightMode::No,
    };
    let emitter = match get("emitter") {
        Some((v, l)) => parse_yes_no(v)
            .ok_or_else(|| bad(l, "emitter", format!("expected yes | no, got '{v}'")))?,
        None => false,
    };
    let note = get("note").map(|(v, _)| v.to_string());
    let shape = match get("shape") {
        Some((v, l)) => {
            let ok = shapes_for(kind);
            if !ok.contains(&v) {
                return Err(bad(
                    l,
                    "shape",
                    format!(
                        "unknown shape '{v}' for {} (known: {})",
                        kind.as_str(),
                        ok.join(", ")
                    ),
                ));
            }
            Some(v.to_string())
        }
        None => None,
    };
    if kind == PropKind::FurnitureTrace && shape.is_none() {
        return Err(PropError::new(
            def.line,
            format!("[prop {name}] furniture_trace needs 'shape ='"),
        ));
    }
    let count = match get("count") {
        Some((v, l)) => Some(
            v.parse::<u32>()
                .ok()
                .filter(|&c| c >= 1)
                .ok_or_else(|| bad(l, "count", format!("expected a count >= 1, got '{v}'")))?,
        ),
        None => None,
    };

    let yes_no = |k: &str, d: bool| -> Result<bool, PropError> {
        match get(k) {
            Some((v, l)) => {
                parse_yes_no(v).ok_or_else(|| bad(l, k, format!("expected yes | no, got '{v}'")))
            }
            None => Ok(d),
        }
    };
    let uint = |k: &str, lo: u32, hi: u32| -> Result<Option<u32>, PropError> {
        match get(k) {
            Some((v, l)) => v
                .parse::<u32>()
                .ok()
                .filter(|x| (lo..=hi).contains(x))
                .map(Some)
                .ok_or_else(|| {
                    bad(
                        l,
                        k,
                        format!("expected an integer in {lo}..={hi}, got '{v}'"),
                    )
                }),
            None => Ok(None),
        }
    };

    let gap = uint("gap", 2, 64)?;
    let dead = match get("dead") {
        Some((v, l)) => v
            .parse::<f32>()
            .ok()
            .filter(|x| (0.0..=1.0).contains(x))
            .ok_or_else(|| bad(l, "dead", format!("expected a fraction 0.0-1.0, got '{v}'")))?,
        None => 0.0,
    };
    let tray = yes_no("tray", false)?;
    let power_each = match get("power_each") {
        Some((v, l)) => v
            .split(',')
            .map(|t| {
                PowerSpec::parse(t)
                    .ok_or_else(|| bad(l, "power_each", format!("bad source '{}'", t.trim())))
            })
            .collect::<Result<Vec<_>, _>>()?,
        None => Vec::new(),
    };
    let monitor = yes_no("monitor", false)?;
    let chair = yes_no("chair", false)?;
    let hatch = match get("hatch") {
        Some((v, l)) => Some(match v {
            "/" => '/',
            "|" => '|',
            "\\" => '\\',
            _ => return Err(bad(l, "hatch", format!("expected / | | | \\, got '{v}'"))),
        }),
        None => None,
    };
    let faces_both = match get("faces") {
        Some((v, l)) => match v {
            "both" => true,
            "front" => false,
            _ => return Err(bad(l, "faces", format!("expected front | both, got '{v}'"))),
        },
        None => false,
    };
    let open = uint("open", 0, 63)?.map(|x| x as i32);
    let ribs = yes_no("ribs", false)?;
    let door_gap = match get("door_gap") {
        Some((v, l)) => {
            let (a, b) = v.split_once("..").unwrap_or((v, v));
            let (a, b) = (a.trim().parse::<i32>(), b.trim().parse::<i32>());
            match (a, b) {
                (Ok(a), Ok(b)) if a >= 0 && b >= a => Some((a, b)),
                _ => return Err(bad(l, "door_gap", format!("expected 'U0..U1', got '{v}'"))),
            }
        }
        None => None,
    };
    let look = match get("look") {
        Some((v, l)) => Some(
            parse_f3(v).ok_or_else(|| bad(l, "look", format!("expected 'X Y Z', got '{v}'")))?,
        ),
        None => None,
    };
    let screen = match get("screen") {
        Some((v, l)) => match v {
            "live" => Screen::Live,
            "dead" => Screen::Dead,
            "mixed" => Screen::Mixed,
            "none" => Screen::None,
            _ => {
                return Err(bad(
                    l,
                    "screen",
                    format!("expected live | dead | mixed | none, got '{v}'"),
                ))
            }
        },
        None => Screen::None,
    };
    let face = match get("face") {
        Some((v, l)) => {
            if !matches!(
                v,
                "fabric" | "lattice" | "watcher" | "codewall" | "redacted"
            ) {
                return Err(bad(
                    l,
                    "face",
                    format!("expected fabric | lattice | watcher | codewall | redacted, got '{v}'"),
                ));
            }
            Some(v.to_string())
        }
        None => None,
    };
    let form_src = get("form").map(|(v, l)| (v, l, "form")).or_else(|| {
        (kind == PropKind::Stair)
            .then(|| get("shape").map(|(v, l)| (v, l, "shape")))
            .flatten()
    });
    let form = match form_src {
        Some((v, l, key)) => match v {
            "straight" => StairForm::Straight,
            "switchback" => StairForm::Switchback,
            "spiral" => StairForm::Spiral,
            _ => {
                return Err(bad(
                    l,
                    key,
                    format!("expected straight | switchback | spiral, got '{v}'"),
                ))
            }
        },
        None => StairForm::Straight,
    };
    let rise = uint("rise", 1, 64)?;
    let radius = uint("radius", 1, 8)?.map_or(1, |r| r as i32);
    let post_every = uint("post_every", 1, 32)?.map_or(4, |p| p as i32);
    let posts = match get("posts") {
        None => GalleryPosts::Auto,
        Some((v, l)) => GalleryPosts::parse(v).ok_or_else(|| {
            bad(
                l,
                "posts",
                format!(
                    "expected auto | no_ends | none | a list of deck cells 'U V; U V …', got '{v}'"
                ),
            )
        })?,
    };
    let stair = match get("stair") {
        Some((v, l)) => {
            let t: Vec<&str> = v
                .split([' ', '\t', ','])
                .filter(|s| !s.is_empty())
                .collect();
            let parsed = (t.len() == 5)
                .then(|| {
                    let x = t[0].parse::<i32>().ok()?;
                    let y = t[1].parse::<i32>().ok()?;
                    let z = t[2].parse::<i32>().ok()?;
                    let f = Facing::parse(t[3])?;
                    let r = t[4].parse::<u32>().ok().filter(|r| *r >= 1)?;
                    Some((IVec3::new(x, y, z), f, r))
                })
                .flatten();
            Some(parsed.ok_or_else(|| {
                bad(
                    l,
                    "stair",
                    format!("expected 'X Y Z FACING RISE', got '{v}'"),
                )
            })?)
        }
        None => None,
    };
    let riser = match get("riser") {
        Some((v, l)) => match v {
            "none" => Riser::None,
            "start" => Riser::Start,
            "end" => Riser::End,
            "both" => Riser::Both,
            _ => {
                return Err(bad(
                    l,
                    "riser",
                    format!("expected none | start | end | both, got '{v}'"),
                ))
            }
        },
        None => Riser::None,
    };
    let end_vent = match get("end") {
        Some((v, l)) => match v {
            "vent" => true,
            "cap" | "none" => false,
            _ => return Err(bad(l, "end", format!("expected vent | none, got '{v}'"))),
        },
        None => false,
    };
    let strands = uint("strands", 1, 3)?.unwrap_or(1);
    let bundle = uint("bundle", 1, 3)?.unwrap_or(1);
    let base = yes_no("base", false)?;
    let solid = yes_no("solid", false)?;
    if solid
        && !shape
            .as_deref()
            .is_some_and(|sh| SOLID_FURNITURE_SHAPES.contains(&sh))
    {
        let l = get("solid").map_or(def.line, |(_, l)| l);
        return Err(bad(
            l,
            "solid",
            format!(
                "solid = yes needs shape = {} (got '{}')",
                SOLID_FURNITURE_SHAPES.join(" | "),
                shape.as_deref().unwrap_or("")
            ),
        ));
    }
    let floor = yes_no("floor", false)?;
    let height = match get("height") {
        Some((v, l)) => Some(
            v.parse::<f32>()
                .ok()
                .filter(|h| h.is_finite() && *h > 0.0)
                .ok_or_else(|| bad(l, "height", format!("expected a height > 0, got '{v}'")))?,
        ),
        None => None,
    };

    let spec = PropSpec {
        name: name.clone(),
        line: def.line,
        kind,
        at,
        facing,
        size,
        size_set,
        to,
        follow,
        lift,
        extend,
        mode,
        style,
        holo_style,
        power,
        invert,
        seed,
        mats,
        light,
        emitter,
        note,
        shape,
        count,
        gap,
        dead,
        tray,
        power_each,
        monitor,
        chair,
        hatch,
        faces_both,
        open,
        ribs,
        door_gap,
        look,
        screen,
        face,
        form,
        rise,
        radius,
        post_every,
        posts,
        stair,
        riser,
        end_vent,
        strands,
        bundle,
        base,
        solid,
        floor,
        height,
        key_lines,
        warnings,
    };
    Ok((spec, inapplicable))
}

// ─────────────────────────────── materials ───────────────────────────────────

fn rgb(c: u32) -> (f32, f32, f32) {
    (
        ((c >> 16) & 0xFF) as f32,
        ((c >> 8) & 0xFF) as f32,
        (c & 0xFF) as f32,
    )
}

fn pack_rgb(r: f32, g: f32, b: f32) -> u32 {
    let c = |x: f32| x.round().clamp(0.0, 255.0) as u32;
    (c(r) << 16) | (c(g) << 8) | c(b)
}

/// Scale a colour's brightness (the cheap stand-in for an OK_L shift).
pub fn shade(c: u32, k: f32) -> u32 {
    let (r, g, b) = rgb(c);
    pack_rgb(r * k, g * k, b * k)
}

fn recipe(s: &str) -> Option<MaterialRecipe> {
    parse_recipe(s)
}

fn mat(
    name: String,
    recipe_s: Option<&str>,
    preset: Option<&str>,
    fg: u32,
    bg: u32,
) -> LevelMaterial {
    LevelMaterial {
        name,
        fg,
        bg,
        recipe: recipe_s.and_then(recipe),
        preset: preset.map(str::to_string),
        ..LevelMaterial::default()
    }
}

/// The pack colour an INK stroke of this style draws in (§3.5 `ink` column).
pub fn pack_ink(style: PropStyle) -> u32 {
    match style {
        PropStyle::Lab => 0x6C_7888,
        PropStyle::Range => 0x5A_6E82,
        PropStyle::Vault => 0x4E_5E56,
        PropStyle::Archive => 0x34_3C48,
        PropStyle::Dead => 0x2A_2E34,
        PropStyle::Void => 0x9A_8AFF,
    }
}

/// The pack's powered hue when the colour law names none (`door`, `seq:`).
pub fn pack_lit_rgb(style: PropStyle) -> u32 {
    match style {
        PropStyle::Lab => 0x5C_FF9A,
        PropStyle::Range => 0x7F_E8FF,
        PropStyle::Vault => 0xFF_B060,
        PropStyle::Archive => 0x45_D962,
        PropStyle::Dead => 0x2A_2E34,
        PropStyle::Void => 0x9A_8AFF,
    }
}

/// The synthesized `[material __prop_<style>_<slot>]` for one pack slot (§3.5).
pub fn pack_material(style: PropStyle, slot: MatSlot) -> LevelMaterial {
    use MatSlot as S;
    use PropStyle as P;
    let name = format!("{PROP_MATERIAL_PREFIX}{}_{}", style.as_str(), slot.as_str());
    // `dead` never lights: its lit slot IS its dead slot.
    let slot = if style == P::Dead && slot == S::Lit {
        S::Dead
    } else {
        slot
    };
    match (style, slot) {
        // The room walls' brick proportions: a body reads in a dark room by its mortar
        // courses (the blanked body cells sit at the legibility floor), and the old
        // `aspect=3 mortar_w=0.10` almost never inked one, so a lab body was a black
        // slab. The backing is lifted with it.
        (P::Lab, S::Body) => mat(
            name,
            Some("brick aspect=2 mortar_w=0.18 grain=0.25"),
            None,
            0x7A_8698,
            0x18_1E28,
        ),
        (P::Lab, S::Face) => mat(
            name,
            Some("circuit scale=0.9 trace_w=0.14"),
            None,
            0x5F_A8C8,
            0x06_121A,
        ),
        (P::Lab, S::Trim) => mat(
            name,
            Some("grain scale=1 speck=0.05"),
            None,
            0xB8_C4D4,
            0x0E_1218,
        ),
        (P::Lab, S::Lit) => mat(
            name,
            None,
            Some("monitor"),
            0x5C_FF9A,
            shade(0x5C_FF9A, LIT_BACKING),
        ),
        (P::Lab, S::Dead) => mat(name, Some("grain speck=0.02"), None, 0x2A_3038, 0x07_090C),
        (P::Lab, S::Glass) => mat(name, None, Some("glass"), 0, 0),
        (P::Range, S::Body) => mat(name, Some("circuit scale=2.2"), None, 0x4E_6A80, 0x06_0A10),
        (P::Range, S::Face) => mat(name, None, Some("lattice"), 0, 0),
        (P::Range, S::Trim) => mat(name, Some("grain"), None, 0xC8_D8E8, 0x0A_0E14),
        (P::Range, S::Lit) => mat(name, Some("scanline"), None, 0x7F_E8FF, 0x06_1418),
        (P::Range, S::Dead) => mat(name, None, Some("static_noise"), 0, 0),
        (P::Range, S::Glass) => mat(name, None, Some("vitrine"), 0, 0),
        (P::Vault, S::Body) => mat(name, None, Some("stone"), 0x5E_6A64, 0x07_0A08),
        (P::Vault, S::Face) => mat(name, None, Some("frost"), 0, 0),
        (P::Vault, S::Trim) => mat(name, None, Some("metal"), 0, 0),
        (P::Vault, S::Lit) => mat(name, Some("scanline"), None, 0xFF_B060, 0x14_0C04),
        (P::Vault, S::Dead) => mat(name, None, Some("wet_stone"), 0, 0),
        (P::Vault, S::Glass) => mat(name, None, Some("ice"), 0, 0),
        (P::Archive, S::Body) => mat(name, Some("grain speck=0.04"), None, 0x1C_2028, 0x03_0406),
        (P::Archive, S::Face) => mat(name, None, Some("redacted"), 0, 0),
        (P::Archive, S::Trim) => mat(name, None, Some("decay"), 0, 0),
        (P::Archive, S::Lit) => mat(name, Some("codewall"), None, 0x45_D962, 0x02_0803),
        (P::Archive, S::Dead) => mat(name, None, Some("decay"), 0, 0),
        (P::Archive, S::Glass) => mat(name, None, Some("screen"), 0, 0),
        (P::Dead, S::Body) => mat(name, None, Some("decay"), 0, 0),
        // `static_noise` animates; a FROZEN static is a dense speckle grain (the
        // material system has no "hold this preset still" knob).
        (P::Dead, S::Face) => mat(
            name,
            Some("grain scale=2 speck=0.45"),
            None,
            0x3A_3E44,
            0x0A_0B0D,
        ),
        (P::Dead, S::Trim) => mat(name, None, Some("rust"), 0, 0),
        (P::Dead, S::Dead | S::Lit) => mat(name, None, Some("decay"), 0, 0),
        (P::Dead, S::Glass) => mat(name, None, Some("decay"), 0, 0),
        (P::Void, S::Body) => mat(name, None, Some("void"), 0, 0),
        (P::Void, S::Face) => mat(name, None, Some("veil"), 0, 0),
        (P::Void, S::Trim) => mat(name, None, Some("vitrine"), 0, 0),
        (P::Void, S::Lit) => mat(name, Some("fractal"), None, 0x9A_8AFF, 0x0A_0818),
        (P::Void, S::Dead) => mat(name, None, Some("void"), 0, 0),
        (P::Void, S::Glass) => mat(name, None, Some("veil"), 0, 0),
        (style, S::Ink) => mat(name, None, None, pack_ink(style), 0),
    }
}

/// A flat single-colour material (line inks, lit hues, `mat.SLOT = #RRGGBB`).
fn flat_material(c: u32) -> LevelMaterial {
    mat(
        format!("{PROP_MATERIAL_PREFIX}c_{c:06x}"),
        None,
        None,
        c,
        shade(c, 0.18),
    )
}

/// Backing brightness of a lit screen, as a fraction of its hue.
///
/// A lit slot is an EMITTER: the renderer keeps every scanline band on it and spares
/// it most of the fog (`re_render::ascii_shader::SELF_LIT_EMISSIVE`), but its gaps
/// show the backing. At the old 12 % the gaps read darker than lamp-lit brick and the
/// powered payoff was invisible (set-dressing notes 2026-09-24); at half the hue the
/// screen is the brightest thing in a dark room, as a powered part should be.
pub const LIT_BACKING: f32 = 0.5;

/// A lit SCREEN in the colour-law hue: the `monitor` look retinted over a lit backing.
fn lit_screen_material(c: u32) -> LevelMaterial {
    mat(
        format!("{PROP_MATERIAL_PREFIX}lit_{c:06x}"),
        None,
        Some("monitor"),
        c,
        shade(c, LIT_BACKING),
    )
}

/// Backing of a rack's LED slot, as a fraction of its hue: dark, so the bands of
/// light read against it the way a lit machine's indicators do.
pub const RACK_LED_BACKING: f32 = 0.14;

/// A powered RACK blade: the `neon` emitter re-cut as dense thin scanline bands
/// (LED rows) in the source's hue, lifted to full value, over a dark backing. Brick
/// of the same hue and a flat neon slab were indistinguishable at cell scale; bright
/// thin bands with dark gaps are not brick at any distance. Every rack and `rack_row`
/// lit slot uses it, whatever `mat.lit` says (that override still colours the gate's
/// lines, lamps and emitters).
pub fn rack_led_material(c: u32) -> LevelMaterial {
    let r = ((c >> 16) & 0xFF) as f32;
    let g = ((c >> 8) & 0xFF) as f32;
    let b = (c & 0xFF) as f32;
    let peak = r.max(g).max(b).max(1.0);
    let lift = |v: f32| ((v * 255.0 / peak).round().clamp(0.0, 255.0)) as u32;
    let fg = if peak < 8.0 {
        0xFF_FFFF
    } else {
        (lift(r) << 16) | (lift(g) << 8) | lift(b)
    };
    mat(
        format!("{PROP_MATERIAL_PREFIX}led_{c:06x}"),
        Some("scanline freq=14 duty=0.32 warp=0 axis=0"),
        Some("neon"),
        fg,
        shade(fg, RACK_LED_BACKING),
    )
}

/// The floor wash under a floor-standing trace piece: its ink darkened to a contact
/// shadow, on a flat grain so it reads as one patch, not a second texture.
pub fn trace_base_material(ink: u32) -> LevelMaterial {
    mat(
        format!("{PROP_MATERIAL_PREFIX}base_{ink:06x}"),
        Some("grain scale=2 speck=0.04"),
        None,
        shade(ink, 0.55),
        shade(ink, 0.20),
    )
}

/// The `furniture_trace` shapes that have a `solid = yes` form: the "signs of
/// people" pieces a player can actually collide with.
pub const SOLID_FURNITURE_SHAPES: [&str; 4] = ["bench", "chair", "table", "stack"];

/// Where the bright top of a solid furniture block starts, as a fraction of its band.
pub const SOLID_FURNITURE_TOP_FROM: f32 = 0.7;

/// The band height (a fraction of the cell, from its floor) of a `solid = yes` piece.
/// All sit below the standing eye (`0.5` above the floor) so the lid — drawn in the top
/// colour — faces the player: bench / chair seat 0.20, table 0.30, stack 0.42.
pub fn solid_furniture_band(shape: &str) -> f32 {
    match shape {
        "table" => 0.30,
        "stack" => 0.42,
        _ => 0.20,
    }
}

/// The z-band material-name suffix (`re_render::zband::suffix`, mirrored here because
/// `re_level` does not depend on the renderer; a cross-crate test pins the two equal).
pub fn zband_suffix(lo: f32, hi: f32, top: Option<(u32, f32)>) -> String {
    let pct = |v: f32| (v.clamp(0.0, 1.0) * 100.0).round() as u32;
    let mut s = format!("~z{}_{}", pct(lo), pct(hi));
    if let Some((rgb, split)) = top {
        use std::fmt::Write as _;
        let _ = write!(s, "~t{:06x}_{}", rgb & 0xFF_FFFF, pct(split));
    }
    s
}

/// The synthesized material of a `solid = yes` furniture block: a dark grain body in
/// the ink's hue, solid over [`solid_furniture_band`] of the cell, with its top
/// ([`SOLID_FURNITURE_TOP_FROM`] of the band up, and the lid) in a bright tint of the
/// ink — the rim that makes a knee-high block read at one cell of screen.
pub fn solid_furniture_material(ink: u32, shape: &str) -> LevelMaterial {
    let top = brighten(ink, 1.9);
    mat(
        format!(
            "{PROP_MATERIAL_PREFIX}solid_{ink:06x}_{shape}{}",
            zband_suffix(
                0.0,
                solid_furniture_band(shape),
                Some((top, SOLID_FURNITURE_TOP_FROM))
            )
        ),
        Some("grain scale=2 speck=0.04"),
        None,
        shade(ink, 0.85),
        shade(ink, 0.30),
    )
}

/// `c` scaled by `k` per channel, lifted toward white where a channel would clip.
fn brighten(c: u32, k: f32) -> u32 {
    let (r, g, b) = rgb(c);
    let lift = |v: f32| (v * k).max(v + (255.0 - v) * (k - 1.0) * 0.35).min(255.0);
    pack_rgb(lift(r), lift(g), lift(b))
}

fn parse_hex(v: &str) -> Option<u32> {
    let h = v.strip_prefix('#')?;
    (h.len() == 6)
        .then(|| u32::from_str_radix(h, 16).ok())
        .flatten()
}

/// The material interner for one expansion: authored names resolve to themselves,
/// synthesized materials are appended once (deduplicated by name). An authored
/// `[material]` with a synthesized name wins (it is never synthesized).
#[derive(Debug, Default)]
struct Mats {
    authored: HashSet<String>,
    synth: Vec<LevelMaterial>,
    index: HashMap<String, usize>,
}

impl Mats {
    fn new(level: &Level) -> Self {
        Mats {
            authored: level.materials.iter().map(|m| m.name.clone()).collect(),
            ..Mats::default()
        }
    }

    fn intern(&mut self, m: LevelMaterial) -> String {
        if !self.authored.contains(&m.name) && !self.index.contains_key(&m.name) {
            self.index.insert(m.name.clone(), self.synth.len());
            self.synth.push(m.clone());
        }
        m.name
    }

    /// Resolve a `mat.SLOT = X` / `face =` value: an authored material name, a
    /// `#RRGGBB` flat colour, a catalog preset, or a recipe line.
    fn resolve_value(&mut self, v: &str) -> Option<String> {
        if self.authored.contains(v) {
            return Some(v.to_string());
        }
        if let Some(c) = parse_hex(v) {
            return Some(self.intern(flat_material(c)));
        }
        if CATALOG_PRESETS.contains(&v) {
            return Some(self.intern(mat(
                format!("{PROP_MATERIAL_PREFIX}p_{v}"),
                None,
                Some(v),
                0,
                0,
            )));
        }
        if let Some(r) = recipe(v) {
            let name = format!("{PROP_MATERIAL_PREFIX}r_{:08x}", fnv1a(v.as_bytes()));
            return Some(self.intern(LevelMaterial {
                name,
                recipe: Some(r),
                fg: 0x9A_A4B0,
                bg: 0x0A_0C10,
                ..LevelMaterial::default()
            }));
        }
        None
    }
}

// ──────────────────────────── expansion output ───────────────────────────────

/// Why a stamped cell exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellMode {
    /// A new cell stamped into Empty air.
    Build,
    /// An existing cell re-materialed (kind/shape untouched).
    Skin,
}

/// One cell a prop stamps.
#[derive(Clone, Debug, PartialEq)]
pub struct PropCell {
    /// The cell coordinate.
    pub pos: Cell3,
    /// The map cell kind this cell is stamped as.
    pub kind: CellKind,
    /// The cell's geometry shape (full cube, wedge, ramp, …).
    pub shape: Shape,
    /// The material NAME (authored or synthesized), or `None` for an UNTAGGED cell
    /// (a notice board's placard host keeps tag 0 so the note binder paints it).
    pub mat: Option<String>,
    /// Whether this cell was built into Empty air or skins an existing cell.
    pub mode: CellMode,
    /// Index of the owning prop in `Level::props`.
    pub prop: usize,
}

/// One stroke a prop draws (world coordinates).
#[derive(Clone, Debug, PartialEq)]
pub struct PropLine {
    /// The polyline's world-space points, in draw order.
    pub points: Vec<Vec3>,
    /// The UNLIT (ink) material — or the only material of an ungated stroke.
    pub mat: String,
    /// Index of the owning prop in `Level::props`.
    pub prop: usize,
    /// `Some(gate index)` when this stroke switches ink ↔ lit with power.
    pub gate: Option<usize>,
    /// The lit material when gated.
    pub lit_mat: Option<String>,
    /// Drawn lit at load (the gate's initial state).
    pub lit_now: bool,
}

impl PropLine {
    /// The material the built map draws this stroke in.
    pub fn drawn_mat(&self) -> &str {
        match (&self.lit_mat, self.lit_now) {
            (Some(l), true) => l,
            _ => &self.mat,
        }
    }
}

/// One hologram volume.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PropHolo {
    /// The hologram's silhouette (box / pyramid / sphere).
    pub shape: HoloShape,
    /// The volume's min corner, in cells.
    pub min: Cell3,
    /// The volume's max corner, in cells.
    pub max: Cell3,
    /// Numeric hologram style tag (hue or catalog id).
    pub style: u16,
    /// Index of the owning prop in `Level::props`.
    pub prop: usize,
    /// Shown at load (false ⇒ a powered holo waiting for its source).
    pub visible: bool,
    /// `Some(gate index)` when this hologram is shown only while powered.
    pub gate: Option<usize>,
}

impl PropHolo {
    fn object(&self) -> re_world::HoloObject {
        re_world::HoloObject {
            shape: self.shape,
            x0: self.min.0,
            y0: self.min.1,
            z0: self.min.2,
            x1: self.max.0,
            y1: self.max.1,
            z1: self.max.2,
            style: self.style,
        }
    }
}

/// A floor/wall decal: re-material the solid cells of a box (a `[zone]` that only
/// the prop owns — e.g. overgrowth's moss floor).
#[derive(Clone, Debug, PartialEq)]
pub struct PropDecal {
    /// The box's min corner, in cells.
    pub min: Cell3,
    /// The box's max corner, in cells.
    pub max: Cell3,
    /// The material this box's solid cells are re-materialed to.
    pub mat: String,
    /// Index of the owning prop in `Level::props`.
    pub prop: usize,
}

/// An R1 lit cell: an Empty slot that fills with `mat` while powered.
#[derive(Clone, Debug, PartialEq)]
pub struct AbsPart {
    /// The cell coordinate.
    pub pos: Cell3,
    /// The material this slot fills with while lit.
    pub mat: String,
    /// Index of the owning prop in `Level::props`.
    pub prop: usize,
    /// Index of the gate that controls this slot.
    pub gate: usize,
}

/// How bad an expansion finding is (maps onto lvlcheck WARN / FAIL).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum IssueLevel {
    /// A soft finding: worth a look, not blocking.
    Warn,
    /// A hard finding: `lvlcheck`'s `props` rule fails the level.
    Fail,
}

/// A non-fatal expansion finding, surfaced by `lvlcheck`'s `props` rule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PropIssue {
    /// The name of the `[prop]` this finding is about.
    pub prop: String,
    /// How bad this finding is.
    pub level: IssueLevel,
    /// Human-readable description of the finding.
    pub msg: String,
    /// The cell the finding concerns, if any.
    pub pos: Option<Cell3>,
}

/// A note re-homed onto a prop (`note = KIND`): the note index, its new reader
/// cell and its new facing (a `[note]` yaw column).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NoteMove {
    /// Index into `Level::notes`.
    pub note: usize,
    /// The note's new reader cell.
    pub pos: Cell3,
    /// The note's new `[note] facing` yaw column.
    pub facing: i32,
}

/// Where a re-homed note was AUTHORED (`Level::prop_note_origins`): the writer
/// emits this position, never the moved one, so a save does not bake the move.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NoteOrigin {
    /// Index into `Level::notes`.
    pub note: usize,
    /// Authored X coordinate.
    pub x: i32,
    /// Authored Y coordinate.
    pub y: i32,
    /// Authored Z coordinate.
    pub z: i32,
    /// Authored `[note] facing` yaw column.
    pub facing: i32,
}

/// Internal gate under construction (material names, prop-local indices).
#[derive(Clone, Debug)]
pub(crate) struct GateDraft {
    pub(crate) prop: String,
    pub(crate) spec: PowerSpec,
    pub(crate) source: PowerSource,
    pub(crate) invert: bool,
    pub(crate) initial_lit: bool,
    pub(crate) lit_rgb: u32,
    /// `(index into the expansion's lights, the LIT light)`.
    pub(crate) lights: Vec<(usize, PointLight)>,
    /// `(index into the expansion's emitters, the OTHER-state spec)`.
    pub(crate) emitters: Vec<(usize, EmitterSpec)>,
}

/// Everything the level's props expand to.
#[derive(Clone, Debug, Default)]
pub struct PropExpansion {
    /// Solid/skin cells every prop stamps.
    pub cells: Vec<PropCell>,
    /// Line strokes every prop draws.
    pub lines: Vec<PropLine>,
    /// Hologram volumes every prop places.
    pub holos: Vec<PropHolo>,
    /// Floor/wall decals (re-materialed boxes) every prop places.
    pub decals: Vec<PropDecal>,
    /// Synthesized (`synthesized == true`) emitters, in the initial state.
    pub emitters: Vec<EmitterSpec>,
    /// Prop lamps (`light = yes`), in their initial state.
    pub lights: Vec<PointLight>,
    /// R1 lit slots (all of them; filled at load iff their gate is initially lit).
    pub parts: Vec<AbsPart>,
    /// Synthesized materials, in palette order after `Level::materials`.
    pub materials: Vec<LevelMaterial>,
    /// Notes re-homed onto a prop (`note = KIND`).
    pub note_moves: Vec<NoteMove>,
    /// Non-fatal findings from the expansion pass.
    pub issues: Vec<PropIssue>,
    /// Stair tread + landing cells the kit emitted (the shortcut rule's exemption).
    pub stair_cells: Vec<Cell3>,
    /// Total line segments (the §3.7 ≤ 400 budget).
    pub segments: usize,
    /// Keys a prop's KIND does not use: `(line, key)`.
    pub unused_keys: Vec<(usize, String)>,
    /// Power gates still under construction (material names, prop-local indices).
    pub(crate) gates: Vec<GateDraft>,
}

impl PropExpansion {
    /// The 1-based palette id of material `name` in the combined palette
    /// (`Level::materials` followed by [`PropExpansion::materials`]).
    pub fn palette_id(&self, level: &Level, name: &str) -> u16 {
        if let Some(i) = level.materials.iter().position(|m| m.name == name) {
            return (i + 1) as u16;
        }
        match self.materials.iter().position(|m| m.name == name) {
            Some(i) => (level.materials.len() + i + 1) as u16,
            None => 0,
        }
    }

    /// Is this R1 slot filled at load (its gate starts lit)?
    fn part_initially_filled(&self, part: &AbsPart) -> bool {
        self.gates.get(part.gate).is_some_and(|g| g.initial_lit)
    }

    /// Resolve the drafted gates into [`PowerGate`]s with absolute indices.
    /// `line_base` = the number of authored `[line]`s (prop lines follow them),
    /// `emitter_base` = the index of the first prop emitter in `Level::emitters`,
    /// `light_base` = the number of authored `[light]`s.
    pub fn power_gates(
        &self,
        level: &Level,
        line_base: usize,
        emitter_base: usize,
        light_base: usize,
    ) -> Vec<PowerGate> {
        self.gates
            .iter()
            .enumerate()
            .map(|(gi, g)| PowerGate {
                prop: g.prop.clone(),
                source: g.source,
                invert: g.invert,
                initial_lit: g.initial_lit,
                lit_rgb: g.lit_rgb,
                parts: self
                    .parts
                    .iter()
                    .filter(|p| p.gate == gi)
                    .map(|p| (p.pos, self.palette_id(level, &p.mat)))
                    .collect(),
                lines: self
                    .lines
                    .iter()
                    .enumerate()
                    .filter(|(_, l)| l.gate == Some(gi))
                    .map(|(i, l)| {
                        let lit = l.lit_mat.as_deref().unwrap_or(&l.mat);
                        (
                            line_base + i,
                            self.palette_id(level, &l.mat),
                            self.palette_id(level, lit),
                        )
                    })
                    .collect(),
                holos: self
                    .holos
                    .iter()
                    .filter(|h| h.gate == Some(gi))
                    .map(PropHolo::object)
                    .collect(),
                emitters: g
                    .emitters
                    .iter()
                    .map(|(i, e)| (emitter_base + i, e.clone()))
                    .collect(),
                lights: g.lights.iter().map(|(i, l)| (light_base + i, *l)).collect(),
            })
            .collect()
    }
}

// ──────────────────────────────── keep-out ───────────────────────────────────

/// One keep-out cell: what puzzle element owns it, and whether a solid prop there
/// is a LOAD error (`hard`: the element itself) or an lvlcheck FAIL (a ring).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeepOutEntry {
    /// `true` ⇒ a solid prop cell here is a LOAD error (the element itself);
    /// `false` ⇒ an lvlcheck FAIL (a keep-out ring around it).
    pub hard: bool,
    /// Human-readable name of the owning puzzle element (for error messages).
    pub what: String,
}

/// The §3.7 keep-out set K.
#[derive(Clone, Debug, Default)]
pub struct KeepOut {
    /// Every reserved cell, keyed by coordinate.
    pub cells: HashMap<Cell3, KeepOutEntry>,
}

impl KeepOut {
    fn add(&mut self, c: Cell3, hard: bool, what: impl Into<String>) {
        let e = self.cells.entry(c).or_insert(KeepOutEntry {
            hard,
            what: String::new(),
        });
        // A hard claim upgrades a soft one and names the element.
        if e.what.is_empty() || (hard && !e.hard) {
            e.what = what.into();
            e.hard = e.hard || hard;
        }
    }

    /// The keep-out entry reserving cell `c`, if any.
    pub fn get(&self, c: Cell3) -> Option<&KeepOutEntry> {
        self.cells.get(&c)
    }

    /// The switch columns `(x, y, floor+1 ..= switch z)` — the cells a TRACE prop
    /// may never occupy.
    pub fn is_switch_column(&self, c: Cell3) -> bool {
        self.cells
            .get(&c)
            .is_some_and(|e| e.what.starts_with("switch column"))
    }
}

/// The first solid cell at or below `z` in column `(x, y)`, or `-1`.
fn floor_below(map: &Map, x: i32, y: i32, z: i32) -> i32 {
    let mut zz = z;
    while zz >= 0 {
        let k = map.cell_at(x, y, zz);
        if k.is_solid() || k == CellKind::Floor {
            return zz;
        }
        zz -= 1;
    }
    -1
}

/// The world cells an `[emit]` reward can occupy (mirrors
/// `re_runtime::emit_parts::part_cells` and lvlcheck's `emit_part_cells`).
fn emit_part_cells(level: &Level) -> Vec<Cell3> {
    let mut out = Vec::new();
    let ped = &level.pedestal;
    for emit in &level.emits {
        let Some(part) = level.parts.get(emit.part) else {
            continue;
        };
        let (w, h, d) = (part.w.max(0), part.h.max(0), part.d.max(0));
        let bx = ped.cx - ped.w / 2 + part.off_x;
        let by = ped.cy - ped.h / 2 + part.off_y;
        let bz = ped.cz + part.off_z;
        for pz in 0..d {
            for py in 0..h {
                for px in 0..w {
                    let idx =
                        px as usize + (w as usize) * (py as usize + (h as usize) * pz as usize);
                    if part.mask.get(idx).copied().unwrap_or(0) != 0 {
                        out.push((bx + px, by + py, bz + pz));
                    }
                }
            }
        }
    }
    out
}

/// Build the keep-out set K from the level's puzzle and the prop-free map.
/// `moved` lists the notes a prop re-homes (`note =`): they reserve nothing.
pub fn keep_out(level: &Level, base: &Map, moved: &HashSet<usize>) -> KeepOut {
    let mut k = KeepOut::default();
    // 1 + 2: pipe paths, switch columns.
    for p in &level.pipes {
        let label = format!("{}->{}", p.from, p.to);
        for &c in &p.path {
            k.add(c, true, format!("pipe {label} path"));
        }
        if let Some(sc) = p.switch_cell.and_then(|i| p.path.get(i)).copied() {
            let fl = floor_below(base, sc.0, sc.1, sc.2 - 1);
            for z in (fl + 1).max(0)..=sc.2 {
                k.add(
                    (sc.0, sc.1, z),
                    true,
                    format!("switch column of pipe {label}"),
                );
            }
        }
    }
    // 3: nodes.
    for n in &level.nodes {
        k.add((n.x, n.y, n.z), true, format!("node {}", n.name));
    }
    // 4: pedestal box (hard) + 1-cell ring (soft), z = cz..cz+2.
    let ped = &level.pedestal;
    if ped.w > 0 && ped.h > 0 {
        let (x0, y0) = (ped.cx - ped.w / 2, ped.cy - ped.h / 2);
        let (x1, y1) = (x0 + ped.w - 1, y0 + ped.h - 1);
        for z in ped.cz..=ped.cz + 2 {
            for y in y0 - 1..=y1 + 1 {
                for x in x0 - 1..=x1 + 1 {
                    let inside = (x0..=x1).contains(&x) && (y0..=y1).contains(&y);
                    k.add(
                        (x, y, z),
                        inside,
                        if inside { "pedestal" } else { "pedestal ring" },
                    );
                }
            }
        }
    }
    // 5: door cell (hard) + Manhattan-2 approach, 2 tall (soft).
    let d = &level.door;
    for dz in 0..2 {
        for dy in -2..=2i32 {
            for dx in -2..=2i32 {
                if dx.abs() + dy.abs() <= 2 {
                    let c = (d.x + dx, d.y + dy, d.z + dz);
                    let hard = dx == 0 && dy == 0 && dz == 0;
                    k.add(c, hard, if hard { "door" } else { "door approach" });
                }
            }
        }
    }
    // 6: spawn cell + headroom (hard) + 8-ring, 2 tall (soft).
    let s = &level.player;
    for dz in 0..2 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let hard = dx == 0 && dy == 0;
                k.add(
                    (s.x + dx, s.y + dy, s.z + dz),
                    hard,
                    if hard { "spawn" } else { "spawn ring" },
                );
            }
        }
    }
    // 7: note reader cells + the wall each placard mounts on.
    // A note a prop re-homed (`note =`) belongs to that prop; its old and new cells
    // are the prop's business, so only the notes left where they were authored
    // reserve their reader cell and placard wall.
    //
    // HARD, and the whole READER SPACE, not just the authored cell: the binder puts
    // the placard on the first masonry along the reader's gaze at the eye layer
    // (`Level::note_host_cells`), so a prop cell anywhere between the reader and
    // that wall becomes the new "first masonry" and the note silently re-homes onto
    // the prop (the L4 gallery took `system_log` this way). Reader cell, gaze run and
    // placard wall — at the authored z and at the eye layer the binder uses — are
    // therefore refused like a node or a pipe.
    for (i, n) in level.notes.iter().enumerate() {
        if moved.contains(&i) {
            continue;
        }
        let one = Level {
            size: level.size,
            storey_height: level.storey_height,
            notes: vec![n.clone()],
            ..Level::default()
        };
        let label = |what: &str| format!("note {} {what}", note_label(n, i));
        k.add((n.x, n.y, n.z), true, label("reader cell"));
        let (dx, dy) = note_gaze(n.facing);
        for (hx, hy, eye) in one.note_host_cells(base) {
            k.add((hx, hy, eye), true, label("placard wall"));
            // The run from the reader to the wall, on both planes the binder reads.
            let steps = (hx - n.x).abs().max((hy - n.y).abs());
            for s in 0..steps {
                let (cx, cy) = (n.x - dx * s, n.y - dy * s);
                for z in [n.z, eye] {
                    k.add((cx, cy, z), true, label("reader space"));
                }
            }
        }
    }
    // 8: portal openings + 2 cells straight out from them (soft: stamped, lvlcheck
    // FAIL — a prop there blocks the warp or sits in the view it frames).
    for pt in &level.portals {
        for (a, b) in [pt.region_a(), pt.region_b()] {
            face_ring(
                &mut k,
                a,
                b,
                2,
                false,
                &format!("portal {} opening", pt.name),
            );
        }
    }
    // 9: mirrors + 1 cell straight out (soft, like the portals). A mirror is a
    // `[zone]` OR a `[wall_decal]` re-kinding a patch to a surface whose bound
    // `[material]` is reflective (the L4 range mirror is the decal form, which the
    // zone-only rule missed). Soft, not hard, so a dressed level that already crowds
    // a mirror still loads and lvlcheck names it.
    let reflective = |name: &str| {
        level
            .materials
            .iter()
            .any(|m| m.reflective && m.name == name.trim())
    };
    for z in &level.overlay.zones {
        if let (true, re_core::Region::Box { min, max }) = (reflective(&z.material.0), &z.region) {
            face_ring(&mut k, *min, *max, 1, false, "mirror face");
        }
    }
    for d in &level.decals {
        let sm = &level.surface_materials;
        let bound = match d.surface_kind {
            CellKind::WallSolid => sm.wall_solid.as_deref(),
            CellKind::WallTrace => sm.wall_trace.as_deref(),
            CellKind::WallMatrix => sm.wall_matrix.as_deref(),
            _ => None,
        };
        if bound.is_some_and(reflective) {
            let name = if d.name.is_empty() {
                format!("({},{},{})-({},{},{})", d.x0, d.y0, d.z0, d.x1, d.y1, d.z1)
            } else {
                d.name.clone()
            };
            let what = format!("mirror decal {name} [{}]", bound.unwrap_or_default());
            face_ring(
                &mut k,
                (d.x0, d.y0, d.z0),
                (d.x1, d.y1, d.z1),
                1,
                false,
                &what,
            );
        }
    }
    // 10: zone inputs.
    for zi in &level.zone_inputs {
        for z in zi.z0..=zi.z1 {
            for y in zi.y0..=zi.y1 {
                for x in zi.x0..=zi.x1 {
                    k.add((x, y, z), false, format!("zone_input {}", zi.name));
                }
            }
        }
    }
    // 11: emit-part targets (a solid prop there would eat the reward).
    for c in emit_part_cells(level) {
        k.add(c, true, "emit-part target");
    }
    // 12: stair treads already in the geometry: 2-cell headroom.
    let (sx, sy, sz) = base.size();
    for z in 0..sz as i32 {
        for y in 0..sy as i32 {
            for x in 0..sx as i32 {
                if matches!(base.shape_at(x, y, z), Shape::RampZ(_))
                    && base.cell_at(x, y, z).is_solid()
                {
                    k.add((x, y, z + 1), false, "stair headroom");
                    k.add((x, y, z + 2), false, "stair headroom");
                }
            }
        }
    }
    k
}

/// The note binder's gaze quadrants: the reader looks along `-(dx, dy)` at the wall
/// (a `[note] facing` yaw column, 480 per turn), exactly as `Level::note_host_cells`.
fn note_gaze(facing: i32) -> (i32, i32) {
    match facing.rem_euclid(480) {
        60..=179 => (1, 0),
        180..=299 => (0, -1),
        300..=419 => (-1, 0),
        _ => (0, 1),
    }
}

/// A note's name for keep-out messages: its first text line, trimmed, else its index.
fn note_label(n: &crate::LevelNote, i: usize) -> String {
    let first = n
        .text
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("");
    let short: String = first.chars().take(24).collect();
    if short.is_empty() {
        format!("#{i} ({:?} at {},{},{})", n.kind, n.x, n.y, n.z)
    } else {
        format!(
            "#{i} ({:?} at {},{},{}, \"{short}\")",
            n.kind, n.x, n.y, n.z
        )
    }
}

/// Keep-out for a WALL-FACE region (a mirror, a portal opening): the face plus `pad`
/// cells straight out from it, on both sides (which side is the room is not known
/// here; the other is masonry). A region flat in x pads in x only, flat in y pads in
/// y only — so a prop standing diagonally past the face's end is not caught, only
/// one standing in front of it. A region flat in neither pads every way.
fn face_ring(k: &mut KeepOut, a: Cell3, b: Cell3, pad: i32, hard: bool, what: &str) {
    let flat_x = a.0 == b.0;
    let flat_y = a.1 == b.1;
    let (px, py) = match (flat_x, flat_y) {
        (true, false) => (pad, 0),
        (false, true) => (0, pad),
        _ => (pad, pad),
    };
    let (x0, x1) = (a.0.min(b.0) - px, a.0.max(b.0) + px);
    let (y0, y1) = (a.1.min(b.1) - py, a.1.max(b.1) + py);
    for z in a.2.min(b.2)..=a.2.max(b.2) {
        for y in y0..=y1 {
            for x in x0..=x1 {
                k.add((x, y, z), hard, what);
            }
        }
    }
}

// ───────────────────────────────── expansion ─────────────────────────────────

/// Initial (load-time) value of a power source, evaluated statically: an input
/// node's `[inputs]` activation, a pipe authored `On`, a budget over those. Hidden
/// nodes, the door and sequence steps start at 0 (their value needs the forward
/// pass, which `re_level` does not run).
fn initial_value(level: &Level, src: PowerSource) -> bool {
    let on = |p: &crate::Pipe| p.state == crate::PipeState::On;
    match src {
        PowerSource::Node(i) => level.nodes.get(i).is_some_and(|n| {
            n.is_input
                && level
                    .inputs
                    .iter()
                    .any(|(name, v)| name == &n.name && *v != 0)
        }),
        PowerSource::Pipe(i) => level.pipes.get(i).is_some_and(on),
        PowerSource::Budget(k) => level.pipes.iter().filter(|p| on(p)).count() as u32 >= k,
        PowerSource::Door | PowerSource::Seq(_) => false,
    }
}

/// The colour-law hue for a source (§3.6).
fn source_hue(level: &Level, src: PowerSource, style: PropStyle) -> u32 {
    match src {
        PowerSource::Node(i) => level
            .nodes
            .get(i)
            .map_or(pack_lit_rgb(style), |n| n.color_rgb & 0xFF_FFFF),
        PowerSource::Pipe(i) => level.pipes.get(i).map_or(pack_lit_rgb(style), |p| {
            let from = level.nodes.get(p.from_node).map(|n| n.color_rgb);
            from.unwrap_or(p.color_rgb) & 0xFF_FFFF
        }),
        PowerSource::Budget(_) => 0xFF_B040,
        PowerSource::Door | PowerSource::Seq(_) => pack_lit_rgb(style),
    }
}

/// Resolve an authored power spec against the level.
fn resolve_power(level: &Level, p: &PowerSpec) -> Result<Option<PowerSource>, String> {
    Ok(Some(match p {
        PowerSpec::None => return Ok(None),
        PowerSpec::Node(n) => PowerSource::Node(
            level
                .node_index(n)
                .ok_or_else(|| format!("unknown node '{n}'"))?,
        ),
        PowerSpec::Pipe(a, b) => PowerSource::Pipe(
            level
                .pipes
                .iter()
                .position(|p| &p.from == a && &p.to == b)
                .ok_or_else(|| format!("unknown pipe '{a}->{b}'"))?,
        ),
        PowerSpec::Door => PowerSource::Door,
        PowerSpec::Budget(k) => PowerSource::Budget(*k),
        PowerSpec::Seq(i) => {
            let steps = level.sequence.patterns.len() as u32;
            if *i > steps {
                return Err(format!(
                    "seq:{i} but the level's [sequence] has {steps} step(s)"
                ));
            }
            PowerSource::Seq(*i)
        }
    }))
}

/// The AUTHORED position of note `i` (before any prop re-homed it).
fn note_origin(level: &Level, i: usize) -> Option<(Cell3, i32)> {
    if let Some(o) = level.prop_note_origins.iter().find(|o| o.note == i) {
        return Some(((o.x, o.y, o.z), o.facing));
    }
    level.notes.get(i).map(|n| ((n.x, n.y, n.z), n.facing))
}

/// Resolve every `note = KIND`: the unclaimed note of that kind nearest (by its
/// AUTHORED position) to the prop's anchor. Ties break on the lower index, so
/// the result is deterministic and stable across the load-time move.
fn claim_notes(
    level: &Level,
    specs: &[(usize, PropSpec)],
    errors: &mut Vec<PropError>,
) -> HashMap<usize, usize> {
    let mut claims: HashMap<usize, usize> = HashMap::new();
    let mut taken: HashSet<usize> = HashSet::new();
    for (pi, spec) in specs {
        let Some(kind) = &spec.note else { continue };
        let a = spec.at_cell();
        let best = level
            .notes
            .iter()
            .enumerate()
            .filter(|(i, n)| !taken.contains(i) && note_kind_str(n.kind) == kind.as_str())
            .filter_map(|(i, _)| note_origin(level, i).map(|(p, _)| (i, p)))
            .min_by_key(|&(i, p)| {
                let d = (p.0 - a.x).pow(2) + (p.1 - a.y).pow(2) + (p.2 - a.z).pow(2);
                (d, i)
            });
        match best {
            Some((i, _)) => {
                taken.insert(i);
                claims.insert(*pi, i);
            }
            None => errors.push(PropError::new(
                spec.line_of("note"),
                format!(
                    "[prop {}] note: no unclaimed [note {kind} ...] in this level                      (known kinds: researcher, system_log, model_thought, stencil)",
                    spec.name
                ),
            )),
        }
    }
    claims
}

/// The `[note KIND]` token of a note kind.
pub fn note_kind_str(k: crate::NoteKind) -> &'static str {
    match k {
        crate::NoteKind::Researcher => "researcher",
        crate::NoteKind::SystemLog => "system_log",
        crate::NoteKind::ModelThought => "model_thought",
        crate::NoteKind::Stencil => "stencil",
    }
}

/// Expand every prop of `level` against `base` (the level built WITHOUT props).
/// Returns the expansion plus every hard error (keep-out violations, generator
/// refusals, bad references). A prop that errors contributes nothing.
///
/// The default `style` is [`Level::prop_default_style`] (set by the parser from the
/// level's own `[meta] prop_style`, default `lab`).
pub fn expand_level(level: &Level, base: &Map) -> (PropExpansion, Vec<PropError>) {
    let mut exp = PropExpansion::default();
    let mut errors = Vec::new();
    if level.props.is_empty() {
        return (exp, errors);
    }
    let default_style = level.prop_default_style;
    // Parse every prop first: the note claims must be known before the keep-out
    // set is built (a re-homed note reserves nothing).
    let mut specs: Vec<(usize, PropSpec)> = Vec::new();
    let mut seen: HashSet<&str> = HashSet::new();
    for (pi, def) in level.props.iter().enumerate() {
        if !seen.insert(def.name.as_str()) {
            errors.push(PropError::new(
                def.line,
                format!("[prop {}] duplicate prop name", def.name),
            ));
            continue;
        }
        match parse_spec(def, &level.name, default_style) {
            Ok((s, unused)) => {
                exp.unused_keys.extend(unused);
                specs.push((pi, s));
            }
            Err(e) => errors.push(e),
        }
    }
    let claims = claim_notes(level, &specs, &mut errors);
    let moved: HashSet<usize> = claims.values().copied().collect();
    let ko = keep_out(level, base, &moved);
    let mut mats = Mats::new(level);
    let mut occupied: HashMap<Cell3, usize> = HashMap::new();
    let mut shared = kinds::Shared::default();
    let note_hosts = level.note_host_cells(base);
    let cx = kinds::Ctx {
        level,
        base,
        ko: &ko,
        occ: &HashMap::new(),
        note_hosts: &note_hosts,
    };
    for (pi, spec) in &specs {
        let pi = *pi;
        for w in &spec.warnings {
            exp.issues.push(PropIssue {
                prop: spec.name.clone(),
                level: IssueLevel::Warn,
                msg: w.clone(),
                pos: None,
            });
        }
        let note = claims.get(&pi).copied();
        let cx = kinds::Ctx {
            occ: &occupied,
            ..cx
        };
        let mut b = kinds::Builder::new(&cx, spec, pi, &mut mats, &mut shared, note);
        match b.run() {
            Ok(()) => {
                let out = b.finish();
                for c in &out.cells {
                    if c.mode == CellMode::Build {
                        occupied.insert(c.pos, pi);
                    }
                }
                exp.absorb(out);
            }
            Err(e) => errors.push(e),
        }
    }
    exp.materials = mats.synth;
    exp.segments = exp
        .lines
        .iter()
        .map(|l| l.points.len().saturating_sub(1))
        .sum();
    (exp, errors)
}

impl PropExpansion {
    fn absorb(&mut self, mut o: kinds::BuiltProp) {
        let gate_base = self.gates.len();
        let emitter_base = self.emitters.len();
        let light_base = self.lights.len();
        for mut g in o.gates.drain(..) {
            for (i, _) in g.emitters.iter_mut() {
                *i += emitter_base;
            }
            for (i, _) in g.lights.iter_mut() {
                *i += light_base;
            }
            self.gates.push(g);
        }
        self.cells.extend(o.cells);
        for mut l in o.lines {
            l.gate = l.gate.map(|g| g + gate_base);
            self.lines.push(l);
        }
        for mut h in o.holos {
            h.gate = h.gate.map(|g| g + gate_base);
            self.holos.push(h);
        }
        self.decals.extend(o.decals);
        self.emitters.extend(o.emitters);
        self.lights.extend(o.lights);
        for mut p in o.parts {
            p.gate += gate_base;
            self.parts.push(p);
        }
        self.note_moves.extend(o.note_moves);
        self.issues.extend(o.issues);
        self.stair_cells.extend(o.stair_cells);
    }
}

// ───────────────────────────────── stamping ──────────────────────────────────

/// Apply an expansion to a map built WITHOUT props (the tail of
/// `Level::build_map`): cells, holograms, decals, the synthesized palette, the
/// prop strokes (with their palette ids) and the prop lamps.
pub fn stamp_expansion(level: &Level, exp: &PropExpansion, map: &mut Map) {
    // Palette: `build_map` already appended every authored material after the five
    // surface slots (it always does when the level has props); the synthesized
    // ones follow, so their 1-based ids continue the sequence.
    if !exp.materials.is_empty() {
        let mut defs = map.materials().to_vec();
        if defs.len() < 5 {
            defs.resize(5, re_world::MaterialDef::fallback());
        }
        let no_tex = HashMap::new();
        if defs.len() < 5 + level.materials.len() {
            for m in &level.materials[defs.len() - 5..] {
                defs.push(Level::material_to_def(m, &no_tex));
            }
        }
        for m in &exp.materials {
            defs.push(Level::material_to_def(m, &no_tex));
        }
        map.set_materials(defs);
    }
    let pid = |name: &Option<String>| name.as_ref().map_or(0, |n| exp.palette_id(level, n));
    for c in &exp.cells {
        let (x, y, z) = c.pos;
        if x < 0 || y < 0 || z < 0 || !map.in_bounds(x, y, z) {
            continue;
        }
        match c.mode {
            CellMode::Build => {
                if map.cell_at(x, y, z) != CellKind::Empty {
                    continue;
                }
                map.set_cell(x as usize, y as usize, z as usize, c.kind, pid(&c.mat));
                map.set_shape(x as usize, y as usize, z as usize, c.shape);
            }
            CellMode::Skin => {
                let kind = map.cell_at(x, y, z);
                map.set_cell(x as usize, y as usize, z as usize, kind, pid(&c.mat));
            }
        }
    }
    for p in &exp.parts {
        if !exp.part_initially_filled(p) {
            continue;
        }
        let (x, y, z) = p.pos;
        if x >= 0
            && y >= 0
            && z >= 0
            && map.cell_at(x, y, z) == CellKind::Empty
            && map.in_bounds(x, y, z)
        {
            let id = exp.palette_id(level, &p.mat);
            map.set_cell(x as usize, y as usize, z as usize, CellKind::WallSolid, id);
            // An R1 lit cell is light/projection, not furniture — walk-through
            // from the very first frame a gate starts lit (mirrors what
            // `re_runtime::prop_power::apply_map` does on every later flip).
            map.mark_walkthrough(x as usize, y as usize, z as usize, true);
        }
    }
    let holos: Vec<re_world::HoloObject> = exp
        .holos
        .iter()
        .filter(|h| h.visible)
        .map(PropHolo::object)
        .collect();
    if !holos.is_empty() {
        re_world::stamp_holos(map, &holos);
    }
    for d in &exp.decals {
        let id = exp.palette_id(level, &d.mat);
        for z in d.min.2..=d.max.2 {
            for y in d.min.1..=d.max.1 {
                for x in d.min.0..=d.max.0 {
                    if x < 0 || y < 0 || z < 0 || !map.in_bounds(x, y, z) {
                        continue;
                    }
                    let k = map.cell_at(x, y, z);
                    let geometry = matches!(
                        k,
                        CellKind::WallSolid
                            | CellKind::WallTrace
                            | CellKind::WallMatrix
                            | CellKind::Floor
                    );
                    if geometry && re_core::decode_pipe_tag(map.side_data_at(x, y, z)).is_none() {
                        map.set_cell(x as usize, y as usize, z as usize, k, id);
                    }
                }
            }
        }
    }
    if !exp.lines.is_empty() {
        let mut overlay = map.overlay().clone();
        let authored = overlay.lines.len();
        let mut ids: Vec<u16> = (0..authored).map(|i| map.line_palette_id(i)).collect();
        for (i, l) in exp.lines.iter().enumerate() {
            overlay.lines.push(re_core::Line {
                name: format!("{PROP_MATERIAL_PREFIX}{}_{i}", l.prop),
                path: re_core::LinePath {
                    points: l.points.clone(),
                },
                material: re_core::LineMaterialId(l.drawn_mat().to_string()),
                scale: re_core::ScaleRule::None,
            });
            ids.push(exp.palette_id(level, l.drawn_mat()));
        }
        map.set_overlay(overlay);
        map.set_line_palette_ids(ids);
    }
    if !exp.lights.is_empty() {
        let mut atmo = map.atmosphere().clone();
        atmo.lights.extend(exp.lights.iter().copied());
        map.set_atmosphere(atmo);
    }
}

// ──────────────────────────────── round trip ─────────────────────────────────

/// Remove every prop-stamped cell from an editor capture of the BUILT map, so a
/// save writes the `[prop]` sections and never bakes their output into `[cells]`.
///
/// `level` is the level as loaded (its own `cells` still prop-free): the expansion
/// is recomputed from it, and a captured cell is reverted only where it still holds
/// exactly what the prop stamped — a cell the author repainted is kept.
pub fn strip_prop_cells(level: &Level, grid: &mut crate::CellGrid) {
    if level.props.is_empty() {
        return;
    }
    let base = level.build_map_base();
    let (exp, _) = expand_level(level, &base);
    let (sx, sy, sz) = grid.size;
    let idx = |(x, y, z): Cell3| -> Option<usize> {
        (x >= 0 && y >= 0 && z >= 0 && (x as usize) < sx && (y as usize) < sy && (z as usize) < sz)
            .then(|| x as usize + sx * (y as usize + sy * z as usize))
    };
    let mut drop_ids: HashMap<Cell3, u16> = HashMap::new();
    let mut revert: HashSet<Cell3> = HashSet::new();
    let shape_of: HashMap<Cell3, Shape> = grid
        .shapes
        .iter()
        .map(|&(x, y, z, s)| ((x, y, z), s))
        .collect();
    let pid = |m: &Option<String>| m.as_ref().map_or(0, |n| exp.palette_id(level, n));
    for c in &exp.cells {
        let Some(i) = idx(c.pos) else { continue };
        match c.mode {
            CellMode::Build => {
                let shape = shape_of.get(&c.pos).copied().unwrap_or(Shape::Full);
                if grid.kinds[i] == c.kind && shape == c.shape {
                    revert.insert(c.pos);
                    drop_ids.insert(c.pos, pid(&c.mat));
                }
            }
            CellMode::Skin => {
                drop_ids.insert(c.pos, pid(&c.mat));
            }
        }
    }
    for p in &exp.parts {
        if let Some(i) = idx(p.pos) {
            if grid.kinds[i] == CellKind::WallSolid {
                revert.insert(p.pos);
                drop_ids.insert(p.pos, exp.palette_id(level, &p.mat));
            }
        }
    }
    for h in &exp.holos {
        for z in h.min.2..=h.max.2 {
            for y in h.min.1..=h.max.1 {
                for x in h.min.0..=h.max.0 {
                    if let Some(i) = idx((x, y, z)) {
                        if grid.kinds[i] == CellKind::Hologram
                            && base.cell_at(x, y, z) == CellKind::Empty
                        {
                            revert.insert((x, y, z));
                            drop_ids.insert((x, y, z), h.style);
                        }
                    }
                }
            }
        }
    }
    for d in &exp.decals {
        let id = exp.palette_id(level, &d.mat);
        for z in d.min.2..=d.max.2 {
            for y in d.min.1..=d.max.1 {
                for x in d.min.0..=d.max.0 {
                    drop_ids.insert((x, y, z), id);
                }
            }
        }
    }
    for &c in &revert {
        if let Some(i) = idx(c) {
            grid.kinds[i] = CellKind::Empty;
        }
    }
    grid.shapes
        .retain(|&(x, y, z, _)| !revert.contains(&(x, y, z)));
    grid.material_ids
        .retain(|&(x, y, z, id)| drop_ids.get(&(x, y, z)).is_none_or(|&want| want != id));
}
