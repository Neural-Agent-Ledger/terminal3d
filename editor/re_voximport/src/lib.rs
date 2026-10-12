// Test-module assert_eq! on f32 values are intentional exact-equal checks
// (round-trip serialization: portal.rotate must survive vox→lvl→vox unchanged).
// Allow float_cmp only in test compilation units; lib code remains denied.
#![cfg_attr(test, allow(clippy::float_cmp))]
#![warn(missing_docs)]
//! `re_voximport` — `.vox` + `.toml` → canonical `.lvl` converter.
//!
//! Authoring flow: sculpt a level in a voxel editor (Vengi / MagicaVoxel /
//! Goxel), export the universal `.vox` exchange format, and write a small
//! companion `.toml` for the non-spatial metadata. This crate turns that pair
//! into terminal3d's canonical `.lvl` text, after which the *unchanged*
//! [`re_level`] parser + `finalize` + `re_world::stamp_geometry` take over.
//!
//! See `docs/VOX_IMPORT.md` for the authoritative spec: the color-tag table,
//! derivation rules 1–8, and the `.toml` schema.
//!
//! ## Hybrid loader
//!
//! The [`dot_vox`] crate parses the binary `.vox` container; OUR code
//! ([`derive_level`]) derives the puzzle graph from color-tagged voxels.
//! [`vox_to_level`] adapts `dot_vox`'s data into our own small [`Voxel`] list
//! and calls the pure-data core, so the derivation logic is unit-testable with
//! NO `.vox` file and NO `dot_vox` in the test path.

use std::collections::{BTreeMap, HashSet};
use std::fmt;

use re_level::{Door, Level, LevelMaterial, Node, Pipe, PipeState, PlayerSpawn, WallDecal};
use re_world::{MaterialAnim, MaterialAnimKind, MaterialRecipe};
use serde::Deserialize;

/// One voxel: integer cell position + its true-color RGB (alpha dropped).
///
/// This is OUR struct, intentionally independent of `dot_vox::Voxel`, so the
/// derivation core and its tests never touch the `.vox` loader.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Voxel {
    /// World-space X.
    pub x: i32,
    /// World-space Y.
    pub y: i32,
    /// World-space Z.
    pub z: i32,
    /// True-color RGB (the `.vox` palette entry's colour, alpha dropped).
    pub rgb: [u8; 3],
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Everything that can go wrong importing a `.vox` + `.toml` pair.
#[derive(Clone, Debug)]
pub enum ImportError {
    /// `dot_vox` rejected the `.vox` bytes.
    VoxParse(String),
    /// The companion `.toml` failed to parse.
    TomlParse(String),
    /// A required `[meta]` field is missing.
    MissingMeta(String),
    /// A `.toml` color string was not a `#RRGGBB`.
    BadColor(String),
    /// A maximal pipe run did not terminate at exactly two node voxels.
    /// Carries the count found and the offending run's cells.
    PipeEndpoints {
        /// How many endpoint voxels the run actually terminated at.
        found: usize,
        /// The offending run's cells, for diagnostics.
        cells: Vec<(i32, i32, i32)>,
    },
    /// A `switch` voxel is not on / adjacent to exactly one pipe path.
    SwitchUnattached {
        /// The offending switch voxel's cell.
        cell: (i32, i32, i32),
        /// How many pipe paths it actually touches (never exactly 1).
        touching: usize,
    },
    /// The level the derivation produced was rejected by `re_level::finalize`
    /// (e.g. missing pedestal / door / player), or by `load_string` on a
    /// round-trip.
    Level(String),
    /// A structural derivation failure with a free-form message (e.g. a node
    /// voxel had no matching `[[node]]` and could not be defaulted).
    Derive(String),
}

impl fmt::Display for ImportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ImportError::VoxParse(m) => write!(f, "vox parse failed: {m}"),
            ImportError::TomlParse(m) => write!(f, "toml parse failed: {m}"),
            ImportError::MissingMeta(m) => write!(f, "missing required toml meta: {m}"),
            ImportError::BadColor(m) => write!(f, "bad color: {m}"),
            ImportError::PipeEndpoints { found, cells } => write!(
                f,
                "pipe run must end at exactly two nodes, found {found}; cells {cells:?}"
            ),
            ImportError::SwitchUnattached { cell, touching } => write!(
                f,
                "switch voxel {cell:?} touches {touching} pipe paths (need exactly 1)"
            ),
            ImportError::Level(m) => write!(f, "level finalize failed: {m}"),
            ImportError::Derive(m) => write!(f, "derivation failed: {m}"),
        }
    }
}

impl std::error::Error for ImportError {}

// ---------------------------------------------------------------------------
// Companion .toml schema
// ---------------------------------------------------------------------------

/// The full parsed companion `.toml`. Every table beyond `[meta]` is optional;
/// an absent table simply omits its `.lvl` section.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct Config {
    /// `[meta]` — the one required table.
    pub meta: Meta,
    /// Optional tag→RGB overrides (hex strings like `"#8000FF"`).
    #[serde(default)]
    pub tags: BTreeMap<String, String>,
    /// Optional explicit room size override `[room] size`.
    #[serde(default)]
    pub room: Option<RoomCfg>,
    /// `[[node]]` metadata entries, keyed to node voxels by position.
    #[serde(default, rename = "node")]
    pub nodes: Vec<NodeCfg>,
    /// `[[edge]]` pipe-direction disambiguation entries.
    #[serde(default, rename = "edge")]
    pub edges: Vec<EdgeCfg>,
    /// `[inputs]` initial activation per node name.
    #[serde(default)]
    pub inputs: BTreeMap<String, i32>,
    /// `[effects]` overrides.
    #[serde(default)]
    pub effects: Option<EffectsCfg>,
    /// `[palette]` fog overrides.
    #[serde(default)]
    pub palette: Option<PaletteCfg>,
    /// `[[material]]` authored material set (the "color = material KEY" pipeline).
    #[serde(default, rename = "material")]
    pub materials: Vec<MaterialCfg>,
    /// `[surface]` table: maps a surface kind → an authored material NAME, e.g.
    /// `floor = "floor_neon"`. The bound material is emitted into the `.lvl` and
    /// the surface tagged with it. Keys: `wall_solid`/`wall_trace`/`wall_matrix`/
    /// `floor`/`ceiling`.
    #[serde(default)]
    pub surface: BTreeMap<String, String>,
    /// `[[color_material]]` — per-voxel-colour material overrides (variant-3
    /// colour-keyed materials). Each entry maps a palette colour (`color =
    /// "#RRGGBB"`) to a full material definition; voxels whose colour matches
    /// (nearest within `color_material_tolerance`) carry that material's id in
    /// their per-cell `side_data` channel, enabling per-voxel shader dispatch.
    /// The material name is auto-generated as `cm_RRGGBB` when `name` is absent.
    #[serde(default, rename = "color_material")]
    pub color_materials: Vec<ColorMaterialCfg>,
    /// Per-channel Euclidean² tolerance for `[[color_material]]` colour matching.
    /// Defaults to [`COLOR_MATERIAL_TOLERANCE`] when absent (≈ same as tag
    /// tolerance, tight enough to avoid cross-palette collisions). Squared
    /// Euclidean: `dr²+dg²+db²` vs this value.
    #[serde(default)]
    pub color_material_tolerance: Option<i32>,
}

/// One `[[color_material]]` — a per-palette-colour material override.
///
/// `color` is the palette RGB (`"#RRGGBB"`) the converter matches against each
/// voxel. The remaining fields are identical to [`MaterialCfg`]; the material
/// `name` defaults to `cm_RRGGBB` (auto-derived from the hex colour) when
/// absent. An auto-name is deterministic and unique per colour — two entries
/// with different colours never share a name.
///
/// The material is emitted into the `.lvl` as a `[material cm_RRGGBB]` section
/// (reusing the existing `re_level` `[material]` parser) and a `[color_cell_ids]`
/// section carries per-cell `x,y,z = id` assignments (id = 1-based index into
/// the `[[color_material]]` list). `re_level` parses `[color_cell_ids]` and
/// stamps each id into its cell's `side_data`, and `re_render` dispatches a
/// non-zero id through the per-voxel material override instead of the
/// surface-kind default.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct ColorMaterialCfg {
    /// The palette colour this entry matches, as `"#RRGGBB"`.
    pub color: String,
    /// Optional explicit name for the emitted `[material NAME]` section.
    /// Defaults to `cm_RRGGBB` (auto from the hex colour).
    #[serde(default)]
    pub name: Option<String>,
    /// Foreground colour, `#RRGGBB`.
    #[serde(default)]
    pub fg: Option<String>,
    /// Background colour, `#RRGGBB`.
    #[serde(default)]
    pub bg: Option<String>,
    /// A single fixed glyph override.
    #[serde(default)]
    pub glyph: Option<String>,
    /// Pattern recipe (`"brick"|"circuit"|"grain"|"sky"`) with optional ` k=v` params.
    #[serde(default)]
    pub recipe: Option<String>,
    /// Optional animation applied to this material.
    #[serde(default)]
    pub animation: Option<AnimationCfg>,
    /// LOD rank bias.
    #[serde(default)]
    pub rank: Option<u8>,
}

/// One `[[material]]` — an authored material the converter emits into the `.lvl`
/// `[material NAME]` section. `fg`/`bg` are `#RRGGBB`; `glyph` is a single char;
/// `recipe` is `"brick"|"circuit"|"grain"|"sky"` with optional ` k=v` params;
/// `animation` is an optional `{ kind, speed, amplitude }`.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct MaterialCfg {
    /// The material's name, bound to surfaces by `[material NAME]`.
    pub name: String,
    /// Foreground colour, `#RRGGBB`.
    #[serde(default)]
    pub fg: Option<String>,
    /// Background colour, `#RRGGBB`.
    #[serde(default)]
    pub bg: Option<String>,
    /// A single fixed glyph override.
    #[serde(default)]
    pub glyph: Option<String>,
    /// Pattern recipe (`"brick"|"circuit"|"grain"|"sky"`) with optional ` k=v` params.
    #[serde(default)]
    pub recipe: Option<String>,
    /// Optional animation applied to this material.
    #[serde(default)]
    pub animation: Option<AnimationCfg>,
    /// LOD rank bias.
    #[serde(default)]
    pub rank: Option<u8>,
}

/// `[[material]].animation` — `kind` is `pulse`/`shift`/`flicker`/`none`.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct AnimationCfg {
    /// `"pulse"|"shift"|"flicker"|"none"`.
    #[serde(default)]
    pub kind: String,
    /// Cycles per second (animation-kind dependent).
    #[serde(default)]
    pub speed: Option<f32>,
    /// Effect strength (animation-kind dependent).
    #[serde(default)]
    pub amplitude: Option<f32>,
}

/// `[meta]` — the one required table. `name` is mandatory.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct Meta {
    /// The level's `[meta] name`.
    #[serde(default)]
    pub name: String,
    /// The level's `[meta] title`.
    #[serde(default)]
    pub title: String,
    /// → `[room] ambient`.
    #[serde(default)]
    pub ambient: String,
}

/// Optional `[room]` overrides.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct RoomCfg {
    /// `size = [sx, sy, sz]`.
    #[serde(default)]
    pub size: Option<[i32; 3]>,
}

/// One `[[node]]` — metadata for the node voxel at `pos`.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct NodeCfg {
    /// The node voxel's cell position this entry describes.
    pub pos: [i32; 3],
    /// The node's name.
    #[serde(default)]
    pub name: String,
    /// A single fixed glyph override.
    #[serde(default)]
    pub glyph: Option<String>,
    /// Candidate display words this node may draw its label from.
    #[serde(default)]
    pub pool: Vec<String>,
    /// Alternate-language candidate display words.
    #[serde(default)]
    pub pool_alt: Vec<String>,
    /// Display colour, `#RRGGBB`.
    #[serde(default)]
    pub color: Option<String>,
    /// Activation threshold (hidden/output nodes only).
    #[serde(default)]
    pub threshold: Option<i32>,
    /// Override the tag-inferred role.
    #[serde(default)]
    pub is_input: Option<bool>,
    /// Override the tag-inferred role.
    #[serde(default)]
    pub is_output: Option<bool>,
}

/// One `[[edge]]` — optional pipe-direction disambiguation by node name.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct EdgeCfg {
    /// The pipe's source node name.
    pub from: String,
    /// The pipe's destination node name.
    pub to: String,
}

/// `[effects]`.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct EffectsCfg {
    /// Snowfall density, `0.0..=1.0`.
    #[serde(default)]
    pub snow: Option<f32>,
    /// Whether ambient steam is on.
    #[serde(default)]
    pub steam: Option<bool>,
    /// Matrix-rain bleed density, `0.0..=1.0`.
    #[serde(default)]
    pub matrix: Option<f32>,
}

/// `[palette]` fog override.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct PaletteCfg {
    /// Fog colour, `#RRGGBB`.
    #[serde(default)]
    pub fog_color: Option<String>,
    /// Distance at which fog begins.
    #[serde(default)]
    pub fog_near: Option<f32>,
    /// Distance at which fog fully occludes.
    #[serde(default)]
    pub fog_far: Option<f32>,
}

// ---------------------------------------------------------------------------
// Tags
// ---------------------------------------------------------------------------

/// The semantic tags a voxel colour can map to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tag {
    /// A plain solid wall cell.
    WallSolid,
    /// A wall cell with the circuit/trace decal.
    WallTrace,
    /// A wall cell with the matrix-rain decal.
    WallMatrix,
    /// A floor cell.
    Floor,
    /// A ceiling cell.
    Ceiling,
    /// A puzzle input node.
    NodeInput,
    /// A puzzle output node.
    NodeOutput,
    /// A puzzle hidden node.
    NodeHidden,
    /// A pipe path cell.
    Pipe,
    /// A pipe's switch cell.
    Switch,
    /// The level's exit door.
    Door,
    /// The player spawn marker.
    Player,
    /// The reward pedestal volume.
    Pedestal,
    /// A transparent hologram cell (`CellKind::Hologram`). The DDA passes
    /// through it; the renderer draws a glowing wireframe on the silhouette.
    Hologram,
    /// A non-euclidean warp-portal face (`CellKind::Portal`). The DDA stops
    /// on it; the renderer re-bases the ray through the portal's delta.
    Portal,
    /// A generic solid object cell (`CellKind::Object`). The DDA stops on
    /// it; game-defined `side_data` carries the object's identity tag.
    Object,
}

impl Tag {
    /// The `.lvl`/`.toml` string key used to name this tag in authored data
    /// (e.g. `"wall_solid"`, `"node_input"`).
    pub fn key(self) -> &'static str {
        match self {
            Tag::WallSolid => "wall_solid",
            Tag::WallTrace => "wall_trace",
            Tag::WallMatrix => "wall_matrix",
            Tag::Floor => "floor",
            Tag::Ceiling => "ceiling",
            Tag::NodeInput => "node_input",
            Tag::NodeOutput => "node_output",
            Tag::NodeHidden => "node_hidden",
            Tag::Pipe => "pipe",
            Tag::Switch => "switch",
            Tag::Door => "door",
            Tag::Player => "player",
            Tag::Pedestal => "pedestal",
            Tag::Hologram => "hologram",
            Tag::Portal => "portal",
            Tag::Object => "object",
        }
    }

    /// This tag's default palette colour, used when the imported `.vox`/`.toml`
    /// carries no explicit colour override for a cell with this tag.
    pub fn default_rgb(self) -> [u8; 3] {
        match self {
            Tag::WallSolid => [0x80, 0x80, 0x80],
            Tag::WallTrace => [0x00, 0xA0, 0xFF],
            Tag::WallMatrix => [0x00, 0xFF, 0x50],
            Tag::Floor => [0x40, 0x40, 0x40],
            Tag::Ceiling => [0x20, 0x20, 0x20],
            Tag::NodeInput => [0xFF, 0xD0, 0x00],
            Tag::NodeOutput => [0xFF, 0x30, 0x30],
            Tag::NodeHidden => [0xFF, 0xFF, 0xFF],
            Tag::Pipe => [0x80, 0x00, 0xFF],
            Tag::Switch => [0xFF, 0x00, 0xFF],
            Tag::Door => [0x00, 0xFF, 0xFF],
            Tag::Player => [0xFF, 0xA5, 0x00],
            Tag::Pedestal => [0x60, 0x60, 0x60],
            // New tags — chosen to be > √36 Euclidean distance from every
            // existing tag colour so the nearest-tag matcher never ambiguates.
            Tag::Hologram => [0x40, 0xFF, 0xFF], // cyan-blue; far from door #00FFFF (√4096)
            Tag::Portal => [0xFF, 0x40, 0xC0],   // magenta-rose; far from switch #FF00FF (√8065)
            Tag::Object => [0xC0, 0x60, 0x00],   // burnt orange; far from all existing
        }
    }

    /// Map a [`Tag`] to its 0-based index in [`ALL_TAGS`].
    /// Matches the order exported in the vox palette by [`default_vox_palette`].
    #[inline]
    pub fn palette_index(self) -> u8 {
        ALL_TAGS.iter().position(|&t| t == self).unwrap_or(0) as u8
    }

    /// The [`re_world::CellKind`] this tag denotes in the voxel map.
    ///
    /// Returns `None` for non-geometry tags (graph elements — nodes, pipes,
    /// switches, door, player, pedestal) whose spatial role is handled by
    /// derivation rules 4–7 rather than by direct cell stamping.
    pub fn cell_kind(self) -> Option<re_world::CellKind> {
        match self {
            Tag::WallSolid => Some(re_world::CellKind::WallSolid),
            Tag::WallTrace => Some(re_world::CellKind::WallTrace),
            Tag::WallMatrix => Some(re_world::CellKind::WallMatrix),
            Tag::Floor => Some(re_world::CellKind::Floor),
            Tag::Ceiling => Some(re_world::CellKind::Ceiling),
            Tag::Hologram => Some(re_world::CellKind::Hologram),
            Tag::Portal => Some(re_world::CellKind::Portal),
            Tag::Object => Some(re_world::CellKind::Object),
            // Graph elements — no direct cell-kind mapping.
            Tag::NodeInput
            | Tag::NodeOutput
            | Tag::NodeHidden
            | Tag::Pipe
            | Tag::Switch
            | Tag::Door
            | Tag::Player
            | Tag::Pedestal => None,
        }
    }
}

/// All tags in canonical palette-index order.  The index of each entry in this
/// array is its palette index in the `.vox` file produced by
/// [`default_vox_palette`] and [`level_to_vox_bytes`].
///
/// **Append-only contract**: never reorder or remove entries — the palette
/// index is stable across all `.vox` files the converter has ever written.
/// New tags are appended at the end.
pub const ALL_TAGS: [Tag; 16] = [
    // indices 0–12  (original 13 tags — order frozen)
    Tag::WallSolid,
    Tag::WallTrace,
    Tag::WallMatrix,
    Tag::Floor,
    Tag::Ceiling,
    Tag::NodeInput,
    Tag::NodeOutput,
    Tag::NodeHidden,
    Tag::Pipe,
    Tag::Switch,
    Tag::Door,
    Tag::Player,
    Tag::Pedestal,
    // indices 13–15 (new dedicated geometry tags)
    Tag::Hologram,
    Tag::Portal,
    Tag::Object,
];

// ---------------------------------------------------------------------------
// Public CellKind ↔ palette-index mapping (the single source of truth)
// ---------------------------------------------------------------------------

/// Map a [`re_world::CellKind`] (and the cell's `side_data` tag) to a
/// **0-based `.vox` palette index** compatible with [`default_vox_palette`].
///
/// This is the **single, authoritative forward mapping** — callers such as
/// [`level_to_vox_bytes`] use it instead of duplicating the table.
///
/// Returns `None` for [`re_world::CellKind::Empty`] (empty cells are not
/// stamped into a `.vox` model).
///
/// `side_data` distinguishes `Object` cells: a non-zero `side_data` value
/// means a node was painted here (saved as `node_hidden`, palette index 7,
/// so `vox_to_level` re-derives it as a hidden node).  A zero `side_data`
/// generic object uses the dedicated `object` tag (palette index 15).
pub fn cell_kind_to_palette_index(kind: re_world::CellKind, side_data: u16) -> Option<u8> {
    match kind {
        re_world::CellKind::Empty => None,
        re_world::CellKind::WallSolid => Some(Tag::WallSolid.palette_index()),
        re_world::CellKind::WallTrace => Some(Tag::WallTrace.palette_index()),
        re_world::CellKind::WallMatrix => Some(Tag::WallMatrix.palette_index()),
        re_world::CellKind::Floor => Some(Tag::Floor.palette_index()),
        re_world::CellKind::Ceiling => Some(Tag::Ceiling.palette_index()),
        re_world::CellKind::Hologram => Some(Tag::Hologram.palette_index()),
        re_world::CellKind::Portal => Some(Tag::Portal.palette_index()),
        re_world::CellKind::Object => {
            // A non-zero side_data = a painted node → save as node_hidden so
            // vox_to_level re-derives it as a hidden node on round-trip.
            if side_data != 0 {
                Some(Tag::NodeHidden.palette_index())
            } else {
                Some(Tag::Object.palette_index())
            }
        }
    }
}

/// Map a [`Tag`] (from the `.vox` palette) back to the [`re_world::CellKind`]
/// it represents in the voxel map.
///
/// This is the **single, authoritative inverse mapping** — [`vox_to_level`]'s
/// derivation calls [`Tag::cell_kind`] directly, but this free function is
/// provided as a convenience alias for callers that have a `Tag` in hand.
///
/// Returns `None` for graph-element tags (nodes, pipes, switches, door, player,
/// pedestal) which are handled by derivation rules 4–7, not by cell stamping.
#[inline]
pub fn tag_to_cell_kind(tag: Tag) -> Option<re_world::CellKind> {
    tag.cell_kind()
}

/// Default RGB tolerance (per-channel Euclidean) for tag matching. Exact match
/// is distance 0; a few units absorbs editor rounding without colliding tags.
const TAG_TOLERANCE: i32 = 6;

/// Default squared-Euclidean tolerance for `[[color_material]]` colour matching.
/// Tighter than the tag tolerance because colour-materials are user-defined and
/// can be chosen to sit well apart; 36 = 6² ≈ ±2 per channel headroom.
pub const COLOR_MATERIAL_TOLERANCE: i32 = 36;

/// The resolved tag→RGB table for one import (defaults + `.toml` overrides).
struct TagTable {
    rgb: [[u8; 3]; 16],
}

impl TagTable {
    fn from_config(cfg: &Config) -> Result<TagTable, ImportError> {
        let mut rgb = [[0u8; 3]; 16];
        for (i, t) in ALL_TAGS.iter().enumerate() {
            rgb[i] = t.default_rgb();
        }
        for (k, v) in &cfg.tags {
            if let Some(i) = ALL_TAGS.iter().position(|t| t.key() == k) {
                rgb[i] = parse_hex_rgb(v)?;
            }
        }
        Ok(TagTable { rgb })
    }

    /// Nearest tag within [`TAG_TOLERANCE`]; `None` for an unmatched colour.
    fn match_tag(&self, rgb: [u8; 3]) -> Option<Tag> {
        let mut best: Option<(i32, Tag)> = None;
        for (i, t) in ALL_TAGS.iter().enumerate() {
            let d = rgb_dist2(rgb, self.rgb[i]);
            if best.is_none_or(|(bd, _)| d < bd) {
                best = Some((d, *t));
            }
        }
        match best {
            Some((d, t)) if d <= TAG_TOLERANCE * TAG_TOLERANCE => Some(t),
            _ => None,
        }
    }
}

#[inline]
fn rgb_dist2(a: [u8; 3], b: [u8; 3]) -> i32 {
    let dr = a[0] as i32 - b[0] as i32;
    let dg = a[1] as i32 - b[1] as i32;
    let db = a[2] as i32 - b[2] as i32;
    dr * dr + dg * dg + db * db
}

/// Parse `"#RRGGBB"` → `[r, g, b]`.
fn parse_hex_rgb(s: &str) -> Result<[u8; 3], ImportError> {
    let h = s.trim().strip_prefix('#').unwrap_or(s.trim());
    if h.len() < 6 {
        return Err(ImportError::BadColor(s.to_string()));
    }
    let r = u8::from_str_radix(&h[0..2], 16).map_err(|_| ImportError::BadColor(s.to_string()))?;
    let g = u8::from_str_radix(&h[2..4], 16).map_err(|_| ImportError::BadColor(s.to_string()))?;
    let b = u8::from_str_radix(&h[4..6], 16).map_err(|_| ImportError::BadColor(s.to_string()))?;
    Ok([r, g, b])
}

/// Pack `[r,g,b]` into the `0x00RRGGBB` form `re_level` uses.
#[inline]
fn pack_rgb(rgb: [u8; 3]) -> u32 {
    ((rgb[0] as u32) << 16) | ((rgb[1] as u32) << 8) | rgb[2] as u32
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// The result of [`read_solid_voxels`]: the deduplicated solid cell list plus the
/// post-scale bounding extent `(ex, ey, ez)` (max coord + 1 per axis).
pub type SolidVoxels = (Vec<(i32, i32, i32)>, (i32, i32, i32));

/// The full output of [`derive_level_full`]: a [`Level`] plus the per-cell
/// material-id assignments produced by `[[color_material]]` matching.
///
/// `cell_material_ids` maps each voxel grid position `(x, y, z)` to a **1-based
/// material id** — the index into the `[[color_material]]` list in the `.toml`
/// (so id `1` is the first `[[color_material]]` entry). Id `0` means "no
/// colour-material match; use the surface-kind default". A level with no
/// `[[color_material]]` entries always yields an empty map.
///
/// The id is the per-cell channel the renderer uses for per-voxel dispatch:
/// `re_level` parses `[color_cell_ids]` into `Level::color_cell_ids` and
/// `build_map` stamps each id into the matching cell's `side_data`
/// (`Map::side_data_at`); `re_render` then resolves a non-zero id through
/// `MaterialRegistry::color_material` instead of the surface-kind default.
#[derive(Clone, Debug, Default)]
pub struct DerivedLevel {
    /// The imported level's geometry, nodes, edges and metadata.
    pub level: Level,
    /// Per-cell colour-material id (1-based; 0 = no override).
    pub cell_material_ids: BTreeMap<(i32, i32, i32), u16>,
}

/// Read the SOLID voxel cells of a `.vox` byte blob, with an optional integer
/// `scale` downscale, as a deduplicated list of `(x, y, z)` cells (z-up, matching
/// our world's coordinate convention — `.vox` is already right-handed Z-up).
///
/// This is the shared, reusable `.vox` → solid-cells extractor: it flattens
/// EVERY model's voxels (palette index → solidity; a `.vox` voxel is solid by
/// definition — it occupies its cell) and, when `scale > 1`, keeps every `scale`-th
/// voxel on each axis and divides the coordinate, so a large building shrinks to a
/// sane footprint (`scale = 4` ⇒ every 4th voxel, ¼ the extent per axis). The
/// result is the building's silhouette as integer cells, ready to be stamped as
/// solid wall cells by a level's `[vox]` section.
///
/// It uses `dot_vox::load_bytes` — the SAME loader as [`vox_to_level`] — so no
/// `.vox` parsing is reimplemented. Returns the cells AND the post-scale bounding
/// extent `(ex, ey, ez)` (max coord + 1 per axis), so a caller can size/centre the
/// stamp. An empty model yields an empty list and a `(0, 0, 0)` extent.
pub fn read_solid_voxels(vox_bytes: &[u8], scale: u32) -> Result<SolidVoxels, ImportError> {
    let data = dot_vox::load_bytes(vox_bytes).map_err(|e| ImportError::VoxParse(e.to_string()))?;
    let scale = scale.max(1);
    let mut seen: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut cells: Vec<(i32, i32, i32)> = Vec::new();
    let (mut ex, mut ey, mut ez) = (0i32, 0i32, 0i32);
    for model in &data.models {
        for v in &model.voxels {
            // Decimate: keep only voxels on the scale lattice, then divide so the
            // kept cells pack contiguously (no gaps) at the smaller extent.
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
                ex = ex.max(cell.0 + 1);
                ey = ey.max(cell.1 + 1);
                ez = ez.max(cell.2 + 1);
                cells.push(cell);
            }
        }
    }
    Ok((cells, (ex, ey, ez)))
}

/// Parse a `.vox` byte blob + a companion `.toml` string into a [`Level`].
///
/// `dot_vox` parses the container; we flatten its first model's voxels (palette
/// index → RGB) into our [`Voxel`] list and hand off to [`derive_level`].
pub fn vox_to_level(vox_bytes: &[u8], toml_str: &str) -> Result<Level, ImportError> {
    vox_to_level_full(vox_bytes, toml_str).map(|dr| dr.level)
}

/// Like [`vox_to_level`] but also returns the per-cell colour-material ids
/// produced by any `[[color_material]]` entries in the `.toml`.
///
/// The `cell_material_ids` map in the returned [`DerivedLevel`] carries the
/// 1-based id for every voxel position that matched a `[[color_material]]`
/// entry. Use [`to_lvl_text_with_ids`] to serialise the result to a `.lvl`
/// that includes the `[color_cell_ids]` section.
pub fn vox_to_level_full(vox_bytes: &[u8], toml_str: &str) -> Result<DerivedLevel, ImportError> {
    let data = dot_vox::load_bytes(vox_bytes).map_err(|e| ImportError::VoxParse(e.to_string()))?;
    let cfg: Config =
        toml::from_str(toml_str).map_err(|e| ImportError::TomlParse(e.to_string()))?;

    if cfg.meta.name.trim().is_empty() {
        return Err(ImportError::MissingMeta("[meta] name".into()));
    }

    // Flatten every model's voxels. dot_vox `Voxel.i` indexes `palette`.
    let mut voxels: Vec<Voxel> = Vec::new();
    let mut size = (0u32, 0u32, 0u32);
    for model in &data.models {
        size.0 = size.0.max(model.size.x);
        size.1 = size.1.max(model.size.y);
        size.2 = size.2.max(model.size.z);
        for v in &model.voxels {
            let c = data
                .palette
                .get(v.i as usize)
                .copied()
                .unwrap_or(dot_vox::Color {
                    r: 0,
                    g: 0,
                    b: 0,
                    a: 255,
                });
            voxels.push(Voxel {
                x: v.x as i32,
                y: v.y as i32,
                z: v.z as i32,
                rgb: [c.r, c.g, c.b],
            });
        }
    }

    derive_level_full(&voxels, size, &cfg)
}

/// The core, engine-side derivation: color-tagged voxels → a [`Level`].
///
/// Implements derivation rules 1–8 from `docs/VOX_IMPORT.md`. Pure data in,
/// pure data out — no `dot_vox`, no I/O — so tests build voxel lists by hand.
///
/// Discards the per-cell colour-material id map (if any `[[color_material]]`
/// entries were authored). Use [`derive_level_full`] to retain it.
pub fn derive_level(
    voxels: &[Voxel],
    size: (u32, u32, u32),
    cfg: &Config,
) -> Result<Level, ImportError> {
    derive_level_full(voxels, size, cfg).map(|dr| dr.level)
}

/// Like [`derive_level`] but returns a [`DerivedLevel`] that also carries the
/// per-cell colour-material id map built from any `[[color_material]]` entries.
///
/// ## Rule 9 — `[[color_material]]` per-voxel material assignment
///
/// Each `[[color_material]]` entry has a `color = "#RRGGBB"` key identifying a
/// palette colour and the usual material fields (`fg`, `bg`, `glyph`, `recipe`,
/// `animation`, `rank`). During derivation every geometry voxel (one whose tag
/// has a `cell_kind`) is tested against the colour-material table (nearest-RGB
/// within `color_material_tolerance`). A matching voxel records a **1-based
/// material id** (index into `cfg.color_materials`) in
/// [`DerivedLevel::cell_material_ids`] at its grid position.
///
/// The colour-material entries are emitted into the `.lvl` as standard
/// `[material cm_RRGGBB]` sections (using existing `re_level` support) and the
/// per-cell id assignments are emitted as a `[color_cell_ids]` section. See
/// [`to_lvl_text_with_ids`].
pub fn derive_level_full(
    voxels: &[Voxel],
    size: (u32, u32, u32),
    cfg: &Config,
) -> Result<DerivedLevel, ImportError> {
    if cfg.meta.name.trim().is_empty() {
        return Err(ImportError::MissingMeta("[meta] name".into()));
    }
    let tags = TagTable::from_config(cfg)?;

    // --- classify every voxel -------------------------------------------------
    // Rule: unknown colour → wall_solid + warning (not an error).
    let mut tagged: Vec<(Voxel, Tag)> = Vec::with_capacity(voxels.len());
    let mut unknown: Vec<[u8; 3]> = Vec::new();
    for &v in voxels {
        match tags.match_tag(v.rgb) {
            Some(t) => tagged.push((v, t)),
            None => {
                tagged.push((v, Tag::WallSolid));
                if !unknown.contains(&v.rgb) {
                    unknown.push(v.rgb);
                }
            }
        }
    }
    if !unknown.is_empty() {
        eprintln!(
            "re_voximport: warning: {} unknown voxel colour(s) treated as wall_solid: {}",
            unknown.len(),
            unknown
                .iter()
                .map(|c| format!("#{:02X}{:02X}{:02X}", c[0], c[1], c[2]))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    // --- Rule 1: room size = bbox(max+1) unless overridden -------------------
    let size = if let Some(rc) = &cfg.room {
        if let Some(s) = rc.size {
            (s[0].max(1) as u32, s[1].max(1) as u32, s[2].max(1) as u32)
        } else {
            bbox_size(voxels, size)
        }
    } else {
        bbox_size(voxels, size)
    };
    let (sx, sy, sz) = (size.0 as i32, size.1 as i32, size.2 as i32);

    let mut level = Level {
        name: cfg.meta.name.clone(),
        title: cfg.meta.title.clone(),
        size: (size.0 as usize, size.1 as usize, size.2 as usize),
        ambient: cfg.meta.ambient.clone(),
        ..Default::default()
    };

    // --- Rule 4: nodes -------------------------------------------------------
    // Each node_* voxel → one [node]. Position keys the [[node]] toml row.
    let mut node_voxels: Vec<(i32, i32, i32)> = Vec::new();
    for &(v, t) in &tagged {
        let role = match t {
            Tag::NodeInput => Some((true, false)),
            Tag::NodeOutput => Some((false, true)),
            Tag::NodeHidden => Some((false, false)),
            _ => None,
        };
        let Some((mut is_in, mut is_out)) = role else {
            continue;
        };
        let pos = (v.x, v.y, v.z);
        let meta = cfg.nodes.iter().find(|n| n.pos == [v.x, v.y, v.z]);
        if let Some(m) = meta {
            if let Some(b) = m.is_input {
                is_in = b;
            }
            if let Some(b) = m.is_output {
                is_out = b;
            }
        }
        let name = match meta.map(|m| m.name.as_str()) {
            Some(n) if !n.is_empty() => n.to_string(),
            _ => format!("n_{}_{}_{}", v.x, v.y, v.z),
        };
        let glyph = meta
            .and_then(|m| m.glyph.as_ref())
            .and_then(|g| g.chars().next())
            .unwrap_or_else(|| name.chars().next().unwrap_or('?').to_ascii_uppercase());
        let color_rgb = match meta.and_then(|m| m.color.as_ref()) {
            Some(c) => pack_rgb(parse_hex_rgb(c)?),
            None => pack_rgb(v.rgb),
        };
        level.nodes.push(Node {
            name,
            x: v.x,
            y: v.y,
            z: v.z,
            glyph,
            pool: meta.map(|m| m.pool.clone()).unwrap_or_default(),
            pool_alt: meta.map(|m| m.pool_alt.clone()).unwrap_or_default(),
            // Anchor word index for the word→story readout (re_level::Node).
            // voximport authors no override → the default 0 (pool's first word).
            anchor: 0,
            color_rgb,
            is_input: is_in,
            is_output: is_out,
            threshold: meta.and_then(|m| m.threshold).unwrap_or(0),
        });
        node_voxels.push(pos);
    }

    // Lookup: cell → node index (only node voxels).
    let node_at: BTreeMap<(i32, i32, i32), usize> = node_voxels
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, i))
        .collect();

    // --- Rule 3: decals — connected boxes of wall_trace / wall_matrix --------
    derive_decals(&tagged, &mut level);

    // --- Rule 5: pipes — maximal 6-connected runs of pipe voxels -------------
    let pipe_cells: HashSet<(i32, i32, i32)> = tagged
        .iter()
        .filter(|(_, t)| *t == Tag::Pipe)
        .map(|(v, _)| (v.x, v.y, v.z))
        .collect();
    let node_list = level.nodes.clone();
    derive_pipes(&pipe_cells, &node_at, &node_list, &cfg.edges, &mut level)?;

    // --- Rule 6: switches map onto pipe paths --------------------------------
    let switch_cells: Vec<(i32, i32, i32)> = tagged
        .iter()
        .filter(|(_, t)| *t == Tag::Switch)
        .map(|(v, _)| (v.x, v.y, v.z))
        .collect();
    derive_switches(&switch_cells, &mut level)?;

    // --- Rule 7: door / player / pedestal ------------------------------------
    let mut door_cell: Option<(i32, i32, i32)> = None;
    let mut player_cell: Option<(i32, i32, i32)> = None;
    let mut pedestal_cells: Vec<(i32, i32, i32)> = Vec::new();
    for &(v, t) in &tagged {
        match t {
            Tag::Door => door_cell = Some((v.x, v.y, v.z)),
            Tag::Player => player_cell = Some((v.x, v.y, v.z)),
            Tag::Pedestal => pedestal_cells.push((v.x, v.y, v.z)),
            _ => {}
        }
    }

    if let Some((x, y, z)) = door_cell {
        level.door = Door {
            x,
            y,
            z,
            ..Default::default()
        };
    } else {
        // Default: far +x interior wall midpoint, on the lowest air layer.
        level.door = Door {
            x: (sx - 1).max(0),
            y: (sy / 2).clamp(0, sy - 1),
            z: 1.min(sz - 1),
            ..Default::default()
        };
    }

    if let Some((x, y, z)) = player_cell {
        level.player = PlayerSpawn {
            x,
            y,
            z,
            facing_yaw_col: 0,
        };
    } else {
        level.player = PlayerSpawn {
            x: (sx / 2).clamp(0, sx - 1),
            y: (sy / 2).clamp(0, sy - 1),
            z: 1.min(sz - 1),
            facing_yaw_col: 0,
        };
    }

    // Pedestal: bounding box of pedestal voxels, else a 1-cell box under the door.
    if pedestal_cells.is_empty() {
        level.pedestal = re_level::Pedestal {
            cx: level.door.x,
            cy: level.door.y,
            cz: 1.min(sz - 1),
            w: 1,
            h: 1,
            d: 1,
        };
    } else {
        let (mn, mx) = bbox(&pedestal_cells);
        level.pedestal = re_level::Pedestal {
            cx: (mn.0 + mx.0) / 2,
            cy: (mn.1 + mx.1) / 2,
            cz: (mn.2 + mx.2) / 2,
            w: mx.0 - mn.0 + 1,
            h: mx.1 - mn.1 + 1,
            d: mx.2 - mn.2 + 1,
        };
    }

    // --- Rule 8: non-spatial data straight from the toml ---------------------
    for (name, val) in &cfg.inputs {
        level.inputs.push((name.clone(), *val));
    }
    if let Some(e) = &cfg.effects {
        if let Some(s) = e.snow {
            level.effects.snow = s.clamp(0.0, 1.0);
        }
        if let Some(s) = e.steam {
            level.effects.steam = s;
        }
        if let Some(m) = e.matrix {
            level.effects.matrix = m.clamp(0.0, 1.0);
        }
    }
    if let Some(p) = &cfg.palette {
        if let Some(c) = &p.fog_color {
            level.palette.fog_color = pack_rgb(parse_hex_rgb(c)?);
            level.palette.has_fog = true;
        }
        if let Some(n) = p.fog_near {
            level.palette.fog_near = n;
            level.palette.has_fog_dist = true;
        }
        if let Some(fa) = p.fog_far {
            level.palette.fog_far = fa;
            level.palette.has_fog_dist = true;
        }
    }

    // --- Materials: the "color = material KEY" payload --------------------
    // Each [[material]] becomes a [material NAME] in the .lvl; the [surface]
    // table binds a surface kind to a material name. The bound material set
    // round-trips through re_level::load and reaches the renderer via build_map.
    derive_materials(cfg, &mut level)?;

    // --- Rule 9: [[color_material]] — per-voxel colour-keyed materials --------
    // Build the colour-material table (resolved RGB + 1-based id per entry).
    let cell_material_ids = derive_color_material_ids(voxels, &tagged, cfg, &mut level)?;

    // Door target_parts default to every output node (matches the parser's
    // finalize when [door] target_parts is omitted).
    level.door.target_parts = level
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.is_output)
        .map(|(i, _)| i)
        .collect();

    Ok(DerivedLevel {
        level,
        cell_material_ids,
    })
}

/// Compute the room size: explicit `.vox` model size if non-zero, else the
/// voxel bounding box `max+1` per axis (rule 1).
fn bbox_size(voxels: &[Voxel], model_size: (u32, u32, u32)) -> (u32, u32, u32) {
    if model_size.0 > 0 && model_size.1 > 0 && model_size.2 > 0 {
        return model_size;
    }
    if voxels.is_empty() {
        return (1, 1, 1);
    }
    let mut mx = (0i32, 0i32, 0i32);
    for v in voxels {
        mx.0 = mx.0.max(v.x);
        mx.1 = mx.1.max(v.y);
        mx.2 = mx.2.max(v.z);
    }
    ((mx.0 + 1) as u32, (mx.1 + 1) as u32, (mx.2 + 1) as u32)
}

/// Inclusive bounding box (min, max) of a non-empty cell list.
fn bbox(cells: &[(i32, i32, i32)]) -> ((i32, i32, i32), (i32, i32, i32)) {
    let mut mn = cells[0];
    let mut mx = cells[0];
    for &(x, y, z) in cells {
        mn.0 = mn.0.min(x);
        mn.1 = mn.1.min(y);
        mn.2 = mn.2.min(z);
        mx.0 = mx.0.max(x);
        mx.1 = mx.1.max(y);
        mx.2 = mx.2.max(z);
    }
    (mn, mx)
}

/// The six axis-neighbours of a cell.
#[inline]
fn neighbors6(c: (i32, i32, i32)) -> [(i32, i32, i32); 6] {
    let (x, y, z) = c;
    [
        (x - 1, y, z),
        (x + 1, y, z),
        (x, y - 1, z),
        (x, y + 1, z),
        (x, y, z - 1),
        (x, y, z + 1),
    ]
}

/// Rule 3 — contiguous boxes of `wall_trace` / `wall_matrix` / `wall_solid`
/// / `hologram` / `portal` / `object` voxels → one `[wall_decal]` each
/// (axis-aligned bbox of the 6-connected run).
///
/// `wall_solid` runs are emitted too so authored (and unknown→solid) interior
/// walls are visible; a fully-walled perimeter can still be left to the shell.
///
/// `hologram`, `portal`, and `object` runs produce a `[wall_decal]` whose
/// `surface_kind` is set to the correct [`re_world::CellKind`]; `build_map` /
/// `stamp_geometry` then stamps each cell with that kind, so a round-trip
/// restores `Hologram`/`Portal`/`Object` cells without data loss.
fn derive_decals(tagged: &[(Voxel, Tag)], level: &mut Level) {
    for (kind_tag, surface) in [
        (Tag::WallTrace, re_world::CellKind::WallTrace),
        (Tag::WallMatrix, re_world::CellKind::WallMatrix),
        (Tag::WallSolid, re_world::CellKind::WallSolid),
        (Tag::Hologram, re_world::CellKind::Hologram),
        (Tag::Portal, re_world::CellKind::Portal),
        (Tag::Object, re_world::CellKind::Object),
    ] {
        let cells: HashSet<(i32, i32, i32)> = tagged
            .iter()
            .filter(|(_, t)| *t == kind_tag)
            .map(|(v, _)| (v.x, v.y, v.z))
            .collect();
        let runs = connected_runs(&cells);
        for run in runs {
            let (mn, mx) = bbox(&run);
            let n = level.decals.len();
            level.decals.push(WallDecal {
                name: format!("d{n}"),
                surface_kind: surface,
                x0: mn.0,
                y0: mn.1,
                z0: mn.2,
                x1: mx.0,
                y1: mx.1,
                z1: mx.2,
                density: 0.0,
                color_head: 0,
                color_tail: 0,
            });
        }
    }
}

/// Partition a cell set into maximal 6-connected runs (flood fill).
fn connected_runs(cells: &HashSet<(i32, i32, i32)>) -> Vec<Vec<(i32, i32, i32)>> {
    let mut seen: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut runs = Vec::new();
    for &start in cells {
        if seen.contains(&start) {
            continue;
        }
        let mut run = Vec::new();
        let mut stack = vec![start];
        seen.insert(start);
        while let Some(c) = stack.pop() {
            run.push(c);
            for n in neighbors6(c) {
                if cells.contains(&n) && !seen.contains(&n) {
                    seen.insert(n);
                    stack.push(n);
                }
            }
        }
        runs.push(run);
    }
    runs
}

/// Rule 5 — each maximal 6-connected pipe run must be 6-adjacent to exactly two
/// node voxels (its endpoints) → one ordered `[pipe from->to]`.
fn derive_pipes(
    pipe_cells: &HashSet<(i32, i32, i32)>,
    node_at: &BTreeMap<(i32, i32, i32), usize>,
    nodes: &[Node],
    edges: &[EdgeCfg],
    level: &mut Level,
) -> Result<(), ImportError> {
    for run in connected_runs(pipe_cells) {
        // Find the (node_index, pipe_endpoint_cell) pairs this run touches.
        let mut endpoints: Vec<(usize, (i32, i32, i32))> = Vec::new();
        let mut seen_nodes: HashSet<usize> = HashSet::new();
        for &pc in &run {
            for nb in neighbors6(pc) {
                if let Some(&ni) = node_at.get(&nb) {
                    if seen_nodes.insert(ni) {
                        endpoints.push((ni, pc));
                    }
                }
            }
        }
        if endpoints.len() != 2 {
            return Err(ImportError::PipeEndpoints {
                found: endpoints.len(),
                cells: {
                    let mut c = run.clone();
                    c.sort_unstable();
                    c
                },
            });
        }

        let (a_node, a_cell) = endpoints[0];
        let (b_node, b_cell) = endpoints[1];

        // Orientation (rule 5): toml [[edge]] wins; else `from` = is_input,
        // else the non-output endpoint; else stable by node index.
        let (from_node, to_node, from_cell, to_cell) =
            orient_pipe((a_node, a_cell), (b_node, b_cell), nodes, edges);

        // Order the run from the `from` endpoint to the `to` endpoint.
        let ordered = order_run(&run, from_cell, to_cell)?;
        // Full cell path including both node anchors (matches `expand_path`
        // output for the equivalent waypoint list).
        let mut path = Vec::with_capacity(ordered.len() + 2);
        path.push(nodes_pos(nodes, from_node));
        path.extend(ordered.iter().copied());
        path.push(nodes_pos(nodes, to_node));

        level.pipes.push(Pipe {
            from: nodes[from_node].name.clone(),
            to: nodes[to_node].name.clone(),
            from_node,
            to_node,
            state: PipeState::Off,
            density: 1.0,
            color_rgb: nodes[from_node].color_rgb,
            path,
            switch_cell: None,
            switch_at_raw: None,
        });
    }
    Ok(())
}

#[inline]
fn nodes_pos(nodes: &[Node], i: usize) -> (i32, i32, i32) {
    (nodes[i].x, nodes[i].y, nodes[i].z)
}

/// A cell coordinate.
type Cell = (i32, i32, i32);
/// An oriented pipe: `(from_node, to_node, from_endpoint_cell, to_endpoint_cell)`.
type Oriented = (usize, usize, Cell, Cell);

/// Decide pipe orientation per rule 5.
fn orient_pipe(a: (usize, Cell), b: (usize, Cell), nodes: &[Node], edges: &[EdgeCfg]) -> Oriented {
    let (an, ac) = a;
    let (bn, bc) = b;
    // toml [[edge]] explicit override.
    for e in edges {
        if e.from == nodes[an].name && e.to == nodes[bn].name {
            return (an, bn, ac, bc);
        }
        if e.from == nodes[bn].name && e.to == nodes[an].name {
            return (bn, an, bc, ac);
        }
    }
    // `from` = the is_input endpoint.
    if nodes[an].is_input && !nodes[bn].is_input {
        return (an, bn, ac, bc);
    }
    if nodes[bn].is_input && !nodes[an].is_input {
        return (bn, an, bc, ac);
    }
    // else `from` = the non-output endpoint.
    if nodes[an].is_output && !nodes[bn].is_output {
        return (bn, an, bc, ac);
    }
    if nodes[bn].is_output && !nodes[an].is_output {
        return (an, bn, ac, bc);
    }
    // stable tie-break by node index.
    if an <= bn {
        (an, bn, ac, bc)
    } else {
        (bn, an, bc, ac)
    }
}

/// Walk a 6-connected run from `start` to `end`, returning the ordered cells.
/// The run is assumed to be a simple path; a BFS-shortest walk recovers order
/// even if the editor left a kink, and errors if `end` is unreachable.
fn order_run(
    run: &[(i32, i32, i32)],
    start: (i32, i32, i32),
    end: (i32, i32, i32),
) -> Result<Vec<(i32, i32, i32)>, ImportError> {
    let set: HashSet<(i32, i32, i32)> = run.iter().copied().collect();
    // BFS for a predecessor map, then reconstruct start..=end.
    let mut prev: BTreeMap<(i32, i32, i32), (i32, i32, i32)> = BTreeMap::new();
    let mut visited: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(start);
    visited.insert(start);
    while let Some(c) = queue.pop_front() {
        if c == end {
            break;
        }
        for nb in neighbors6(c) {
            if set.contains(&nb) && visited.insert(nb) {
                prev.insert(nb, c);
                queue.push_back(nb);
            }
        }
    }
    if !visited.contains(&end) {
        return Err(ImportError::PipeEndpoints {
            found: 2,
            cells: {
                let mut c = run.to_vec();
                c.sort_unstable();
                c
            },
        });
    }
    let mut chain = vec![end];
    let mut cur = end;
    while cur != start {
        cur = prev[&cur];
        chain.push(cur);
    }
    chain.reverse();
    Ok(chain)
}

/// Derive the authored material set (the "color = material KEY" pipeline).
///
/// Each `[[material]]` config row becomes a [`LevelMaterial`]; the `[surface]`
/// table binds a surface kind → a material name. A binding that names a material
/// not defined in `[[material]]` is an error (a typo would otherwise silently
/// fall back to a built-in). Levels with no `[[material]]` / `[surface]` produce
/// an empty set and render exactly as before.
fn derive_materials(cfg: &Config, level: &mut Level) -> Result<(), ImportError> {
    for mc in &cfg.materials {
        if mc.name.trim().is_empty() {
            return Err(ImportError::Derive("[[material]] missing name".into()));
        }
        let fg = match &mc.fg {
            Some(c) => pack_rgb(parse_hex_rgb(c)?),
            None => 0xFF_FFFF,
        };
        let bg = match &mc.bg {
            Some(c) => pack_rgb(parse_hex_rgb(c)?),
            None => 0,
        };
        let glyph = mc
            .glyph
            .as_ref()
            .and_then(|g| g.chars().next())
            .map(u32::from);
        let recipe = match &mc.recipe {
            Some(r) => Some(parse_recipe_cfg(r).ok_or_else(|| {
                ImportError::Derive(format!("[[material]] {}: unknown recipe '{r}'", mc.name))
            })?),
            None => None,
        };
        let anim = match &mc.animation {
            Some(a) => parse_anim_cfg(a)?,
            None => MaterialAnim::default(),
        };
        level.materials.push(LevelMaterial {
            name: mc.name.clone(),
            fg,
            bg,
            glyph,
            recipe,
            anim,
            rank: mc.rank.unwrap_or(0),
            // Additive mirror fields: the .vox import path authors no reflective
            // surfaces, so take the non-reflective defaults.
            ..LevelMaterial::default()
        });
    }

    // Bind surfaces. A name must reference a defined material.
    let known = |name: &str| level.materials.iter().any(|m| m.name == name);
    for (surface, name) in &cfg.surface {
        if !known(name) {
            return Err(ImportError::Derive(format!(
                "[surface] {surface} = '{name}' references an undefined [[material]]"
            )));
        }
        let slot = match surface.as_str() {
            "wall_solid" => &mut level.surface_materials.wall_solid,
            "wall_trace" => &mut level.surface_materials.wall_trace,
            "wall_matrix" => &mut level.surface_materials.wall_matrix,
            "floor" => &mut level.surface_materials.floor,
            "ceiling" => &mut level.surface_materials.ceiling,
            other => {
                return Err(ImportError::Derive(format!(
                    "[surface] unknown surface kind '{other}'"
                )))
            }
        };
        *slot = Some(name.clone());
    }
    Ok(())
}

/// Rule 9 — build the per-cell colour-material id map from `[[color_material]]`
/// entries.
///
/// For each geometry voxel (one whose tag maps to a [`re_world::CellKind`]) we
/// test its RGB against every `[[color_material]]` entry using nearest-neighbour
/// matching within the configured tolerance (squared Euclidean distance).  When a
/// match is found the voxel's grid position is recorded with a **1-based** id
/// (`1` = the first `[[color_material]]` entry, `2` = the second, …).
///
/// The matched colour-material entries are also promoted into `level.materials`
/// as standard named materials (e.g. `cm_20FF80`) so the existing surface-kind
/// binding pipeline — `[material NAME]` sections in the `.lvl` — carries them to
/// the renderer.  A colour-material is given the auto-name `cm_RRGGBB` unless
/// `[[color_material]] name` is explicitly set.
///
/// **Uniqueness guarantee:** two `[[color_material]]` entries MUST NOT resolve to
/// the same auto-name (i.e. their `#RRGGBB` colours must be distinct).  A
/// collision is a hard error (the author painted two entries with the same hex).
///
/// Returns the `(x,y,z) → material_id` map (empty when no
/// `[[color_material]]` entries are configured).
fn derive_color_material_ids(
    voxels: &[Voxel],
    tagged: &[(Voxel, Tag)],
    cfg: &Config,
    level: &mut Level,
) -> Result<BTreeMap<(i32, i32, i32), u16>, ImportError> {
    if cfg.color_materials.is_empty() {
        return Ok(BTreeMap::new());
    }

    // Resolve the colour + name for each [[color_material]] entry.
    // Entry index 0 → material_id 1 (id=0 is reserved for "no override").
    let tol = cfg
        .color_material_tolerance
        .unwrap_or(COLOR_MATERIAL_TOLERANCE);

    struct CmEntry {
        rgb: [u8; 3],
        name: String,
    }

    let mut cm_entries: Vec<CmEntry> = Vec::with_capacity(cfg.color_materials.len());
    for (idx, cm) in cfg.color_materials.iter().enumerate() {
        if cm.color.trim().is_empty() {
            return Err(ImportError::Derive(format!(
                "[[color_material]] index {idx}: missing `color` field"
            )));
        }
        let rgb = parse_hex_rgb(&cm.color)?;
        let name = match &cm.name {
            Some(n) if !n.trim().is_empty() => n.clone(),
            _ => format!("cm_{:02X}{:02X}{:02X}", rgb[0], rgb[1], rgb[2]),
        };
        // Uniqueness check: no two entries may share an auto-name.
        if cm_entries.iter().any(|e| e.name == name) {
            return Err(ImportError::Derive(format!(
                "[[color_material]]: duplicate material name '{name}' \
                 (two entries resolved to the same name — use distinct colours \
                 or explicit `name` fields)"
            )));
        }
        // Emit this colour-material as a standard [material NAME] in the level.
        // If the name was already registered by a [[material]] entry, skip
        // (the author may intentionally alias a [[material]] to a colour).
        if !level.materials.iter().any(|m| m.name == name) {
            let fg = match &cm.fg {
                Some(c) => pack_rgb(parse_hex_rgb(c)?),
                None => 0xFF_FFFF,
            };
            let bg = match &cm.bg {
                Some(c) => pack_rgb(parse_hex_rgb(c)?),
                None => 0,
            };
            let glyph = cm
                .glyph
                .as_ref()
                .and_then(|g| g.chars().next())
                .map(u32::from);
            let recipe = match &cm.recipe {
                Some(r) => Some(parse_recipe_cfg(r).ok_or_else(|| {
                    ImportError::Derive(format!("[[color_material]] {name}: unknown recipe '{r}'"))
                })?),
                None => None,
            };
            let anim = match &cm.animation {
                Some(a) => parse_anim_cfg(a)?,
                None => re_world::MaterialAnim::default(),
            };
            level.materials.push(LevelMaterial {
                name: name.clone(),
                fg,
                bg,
                glyph,
                recipe,
                anim,
                rank: cm.rank.unwrap_or(0),
                ..LevelMaterial::default()
            });
        }
        cm_entries.push(CmEntry { rgb, name });
    }

    // Match every geometry voxel against the colour-material table.
    let mut cell_ids: BTreeMap<(i32, i32, i32), u16> = BTreeMap::new();

    // Build a fast lookup of tagged geometry positions (only voxels with a
    // direct CellKind mapping are candidates; graph-element voxels are not
    // geometry cells and do not carry material ids).
    for &(v, tag) in tagged {
        if tag.cell_kind().is_none() {
            // graph element (node/pipe/switch/door/player/pedestal) — skip.
            continue;
        }
        // Nearest colour-material within tolerance.
        let mut best: Option<(i32, u16)> = None; // (dist2, 1-based id)
        for (idx, entry) in cm_entries.iter().enumerate() {
            let d = rgb_dist2(v.rgb, entry.rgb);
            if d <= tol && best.is_none_or(|(bd, _)| d < bd) {
                best = Some((d, (idx + 1) as u16));
            }
        }
        if let Some((_, id)) = best {
            cell_ids.insert((v.x, v.y, v.z), id);
        }
    }

    // Warn about geometry voxels whose original RGB matched a colour-material
    // entry that was NOT in the tagged set (could happen if the voxel's colour
    // is BOTH close to a tag AND close to a colour-material).
    // This is informational only — the tag classifier already ran first.
    // (No action needed here; the tag takes priority because we only test
    //  voxels that ALREADY made it through tag classification above.)

    // Also handle voxels whose original (pre-tag-matching) RGB was not
    // matched by any tag — such voxels land as wall_solid.  They may still
    // match a [[color_material]] entry via their original RGB; check those too.
    for &v in voxels {
        // If we already assigned an id via the tagged pass, skip.
        if cell_ids.contains_key(&(v.x, v.y, v.z)) {
            continue;
        }
        // Only if this voxel has no tag match (it landed as wall_solid from the
        // "unknown colour" path) should we attempt a colour-material match.
        // We detect this by checking whether the voxel's RGB is NOT an exact
        // tag match (the tag-matcher may have already accepted it).
        // Simpler: re-match the raw voxel RGB against colour-material table.
        let mut best: Option<(i32, u16)> = None;
        for (idx, entry) in cm_entries.iter().enumerate() {
            let d = rgb_dist2(v.rgb, entry.rgb);
            if d <= tol && best.is_none_or(|(bd, _)| d < bd) {
                best = Some((d, (idx + 1) as u16));
            }
        }
        if let Some((_, id)) = best {
            cell_ids.insert((v.x, v.y, v.z), id);
        }
    }

    Ok(cell_ids)
}

/// Parse a `[[material]].recipe` string (`"brick aspect=2.0 …"`) from a `.toml`
/// sidecar.
///
/// DELEGATES to [`re_level::parse_recipe`] — the same parser the `.lvl`
/// `[material] recipe =` key uses. It used to be a hand-copied duplicate that
/// understood four kinds while the `.lvl` parser understood six, so `hologram` and
/// `codewall` were authorable in one surface and a hard import error in the other.
/// One parser means the two authoring surfaces cannot drift apart again.
fn parse_recipe_cfg(v: &str) -> Option<MaterialRecipe> {
    re_level::parse_recipe(v)
}

/// Parse a `[[material]].animation` table into a [`MaterialAnim`].
fn parse_anim_cfg(a: &AnimationCfg) -> Result<MaterialAnim, ImportError> {
    let kind = match a.kind.as_str() {
        "" | "none" | "static" => MaterialAnimKind::None,
        "pulse" => MaterialAnimKind::Pulse,
        "shift" => MaterialAnimKind::Shift,
        "flicker" => MaterialAnimKind::Flicker,
        other => {
            return Err(ImportError::Derive(format!(
                "[[material]] animation: unknown kind '{other}'"
            )))
        }
    };
    Ok(MaterialAnim {
        kind,
        speed: a.speed.unwrap_or(1.0),
        amplitude: a.amplitude.unwrap_or(0.2),
    })
}

/// Rule 6 — each switch voxel must be on / 6-adjacent to exactly one pipe path.
/// Sets that pipe's `switch_cell` to the matching path index.
fn derive_switches(switch_cells: &[(i32, i32, i32)], level: &mut Level) -> Result<(), ImportError> {
    for &sc in switch_cells {
        // Candidate set of cells the switch "is on": itself + 6 neighbours.
        let candidates: Vec<(i32, i32, i32)> = std::iter::once(sc).chain(neighbors6(sc)).collect();
        // Which pipes does it touch, and at what path index?
        let mut hits: Vec<(usize, usize)> = Vec::new(); // (pipe, path index)
        for (pi, pipe) in level.pipes.iter().enumerate() {
            let mut best: Option<usize> = None;
            for cand in &candidates {
                if let Some(idx) = pipe.path.iter().position(|c| c == cand) {
                    best = Some(match best {
                        Some(b) => b.min(idx),
                        None => idx,
                    });
                }
            }
            if let Some(idx) = best {
                hits.push((pi, idx));
            }
        }
        if hits.len() != 1 {
            return Err(ImportError::SwitchUnattached {
                cell: sc,
                touching: hits.len(),
            });
        }
        let (pi, idx) = hits[0];
        level.pipes[pi].switch_cell = Some(idx);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Serializer: Level → .lvl text (inverse of re_level::load_string)
// ---------------------------------------------------------------------------

/// Serialize a [`Level`] back to `.lvl` text that `re_level::load_string`
/// round-trips. Auto-derived shell geometry is left to `stamp_geometry`.
///
/// ## Lossiness (R-H4)
///
/// This is now a **complete** serializer: it emits every populated section,
/// including `[part]` (with its mask), `[emit]`, and `[note]`. A `Level` that
/// the parser accepted serializes back to text the parser accepts with the
/// same data. (Earlier this function silently dropped parts/emits/notes; that
/// was safe only because [`derive_level`] never produces them, but a
/// general-purpose `Level → .lvl` serializer must not assume its input came
/// from `derive_level`.)
///
/// The one deliberate normalization: `[pedestal] volume` dimensions are
/// clamped to a minimum of 1. A zero extent is never valid geometry (the
/// parser requires positive `[room]`/`[part]` dims; a 0-volume pedestal is an
/// uninitialized `Level`, not authored data), so clamping is a defensive
/// floor, not data loss.
pub fn to_lvl_text(level: &Level) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();

    let _ = writeln!(s, "[meta]");
    let _ = writeln!(s, "name = {}", level.name);
    // INERT AUTHORING METADATA (`version`, `spec_rev`, `[room] floor`/`ceiling`,
    // `[door] target_model`). `re_level` parses these but nothing acts on them; the
    // writer used to emit none of them, so every editor save DELETED five authored
    // keys from up to 24 shipped levels at once — invisible to the text-vs-text
    // idempotency sweep, because a key absent from BOTH writes compares equal.
    // `spec_rev` in particular is read off disk by a host game's canon-consistency
    // guard, so erasing it can disarm a live test there. Emitted only when non-empty, and
    // in the shipped files' own key order (name/version/title/spec_rev,
    // size/floor/ceiling/ambient, pos/target_model), so a canonical file's bytes
    // are unchanged.
    if !level.version.is_empty() {
        let _ = writeln!(s, "version = {}", level.version);
    }
    if !level.title.is_empty() {
        let _ = writeln!(s, "title = {}", level.title);
    }
    if !level.spec_rev.is_empty() {
        let _ = writeln!(s, "spec_rev = {}", level.spec_rev);
    }
    // `prop_style` (two-repo split A1, replaces the `campaign.toml`-derived
    // default): emitted ONLY when authored, never derived from `Lab` alone,
    // because `Lab` is both the struct default AND a legitimate authored value
    // — see `Level::prop_style_authored`. Omitting it here for an unauthored
    // level keeps every pre-existing level byte-identical across a save.
    if level.prop_style_authored {
        let _ = writeln!(s, "prop_style = {}", level.prop_default_style.as_str());
    }
    s.push('\n');

    let _ = writeln!(s, "[room]");
    let _ = writeln!(
        s,
        "size = {},{},{}",
        level.size.0, level.size.1, level.size.2
    );
    if !level.room_floor.is_empty() {
        let _ = writeln!(s, "floor = {}", level.room_floor);
    }
    if !level.room_ceiling.is_empty() {
        let _ = writeln!(s, "ceiling = {}", level.room_ceiling);
    }
    if !level.ambient.is_empty() {
        let _ = writeln!(s, "ambient = {}", level.ambient);
    }
    // `[room]` opt-in FLAGS. Both default to `false` and the parser reads them as
    // `[room]` keys, so a writer that dropped them would silently turn a reactive
    // (`render_pipes`) or finale (`cycle_keep`) level into a plain one on save.
    // Emitted ONLY when set (so a level without them stays byte-identical).
    // `render_pipes` MUST round-trip or a saved reactive level loses its tagged
    // pipe traces (the reactive bit-15 tags).
    if level.render_pipes {
        let _ = writeln!(s, "render_pipes = yes");
    }
    if level.cycle_keep {
        let _ = writeln!(s, "cycle_keep = yes");
    }
    // `[room] power_budget` — the simultaneously-ON pipe cap.
    // Omit-when-default: `0` means UNLIMITED and is what every non-opted-in level
    // carries, so an un-budgeted level re-saves BYTE-IDENTICAL (no new line). A
    // budgeted level MUST round-trip or the editor would silently strip a rule
    // that changes whether the level is winnable — the same class of silent-drop
    // bug `render_pipes`/`cycle_keep` above guard against.
    if level.power_budget > 0 {
        let _ = writeln!(s, "power_budget = {}", level.power_budget);
    }
    // `[room] storey_height` — omit-when-default rule: the key is written ONLY
    // when the effective height differs from DEFAULT_STOREY_HEIGHT (4). This keeps
    // all existing single-storey levels BYTE-IDENTICAL on re-save (they have no
    // `storey_height` line and stay that way), while a multi-floor level preserves
    // its value across editor save→reload. The effective height is obtained via
    // `Level::storey_height()`, which maps the raw-field sentinel 0 → default 4,
    // so programmatically-built levels that never set the field also satisfy
    // omit-when-default without special-casing.
    let sh = level.storey_height();
    if sh != re_level::DEFAULT_STOREY_HEIGHT {
        let _ = writeln!(s, "storey_height = {sh}");
    }
    s.push('\n');

    for (ni, n) in level.nodes.iter().enumerate() {
        let _ = writeln!(s, "[node {}]", n.name);
        let _ = writeln!(s, "pos = {},{},{}", n.x, n.y, n.z);
        if n.glyph != '\0' && n.glyph != ' ' {
            let _ = writeln!(s, "glyph = {}", n.glyph);
        }
        if !n.pool.is_empty() {
            let _ = writeln!(s, "pool = {}", n.pool.join(" "));
        }
        if !n.pool_alt.is_empty() {
            let _ = writeln!(s, "pool_alt = {}", n.pool_alt.join(" "));
        }
        // `[node] anchor` — the pool anchor-word index (default 0) the parser reads
        // but the writer previously dropped. Emitted only when non-zero so a
        // default-anchor node stays byte-identical. Round-trips via re_level's `anchor`.
        if n.anchor != 0 {
            let _ = writeln!(s, "anchor = {}", n.anchor);
        }
        let _ = writeln!(s, "color = #{:06X}", n.color_rgb & 0xFF_FFFF);
        if n.is_input {
            let _ = writeln!(s, "is_input = yes");
        }
        if n.is_output {
            let _ = writeln!(s, "is_output = yes");
        }
        if n.threshold != 0 {
            let _ = writeln!(s, "threshold = {}", n.threshold);
        }
        // §4.1 inline sound-field attach: the field carried by THIS node, if
        // any (`re_level::fields::to_inline_line`'s inverse of `attach_inline`).
        if let Some(f) = level
            .field_defs
            .iter()
            .find(|f| f.host == re_level::FieldHost::Node(ni))
        {
            if let Some(line) = re_level::fields::to_inline_line(f) {
                let _ = writeln!(s, "field = {line}");
            }
        }
        s.push('\n');
    }

    for (pi, p) in level.pipes.iter().enumerate() {
        let _ = writeln!(s, "[pipe {}->{}]", p.from, p.to);
        let _ = writeln!(
            s,
            "state = {}",
            if p.state == PipeState::On {
                "ON"
            } else {
                "OFF"
            }
        );
        // The expanded path round-trips losslessly as a per-cell waypoint list
        // (each cell is already 6-connected, so expand_path is the identity).
        let path = p
            .path
            .iter()
            .map(|(x, y, z)| format!("({x},{y},{z})"))
            .collect::<Vec<_>>()
            .join(" ");
        let _ = writeln!(s, "path = {path}");
        if let Some(idx) = p.switch_cell {
            if let Some(&(x, y, z)) = p.path.get(idx) {
                let _ = writeln!(s, "switch_at = {x},{y},{z}");
            }
        }
        if (p.density - 1.0).abs() > f32::EPSILON {
            let _ = writeln!(s, "density = {}", p.density);
        }
        // `[pipe] color` — an authored per-pipe tint the parser reads but the writer
        // previously DROPPED (04.lvl sets it). Emitted only when non-default (0) so a
        // colour-less pipe stays byte-identical. Round-trips via re_level's `color`.
        if p.color_rgb & 0xFF_FFFF != 0 {
            let _ = writeln!(s, "color = #{:06X}", p.color_rgb & 0xFF_FFFF);
        }
        // §4.1 inline sound-field attach carried by THIS pipe, if any.
        if let Some(f) = level
            .field_defs
            .iter()
            .find(|f| f.host == re_level::FieldHost::Pipe(pi))
        {
            if let Some(line) = re_level::fields::to_inline_line(f) {
                let _ = writeln!(s, "field = {line}");
            }
        }
        s.push('\n');
    }

    // Parts: a voxel stamp + its solid mask, indexed `x + w*(y + h*z)`.
    for (pi, part) in level.parts.iter().enumerate() {
        let _ = writeln!(s, "[part {}]", part.name);
        let _ = writeln!(s, "size = {},{},{}", part.w, part.h, part.d);
        if part.off_x != 0 || part.off_y != 0 || part.off_z != 0 {
            let _ = writeln!(s, "offset = {},{},{}", part.off_x, part.off_y, part.off_z);
        }
        let _ = writeln!(s, "color = #{:06X}", part.color_rgb & 0xFF_FFFF);
        // The lit look (`height` / `top` / `top_from`): written only when it differs
        // from the default, so a level that never authored them saves byte-for-byte as
        // before. `part_look` already drops a `top_from` with no `top` colour (it draws
        // nothing), so that combination is not written back either.
        let look = level.part_look(pi);
        // `height` is always an exact value reaching here — the parser sets it to the
        // literal constants `1.0`/`0.5` for `full`/`half`, or an exact `f32::parse` of an
        // authored fraction — but clippy::float_cmp (deny-level) still forbids `==`/`!=`
        // on floats, so compare via an epsilon that cannot change behaviour for any of
        // those exact inputs.
        if (look.height - 1.0).abs() > f32::EPSILON {
            if (look.height - 0.5).abs() <= f32::EPSILON {
                let _ = writeln!(s, "height = half");
            } else {
                let _ = writeln!(s, "height = {}", look.height);
            }
        }
        if let Some((rgb, from)) = look.top {
            let _ = writeln!(s, "top = #{:06X}", rgb & 0xFF_FFFF);
            if from != re_level::PART_TOP_FROM_DEFAULT {
                let _ = writeln!(s, "top_from = {from}");
            }
        }
        let _ = writeln!(s, "mask =");
        let (w, h, d) = (part.w.max(0), part.h.max(0), part.d.max(0));
        for z in 0..d {
            if z > 0 {
                // blank line between z-layers advances the parser's mask_z
                s.push('\n');
            }
            for y in 0..h {
                let mut row = String::with_capacity(w as usize);
                for x in 0..w {
                    let idx = (x + w * (y + h * z)) as usize;
                    let solid = part.mask.get(idx).copied().unwrap_or(0) != 0;
                    row.push(if solid { '#' } else { '.' });
                }
                let _ = writeln!(s, "{row}");
            }
        }
        s.push('\n');
    }

    // Emits: one `[emit SOURCE=1]` per source node, listing the parts it lights.
    if !level.emits.is_empty() {
        // Group part names by source node, preserving the clauses' AUTHORED
        // ORDER: each source's group sits at the position of its FIRST
        // clause. A `BTreeMap<usize, _>` here previously re-sorted the groups
        // by node INDEX ascending, silently reordering `[emit ...]` clauses
        // relative to the source file whenever the authored order wasn't
        // already index-ascending (L3: output=5, h1=3, h2=4 becomes h1, h2,
        // output) — a structural-loss bug caught by
        // `every_shipped_level_survives_a_save_with_no_structural_loss`.
        let mut groups: Vec<(usize, Vec<String>)> = Vec::new();
        for e in &level.emits {
            if let Some(part) = level.parts.get(e.part) {
                match groups.iter_mut().find(|(src, _)| *src == e.source_node) {
                    Some((_, parts)) => parts.push(part.name.clone()),
                    None => groups.push((e.source_node, vec![part.name.clone()])),
                }
            }
        }
        for (src, parts) in groups {
            if let Some(node) = level.nodes.get(src) {
                let _ = writeln!(s, "[emit {}=1]", node.name);
                let _ = writeln!(s, "parts = {}", parts.join(" "));
                s.push('\n');
            }
        }
    }

    for d in &level.decals {
        let _ = writeln!(s, "[wall_decal {}]", d.name);
        let surface = match d.surface_kind {
            re_world::CellKind::WallMatrix => "matrix",
            re_world::CellKind::WallSolid => "solid",
            re_world::CellKind::Hologram => "hologram",
            re_world::CellKind::Portal => "portal",
            re_world::CellKind::Object => "object",
            _ => "trace",
        };
        let _ = writeln!(s, "surface = {surface}");
        let _ = writeln!(
            s,
            "region = ({},{},{})-({},{},{})",
            d.x0, d.y0, d.z0, d.x1, d.y1, d.z1
        );
        // `density` / `color_head` / `color_tail` — parsed by `re_level` but
        // previously DROPPED by the writer (silent data loss on save→reload).
        // Emitted only when non-default so a plain decal stays byte-identical:
        //   density   default = 0.0  (f32 zero)
        //   color_head/tail defaults = 0x00000000
        if d.density != 0.0 {
            let _ = writeln!(s, "density = {}", d.density);
        }
        if d.color_head != 0 {
            let _ = writeln!(s, "color_head = #{:06X}", d.color_head & 0xFF_FFFF);
        }
        if d.color_tail != 0 {
            let _ = writeln!(s, "color_tail = #{:06X}", d.color_tail & 0xFF_FFFF);
        }
        s.push('\n');
    }

    // [portal] PAIRS: emit each non-euclidean warp-portal pair so a load → save
    // → load round-trip is lossless (the editor depends on this). Matches the
    // `[wall_decal]` region-emit style.
    for pt in &level.portals {
        let _ = writeln!(s, "[portal {}]", pt.name);
        let _ = writeln!(
            s,
            "region_a = ({},{},{})-({},{},{})",
            pt.ax0, pt.ay0, pt.az0, pt.ax1, pt.ay1, pt.az1
        );
        let _ = writeln!(
            s,
            "region_b = ({},{},{})-({},{},{})",
            pt.bx0, pt.by0, pt.bz0, pt.bx1, pt.by1, pt.bz1
        );
        if pt.rotate != 0.0 {
            let _ = writeln!(s, "rotate = {}", pt.rotate);
        }
        if !pt.skin.is_empty() {
            let _ = writeln!(s, "skin = {}", pt.skin);
        }
        s.push('\n');
    }

    // [vox] MODEL STAMPS: emit each `.vox` placement so a load → save → load
    // round-trip is lossless (the editor's vox-placement path depends on this).
    // Without this the parsed `level.voxes` were silently DROPPED on save — the
    // same gap `[emitter]` had. Emitted ONLY when the level carries vox stamps,
    // so vox-less levels stay byte-identical. The exact inverse of `re_level`'s
    // `[vox]` parser (file / offset / material / scale).
    for v in &level.voxes {
        write_vox(&mut s, v);
    }

    // [texture] IMAGE TEXTURES: emit each `[texture NAME] file = …` so a load → save →
    // load round-trip is lossless (TX1 carries one). Only the FILE PATH is serialized —
    // the decoded RGBA8 buffer is re-derived on load by `re_level::build_map`. Emitted
    // ONLY when the level carries textures, so texture-less levels stay byte-identical.
    // The exact inverse of `re_level`'s `[texture]` parser (`file`).
    for t in &level.textures {
        let _ = writeln!(s, "[texture {}]", t.name);
        let _ = writeln!(s, "file = {}", t.file);
        // `max_size` is a LOAD-time decode cap (spec 1.4): the longest edge the
        // image keeps in memory. Emitted only when the author set one, so a
        // texture without a cap stays byte-identical.
        if let Some(cap) = t.max_size {
            let _ = writeln!(s, "max_size = {cap}");
        }
        // `height = <path>` is the optional 8-bit RELIEF channel (spec 1.4). Like
        // `file`, only the PATH round-trips - the decoded bytes are re-derived on
        // load. Emitted only when the author set one, so a texture with no relief
        // stays byte-identical.
        if let Some(hp) = &t.height {
            let _ = writeln!(s, "height = {hp}");
        }
        // `frames = <path.gif>` + `fps` are the ANIMATED texture (spec 1.4). Like
        // `file`, only the PATH round-trips - every frame is re-decoded on load.
        // Emitted only when authored, so a still texture stays byte-identical.
        if let Some(fp) = &t.frames {
            let _ = writeln!(s, "frames = {fp}");
        }
        // `fps` is written whenever it is authored, INDEPENDENTLY of `frames`: the
        // parser accepts the two keys separately, so dropping an `fps` that sits
        // beside an unset `frames` would make save -> load lossy the moment an author
        // set the rate before picking the clip.
        if let Some(fps) = t.fps {
            let _ = writeln!(s, "fps = {fps}");
        }
        s.push('\n');
    }

    // [sky] IMAGE CUBEMAP + PROCEDURAL SKY PROGRAM: emit the cylinder / cap_top /
    // cap_bottom paths (image-only round-trip, unchanged) AND the `base =` /
    // `layer =` (repeatable) / `seed =` / `hz =` keys `re_world::sky_spec`
    // parses. Only FILE PATHS are serialized for the cubemap — the decoded
    // pixel buffers are re-derived on
    // load by `re_level::build_map`. The section is emitted when EITHER an
    // image path OR a procedural key is authored, so an image-only or
    // procedural-only `[sky]` round-trips and a sky-less level stays
    // byte-identical. `kind =` shorthand is never re-emitted (it only expands
    // at parse time); the writer always emits the expanded `base`/`layer`
    // lines, which is the exact inverse the parser reads regardless of which
    // form the level was originally authored in.
    let has_procedural = level.sky.base.is_some() || !level.sky.layers.is_empty();
    if !level.sky.is_empty() || has_procedural {
        let _ = writeln!(s, "[sky]");
        if let Some(p) = &level.sky.cylinder {
            let _ = writeln!(s, "cylinder = {p}");
        }
        if let Some(p) = &level.sky.cap_top {
            let _ = writeln!(s, "cap_top = {p}");
        }
        if let Some(p) = &level.sky.cap_bottom {
            let _ = writeln!(s, "cap_bottom = {p}");
        }
        if let Some(base) = &level.sky.base {
            let _ = writeln!(s, "base = {}", write_sky_base(base));
        }
        for layer in &level.sky.layers {
            let _ = writeln!(s, "layer = {}", write_sky_layer(layer));
        }
        if let Some(seed) = level.sky.seed {
            let _ = writeln!(s, "seed = {seed}");
        }
        if let Some(hz) = level.sky.hz {
            let _ = writeln!(s, "hz = {hz}");
        }
        s.push('\n');
    }

    // [holo] HOLOGRAM objects: emit each transparent-wireframe volume so a load →
    // save → load round-trip is lossless (levels 10/13/13b/97/98 carry these).
    // Without this the parsed `level.holos` were silently DROPPED on save (B1).
    // Emitted ONLY when the level carries holos, so holo-less levels stay
    // byte-identical. The exact inverse of `re_level`'s `[holo]` parser
    // (shape / region / style).
    for h in &level.holos {
        write_holo(&mut s, h);
    }

    // [zone_input] FLOOR-ZONE INPUTS: emit each player-position-as-input zone so a
    // load → save → load round-trip is lossless (L5 carries one). Without this the
    // parsed `level.zone_inputs` were silently DROPPED on save (B1). Emitted ONLY
    // when present. The exact inverse of `re_level`'s `[zone_input]` parser
    // (node / region).
    for zi in &level.zone_inputs {
        write_zone_input(&mut s, zi);
    }

    // [trigger] TRIGGER-FIELDS: emit each region→action trigger so a load → save →
    // load round-trip is lossless. Without this the parsed `level.triggers` would be
    // silently DROPPED on save (the recent serialize-gap bug). The exact inverse of
    // `re_level`'s `[trigger]` parser (region / action / once).
    for t in &level.triggers {
        write_trigger(&mut s, t);
    }

    // SCALABLE MATERIAL OVERLAY ([zone]/[line]/[inset]): emit each authored overlay
    // primitive so a load → save → load round-trip is lossless (A1–A5/97/98 carry
    // these). Without this the parsed `level.overlay.{zones,lines,insets}` were
    // silently DROPPED on save (B1). Each helper emits ONLY when its collection is
    // non-empty, so an overlay-less level stays byte-identical. Exact inverses of
    // `re_level`'s `[zone]`/`[line]`/`[inset]` parsers.
    for z in &level.overlay.zones {
        write_zone(&mut s, z);
    }
    for ln in &level.overlay.lines {
        write_line(&mut s, ln);
    }
    for ins in &level.overlay.insets {
        write_inset(&mut s, ins);
    }

    // [hierarchy] OBJECT NESTING: emit each `(child) -> (parent)` link so the
    // editor's parent/child nesting survives save→reload. Without this the parsed
    // `level.hierarchy` would be silently DROPPED on save (the "parses but never
    // serializes" gap). Emitted ONLY when the level carries links, so a
    // hierarchy-less level stays byte-identical. The exact inverse of `re_level`'s
    // `[hierarchy]` parser (`child_kind idx -> parent_kind idx`).
    if !level.hierarchy.is_empty() {
        write_hierarchy(&mut s, &level.hierarchy);
    }

    if !level.inputs.is_empty() {
        let _ = writeln!(s, "[inputs]");
        for (name, val) in &level.inputs {
            let _ = writeln!(s, "{name} = {val}");
        }
        s.push('\n');
    }

    if level.effects.snow > 0.0
        || level.effects.steam
        || level.effects.matrix > 0.0
        || level.effects.glitch > 0.0
    {
        let _ = writeln!(s, "[effects]");
        if level.effects.snow > 0.0 {
            let _ = writeln!(s, "snow = {}", level.effects.snow);
        }
        if level.effects.steam {
            let _ = writeln!(s, "steam = yes");
        }
        if level.effects.matrix > 0.0 {
            let _ = writeln!(s, "matrix = {}", level.effects.matrix);
        }
        // `glitch` is a parsed `[effects]` key (clamped 0..1) that the writer
        // previously DROPPED — another silent-drop gap. Emitted only when set so a
        // glitch-less level stays byte-identical.
        if level.effects.glitch > 0.0 {
            let _ = writeln!(s, "glitch = {}", level.effects.glitch);
        }
        s.push('\n');
    }

    if level.palette.has_fog || level.palette.has_fog_dist {
        let _ = writeln!(s, "[palette]");
        if level.palette.has_fog {
            let _ = writeln!(
                s,
                "fog_color = #{:06X}",
                level.palette.fog_color & 0xFF_FFFF
            );
        }
        if level.palette.has_fog_dist {
            let _ = writeln!(s, "fog_near = {}", level.palette.fog_near);
            let _ = writeln!(s, "fog_far = {}", level.palette.fog_far);
        }
        s.push('\n');
    }

    // [light] REPEATABLE POINT LIGHTS: emit one `[light]` section per authored
    // punctual light so a load → save → load round-trip is lossless. Without this
    // the parsed `level.lights` would be silently DROPPED on save (the same class
    // of gap fixed above for decals/holos/zones/etc). Emitted ONLY when the
    // level carries lights, so a light-less level stays byte-identical. The exact
    // inverse of `re_level`'s `[light]` parser (pos / color / radius / intensity).
    for l in &level.lights {
        let _ = writeln!(s, "[light]");
        let _ = writeln!(s, "pos = {}, {}, {}", l.pos.x, l.pos.y, l.pos.z);
        let _ = writeln!(s, "color = #{:06X}", l.color & 0xFF_FFFF);
        let _ = writeln!(s, "radius = {}", l.radius);
        let _ = writeln!(s, "intensity = {}", l.intensity);
        s.push('\n');
    }

    // [atmosphere] KEY LIGHT / AMBIENT / HEIGHT FOG / POSTFX: emit only the keys
    // that live on `LevelAtmosphere` (the fog-DISTANCE keys are aliases that land
    // in `LevelPalette` and are already emitted by the `[palette]` block above —
    // emitting them here too would duplicate the same meaning in two sections).
    // Emitted ONLY when the level carries at least one atmosphere key, so an
    // atmosphere-less level stays byte-identical. The exact inverse of
    // `re_level`'s `[atmosphere]` parser (key_dir / key_color / ambient /
    // fog_height / postfx).
    let atmos = &level.atmos;
    if atmos.key_dir.is_some()
        || atmos.key_color.is_some()
        || atmos.ambient.is_some()
        || atmos.fog_height.is_some()
        || atmos.postfx.is_some()
    {
        let _ = writeln!(s, "[atmosphere]");
        if let Some(d) = atmos.key_dir {
            let _ = writeln!(s, "key_dir = {}, {}, {}", d.x, d.y, d.z);
        }
        if let Some(c) = atmos.key_color {
            let _ = writeln!(s, "key_color = #{:06X}", c & 0xFF_FFFF);
        }
        if let Some(a) = atmos.ambient {
            let _ = writeln!(s, "ambient = {a}");
        }
        if let Some(hf) = &atmos.fog_height {
            let _ = writeln!(s, "fog_height = {} {}", hf.base, hf.falloff);
        }
        if let Some(p) = &atmos.postfx {
            let _ = writeln!(s, "postfx = {p}");
        }
        s.push('\n');
    }

    // Materials: one [material NAME] per authored material, plus its surface
    // binding (so re_level resolves names → surfaces on load). Emitted before
    // [pedestal] but it can go anywhere — order-independent in the parser.
    for m in &level.materials {
        let _ = writeln!(s, "[material {}]", m.name);
        // `preset = NAME` selects a renderer CATALOG material; explicit keys below
        // still override on top, so emit it FIRST (matches the parser's "preset
        // then overrides" intent). Round-trips via re_level's `"preset"` key.
        if let Some(preset) = &m.preset {
            let _ = writeln!(s, "preset = {preset}");
        }
        let _ = writeln!(s, "fg = #{:06X}", m.fg & 0xFF_FFFF);
        let _ = writeln!(s, "bg = #{:06X}", m.bg & 0xFF_FFFF);
        if let Some(g) = m.glyph {
            if let Some(ch) = char::from_u32(g) {
                let _ = writeln!(s, "glyph = {ch}");
            }
        }
        if let Some(r) = m.recipe {
            let _ = writeln!(s, "recipe = {}", recipe_to_text(r));
        }
        // IMAGE-TEXTURE binding: emit `texture = NAME` so an `[material] recipe = image
        // texture = NAME` round-trips. The `tex_id` itself is NOT serialized — it is a
        // build-time index re-derived from the `[texture NAME]` file on load. Only the
        // NAME (which `re_level` re-resolves to a fresh id) persists. Emitted only when
        // a texture is bound, so non-image materials stay byte-identical.
        if let Some(tex) = &m.texture {
            let _ = writeln!(s, "texture = {tex}");
        }
        // IMAGE-TEXTURE world-anchored span (M3a): emit `span = ALONG_MIN,ALONG_SIZE
        // UP_TOP,UP_SIZE` so a `recipe = image` material that stretches ONE figure
        // across the surface round-trips. Parsed by re_level's `"span"` key
        // (`parse_image_span`). Emitted ONLY when set, so a per-cell-tiled (or
        // non-image) material stays byte-identical. Previously DROPPED → B1 data loss.
        if let Some(span) = &m.image_span {
            let _ = writeln!(
                s,
                "span = {},{} {},{}",
                span.along_min, span.along_size, span.up_top, span.up_size
            );
        }
        // IMAGE-TEXTURE sub-cell detail mode (M3b): emit `subcell = half|quad|sextant|
        // braille` so a `recipe = image` material that renders at 2×1 / 2×2 / 3×2 / 2×4
        // round-trips. Parsed by
        // re_level's `"subcell"` key (`parse_subcell`). `Off` is the default solid
        // block → omitted so a non-sub-cell material stays byte-identical. Previously
        // DROPPED → B1 data loss.
        match m.image_subcell {
            re_world::SubcellMode::Off => {}
            re_world::SubcellMode::Half => {
                let _ = writeln!(s, "subcell = half");
            }
            re_world::SubcellMode::Quad => {
                let _ = writeln!(s, "subcell = quad");
            }
            re_world::SubcellMode::Sextant => {
                let _ = writeln!(s, "subcell = sextant");
            }
            re_world::SubcellMode::Braille => {
                let _ = writeln!(s, "subcell = braille");
            }
            re_world::SubcellMode::Ascii => {
                let _ = writeln!(s, "subcell = ascii");
            }
            re_world::SubcellMode::Contour => {
                let _ = writeln!(s, "subcell = contour");
            }
            re_world::SubcellMode::Lines => {
                let _ = writeln!(s, "subcell = lines");
            }
        }
        // IMAGE-TEXTURE fit mode (Component 1 quality-textures spec): emit
        // `fit = stretch` / `fit = cover` only when non-default. `Contain` is the
        // re_level parser default so omitting it keeps byte-identical output for any
        // existing material that did not set `fit`. Placed after `subcell` so the key
        // order matches the re_level parser's expectation and the canonical .lvl shape.
        match m.image_fit {
            re_world::Fit::Contain => {}
            re_world::Fit::Stretch => {
                let _ = writeln!(s, "fit = stretch");
            }
            re_world::Fit::Cover => {
                let _ = writeln!(s, "fit = cover");
            }
        }
        // IMAGE-TEXTURE span-relative REPEAT: emit `tile = ALONG UP` so a tiled
        // `recipe = image` material round-trips. `None` (one copy) is the parser
        // default and is OMITTED, so every material authored before this key keeps
        // byte-identical output.
        if let Some((ta, tb)) = m.image_tile {
            let _ = writeln!(s, "tile = {ta} {tb}");
        }
        // IMAGE-TEXTURE RELIEF: emit `relief = height_scale=… normal_strength=…` in
        // the same KNOB=VALUE token form the parser reads. Both knobs are always
        // written (the parser defaults them, but writing both makes the saved file
        // state what it means rather than relying on a default that may be retuned).
        // `None` is OMITTED, keeping a self-lit image byte-identical.
        if let Some(r) = m.image_relief {
            let _ = writeln!(
                s,
                "relief = height_scale={} normal_strength={}",
                r.height_scale, r.normal_strength
            );
        }
        if m.anim.kind != MaterialAnimKind::None {
            let kind = match m.anim.kind {
                MaterialAnimKind::Pulse => "pulse",
                MaterialAnimKind::Shift => "shift",
                MaterialAnimKind::Flicker => "flicker",
                MaterialAnimKind::None => unreachable!(),
            };
            let _ = writeln!(
                s,
                "animation = {kind} speed={} amplitude={}",
                m.anim.speed, m.anim.amplitude
            );
        }
        if m.rank != 0 {
            let _ = writeln!(s, "rank = {}", m.rank);
        }
        // MIRROR authoring: emit `reflective` only when true (non-reflective
        // materials stay byte-identical). Emit `clarity` whenever it differs from
        // the default (0.85) regardless of `reflective` — a non-reflective material
        // CAN carry a non-default clarity (e.g. authored for future activation), and
        // dropping it on save→reload would lose the authored value silently.
        // Round-trips via re_level's `"reflective"`/`"clarity"` keys (the parser
        // reads them independently; clarity is clamped to 0..1 on parse).
        if m.reflective {
            let _ = writeln!(s, "reflective = true");
        }
        if (m.clarity - re_world::MaterialDef::DEFAULT_CLARITY).abs() > f32::EPSILON {
            let _ = writeln!(s, "clarity = {}", m.clarity);
        }
        // DETAIL-TIER: emit `detail = on` only when set (the writer once had no
        // case for this key at all, so a `[material] detail = on` silently
        // dropped on every save/reload through the editor; `false` is the
        // parser's default, so omitting it keeps every material authored
        // before this key byte-identical). Round-trips via
        // re_level's `"detail"` key (`is_truthy`, `on`/`true` accepted).
        if m.detail {
            let _ = writeln!(s, "detail = on");
        }
        // Surface binding (the surface this material is bound to, if any).
        if let Some(surface) = surface_for_material(level, &m.name) {
            let _ = writeln!(s, "surface = {surface}");
        }
        s.push('\n');
    }

    // [cells]: explicit dense per-cell geometry (the editor save format). Emitted
    // ONLY when the Level carries one, so cell-less levels (01-07, every shipped
    // procedural level) stay byte-identical. Authoritative on load (stamps last).
    if let Some(cells) = &level.cells {
        write_cells(&mut s, cells);
    }

    let p = &level.pedestal;
    let _ = writeln!(s, "[pedestal]");
    let _ = writeln!(s, "center = {},{},{}", p.cx, p.cy, p.cz);
    let _ = writeln!(s, "volume = {},{},{}", p.w.max(1), p.h.max(1), p.d.max(1));
    s.push('\n');

    let _ = writeln!(s, "[door]");
    let _ = writeln!(
        s,
        "pos = {},{},{}",
        level.door.x, level.door.y, level.door.z
    );
    // `[door] target_model` — inert authoring metadata (see the `[meta]` note
    // above); emitted only when the author wrote one.
    if !level.door.target_model.is_empty() {
        let _ = writeln!(s, "target_model = {}", level.door.target_model);
    }
    if !level.door.target_parts.is_empty() {
        let names = level
            .door
            .target_parts
            .iter()
            .filter_map(|&i| level.nodes.get(i).map(|n| n.name.clone()))
            .collect::<Vec<_>>()
            .join(" ");
        if !names.is_empty() {
            let _ = writeln!(s, "target_parts = {names}");
        }
    }
    // `[door] target_off` — the opt-in INVERTED win targets (L12 `exit 0`): nodes
    // that must be OFF for the win. Parsed by `re_level` but previously DROPPED by
    // the writer, so a saved L12-style level lost its inverted targets. Resolved
    // node indices → names (the inverse of the parser's name→index resolution);
    // emitted ONLY when present so an ordinary level stays byte-identical.
    if !level.door.target_off.is_empty() {
        let names = level
            .door
            .target_off
            .iter()
            .filter_map(|&i| level.nodes.get(i).map(|n| n.name.clone()))
            .collect::<Vec<_>>()
            .join(" ");
        if !names.is_empty() {
            let _ = writeln!(s, "target_off = {names}");
        }
    }
    // §4.1 inline sound-field attach on the door, if any (`Level` owns
    // exactly one, so no per-instance index).
    if let Some(f) = level
        .field_defs
        .iter()
        .find(|f| f.host == re_level::FieldHost::Door)
    {
        if let Some(line) = re_level::fields::to_inline_line(f) {
            let _ = writeln!(s, "field = {line}");
        }
    }
    s.push('\n');

    // `[sequence]` — the opt-in L11 sequence-lock win mode. Parsed by `re_level`
    // into a resolved `SequenceSpec` whose `patterns` discard the authored node
    // NAMES and the observe-window structure; the AUTHORED form
    // (`observe_names` + `step_names`) is preserved on the spec precisely so this
    // writer can reproduce the exact `[sequence]` block. Emitted ONLY when an
    // observe window is present, so a level with no `[sequence]` section stays
    // byte-identical. Exact inverse of the parser: `observe = NAME...` then one
    // `step = NAME...` line per step (an empty step is the "all OFF" step).
    if !level.sequence.observe_names.is_empty() {
        let _ = writeln!(s, "[sequence]");
        let _ = writeln!(s, "observe = {}", level.sequence.observe_names.join(" "));
        for on_names in &level.sequence.step_names {
            let _ = writeln!(s, "step = {}", on_names.join(" "));
        }
        s.push('\n');
    }

    let _ = writeln!(s, "[player]");
    let _ = writeln!(
        s,
        "spawn = {},{},{}",
        level.player.x, level.player.y, level.player.z
    );
    let _ = writeln!(s, "facing = {}", level.player.facing_yaw_col);
    s.push('\n');

    // [emitter]: one section per particle emitter. Emitted ONLY when the level
    // carries emitters, so emitter-less levels stay byte-identical. The inverse
    // of `re_level`'s `[emitter]` parser — every key it reads is written here so
    // an authored/edited emitter round-trips losslessly.
    //
    // SYNTHESIZED emitters (`EmitterSpec::synthesized == true`) are skipped.
    // `re_level` expands `[effects] snow` into a lattice of ambient emitters at
    // load and tags each one; writing them back out as well would emit BOTH
    // `snow = …` and its expansion, and the next load would then double the
    // snowfall — one more effect per save. Filtering on the per-spec flag (rather
    // than a "first N are authored" prefix) survives an editor APPENDING a newly
    // authored emitter after the synthesized ones — the common case, since
    // `ts_editor` always pushes new emitters onto the end of the list.
    for (i, e) in level.emitters.iter().filter(|e| !e.synthesized).enumerate() {
        write_emitter(&mut s, i, e);
    }

    // [prop]: the AUTHORED sections, verbatim and in order (set-dressing §3.4
    // round-trip). Their expansion — cells, lines, holos, lights, emitters —
    // lives only in the built map and is never written as its own section.
    for p in &level.props {
        let _ = writeln!(s, "[prop {}]", p.name);
        for k in &p.keys {
            let _ = writeln!(s, "{} = {}", k.key, k.value);
        }
        s.push('\n');
    }

    // [field]: sound fields, AUTHORED STANDALONE sections written verbatim and
    // in order — the same "kept exactly, resolved elsewhere" contract as
    // `[prop]` above. Inline attaches (`host != Standalone`) are written
    // inline inside their host's own section instead (the
    // `[node]`/`[pipe]`/`[door]` loops above), never here — writing both would
    // double the field on the next load.
    for f in level
        .field_defs
        .iter()
        .filter(|f| f.host == re_level::FieldHost::Standalone)
    {
        let _ = writeln!(s, "[field {}]", f.name);
        for k in &f.keys {
            let _ = writeln!(s, "{} = {}", k.key, k.value);
        }
        s.push('\n');
    }

    // Notes last: each body is captured verbatim by the parser until the next
    // header, so emitting them at the tail avoids a body bleeding into another
    // section.
    for (ni, note) in level.notes.iter().enumerate() {
        // A note a `[prop] note =` re-homed is written where it was AUTHORED: the
        // move is recomputed on every load, so baking it would drift the file.
        let (nx, ny, nz, nf) = level
            .prop_note_origins
            .iter()
            .find(|o| o.note == ni)
            .map_or((note.x, note.y, note.z, note.facing), |o| {
                (o.x, o.y, o.z, o.facing)
            });
        let kind = match note.kind {
            re_level::NoteKind::Researcher => "researcher",
            re_level::NoteKind::SystemLog => "system_log",
            re_level::NoteKind::ModelThought => "model_thought",
            re_level::NoteKind::Stencil => "stencil",
        };
        let _ = writeln!(s, "[note {kind} pos={nx},{ny},{nz} facing={nf}]");
        // Trim trailing blank lines from the body. The parser captures a note body
        // VERBATIM until the next header — INCLUDING the blank separator line we add
        // after each section. Without this trim that blank is re-absorbed into the
        // body on every reparse, so the body grows one line per save → the
        // serialization is NOT idempotent. `trim_end` collapses any accumulated
        // trailing blank to a stable fixed point (a body with no meaningful trailing
        // blank round-trips byte-identically). Leading/interior lines are untouched
        // EXCEPT: each line is passed through `re_level::escape_note_body_line`,
        // which prefixes a line that would otherwise start with `#` in column 0 (the
        // parser's `[note]` comment discriminator) with one space — without this, an
        // authored body line starting with `#` is silently dropped on the next load.
        // Every shipped body line is already indented, so this is a no-op for them.
        let body = note.text.trim_end();
        if body.is_empty() {
            let _ = writeln!(s);
        } else {
            for line in body.lines() {
                let _ = writeln!(s, "{}", re_level::escape_note_body_line(line));
            }
        }
        s.push('\n');
    }

    s
}

/// Emit a `[cells]` section — explicit, lossless, human-readable per-cell
/// geometry. Inverse of `re_level`'s `[cells]` parser.
///
/// ## Grammar (matches the `[part]` mask convention)
///
/// ```text
/// [cells]
/// size = sx,sy,sz
/// <z=0 block: sy rows of sx glyphs, y=0 first>
///
/// <z=1 block>            # blank line separates z-layers
/// ...
/// material_ids =         # OPTIONAL; only when some cell has a non-zero id
/// x,y,z = id             #   1-based colour-material id (same space as
/// ...                    #   [color_cell_ids]); persisted from side_data
/// shapes =               # OPTIONAL; only when some cell is non-Full 45° (M3c)
/// x y z wedgexy ne|nw|se|sw     #   45° vertical diagonal wall
/// x y z rampz   posx|negx|posy|negy   #   45° rising slope
/// ```
///
/// Glyph alphabet (== `ts_editor` chrome / `re_level::cell_kind_to_glyph`):
/// `.`=Empty `#`=WallSolid `/`=WallTrace `%`=WallMatrix `_`=Floor `^`=Ceiling
/// `O`=Object `H`=Hologram `@`=Portal.
fn write_cells(s: &mut String, cells: &re_level::CellGrid) {
    use std::fmt::Write as _;
    let (sx, sy, sz) = cells.size;
    let _ = writeln!(s, "[cells]");
    let _ = writeln!(s, "size = {sx},{sy},{sz}");
    for z in 0..sz {
        if z > 0 {
            // blank line advances the parser's cells_z cursor
            s.push('\n');
        }
        for y in 0..sy {
            let mut row = String::with_capacity(sx);
            for x in 0..sx {
                row.push(re_level::cell_kind_to_glyph(cells.kind_at(x, y, z)));
            }
            let _ = writeln!(s, "{row}");
        }
    }
    // material_ids sub-table — emitted only when present, sorted (z,y,x).
    if !cells.material_ids.is_empty() {
        let mut ids = cells.material_ids.clone();
        ids.sort_unstable_by_key(|&(x, y, z, _)| (z, y, x));
        let _ = writeln!(s, "material_ids =");
        for (x, y, z, id) in ids {
            let _ = writeln!(s, "{x},{y},{z} = {id}");
        }
    }
    // shapes sub-table — emitted only when at least one cell carries a non-Full
    // 45° geometry shape (M3c). All-Full ⇒ skipped ⇒ the format stays Full-only
    // and a shape-less level is byte-identical. Readable direction names
    // (`wedgexy ne`, `rampz posx`, …) via `re_level::shape_to_tokens`.
    if !cells.shapes.is_empty() {
        let mut shapes = cells.shapes.clone();
        shapes.sort_unstable_by_key(|&(x, y, z, _)| (z, y, x));
        let mut wrote_header = false;
        for (x, y, z, shape) in shapes {
            if let Some((shape_kw, dir_kw)) = re_level::shape_to_tokens(shape) {
                if !wrote_header {
                    let _ = writeln!(s, "shapes =");
                    wrote_header = true;
                }
                let _ = writeln!(s, "{x} {y} {z} {shape_kw} {dir_kw}");
            }
        }
    }
    s.push('\n');
}

/// Emit one `[vox]` section — the EXACT inverse of `re_level`'s `[vox]` parser
/// (file / offset / material / scale). Every key the parser reads is written so
/// an authored or editor-placed [`re_level::LevelVox`] round-trips losslessly.
///
/// * `file` is always written (a vox stamp with no file is meaningless, but an
///   empty string still round-trips to an empty string).
/// * `offset` is written only when non-zero (the parser defaults it to `0,0,0`).
/// * `material` uses the canonical token from [`re_level::VoxMaterial::as_str`]
///   so it re-parses to the same variant; written only when non-default (`solid`
///   is the parser default) to keep solid stamps compact.
/// * `scale` is written only when `> 1` (the parser defaults/clamps to `1`).
///
/// The section name carries the stamp's `name` (purely readable, like the
/// parser's `rest`), so a named vox keeps its name across the round-trip.
fn write_vox(s: &mut String, v: &re_level::LevelVox) {
    use std::fmt::Write as _;
    let _ = writeln!(s, "[vox {}]", v.name);
    let _ = writeln!(s, "file = {}", v.file);
    if v.off_x != 0 || v.off_y != 0 || v.off_z != 0 {
        let _ = writeln!(s, "offset = {},{},{}", v.off_x, v.off_y, v.off_z);
    }
    if v.material != re_level::VoxMaterial::Solid {
        let _ = writeln!(s, "material = {}", v.material.as_str());
    }
    if v.scale > 1 {
        let _ = writeln!(s, "scale = {}", v.scale);
    }
    s.push('\n');
}

/// Emit one `[emitter]` section — the inverse of `re_level`'s `[emitter]`
/// parser. Every key the parser reads is written here so an authored or
/// editor-tweaked [`EmitterSpec`] round-trips losslessly. Keyword spellings
/// match the parser's canonical forms (`RampOverAge`, `Off`, `KeepBg`, …) so a
/// save→load→save is idempotent. `idx` only seeds a readable section name; the
/// spec carries none.
/// Inverse of `re_level::parse_distribution` (`a..b` | `m~s` | `a..b^k`).
fn write_distribution(d: re_world::Distribution) -> String {
    match d {
        re_world::Distribution::Uniform { lo, hi } => format!("{lo}..{hi}"),
        re_world::Distribution::Gaussian { mean, std } => format!("{mean}~{std}"),
        re_world::Distribution::PowerLaw { lo, hi, k } => format!("{lo}..{hi}^{k}"),
    }
}

fn write_emitter(s: &mut String, idx: usize, e: &re_world::EmitterSpec) {
    use re_world::{
        BoldMode, ColorMode, DustVisibility, EmitterBlend, EmitterKind, EmitterOrigin, GlyphPick,
        GlyphSource, MotionKind,
    };
    use std::fmt::Write as _;

    let _ = writeln!(s, "[emitter e{idx}]");
    // KIND: emit only when NOT the `Standard` default, so every existing (Standard)
    // emitter section is byte-identical to before this key existed; a `GodRays`
    // emitter round-trips through the parser's case-insensitive `kind` key.
    match e.kind {
        EmitterKind::Standard => {} // default — omitted.
        EmitterKind::GodRays => {
            let _ = writeln!(s, "kind = GodRays");
        }
    }
    match e.origin {
        EmitterOrigin::World(p) => {
            let _ = writeln!(s, "origin_world = {} {} {}", p.x, p.y, p.z);
        }
        EmitterOrigin::Screen { col, row } => {
            let _ = writeln!(s, "origin_screen = {col} {row}");
        }
    }
    let _ = writeln!(
        s,
        "colors = 0x{:06X} 0x{:06X} 0x{:06X}",
        e.colors[0] & 0xFF_FFFF,
        e.colors[1] & 0xFF_FFFF,
        e.colors[2] & 0xFF_FFFF
    );
    let color_mode = match e.color_mode {
        ColorMode::RampOverAge => "RampOverAge",
        ColorMode::RandomPick => "RandomPick",
        ColorMode::Fixed => "Fixed",
    };
    let _ = writeln!(s, "color_mode = {color_mode}");
    let _ = writeln!(s, "family = {}", e.family);
    if let Some(glyphs) = &e.glyph_set {
        let joined: String = glyphs
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        let _ = writeln!(s, "glyph_set = {joined}");
    }
    let bold = match e.bold {
        BoldMode::Off => "Off",
        BoldMode::On => "On",
        BoldMode::ByProximity => "ByProximity",
        BoldMode::ByAge => "ByAge",
    };
    let _ = writeln!(s, "bold = {bold}");
    let _ = writeln!(
        s,
        "direction = {} {} {}",
        e.direction.x, e.direction.y, e.direction.z
    );
    let _ = writeln!(s, "spread_deg = {}", e.spread_deg);
    let _ = writeln!(s, "size = {}", e.size);
    let _ = writeln!(s, "growth = {}", e.growth);
    let _ = writeln!(s, "rate = {}", e.rate);
    if let Some(burst) = e.burst {
        let _ = writeln!(s, "burst = {burst}");
    }
    let _ = writeln!(s, "speed = {}", e.speed);
    let _ = writeln!(s, "speed_spread = {}", e.speed_spread);
    let _ = writeln!(s, "path_jitter = {}", e.path_jitter);
    let _ = writeln!(s, "seed = {}", e.seed);
    let _ = writeln!(s, "swirl = {}", e.swirl);
    let _ = writeln!(s, "lifetime = {}", e.lifetime);
    let _ = writeln!(s, "lifetime_spread = {}", e.lifetime_spread);
    let _ = writeln!(s, "accel = {} {} {}", e.accel.x, e.accel.y, e.accel.z);
    let _ = writeln!(s, "max_particles = {}", e.max_particles);
    let blend = match e.blend {
        EmitterBlend::KeepBg => "KeepBg",
        EmitterBlend::Opaque => "Opaque",
    };
    let _ = writeln!(s, "blend = {blend}");
    // ── VARIETY & RANDOMISATION ──────────────────────────────────────────────
    // Every key below is the EXACT inverse of the `re_level` parser: emitted
    // only when it differs from `EmitterSpec::default()`, so an emitter
    // authored before these fields existed stays byte-identical.
    match e.motion {
        MotionKind::Ballistic => {} // default — omitted.
        MotionKind::Curl {
            scale,
            strength,
            evolve,
        } => {
            let _ = writeln!(
                s,
                "motion = curl scale={scale} strength={strength} evolve={evolve}"
            );
        }
        MotionKind::Attract {
            target,
            strength,
            swirl,
        } => {
            let _ = writeln!(
                s,
                "motion = attract {} {} {} strength={strength} swirl={swirl}",
                target.x, target.y, target.z
            );
        }
        MotionKind::Lorenz { scale } => {
            let _ = writeln!(s, "motion = lorenz scale={scale}");
        }
        MotionKind::Flock {
            n,
            sep,
            align,
            coh,
            seek_light,
        } => {
            let _ = writeln!(
                s,
                "motion = flock n={n} sep={sep} align={align} coh={coh} seek_light={seek_light}"
            );
        }
        MotionKind::Ripple { speed } => {
            let _ = writeln!(s, "motion = ripple speed={speed}");
        }
    }
    if let Some(d) = e.lifetime_dist {
        let _ = writeln!(s, "lifetime_dist = {}", write_distribution(d));
    }
    if let Some(d) = e.speed_dist {
        let _ = writeln!(s, "speed_dist = {}", write_distribution(d));
    }
    if let Some(d) = e.size_dist {
        let _ = writeln!(s, "size_dist = {}", write_distribution(d));
    }
    if (e.randomize - 1.0).abs() > f32::EPSILON {
        let _ = writeln!(s, "randomize = {}", e.randomize);
    }
    if let Some(j) = e.color_jitter {
        let _ = writeln!(
            s,
            "color_jitter = h={} s={} l={} dance={}",
            j.h_deg, j.s, j.l, j.dance
        );
    }
    if let Some(ramp) = &e.color_ramp {
        let joined: String = ramp
            .iter()
            .map(|c| format!("0x{:06X}", c & 0xFF_FFFF))
            .collect::<Vec<_>>()
            .join(" ");
        let _ = writeln!(s, "color_ramp = {joined}");
    }
    match e.glyph_pick {
        GlyphPick::Sequential => {} // default — omitted.
        GlyphPick::Random => {
            let _ = writeln!(s, "glyph_pick = Random");
        }
    }
    match &e.glyph_source {
        GlyphSource::Family => {} // default — omitted.
        GlyphSource::Text(text) => {
            let _ = writeln!(s, "glyph_source = text \"{text}\"");
        }
    }
    if let Some(t) = e.trail {
        let dir_glyphs = if t.dir_glyphs { "on" } else { "off" };
        let _ = writeln!(
            s,
            "trail = {} fade={} dir_glyphs={}",
            t.count, t.fade, dir_glyphs
        );
    }
    if let Some(b) = e.bounce {
        let _ = writeln!(s, "bounce = {b}");
    }
    match e.visibility {
        DustVisibility::Always => {} // default — omitted.
        DustVisibility::Lit { gain } => {
            let _ = writeln!(s, "visibility = lit gain={gain}");
        }
    }
    s.push('\n');
}

/// Inverse of `re_world::sky_spec::SkySpec::parse_base` (`gradient horizon=#…
/// zenith=#… [glow=#…] [glow_width=…]`). `glow_width` is only written when it
/// differs from `SkyBase::default()`'s `0.12`, so an authored `base` with no
/// explicit width round-trips compactly (the parser re-defaults it anyway).
fn write_sky_base(b: &re_world::sky_spec::SkyBase) -> String {
    use std::fmt::Write as _;
    let mut out = format!(
        "gradient horizon=#{:06X} zenith=#{:06X}",
        b.horizon & 0xFF_FFFF,
        b.zenith & 0xFF_FFFF
    );
    if let Some(glow) = b.glow {
        let _ = write!(out, " glow=#{:06X}", glow & 0xFF_FFFF);
    }
    if (b.glow_width - re_world::sky_spec::SkyBase::default().glow_width).abs() > f32::EPSILON {
        let _ = write!(out, " glow_width={}", b.glow_width);
    }
    out
}

/// Inverse of `re_world::sky_spec::SkySpec::parse_layer` — one `layer = <kind>
/// key=value …` line per authored layer. Every knob is always written (not
/// gated on its default) because a layer entry's own presence already means
/// "authored"; unlike the top-level `[emitter]` keys, there is no legacy
/// "layer never existed" state to preserve byte-for-byte.
fn write_sky_layer(layer: &re_world::sky_spec::SkyLayerSpec) -> String {
    use re_world::sky_spec::{AuroraPalette, DataRainSource, SkyLayerSpec};
    match layer {
        SkyLayerSpec::Stars(p) => format!(
            "stars density={} twinkle={} milkyway={} pole_tilt={} wheel={} meteors={}",
            p.density, p.twinkle, p.milkyway, p.pole_tilt_deg, p.wheel, p.meteors
        ),
        SkyLayerSpec::Constellation(p) => {
            let stars = p
                .stars
                .iter()
                .map(|(u, v)| format!("({u},{v})"))
                .collect::<Vec<_>>()
                .join(" ");
            format!(
                "constellation stars={stars} link=tag:{},{},{}",
                p.link[0], p.link[1], p.link[2]
            )
        }
        SkyLayerSpec::Aurora(p) => {
            let palette = match p.palette {
                AuroraPalette::Oxygen => "oxygen",
                AuroraPalette::Violet => "violet",
                AuroraPalette::Mono => "mono",
            };
            format!(
                "aurora curtains={} band={}..{} height={} decay={} palette={palette} activity={} shimmer_hz={} surge_rate={} drift_period={}",
                p.curtains, p.band_lo, p.band_hi, p.height, p.decay, p.activity, p.shimmer_hz, p.surge_rate, p.drift_period
            )
        }
        SkyLayerSpec::Clouds(p) => format!(
            "clouds cover={} soft={} scale={} height={} wind={},{} layers={} lit=#{:06X} shade=#{:06X} haze={} bake={}",
            p.cover, p.soft, p.scale, p.height, p.wind.0, p.wind.1, p.layers,
            p.lit & 0xFF_FFFF, p.shade & 0xFF_FFFF, p.haze, if p.bake { "on" } else { "off" }
        ),
        SkyLayerSpec::Storm(p) => format!("storm cover={} rate={}", p.cover, p.rate),
        SkyLayerSpec::DataRain(p) => {
            let source = match p.source {
                DataRainSource::Binary => "binary",
                DataRainSource::Hex => "hex",
                DataRainSource::Pool => "pool",
                DataRainSource::Shards => "shards",
            };
            format!("datarain source={source} speed0={} speed1={} trail={}", p.speed0, p.speed1, p.trail)
        }
        SkyLayerSpec::Void(p) => format!("void drift={}", p.drift),
        SkyLayerSpec::Fractal(p) => format!("fractal speed={}", p.speed),
        SkyLayerSpec::Watcher(p) => format!("watcher iris=#{:06X}", p.iris & 0xFF_FFFF),
        SkyLayerSpec::Mirror(p) => format!("mirror dim={}", p.dim),
        SkyLayerSpec::Skyline(p) => {
            let style = match p.style {
                re_world::FacadeStyle::Office => "office",
                re_world::FacadeStyle::Residential => "residential",
                re_world::FacadeStyle::Industrial => "industrial",
                re_world::FacadeStyle::ServerRack => "server_rack",
                re_world::FacadeStyle::Lockers => "lockers",
            };
            format!(
                "skyline blocks={} density={} height={}..{} style={style} lit={} landmark_rate={}",
                p.blocks, p.density, p.height_lo, p.height_hi, p.lit, p.landmark_rate
            )
        }
    }
}

/// Emit one `[holo NAME]` section — the EXACT inverse of `re_level`'s `[holo]`
/// parser (`shape` / `region` / `style`). Every key the parser reads is written so
/// an authored or editor-placed [`re_level::LevelHolo`] round-trips losslessly.
///
/// * `shape` uses the canonical keyword from [`re_world::HoloShape`] (`box` /
///   `pyramid` / `sphere`), always written so the parser never falls back to its
///   `box` default for a `pyramid`/`sphere` holo.
/// * `region` matches the `(x0,y0,z0)-(x1,y1,z1)` syntax `re_level::parse_region`
///   reads (same as `[zone]`/`[zone_input]`/`[inset]`).
/// * `style` is written only when non-zero (the parser defaults it to `0`), so a
///   default-style holo stays compact.
fn write_holo(s: &mut String, h: &re_level::LevelHolo) {
    use std::fmt::Write as _;
    let _ = writeln!(s, "[holo {}]", h.name);
    let _ = writeln!(s, "shape = {}", holo_shape_to_str(h.shape));
    let _ = writeln!(
        s,
        "region = ({},{},{})-({},{},{})",
        h.x0, h.y0, h.z0, h.x1, h.y1, h.z1
    );
    if h.style != 0 {
        let _ = writeln!(s, "style = {}", h.style);
    }
    s.push('\n');
}

/// Canonical `shape = …` keyword for a [`re_world::HoloShape`] — the inverse of
/// [`re_world::HoloShape::from_str_opt`] (which accepts `box`/`cube`, `pyramid`,
/// `sphere`/`ellipsoid`; we emit the primary spelling of each).
fn holo_shape_to_str(shape: re_world::HoloShape) -> &'static str {
    match shape {
        re_world::HoloShape::Box => "box",
        re_world::HoloShape::Pyramid => "pyramid",
        re_world::HoloShape::Sphere => "sphere",
    }
}

/// Emit one `[zone_input NAME]` section — the EXACT inverse of `re_level`'s
/// `[zone_input]` parser (`node` / `region`). Drives an input node HIGH while the
/// player stands in the box. `node_index` is a finalize-resolved cache, NOT
/// authored, so it is intentionally not written — re-resolved on the next load.
fn write_zone_input(s: &mut String, zi: &re_level::ZoneInput) {
    use std::fmt::Write as _;
    let _ = writeln!(s, "[zone_input {}]", zi.name);
    let _ = writeln!(s, "node = {}", zi.node);
    let _ = writeln!(
        s,
        "region = ({},{},{})-({},{},{})",
        zi.x0, zi.y0, zi.z0, zi.x1, zi.y1, zi.z1
    );
    s.push('\n');
}

/// Emit one `[trigger]` section — the EXACT inverse of `re_level`'s `[trigger]`
/// parser (`region` / `action` / `once`). A trigger-field fires its action when
/// the player walks into the region.
///
/// * `region` is written only for the [`re_level::OverlayRegion::Box`] variant (the
///   only trigger shape; the enum is `#[non_exhaustive]`).
/// * `action` writes `activate NODE` for [`re_level::TriggerAction::ActivateNode`]
///   (by NAME — the resolved `node_index` is a finalize cache, re-resolved on
///   load, so it is intentionally not serialized) or `event LABEL` for
///   [`re_level::TriggerAction::Event`].
/// * `once` is always written (a plain bool).
fn write_trigger(s: &mut String, t: &re_level::Trigger) {
    use std::fmt::Write as _;
    let _ = writeln!(s, "[trigger]");
    if let re_level::OverlayRegion::Box { min, max } = t.region {
        let _ = writeln!(
            s,
            "region = ({},{},{})-({},{},{})",
            min.0, min.1, min.2, max.0, max.1, max.2
        );
    }
    match &t.action {
        re_level::TriggerAction::ActivateNode { name, .. } => {
            let _ = writeln!(s, "action = activate {name}");
        }
        re_level::TriggerAction::Event { label } => {
            let _ = writeln!(s, "action = event {label}");
        }
        // `#[non_exhaustive]`: a future action variant has no `.lvl` syntax yet.
        _ => {}
    }
    let _ = writeln!(s, "once = {}", t.once);
    s.push('\n');
}

/// Emit one `[zone NAME]` overlay section — the EXACT inverse of `re_level`'s
/// `[zone]` parser (`region` / `material` / `scale` / `insets`). A Scalable
/// Material Overlay zone (spec primitive 1).
///
/// * `region` is written only for the [`Region::Box`] variant (the only P0 shape;
///   the enum is `#[non_exhaustive]`). An analytic region (P1+) carries no `.lvl`
///   syntax yet and is skipped — there is no parser key to round-trip it through.
/// * `material` is written only when bound (a non-empty name); an unbound zone is
///   inert and the parser defaults it to empty.
/// * `scale` is written only when not the default `none` (so an unscaled zone stays
///   byte-identical).
/// * `insets` is written only when the zone references at least one inset.
fn write_zone(s: &mut String, z: &re_level::OverlayZone) {
    use re_level::OverlayRegion;
    use std::fmt::Write as _;
    let _ = writeln!(s, "[zone {}]", z.name);
    if let OverlayRegion::Box { min, max } = z.region {
        let _ = writeln!(
            s,
            "region = ({},{},{})-({},{},{})",
            min.0, min.1, min.2, max.0, max.1, max.2
        );
    }
    if !z.material.0.is_empty() {
        let _ = writeln!(s, "material = {}", z.material.0);
    }
    if let Some(scale) = scale_rule_to_str(&z.scale) {
        let _ = writeln!(s, "scale = {scale}");
    }
    if !z.insets.is_empty() {
        let names: Vec<&str> = z.insets.iter().map(|i| i.0.as_str()).collect();
        let _ = writeln!(s, "insets = {}", names.join(" "));
    }
    s.push('\n');
}

/// Emit one `[line NAME]` overlay section — the EXACT inverse of `re_level`'s
/// `[line]` parser (`path` / `material` / `scale`). A crisp scalable line (spec
/// primitive 2).
///
/// * `path` is the `;`-separated `x,y,z` polyline `re_level::parse_polyline` reads;
///   written only when the path has at least one vertex (an empty path is inert and
///   the parser leaves it empty).
/// * `material` / `scale` follow the same "write only when non-default" rule as
///   [`write_zone`].
fn write_line(s: &mut String, ln: &re_level::OverlayLine) {
    use std::fmt::Write as _;
    let _ = writeln!(s, "[line {}]", ln.name);
    if !ln.path.points.is_empty() {
        let path = ln
            .path
            .points
            .iter()
            .map(|p| format!("{},{},{}", p.x, p.y, p.z))
            .collect::<Vec<_>>()
            .join("; ");
        let _ = writeln!(s, "path = {path}");
    }
    if !ln.material.0.is_empty() {
        let _ = writeln!(s, "material = {}", ln.material.0);
    }
    if let Some(scale) = scale_rule_to_str(&ln.scale) {
        let _ = writeln!(s, "scale = {scale}");
    }
    s.push('\n');
}

/// Emit one `[inset NAME]` overlay section — the EXACT inverse of `re_level`'s
/// `[inset]` parser (`at_octave` / `region` / `child`). An authored inset
/// ("врезка", spec primitive 4).
///
/// * `at_octave` is written as `N` when the inclusive range is a single octave,
///   else `N-M` (the two forms `re_level::parse_octave_range` accepts).
/// * `region` is the [`Region::Box`] syntax (analytic regions are P1+, no syntax).
/// * `child` is the referenced child-scene path — always written (the parser keys
///   off `child`/`content`/`scene`; an empty reference still round-trips to empty).
fn write_inset(s: &mut String, ins: &re_level::OverlayInset) {
    use re_level::{OverlayInsetContent, OverlayRegion};
    use std::fmt::Write as _;
    let _ = writeln!(s, "[inset {}]", ins.name);
    let (lo, hi) = ins.at_octave;
    if lo == hi {
        let _ = writeln!(s, "at_octave = {lo}");
    } else {
        let _ = writeln!(s, "at_octave = {lo}-{hi}");
    }
    if let OverlayRegion::Box { min, max } = ins.region {
        let _ = writeln!(
            s,
            "region = ({},{},{})-({},{},{})",
            min.0, min.1, min.2, max.0, max.1, max.2
        );
    }
    let OverlayInsetContent::ChildScene(child) = &ins.content;
    let _ = writeln!(s, "child = {child}");
    s.push('\n');
}

/// Emit a `[hierarchy]` section — one `child -> parent` link per line. Inverse of
/// `re_level`'s `[hierarchy]` parser.
///
/// ## Grammar
///
/// ```text
/// [hierarchy]
/// node 3 -> vox 1          # node index 3's parent is vox index 1
/// emitter 0 -> node 3      # transitive: emitter under a node under a vox
/// vox 2 -> door 0          # a singleton (door/spawn/pedestal) index is 0
/// ```
///
/// The `kind index` pair is stable within a saved file's deterministic object
/// order, so a link round-trips losslessly across save→reload. Singleton kinds
/// always write index `0`.
fn write_hierarchy(s: &mut String, links: &[(re_level::HierRef, re_level::HierRef)]) {
    use std::fmt::Write as _;
    let _ = writeln!(s, "[hierarchy]");
    for (child, parent) in links {
        let _ = writeln!(
            s,
            "{} {} -> {} {}",
            child.kind.token(),
            child.index,
            parent.kind.token(),
            parent.index
        );
    }
    s.push('\n');
}

/// Serialize a [`re_level::OverlayScaleRule`] back to its `scale = …` value — the
/// inverse of `re_level::parse_scale_rule`. Returns `None` for the default
/// [`ScaleRule::None`] so the caller can SKIP the key (an unscaled zone/line stays
/// byte-identical, matching the parser default).
///
/// `self_similar` emits its [`SelfSimilarParams`] as `arms:N twist:F warp:F` — the
/// SAME `key:val` tokens the parser reads (it accepts `warp` as `octave_warp`); we
/// emit the short `warp` spelling. Sub-params are always emitted (the parser
/// defaults missing ones, but emitting all three keeps a non-default field from
/// silently reverting and makes the round-trip explicit).
fn scale_rule_to_str(rule: &re_level::OverlayScaleRule) -> Option<String> {
    use re_level::OverlayScaleRule as Sr;
    match rule {
        Sr::None => None,
        Sr::Crisp => Some("crisp".to_string()),
        Sr::SelfSimilar(p) => Some(format!(
            "self_similar arms:{} twist:{} warp:{}",
            p.arms, p.twist, p.octave_warp
        )),
    }
}

/// Like [`to_lvl_text`] but also appends a `[color_cell_ids]` section that
/// records the per-cell colour-material id assignments produced by rule 9.
///
/// ## `.lvl` format of `[color_cell_ids]`
///
/// ```text
/// [color_cell_ids]
/// 2,3,1 = 1
/// 4,5,1 = 2
/// ```
///
/// Each line is `x,y,z = material_id` where `material_id` is the 1-based index
/// into the `[[color_material]]` table (matching the id stored in
/// [`DerivedLevel::cell_material_ids`]).
///
/// `re_level` parses this section and threads the ids into each cell's
/// `side_data` via `Map::set_cell` in `build_map`.
///
/// When `cell_material_ids` is empty (no `[[color_material]]` entries) this
/// function produces the same output as [`to_lvl_text`].
pub fn to_lvl_text_with_ids(
    level: &Level,
    cell_material_ids: &BTreeMap<(i32, i32, i32), u16>,
) -> String {
    use std::fmt::Write as _;
    let mut s = to_lvl_text(level);
    if !cell_material_ids.is_empty() {
        let _ = writeln!(s, "[color_cell_ids]");
        // Sort by (z, y, x) for a deterministic, human-readable order.
        let mut entries: Vec<_> = cell_material_ids.iter().collect();
        entries.sort_unstable_by_key(|&(&(x, y, z), _)| (z, y, x));
        for (&(x, y, z), &id) in &entries {
            let _ = writeln!(s, "{x},{y},{z} = {id}");
        }
        s.push('\n');
    }
    s
}

/// Serialize a [`MaterialRecipe`] back to the `.lvl` `recipe = …` text the
/// `re_level` parser reads (round-trip inverse of `parse_recipe_cfg`/`parse_recipe`).
fn recipe_to_text(r: MaterialRecipe) -> String {
    match r {
        MaterialRecipe::Brick {
            aspect,
            mortar_w,
            grain,
        } => format!("brick aspect={aspect} mortar_w={mortar_w} grain={grain}"),
        MaterialRecipe::Circuit { scale, trace_w } => {
            format!("circuit scale={scale} trace_w={trace_w}")
        }
        MaterialRecipe::Grain {
            scale,
            speck,
            tile_eps,
        } => format!("grain scale={scale} speck={speck} tile_eps={tile_eps}"),
        MaterialRecipe::Sky { scale, density } => format!("sky scale={scale} density={density}"),
        // The glyph-only transparent hologram: round-trips to
        // `hologram line_freq=… scan_w=…` (see re_level's parse_recipe).
        MaterialRecipe::Hologram { line_freq, scan_w } => {
            format!("hologram line_freq={line_freq} scan_w={scan_w}")
        }
        // The strange loop: round-trips to `codewall panel=N` (see re_level's
        // parse_recipe and re_render::code_wall).
        MaterialRecipe::CodeWall { panel } => format!("codewall panel={panel}"),
        // ── the five pattern recipes (spec 2026-09-02 §1.5) ─────────────────
        // Each writes EVERY knob explicitly, so a load→save→load lap is the
        // identity even if a default here ever changes.
        MaterialRecipe::Scanline {
            freq,
            duty,
            warp,
            axis,
        } => format!("scanline freq={freq} duty={duty} warp={warp} axis={axis}"),
        MaterialRecipe::Truchet {
            seed,
            density,
            scale,
        } => format!("truchet seed={seed} density={density} scale={scale}"),
        MaterialRecipe::Fractal {
            arms,
            octave_warp,
            twist,
        } => format!("fractal arms={arms} octave_warp={octave_warp} twist={twist}"),
        MaterialRecipe::Substrate {
            scale,
            held,
            released,
        } => format!("substrate scale={scale} held={held} released={released}"),
        // The CHANNEL is a keyword, matching `re_level::parse_recipe`'s `mode=`.
        MaterialRecipe::Voice { scale, mode } => {
            let mode = match mode {
                re_world::VoiceMode::Hold => "hold",
                re_world::VoiceMode::Release => "release",
            };
            format!("voice scale={scale} mode={mode}")
        }
        // NOT AUTHORABLE, therefore NOT SERIALIZABLE. A `NoteWall` material is
        // SYNTHESIZED at level load from an existing `[note]`'s `pos`/`facing` (it is
        // appended to the runtime `Map`'s palette, never to a parsed `Level`), so it has
        // no `.lvl` surface syntax and there is nothing to round-trip: the `[note]`
        // section IS its serialized form. This arm exists only for exhaustiveness. It
        // emits the plain `codewall`-adjacent fallback text so that, even in the
        // impossible case of a synthesized material reaching a writer, the output stays
        // a parseable recipe rather than a malformed line.
        MaterialRecipe::NoteWall { .. } => "brick".to_string(),
        // IMAGE-TEXTURE wall: round-trips to a bare `image` keyword. The `tex_id` is a
        // build-time index, NOT serialized — the bound `[texture NAME]` is carried by
        // the material's `texture = NAME` key (emitted separately) and re-resolved to a
        // fresh id on load. So the recipe text never names an id.
        MaterialRecipe::Image { .. } => "image".to_string(),
        // The structured bay×floor window grid (spec 2026-09-27 Pass 2 item 7):
        // round-trips to `facade style=… bay=… floor=… lit=… seed=…
        // ground_band=on|off sign_row=on|off` (see re_level's parse_recipe).
        MaterialRecipe::Facade {
            style,
            bay,
            floor_h,
            lit,
            seed,
            ground_band,
            sign_row,
        } => {
            let style = match style {
                re_world::FacadeStyle::Office => "office",
                re_world::FacadeStyle::Residential => "residential",
                re_world::FacadeStyle::Industrial => "industrial",
                re_world::FacadeStyle::ServerRack => "server_rack",
                re_world::FacadeStyle::Lockers => "lockers",
            };
            let ground_band = if ground_band { "on" } else { "off" };
            let sign_row = if sign_row { "on" } else { "off" };
            format!(
                "facade style={style} bay={bay} floor={floor_h} lit={lit} seed={seed} \
                 ground_band={ground_band} sign_row={sign_row}"
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Reverse converter: Level → .vox bytes + companion .toml
// ---------------------------------------------------------------------------

/// Place a single tagged voxel into the building grid, clamping to the `.vox`
/// u8 coordinate range. Out-of-range cells are silently dropped (the caller
/// logs a warning via the export path if needed).
fn vox_grid_place(
    grid: &mut std::collections::HashMap<(u8, u8, u8), u8>,
    x: i32,
    y: i32,
    z: i32,
    tag: Tag,
) {
    if (0..=255).contains(&x) && (0..=255).contains(&y) && (0..=255).contains(&z) {
        grid.insert((x as u8, y as u8, z as u8), tag_palette_index(tag));
    }
}

/// Build a 256-entry RGBA palette from the default tag table (with the 16 tag
/// colours placed at indices 0..15, plus a transparent filler for the rest).
///
/// Palette index semantics (shared constant so the forward/reverse paths agree):
/// ```text
///  0  wall_solid    #808080
///  1  wall_trace    #00A0FF
///  2  wall_matrix   #00FF50
///  3  floor         #404040
///  4  ceiling       #202020
///  5  node_input    #FFD000
///  6  node_output   #FF3030
///  7  node_hidden   #FFFFFF
///  8  pipe          #8000FF
///  9  switch        #FF00FF
/// 10  door          #00FFFF
/// 11  player        #FFA500
/// 12  pedestal      #606060
/// 13  hologram      #40FFFF   (new — transparent glowing hologram cell)
/// 14  portal        #FF40C0   (new — non-euclidean warp-portal face)
/// 15  object        #C06000   (new — generic solid object cell)
/// ```
pub fn default_vox_palette() -> Vec<dot_vox::Color> {
    let mut pal: Vec<dot_vox::Color> = Vec::with_capacity(256);
    for t in ALL_TAGS {
        let [r, g, b] = t.default_rgb();
        pal.push(dot_vox::Color { r, g, b, a: 255 });
    }
    // Fill the remaining 256 - 16 = 240 slots with a transparent/black filler
    // so the palette is always exactly 256 entries (required by dot_vox writer).
    while pal.len() < 256 {
        pal.push(dot_vox::Color {
            r: 0,
            g: 0,
            b: 0,
            a: 0,
        });
    }
    pal
}

/// The 0-based palette index for a [`Tag`] in our canonical vox palette.
/// Delegates to [`Tag::palette_index`] — kept as a crate-private helper so
/// `vox_grid_place` stays readable.
fn tag_palette_index(tag: Tag) -> u8 {
    tag.palette_index()
}

/// Stamp a [`Level`]'s geometry and puzzle graph into a list of tagged voxels,
/// then encode them as a binary `.vox` blob (parseable by `dot_vox::load_bytes`).
///
/// The colour convention is the **same** as the forward path: each semantic
/// element is assigned the exact default-tag RGB so a round-trip is consistent
/// and `vox2lvl` re-derives the level identically.
///
/// ## What is stamped
///
/// | Source | Colour tag used |
/// |--------|-----------------|
/// | Every cell in a `[wall_decal solid]` region | `wall_solid` |
/// | Every cell in a `[wall_decal trace]` region | `wall_trace` |
/// | Every cell in a `[wall_decal matrix]` region | `wall_matrix` |
/// | Nodes — `is_input` | `node_input` |
/// | Nodes — `is_output` | `node_output` |
/// | Nodes — neither | `node_hidden` |
/// | Each pipe's path cells (excluding node endpoints) | `pipe` |
/// | Each pipe's switch cell (if any) | `switch` |
/// | Door position | `door` |
/// | Player spawn | `player` |
/// | Pedestal center cell | `pedestal` |
///
/// Floor / ceiling slabs are **not** stamped (the derivation rules auto-add them
/// from the shell), keeping the model compact. Nodes and the door anchor take
/// priority over any decal voxel at the same cell. Switch takes priority over
/// plain pipe.
///
/// `.vox` coordinates are `u8` (0–255), so rooms larger than 256 in any axis
/// will clip those voxels (dropped with a warning to stderr).
pub fn level_to_vox_bytes(level: &Level) -> Vec<u8> {
    use std::collections::HashMap;

    // Map (x,y,z) → palette index; last writer wins (priority ordering below).
    let mut grid: HashMap<(u8, u8, u8), u8> = HashMap::new();

    // 1. Decal regions (lowest priority — overridden by graph elements).
    // All six geometry surface kinds are now supported via the dedicated tags.
    for d in &level.decals {
        let tag = match d.surface_kind {
            re_world::CellKind::WallTrace => Tag::WallTrace,
            re_world::CellKind::WallMatrix => Tag::WallMatrix,
            re_world::CellKind::Hologram => Tag::Hologram,
            re_world::CellKind::Portal => Tag::Portal,
            re_world::CellKind::Object => Tag::Object,
            // WallSolid (and any other / unexpected kind) → wall_solid tag.
            _ => Tag::WallSolid,
        };
        for x in d.x0..=d.x1 {
            for y in d.y0..=d.y1 {
                for z in d.z0..=d.z1 {
                    vox_grid_place(&mut grid, x, y, z, tag);
                }
            }
        }
    }

    // 1b. Hologram objects (level.holos): the whole bounding box of each [holo]
    // section is stamped with the Hologram tag. Shape-selected interior fill is
    // not reconstructed per-cell here (re_world::HoloObject::contains is
    // private); using the bbox is lossless for Box holos and a conservative
    // superset for Pyramid/Sphere — the vox round-trip re-derives a decal bbox
    // that produces the same outer extent when re-imported, so the authored
    // shape in the .toml remains the source of truth for the exact fill.
    for h in &level.holos {
        let xlo = h.x0.min(h.x1).max(0);
        let xhi = h.x0.max(h.x1);
        let ylo = h.y0.min(h.y1).max(0);
        let yhi = h.y0.max(h.y1);
        let zlo = h.z0.min(h.z1).max(0);
        let zhi = h.z0.max(h.z1);
        for z in zlo..=zhi {
            for y in ylo..=yhi {
                for x in xlo..=xhi {
                    vox_grid_place(&mut grid, x, y, z, Tag::Hologram);
                }
            }
        }
    }

    // 2. Pipe paths (middle priority).
    // All non-endpoint path cells are stamped as `pipe`.
    // Switch cells are handled in a DEFERRED pass (step 5) after ALL other elements
    // have been stamped, so we can find a neighbor that is truly unoccupied.
    let mut switch_candidates: Vec<(i32, i32, i32)> = Vec::new();
    for pipe in &level.pipes {
        for (pi, &(x, y, z)) in pipe.path.iter().enumerate() {
            let is_endpoint = pi == 0 || pi + 1 == pipe.path.len();
            if !is_endpoint {
                vox_grid_place(&mut grid, x, y, z, Tag::Pipe);
            }
        }
        if let Some(si) = pipe.switch_cell {
            if let Some(&cell) = pipe.path.get(si) {
                switch_candidates.push(cell);
            }
        }
    }

    // 3. Nodes (higher priority — overrides any pipe/decal at the same cell).
    for n in &level.nodes {
        let tag = if n.is_input {
            Tag::NodeInput
        } else if n.is_output {
            Tag::NodeOutput
        } else {
            Tag::NodeHidden
        };
        vox_grid_place(&mut grid, n.x, n.y, n.z, tag);
    }

    // 4. Door / player / pedestal center.
    // Pedestal first so door and player can overwrite it if they share the same cell.
    let ped = &level.pedestal;
    vox_grid_place(&mut grid, ped.cx, ped.cy, ped.cz, Tag::Pedestal);
    // Door and player last — they win over pedestal at the same coordinate.
    vox_grid_place(
        &mut grid,
        level.door.x,
        level.door.y,
        level.door.z,
        Tag::Door,
    );
    vox_grid_place(
        &mut grid,
        level.player.x,
        level.player.y,
        level.player.z,
        Tag::Player,
    );

    // 5. Switch voxels — placed AFTER all other elements so we find a genuinely
    // unoccupied 6-neighbor of each switch path cell.
    // The derivation rule (rule 6) allows a switch to be ON or 6-adjacent to a
    // pipe path cell; placing it adjacent keeps the pipe run contiguous as `pipe`
    // voxels, which is required for pipe-endpoint detection.
    for (sx2, sy2, sz2) in switch_candidates {
        // Prefer neighbors in a consistent order (deterministic tie-break).
        let neighbors = [
            (sx2, sy2 + 1, sz2),
            (sx2, sy2 - 1, sz2),
            (sx2 + 1, sy2, sz2),
            (sx2 - 1, sy2, sz2),
            (sx2, sy2, sz2 + 1),
            (sx2, sy2, sz2 - 1),
        ];
        let placed = neighbors.iter().any(|&(cx, cy, cz)| {
            if !(0..=255).contains(&cx) || !(0..=255).contains(&cy) || !(0..=255).contains(&cz) {
                return false;
            }
            let key = (cx as u8, cy as u8, cz as u8);
            if grid.contains_key(&key) {
                return false;
            }
            grid.insert(key, tag_palette_index(Tag::Switch));
            true
        });
        if !placed {
            // All 6 neighbors are occupied (extremely rare in practice — e.g. a
            // switch inside a dense cluster). Fall back to placing ON the pipe cell:
            // the derivation allows `switch_at` to be ON the path itself.
            vox_grid_place(&mut grid, sx2, sy2, sz2, Tag::Switch);
        }
    }

    // Build the SIZE and XYZI.
    let (sx, sy, sz) = level.size;
    let size = dot_vox::Size {
        x: sx.min(255) as u32,
        y: sy.min(255) as u32,
        z: sz.min(255) as u32,
    };

    let mut voxels: Vec<dot_vox::Voxel> = grid
        .into_iter()
        .map(|((x, y, z), i)| dot_vox::Voxel { x, y, z, i })
        .collect();
    // Deterministic output: sort by (z, y, x).
    voxels.sort_unstable_by_key(|v| (v.z, v.y, v.x));

    let data = dot_vox::DotVoxData {
        version: 150,
        models: vec![dot_vox::Model { size, voxels }],
        palette: default_vox_palette(),
        index_map: dot_vox::DEFAULT_INDEX_MAP.to_vec(),
        materials: Vec::new(),
        scenes: Vec::new(),
        layers: Vec::new(),
    };

    let mut buf: Vec<u8> = Vec::new();
    data.write_vox(&mut buf).expect("dot_vox write_vox failed");
    buf
}

/// Serialize a [`Level`]'s non-spatial metadata to a companion `.toml` string
/// that, together with the `.vox` produced by [`level_to_vox_bytes`], can be
/// fed back to [`vox_to_level`] to reconstruct the level's geometry and graph.
///
/// ## What is lossless
///
/// All non-spatial metadata is captured verbatim: `[meta]`, `[room] ambient`,
/// per-node name/glyph/pool/pool_alt/anchor/threshold/color/is_input/is_output,
/// `[[edge]]` for every pipe (to preserve `from→to` orientation),
/// `[inputs]`, `[effects]`, `[palette]`, `[[material]]`, `[surface]`.
///
/// ## What is lossy (and why — honest accounting)
///
/// | Omitted field | Reason |
/// |---------------|--------|
/// | `[part]` / `[emit]` / `[note]` / `[holo]` / `[vox]` sections | Not in the `.toml` sidecar schema; they are preserved in the `.lvl` directly. Round-trip via `lvl2vox` → `vox2lvl` drops them. |
/// | Pipe `density` / `color_rgb` / `state` | Not in the `.toml` schema; re-derived from node colours on import. |
/// | Node `anchor == 0` | Zero is the default; omitting is lossless. |
/// | Door `target_parts` | Re-derived from output nodes (matches parser auto-default). |
/// | Pedestal box dimensions (w/h/d) | Only the center voxel is stamped; re-import sets a 1×1×1 box. |
pub fn level_to_toml(level: &Level) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();

    // [meta]
    let _ = writeln!(s, "[meta]");
    let _ = writeln!(s, "name = {:?}", level.name);
    if !level.title.is_empty() {
        let _ = writeln!(s, "title = {:?}", level.title);
    }
    if !level.ambient.is_empty() {
        let _ = writeln!(s, "ambient = {:?}", level.ambient);
    }
    s.push('\n');

    // [[node]] — one entry per node.
    for n in &level.nodes {
        let _ = writeln!(s, "[[node]]");
        let _ = writeln!(s, "pos = [{}, {}, {}]", n.x, n.y, n.z);
        let _ = writeln!(s, "name = {:?}", n.name);
        if n.glyph != '\0' && n.glyph != ' ' {
            let _ = writeln!(s, "glyph = {:?}", n.glyph.to_string());
        }
        if !n.pool.is_empty() {
            let pools = n
                .pool
                .iter()
                .map(|w| format!("{w:?}"))
                .collect::<Vec<_>>()
                .join(", ");
            let _ = writeln!(s, "pool = [{pools}]");
        }
        if !n.pool_alt.is_empty() {
            let pools = n
                .pool_alt
                .iter()
                .map(|w| format!("{w:?}"))
                .collect::<Vec<_>>()
                .join(", ");
            let _ = writeln!(s, "pool_alt = [{pools}]");
        }
        if n.anchor != 0 {
            let _ = writeln!(s, "anchor = {}", n.anchor);
        }
        let _ = writeln!(s, "color = \"#{:06X}\"", n.color_rgb & 0xFF_FFFF);
        if n.threshold != 0 {
            let _ = writeln!(s, "threshold = {}", n.threshold);
        }
        if n.is_input {
            let _ = writeln!(s, "is_input = true");
        }
        if n.is_output {
            let _ = writeln!(s, "is_output = true");
        }
        s.push('\n');
    }

    // [[edge]] — one entry per pipe to preserve orientation.
    for pipe in &level.pipes {
        let _ = writeln!(s, "[[edge]]");
        let _ = writeln!(s, "from = {:?}", pipe.from);
        let _ = writeln!(s, "to = {:?}", pipe.to);
        s.push('\n');
    }

    // [inputs]
    if !level.inputs.is_empty() {
        let _ = writeln!(s, "[inputs]");
        for (name, val) in &level.inputs {
            let _ = writeln!(s, "{} = {}", name, val);
        }
        s.push('\n');
    }

    // [effects]
    let e = &level.effects;
    if e.snow > 0.0 || e.steam || e.matrix > 0.0 || e.glitch > 0.0 {
        let _ = writeln!(s, "[effects]");
        if e.snow > 0.0 {
            let _ = writeln!(s, "snow = {}", e.snow);
        }
        if e.steam {
            let _ = writeln!(s, "steam = true");
        }
        if e.matrix > 0.0 {
            let _ = writeln!(s, "matrix = {}", e.matrix);
        }
        s.push('\n');
    }

    // [palette]
    let pal = &level.palette;
    if pal.has_fog || pal.has_fog_dist {
        let _ = writeln!(s, "[palette]");
        if pal.has_fog {
            let _ = writeln!(s, "fog_color = \"#{:06X}\"", pal.fog_color & 0xFF_FFFF);
        }
        if pal.has_fog_dist {
            let _ = writeln!(s, "fog_near = {}", pal.fog_near);
            let _ = writeln!(s, "fog_far = {}", pal.fog_far);
        }
        s.push('\n');
    }

    // [[material]] + [surface]
    if !level.materials.is_empty() {
        for m in &level.materials {
            let _ = writeln!(s, "[[material]]");
            let _ = writeln!(s, "name = {:?}", m.name);
            let _ = writeln!(s, "fg = \"#{:06X}\"", m.fg & 0xFF_FFFF);
            let _ = writeln!(s, "bg = \"#{:06X}\"", m.bg & 0xFF_FFFF);
            if let Some(g) = m.glyph {
                if let Some(ch) = char::from_u32(g) {
                    let _ = writeln!(s, "glyph = {:?}", ch.to_string());
                }
            }
            if let Some(r) = m.recipe {
                let _ = writeln!(s, "recipe = {:?}", recipe_to_text(r));
            }
            if m.anim.kind != re_world::MaterialAnimKind::None {
                let kind = match m.anim.kind {
                    re_world::MaterialAnimKind::Pulse => "pulse",
                    re_world::MaterialAnimKind::Shift => "shift",
                    re_world::MaterialAnimKind::Flicker => "flicker",
                    re_world::MaterialAnimKind::None => unreachable!(),
                };
                let _ = writeln!(
                    s,
                    "animation = {{ kind = {:?}, speed = {}, amplitude = {} }}",
                    kind, m.anim.speed, m.anim.amplitude
                );
            }
            if m.rank != 0 {
                let _ = writeln!(s, "rank = {}", m.rank);
            }
            s.push('\n');
        }
        // [surface] bindings
        let sm = &level.surface_materials;
        let has_surface = sm.wall_solid.is_some()
            || sm.wall_trace.is_some()
            || sm.wall_matrix.is_some()
            || sm.floor.is_some()
            || sm.ceiling.is_some();
        if has_surface {
            let _ = writeln!(s, "[surface]");
            if let Some(n) = &sm.wall_solid {
                let _ = writeln!(s, "wall_solid = {n:?}");
            }
            if let Some(n) = &sm.wall_trace {
                let _ = writeln!(s, "wall_trace = {n:?}");
            }
            if let Some(n) = &sm.wall_matrix {
                let _ = writeln!(s, "wall_matrix = {n:?}");
            }
            if let Some(n) = &sm.floor {
                let _ = writeln!(s, "floor = {n:?}");
            }
            if let Some(n) = &sm.ceiling {
                let _ = writeln!(s, "ceiling = {n:?}");
            }
            s.push('\n');
        }
    }

    s
}

/// Which surface kind (if any) a level binds `name` to — for emitting the
/// `surface = …` line in `to_lvl_text`. The first matching binding wins.
fn surface_for_material(level: &Level, name: &str) -> Option<&'static str> {
    let sm = &level.surface_materials;
    let hit = |b: &Option<String>| b.as_deref() == Some(name);
    if hit(&sm.wall_solid) {
        Some("wall_solid")
    } else if hit(&sm.wall_trace) {
        Some("wall_trace")
    } else if hit(&sm.wall_matrix) {
        Some("wall_matrix")
    } else if hit(&sm.floor) {
        Some("floor")
    } else if hit(&sm.ceiling) {
        Some("ceiling")
    } else {
        None
    }
}

/// The generated, freely-licensed PNG test plate for the image-texture sampler
/// (`levels/TX5-genplate.lvl`). Level-authoring asset generation belongs with the
/// rest of the offline level tooling in this crate; see the module docs for the
/// regeneration command and for why each of the plate's three properties exists.
pub mod texfixture;

#[cfg(test)]
mod tests;
