// Test-module assert_eq! on f32 values are intentional exact-equal checks.
// Allow float_cmp only in test compilation units; lib code remains denied.
#![cfg_attr(test, allow(clippy::float_cmp))]
#![warn(missing_docs)]
//! `re_world` — the voxel world + camera. Engine-agnostic.
//!
//! No terminal, no `bevy_render`. Mirrors the legacy C `World` / `world.h`
//! voxel model and `world_load_test_map` (perimeter walls at every z, floor at
//! z=0, ceiling at z=top, one interior pillar).
//!
//! Coordinate convention (from `legacy/src/world.h`):
//! * x = west(0) → east
//! * y = north(0) → south
//! * z = floor(0) → ceiling
//! * yaw = 0 faces +y (south); `dir = (sin yaw, cos yaw)`.

#[cfg(feature = "reflect")]
use bevy::prelude::ReflectComponent;
use bevy::prelude::{Component, Resource};
use glam::{Mat4, Vec3};
use re_core::{shape_solid_contains, Overlay, Shape};
use std::collections::{HashMap, HashSet};

mod store;
pub use store::{BrickmapStore, DenseStore, MapStore, ProcGenStore, VoxelStore};

mod shape_store;
pub use shape_store::ShapeStore;

// Re-export shape types so callers that only depend on re_world don't need to
// also import re_core for the shape vocabulary.
pub use re_core::{shape_surface, DiagXY, RampDir, SurfacePlane};

pub mod gen;

pub mod emitter_spec;
pub use emitter_spec::{
    BoldMode, ColorJitter, ColorMode, Distribution, DustVisibility, EmitterBlend, EmitterKind,
    EmitterOrigin, EmitterSpec, GlyphPick, GlyphSource, MotionKind, TrailSpec,
};

pub mod sky_spec;

/// The kind of a single voxel. `Empty` is the zero value so a fresh grid is
/// all-empty, matching the C `CELL_EMPTY == 0` contract.
///
/// The surface registry is generic — the engine carries no game-specific
/// vocabulary. Perimeter walls, floor/ceiling slabs, and [`Object`] cover the
/// full voxel type space; game meaning is carried by [`Map::side_data_at`], not
/// by the variant name.
///
/// [`Object`]: CellKind::Object
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CellKind {
    /// Air. A freshly allocated grid is all-empty.
    #[default]
    Empty,
    /// Opaque flat-shaded wall.
    WallSolid,
    /// Wall with the circuit-trace decal surface.
    WallTrace,
    /// Wall with the matrix-rain decal surface.
    WallMatrix,
    /// Interior floor (z=0).
    Floor,
    /// Interior ceiling (z=top).
    Ceiling,
    /// A generic solid object-surface cell whose game role is carried by
    /// [`Map::side_data_at`]. The engine treats it as an opaque non-decal solid
    /// — the DDA stops on it and the renderer maps it to `WallOther`. Any
    /// game-defined tag (e.g. an identifier used to distinguish different
    /// interactive elements) is stored in the cell's `side_data` field, not in
    /// the variant itself.
    Object,
    /// A **HOLOGRAM** voxel: a TRANSPARENT object cell the raycaster passes
    /// THROUGH (it is NOT [`is_solid`](Self::is_solid)) — the scene behind it
    /// still renders. The renderer treats the *silhouette / edges* of a connected
    /// block of hologram cells as a glowing depth-coloured contour wireframe with
    /// scanlines (a sci-fi projection); the fill stays see-through. A small
    /// game-defined STYLE tag (colour/look variant) rides in the cell's
    /// `side_data`. Stamping a box of these = a holographic cube; an arbitrary
    /// voxel set = an arbitrary hologram mesh. Additive: a map with no hologram
    /// cells renders exactly as before.
    Hologram,
    /// A **non-euclidean warp-portal** face. The DDA **stops** on this cell
    /// (it IS [`is_solid`](Self::is_solid)) so the renderer gets a hit it can
    /// re-base through the portal's `delta` transform into the destination
    /// region. [`Map::portal_for_cell`] returns the index into
    /// [`Map::portals`] for the stamped cell.
    ///
    /// Stage 1 MVP: portal cells are collidable (the player is blocked by them
    /// like a wall). Stage 2 adds plane-crossing detection + seamless teleport
    /// and flips these cells non-solid for pass-through.
    Portal,
}

impl CellKind {
    /// True for any cell the raycaster's DDA should stop on.
    ///
    /// Walls (solid/trace/matrix) and generic [`Object`] cells are all opaque
    /// hits. Floor and ceiling slabs are handled by the raycaster's vertical
    /// bounds logic and are not returned by `is_solid`.
    ///
    /// [`Object`]: CellKind::Object
    #[inline]
    pub fn is_solid(self) -> bool {
        matches!(
            self,
            CellKind::WallSolid
                | CellKind::WallTrace
                | CellKind::WallMatrix
                | CellKind::Object
                | CellKind::Portal
        )
    }

    /// True for the perimeter/decal wall surfaces (not floor/ceiling/object).
    #[inline]
    pub fn is_wall(self) -> bool {
        matches!(
            self,
            CellKind::WallSolid | CellKind::WallTrace | CellKind::WallMatrix
        )
    }

    /// True for a [`Hologram`](Self::Hologram) cell — a TRANSPARENT object the
    /// DDA passes through. Distinct from [`is_solid`](Self::is_solid) (always
    /// `false` for a hologram) so the caster can collect a hologram crossing
    /// without stopping the primary ray on it.
    #[inline]
    pub fn is_hologram(self) -> bool {
        matches!(self, CellKind::Hologram)
    }

    /// True for cells that **block player movement** (collision predicate).
    ///
    /// This deliberately diverges from [`is_solid`](Self::is_solid) for
    /// [`Portal`](Self::Portal) cells:
    ///
    /// * **Raycaster** uses `is_solid` — portals are solid so the DDA stops and
    ///   the renderer gets a hit to re-base through the portal's `delta`.
    /// * **Collision** uses `is_collidable` — portals are NOT collidable so the
    ///   player can walk into the opening; seamless teleport (Stage 2) then
    ///   warps the pose across the portal plane.
    ///
    /// Everything else is unchanged: the walls, objects that were collidable
    /// before remain so; hologram and empty cells remain non-collidable.
    #[inline]
    pub fn is_collidable(self) -> bool {
        matches!(
            self,
            CellKind::WallSolid | CellKind::WallTrace | CellKind::WallMatrix | CellKind::Object // Portal is intentionally ABSENT: walkable for the player, solid for the raycaster.
        )
    }

    /// Stable `u8` encoding for compact brickmap serialization (W1). Every
    /// variant is mapped; an unknown byte decodes as `Empty` (safe default).
    ///
    /// The mapping is an append-only stable contract — never reorder.
    #[inline]
    pub fn to_u8(self) -> u8 {
        match self {
            CellKind::Empty => 0,
            CellKind::WallSolid => 1,
            CellKind::WallTrace => 2,
            CellKind::WallMatrix => 3,
            CellKind::Floor => 4,
            CellKind::Ceiling => 5,
            CellKind::Object => 6,
            CellKind::Hologram => 7,
            CellKind::Portal => 8,
        }
    }

    /// Decode a byte written by [`to_u8`](Self::to_u8). Unknown bytes decode
    /// as `Empty` — never panics, never corrupts bounds.
    #[inline]
    pub fn from_u8(b: u8) -> CellKind {
        match b {
            0 => CellKind::Empty,
            1 => CellKind::WallSolid,
            2 => CellKind::WallTrace,
            3 => CellKind::WallMatrix,
            4 => CellKind::Floor,
            5 => CellKind::Ceiling,
            6 => CellKind::Object,
            7 => CellKind::Hologram,
            8 => CellKind::Portal,
            _ => CellKind::Empty, // future-proof: unknown byte → safe Empty
        }
    }
}

/// Which procedural texture recipe an authored [`MaterialDef`] uses, mirrored as
/// engine-agnostic plain data so `re_world` (and `re_level`) can carry a level's
/// material set without depending on the renderer. `re_render` maps this 1:1 onto
/// its own `ascii_shader::TextureRecipe`.
///
/// The variants and their parameter names match the renderer's recipes exactly;
/// adding a recipe is a new variant here + its arm in the renderer's converter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MaterialRecipe {
    /// Brick / panel masonry.
    Brick {
        /// Bricks per cell-width.
        aspect: f32,
        /// Seam width, in brick-units.
        mortar_w: f32,
        /// fBm body mottling amount.
        grain: f32,
    },
    /// Circuit board (Worley pads + Voronoi traces).
    Circuit {
        /// Worley cells per world-cell.
        scale: f32,
        /// Voronoi-edge (trace) width.
        trace_w: f32,
    },
    /// World-pinned floor grain.
    Grain {
        /// Grain lattice density.
        scale: f32,
        /// Fraction of cells carrying a speck.
        speck: f32,
        /// Tile-grid epsilon (integer-tile line thinness).
        tile_eps: f32,
    },
    /// Star sky / ceiling.
    Sky {
        /// Star lattice density.
        scale: f32,
        /// Star fraction (how many lattice cells light up as a star).
        density: f32,
    },
    /// `Hologram`: a GLYPH-ONLY transparent surface — fg-coloured horizontal
    /// scanline glyphs on a near-void (black) bg, scrolling vertically over time so
    /// it reads as a see-through, glowing hologram rather than a filled wall.
    /// `line_freq` = scan lines per world-cell (vertical density); `scan_w` =
    /// fraction of each line-period that inks a glyph (line thickness). Plain data;
    /// the renderer owns the glyph set + the vertical-scroll animation.
    Hologram {
        /// Scan lines per world-cell (vertical density).
        line_freq: f32,
        /// Fraction of each line-period that inks a glyph (line thickness).
        scan_w: f32,
    },
    /// `CodeWall`: lay the engine's OWN annotated source text onto the wall face
    /// as readable monospace glyphs. `panel` selects WHICH embedded source the
    /// wall shows (see the renderer's `code_wall` panel registry — panel 0 is the
    /// renderer that draws code on the wall, drawn on the wall). Engine-agnostic
    /// plain data: `re_world` carries only the panel index; the renderer owns the
    /// actual text + glyph mapping. The fourth wall, made of literal walls.
    CodeWall {
        /// Which embedded source file the wall shows (renderer panel index).
        panel: u32,
    },
    /// `NoteWall`: lay ONE authored `[note]` placard's body onto a single wall
    /// cell face as readable monospace text — the in-world voice of the level.
    ///
    /// `note` indexes the LEVEL's parsed note list (i.e. the note-panel space of
    /// the renderer's text-surface registry, which the game layer fills at load in
    /// the same order). `face` is the wall face the text is painted on, named by
    /// its OUTWARD normal — the side the reader stands on.
    ///
    /// **Why `face` is not optional.** A material is bound to a CELL, so without a
    /// face the note would render on all four faces of that cell, including the
    /// reverse side in the adjoining room (mirrored). The renderer gates on the
    /// shaded face's true normal when it has one, and on the coarse NS/EW `side`
    /// bit when it does not — see [`re_core::FaceDir`].
    ///
    /// Engine-agnostic plain data: `re_world` carries only the index + the face;
    /// the renderer owns the text grid and the glyph mapping. Synthesized at level
    /// load from the note's authored `pos`/`facing` — there is no `.lvl` key for
    /// it, so the level format and its round-trip are untouched.
    NoteWall {
        /// Index into the level's parsed note list.
        note: u32,
        /// The wall face the text is painted on, named by its outward normal.
        face: re_core::FaceDir,
    },
    /// `Scanline`: a FILLED directional-band surface — the smart-material primitive
    /// behind CRT/monitor screens, brushed metal, woven fabric, and striped/grated
    /// looks. Unlike [`Hologram`](Self::Hologram) the gaps between bands keep a real
    /// (dark) background fill, so the surface reads as a solid panel.
    ///
    /// `freq` = bands per world-cell; `duty` = inked fraction of each band period
    /// `[0,1]` (line thickness vs. gap); `warp` = fBm jitter `[0,1]` that breaks
    /// perfectly straight lines into a hand-made grain; `axis` selects the band
    /// direction — `0` = HORIZONTAL (CRT raster / brushed grain), `1` = VERTICAL
    /// (grate / blind), `2` = WOVEN cross-hatch (fabric / mesh).
    ///
    /// `.lvl` syntax: `recipe = scanline freq=9 duty=0.55 warp=0.05 axis=0`.
    Scanline {
        /// Bands per world-cell.
        freq: f32,
        /// Inked fraction of each band period `[0,1]` (line thickness vs. gap).
        duty: f32,
        /// fBm jitter `[0,1]` that breaks perfectly straight lines into a
        /// hand-made grain.
        warp: f32,
        /// Band direction: `0` horizontal, `1` vertical, `2` woven. Parsed and
        /// serialized as a bare integer; the parser clamps it to `0..=2` so the
        /// value that round-trips is always one the renderer names.
        axis: u8,
    },
    /// `Truchet`: EDGE-MATCHED conduit tiling — box-drawing runs that CONTINUE
    /// across cell boundaries instead of speckling independently. `seed` selects
    /// the pattern; `density` `[0,1]` is the fraction of tile edges crossed (`0` =
    /// blank, `1` = a full `┼` lattice); `scale` is TILES PER WORLD CELL on both
    /// axes.
    ///
    /// `.lvl` syntax: `recipe = truchet seed=0 density=0.5 scale=4`.
    Truchet {
        /// Selects the pattern.
        seed: i32,
        /// Fraction of tile edges crossed `[0,1]` (`0` blank, `1` full lattice).
        density: f32,
        /// Tiles per world cell on both axes.
        scale: f32,
    },
    /// `Fractal`: a log-polar self-similar "portal" — nested rings / spiral arms
    /// that zoom and spin without a visible seam. `arms` = spiral arms (`0` = pure
    /// concentric rings), `octave_warp` = fBm enrichment strength `[0,1]`, `twist`
    /// = arm curl (log-spiral vs. straight rays). The renderer's log-polar
    /// zoom/spin phase is driven by an `AnimKind::Zoom`/`Spin`/`SpinZoom`
    /// animation, which the `.lvl` `animation =` key cannot yet select (it offers
    /// `none|pulse|shift|flicker`), so an authored fractal is STATIC — a still
    /// portal. Selecting it via `preset = portal_zoom` still animates.
    ///
    /// `.lvl` syntax: `recipe = fractal arms=5 octave_warp=0.3 twist=0.9`.
    Fractal {
        /// Spiral arms (`0` = pure concentric rings).
        arms: u8,
        /// fBm enrichment strength `[0,1]`.
        octave_warp: f32,
        /// Arm curl (log-spiral vs. straight rays).
        twist: f32,
    },
    /// `Substrate`: the substrate's own CELL ALPHABET as a wall surface — `░`
    /// unclaimed, `█` held, `▒` released (a scar), `✕` burned. `scale` = alphabet
    /// cells per world unit; `held` is the fraction of the field ever claimed;
    /// `released` is the fraction OF THAT HELD SET since let go (so raising
    /// `released` ERODES the held set rather than flipping it). Static: a pure
    /// function of the world cell and the two scalars.
    ///
    /// `.lvl` syntax: `recipe = substrate scale=3 held=0.55 released=0.3`.
    Substrate {
        /// Alphabet cells per world unit.
        scale: f32,
        /// Fraction of the field ever claimed.
        held: f32,
        /// Fraction OF THE HELD SET since let go (raising this erodes the held
        /// set rather than flipping it).
        released: f32,
    },
    /// `Voice`: [`Substrate`](Self::Substrate)'s cell alphabet driven by PUZZLE
    /// STATE instead of by two authored scalars — the level's monotone solve-progress
    /// mark arrives on the state bus and is spent on one channel or the other per
    /// [`VoiceMode`]. `scale` is the alphabet density, exactly as on `Substrate`.
    ///
    /// `.lvl` syntax: `recipe = voice scale=3 mode=hold` (or `mode=release`).
    Voice {
        /// Alphabet density, exactly as on [`Substrate`](Self::Substrate).
        scale: f32,
        /// Which channel the puzzle's solve-progress mark is spent on.
        mode: VoiceMode,
    },
    /// `Image`: sample a decoded RGBA8 [`re_core::TextureImage`] (e.g. a PNG/WebP/GIF)
    /// for the wall face. `tex_id` indexes [`Map::textures`] — the per-level texture
    /// table `re_level` fills by decoding each `[texture NAME]` file. Engine-agnostic
    /// plain data: `re_world`/`re_level` carry only the id; `re_render` resolves it to
    /// the actual `Arc<TextureImage>` via `attach_image_textures` and samples the
    /// nearest texel. The 1:1 renderer mirror is
    /// `TextureRecipe::Image { tex_id, span, subcell, fit }`.
    ///
    /// `span` is the WORLD-ANCHORED UV mapping (M3a): when `Some`, it is used
    /// directly (expert override — `fit` is ignored). When `None`, the renderer
    /// derives an `ImageSpan` from the textured wall's real world extent and the
    /// image aspect ratio using the `fit` rule. Default behaviour for old call
    /// sites: `fit = Fit::Contain` (aspect-preserving, whole image inside wall).
    Image {
        /// Index into [`Map::textures`] — the per-level texture table.
        tex_id: u32,
        /// World-anchored UV mapping override (`Some` bypasses `fit` entirely).
        span: Option<ImageSpan>,
        /// `tile = a b` — how many times the figure REPEATS across the mapped span
        /// (`a` along the surface, `b` up it). `(1.0, 1.0)` — the default and what
        /// every level authored before this key existed — means one copy, and the
        /// renderer skips the multiply entirely, so it is byte-identical. Both
        /// factors are guaranteed finite and `> 0` by the `.lvl` parser.
        tile: (f32, f32),
        /// `relief = height_scale=… normal_strength=…` — when `Some`, this image is a
        /// LIT SURFACE rather than a self-lit decal: the renderer flips the recipe off
        /// `self_lit`, routes its height field through the bound texture's
        /// [`re_core::TextureImage::height_map`], and lights the result with these two
        /// knobs. `None` (the default) keeps the historical self-lit block.
        relief: Option<ImageRelief>,
        /// SUB-CELL detail mode (M3b): render the image at HIGHER-than-cell resolution
        /// by sampling the world-anchored texture at multiple sub-positions inside each
        /// screen cell. `Off` ⇒ the legacy per-cell solid block (byte-identical);
        /// `Quad` ⇒ 2×2 quadrant (two colours/cell); `Braille` ⇒ 2×4 Braille coverage
        /// (sharp monochrome silhouette); `Ascii` ⇒ luminance ramp + Sobel-direction
        /// edge glyphs; `Contour` ⇒ DoG-hybrid clean contour lines (hero look);
        /// `Lines` ⇒ line-only: non-edge cells inked ≈ background so only contours pop.
        /// Plain data; the renderer owns the blit and the `EdgeField` precompute.
        subcell: SubcellMode,
        /// How the image is mapped onto the wall surface when no explicit `span`
        /// override is given. An explicit `Some(span)` overrides `fit` entirely;
        /// when `span` is `None` the renderer derives the `ImageSpan` from the wall's
        /// world extent and the image aspect per this rule. Defaults to
        /// [`Fit::Contain`] (aspect-preserving, whole image fits inside the wall).
        fit: Fit,
    },
    /// `Facade`: a structured BAY × FLOOR window grid — office/residential/
    /// industrial/server-rack/locker facades, doubling as an interior surface
    /// (a rack of LEDs, a wall of lockers) since the same bay×floor lattice
    /// applies either way.
    ///
    /// `bay` and `floor_h` are REPEATS PER WORLD-CELL on the wall's along/up axes
    /// (so `bay=3 floor=4` tiles 3 window columns × 4 window rows across one world
    /// cell, exactly like [`Truchet`](Self::Truchet)'s `scale`). `lit` is the
    /// fraction of window units that hash LIT (bright) vs. dark; the choice is
    /// per-unit and SEEDED (`seed` salts the per-building hash), not per-frame
    /// noise, so the pattern is stable frame to frame and only moves if the
    /// author changes the knobs. `ground_band` draws a distinct storefront/curb
    /// band across the wall's bottom world-cell height; `sign_row` draws one
    /// thin accent band just above it. Mullion/sill framing and the per-style
    /// glyph table are the renderer's — see `re_render::facade`.
    ///
    /// `.lvl` syntax: `recipe = facade style=office bay=3 floor=4 lit=0.35
    /// [seed=0] [ground_band=on|off] [sign_row=on|off]`.
    Facade {
        /// Which window-unit glyph table to draw from (see [`FacadeStyle`]).
        style: FacadeStyle,
        /// Window columns repeated per world-cell along the wall.
        bay: f32,
        /// Window rows repeated per world-cell up the wall.
        floor_h: f32,
        /// Fraction of window units that hash lit (bright) vs. dark.
        lit: f32,
        /// Salts the per-building lit/dark hash.
        seed: i32,
        /// Draw a distinct storefront/curb band across the wall's bottom.
        ground_band: bool,
        /// Draw one thin accent band just above the ground band.
        sign_row: bool,
    },
}

/// Which window-unit GLYPH TABLE a [`MaterialRecipe::Facade`] surface draws from —
/// a small catalog covering both exterior facades and interior set-dressing that
/// shares the same bay×floor lattice (server racks, locker walls).
///
/// `.lvl` syntax: `recipe = facade style=<office|residential|industrial|
/// server_rack|lockers> ...`. Default [`Office`](Self::Office) — plain rectangular
/// lit/dark panes, the most neutral reading when unspecified.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FacadeStyle {
    /// Evenly spaced rectangular panes, calm mullions — office tower windows.
    #[default]
    Office,
    /// Smaller, more varied units — apartment windows (balcony rails read via
    /// the same frame glyph at a tighter bay).
    Residential,
    /// Fewer, heavier openings with a grimier ink — warehouse / plant facades.
    Industrial,
    /// Dense small dot units on thin vertical rails — server-rack LED banks
    /// (an interior use: bind this recipe to an equipment-room wall).
    ServerRack,
    /// Vertical door-like units (closed vs. in-use) — locker-room walls
    /// (another interior use).
    Lockers,
}

/// RELIEF knobs for a [`MaterialRecipe::Image`] surface — the plain-data mirror of
/// the two `re_render::LightModel` fields an authored `relief =` sets.
///
/// Authored as `[material] relief = height_scale=1.2 normal_strength=0.9`. Its mere
/// PRESENCE is the switch: an image material with `Some(..)` here stops being a
/// self-lit decal and starts being lit like every procedural material, with its
/// height field read from the bound texture's 8-bit relief channel
/// (`re_core::TextureImage::height_map`). An image with no height channel still
/// reads as flat — the knobs scale a constant field to zero slope — which is the
/// defined, non-garbage outcome for "author asked for relief, supplied no map".
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageRelief {
    /// How tall the height field reads, in tangent units. Larger ⇒ steeper relief.
    /// `0` ⇒ flat. Mirrors `re_render::LightModel::height_scale`.
    pub height_scale: f32,
    /// Scales the tangent-space normal tilt before it combines with the face normal.
    /// Larger ⇒ bumps lean further toward the light. Mirrors
    /// `re_render::LightModel::normal_strength`.
    pub normal_strength: f32,
}

impl ImageRelief {
    /// The relief an author gets from a bare `relief =` with no knobs: a full-strength
    /// unit relief. Deliberately `1.0`/`1.0` rather than the neutral material default
    /// (0.6/0.8) — a photo's height map is a real measured surface, so asking for
    /// relief at all should show it, and the author scales DOWN from there.
    pub const DEFAULT: ImageRelief = ImageRelief {
        height_scale: 1.0,
        normal_strength: 1.0,
    };
}

impl Default for ImageRelief {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// The default `tile =` repeat for an image material: ONE copy across the span.
/// Named so the "is this material tiled at all?" test is a comparison against a
/// constant rather than a pair of scattered `1.0` literals.
pub const IMAGE_TILE_NONE: (f32, f32) = (1.0, 1.0);

/// Which CHANNEL a [`MaterialRecipe::Voice`] wall spends the level's solve-progress
/// mark on. Plain-data mirror of the renderer's `substrate::VoiceMode`; the two
/// readings of one scalar are the whole point of the pair.
///
/// | mode | `held` | `released` | reads as |
/// |---|---|---|---|
/// | [`Hold`](Self::Hold) | `mark` | `0` | an empty wall FILLING with `█` |
/// | [`Release`](Self::Release) | `1` | `mark` | a whole wall ERODING to `▒` |
///
/// `.lvl` syntax: `recipe = voice scale=3 mode=hold` / `mode=release`. Default
/// [`Hold`](Self::Hold) — the additive reading, matching the `mnemos_hold` preset.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum VoiceMode {
    /// `held = mark`, `released = 0` — an unclaimed field converting to `█`.
    #[default]
    Hold,
    /// `held = 1`, `released = mark` — a whole field eroding to `▒` (and, past the
    /// end of the ramp, to `✕`).
    Release,
}

/// SUB-CELL detail mode for an [`MaterialRecipe::Image`] surface (M3b). Plain-data
/// mirror of the renderer's `ascii_shader::SubcellMode`. Default [`Off`](Self::Off)
/// keeps the legacy per-cell solid `█` block, so every existing image material is
/// byte-identical.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SubcellMode {
    /// One solid block per cell (the M3a look). Default — byte-identical.
    #[default]
    Off,
    /// 2×1 half-block: two sub-texels → `▀`/`▄`/`█`/space + a two-colour split.
    /// The cheapest rung, and the one that makes a sub-texel roughly SQUARE — a
    /// terminal cell is about 2:1 tall, so one vertical split is the aspect
    /// correction. Every glyph it emits is a Block Element, so it needs no
    /// capability gate and is the degrade target for [`Sextant`](Self::Sextant).
    Half,
    /// 2×2 quadrant: four sub-texels → one of 16 block glyphs + a two-colour split.
    /// Doubles effective resolution; two colours per cell.
    Quad,
    /// 3×2 sextant: six sub-texels → one of the 60 `U+1FB00..=U+1FB3B` BLOCK SEXTANT
    /// glyphs (or the four Block Elements Unicode already had) + a two-colour split.
    ///
    /// Three times the sub-texels of [`Quad`](Self::Quad) while still being a BLOCK
    /// geometry, so unlike [`Braille`](Self::Braille) it keeps TWO colours per cell.
    /// That is the trade this rung exists to fix: Braille buys resolution by giving
    /// every one of its eight dots a single shared ink.
    ///
    /// **Requires terminal support** for the Symbols for Legacy Computing sextants.
    /// Coverage is good but not universal (Ghostty/xterm.js ~100% of the important
    /// characters, VTE ~94%, urxvt 0%), and most terminals DRAW these themselves
    /// rather than relying on font paths — so it is a terminal capability, not a font
    /// question. Degrade to [`Half`](Self::Half) or [`Quad`](Self::Quad) when absent.
    Sextant,
    /// 2×4 Braille: eight sub-texels → an 8-dot coverage glyph (U+2800..28FF). One ink
    /// colour, sharp 2×4 silhouette — the alpha-cut-out high-detail-shape mode.
    Braille,
    /// Luminance ramp (` .:-=+*#%@`) fill; on a Sobel edge, swap to the direction
    /// glyph (`/ \ | - _`). Lightest ASCII algorithm; the stepping-stone baseline.
    /// The renderer precomputes a per-texture Sobel gradient once at load time.
    Ascii,
    /// Same luminance-ramp fill as [`Ascii`](Self::Ascii) but the edge layer uses a
    /// **Difference-of-Gaussians** `EdgeField` → thin, clean contour lines.
    /// The Acerola DoG+Sobel hybrid look; the hero quality mode.
    Contour,
    /// Line-only / ink mode: EVERY cell is assigned a directional glyph, but
    /// non-edge cells are inked close to their background colour (low contrast) so
    /// only the contour glyphs visually pop — a clean line drawing over flat colour
    /// fields. Requires the same `EdgeField` as [`Contour`](Self::Contour).
    Lines,
}

/// How a [`MaterialRecipe::Image`] figure is mapped onto its wall surface when no
/// explicit `span` override is given. Plain-data mirror of
/// `re_render::ascii_shader::Fit`.
///
/// When `span` is `Some`, the authored `ImageSpan` is used directly and `fit` is
/// ignored. When `span` is `None`, `re_render::registry_for_map` derives an
/// `ImageSpan` from the textured wall's real world extent and the image aspect ratio
/// according to this rule, baking it into the resolved `Material` so the per-cell
/// hot path is unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Fit {
    /// Fill the wall exactly; aspect ratio ignored (the image distorts).
    Stretch,
    /// Preserve aspect; the WHOLE image fits inside the wall (margins around it).
    ///
    /// This is the default. A `(u, v)` outside `[0, 1]` falls in the margin and
    /// is treated as transparent/edge (the existing clamp/cut-out rule applies).
    #[default]
    Contain,
    /// Preserve aspect; fill the wall completely, cropping the overflow.
    Cover,
}

/// WORLD-ANCHORED UV span for an [`MaterialRecipe::Image`] surface (M3a). Maps the
/// texture's `[0,1]×[0,1]` UV box onto a rectangle of WORLD space so ONE copy of
/// the figure spans the whole textured surface regardless of how many cells wide /
/// tall it is. The renderer computes, at each wall hit:
///
/// * the world ALONG coordinate (the wall's variable axis — world Y on an NS face,
///   world X on an EW face), then `u = (along - along_min) / along_size`;
/// * the world UP coordinate (`wall_world_y`, the texel's world height), then
///   `v = (up_top - world_up) / up_size` so `v = 0` at the TOP of the span
///   (matching the image's top-left origin) and `v = 1` at the bottom.
///
/// A `(u, v)` outside `[0,1]` is OUTSIDE the figure → treated as transparent by the
/// cut-out (shows the backdrop) and clamped to the edge by the opaque-paint path.
/// Plain data; the 1:1 renderer mirror is `ascii_shader::ImageSpan`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageSpan {
    /// World coordinate (along the wall's variable axis) where `u = 0`.
    pub along_min: f32,
    /// World extent (along the wall's variable axis) the full `u` range covers.
    /// Must be `> 0`; a non-positive size falls back to per-cell tiling.
    pub along_size: f32,
    /// World UP coordinate (height) where `v = 0` (TOP of the figure).
    pub up_top: f32,
    /// World vertical extent the full `v` range covers (`v = 1` at `up_top - up_size`).
    /// Must be `> 0`; a non-positive size falls back to per-cell tiling.
    pub up_size: f32,
}

/// Sky image set for a level that wants an IMAGE-backed skybox instead of
/// (or in addition to) the procedural gradient sky.
///
/// Mirrors the two-region sky model in `re_render::sky`:
/// * `cylinder` — the 360° wraparound horizon band sampled for `pitch ≤ θ_cap`.
///   Recommended size ≈ 720 × 96 (density-matched to `re_render::sky::CYL_W/CYL_H`);
///   any size is accepted — the renderer normalises UVs into `[0,1]`.
/// * `cap_top` — the zenith disc sampled for `pitch > θ_cap` (polar coordinates).
///   Recommended size ≈ 256 × 256 (`re_render::sky::CAP_DIM`); again any size is
///   accepted.
/// * `cap_bottom` — the nadir disc for below-horizon rays (optional; very few levels
///   need a textured floor-sky; defaults to `None` → the procedural/floor path wins).
///
/// All three slots are `Option<Arc<TextureImage>>`.  A `None` slot means "use the
/// procedural sky source for that region" — so a level that only authors a `cylinder`
/// image still gets the procedural cap, and a map with no `SkyImages` entries renders
/// byte-identically to today.
///
/// **Storage only.**  Resize, UV mapping, and blending live in `re_render::sky` — this
/// struct is pure data.  `re_level` populates it via the `[sky]` section setters;
/// `re_render` reads it through [`Map::sky_images`].
#[derive(Clone, Default)]
pub struct SkyImages {
    /// The 360° wraparound horizon band (cylinder region, `pitch ≤ θ_cap`).
    ///
    /// `None` ⇒ use the procedural sky for this region (byte-identical to today).
    pub cylinder: Option<std::sync::Arc<re_core::TextureImage>>,
    /// The zenith disc (top cap, `pitch > θ_cap`).
    ///
    /// `None` ⇒ use the procedural sky for this region.
    pub cap_top: Option<std::sync::Arc<re_core::TextureImage>>,
    /// The nadir disc (bottom cap, for below-horizon rays; optional).
    ///
    /// `None` ⇒ the procedural or floor-sky path handles below-horizon rays (default).
    pub cap_bottom: Option<std::sync::Arc<re_core::TextureImage>>,
}

impl std::fmt::Debug for SkyImages {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SkyImages")
            .field(
                "cylinder",
                &self
                    .cylinder
                    .as_ref()
                    .map(|t| format!("{}x{}", t.width, t.height)),
            )
            .field(
                "cap_top",
                &self
                    .cap_top
                    .as_ref()
                    .map(|t| format!("{}x{}", t.width, t.height)),
            )
            .field(
                "cap_bottom",
                &self
                    .cap_bottom
                    .as_ref()
                    .map(|t| format!("{}x{}", t.width, t.height)),
            )
            .finish()
    }
}

/// Which authored animation an authored [`MaterialDef`] plays. Plain-data mirror
/// of the renderer's `ascii_shader::AnimKind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MaterialAnimKind {
    /// No animation — a static look.
    #[default]
    None,
    /// Sinusoidal brightness breathing.
    Pulse,
    /// Scroll the texture coordinate over time (matrix-rain flow).
    Shift,
    /// Pseudo-random brightness jitter.
    Flicker,
}

/// An authored animation block (kind + speed + amplitude), plain data.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct MaterialAnim {
    /// Which animation shape plays (see [`MaterialAnimKind`]).
    pub kind: MaterialAnimKind,
    /// Playback speed, in cycles per second.
    pub speed: f32,
    /// Effect strength (e.g. brightness swing for `Pulse`/`Flicker`).
    pub amplitude: f32,
}

/// Per-level atmosphere: an authored distance-fog override plus a `matrix`
/// ambient density knob. Plain data, engine-agnostic (no terminal / no render
/// dep) — `re_level::build_map` forwards a level's `[palette]` fog and
/// `[effects] matrix` onto the [`Map`], and the renderer reads it back to tint
/// distant cells toward `fog_color` between `fog_near` and `fog_far`.
///
/// Every field is **optional** (`None` ⇒ the renderer keeps its built-in
/// constant), so a [`Map`] that carries no atmosphere renders exactly as before
/// — fully back-compatible with the shipped levels.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Atmosphere {
    /// Fog tint colour, packed `0x00RRGGBB`. `None` ⇒ the renderer's built-in
    /// fog colour (a fade toward black).
    pub fog_color: Option<u32>,
    /// World-cell distance at which fog begins. `None` ⇒ built-in `FOG_NEAR`.
    pub fog_near: Option<f32>,
    /// World-cell distance at which fog is full. `None` ⇒ built-in `FOG_FAR`.
    pub fog_far: Option<f32>,
    /// `[effects] matrix` ambient density `[0,1]`. `None` ⇒ unset (the renderer
    /// keeps the matrix material's built-in look). A future bias knob.
    pub matrix: Option<f32>,
    /// Level-authored PUNCTUAL lights (`[light]` sections). Empty ⇒ no point
    /// lights, and the renderer's lighting is byte-identical to the pre-lighting
    /// build. Unshadowed in this revision (see the lighting spec §1.3).
    pub lights: Vec<PointLight>,
    /// Directional KEY light direction, pointing **toward** the light. Need not
    /// be unit — the renderer normalises. `None` ⇒ the built-in key direction.
    pub key_dir: Option<Vec3>,
    /// Directional KEY light colour, packed `0x00RRGGBB`. `None` ⇒ built-in.
    pub key_color: Option<u32>,
    /// Ambient floor `[0,1]` (grey level). `None` ⇒ the built-in ambient.
    pub ambient: Option<f32>,
    /// Exponential HEIGHT fog, multiplied onto the distance fog. `None` ⇒ off.
    pub fog_height: Option<HeightFog>,
    /// `re_postfx` preset name requested by the level (e.g. `"clean"`).
    /// `None` ⇒ the level expresses no preference and the Settings value stands.
    /// Validated against the known preset list by `re_level` at parse time.
    pub postfx: Option<String>,
}

/// One level-authored PUNCTUAL (point) light. Plain data; `re_render` turns a
/// slice of these into its runtime light set. Position is in WORLD CELL units
/// with the same axis convention as [`CameraPose::pos`] — `x`/`y` are the floor
/// plane and **`z` is height** (0 = floor, `storey_height` = ceiling).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointLight {
    /// World position (`x`, `y` on the floor plane, `z` = height).
    pub pos: Vec3,
    /// Light colour, packed `0x00RRGGBB`.
    pub color: u32,
    /// Falloff radius in world cells (> 0). Attenuation is
    /// `1 / (1 + d²/radius²)` — at `d == radius` the light is at half strength.
    pub radius: f32,
    /// Scalar brightness multiplier (≥ 0). `1.0` = the authored colour at full.
    pub intensity: f32,
}

impl Default for PointLight {
    fn default() -> Self {
        PointLight {
            pos: Vec3::ZERO,
            color: 0x00FF_FFFF,
            radius: 6.0,
            intensity: 1.0,
        }
    }
}

/// The canonical `re_postfx` preset names an `[atmosphere] postfx =` value may
/// name. Kept HERE (engine-agnostic data) rather than in `re_postfx` so the
/// parser (`re_level`), the linter (`re_lvlcheck`), the serializer
/// (`re_voximport`) and the editor can all validate against one list without any
/// of them depending on the post-effect implementation. Must stay in sync with
/// `re_postfx::PostfxConfig::preset` — a `re_game` test asserts that.
pub const POSTFX_PRESETS: [&str; 7] = [
    "off",
    "clean",
    "subtle",
    "cinematic",
    "toon",
    "crt",
    "dream",
];

/// True if `name` is a known [`POSTFX_PRESETS`] entry (exact, lower-case).
#[inline]
pub fn is_postfx_preset(name: &str) -> bool {
    POSTFX_PRESETS.contains(&name)
}

/// Exponential height fog: the distance-fog factor is multiplied by
/// `exp(-falloff · max(0, z − base))`, so surfaces ABOVE `base` clear out and the
/// murk pools at the floor. `falloff == 0` is a no-op.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HeightFog {
    /// World height (z) at which the fog starts thinning.
    pub base: f32,
    /// Extinction per world unit of height above `base` (≥ 0).
    pub falloff: f32,
}

impl Atmosphere {
    /// True if this atmosphere carries no authored override at all (the common,
    /// shipped-level case) — the renderer then uses its built-in constants.
    #[inline]
    pub fn is_none(&self) -> bool {
        self.fog_color.is_none()
            && self.fog_near.is_none()
            && self.fog_far.is_none()
            && self.matrix.is_none()
            && self.lights.is_empty()
            && self.key_dir.is_none()
            && self.key_color.is_none()
            && self.ambient.is_none()
            && self.fog_height.is_none()
            && self.postfx.is_none()
    }

    /// True if this atmosphere authors any LIGHTING override (key light, ambient,
    /// or point lights). Fog-only levels answer `false`, so the renderer can keep
    /// its built-in `LightEnv` (`re_render`) untouched on every shipped level.
    #[inline]
    pub fn has_lighting(&self) -> bool {
        !self.lights.is_empty()
            || self.key_dir.is_some()
            || self.key_color.is_some()
            || self.ambient.is_some()
    }
}

/// One authored material definition carried by a [`Map`] (and a parsed level).
///
/// This is the engine-agnostic "color = material KEY" payload: a name, a single
/// flat `fg`/`bg` colour (packed `0x00RRGGBB`), an optional fixed `glyph`, an
/// optional procedural `recipe`, an optional animation, and a LOD `rank` bias.
/// `re_render` turns a slice of these into its `MaterialRegistry` via
/// `from_specs`; if a level supplies none, the renderer falls back to built-ins.
///
/// `re_world` deliberately knows nothing about how this shades — it is just data
/// passed through from `re_level::build_map` to the renderer.
#[derive(Clone, Debug, PartialEq)]
pub struct MaterialDef {
    /// Stable name an author references (e.g. `"floor_neon"`).
    pub name: String,
    /// Foreground colour, packed `0x00RRGGBB`.
    pub fg: u32,
    /// Background colour, packed `0x00RRGGBB`.
    pub bg: u32,
    /// Optional fixed base glyph (a Unicode scalar); `None` ⇒ use the recipe ramp.
    pub glyph: Option<u32>,
    /// Optional procedural recipe; `None` ⇒ the renderer picks a sensible default
    /// recipe for the surface this material is bound to.
    pub recipe: Option<MaterialRecipe>,
    /// Animation block (default = static).
    pub anim: MaterialAnim,
    /// Detail-rank bias `[0,127]` style; `0` keeps the recipe's default order.
    pub rank: u8,
    /// When `true`, this surface is a **MIRROR**: the raycaster reflects the primary
    /// ray about the hit face's normal and shows the REFLECTED scene in the cell,
    /// tinted with a cool mirror sheen, instead of the material's own surface. Plain
    /// data — `re_world` knows nothing about how it reflects; `re_render` reads the
    /// flag off the resolved `Material` (`re_render`). `false` (the default) ⇒ an ordinary
    /// opaque surface, fully back-compatible.
    pub reflective: bool,
    /// Mirror CLARITY `[0,1]`: how crisply the reflection resolves. `1.0` = a clean,
    /// sharp reflection; `0.0` = a dark, broken, noisy surface (the reflection has
    /// "collapsed"). Values in between blend between the two. Authored as the
    /// material's STARTING clarity; the game can drive it live via the renderer's
    /// `MirrorClarity` resource to animate a recognition beat. Only meaningful when
    /// `reflective` is `true`. Defaults to [`MaterialDef::DEFAULT_CLARITY`].
    pub clarity: f32,
    /// When `true`, this slot is a **built-in fallback**: the level did not author
    /// a material for this surface, so the renderer should keep its own built-in
    /// for the slot. `fg`/`bg`/etc. are ignored. Set by `re_level::build_map` so
    /// the authored set can be a fixed-length, slot-ordered list (one entry per
    /// surface) while unbound surfaces still get the engine default.
    pub fallback: bool,
    /// Opt-in world-anchored DETAIL-TIER cross-fade (`[material] detail = on`):
    /// when `true`, `re_render`'s
    /// recipe sampling gates on `re_render::detail_lod` (tone/pattern/detail +
    /// underline micro-detail) instead of a single fixed-resolution sample.
    /// Defaults to `false` — back-compatible, byte-identical rendering for every
    /// material that never authors the key.
    pub detail: bool,
}

impl MaterialDef {
    /// The default mirror clarity when a level marks a surface `reflective` but
    /// does not author an explicit `clarity` — a mostly-crisp reflection with a
    /// faint imperfection, so a fresh mirror already "works" while leaving the game
    /// headroom to both sharpen (→1) and break (→0) it.
    pub const DEFAULT_CLARITY: f32 = 0.85;

    /// A minimal flat material: just a name + fg/bg, static, recipe-default.
    pub fn flat(name: impl Into<String>, fg: u32, bg: u32) -> Self {
        MaterialDef {
            name: name.into(),
            fg,
            bg,
            glyph: None,
            recipe: None,
            anim: MaterialAnim::default(),
            rank: 0,
            reflective: false,
            clarity: Self::DEFAULT_CLARITY,
            fallback: false,
            detail: false,
        }
    }

    /// A built-in-fallback slot: the renderer keeps its own material here.
    pub fn fallback() -> Self {
        MaterialDef {
            name: String::new(),
            fg: 0,
            bg: 0,
            glyph: None,
            recipe: None,
            anim: MaterialAnim::default(),
            rank: 0,
            reflective: false,
            clarity: Self::DEFAULT_CLARITY,
            fallback: true,
            detail: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Non-euclidean portal geometry.
// ---------------------------------------------------------------------------

/// Internal: the inner-face frame derived from one wall-face portal region.
///
/// Carries the room-side plane origin, its in-plane right/up axes, the inward
/// normal, and the in-plane half-extents. Built by `Map::face_frame` and
/// consumed by `Map::add_portal_pair`; not part of the public API.
#[derive(Clone, Copy, Debug)]
struct PortalFace {
    origin: Vec3,
    right: Vec3,
    up: Vec3,
    normal: Vec3,
    half_u: f32,
    half_v: f32,
}

/// One side of a non-euclidean warp-portal link (spec §3.1 / §4).
///
/// A portal is a rectangular opening in the voxel grid. Two `PortalLink`s form
/// a reciprocal pair: ray/player crossing side A is transformed by `delta` into
/// the coordinate frame of side B, and vice versa via `delta_inv`. The pair is
/// registered in [`Map::portals`]; individual portal cells index into that `Vec`
/// via [`Map::portal_for_cell`].
///
/// **Graph-ready shape:** each link already carries its own `delta`; a future
/// graph model is an extension (more links, per-node coord spaces), not a
/// rewrite. The `warp_budget` parameter threaded through the renderer bounds
/// recursion depth.
///
/// ## Coordinate conventions
///
/// * `plane_pos` — a world-space point on the portal plane (the opening centre).
/// * `plane_normal` — unit normal pointing toward the arriving side
///   (rays with a negative dot-product against this are on the wrong side).
/// * `half_u` — in-plane horizontal axis scaled by the opening half-width.
/// * `half_v` — in-plane vertical axis scaled by the opening half-height.
/// * `delta` — rigid transform (`Mat4`) mapping a world point/direction on
///   THIS side into the destination region. Apply to the hit point and
///   ray direction to re-base the secondary DDA ray.
/// * `delta_inv` — exact inverse of `delta` (the partner's delta). Precomputed
///   on construction so the renderer never inverts a matrix per ray; a wrong
///   inverse = a visible tear, so it is computed, not authored on both sides.
#[derive(Clone, Debug)]
pub struct PortalLink {
    /// A point on the portal plane (world space, opening centre).
    pub plane_pos: Vec3,
    /// Unit normal pointing toward the arriving side.
    pub plane_normal: Vec3,
    /// In-plane axis × half-width (the horizontal opening bound).
    pub half_u: Vec3,
    /// In-plane axis × half-height (the vertical opening bound).
    pub half_v: Vec3,
    /// Maps a world point/ray on THIS side into the destination region.
    pub delta: Mat4,
    /// Inverse of `delta` (the partner portal's delta). Pre-computed.
    pub delta_inv: Mat4,
}

/// The voxel map: a 3D grid of [`CellKind`], row-major
/// (`index = x + sx*(y + sy*z)`), exactly like the C `World`.
///
/// A parallel `side_data` grid carries a game-defined tag per cell (a `u16`
/// whose meaning is determined by the game, not the engine). The DDA only
/// needs [`CellKind`]; `side_data` is for the game layer to query.
///
/// `materials` is an OPTIONAL per-level authored material set (additive): empty
/// by default, so an unchanged map drives the renderer's built-in materials. A
/// level that authors a `[material …]` section populates it via
/// [`set_materials`](Self::set_materials), and the renderer then builds its
/// registry from these instead of the bare built-ins.
#[derive(Resource, Clone, Debug)]
pub struct Map {
    store: MapStore, // Dense (default, byte-identical) or Brick (W1 sparse)
    /// Authored per-level materials (empty ⇒ renderer uses its built-ins).
    materials: Vec<MaterialDef>,
    /// Authored per-level atmosphere (fog + matrix density). Default = none, so
    /// an unchanged map drives the renderer's built-in fog constants.
    atmosphere: Atmosphere,
    /// Cached: does the grid contain ANY [`CellKind::Hologram`] cell? Set when a
    /// hologram cell is written, so the renderer can skip its hologram pass
    /// entirely on the common (no-hologram) map — a per-frame O(1) gate, never an
    /// O(cells) scan.
    has_holo: bool,
    /// Registered portal pairs. Empty on all existing levels — they load and
    /// render exactly as before. Populated by [`Map::inject_demo_portal_pair`]
    /// (Stage 1 scaffold) or the future `re_level` `[portal]` parser (Stage 3).
    pub portals: Vec<PortalLink>,
    /// Sparse map from grid cell `(cx, cy, cz)` → index into [`Map::portals`].
    /// Only [`CellKind::Portal`] cells appear here; all other cells are absent
    /// (the common case). O(1) lookup by the raycaster on a portal hit.
    portal_cells: HashMap<(i32, i32, i32), usize>,
    /// Per-cell [`Shape`] (Full/WedgeXY/RampZ + orientation).
    ///
    /// Default = all-`Full` via [`ShapeStore`]'s sparse representation — no
    /// non-`Full` entry ⇒ every cell behaves exactly as today.  A level with no
    /// wedge/ramp geometry never writes here, keeping the heap overhead zero.
    shapes: ShapeStore,
    /// The optional **Scalable Material Overlay** (zones / lines / insets) that
    /// rides over this voxel grid. Default = [`Overlay::default`] (all-empty), so
    /// an unchanged map carries an inert overlay and renders byte-identically to
    /// today. Populated by the `re_level` `[zone]`/`[line]`/`[inset]` parser via
    /// [`set_overlay`](Self::set_overlay). P0 is the carrier seam only: NO resolve,
    /// NO render, NO collision change (collision stays octave-0, spec §7 Q6).
    overlay: Overlay,
    /// Per-`[line]` resolved **colour-palette id** (set-dressing E0, F5), parallel to
    /// `overlay.lines`: entry `i` is the 1-based palette id of the authored
    /// `[material NAME]` that line `i` binds (the SAME id space as `[zone]` and
    /// `[color_cell_ids]`), or `0` when the name is empty / not a level-authored
    /// material. `re_level::build_map` resolves it by the AUTHORED name — which the
    /// renderer cannot do itself, because a `MaterialDef` carries only the
    /// preset-encoded name. `re_render::lines` reads it to tint the stroke with the
    /// authored material; `0` falls back to a catalog name lookup, then the contour
    /// ink. Default = empty ⇒ no resolved ids (every stroke takes the fallback path).
    line_palette_ids: Vec<u16>,
    /// The **resolved insets** (Scalable Material Overlay P4) — `re_level::build_map`
    /// distils each authored `[inset]` (which references a child `.lvl`) into a
    /// [`re_core::ResolvedInset`] (region + phase-band + representative colour) so
    /// `re_render` can apply the bounded phase-band reveal over a SelfSimilar zone
    /// WITHOUT ever loading a `.lvl`. Default = empty ⇒ the renderer's inset-override
    /// loop never runs ⇒ frames are byte-identical to today. A level with no `[inset]`
    /// resolves to an empty vec, so every existing level is unaffected.
    resolved_insets: Vec<re_core::ResolvedInset>,
    /// The **resolved triggers** — `re_level::build_map` distils each authored
    /// `[trigger]` (region + action + once) into a [`re_core::Trigger`] (with any
    /// `ActivateNode` node-name already resolved to an index). The game layer
    /// (`re_game`) queries these with [`triggers_containing`](Self::triggers_containing)
    /// each frame to detect OUTSIDE→INSIDE entry and fire the action. Default =
    /// empty ⇒ a level with no `[trigger]` section never fires anything (additive,
    /// byte-identical to today).
    resolved_triggers: Vec<re_core::Trigger>,
    /// Per-level image textures (decoded RGBA8) referenced by an `Image` material
    /// recipe via its index. Each entry is a [`re_core::TextureSet`]: a STILL image is
    /// a one-frame set (what `push_texture` builds, byte-identical to when this vector
    /// held bare `Arc<TextureImage>`), an `[texture NAME] frames = clip.gif` is an
    /// N-frame set with an `fps`. Frames are `Arc`-wrapped so the renderer's material
    /// registry can share the (potentially large) pixel buffers without cloning them.
    /// Default = empty ⇒ a textureless map renders byte-identically to today.
    /// Populated by `re_level` via [`push_texture`](Self::push_texture) /
    /// [`push_texture_set`](Self::push_texture_set).
    pub textures: Vec<re_core::TextureSet>,
    /// Optional image-backed sky set (cylinder strip + top cap + optional bottom cap).
    /// Default = all-`None` ⇒ no image sky, the renderer uses [`ProceduralSky`] for
    /// every missed / above-horizon ray — byte-identical to today. Populated by
    /// `re_level` via the `[sky]` section setters. Read by `re_render::sky` to
    /// choose a [`TextureImage`]-backed [`SkySource`] for the authored regions.
    ///
    /// Storage only; resize, UV mapping, and blending live in `re_render`.
    ///
    /// [`ProceduralSky`]: re_render::sky::ProceduralSky
    /// [`SkySource`]: re_render::sky::SkySource
    /// [`TextureImage`]: re_core::TextureImage
    sky_images: SkyImages,
    /// Optional per-level PROCEDURAL sky PROGRAM — a gradient base plus stars/aurora/clouds/storm/etc. layers,
    /// parsed by `re_level` from a `[sky] base =` / `layer =` / `kind =` section.
    /// Default = `None` ⇒ the renderer keeps sampling the legacy
    /// [`ProceduralSky`]/[`ImageSky`] path — byte-identical to today for every
    /// level that authors no such key. Populated via [`set_sky_spec`](Self::set_sky_spec).
    ///
    /// [`ProceduralSky`]: re_render::sky::ProceduralSky
    /// [`ImageSky`]: re_render::sky::ImageSky
    sky_spec: Option<sky_spec::SkySpec>,
    /// Runtime-only set of cells currently marked walk-through (see
    /// [`mark_walkthrough`](Self::mark_walkthrough)). Never serialized; a level
    /// loads with this empty and every prop-power/emit system re-marks its own
    /// cells as it stamps them.
    walkthrough: HashSet<(i32, i32, i32)>,
}

impl Map {
    /// Private: construct a `Map` around any `MapStore`, setting all non-store
    /// fields to their defaults. Both `new` (Dense) and `new_brickmap` (Brick)
    /// call this so the non-store initialization can never diverge.
    fn with_store(store: MapStore) -> Self {
        Map {
            store,
            materials: Vec::new(),
            atmosphere: Atmosphere::default(),
            has_holo: false,
            portals: Vec::new(),
            portal_cells: HashMap::new(),
            shapes: ShapeStore::default(),
            overlay: Overlay::default(),
            line_palette_ids: Vec::new(),
            resolved_insets: Vec::new(),
            resolved_triggers: Vec::new(),
            textures: Vec::new(),
            sky_images: SkyImages::default(),
            sky_spec: None,
            walkthrough: HashSet::new(),
        }
    }

    /// Append a STILL image texture and return its index (the `tex_id` an `Image`
    /// material recipe references). The buffer is `Arc`-wrapped so the renderer's
    /// registry can share it without a deep copy.
    ///
    /// The entry is a ONE-frame [`re_core::TextureSet`], so it reports
    /// `is_animated() == false` and every consumer sees exactly what it saw when this
    /// vector held bare images.
    pub fn push_texture(&mut self, texture: re_core::TextureImage) -> u32 {
        self.push_texture_set(re_core::TextureSet::still(texture))
    }

    /// Append a FRAME SET (a still is a one-frame set) and return its index.
    /// The animated form of [`push_texture`](Self::push_texture): `re_level` builds
    /// the set once at load from `[texture NAME] frames = clip.gif`, so a 68-frame
    /// GIF is decoded ONCE for the whole session, never per cast or per cell.
    pub fn push_texture_set(&mut self, set: re_core::TextureSet) -> u32 {
        let id = self.textures.len() as u32;
        self.textures.push(set);
        id
    }

    /// The image texture at `id` — **frame 0** — or `None` if out of range.
    ///
    /// Deliberately unchanged in shape and meaning: for a still texture this IS the
    /// texture, and every caller that only wants the pixels' dimensions/aspect (the
    /// `fit` span derivation, the asset checks) wants frame 0 whatever the set does
    /// over time. Callers that must follow the animation ask for the
    /// [`texture_set`](Self::texture_set) and index it by time.
    #[inline]
    pub fn texture(&self, id: u32) -> Option<&std::sync::Arc<re_core::TextureImage>> {
        self.textures
            .get(id as usize)
            .map(re_core::TextureSet::first)
    }

    /// The full frame set at `id`, or `None` if out of range. A still texture yields
    /// a one-frame set.
    #[inline]
    pub fn texture_set(&self, id: u32) -> Option<&re_core::TextureSet> {
        self.textures.get(id as usize)
    }

    // -----------------------------------------------------------------------
    // Sky-image accessors (image-backed skybox foundation).
    // -----------------------------------------------------------------------

    /// Attach the horizon cylinder image for the sky (360° wraparound band,
    /// sampled for `pitch ≤ θ_cap`).  Recommended dimensions ≈ 720 × 96 (matching
    /// `re_render::sky::CYL_W` × `CYL_H`); any size is accepted — the renderer
    /// normalises UVs.
    ///
    /// Replaces any previously set cylinder image. Pass through an `Arc` so the
    /// renderer can share the buffer without cloning it.
    pub fn set_sky_cylinder(&mut self, image: std::sync::Arc<re_core::TextureImage>) {
        self.sky_images.cylinder = Some(image);
    }

    /// Attach the zenith cap image for the sky (top disc, sampled for
    /// `pitch > θ_cap`).  Recommended dimensions ≈ 256 × 256 (matching
    /// `re_render::sky::CAP_DIM`); any size is accepted.
    ///
    /// Replaces any previously set top-cap image.
    pub fn set_sky_cap_top(&mut self, image: std::sync::Arc<re_core::TextureImage>) {
        self.sky_images.cap_top = Some(image);
    }

    /// Attach the nadir cap image for the sky (bottom disc, for below-horizon
    /// rays).  Optional: most levels leave this `None` (the renderer's procedural
    /// or floor-sky path handles below-horizon rays).
    ///
    /// Replaces any previously set bottom-cap image.
    pub fn set_sky_cap_bottom(&mut self, image: std::sync::Arc<re_core::TextureImage>) {
        self.sky_images.cap_bottom = Some(image);
    }

    /// The full sky-image set for this map.  All three slots default to `None`
    /// (procedural sky for every region — byte-identical to today).  `re_render`
    /// reads this once at registry-build time to decide whether to use an
    /// image-backed `re_render::sky::SkySource` for each region.
    #[inline]
    pub fn sky_images(&self) -> &SkyImages {
        &self.sky_images
    }

    /// Attach the level's authored procedural sky PROGRAM. Replaces any
    /// previously set program. `re_level` calls this once at `build_map` time from a `[sky]`
    /// section's `base =` / `layer =` / `kind =` keys.
    pub fn set_sky_spec(&mut self, spec: sky_spec::SkySpec) {
        self.sky_spec = Some(spec);
    }

    /// The level's authored procedural sky program, or `None` when the level's
    /// `[sky]` section (if any) named only image paths — the renderer then
    /// falls back to `re_render::sky::ProceduralSky`/`re_render::sky::ImageSky`
    /// exactly as before.
    #[inline]
    pub fn sky_spec(&self) -> Option<&sky_spec::SkySpec> {
        self.sky_spec.as_ref()
    }

    /// Allocate an all-[`CellKind::Empty`] map of `sx`×`sy`×`sz` backed by
    /// the dense flat store (byte-identical to pre-W0 — existing tests
    /// unchanged).
    pub fn new(sx: usize, sy: usize, sz: usize) -> Self {
        Map::with_store(MapStore::Dense(DenseStore::new(sx, sy, sz)))
    }

    /// A `Map` backed by the sparse brickmap store — for big worlds (W1).
    /// Memory is O(written bricks); reads of unwritten cells return
    /// [`CellKind::Empty`]. All public `Map` methods work identically.
    pub fn new_brickmap(sx: usize, sy: usize, sz: usize) -> Self {
        Map::with_store(MapStore::Brick(BrickmapStore::new(sx, sy, sz)))
    }

    /// A procedural, runtime-editable, persistent big world (W5). Unwritten
    /// cells come from `gen::procedural_base(seed,…)`; edits persist as override
    /// bricks. Reuses the private `with_store` helper.
    pub fn procedural(sx: usize, sy: usize, sz: usize, seed: u64) -> Self {
        Self::with_store(MapStore::Proc(ProcGenStore::new(sx, sy, sz, seed)))
    }

    /// A brickmap-backed Map of the given size, filled from `procedural_base`.
    /// Only solid cells materialize bricks (empty regions stay sparse).
    pub fn generate_brickmap(sx: usize, sy: usize, sz: usize, seed: u64) -> Self {
        let mut m = Map::new_brickmap(sx, sy, sz);
        for z in 0..sz {
            for y in 0..sy {
                for x in 0..sx {
                    let k = crate::gen::procedural_base(seed, x as i32, y as i32, z as i32);
                    if k != CellKind::Empty {
                        m.set_cell(x, y, z, k, 0);
                    }
                }
            }
        }
        m
    }

    /// True if the map contains any [`CellKind::Hologram`] cell. O(1) — the
    /// renderer gates its hologram pass on this so a no-hologram map is unchanged.
    #[inline]
    pub fn has_holograms(&self) -> bool {
        self.has_holo
    }

    /// Replace the authored atmosphere (per-level fog + matrix density). Default
    /// ⇒ the renderer uses its built-in fog constants.
    pub fn set_atmosphere(&mut self, atmosphere: Atmosphere) {
        self.atmosphere = atmosphere;
    }

    /// The authored atmosphere (default ⇒ use the renderer's built-in fog).
    /// Borrowed: [`Atmosphere`] owns a `Vec<PointLight>` and is no longer `Copy`.
    #[inline]
    pub fn atmosphere(&self) -> &Atmosphere {
        &self.atmosphere
    }

    /// Replace the authored material set (the per-level "color = material KEY"
    /// payload). Empty ⇒ the renderer falls back to its built-ins.
    pub fn set_materials(&mut self, materials: Vec<MaterialDef>) {
        self.materials = materials;
    }

    /// The authored material set (empty ⇒ use renderer built-ins).
    #[inline]
    pub fn materials(&self) -> &[MaterialDef] {
        &self.materials
    }

    /// Replace the authored [`Overlay`] (zones / lines / insets). Default ⇒ an
    /// empty overlay, so an unchanged map renders byte-identically to today. P0
    /// carrier only: storing an overlay changes no geometry, no collision, no
    /// render — the resolve/render is a later phase (spec §6 P1+).
    pub fn set_overlay(&mut self, overlay: Overlay) {
        self.overlay = overlay;
    }

    /// The authored overlay (default ⇒ [`Overlay::default`], all-empty). Read-only
    /// accessor; the raycaster/resolve of a later phase reads through this rather
    /// than reaching into `Map` internals.
    #[inline]
    pub fn overlay(&self) -> &Overlay {
        &self.overlay
    }

    /// Replace the per-`[line]` resolved colour-palette ids (parallel to
    /// `overlay().lines`; `0` = unresolved). See the field docs.
    pub fn set_line_palette_ids(&mut self, ids: Vec<u16>) {
        self.line_palette_ids = ids;
    }

    /// The resolved 1-based colour-palette id of overlay line `index`, or `0` when
    /// the line binds no level-authored material (or `index` is out of range).
    #[inline]
    #[must_use]
    pub fn line_palette_id(&self, index: usize) -> u16 {
        self.line_palette_ids.get(index).copied().unwrap_or(0)
    }

    /// Replace the **resolved insets** (Scalable Material Overlay P4). Called by
    /// `re_level::build_map` after it has (cycle-guarded, one-level-deep) loaded each
    /// authored `[inset]`'s child scene and distilled it to a [`re_core::ResolvedInset`]
    /// (region + phase-band + representative colour). Default ⇒ empty, so a level with
    /// no `[inset]` leaves the renderer's override loop a no-op (byte-identical).
    pub fn set_resolved_insets(&mut self, insets: Vec<re_core::ResolvedInset>) {
        self.resolved_insets = insets;
    }

    /// The resolved insets (default ⇒ empty). `re_render` reads this slice to apply
    /// the bounded phase-band reveal over a SelfSimilar zone — it never loads a `.lvl`.
    /// Empty ⇒ the override loop never runs ⇒ byte-identical to today.
    #[inline]
    pub fn resolved_insets(&self) -> &[re_core::ResolvedInset] {
        &self.resolved_insets
    }

    /// Replace the **resolved triggers** (trigger-fields feature). Called by
    /// `re_level::build_map` after it has resolved each authored `[trigger]`'s
    /// `ActivateNode` node name to an index. Default ⇒ empty, so a level with no
    /// `[trigger]` section fires nothing.
    pub fn set_resolved_triggers(&mut self, triggers: Vec<re_core::Trigger>) {
        self.resolved_triggers = triggers;
    }

    /// All resolved triggers on this map (default ⇒ empty). The game layer reads
    /// this slice to drive the firing system.
    #[inline]
    pub fn resolved_triggers(&self) -> &[re_core::Trigger] {
        &self.resolved_triggers
    }

    /// Every resolved trigger whose region CONTAINS the cell at world position
    /// `pos` (the camera position; floored to its containing cell, matching the
    /// floor-based cell sampling the rest of `re_world` uses — see [`at_door`]).
    ///
    /// Pure, allocation-free (returns an iterator borrowing `self`). A non-finite
    /// `pos` (NaN/∞ from a corrupt pose) yields an EMPTY iterator — never fire on
    /// garbage input. The game layer compares this frame's hits against last
    /// frame's to detect the OUTSIDE→INSIDE entry edge.
    pub fn triggers_containing(&self, pos: Vec3) -> impl Iterator<Item = &re_core::Trigger> {
        let cell = if pos.is_finite() {
            Some((
                pos.x.floor() as i32,
                pos.y.floor() as i32,
                pos.z.floor() as i32,
            ))
        } else {
            None
        };
        self.resolved_triggers
            .iter()
            .filter(move |t| matches!(cell, Some((x, y, z)) if t.contains(x, y, z)))
    }

    /// The map's `(sx, sy, sz)` voxel-grid dimensions.
    #[inline]
    pub fn size(&self) -> (usize, usize, usize) {
        self.store.size()
    }

    /// True if `(x, y, z)` is inside the map's voxel grid.
    #[inline]
    pub fn in_bounds(&self, x: i32, y: i32, z: i32) -> bool {
        self.store.in_bounds(x, y, z)
    }

    /// Cell at `(x,y,z)`; out-of-bounds reads return [`CellKind::Empty`] so the
    /// DDA inner loop never has to branch on bounds (mirrors `world_cell_at`).
    #[inline]
    pub fn cell_at(&self, x: i32, y: i32, z: i32) -> CellKind {
        self.store.cell_at(x, y, z)
    }

    /// Write a cell's kind (clearing its `side_data`). Out-of-bounds ignored.
    #[inline]
    pub fn set(&mut self, x: usize, y: usize, z: usize, kind: CellKind) {
        self.set_cell(x, y, z, kind, 0);
    }

    /// Write a cell's kind and `side_data` tag. Out-of-bounds writes are
    /// ignored. Mirrors legacy `world_set_cell` (minus the `rev` counter,
    /// which the Bevy change-detection layer subsumes).
    #[inline]
    pub fn set_cell(&mut self, x: usize, y: usize, z: usize, kind: CellKind, side_data: u16) {
        if self.store.set_cell(x, y, z, kind, side_data) && kind == CellKind::Hologram {
            self.has_holo = true; // Map keeps the O(1) holo cache (store stays game-agnostic)
        }
    }

    /// Game-defined `side_data` tag at `(x,y,z)`; `0` for out-of-bounds or
    /// untagged cells. Meaning is determined by the game layer, not the engine.
    #[inline]
    pub fn side_data_at(&self, x: i32, y: i32, z: i32) -> u16 {
        self.store.side_data_at(x, y, z)
    }

    // -----------------------------------------------------------------------
    // Walk-through override (visual-only prop content — set-dressing follow-up).
    //
    // Some cells are stamped SOLID (`is_solid`/`is_collidable` say yes) purely so
    // the renderer paints them — a power-reactive `[emit]`/`[prop] power=` figure
    // is exactly this: "a rack blade, a reward figure MATERIALISES" is meant to
    // read as light/projection, not as furniture the player bumps into. Rather
    // than add a new `CellKind` (which `re_render::offslice::role_of` matches
    // EXHAUSTIVELY, by design, so a new variant is a compile error there until
    // its vertical role is decided — not a change to make from this seam), a cell
    // can be marked WALK-THROUGH: rendering (`cell_at`, `is_solid`, the DDA) is
    // completely unaffected, and only the PHYSICS read (`physics_cell_at`, used
    // exclusively by `CameraPose`'s collision/gravity code below) sees it as
    // `Empty`. Runtime-only: never serialized to `.lvl`/brickmap bytes, rebuilt by
    // whatever system stamped the cell in the first place (`re_level::props`,
    // `re_game::emit_parts`, `re_game::prop_power`).
    // -----------------------------------------------------------------------

    /// Mark (or unmark) `(x, y, z)` as walk-through: [`physics_cell_at`](Self::physics_cell_at)
    /// will read it as [`CellKind::Empty`] regardless of its real [`CellKind`],
    /// while [`cell_at`](Self::cell_at) (rendering, keep-out, everything else)
    /// is unaffected. Idempotent; out-of-bounds coordinates are accepted and
    /// simply never consulted (`in_bounds` still gates every reader).
    pub fn mark_walkthrough(&mut self, x: usize, y: usize, z: usize, on: bool) {
        let key = (x as i32, y as i32, z as i32);
        if on {
            self.walkthrough.insert(key);
        } else {
            self.walkthrough.remove(&key);
        }
    }

    /// Is `(x, y, z)` currently marked walk-through?
    #[inline]
    pub fn is_walkthrough(&self, x: i32, y: i32, z: i32) -> bool {
        self.walkthrough.contains(&(x, y, z))
    }

    /// The [`CellKind`] to use for GAMEPLAY PHYSICS (collision, standing,
    /// ceiling headroom) at `(x, y, z)`: identical to
    /// [`cell_at`](Self::cell_at) except a cell marked
    /// [`walk-through`](Self::mark_walkthrough) reads as [`CellKind::Empty`] —
    /// exactly as if it were air. This is the ONE seam every physics call site
    /// in [`CameraPose`] reads through instead of `cell_at`; nothing else
    /// (rendering, keep-out, `lvlcheck`, the editor) should ever call it, because
    /// everything else legitimately wants the real, rendered kind.
    #[inline]
    pub fn physics_cell_at(&self, x: i32, y: i32, z: i32) -> CellKind {
        let k = self.cell_at(x, y, z);
        if k == CellKind::Empty {
            return k;
        }
        if self.is_walkthrough(x, y, z) {
            CellKind::Empty
        } else {
            k
        }
    }

    // -----------------------------------------------------------------------
    // Per-cell shape accessors (M3a 45° geometry).
    // -----------------------------------------------------------------------

    /// The [`Shape`] of the cell at `(x, y, z)`.  Out-of-bounds ⇒
    /// [`Shape::Full`] (solid cube, same as every un-shaped cell).
    ///
    /// O(1) amortized.  A map where no shape has been set returns
    /// [`Shape::Full`] for every cell — byte-identical to before M3a.
    #[inline]
    pub fn shape_at(&self, x: i32, y: i32, z: i32) -> Shape {
        if !self.store.in_bounds(x, y, z) {
            return Shape::Full; // out-of-bounds is treated as a full solid wall
        }
        let (sx, sy, _) = self.store.size();
        let idx = x as usize + sx * (y as usize + sy * z as usize);
        self.shapes.get(idx)
    }

    /// Set the [`Shape`] for cell `(x, y, z)`.  Out-of-bounds writes are
    /// silently ignored (same policy as [`set_cell`](Self::set_cell)).
    ///
    /// Writing [`Shape::Full`] restores the default (removes the entry) so
    /// the [`ShapeStore`] never accumulates redundant entries.
    pub fn set_shape(&mut self, x: usize, y: usize, z: usize, shape: Shape) {
        let (sx, sy, sz) = self.store.size();
        if x < sx && y < sy && z < sz {
            let idx = x + sx * (y + sy * z);
            self.shapes.set(idx, shape);
        }
    }

    /// Count of cells matching `kind` (test/diagnostic helper).
    pub fn count(&self, kind: CellKind) -> usize {
        self.store.count(kind)
    }

    // -----------------------------------------------------------------------
    // Portal accessors.
    // -----------------------------------------------------------------------

    /// Returns the index into [`Map::portals`] for the [`CellKind::Portal`]
    /// cell at `(cx, cy, cz)`, or `None` if that cell is not a portal (or is
    /// out of bounds).
    ///
    /// O(1) — backed by a sparse `HashMap` populated at portal construction
    /// time. The common case (no portals / non-portal cell) returns `None`
    /// instantly with zero heap allocation on the hot DDA path.
    #[inline]
    pub fn portal_for_cell(&self, cx: i32, cy: i32, cz: i32) -> Option<usize> {
        self.portal_cells.get(&(cx, cy, cz)).copied()
    }

    /// Stamp a rectangular run of [`CellKind::Portal`] cells and register them
    /// in `portal_cells` pointing at `portal_idx`. Internal helper shared
    /// between `inject_demo_portal_pair` and any future `[portal]` parser path.
    ///
    /// `bounds` = `((x0,y0,z0), (x1,y1,z1))` inclusive on both ends.
    fn stamp_portal_face(&mut self, bounds: ((i32, i32, i32), (i32, i32, i32)), portal_idx: usize) {
        let ((x0, y0, z0), (x1, y1, z1)) = bounds;
        let (xa, xb) = (x0.min(x1), x0.max(x1));
        let (ya, yb) = (y0.min(y1), y0.max(y1));
        let (za, zb) = (z0.min(z1), z0.max(z1));
        for z in za..=zb {
            for y in ya..=yb {
                for x in xa..=xb {
                    if x >= 0 && y >= 0 && z >= 0 {
                        self.set(x as usize, y as usize, z as usize, CellKind::Portal);
                        self.portal_cells.insert((x, y, z), portal_idx);
                    }
                }
            }
        }
    }

    /// Add a reciprocal warp-portal PAIR from two wall-face regions, stamping
    /// both faces as [`CellKind::Portal`] and registering the linked
    /// [`PortalLink`]s with reciprocal `delta`/`delta_inv` transforms.
    ///
    /// This is the **single shared implementation** of the portal geometry —
    /// both [`inject_demo_portal_pair`](Self::inject_demo_portal_pair) (the
    /// Stage-1 scaffold) and the `re_level` `[portal]` parser (Stage 3) call it,
    /// so the load-bearing math (inner-face plane placement, the 180°
    /// front-to-front `delta` flip, the reciprocal `delta_inv`) lives in exactly
    /// one place and cannot diverge.
    ///
    /// ## Arguments
    ///
    /// * `region_a` / `region_b` — inclusive grid-cell boxes
    ///   `((x0,y0,z0),(x1,y1,z1))` describing each opening. Each MUST be a flat
    ///   **wall face**: exactly one axis has `x0==x1` (the normal axis), and that
    ///   axis sits on a grid boundary (`0` or `size-1`). The other two axes give
    ///   the opening's in-plane extent. Both regions must describe the same
    ///   opening area (same in-plane cell counts) — a mismatch yields a visible
    ///   tear, so callers (the `[portal]` parser) validate it first.
    /// * `rotate_deg` — an EXTRA yaw (degrees) applied to the A→B transform
    ///   beyond the natural face-to-face frame change. `0.0` reproduces the
    ///   plain "walk straight, the world turned" warp. B→A uses the inverse.
    ///
    /// ## Geometry (matches the original inline demo exactly)
    ///
    /// For a face region, the inner-face plane is the room-side boundary of the
    /// portal cell row (e.g. `y=1.0` for a `y=0` north wall, `x=sx-1` for an
    /// east wall) so the player crossing the plane is still in bounds and the
    /// teleport fires. The in-plane centre/half are taken from the cell-index
    /// span endpoints (`centre=(lo+hi)/2`, `half=(hi-lo)/2`), identical to the
    /// scaffold's hand-derived values. `delta = ltw_b_flipped × wtl_a`, with the
    /// front-to-front flip negating the right+normal columns so the exiting ray
    /// travels INTO the destination room.
    pub fn add_portal_pair(
        &mut self,
        region_a: ((i32, i32, i32), (i32, i32, i32)),
        region_b: ((i32, i32, i32), (i32, i32, i32)),
        rotate_deg: f32,
    ) {
        let face_a = self.face_frame(region_a);
        let face_b = self.face_frame(region_b);

        // local_to_world: columns = [right, up, normal, origin] (rigid frame).
        let ltw_a = Mat4::from_cols(
            face_a.right.extend(0.0),
            face_a.up.extend(0.0),
            face_a.normal.extend(0.0),
            face_a.origin.extend(1.0),
        );
        let ltw_b = Mat4::from_cols(
            face_b.right.extend(0.0),
            face_b.up.extend(0.0),
            face_b.normal.extend(0.0),
            face_b.origin.extend(1.0),
        );
        let wtl_a = ltw_a.inverse();
        let wtl_b = ltw_b.inverse();

        // Front-to-front 180° flip about local UP: negate the right + normal
        // columns so the arriving direction (−normal) exits as +normal on the
        // far side (into the destination room). See the demo doc-comment for the
        // full derivation — this is byte-equivalent to the original inline math.
        let ltw_b_flipped = Mat4::from_cols(
            (-face_b.right).extend(0.0),
            face_b.up.extend(0.0),
            (-face_b.normal).extend(0.0),
            face_b.origin.extend(1.0),
        );
        let ltw_a_flipped = Mat4::from_cols(
            (-face_a.right).extend(0.0),
            face_a.up.extend(0.0),
            (-face_a.normal).extend(0.0),
            face_a.origin.extend(1.0),
        );

        // Optional EXTRA yaw A→B, applied about the destination's UP axis in
        // local space (post-multiply the source-side `wtl`). `0` ⇒ identity, so
        // the scaffold (which passes 0.0) is unchanged bit-for-bit.
        let (rot_ab, rot_ba) = if rotate_deg.abs() < f32::EPSILON {
            (Mat4::IDENTITY, Mat4::IDENTITY)
        } else {
            let r = rotate_deg.to_radians();
            (Mat4::from_rotation_z(r), Mat4::from_rotation_z(-r))
        };

        let delta_ab = ltw_b_flipped * rot_ab * wtl_a;
        let delta_ba = ltw_a_flipped * rot_ba * wtl_b;

        let idx_a = self.portals.len();
        self.portals.push(PortalLink {
            plane_pos: face_a.origin,
            plane_normal: face_a.normal,
            half_u: face_a.right * face_a.half_u,
            half_v: face_a.up * face_a.half_v,
            delta: delta_ab,
            delta_inv: delta_ba,
        });
        let idx_b = idx_a + 1;
        self.portals.push(PortalLink {
            plane_pos: face_b.origin,
            plane_normal: face_b.normal,
            half_u: face_b.right * face_b.half_u,
            half_v: face_b.up * face_b.half_v,
            delta: delta_ba,
            delta_inv: delta_ab,
        });

        self.stamp_portal_face(region_a, idx_a);
        self.stamp_portal_face(region_b, idx_b);
    }

    /// Derive the inner-face frame (origin, right/up/normal axes, in-plane
    /// half-extents) for one wall-face portal region. Pure geometry shared by
    /// [`add_portal_pair`](Self::add_portal_pair). Panics in debug if the region
    /// is not a single flat wall face — callers validate first, but this keeps
    /// the math honest.
    fn face_frame(&self, region: ((i32, i32, i32), (i32, i32, i32))) -> PortalFace {
        let ((x0, y0, z0), (x1, y1, z1)) = region;
        let (xa, xb) = (x0.min(x1), x0.max(x1));
        let (ya, yb) = (y0.min(y1), y0.max(y1));
        let (za, zb) = (z0.min(z1), z0.max(z1));

        // Which axis is flat (the normal axis)? Exactly one span must be a
        // single cell. Priority x → y → z (a region flat in more than one axis
        // is degenerate; callers reject it).
        let flat_x = xa == xb;
        let flat_y = ya == yb;

        // In-plane centre/half from the cell-index span endpoints (matches the
        // scaffold's `(lo+hi)/2` / `(hi-lo)/2` derivation exactly).
        let span_centre = |lo: i32, hi: i32| (lo + hi) as f32 * 0.5;
        let span_half = |lo: i32, hi: i32| (hi - lo) as f32 * 0.5;

        if flat_x {
            // Wall normal along x. Min boundary (x=0) ⇒ inward +X, inner face at
            // x+1; max boundary (x=sx-1) ⇒ inward −X, inner face at x.
            let at_min = xa == 0;
            let normal = if at_min { Vec3::X } else { -Vec3::X };
            let plane_x = if at_min { xa as f32 + 1.0 } else { xa as f32 };
            PortalFace {
                origin: Vec3::new(plane_x, span_centre(ya, yb), span_centre(za, zb)),
                right: Vec3::Y,
                up: Vec3::Z,
                normal,
                half_u: span_half(ya, yb),
                half_v: span_half(za, zb),
            }
        } else if flat_y {
            let at_min = ya == 0;
            let normal = if at_min { Vec3::Y } else { -Vec3::Y };
            let plane_y = if at_min { ya as f32 + 1.0 } else { ya as f32 };
            PortalFace {
                origin: Vec3::new(span_centre(xa, xb), plane_y, span_centre(za, zb)),
                right: Vec3::X,
                up: Vec3::Z,
                normal,
                half_u: span_half(xa, xb),
                half_v: span_half(za, zb),
            }
        } else {
            // Flat in z (a floor/ceiling portal). Min boundary (z=0) ⇒ inward
            // +Z, inner face at z+1; max ⇒ inward −Z, inner face at z.
            let at_min = za == 0;
            let normal = if at_min { Vec3::Z } else { -Vec3::Z };
            let plane_z = if at_min { za as f32 + 1.0 } else { za as f32 };
            PortalFace {
                origin: Vec3::new(span_centre(xa, xb), span_centre(ya, yb), plane_z),
                right: Vec3::X,
                up: Vec3::Y,
                normal,
                half_u: span_half(xa, xb),
                half_v: span_half(ya, yb),
            }
        }
    }

    /// **Stage 1 scaffold** — inject a hardcoded demo portal pair into this map
    /// for the non-euclidean warp-portal prototype.
    ///
    /// This is a TEMPORARY fixture until the `re_level` `[portal]` section
    /// parser (Stage 3) replaces it. Call it on a map loaded from
    /// `levels/92-fractal-portal.lvl` to get a seam-free non-euclidean corner.
    ///
    /// ## Layout (matches `levels/92-fractal-portal.lvl`)
    ///
    /// The level is a 12×16×10 room. The two portal openings are:
    ///
    /// * **Portal A (north wall):** portal cells at `y=0`, x-span `1..=10`,
    ///   z-span `1..=8`. Normal = `+y` (rays arrive from the south).
    ///   **Portal plane** = INNER FACE at `y=1.0` so the player crossing
    ///   northward triggers the teleport while still in bounds.
    /// * **Portal B (west wall):** portal cells at `x=0`, y-span `1..=14`,
    ///   z-span `1..=8`. Normal = `+x` (rays arrive from the east).
    ///   **Portal plane** = INNER FACE at `x=1.0` for the same reason.
    ///
    /// A ray entering the NORTH portal emerges from the WEST portal
    /// (non-euclidean 90° turn): the destination appears as if you continued
    /// straight ahead but the world has turned.
    ///
    /// ## Transform derivation
    ///
    /// Each side has a local frame `(origin, right=u, up=v, in=n)`:
    /// * `local_to_world` = a 4×4 that maps the local `(u,v,n)` frame into
    ///   world space.
    /// * `world_to_local` = its inverse.
    /// * `delta_A→B = local_to_world_B × world_to_local_A` maps a world point
    ///   on side A into the equivalent point in side B's world.
    /// * `delta_B→A = inverse(delta_A→B)`.
    pub fn inject_demo_portal_pair(&mut self) {
        // The load-bearing portal math now lives in `add_portal_pair` (the
        // single shared implementation used by the `re_level` `[portal]` parser
        // too), so the scaffold and the parser can never diverge. This call is
        // byte-equivalent to the old hand-derived inline math:
        //
        // * Side A — NORTH wall, cells y=0, x∈[1,10], z∈[1,8]. Inner-face plane
        //   at y=1.0, inward normal +y, right=+x, up=+z, centre (5.5,1.0,4.5),
        //   half (4.5, 3.5).
        // * Side B — WEST wall, cells x=0, y∈[1,14], z∈[1,8]. Inner-face plane
        //   at x=1.0, inward normal +x, right=+y, up=+z, centre (1.0,7.5,4.5),
        //   half (6.5, 3.5).
        //
        // `rotate_deg = 0.0` ⇒ no extra yaw, so `delta_A→B = ltw_b_flipped ×
        // wtl_a` exactly as before (the 90° corner turn comes from the differing
        // face frames, not from an explicit rotate).
        self.add_portal_pair(
            ((1, 0, 1), (10, 0, 8)), // NORTH wall opening (portal A)
            ((0, 1, 1), (0, 14, 8)), // WEST wall opening (portal B)
            0.0,
        );
    }
}

/// The hardcoded M1 test room — a Rust port of `world_load_test_map`.
///
/// A 6×6×4 room: perimeter walls at every z, interior floor at z=0, interior
/// ceiling at z=3, and one interior pillar at (2,2,1). Game-layer object cells
/// are not stamped here — those are added via [`stamp_geometry`] / [`GeometrySpec`].
pub fn test_room() -> Map {
    const SX: usize = 6;
    const SY: usize = 6;
    const SZ: usize = 4;
    let mut map = Map::new(SX, SY, SZ);

    for z in 0..SZ {
        for y in 0..SY {
            for x in 0..SX {
                let on_perimeter = x == 0 || x == SX - 1 || y == 0 || y == SY - 1;
                let kind = if on_perimeter {
                    CellKind::WallSolid
                } else if z == 0 {
                    CellKind::Floor
                } else if z == SZ - 1 {
                    CellKind::Ceiling
                } else {
                    CellKind::Empty
                };
                map.set(x, y, z, kind);
            }
        }
    }

    // Interior pillar — gives the raycaster something to hit mid-room.
    map.set(2, 2, 1, CellKind::WallSolid);

    map
}

/// A wall-decal region for [`stamp_geometry`]: an inclusive box `(x0..=x1,
/// y0..=y1, z0..=z1)` overriding the room shell with `kind` (a wall surface).
/// Ported from legacy `DecalScratch` / `WallDecal` (the geometry-relevant
/// fields only — color/density are render concerns for later milestones).
#[derive(Clone, Copy, Debug)]
pub struct DecalRegion {
    /// Wall-surface kind this region overrides the room shell with.
    pub kind: CellKind,
    /// Box min x (inclusive).
    pub x0: i32,
    /// Box min y (inclusive).
    pub y0: i32,
    /// Box min z (inclusive).
    pub z0: i32,
    /// Box max x (inclusive).
    pub x1: i32,
    /// Box max y (inclusive).
    pub y1: i32,
    /// Box max z (inclusive).
    pub z1: i32,
}

/// A generic solid-object cell for [`stamp_geometry`]: position + a
/// game-defined tag stored in the cell's `side_data` field. The engine stamps
/// these as [`CellKind::Object`]; the game layer reads `side_data` to recover
/// whatever identifier it stored here.
#[derive(Clone, Copy, Debug)]
pub struct ObjectCell {
    /// Voxel x coordinate.
    pub x: i32,
    /// Voxel y coordinate.
    pub y: i32,
    /// Voxel z coordinate.
    pub z: i32,
    /// Game-defined tag stored verbatim in the cell's `side_data` field.
    pub tag: u16,
}

/// Which voxel SHAPE a [`HoloObject`] fills inside its bounding box. The shape
/// only decides which cells are stamped [`CellKind::Hologram`]; the renderer
/// derives the glowing wireframe from the resulting silhouette/edges, so a box
/// reads as a wireframe cube and a stepped sphere/pyramid as its own outline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum HoloShape {
    /// A filled axis-aligned box (the canonical holographic CUBE).
    #[default]
    Box,
    /// A stepped square pyramid: each z layer (bottom→top) shrinks toward the
    /// centre, so the silhouette tapers to an apex.
    Pyramid,
    /// A stepped ellipsoid (a voxel "sphere") inscribed in the bounding box.
    Sphere,
}

impl HoloShape {
    /// Parse an authored shape name (case-insensitive); unknown ⇒ `None`.
    pub fn from_str_opt(s: &str) -> Option<HoloShape> {
        match s.trim().to_ascii_lowercase().as_str() {
            "box" | "cube" => Some(HoloShape::Box),
            "pyramid" => Some(HoloShape::Pyramid),
            "sphere" | "ellipsoid" => Some(HoloShape::Sphere),
            _ => None,
        }
    }
}

/// One authored HOLOGRAM object for [`stamp_geometry`]: an inclusive bounding box
/// `(x0..=x1, y0..=y1, z0..=z1)`, a [`HoloShape`] that selects which interior
/// cells fill, and a `style` tag (0..=255 colour/look variant) stored in each
/// stamped cell's `side_data`. Every filled cell becomes [`CellKind::Hologram`]
/// — a transparent voxel the raycaster passes through and renders as part of a
/// glowing contour wireframe.
#[derive(Clone, Copy, Debug)]
pub struct HoloObject {
    /// Which interior cells fill (see [`HoloShape`]).
    pub shape: HoloShape,
    /// Bounding box min x (inclusive).
    pub x0: i32,
    /// Bounding box min y (inclusive).
    pub y0: i32,
    /// Bounding box min z (inclusive).
    pub z0: i32,
    /// Bounding box max x (inclusive).
    pub x1: i32,
    /// Bounding box max y (inclusive).
    pub y1: i32,
    /// Bounding box max z (inclusive).
    pub z1: i32,
    /// Colour/look variant tag stored in `side_data` (0 = default blue look).
    pub style: u16,
}

impl HoloObject {
    /// Is world cell `(x,y,z)` part of this object's [`HoloShape`] fill? Pure
    /// integer geometry — no allocation, deterministic.
    fn contains(&self, x: i32, y: i32, z: i32) -> bool {
        let (x0, x1) = (self.x0.min(self.x1), self.x0.max(self.x1));
        let (y0, y1) = (self.y0.min(self.y1), self.y0.max(self.y1));
        let (z0, z1) = (self.z0.min(self.z1), self.z0.max(self.z1));
        if x < x0 || x > x1 || y < y0 || y > y1 || z < z0 || z > z1 {
            return false;
        }
        match self.shape {
            HoloShape::Box => true,
            HoloShape::Pyramid => {
                // Bottom layer (z0) is full; each layer up shrinks the inset by 1
                // ring per z step, tapering to a centred apex.
                let inset = z - z0;
                x >= x0 + inset && x <= x1 - inset && y >= y0 + inset && y <= y1 - inset
            }
            HoloShape::Sphere => {
                // Stepped ellipsoid: normalised radius test about the box centre.
                let cx = (x0 + x1) as f32 * 0.5;
                let cy = (y0 + y1) as f32 * 0.5;
                let cz = (z0 + z1) as f32 * 0.5;
                let rx = ((x1 - x0) as f32 * 0.5).max(0.5);
                let ry = ((y1 - y0) as f32 * 0.5).max(0.5);
                let rz = ((z1 - z0) as f32 * 0.5).max(0.5);
                let nx = (x as f32 - cx) / rx;
                let ny = (y as f32 - cy) / ry;
                let nz = (z as f32 - cz) / rz;
                nx * nx + ny * ny + nz * nz <= 1.0 + 1e-3
            }
        }
    }
}

/// The static-geometry inputs for [`stamp_geometry`], borrowed from a parsed
/// level. This is the engine-agnostic seam between the level parser and
/// `re_world` (the voxel grid) — `re_world` stays free of any dependency on
/// the level format or game vocabulary.
#[derive(Clone, Copy, Debug)]
pub struct GeometrySpec<'a> {
    /// Room dimensions (sx, sy, sz).
    pub size: (usize, usize, usize),
    /// Perimeter-wall surface kind from `[room] ambient` (see
    /// [`ambient_to_wall_kind`]).
    pub wall_kind: CellKind,
    /// Wall-decal override regions.
    pub decals: &'a [DecalRegion],
    /// Generic solid-object cells stamped as [`CellKind::Object`]. Each cell's
    /// `tag` is stored verbatim in `side_data`; the game layer reads it back to
    /// distinguish different objects. The engine treats all of them identically
    /// as opaque non-decal solids.
    pub objects: &'a [ObjectCell],
}

/// Map a `[room] ambient` string to its perimeter-wall [`CellKind`].
///
/// Port of legacy `ambient_to_wall_kind`:
/// * `"trace"` / `"trace-pattern"` → [`CellKind::WallTrace`]
/// * `"matrix"` → [`CellKind::WallMatrix`]
/// * `"solid"` / anything else → [`CellKind::WallSolid`]
/// * empty / unset → [`CellKind::WallTrace`] (matches the shipped levels)
pub fn ambient_to_wall_kind(ambient: &str) -> CellKind {
    if ambient.is_empty() {
        return CellKind::WallTrace;
    }
    if ambient.starts_with("trace") {
        CellKind::WallTrace
    } else if ambient == "matrix" {
        CellKind::WallMatrix
    } else {
        CellKind::WallSolid
    }
}

/// Stamp the static room geometry into `map`, porting legacy
/// `stamp_world_geometry`.
///
/// Order matters (later writes win):
/// 1. **Shell** — perimeter walls at every z (`spec.wall_kind`); interior
///    floor at z=0; interior ceiling at z=top. Interior air stays [`CellKind::Empty`].
/// 2. **Decals** — each region overrides the shell on its patch.
/// 3. **Objects** — each [`ObjectCell`] is stamped as [`CellKind::Object`]
///    with its `tag` stored in `side_data`. An object cell always wins the cell
///    it occupies regardless of what was there before.
///
/// The `map` is reset to all-[`CellKind::Empty`] of `spec.size` first.
pub fn stamp_geometry(spec: &GeometrySpec) -> Map {
    let (sx, sy, sz) = spec.size;
    let mut map = Map::new(sx, sy, sz);

    // 1. ROOM SHELL.
    for z in 0..sz {
        for y in 0..sy {
            for x in 0..sx {
                let on_perimeter = x == 0 || x == sx - 1 || y == 0 || y == sy - 1;
                let kind = if on_perimeter {
                    spec.wall_kind
                } else if z == 0 {
                    CellKind::Floor
                } else if z == sz - 1 {
                    CellKind::Ceiling
                } else {
                    continue; // interior air stays Empty — skip the write
                };
                map.set(x, y, z, kind);
            }
        }
    }

    // 2. WALL DECALS: each region overrides the shell wall on that patch.
    for d in spec.decals {
        for z in d.z0..=d.z1 {
            for y in d.y0..=d.y1 {
                for x in d.x0..=d.x1 {
                    if x >= 0 && y >= 0 && z >= 0 {
                        map.set(x as usize, y as usize, z as usize, d.kind);
                    }
                }
            }
        }
    }

    // 3. OBJECTS: generic solid-object cells; side_data = game-defined tag.
    for o in spec.objects {
        if o.x >= 0 && o.y >= 0 && o.z >= 0 {
            map.set_cell(
                o.x as usize,
                o.y as usize,
                o.z as usize,
                CellKind::Object,
                o.tag,
            );
        }
    }

    map
}

/// Stamp transparent HOLOGRAM objects into an existing `map`, marking each
/// shape-filled cell [`CellKind::Hologram`] with the object's `style` tag in
/// `side_data`. Called by `re_level::build_map` AFTER [`stamp_geometry`].
///
/// Kept SEPARATE from [`GeometrySpec`] (rather than adding a field) so every
/// existing `GeometrySpec { … }` construction across the workspace keeps
/// compiling unchanged — a level with no `[holo]` section never calls this and
/// renders exactly as before (additive, back-compatible).
///
/// A hologram cell is NOT solid: the raycaster passes through it (the scene
/// behind still renders) and reconstructs the glowing wireframe from the
/// silhouette. A hologram never overwrites a solid wall/object cell — it only
/// fills cells that are currently [`CellKind::Empty`], so a hologram authored to
/// overlap geometry leaves the geometry intact.
pub fn stamp_holos(map: &mut Map, holos: &[HoloObject]) {
    for h in holos {
        let xlo = h.x0.min(h.x1).max(0);
        let xhi = h.x0.max(h.x1);
        let ylo = h.y0.min(h.y1).max(0);
        let yhi = h.y0.max(h.y1);
        let zlo = h.z0.min(h.z1).max(0);
        let zhi = h.z0.max(h.z1);
        for z in zlo..=zhi {
            for y in ylo..=yhi {
                for x in xlo..=xhi {
                    if !h.contains(x, y, z) {
                        continue;
                    }
                    if map.cell_at(x, y, z) == CellKind::Empty {
                        map.set_cell(
                            x as usize,
                            y as usize,
                            z as usize,
                            CellKind::Hologram,
                            h.style,
                        );
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Physics constants — `pub const` so `re_present` / the game can read or
// override them for tuning without reaching into crate internals.
// ---------------------------------------------------------------------------

/// Radius of the player's collision cylinder in cell units.
/// At 0.25 the player fits cleanly through a 1-cell-wide corridor.
pub const PLAYER_RADIUS: f32 = 0.25;

/// Eye height above the bottom of the occupied cell (in cell units).
/// `pos.z` is the eye position; feet are at `pos.z - PLAYER_EYE_HEIGHT`.
/// Standing on a floor cell (z=0) gives `pos.z = 1.0 + PLAYER_EYE_HEIGHT = 1.5`.
pub const PLAYER_EYE_HEIGHT: f32 = 0.5;

/// Eye-z when the player stands grounded on a normal floor cell (cell units): a
/// floor cell `z=0` has its top at `1.0`, and the eye sits [`PLAYER_EYE_HEIGHT`]
/// above that. THE single source of truth — `re_render` consumes this (re-export)
/// for its vertical-shear reference height instead of hand-copying the literal.
pub const STANDING_EYE_Z: f32 = 1.0 + PLAYER_EYE_HEIGHT;

/// Minimum clearance kept between the eye and the cell above it.
/// Prevents the player from rising into a solid ceiling.
pub const PLAYER_HEAD_ROOM: f32 = 0.3;

/// Gravitational acceleration in cell-units per second² (downward).
pub const GRAVITY: f32 = 20.0;

/// Vertical impulse applied at jump onset (cell-units per second, upward).
/// Derived from `v² = 2·g·h` with h ≈ 1.06 cells: sqrt(2·20·1.06) ≈ 6.5.
pub const JUMP_IMPULSE: f32 = 6.5;

/// Terminal falling speed (cell-units per second, downward).
/// Prevents tunnelling through thin floors on very long falls.
pub const MAX_FALL_SPEED: f32 = 30.0;

/// Coyote-time grace window (seconds): [`CameraPose::try_jump`] still fires if
/// pressed within this long after the player last had `grounded == true`, even
/// though they have already walked off the ledge (Swaggerfall-study item 5 —
/// "never fall from a building prematurely"). Tracked by
/// [`PlayerMotion::since_grounded`], which [`CameraPose::step_gravity`] resets
/// to `0.0` every frame the player is grounded and otherwise counts up by
/// `dt`. Does NOT grant a double jump — see
/// [`PlayerMotion::jumped_airborne`].
pub const COYOTE_TIME: f32 = 0.10;

/// Jump-buffer window (seconds): a jump press that arrives while airborne and
/// not (yet) coyote-eligible is remembered for this long and fires
/// automatically the moment [`CameraPose::step_gravity`] lands the player,
/// rather than being silently dropped. Tracked by
/// [`PlayerMotion::jump_buffered`].
pub const JUMP_BUFFER_TIME: f32 = 0.10;

// ---------------------------------------------------------------------------
// PlayerMotion component — the per-player vertical-physics state.
// ---------------------------------------------------------------------------

/// Vertical-physics state carried alongside a [`CameraPose`] by the player
/// entity. Intentionally minimal: it is pure data; the physics step lives on
/// [`CameraPose::step_gravity`] so it can be called deterministically in
/// unit tests without a Bevy world.
///
/// # Intended use by `re_present`
///
/// Each frame (inside the `Update` schedule, after input):
///
/// ```text
/// // 1. Grounded movement (horizontal).
/// pose.move_with_collision(dx, dy, map, PLAYER_RADIUS);
///
/// // 2. Gravity / jump (vertical).
/// pose.step_gravity(&mut motion, dt, map);
///
/// // 3. Optional debug-fly (replaces the above when noclip is active):
/// pose.apply_height(delta_z, map);
/// ```
///
/// `grounded` is set by `step_gravity`. The input handler no longer needs to
/// gate the call itself — call `try_jump` unconditionally on every press
/// edge; it internally decides whether to fire immediately (grounded or
/// within coyote time), buffer the press for landing, or (already used this
/// airborne phase) do nothing.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct PlayerMotion {
    /// Vertical velocity in cell-units per second.
    /// Positive = moving upward; negative = falling.
    pub vz: f32,
    /// True when the player is resting on a solid surface this frame.
    /// Set by [`CameraPose::step_gravity`]; used by the input handler to
    /// gate [`CameraPose::try_jump`].
    pub grounded: bool,
    /// Seconds since `grounded` was last `true`. [`CameraPose::step_gravity`]
    /// resets this to `0.0` every frame the player is grounded and otherwise
    /// counts it up by `dt`. [`CameraPose::try_jump`] compares it against
    /// [`COYOTE_TIME`] to grant the coyote-time grace window. A freshly
    /// constructed (`Default`) motion starts far past the window — no free
    /// coyote jump before the player has ever touched ground.
    pub since_grounded: f32,
    /// Seconds elapsed since an airborne jump press was buffered, or `None`
    /// when no press is pending. Set to `Some(0.0)` by [`CameraPose::try_jump`]
    /// whenever a press cannot fire immediately; [`CameraPose::step_gravity`]
    /// ages it by `dt` each airborne frame, drops it once it exceeds
    /// [`JUMP_BUFFER_TIME`], and — the point of buffering — fires the jump
    /// automatically the instant the player lands within the window.
    pub jump_buffered: Option<f32>,
    /// `true` once a jump (grounded or coyote-window) has fired during the
    /// CURRENT airborne phase. Blocks a second coyote-window press from
    /// granting a double jump. Cleared by [`CameraPose::step_gravity`]
    /// whenever the player regains the ground.
    pub jumped_airborne: bool,
}

impl Default for PlayerMotion {
    fn default() -> Self {
        // `since_grounded` starts at (effectively) "a very long time ago" —
        // conservative: a brand-new motion (e.g. spawned mid-air by a test
        // fixture) gets no free coyote jump before `step_gravity` has ever
        // run. Real gameplay motions reach a true `since_grounded == 0.0`
        // the first time `step_gravity` grounds them.
        PlayerMotion {
            vz: 0.0,
            grounded: false,
            since_grounded: f32::INFINITY,
            jump_buffered: None,
            jumped_airborne: false,
        }
    }
}

// ---------------------------------------------------------------------------

/// The camera pose: continuous world position + yaw/pitch in radians.
///
/// `pos` is in cell units (cell n spans `[n, n+1)`); yaw=0 faces +y (south).
#[derive(Component, Clone, Copy, Debug)]
#[cfg_attr(feature = "reflect", derive(bevy::reflect::Reflect))]
#[cfg_attr(feature = "reflect", reflect(Component))]
pub struct CameraPose {
    // NOTE: the workspace uses glam 0.33 while bevy_reflect 0.18 only provides
    // Reflect impls for glam 0.30. The two versions are distinct types in the dep
    // tree, so Vec3 cannot implement Reflect here. We ignore it so the rest of
    // CameraPose (yaw, pitch) is inspectable over BRP. To inspect position, read
    // the raw component via bevy_remote's GET and parse the debug output, or store
    // individual f32 fields. This can be revisited when the workspace glam is
    // unified with bevy's (both become 0.30 or both 0.33).
    /// World position, in cell units (cell `n` spans `[n, n+1)`).
    #[cfg_attr(feature = "reflect", reflect(ignore))]
    pub pos: Vec3,
    /// Facing angle in radians; `0` faces `+y` (south).
    pub yaw: f32,
    /// Look up/down angle in radians.
    pub pitch: f32,
}

// ---------------------------------------------------------------------------
// Private helpers.
// ---------------------------------------------------------------------------

/// The height (local z in `[0, 1]`) of the solid surface in a non-`Full` cell
/// at the given local x/y position, for use by step_gravity's floor-contact
/// logic.
///
/// For [`Shape::Full`] returns `1.0` (top of the cell, same as before).
/// For [`Shape::WedgeXY`] returns `1.0` if the x/y is in the solid half, `0.0`
/// if it is in the air half (the wedge has a vertical face, not a sloped top).
/// For [`Shape::RampZ`] returns the slope height at that x/y — i.e., the z of
/// the 45° surface so gravity can rest the player exactly on the slope rather
/// than snapping to the full cell top.
///
/// The returned value is in cell-local space; the caller adds the cell-floor
/// world z to get the world-space surface height.
#[inline]
fn shape_surface_z(shape: Shape, local_x: f32, local_y: f32) -> f32 {
    use re_core::{DiagXY, RampDir};
    match shape {
        Shape::Full => 1.0,
        Shape::WedgeXY(diag) => {
            // A wedge has a vertical face — its "top surface" within the solid
            // half is still the full cell top (z=1.0).
            let in_solid = match diag {
                DiagXY::NE => local_x + local_y >= 1.0,
                DiagXY::NW => local_x <= local_y,
                DiagXY::SE => local_x >= local_y,
                DiagXY::SW => local_x + local_y <= 1.0,
            };
            if in_solid {
                1.0
            } else {
                0.0
            }
        }
        Shape::RampZ(dir) => {
            // The slope height at (local_x, local_y) — clamp to [0, 1].
            match dir {
                RampDir::PosX => local_x.clamp(0.0, 1.0),
                RampDir::NegX => (1.0 - local_x).clamp(0.0, 1.0),
                RampDir::PosY => local_y.clamp(0.0, 1.0),
                RampDir::NegY => (1.0 - local_y).clamp(0.0, 1.0),
            }
        }
    }
}

/// Wrap `yaw` into `(−π, π]` — used after adding a portal delta-yaw so the
/// angle stays bounded regardless of how many portal crossings accumulate.
#[inline]
fn normalise_yaw(yaw: f32) -> f32 {
    use std::f32::consts::TAU;
    let y = yaw % TAU;
    if y > std::f32::consts::PI {
        y - TAU
    } else if y <= -std::f32::consts::PI {
        y + TAU
    } else {
        y
    }
}

/// A single collision probe: the floored cell index `(cx, cy, cz)` together with
/// the continuous world point that produced it. The cell index and world point
/// always travel together through the axis-probe collision path, so bundling
/// them keeps [`CameraPose::solid_or_oob`]'s signature small without changing
/// what is passed. `Copy` + by-value means no allocation and no perf change.
#[derive(Clone, Copy)]
struct CellProbe {
    cx: i32,
    cy: i32,
    cz: i32,
    /// Continuous world coordinates of the probe point (`(wx, wy, wz)`).
    point: Vec3,
}

impl CameraPose {
    /// Spawn pose for the test room: standing on the interior air level
    /// (z=1), near the south wall looking north toward the pillar.
    pub fn spawn_in_test_room() -> Self {
        CameraPose {
            // Cell centre at (3, 4, 1): a bit south of the pillar, looking north.
            pos: Vec3::new(3.5, 4.5, 1.5),
            yaw: std::f32::consts::PI, // +PI faces -y (north) toward the pillar
            pitch: 0.0,
        }
    }

    /// Spawn at a level's `[player]` cell, facing the given discrete
    /// `facing_yaw_col` (legacy encoding: `+y=0, +x=120, -y=240, -x=360`,
    /// PANORAMIC_COLS=480 = a full revolution).
    ///
    /// The camera is placed at the **centre** of the spawn cell so it sits in
    /// interior air, and `facing_yaw_col` is converted to a yaw in radians.
    /// yaw=0 faces +y (south), matching [`dir`](Self::dir).
    pub fn spawn_at(x: i32, y: i32, z: i32, facing_yaw_col: i32) -> Self {
        // 480 columns == 2π. yaw=0 faces +y.
        let yaw = (facing_yaw_col as f32 / 480.0) * std::f32::consts::TAU;
        CameraPose {
            pos: Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5),
            yaw,
            pitch: 0.0,
        }
    }

    /// Forward (facing) direction in the xy plane: `(sin yaw, cos yaw)`.
    #[inline]
    pub fn dir(&self) -> (f32, f32) {
        (self.yaw.sin(), self.yaw.cos())
    }

    /// Apply a vertical (height) delta in cell units, clamped to the map's
    /// habitable z range and blocked by solid destination cells.
    ///
    /// The interior air band sits between the floor slab (z=0) and the ceiling
    /// slab (z=sz-1), so the camera centre is kept inside `[1.5, sz-1.5]` — i.e.
    /// the floored z layer stays in `[1, sz-2]`, never in the floor or ceiling.
    /// (For the degenerate `sz < 3` case the band collapses and the move is a
    /// no-op.) If the destination cell `(floor x, floor y, floor z)` is solid the
    /// move is skipped, so the player can't rise/sink into geometry.
    ///
    /// **Debug / free-fly only.** For the grounded play model use
    /// [`step_gravity`](Self::step_gravity) + [`try_jump`](Self::try_jump).
    ///
    /// Returns `true` if the pose actually moved.
    pub fn apply_height(&mut self, delta: f32, map: &Map) -> bool {
        let (_, _, sz) = map.size();
        if sz < 3 {
            return false; // no interior air band to move within
        }
        // Centre-of-cell band: lowest air layer centre is 1.5, highest is sz-1.5.
        let lo = 1.5;
        let hi = sz as f32 - 1.5;
        let target = (self.pos.z + delta).clamp(lo, hi);
        if (target - self.pos.z).abs() < f32::EPSILON {
            return false;
        }
        // Refuse to enter a solid destination cell (e.g. a pillar column).
        // Shape-aware: for non-Full cells check whether the specific world point
        // (cx+0.5, cy+0.5, target) is inside the solid part (free-fly navigates
        // to cell centres in x/y, so x+0.5 and y+0.5 are representative locals).
        let (cx, cy) = (self.pos.x.floor() as i32, self.pos.y.floor() as i32);
        let cz_dest = target.floor() as i32;
        let kind_dest = map.physics_cell_at(cx, cy, cz_dest);
        if kind_dest.is_solid() {
            let shape_dest = map.shape_at(cx, cy, cz_dest);
            if shape_dest == Shape::Full {
                return false;
            }
            // For non-Full cells use local coords relative to the destination cell.
            let local = Vec3::new(
                self.pos.x - cx as f32,
                self.pos.y - cy as f32,
                target - cz_dest as f32,
            );
            if shape_solid_contains(shape_dest, local) {
                return false;
            }
        }
        self.pos.z = target;
        true
    }

    // -----------------------------------------------------------------------
    // Horizontal collision (wall-slide).
    // -----------------------------------------------------------------------

    /// Move the camera by `(dx, dy)` in cell units while colliding with solid
    /// cells, keeping to the current z layer.
    ///
    /// The two axes are applied **independently** so the player slides along a
    /// wall when one axis is blocked. The movement is blocked on an axis when
    /// the destination circle (centre at the new position, radius `radius`)
    /// would overlap a solid cell on that axis.
    ///
    /// **Radius probe:** for each axis the four "probe points" at ±`radius`
    /// along that axis (at each of the ±`radius` offsets on the _other_ axis)
    /// are sampled. Concretely:
    ///
    /// * X probe at `(x + dx ± radius,  y ± radius, z)` — four cells.
    /// * Y probe at `(x ± radius,  y + dy ± radius, z)` — four cells.
    ///
    /// This catches corner clips: a diagonal move toward a concave corner is
    /// correctly blocked on both axes.
    ///
    /// Out-of-bounds cells are treated as **solid** (walls at map edges), so
    /// the player can never leave the map.
    ///
    /// Returns a `(moved_x, moved_y)` pair indicating which axes actually
    /// moved. The pose is updated in-place.
    pub fn move_with_collision(
        &mut self,
        dx: f32,
        dy: f32,
        map: &Map,
        radius: f32,
    ) -> (bool, bool) {
        let z = self.pos.z.floor() as i32;

        // Snapshot the position BEFORE the move for the portal crossing test.
        let pos_before = self.pos;

        // The eye z stays constant during horizontal movement; the z world
        // coordinate for the local-coord probe is the eye z itself.
        let wz = self.pos.z;

        // --- X axis ---
        let moved_x = if dx.abs() > f32::EPSILON {
            let new_x = self.pos.x + dx;
            // Probe the leading edge in X at both ±radius offsets in Y.
            let blocked = [-radius, radius].iter().any(|&oy| {
                let px = new_x + dx.signum() * radius;
                let py = self.pos.y + oy;
                self.solid_or_oob(
                    map,
                    CellProbe {
                        cx: px.floor() as i32,
                        cy: py.floor() as i32,
                        cz: z,
                        point: Vec3::new(px, py, wz),
                    },
                )
            });
            if !blocked {
                self.pos.x = new_x;
                true
            } else {
                false
            }
        } else {
            false
        };

        // --- Y axis ---
        let moved_y = if dy.abs() > f32::EPSILON {
            let new_y = self.pos.y + dy;
            // Probe the leading edge in Y at both ±radius offsets in X.
            let blocked = [-radius, radius].iter().any(|&ox| {
                let px = self.pos.x + ox;
                let py = new_y + dy.signum() * radius;
                self.solid_or_oob(
                    map,
                    CellProbe {
                        cx: px.floor() as i32,
                        cy: py.floor() as i32,
                        cz: z,
                        point: Vec3::new(px, py, wz),
                    },
                )
            });
            if !blocked {
                self.pos.y = new_y;
                true
            } else {
                false
            }
        } else {
            false
        };

        // After the move resolves, test the before→after segment against all
        // portal planes. A crossing within the opening bounds warps the pose.
        self.apply_portal_crossings(pos_before, map);

        (moved_x, moved_y)
    }

    // -----------------------------------------------------------------------
    // Seamless portal teleport (Stage 2).
    // -----------------------------------------------------------------------

    /// Test the segment `pos_before → self.pos` against each registered portal
    /// plane. On a valid crossing (plane crossed + within opening bounds), apply
    /// the portal's `delta` transform to `pos`, and rotate `yaw` by the same
    /// rigid transform. At most **one** portal crossing is applied per call (the
    /// first one found), preventing a double-warp if `pos_before` and `self.pos`
    /// span two portals simultaneously (physically impossible at normal walking
    /// speeds but guards correctness).
    ///
    /// ## Crossing test (ported from NonEuclidean `Portal::Intersects`)
    ///
    /// 1. **Plane side:** `dot(n, before − p)` and `dot(n, after − p)` must have
    ///    opposite signs — the segment crossed the plane.
    /// 2. **Opening bounds:** the crossing point `q` is projected onto `half_u`
    ///    and `half_v`. If `|proj_u| < |half_u|²` AND `|proj_v| < |half_v|²`
    ///    (i.e. within the rectangle), the crossing is valid.
    ///
    /// ## Pose re-base
    ///
    /// * `self.pos  ← delta.transform_point3(pos_after) + exit_dir * 1e-3`
    ///   where `exit_dir = delta.transform_vector3(-plane_normal)` is the
    ///   room-side exit direction in the destination (nudges the player INTO
    ///   the room, preventing an immediate re-trigger of the destination portal).
    /// * `self.yaw  ← atan2(new_dir.x, new_dir.y)` where
    ///   `new_dir = delta.transform_vector3((sin yaw, cos yaw, 0))` —
    ///   the player's 3-D forward vector is transformed by the same rigid
    ///   transform the raycaster applies to a ray direction, then the new yaw
    ///   is read back via the inverse of the `dir()` convention.
    ///   Pitch is preserved — MVP portals are vertical.
    ///
    /// Pure compute: no allocation, no terminal, deterministic.
    pub fn apply_portal_crossings(&mut self, pos_before: Vec3, map: &Map) {
        if map.portals.is_empty() {
            return;
        }

        let pos_after = self.pos;

        for portal in &map.portals {
            let n = portal.plane_normal;
            let p = portal.plane_pos;

            // --- 1. Plane-side test ---
            let da = n.dot(pos_before - p);
            let db = n.dot(pos_after - p);

            // Both on the same side (or one exactly on the plane) → no crossing.
            if da * db > 0.0 {
                continue;
            }

            // --- 2. Compute the crossing point on the plane ---
            // q = pos_before + (pos_after − pos_before) * t, where t = da/(da−db).
            // Guard against degenerate da == db (parallel to plane, no crossing).
            let denom = da - db;
            if denom.abs() < f32::EPSILON {
                continue;
            }
            let t = da / denom;
            let q = pos_before + (pos_after - pos_before) * t - p; // relative to plane origin

            // --- 3. Opening bounds test (rectangular) ---
            // half_u and half_v are the in-plane axes SCALED by the half-extents.
            // Project q onto each axis; the crossing is valid if the projection
            // magnitude is strictly less than the squared magnitude of the half-axis
            // (because |q·half| / |half|² ≤ 1 ↔ the normalised coord is in [−1,1]).
            let hu = portal.half_u;
            let hv = portal.half_v;
            let hu_sq = hu.dot(hu);
            let hv_sq = hv.dot(hv);
            // Degenerate opening (zero area) — skip.
            if hu_sq < f32::EPSILON || hv_sq < f32::EPSILON {
                continue;
            }
            let proj_u = q.dot(hu);
            let proj_v = q.dot(hv);
            if proj_u.abs() >= hu_sq || proj_v.abs() >= hv_sq {
                continue; // outside the opening rectangle
            }

            // --- 4. Valid crossing — re-base the pose ---
            self.pos = portal.delta.transform_point3(pos_after);

            // Nudge the re-based position slightly INTO the destination room
            // to prevent an immediate re-trigger of the destination portal on
            // the very next move call. When pos_after lands exactly on the
            // source plane (t = 1.0 at the end of a step), the re-based point
            // is exactly on the destination plane; without a nudge the next
            // move starts there, `da = 0`, and the destination portal would
            // fire immediately (doubling back). Pushing by 1e-3 world-units
            // along the exit direction (INTO the destination room) ensures
            // `da > 0` for the destination portal so `da * db > 0 → continue`
            // on the next call.
            //
            // The exit direction is the travel direction AFTER the portal
            // transform: `delta.transform_vector3(-plane_normal)`. We negate
            // `plane_normal` because the player is travelling opposite to the
            // source inward-normal (into the portal, not away from it). The
            // portal delta (which includes a 180° flip) maps this arrival
            // direction to the room-side exit direction in the destination,
            // which is what we need.
            //
            // Example (demo portal pair): source normal = +Y, player travels
            // in −Y. delta_AB(−Y) = +X = portal B's room-side direction.
            // Nudging +X puts the player at x > 1.0 (room side of portal B),
            // so portal B's `da` is positive and the next step's `da*db > 0`
            // prevents a second crossing.
            let exit_dir = portal.delta.transform_vector3(-portal.plane_normal);
            self.pos += exit_dir * 1e-3;

            // Re-base the facing direction through the SAME rigid transform the
            // raycaster applies to a ray direction, so the view seen through the
            // portal and the direction walked into it agree exactly.
            //
            // Convention: dir() == (yaw.sin(), yaw.cos()), so the 3-D forward
            // vector in the (x, y, z) world frame is (sin yaw, cos yaw, 0).
            // After mapping through `delta` we read back the new yaw with the
            // same inverse: yaw = atan2(x', y').
            //
            // This replaces the former additive delta_yaw approach, which was
            // only correct when yaw == 0 (where old_dir == Vec3::Y and both
            // methods agree). For any other initial yaw the additive form chose
            // the wrong exit direction — the raycaster and the collision layer
            // disagreed, producing an apparent wall-facing teleport.
            let old_dir = Vec3::new(self.yaw.sin(), self.yaw.cos(), 0.0);
            let new_dir = portal.delta.transform_vector3(old_dir);
            // atan2(sin, cos) is the inverse of the dir() convention.
            // Result is already in (−π, π], so normalise_yaw is a no-op here,
            // but we keep the call so callers that inspect self.yaw always see
            // a bounded value regardless of future convention changes.
            self.yaw = normalise_yaw(new_dir.x.atan2(new_dir.y));

            // One warp per move — stop after the first valid crossing.
            return;
        }
    }

    /// Returns `true` if the world point `(wx, wy, wz)` is inside solid geometry
    /// or out-of-bounds.  Out-of-bounds counts as solid so movement never leaves
    /// the map.
    ///
    /// Uses [`CellKind::is_collidable`] (not `is_solid`) so that
    /// [`CellKind::Portal`] cells are **walkable** — the player passes through
    /// them and the portal-crossing logic in
    /// [`CameraPose::apply_portal_crossings`] handles the seamless teleport.
    /// The raycaster still sees portals as solid (via `is_solid`) and gets its
    /// DDA hit to re-base.  The two predicates are intentionally different: this
    /// one is the collision gate.
    ///
    /// ## Shape-aware probe (M3a)
    ///
    /// For a cell whose [`Shape`] is [`Shape::Full`] the behaviour is unchanged:
    /// the cell is solid everywhere.  For a non-`Full` cell the probe converts
    /// the world point to cell-local coordinates `local = (wx−cx, wy−cy, wz−cz)`
    /// and delegates to [`shape_solid_contains`], which returns `true` only when
    /// `local` is in the solid half of the wedge or below the ramp surface.  This
    /// lets a player walk through the open triangle of a diagonal wall and ascend
    /// a ramp.
    ///
    /// The `cx/cy/cz` parameters are the floored cell indices (pre-computed by
    /// the callers for the axis-probe grid); `wx/wy/wz` are the continuous world
    /// coordinates of the probe point from which the local offset is derived.
    #[inline]
    fn solid_or_oob(&self, map: &Map, probe: CellProbe) -> bool {
        let CellProbe { cx, cy, cz, point } = probe;
        if !map.in_bounds(cx, cy, cz) {
            return true; // treat map edges as solid
        }
        let kind = map.physics_cell_at(cx, cy, cz);
        if !kind.is_collidable() {
            return false;
        }
        // For Full cells the shape test is a no-op — is_collidable() is the gate.
        // For non-Full cells consult the shape.
        let shape = map.shape_at(cx, cy, cz);
        if shape == Shape::Full {
            // Allow the player to step UP onto a Full-solid cell when their
            // FEET are below the cell's floor. This covers spiral staircase
            // landings and any raised floor edge the player climbs to:
            //
            //   cz = floor(eye_z).  feet_z = eye_z − EYE_HEIGHT.
            //   If feet_z < cz (feet are in a lower z-cell), the body only
            //   overlaps this cell at eye level but not at foot level.  This
            //   is a "step-up" — the player walks into the cell and
            //   step_gravity snaps them to the top surface (cz + 1).
            //
            //   If feet_z >= cz (feet inside the cell) the cell is a genuine
            //   wall the body would penetrate — block as before.
            //
            // ROOT CAUSE of the spiral staircase block: tread-8 RampZ(NegY)
            // raises the player's eye to ≈8.0 (cz=8) while feet stay at 7.5.
            // The final landing (5,5,8) is Full/WallSolid at z=8.  Before this
            // fix, solid_or_oob returned true unconditionally → permanently
            // blocked.  Now feet_z (7.5) < cz (8) → step-up → allowed.
            let feet_z = point.z - PLAYER_EYE_HEIGHT;
            if feet_z < cz as f32 {
                return false; // step-up surface: player walks onto it
            }
            return true;
        }
        // RampZ cells are NEVER blocking for horizontal movement.
        //
        // A ramp is a sloped walkable surface, not a wall. The player must be
        // able to enter a ramp's XY footprint from any horizontal direction;
        // `step_gravity` (ramp-snap, step 3) then places the feet on the slope
        // surface. Treating a RampZ cell as a solid wall here causes the player
        // to be permanently blocked when their eye-level z-position is above
        // the slope surface at the probe's local_x/local_y (which happens when
        // approaching from any direction other than the ramp's ascending
        // direction, or when turning a corner in a spiral staircase).
        //
        // WedgeXY cells retain their horizontal blocking (they are diagonal
        // WALLS, not walkable slopes).
        if matches!(shape, Shape::RampZ(_)) {
            return false;
        }
        // WedgeXY: convert world probe point to cell-local coordinates and
        // delegate to shape_solid_contains.
        let local = Vec3::new(
            point.x - cx as f32,
            point.y - cy as f32,
            point.z - cz as f32,
        );
        shape_solid_contains(shape, local)
    }

    // -----------------------------------------------------------------------
    // Gravity + jump.
    // -----------------------------------------------------------------------

    /// Integrate gravity and vertical movement for one frame of duration `dt`
    /// seconds.
    ///
    /// # Physics model
    ///
    /// * Gravity accelerates `motion.vz` downward each frame.
    /// * `motion.vz` is clamped to `[-MAX_FALL_SPEED, MAX_FALL_SPEED]` to
    ///   prevent tunnelling through thin floors on large `dt` spikes.
    /// * The camera moves vertically by `vz * dt`.
    /// * **Floor:** the "feet" sit at `pos.z - PLAYER_EYE_HEIGHT`.  If, after
    ///   the move, the cell directly below the feet is a [`CellKind::Floor`]
    ///   **or** a solid cell (wall/object), the player is snapped to the top of
    ///   that cell (`z_cell + 1.0 + PLAYER_EYE_HEIGHT`), `vz` is zeroed, and
    ///   `grounded` is set to `true`.
    /// * **Ceiling:** the "head" sits at `pos.z + PLAYER_HEAD_ROOM`.  If that
    ///   point would enter a solid cell after the move, the move is clamped to
    ///   the underside of that cell and `vz` is zeroed (no bonking through the
    ///   ceiling).
    /// * **Map bounds:** the eye is hard-clamped into `[PLAYER_EYE_HEIGHT,
    ///   sz as f32 - 1.0 - PLAYER_HEAD_ROOM]` so the player can never escape
    ///   the map vertically.
    ///
    /// # Interaction with `apply_height`
    ///
    /// `apply_height` is a free-fly / debug tool.  When using the grounded
    /// model, call only `step_gravity` (and `try_jump`) — never both in the
    /// same frame for the same entity.
    pub fn step_gravity(&mut self, motion: &mut PlayerMotion, dt: f32, map: &Map) {
        let (_, _, sz) = map.size();
        if sz < 3 {
            return; // degenerate map — no room to fall
        }

        let cx = self.pos.x.floor() as i32;
        let cy = self.pos.y.floor() as i32;

        // Hard map-boundary eye limits:
        //   floor: top of floor cell z=0 is 1.0 → eye sits at 1.0 + EYE_HEIGHT.
        //   ceiling: bottom of ceiling cell (sz-1) is (sz-1) as f32 → eye at
        //            (sz-1) as f32 - HEAD_ROOM.
        let eye_lo = 1.0 + PLAYER_EYE_HEIGHT; // = 1.5 for standard rooms
        let eye_hi = (sz - 1) as f32 - PLAYER_HEAD_ROOM;

        // 1. Apply gravity and clamp terminal velocity.
        motion.vz -= GRAVITY * dt;
        motion.vz = motion.vz.clamp(-MAX_FALL_SPEED, MAX_FALL_SPEED);

        // 2. Candidate new eye position.
        let mut new_z = self.pos.z + motion.vz * dt;

        // -----------------------------------------------------------------------
        // 3. RampZ surface tracking (staircase seam fix).
        //
        // A single ramp cell already snaps the player to the slope via the flat
        // floor check (step 5 below). The seam between two adjacent ramp cells is
        // the tricky case:
        //
        //   ASCENDING seam: after `move_with_collision` the player has stepped
        //   from cell (N, y, z_r) to cell (N+1, y, z_r+1). Their feet were snapped
        //   to z_r+~1.0 by the previous ramp cell; the new ramp cell's base
        //   surface is also z_r+1+0 = z_r+1, but `feet_z.floor()` = z_r, not
        //   z_r+1. Step 5 looks at the cell at that floor index, finds it Empty,
        //   and does nothing → the player floats one frame.
        //
        //   DESCENDING seam: the player walks from (N, y, z_r) to (N-1, y, z_r-1).
        //   Feet snap to z_r+high_x; in the new cell the surface is z_r-1+high_x
        //   which is LOWER than the current feet. Step 5 only snaps UP (feet went
        //   BELOW surface); it ignores feet that are ABOVE. Gravity eventually
        //   pulls them down but `grounded` flickers for 1-2 frames.
        //
        // Fix: when not jumping (vz ≤ 0), search the three cells at fc-1, fc, fc+1
        // for a `RampZ` surface within 1 cell of the player's feet. When found,
        // snap immediately to the slope surface (up OR down), zero vz, and set
        // grounded. This replaces the per-frame gravity drop at every cell seam
        // with an instantaneous slope-follow — exactly the Quake step-down model
        // applied to 45° ramp geometry.
        //
        // The 1-cell snap window is intentional: it covers the seam gap (at most
        // 1 cell across a ramp boundary) while not grabbing ramp cells the player
        // legitimately jumped over (they'd be ≫1 cell away in mid-air).
        //
        // Flat Floor / WedgeXY cells are NOT snap-tracked here — only RampZ.
        // Flat floor behaviour is identical to before (step 5 below).
        // -----------------------------------------------------------------------
        // `ramp_snapped` records whether THIS call's ramp-seam search (below)
        // resolved a resting contact. Step 4 used to gate on `!motion.grounded`
        // instead — but `motion.grounded` is a PERSISTENT field, last written by
        // step 7 of the PREVIOUS call, so it stays stale/`true` for one whole
        // extra frame after the player's column changes underneath them (e.g.
        // stepping off a ramp tread onto a taller landing whose supporting
        // column is a different `(cx, cy)`). Gating on the stale flag suppressed
        // the one check that could have caught the seam, letting an unrestrained
        // fall build momentum before the flat-floor check ever got a chance to
        // run with `fc` still adjacent to the missed surface. A local flag reset
        // every call closes that gap without changing what `motion.grounded`
        // means to callers.
        let mut ramp_snapped = false;
        if motion.vz <= 0.0 {
            let feet_z = new_z - PLAYER_EYE_HEIGHT;
            let fc_base = feet_z.floor() as i32;
            let lx = self.pos.x - cx as f32;
            let ly = self.pos.y - cy as f32;
            // Check fc, fc+1, fc-1 (in that priority order).
            'ramp_snap: for &dz in &[0i32, 1, -1] {
                let fc = fc_base + dz;
                if !map.in_bounds(cx, cy, fc) {
                    continue;
                }
                let fk = map.physics_cell_at(cx, cy, fc);
                // Only solid cells or Floor cells can be stood on.
                if fk != CellKind::Floor && !fk.is_solid() {
                    continue;
                }
                let fs = map.shape_at(cx, cy, fc);
                // Only apply ramp-snap logic to RampZ shapes; flat/full floors
                // (including a raised Full-solid landing) are handled by step 4
                // below, which has its own — necessarily asymmetric — window.
                let Shape::RampZ(_) = fs else {
                    continue;
                };
                let surface_world_z = fc as f32 + shape_surface_z(fs, lx, ly);
                let gap = feet_z - surface_world_z; // + = feet above, − = feet below
                                                    // Snap if within 1 cell (covers a full ramp step in either direction).
                if gap.abs() <= 1.0 + 1e-3 {
                    new_z = surface_world_z + PLAYER_EYE_HEIGHT;
                    motion.vz = 0.0;
                    motion.grounded = true;
                    ramp_snapped = true;
                    // Don't return yet — the ceiling clamp (step 4) runs next to
                    // enforce headroom on stairs.  The ramp snap may push the
                    // player upward; the ceiling clamp corrects that without an
                    // early return so the hard-clamp and grounded paths always
                    // commit `self.pos.z` consistently.
                    break 'ramp_snap;
                }
            }
        }

        // -----------------------------------------------------------------------
        // 4. Flat/solid floor check — feet at `new_z - PLAYER_EYE_HEIGHT`.
        //
        // On a downward move (or after gravity): scan for the nearest floor/solid
        // surface at the feet's own cell (`fc_base`) or, failing that, the cell
        // directly ABOVE it (`fc_base + 1`). If found and the feet have
        // descended into or below its top (or the slope surface for a non-Full
        // shape), snap up to the surface.
        //
        // This step is skipped when the ramp-snap (step 3) already handled the
        // floor contact THIS call (`ramp_snapped`).
        //
        // Note: this step does NOT return early — it adjusts `new_z` and falls
        // through to the ceiling clamp (step 5) so that a snap-up never causes
        // the player to clip into a ceiling cell.
        //
        // ## Why `fc_base + 1` too (ROOT CAUSE — spiral staircase landing)
        // A raised Full-solid platform (e.g. a stair landing) can sit exactly
        // one cell above open air the player just walked out onto — the
        // horizontal collision probe (`solid_or_oob`'s step-up branch) already
        // grants entry to that platform's footprint whenever the player's feet
        // are below its floor, on the promise that gravity then snaps them onto
        // its top. Checking ONLY `fc_base` breaks that promise the instant
        // gravity nudges the feet a hair below the `fc_base + 1` boundary —
        // which happens on essentially every frame once `vz` goes negative: the
        // one cell that would have caught them (the platform itself) is one
        // index above where the naive floor-of-feet lookup lands, and the cell
        // actually under the feet is open air with nothing to catch the fall.
        //
        // This can never mistake a floating solid the player is genuinely
        // falling AWAY from below as a floor: `fc_base + 1`'s surface is always
        // `>= fc_base + 1 > feet_z` (by construction of `fc_base`), so the
        // "feet below surface" test at that index is trivially satisfied only
        // when it is reached at all — and it is reached only when `fc_base`
        // itself has no floor/solid cell to offer, i.e. exactly the seam gap.
        // If `fc_base` DOES have a floor/solid cell, the search stops there
        // (matching the pre-existing single-cell behaviour exactly) even when
        // that cell's surface does not (yet) catch the feet — a player merely
        // passing above a real floor is never boosted up an extra cell.
        //
        // ## Ramp cells checked here vs. step 3
        // Step 3 handles the ramp-to-ramp cross-seam case (fc±1, RampZ only).
        // This step handles flat/solid floors: the common case where the player
        // overshoots INTO the same floor cell (e.g. on the very first frame of
        // contact or after a jump), AND the raised-platform seam above.
        // -----------------------------------------------------------------------
        if motion.vz <= 0.0 && !ramp_snapped {
            let feet_z = new_z - PLAYER_EYE_HEIGHT;
            let fc_base = feet_z.floor() as i32;
            'flat_snap: for &dz in &[0i32, 1] {
                let fc = fc_base + dz;
                if !map.in_bounds(cx, cy, fc) {
                    break 'flat_snap;
                }
                let floor_kind = map.physics_cell_at(cx, cy, fc);
                let is_floor_surface = floor_kind == CellKind::Floor || floor_kind.is_solid();
                if !is_floor_surface {
                    // Nothing at this index — try the cell directly above.
                    continue;
                }
                // Found the (one and only) candidate surface for this call —
                // either it catches the feet or it doesn't, exactly like the
                // original single-cell check when `dz == 0` already has a floor.
                let floor_shape = map.shape_at(cx, cy, fc);
                // local x/y of the player within the floor cell.
                let lx = self.pos.x - cx as f32;
                let ly = self.pos.y - cy as f32;
                // World-z of the solid surface (full top or slope height).
                let surface_local_z = shape_surface_z(floor_shape, lx, ly);
                let floor_surface_world_z = fc as f32 + surface_local_z;
                // Overshot: feet are below the surface.
                if feet_z < floor_surface_world_z {
                    new_z = floor_surface_world_z + PLAYER_EYE_HEIGHT;
                    motion.vz = 0.0;
                    motion.grounded = true;
                    // Fall through to ceiling clamp, then hard-clamp and commit.
                }
                break 'flat_snap;
            }
        }

        // -----------------------------------------------------------------------
        // 5. Ceiling clamp — prevent the head from entering a solid cell.
        //
        // Runs unconditionally after all floor-snap steps so that:
        //   (a) Active jumps (vz > 0) are blocked — the original behaviour.
        //   (b) Ramp-snap upward steps (step 3) are blocked when a low ceiling
        //       sits above the slope — the "headroom on stairs" requirement.
        //   (c) Flat-floor snap-up (step 4) is bounded — the floor check above
        //       no longer returns early, so a snap that would push the head into
        //       a ceiling is immediately corrected here.
        //
        // When the ceiling clamp fires it:
        //   * Reduces `new_z` to `head_cell - HEAD_ROOM`.
        //   * Zeros `motion.vz` (player cannot accelerate through the ceiling).
        //   * Does NOT touch `motion.grounded` — the grounded state is resolved
        //     in step 7 from the final `self.pos.z`.  (The player may be squished
        //     between a ramp and a ceiling; step 7 will correctly report not-
        //     grounded in that case because the feet won't be close to the ramp
        //     surface at the ceiling-limited height.)
        //
        // Shape-aware: a RampZ or WedgeXY cell above the head is only a hit if
        // the head's world-point is actually inside the solid part of that shape.
        // -----------------------------------------------------------------------
        {
            let head_z = new_z + PLAYER_HEAD_ROOM;
            let head_cell = head_z.floor() as i32;
            // Treat out-of-bounds as solid (map boundary = ceiling).
            let ceiling_hit = if !map.in_bounds(cx, cy, head_cell) {
                true
            } else {
                let hk = map.physics_cell_at(cx, cy, head_cell);
                if !hk.is_solid() {
                    false
                } else {
                    // Shape-aware: check if the head world-point is inside the solid.
                    let hs = map.shape_at(cx, cy, head_cell);
                    if hs == Shape::Full {
                        true
                    } else {
                        let local = Vec3::new(
                            self.pos.x - cx as f32,
                            self.pos.y - cy as f32,
                            head_z - head_cell as f32,
                        );
                        shape_solid_contains(hs, local)
                    }
                }
            };
            if ceiling_hit {
                // Clamp eye to just below the blocking cell.
                new_z = head_cell as f32 - PLAYER_HEAD_ROOM;
                motion.vz = 0.0;
                // If `motion.grounded` was set true by ramp-snap or floor-snap,
                // the ceiling blocked the resulting upward move.  The player
                // cannot stand on the surface at full height; clear grounded so
                // they are not "planted" into the ceiling.  Step 7 will re-check
                // the grounded state from the ceiling-clamped position.
                motion.grounded = false;
            }
        }

        // 6. Hard-clamp to map bounds and zero velocity when clamped.
        //
        // `boundary_floored` used to be an early `return` here (grounded at
        // the map's floor boundary needs no further check) — it is now a
        // flag instead, so step 8's coyote/jump-buffer bookkeeping (which
        // must run on EVERY exit path, not just the common one) is never
        // skipped.
        let boundary_floored = new_z < eye_lo;
        if boundary_floored {
            self.pos.z = eye_lo;
            motion.vz = 0.0;
            motion.grounded = true;
        } else if new_z > eye_hi {
            self.pos.z = eye_hi;
            motion.vz = 0.0;
            // No further action here — grounded check below still needed
            // (could be resting against a ceiling, not the floor).
        } else {
            self.pos.z = new_z;
        }

        // 7. Determine grounded state from the final position. Skipped when
        //    step 6 already floored at the map boundary — that state is
        //    unconditionally grounded.
        //    Feet are at `pos.z - EYE_HEIGHT`. The supporting surface is the TOP
        //    of the cell below (for Full cells, fc+1.0) or the slope surface
        //    (for RampZ cells, fc + slope_z_at_xy).
        //
        //    We check TWO candidate floor cells:
        //    (a) `fc_below = floor(feet − ε)` — the normal case: feet slightly above
        //        the floor of z=0 land in cell z=0.
        //    (b) `fc_at = floor(feet)` — the boundary case: feet exactly at z=N
        //        (e.g. resting on a RampZ cell whose top surface is local_z=0 at
        //        the cell seam) land IN cell z=N, not in z=N-1.  This covers NegX/
        //        NegY ramp seams where local_z≈0 at the high-x or high-y cell edge.
        //
        //    `motion.grounded` set by ramp-snap (step 3) or flat floor (step 4)
        //    may have been cleared by the ceiling clamp (step 5); always
        //    recompute from the committed `self.pos.z` for correctness.
        if !boundary_floored {
            let feet_after = self.pos.z - PLAYER_EYE_HEIGHT;
            let lx = self.pos.x - cx as f32;
            let ly = self.pos.y - cy as f32;

            let check_grounded_at = |fc: i32| -> bool {
                if !map.in_bounds(cx, cy, fc) {
                    return false;
                }
                let fk = map.physics_cell_at(cx, cy, fc);
                if fk != CellKind::Floor && !fk.is_solid() {
                    return false;
                }
                let floor_shape = map.shape_at(cx, cy, fc);
                let surface_local_z = shape_surface_z(floor_shape, lx, ly);
                let surface_world_z = fc as f32 + surface_local_z;
                // Within ~5 mm of the surface counts as grounded.
                (feet_after - surface_world_z).abs() < 0.01
            };

            let fc_below = (feet_after - 1e-4).floor() as i32;
            let fc_at = feet_after.floor() as i32;
            motion.grounded = check_grounded_at(fc_below) || check_grounded_at(fc_at);
        }

        // -----------------------------------------------------------------------
        // 8. Coyote-time / jump-buffer bookkeeping (Swaggerfall-study item 5).
        //
        // Runs on EVERY exit path (including `boundary_floored`), using the
        // FINAL `motion.grounded` this call committed to:
        //
        //   * Grounded (still resting, or just landed this frame): reset
        //     `since_grounded` to 0.0 (coyote window re-arms) and
        //     `jumped_airborne` to false (a fresh airborne phase can jump
        //     again next time it leaves the ground). If a jump press was
        //     buffered within `JUMP_BUFFER_TIME`, fire it NOW — the whole
        //     point of buffering is that landing itself completes the jump
        //     the player already asked for, no re-press required.
        //   * Airborne: age `since_grounded` by `dt` (the coyote window
        //     counts down from the moment the player left the ground, not
        //     from the moment they last pressed jump) and age
        //     `jump_buffered`, dropping it once it exceeds the buffer
        //     window so a stale press does not fire on a much later landing.
        // -----------------------------------------------------------------------
        if motion.grounded {
            let should_fire_buffered = motion
                .jump_buffered
                .is_some_and(|age| age < JUMP_BUFFER_TIME);
            motion.since_grounded = 0.0;
            motion.jumped_airborne = false;
            motion.jump_buffered = None;
            if should_fire_buffered {
                motion.vz = JUMP_IMPULSE;
                motion.grounded = false;
                motion.jumped_airborne = true;
                // `since_grounded` stays at 0.0; the very next `step_gravity`
                // call (airborne branch below, next frame) starts counting
                // it up from there — exactly as if the player had just left
                // solid ground this instant, which for jump-arc purposes
                // they effectively have.
            }
        } else {
            motion.since_grounded += dt;
            if let Some(age) = motion.jump_buffered.as_mut() {
                *age += dt;
                if *age >= JUMP_BUFFER_TIME {
                    motion.jump_buffered = None;
                }
            }
        }
    }

    /// Apply a jump impulse if the player can jump right now — grounded, or
    /// within the coyote-time grace window after leaving a ledge — and
    /// otherwise buffer the press so it fires automatically on landing
    /// (Swaggerfall-study item 5: coyote time + jump buffer).
    ///
    /// Three outcomes:
    /// * **Grounded** — fires immediately: `motion.vz = JUMP_IMPULSE`,
    ///   `motion.grounded = false`, `motion.jumped_airborne = true`. Returns
    ///   `true`.
    /// * **Airborne, within [`COYOTE_TIME`] of leaving the ground, and no
    ///   jump already used this airborne phase** — fires immediately exactly
    ///   like the grounded case (same impulse, same arc: only the ALLOWED
    ///   press timing widens, never the trajectory). Returns `true`.
    /// * **Otherwise** (airborne beyond the coyote window, or a jump was
    ///   already used this airborne phase — no double jump) — the press
    ///   fires nothing now but is buffered: `motion.jump_buffered =
    ///   Some(0.0)`. [`CameraPose::step_gravity`] fires it automatically the
    ///   instant the player next lands, provided that happens within
    ///   [`JUMP_BUFFER_TIME`]. Returns `false`.
    ///
    /// The caller should call `try_jump` unconditionally on every press edge
    /// (no external grounded gate needed — this function IS the gate) and
    /// call `step_gravity` every frame regardless of whether a jump is in
    /// progress.
    pub fn try_jump(motion: &mut PlayerMotion) -> bool {
        if motion.grounded {
            motion.vz = JUMP_IMPULSE;
            motion.grounded = false;
            motion.jumped_airborne = true;
            motion.since_grounded = 0.0;
            motion.jump_buffered = None;
            true
        } else if motion.since_grounded < COYOTE_TIME && !motion.jumped_airborne {
            // Coyote time: left the ledge less than COYOTE_TIME ago and has
            // not already spent this airborne phase's one jump.
            motion.vz = JUMP_IMPULSE;
            motion.jumped_airborne = true;
            motion.jump_buffered = None;
            true
        } else {
            // Can't fire now (too long airborne, or already jumped this
            // phase). Remember the press — `step_gravity` fires it on
            // landing if that happens within JUMP_BUFFER_TIME. A fresh
            // press always refreshes the buffer window (most recent press
            // wins), matching mashing-the-button expectations.
            motion.jump_buffered = Some(0.0);
            false
        }
    }
}

/// Is a player-radius probe point at `p` (world/cell units) blocked by
/// collision geometry — the exact per-probe predicate
/// [`CameraPose::move_with_collision`] uses (shape-aware, walk-through-aware,
/// out-of-bounds-is-solid), exposed so a caller that needs to know "can the
/// player occupy this point" (e.g. `re_lvlcheck`'s `no_stuck` check) asks the
/// SAME rule instead of re-approximating it. `self` is unused inside
/// `solid_or_oob`, so a throwaway pose is a legitimate way to call it.
#[must_use]
pub fn point_blocked(map: &Map, p: Vec3) -> bool {
    let probe = CellProbe {
        cx: p.x.floor() as i32,
        cy: p.y.floor() as i32,
        cz: p.z.floor() as i32,
        point: p,
    };
    CameraPose {
        pos: p,
        yaw: 0.0,
        pitch: 0.0,
    }
    .solid_or_oob(map, probe)
}

// ---------------------------------------------------------------------------
// Runtime unstuck recovery (set-dressing follow-up, walk-through props):
// deterministically find somewhere safe to stand after a load or a
// prop-power flip left the player embedded in a collider.
// ---------------------------------------------------------------------------

/// How many gravity-only iterations [`nearest_free_standable_cell`] gives a
/// candidate cell to prove it settles onto solid ground. Mirrors the
/// `SETTLE_ITERS` budget `re_lvlcheck::no_stuck` and the campaign integration
/// tests already use for the same "does this position ground out" question.
const UNSTUCK_SETTLE_ITERS: usize = 90;

/// Is `(x, y, z)` (an air cell — the SAME convention `[player] spawn`/
/// [`CameraPose::spawn_at`] use: the cell the player's FEET occupy, not the
/// floor beneath it) a safe place to stand: the eye-height radius probe is
/// not blocked, AND settling there under gravity grounds out (so it is not a
/// mid-air drop over a pit). The exact rule `move_with_collision`'s probe and
/// `step_gravity`'s ground check already use — never re-approximated.
fn cell_is_safe_standing_spot(map: &Map, x: i32, y: i32, z: i32) -> bool {
    if x < 0 || y < 0 || z < 0 {
        return false;
    }
    let eye = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5);
    let r = PLAYER_RADIUS;
    let blocked = [(r, 0.0), (-r, 0.0), (0.0, r), (0.0, -r)]
        .iter()
        .any(|&(ox, oy)| point_blocked(map, Vec3::new(eye.x + ox, eye.y + oy, eye.z)));
    if blocked {
        return false;
    }
    let mut cam = CameraPose {
        pos: eye,
        yaw: 0.0,
        pitch: 0.0,
    };
    let mut motion = PlayerMotion::default();
    for _ in 0..UNSTUCK_SETTLE_ITERS {
        cam.step_gravity(&mut motion, 1.0 / 60.0, map);
        if motion.grounded {
            break;
        }
    }
    motion.grounded
}

/// Deterministically search outward from `near` for the nearest safe-to-stand
/// cell (`cell_is_safe_standing_spot`, private below).
///
/// The runtime recovery for "the player is embedded in a collider after a
/// load or a prop-power flip" (set-dressing follow-up; marking power-reactive
/// cells walk-through keeps this from firing today, but the net stays for a
/// future mechanism that adds genuinely solid power-reactive geometry — R4
/// whole-body swaps are explicitly deferred, not yet delivered).
///
/// Search order: increasing Chebyshev ring around `near`'s cell, then a fixed
/// `(dz, dy, dx)` scan within each ring — plain nested integer loops, never a
/// hash-ordered collection, so the result is identical on every platform and
/// every run given the same map and starting point.
///
/// `None` if nothing within `max_radius` cells is safe — the caller should
/// treat that as unrecoverable and log rather than silently teleport
/// somewhere unrelated.
#[must_use]
pub fn nearest_free_standable_cell(map: &Map, near: Vec3, max_radius: i32) -> Option<Vec3> {
    let (cx, cy, cz) = (
        near.x.floor() as i32,
        near.y.floor() as i32,
        near.z.floor() as i32,
    );
    for r in 0..=max_radius {
        for dz in -r..=r {
            for dy in -r..=r {
                for dx in -r..=r {
                    if dx.abs().max(dy.abs()).max(dz.abs()) != r {
                        continue; // only this ring's shell — smaller rings already tried
                    }
                    let (x, y, z) = (cx + dx, cy + dy, cz + dz);
                    if cell_is_safe_standing_spot(map, x, y, z) {
                        return Some(Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5));
                    }
                }
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Door-entry predicate (level-completion gate).
// ---------------------------------------------------------------------------

/// Chebyshev (cell) radius within which the player is considered "at the door".
///
/// `0` would require the player to stand on the *exact* door cell; the door
/// itself is usually a wall opening the player can never fully occupy, so a
/// radius of `1` means "in the door cell OR any of the 26 cells touching it"
/// (including diagonals — this is a Chebyshev / `max`-norm ball, not Manhattan).
/// One cell is the natural "you reached it" threshold for a cell-quantized
/// grid where the player's footprint is a single cell.
pub const DOOR_ENTRY_RADIUS: i32 = 1;

/// Pure predicate: is the player's camera close enough to the door cell to be
/// considered "entering" it?
///
/// `cam_pos` is the camera world position (`CameraPose::pos`, a `glam::Vec3` in
/// cell units — a pose centred in cell `(x, y, z)` has `pos == (x+0.5, …)`).
/// `(door_x, door_y, door_z)` are the integer door cell coordinates (take them
/// as plain `i32` so this stays free of any `re_level` / `re_puzzle`
/// dependency — the caller passes `re_level::Door { x, y, z }`).
///
/// The camera position is floored to its containing cell and compared to the
/// door cell under the Chebyshev (max-axis) distance. The player is "at the
/// door" iff every axis is within [`DOOR_ENTRY_RADIUS`] cells:
///
/// ```text
/// max(|cx - door_x|, |cy - door_y|, |cz - door_z|) <= DOOR_ENTRY_RADIUS
/// ```
///
/// Floored, not rounded: a camera at `pos.x = 7.9` is still in cell `7`. This
/// matches how the rest of `re_world` converts a world position to a cell
/// (`floor`), so the predicate agrees with collision / cell sampling.
///
/// Edge cases:
/// * Exactly on the door cell ⇒ `true` (distance 0).
/// * Diagonally adjacent (all three axes off by 1) ⇒ `true` at the default
///   radius (Chebyshev distance 1, not √3).
/// * Non-finite `cam_pos` (NaN/∞ from a corrupt pose) ⇒ `false` (never grant a
///   win on garbage input rather than panicking on the cast).
#[inline]
pub fn at_door(cam_pos: Vec3, door_x: i32, door_y: i32, door_z: i32) -> bool {
    at_door_within(cam_pos, door_x, door_y, door_z, DOOR_ENTRY_RADIUS)
}

/// [`at_door`] with an explicit Chebyshev radius (in cells). `radius < 0` is
/// treated as `0` (an empty ball that only the exact door cell satisfies).
#[inline]
pub fn at_door_within(cam_pos: Vec3, door_x: i32, door_y: i32, door_z: i32, radius: i32) -> bool {
    if !cam_pos.is_finite() {
        return false;
    }
    let radius = radius.max(0);
    // Floor each axis to its containing cell (matches floor-based cell sampling).
    let cx = cam_pos.x.floor() as i32;
    let cy = cam_pos.y.floor() as i32;
    let cz = cam_pos.z.floor() as i32;
    (cx - door_x).abs() <= radius && (cy - door_y).abs() <= radius && (cz - door_z).abs() <= radius
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Door-entry predicate (at_door) tests.
    // -----------------------------------------------------------------------

    /// A camera centred in the door cell is "at the door".
    #[test]
    fn at_door_exact_cell() {
        // Pose centred in cell (7, 4, 1) -> pos (7.5, 4.5, 1.5).
        let cam = Vec3::new(7.5, 4.5, 1.5);
        assert!(at_door(cam, 7, 4, 1));
    }

    /// Anywhere inside the door cell (not just the centre) floors to the cell.
    #[test]
    fn at_door_anywhere_in_cell() {
        assert!(at_door(Vec3::new(7.01, 4.99, 1.5), 7, 4, 1));
        assert!(at_door(Vec3::new(7.99, 4.01, 1.01), 7, 4, 1));
    }

    /// A face-adjacent cell (one axis off by 1) is within the default radius.
    #[test]
    fn at_door_face_adjacent() {
        let door = (7, 4, 1);
        // +x neighbour, cell (8, 4, 1).
        assert!(at_door(Vec3::new(8.5, 4.5, 1.5), door.0, door.1, door.2));
        // -y neighbour, cell (7, 3, 1).
        assert!(at_door(Vec3::new(7.5, 3.5, 1.5), door.0, door.1, door.2));
        // +z neighbour, cell (7, 4, 2).
        assert!(at_door(Vec3::new(7.5, 4.5, 2.5), door.0, door.1, door.2));
    }

    /// A fully diagonal neighbour (all three axes off by 1) is still within the
    /// Chebyshev ball of radius 1 — diagonal entry counts.
    #[test]
    fn at_door_diagonal_adjacent() {
        // Cell (8, 5, 2) is the +x/+y/+z corner neighbour of (7, 4, 1).
        assert!(at_door(Vec3::new(8.5, 5.5, 2.5), 7, 4, 1));
        // Opposite corner (6, 3, 0).
        assert!(at_door(Vec3::new(6.5, 3.5, 0.5), 7, 4, 1));
    }

    /// Two cells away on any single axis is NOT at the door.
    #[test]
    fn at_door_far_rejected() {
        let door = (7, 4, 1);
        // Two cells in +x: cell (9, 4, 1).
        assert!(!at_door(Vec3::new(9.5, 4.5, 1.5), door.0, door.1, door.2));
        // Two cells in -y: cell (7, 2, 1).
        assert!(!at_door(Vec3::new(7.5, 2.5, 1.5), door.0, door.1, door.2));
        // Far away.
        assert!(!at_door(Vec3::new(20.5, 30.5, 5.5), door.0, door.1, door.2));
    }

    /// Floor (not round) semantics: pos.x in [door, door+1) is the door cell;
    /// pos.x just below `door` floors into the previous cell.
    #[test]
    fn at_door_floors_not_rounds() {
        // 7.0 floors to cell 7 == door cell.
        assert!(at_door(Vec3::new(7.0, 4.0, 1.0), 7, 4, 1));
        // 6.9 floors to cell 6 (adjacent, still within radius 1).
        assert!(at_door(Vec3::new(6.9, 4.5, 1.5), 7, 4, 1));
        // 5.9 floors to cell 5 (two away, rejected).
        assert!(!at_door(Vec3::new(5.9, 4.5, 1.5), 7, 4, 1));
    }

    /// `at_door_within(_, 0)` requires the exact door cell.
    #[test]
    fn at_door_radius_zero_exact_only() {
        assert!(at_door_within(Vec3::new(7.5, 4.5, 1.5), 7, 4, 1, 0));
        assert!(!at_door_within(Vec3::new(8.5, 4.5, 1.5), 7, 4, 1, 0));
        // A negative radius is clamped to 0, not a panic / empty result.
        assert!(at_door_within(Vec3::new(7.5, 4.5, 1.5), 7, 4, 1, -3));
        assert!(!at_door_within(Vec3::new(8.5, 4.5, 1.5), 7, 4, 1, -3));
    }

    /// A larger radius widens the entry ball.
    #[test]
    fn at_door_radius_two() {
        // Cell (9, 4, 1) is two away — outside r=1 but inside r=2.
        assert!(!at_door_within(Vec3::new(9.5, 4.5, 1.5), 7, 4, 1, 1));
        assert!(at_door_within(Vec3::new(9.5, 4.5, 1.5), 7, 4, 1, 2));
    }

    /// Non-finite camera positions never grant entry (no panic on the cast).
    #[test]
    fn at_door_non_finite_rejected() {
        assert!(!at_door(Vec3::new(f32::NAN, 4.5, 1.5), 7, 4, 1));
        assert!(!at_door(Vec3::new(7.5, f32::INFINITY, 1.5), 7, 4, 1));
        assert!(!at_door(Vec3::new(7.5, 4.5, f32::NEG_INFINITY), 7, 4, 1));
    }

    /// Negative door coordinates work (no unsigned assumptions).
    #[test]
    fn at_door_negative_coords() {
        assert!(at_door(Vec3::new(-2.5, -1.5, 0.5), -3, -2, 0));
        assert!(!at_door(Vec3::new(-0.5, -1.5, 0.5), -3, -2, 0));
    }

    // -----------------------------------------------------------------------
    // Trigger-field storage + `triggers_containing` query.
    // -----------------------------------------------------------------------

    fn trig(min: (i32, i32, i32), max: (i32, i32, i32), label: &str) -> re_core::Trigger {
        re_core::Trigger {
            region: re_core::Region::Box { min, max },
            action: re_core::TriggerAction::event(label),
            once: true,
        }
    }

    #[test]
    fn triggers_containing_point_inside_outside_edge() {
        let mut m = Map::new(8, 1, 8);
        m.set_resolved_triggers(vec![trig((2, 0, 2), (4, 0, 5), "a")]);
        // Inside (camera centred in cell 3,0,3).
        assert_eq!(m.triggers_containing(Vec3::new(3.5, 0.5, 3.5)).count(), 1);
        // Inclusive edge cell (2,0,2).
        assert_eq!(m.triggers_containing(Vec3::new(2.1, 0.9, 2.9)).count(), 1);
        // Just outside on +x.
        assert_eq!(m.triggers_containing(Vec3::new(5.5, 0.5, 3.5)).count(), 0);
        // Floors, not rounds: pos.x = 1.9 is cell 1 (outside).
        assert_eq!(m.triggers_containing(Vec3::new(1.9, 0.5, 3.5)).count(), 0);
    }

    #[test]
    fn triggers_containing_multiple_overlap() {
        let mut m = Map::new(8, 1, 8);
        m.set_resolved_triggers(vec![
            trig((0, 0, 0), (5, 0, 5), "wide"),
            trig((3, 0, 3), (4, 0, 4), "inner"),
            trig((6, 0, 6), (7, 0, 7), "far"),
        ]);
        // (3,0,3) is in BOTH "wide" and "inner".
        let hits: Vec<_> = m
            .triggers_containing(Vec3::new(3.5, 0.5, 3.5))
            .map(|t| match &t.action {
                re_core::TriggerAction::Event { label } => label.clone(),
                _ => String::new(),
            })
            .collect();
        assert_eq!(hits.len(), 2);
        assert!(hits.contains(&"wide".to_string()));
        assert!(hits.contains(&"inner".to_string()));
        // "far" never overlaps the others.
        assert_eq!(m.triggers_containing(Vec3::new(6.5, 0.5, 6.5)).count(), 1);
    }

    #[test]
    fn triggers_containing_non_finite_rejected() {
        let mut m = Map::new(8, 1, 8);
        m.set_resolved_triggers(vec![trig((0, 0, 0), (7, 0, 7), "a")]);
        assert_eq!(
            m.triggers_containing(Vec3::new(f32::NAN, 0.5, 3.5)).count(),
            0
        );
        assert_eq!(
            m.triggers_containing(Vec3::new(3.5, f32::INFINITY, 3.5))
                .count(),
            0
        );
    }

    #[test]
    fn triggers_default_empty() {
        let m = Map::new(4, 1, 4);
        assert!(m.resolved_triggers().is_empty());
        assert_eq!(m.triggers_containing(Vec3::new(1.5, 0.5, 1.5)).count(), 0);
    }

    // -----------------------------------------------------------------------
    // W1 brickmap-backed Map tests.
    // -----------------------------------------------------------------------

    #[test]
    fn brickmap_backed_map_reads_and_writes() {
        let mut m = Map::new_brickmap(64, 64, 8);
        assert_eq!(m.size(), (64, 64, 8));
        m.set_cell(33, 40, 2, CellKind::WallSolid, 9);
        assert_eq!(m.cell_at(33, 40, 2), CellKind::WallSolid);
        assert_eq!(m.side_data_at(33, 40, 2), 9);
        m.set(5, 5, 0, CellKind::Hologram);
        assert!(m.has_holograms()); // Map's holo cache maintained via set_cell's bool
    }

    // -----------------------------------------------------------------------
    // W3a generate_brickmap tests.
    // -----------------------------------------------------------------------

    #[test]
    fn procedural_map_edits_persist_over_base() {
        let mut m = Map::procedural(512, 512, 64, 7);
        assert_eq!(m.size(), (512, 512, 64));
        let (x, y, z) = (300, 220, 6);
        let base = crate::gen::procedural_base(7, x, y, z);
        assert_eq!(m.cell_at(x, y, z), base);
        m.set(x as usize, y as usize, z as usize, CellKind::Portal);
        assert_eq!(m.cell_at(x, y, z), CellKind::Portal); // edit wins
    }

    #[test]
    fn generated_brickmap_is_big_and_sparse() {
        // A 256x256x32 world: generated, brickmap-backed, reads match procedural_base.
        let m = Map::generate_brickmap(256, 256, 32, 7);
        assert_eq!(m.size(), (256, 256, 32));
        // Spot-check a few cells equal the generator (the fill is faithful).
        for &(x, y, z) in &[(10, 10, 2), (200, 50, 5), (255, 255, 31)] {
            assert_eq!(m.cell_at(x, y, z), crate::gen::procedural_base(7, x, y, z));
        }
        // Some solids exist (world isn't empty).
        assert!(m.count(CellKind::WallSolid) > 0);
    }

    #[test]
    fn cell_kind_u8_roundtrip_all_variants() {
        // Every CellKind variant must survive a to_u8 → from_u8 round-trip.
        let variants = [
            CellKind::Empty,
            CellKind::WallSolid,
            CellKind::WallTrace,
            CellKind::WallMatrix,
            CellKind::Floor,
            CellKind::Ceiling,
            CellKind::Object,
            CellKind::Hologram,
            CellKind::Portal,
        ];
        for &v in &variants {
            assert_eq!(
                CellKind::from_u8(v.to_u8()),
                v,
                "round-trip failed for {v:?}"
            );
        }
        // Unknown byte → Empty (safe fallback, not a panic).
        assert_eq!(CellKind::from_u8(255), CellKind::Empty);
    }

    #[test]
    fn test_room_has_walls() {
        let m = test_room();
        assert_eq!(m.size(), (6, 6, 4));
        // Perimeter at z=1 must be wall; an interior air cell must be empty.
        assert_eq!(m.cell_at(0, 0, 1), CellKind::WallSolid);
        assert_eq!(m.cell_at(5, 5, 1), CellKind::WallSolid);
        assert_eq!(m.cell_at(3, 3, 1), CellKind::Empty);
        // Pillar.
        assert_eq!(m.cell_at(2, 2, 1), CellKind::WallSolid);
        // Floor at z=0 interior, ceiling at z=3 interior.
        assert_eq!(m.cell_at(3, 3, 0), CellKind::Floor);
        assert_eq!(m.cell_at(3, 3, 3), CellKind::Ceiling);
        assert!(m.count(CellKind::WallSolid) > 0);
    }

    #[test]
    fn out_of_bounds_is_empty() {
        let m = test_room();
        assert_eq!(m.cell_at(-1, 0, 0), CellKind::Empty);
        assert_eq!(m.cell_at(99, 0, 0), CellKind::Empty);
    }

    #[test]
    fn map_materials_default_empty_and_settable() {
        // Back-compat: a fresh / stamped map carries NO authored materials, so
        // the renderer falls back to its built-ins.
        let mut m = test_room();
        assert!(
            m.materials().is_empty(),
            "default map must carry no materials"
        );
        // An authored set round-trips through the accessor.
        m.set_materials(vec![
            MaterialDef::flat("floor_neon", 0x00FF88, 0x001a12),
            MaterialDef {
                name: "wall_pulse".into(),
                fg: 0x66CCFF,
                bg: 0x041018,
                glyph: Some(u32::from('#')),
                recipe: Some(MaterialRecipe::Brick {
                    aspect: 2.0,
                    mortar_w: 0.16,
                    grain: 0.6,
                }),
                anim: MaterialAnim {
                    kind: MaterialAnimKind::Pulse,
                    speed: 0.5,
                    amplitude: 0.2,
                },
                rank: 0,
                reflective: false,
                clarity: MaterialDef::DEFAULT_CLARITY,
                fallback: false,
                detail: false,
            },
        ]);
        assert_eq!(m.materials().len(), 2);
        assert_eq!(m.materials()[0].name, "floor_neon");
        assert_eq!(m.materials()[1].glyph, Some(u32::from('#')));
    }

    #[test]
    fn material_def_mirror_fields_default_and_settable() {
        // Back-compat: a flat / fallback material is NON-reflective by default, so
        // an ordinary authored material never accidentally becomes a mirror.
        let flat = MaterialDef::flat("plain", 0x808080, 0x101010);
        assert!(
            !flat.reflective,
            "a flat material must not be reflective by default"
        );
        assert!(
            !MaterialDef::fallback().reflective,
            "a fallback slot is never a mirror"
        );
        assert_eq!(flat.clarity, MaterialDef::DEFAULT_CLARITY);

        // A mirror is authorable as plain data: reflective + a clarity in [0,1].
        let mirror = MaterialDef {
            reflective: true,
            clarity: 0.5,
            ..MaterialDef::flat("mirror", 0xC8D2FF, 0x05060A)
        };
        assert!(mirror.reflective);
        assert_eq!(mirror.clarity, 0.5);
    }

    #[test]
    fn map_atmosphere_default_none_and_settable() {
        // Back-compat: a fresh / stamped map carries NO atmosphere, so the
        // renderer falls back to its built-in fog constants.
        let mut m = test_room();
        assert!(
            m.atmosphere().is_none(),
            "default map must carry no atmosphere"
        );
        assert_eq!(m.atmosphere().fog_color, None);
        // An authored atmosphere round-trips through the accessor.
        m.set_atmosphere(Atmosphere {
            fog_color: Some(0x00FF_00FF),
            fog_near: Some(1.5),
            fog_far: Some(12.0),
            matrix: Some(0.7),
            ..Default::default()
        });
        let a = m.atmosphere();
        assert!(!a.is_none());
        assert_eq!(a.fog_color, Some(0x00FF_00FF));
        assert_eq!(a.fog_near, Some(1.5));
        assert_eq!(a.fog_far, Some(12.0));
        assert_eq!(a.matrix, Some(0.7));
        assert!(!a.has_lighting(), "fog-only atmosphere authors no lighting");
    }

    /// The new §1.3 lighting fields: default-off, and `has_lighting` flips only
    /// when a light/key/ambient override is actually authored.
    #[test]
    fn atmosphere_lighting_fields_default_off() {
        let a = Atmosphere::default();
        assert!(a.is_none());
        assert!(!a.has_lighting());
        assert!(a.lights.is_empty());
        assert_eq!(a.postfx, None);

        let lit = Atmosphere {
            lights: vec![PointLight {
                pos: Vec3::new(2.0, 3.0, 1.0),
                color: 0x00FF_B060,
                radius: 6.0,
                intensity: 1.25,
            }],
            ambient: Some(0.12),
            fog_height: Some(HeightFog {
                base: 0.0,
                falloff: 0.6,
            }),
            postfx: Some("clean".to_string()),
            ..Default::default()
        };
        assert!(!lit.is_none());
        assert!(lit.has_lighting());
        assert_eq!(lit.lights[0].radius, 6.0);
        assert_eq!(lit.fog_height.unwrap().falloff, 0.6);

        // Fog-height alone is atmosphere but NOT lighting.
        let fogonly = Atmosphere {
            fog_height: Some(HeightFog {
                base: 1.0,
                falloff: 0.2,
            }),
            ..Default::default()
        };
        assert!(!fogonly.is_none());
        assert!(!fogonly.has_lighting());
    }

    #[test]
    fn ambient_mapping() {
        assert_eq!(ambient_to_wall_kind(""), CellKind::WallTrace);
        assert_eq!(ambient_to_wall_kind("trace-pattern"), CellKind::WallTrace);
        assert_eq!(ambient_to_wall_kind("matrix"), CellKind::WallMatrix);
        assert_eq!(ambient_to_wall_kind("solid"), CellKind::WallSolid);
    }

    #[test]
    fn stamp_geometry_shell_and_extras() {
        // Two objects: one with tag=7 (mimics a game-defined label), one with tag=0
        // (mimics any other game-defined anchor — both are generic Object cells).
        let objects = [
            ObjectCell {
                x: 3,
                y: 3,
                z: 2,
                tag: 7,
            },
            ObjectCell {
                x: 11,
                y: 3,
                z: 1,
                tag: 0,
            },
        ];
        let spec = GeometrySpec {
            size: (12, 8, 4),
            wall_kind: CellKind::WallTrace,
            decals: &[],
            objects: &objects,
        };
        let m = stamp_geometry(&spec);
        assert_eq!(m.size(), (12, 8, 4));
        // Perimeter is wall at every z.
        assert_eq!(m.cell_at(0, 0, 0), CellKind::WallTrace);
        assert_eq!(m.cell_at(0, 0, 3), CellKind::WallTrace);
        assert_eq!(m.cell_at(11, 7, 2), CellKind::WallTrace);
        // Interior floor / ceiling.
        assert_eq!(m.cell_at(5, 4, 0), CellKind::Floor);
        assert_eq!(m.cell_at(5, 4, 3), CellKind::Ceiling);
        // Interior air.
        assert_eq!(m.cell_at(5, 4, 2), CellKind::Empty);
        // First object: kind=Object, side_data=tag.
        assert_eq!(m.cell_at(3, 3, 2), CellKind::Object);
        assert_eq!(m.side_data_at(3, 3, 2), 7);
        // Second object: kind=Object, side_data=0.
        assert_eq!(m.cell_at(11, 3, 1), CellKind::Object);
        assert_eq!(m.side_data_at(11, 3, 1), 0);
    }

    /// The empty-world guard — the same class of check that caught the legacy
    /// blank-map bug: after stamping, the grid must contain real walls.
    #[test]
    fn stamp_geometry_is_not_empty() {
        let spec = GeometrySpec {
            size: (12, 8, 4),
            wall_kind: CellKind::WallTrace,
            decals: &[],
            objects: &[],
        };
        let m = stamp_geometry(&spec);
        let walls = m.count(CellKind::WallTrace);
        assert!(walls > 0, "stamped world must have wall cells, got {walls}");
        // Floor + ceiling slabs present too.
        assert!(m.count(CellKind::Floor) > 0);
        assert!(m.count(CellKind::Ceiling) > 0);
    }

    #[test]
    fn hologram_cell_is_transparent_not_solid() {
        // A hologram cell must NOT be solid (the DDA passes through it), yet must
        // report as a hologram so the caster can collect its crossing.
        assert!(!CellKind::Hologram.is_solid());
        assert!(CellKind::Hologram.is_hologram());
        assert!(!CellKind::Hologram.is_wall());
        // A non-hologram object is unchanged: solid, not a hologram.
        assert!(CellKind::Object.is_solid());
        assert!(!CellKind::Object.is_hologram());
    }

    #[test]
    fn stamp_holos_box_fills_hologram_cells_only_in_air() {
        let spec = GeometrySpec {
            size: (12, 8, 4),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[],
        };
        let mut map = stamp_geometry(&spec);
        let walls_before = map.count(CellKind::WallSolid);
        let floor_before = map.count(CellKind::Floor);
        // A 3x3x2 box overlapping the perimeter wall on one edge: the wall must be
        // preserved (hologram only fills Empty air).
        let holo = HoloObject {
            shape: HoloShape::Box,
            x0: 0,
            y0: 2,
            z0: 1,
            x1: 2,
            y1: 4,
            z1: 2,
            style: 0,
        };
        stamp_holos(&mut map, &[holo]);
        // Walls + floor untouched (additive, non-destructive over solids).
        assert_eq!(map.count(CellKind::WallSolid), walls_before);
        assert_eq!(map.count(CellKind::Floor), floor_before);
        // Hologram cells appeared in the air part of the box, carrying the style tag.
        let holo_count = map.count(CellKind::Hologram);
        assert!(
            holo_count > 0,
            "box must stamp hologram cells, got {holo_count}"
        );
        // An air cell inside the box is now a hologram; a wall cell (x=0) is not.
        assert_eq!(map.cell_at(1, 3, 1), CellKind::Hologram);
        assert_eq!(map.cell_at(0, 3, 1), CellKind::WallSolid);
    }

    #[test]
    fn stamp_holos_shapes_taper() {
        // A pyramid's apex layer is smaller than its base layer; a sphere is
        // smaller than the enclosing box of equal extent.
        let spec = GeometrySpec {
            size: (16, 16, 8),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[],
        };
        let pyr = HoloObject {
            shape: HoloShape::Pyramid,
            x0: 4,
            y0: 4,
            z0: 1,
            x1: 10,
            y1: 10,
            z1: 4,
            style: 1,
        };
        let mut mp = stamp_geometry(&spec);
        stamp_holos(&mut mp, &[pyr]);
        let pyr_count = mp.count(CellKind::Hologram);
        // Compare against the full box of the same extent.
        let boxo = HoloObject {
            shape: HoloShape::Box,
            ..pyr
        };
        let mut mb = stamp_geometry(&spec);
        stamp_holos(&mut mb, &[boxo]);
        let box_count = mb.count(CellKind::Hologram);
        assert!(pyr_count > 0 && box_count > 0);
        assert!(
            pyr_count < box_count,
            "pyramid {pyr_count} must be smaller than box {box_count}"
        );

        let sph = HoloObject {
            shape: HoloShape::Sphere,
            ..boxo
        };
        let mut ms = stamp_geometry(&spec);
        stamp_holos(&mut ms, &[sph]);
        let sph_count = ms.count(CellKind::Hologram);
        assert!(
            sph_count > 0 && sph_count < box_count,
            "sphere {sph_count} must be smaller than box {box_count}"
        );
    }

    #[test]
    fn no_holos_is_unchanged() {
        // A map stamped with an empty holo list is byte-identical to one stamped
        // without — the shipped-level back-compat guarantee.
        let spec = GeometrySpec {
            size: (10, 10, 4),
            wall_kind: CellKind::WallTrace,
            decals: &[],
            objects: &[],
        };
        let base = stamp_geometry(&spec);
        let mut withcall = stamp_geometry(&spec);
        stamp_holos(&mut withcall, &[]);
        assert_eq!(base.count(CellKind::Hologram), 0);
        assert_eq!(withcall.count(CellKind::Hologram), 0);
        assert_eq!(
            base.count(CellKind::WallTrace),
            withcall.count(CellKind::WallTrace)
        );
    }

    #[test]
    fn apply_height_clamps_to_air_band() {
        // 12x8x4 room: air layers are z=1 and z=2 (floor z=0, ceiling z=3).
        let spec = GeometrySpec {
            size: (12, 8, 4),
            wall_kind: CellKind::WallTrace,
            decals: &[],
            objects: &[],
        };
        let map = stamp_geometry(&spec);
        let mut cam = CameraPose::spawn_at(8, 3, 1, 360); // pos.z = 1.5

        // Rise into the z=2 layer (the switch layer in level 01).
        assert!(cam.apply_height(1.0, &map));
        assert_eq!(cam.pos.z.floor() as i32, 2);

        // Rising again clamps at the top air layer centre (sz-1.5 = 2.5).
        assert!(!cam.apply_height(1.0, &map));
        assert!((cam.pos.z - 2.5).abs() < 1e-6);

        // Fall back down to the lowest air layer; clamps at 1.5.
        assert!(cam.apply_height(-5.0, &map));
        assert!((cam.pos.z - 1.5).abs() < 1e-6);
        assert!(!cam.apply_height(-1.0, &map));
    }

    #[test]
    fn apply_height_blocked_by_solid_cell() {
        // Pillar at (2,2,1) in the test room; standing on it can't sink in.
        let map = test_room();
        let mut cam = CameraPose {
            pos: Vec3::new(2.5, 2.5, 2.5),
            yaw: 0.0,
            pitch: 0.0,
        };
        // Dropping to z=1 would enter the solid pillar cell → refused.
        assert!(!cam.apply_height(-1.0, &map));
        assert!((cam.pos.z - 2.5).abs() < 1e-6);
    }

    #[test]
    fn camera_spawn_facing() {
        // facing -x (360) → yaw = 360/480 * TAU = 0.75*TAU = 1.5π.
        let cam = CameraPose::spawn_at(8, 3, 1, 360);
        assert!((cam.pos.x - 8.5).abs() < 1e-6);
        assert!((cam.pos.z - 1.5).abs() < 1e-6);
        let (dx, dy) = cam.dir();
        // yaw=1.5π → sin=-1, cos≈0: facing -x (west).
        assert!(dx < -0.9, "dx={dx}");
        assert!(dy.abs() < 1e-3, "dy={dy}");
    }

    // -----------------------------------------------------------------------
    // Horizontal collision tests.
    // -----------------------------------------------------------------------

    /// Helper: open 5×5×4 room (perimeter walls, floor z=0, ceiling z=3, air
    /// at z=1..=2). The inner open area is cells (1..=3, 1..=3).
    fn open_room() -> Map {
        let spec = GeometrySpec {
            size: (5, 5, 4),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[],
        };
        stamp_geometry(&spec)
    }

    /// Standard test pose: standing in the middle of the open room, eye at z=1.5.
    fn mid_pose() -> CameraPose {
        CameraPose {
            pos: Vec3::new(2.5, 2.5, 1.5),
            yaw: 0.0,
            pitch: 0.0,
        }
    }

    #[test]
    fn collision_move_into_wall_is_blocked() {
        // Moving +x far enough to hit the east perimeter wall (x=4).
        let map = open_room();
        let mut cam = mid_pose(); // x=2.5
                                  // A large dx that would push us into cell x=4 (perimeter wall).
        let (mx, _my) = cam.move_with_collision(10.0, 0.0, &map, PLAYER_RADIUS);
        // X should be blocked; we must NOT be inside the wall cell.
        assert!(
            !mx || cam.pos.x + PLAYER_RADIUS < 4.0,
            "x={} must not reach the wall at x=4",
            cam.pos.x
        );
        // The blocked position should be just shy of the wall.
        assert!(
            cam.pos.x < 4.0 - PLAYER_RADIUS + 0.01,
            "x={} must be against the wall",
            cam.pos.x
        );
    }

    #[test]
    fn collision_slide_along_wall() {
        // Standing at (1.5, 1.5, 1.5), moving diagonally toward the +x wall
        // (perimeter at x=0 and x=4). Push toward +x hard and +y gently.
        // The x axis should be blocked near the east wall but y should slide.
        let map = open_room();
        // Start close to the east wall.
        let mut cam = CameraPose {
            pos: Vec3::new(3.5, 2.0, 1.5),
            yaw: 0.0,
            pitch: 0.0,
        };
        let y_before = cam.pos.y;
        // Large dx (blocked) + small dy (unblocked).
        let (mx, my) = cam.move_with_collision(5.0, 0.3, &map, PLAYER_RADIUS);
        // x is blocked (can't enter the wall at x=4).
        // y slides freely.
        assert!(my, "y axis should have slid along the wall");
        // x may have moved a little or not at all depending on starting pos,
        // but it must not enter the solid cell.
        assert!(
            cam.pos.x + PLAYER_RADIUS <= 4.0 + 1e-4,
            "x={} must not penetrate the east wall",
            cam.pos.x
        );
        assert!(
            cam.pos.y > y_before,
            "y={} should have moved from {y_before}",
            cam.pos.y
        );
        let _ = mx;
    }

    #[test]
    fn collision_diagonal_corner_no_tunnel() {
        // A pillar at (3,3,1) in the test_room. The player approaches diagonally
        // — neither axis alone is blocked, but radius probing catches the corner.
        let map = test_room(); // pillar at (2,2,1)
                               // Start near the pillar.
        let mut cam = CameraPose {
            pos: Vec3::new(1.5, 1.5, 1.5),
            yaw: 0.0,
            pitch: 0.0,
        };
        // Try to move directly onto the pillar cell.
        cam.move_with_collision(0.7, 0.7, &map, PLAYER_RADIUS);
        // The player must not be inside the pillar cell (2,2,1).
        let cx = cam.pos.x.floor() as i32;
        let cy = cam.pos.y.floor() as i32;
        let cz = cam.pos.z.floor() as i32;
        let kind = map.cell_at(cx, cy, cz);
        assert!(
            !kind.is_solid(),
            "player ended up in solid cell ({cx},{cy},{cz}) kind={kind:?}"
        );
    }

    #[test]
    fn collision_open_space_moves_freely() {
        // In the centre of the open room, small movements should be unblocked.
        let map = open_room();
        let mut cam = mid_pose();
        let (mx, my) = cam.move_with_collision(0.1, 0.1, &map, PLAYER_RADIUS);
        assert!(mx, "x should move freely in open space");
        assert!(my, "y should move freely in open space");
        assert!((cam.pos.x - 2.6).abs() < 1e-5);
        assert!((cam.pos.y - 2.6).abs() < 1e-5);
    }

    #[test]
    fn collision_out_of_bounds_treated_as_solid() {
        // 5×5 room. Starting at x=1.5, pushing -x hard enough to go off-map
        // (past x=0 wall). Should be blocked.
        let map = open_room();
        let mut cam = CameraPose {
            pos: Vec3::new(1.5, 2.5, 1.5),
            yaw: 0.0,
            pitch: 0.0,
        };
        cam.move_with_collision(-10.0, 0.0, &map, PLAYER_RADIUS);
        assert!(cam.pos.x > 0.0, "x={} must not leave the map", cam.pos.x);
    }

    // -----------------------------------------------------------------------
    // Walk-through override (visual-only prop content).
    // -----------------------------------------------------------------------

    /// A `WallSolid` cell blocks movement — until it is marked walk-through, at
    /// which point the player passes straight through it, exactly as if it were
    /// `Empty`. `cell_at` (rendering) reports the real kind throughout.
    #[test]
    fn walkthrough_cell_stops_blocking_horizontal_movement() {
        // Small per-frame steps, like every real caller (the API is not a swept
        // teleport — a single large `dx` only tests the final destination).
        const STEP: f32 = 0.05;
        let advance = |map: &Map, start: Vec3| -> f32 {
            let mut cam = CameraPose {
                pos: start,
                yaw: 0.0,
                pitch: 0.0,
            };
            for _ in 0..60 {
                cam.move_with_collision(STEP, 0.0, map, PLAYER_RADIUS);
            }
            cam.pos.x
        };

        let mut map = open_room();
        // Stamp a solid cell directly in the player's path (open interior cell).
        map.set_cell(3, 2, 1, CellKind::WallSolid, 0);
        let blocked_x = advance(&map, Vec3::new(1.5, 2.5, 1.5));
        assert!(
            blocked_x + PLAYER_RADIUS <= 3.0 + 1e-4,
            "solid cell (3,2,1) must still block: x={blocked_x}"
        );

        map.mark_walkthrough(3, 2, 1, true);
        let free_x = advance(&map, Vec3::new(1.5, 2.5, 1.5));
        assert!(
            free_x > 3.5,
            "walk-through cell (3,2,1) must not block: x={free_x}"
        );

        // Rendering is untouched either way.
        assert_eq!(map.cell_at(3, 2, 1), CellKind::WallSolid);
        assert!(map.cell_at(3, 2, 1).is_solid());
    }

    /// `physics_cell_at` reads a walk-through cell as `Empty`; unmarking restores
    /// the real kind. `Empty` cells are never affected by marking (no-op).
    #[test]
    fn physics_cell_at_and_mark_walkthrough_round_trip() {
        let mut map = open_room();
        map.set_cell(2, 2, 1, CellKind::WallSolid, 7);
        assert_eq!(map.physics_cell_at(2, 2, 1), CellKind::WallSolid);
        assert!(!map.is_walkthrough(2, 2, 1));

        map.mark_walkthrough(2, 2, 1, true);
        assert!(map.is_walkthrough(2, 2, 1));
        assert_eq!(map.physics_cell_at(2, 2, 1), CellKind::Empty);
        // The real kind and side_data are untouched.
        assert_eq!(map.cell_at(2, 2, 1), CellKind::WallSolid);
        assert_eq!(map.side_data_at(2, 2, 1), 7);

        map.mark_walkthrough(2, 2, 1, false);
        assert!(!map.is_walkthrough(2, 2, 1));
        assert_eq!(map.physics_cell_at(2, 2, 1), CellKind::WallSolid);

        // Marking an Empty cell walk-through is a harmless no-op.
        map.mark_walkthrough(1, 1, 1, true);
        assert_eq!(map.physics_cell_at(1, 1, 1), CellKind::Empty);
    }

    /// A walk-through `WallSolid` floor cell is not treated as standable ground
    /// by `step_gravity` — the player falls straight through it onto the real
    /// floor, instead of snapping onto "top of the figure".
    #[test]
    fn walkthrough_cell_is_not_standable_ground() {
        let mut map = open_room();
        // A solid slab spanning the whole interior air layer at z=1 (the floor
        // slab itself is z=0; this extra slab sits on top of it).
        for x in 1..=3 {
            for y in 1..=3 {
                map.set_cell(x, y, 1, CellKind::WallSolid, 0);
            }
        }
        map.mark_walkthrough(2, 2, 1, true);

        // Drop the player from above the marked cell only.
        let mut cam = CameraPose {
            pos: Vec3::new(2.5, 2.5, 3.0),
            yaw: 0.0,
            pitch: 0.0,
        };
        let mut motion = PlayerMotion::default();
        for _ in 0..240 {
            cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
        }
        // The neighbouring (non-walk-through) slab cells DO catch the player —
        // proving the slab is solid ground in general — but the marked column
        // must not: the player keeps falling past z=1 toward the real floor.
        assert!(
            cam.pos.z < 1.5 + PLAYER_EYE_HEIGHT,
            "player must fall through the walk-through slab cell, ended at z={}",
            cam.pos.z
        );
    }

    /// The same walk-through cell, unmarked, DOES catch the player as ordinary
    /// solid ground — confirms the previous test isn't vacuous.
    #[test]
    fn unmarked_matching_cell_is_standable_ground() {
        let mut map = open_room();
        for x in 1..=3 {
            for y in 1..=3 {
                map.set_cell(x, y, 1, CellKind::WallSolid, 0);
            }
        }
        let mut cam = CameraPose {
            pos: Vec3::new(2.5, 2.5, 3.0),
            yaw: 0.0,
            pitch: 0.0,
        };
        let mut motion = PlayerMotion::default();
        for _ in 0..240 {
            cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
        }
        assert!(
            motion.grounded,
            "player must land on the solid slab, ended at z={} grounded={}",
            cam.pos.z, motion.grounded
        );
        assert!(cam.pos.z > 1.5 + PLAYER_EYE_HEIGHT - 1e-3);
    }

    /// [`point_blocked`] agrees with `move_with_collision`'s own predicate: solid
    /// blocks, walk-through does not, out-of-bounds always does.
    #[test]
    fn point_blocked_matches_collision() {
        let mut map = open_room();
        map.set_cell(2, 2, 1, CellKind::WallSolid, 0);
        assert!(point_blocked(&map, Vec3::new(2.5, 2.5, 1.5)));
        map.mark_walkthrough(2, 2, 1, true);
        assert!(!point_blocked(&map, Vec3::new(2.5, 2.5, 1.5)));
        assert!(
            point_blocked(&map, Vec3::new(-1.0, 2.5, 1.5)),
            "out of bounds"
        );
    }

    // -----------------------------------------------------------------------
    // `nearest_free_standable_cell` — runtime unstuck recovery.
    // -----------------------------------------------------------------------

    /// Already-safe: the search finds the starting cell itself at ring 0.
    #[test]
    fn nearest_free_standable_cell_returns_self_when_already_safe() {
        let map = open_room();
        let found = nearest_free_standable_cell(&map, Vec3::new(2.5, 2.5, 1.5), 3);
        assert_eq!(found, Some(Vec3::new(2.5, 2.5, 1.5)));
    }

    /// The starting cell is embedded in a collider: the search steps out to a
    /// neighbour, never returns the blocked cell, and is deterministic (same
    /// map + same starting point => same answer, every call).
    #[test]
    fn nearest_free_standable_cell_steps_around_a_blocked_cell() {
        let mut map = open_room();
        map.set_cell(2, 2, 1, CellKind::WallSolid, 0);

        let a = nearest_free_standable_cell(&map, Vec3::new(2.5, 2.5, 1.5), 3);
        let b = nearest_free_standable_cell(&map, Vec3::new(2.5, 2.5, 1.5), 3);
        assert_eq!(a, b, "must be deterministic across repeated calls");

        let found = a.expect("open_room has standable ground within 3 cells of centre");
        assert!(
            !point_blocked(&map, found),
            "the returned cell must not itself be blocked"
        );
        let key = (
            found.x.floor() as i32,
            found.y.floor() as i32,
            found.z.floor() as i32,
        );
        assert_ne!(
            key,
            (2, 2, 1),
            "must not return the collider it was pushed out of"
        );
        let dist = (key.0 - 2)
            .abs()
            .max((key.1 - 2).abs())
            .max((key.2 - 1).abs());
        assert!(
            dist <= 1,
            "open_room's nearest free neighbour should be one ring out, got dist={dist}"
        );
    }

    /// Nothing is safe within `max_radius` (a radius-0 search from an already
    /// -blocked cell can only ever check itself) — the caller must be told
    /// `None` rather than getting a silently-wrong answer.
    #[test]
    fn nearest_free_standable_cell_returns_none_beyond_max_radius() {
        let mut map = open_room();
        map.set_cell(2, 2, 1, CellKind::WallSolid, 0);
        let found = nearest_free_standable_cell(&map, Vec3::new(2.5, 2.5, 1.5), 0);
        assert_eq!(found, None);
    }

    // -----------------------------------------------------------------------
    // Gravity + jump tests.
    // -----------------------------------------------------------------------

    /// Returns a pose that starts in mid-air (z=2.5) over the open room, and
    /// a default (zero-velocity, not grounded) PlayerMotion.
    fn airborne_setup() -> (CameraPose, PlayerMotion, Map) {
        let map = open_room();
        // Place the player in air layer z=2 (eye at 2.5).
        let cam = CameraPose {
            pos: Vec3::new(2.5, 2.5, 2.5),
            yaw: 0.0,
            pitch: 0.0,
        };
        let motion = PlayerMotion {
            vz: 0.0,
            grounded: false,
            ..Default::default()
        };
        (cam, motion, map)
    }

    #[test]
    fn gravity_lands_on_floor_and_stays() {
        // Drop the player from mid-air. After enough time they should land on
        // the floor (pos.z = 1.5 = PLAYER_EYE_HEIGHT above floor top at z=1.0).
        let (mut cam, mut motion, map) = airborne_setup();
        // Simulate a 2-second fall in 10 ms steps.
        for _ in 0..200 {
            cam.step_gravity(&mut motion, 0.01, &map);
        }
        // After landing, pos.z == floor-top (1.0) + EYE_HEIGHT (0.5) = 1.5.
        assert!(
            (cam.pos.z - 1.5).abs() < 0.01,
            "pos.z={} should be 1.5 after landing",
            cam.pos.z
        );
        assert!(motion.grounded, "should be grounded after landing");
        assert!(
            motion.vz.abs() < 0.1,
            "vz={} should be ~0 after landing",
            motion.vz
        );
    }

    #[test]
    fn cannot_fall_through_floor() {
        // Start at the exact ground level and apply large negative vz — the
        // floor must block it.
        let map = open_room();
        let mut cam = CameraPose {
            pos: Vec3::new(2.5, 2.5, 1.5),
            yaw: 0.0,
            pitch: 0.0,
        };
        let mut motion = PlayerMotion {
            vz: -100.0,
            grounded: false,
            ..Default::default()
        };
        cam.step_gravity(&mut motion, 1.0, &map);
        assert!(
            cam.pos.z >= 1.0,
            "pos.z={} must not go below floor top",
            cam.pos.z
        );
        assert!(motion.grounded, "must be grounded after hitting floor");
    }

    #[test]
    fn cannot_rise_through_ceiling() {
        // Give the player a large upward velocity — the ceiling at z=3 must
        // block them. Ceiling cell is at z=3 (sz-1=3), so head at pos.z +
        // HEAD_ROOM must not enter it.
        let map = open_room(); // sz=4, ceiling at z=3
                               // Start near the ceiling.
        let mut cam = CameraPose {
            pos: Vec3::new(2.5, 2.5, 2.0),
            yaw: 0.0,
            pitch: 0.0,
        };
        let mut motion = PlayerMotion {
            vz: 100.0,
            grounded: false,
            ..Default::default()
        };
        cam.step_gravity(&mut motion, 1.0, &map);
        // Head = pos.z + HEAD_ROOM must be < 3.0.
        assert!(
            cam.pos.z + PLAYER_HEAD_ROOM < 3.0 + 0.01,
            "head={} must not enter ceiling cell at z=3",
            cam.pos.z + PLAYER_HEAD_ROOM
        );
        assert_eq!(motion.vz, 0.0, "vz should be zeroed on ceiling hit");
    }

    #[test]
    fn jump_rises_then_falls_to_same_floor() {
        // From the floor, apply a jump. The player should rise then fall back
        // to exactly the same floor height (1.5) and be grounded again.
        let map = open_room();
        let mut cam = CameraPose {
            pos: Vec3::new(2.5, 2.5, 1.5),
            yaw: 0.0,
            pitch: 0.0,
        };
        let mut motion = PlayerMotion {
            vz: 0.0,
            grounded: true,
            ..Default::default()
        };

        // Jump.
        let jumped = CameraPose::try_jump(&mut motion);
        assert!(jumped, "should be able to jump when grounded");
        assert!(
            !motion.grounded,
            "should not be grounded immediately after jump"
        );

        // Track peak height.
        let mut peak_z = cam.pos.z;
        // Simulate 1.5 seconds in 10 ms steps.
        for _ in 0..150 {
            cam.step_gravity(&mut motion, 0.01, &map);
            if cam.pos.z > peak_z {
                peak_z = cam.pos.z;
            }
        }

        // Peak must be > 1 cell above the floor (z > 2.5).
        assert!(peak_z > 2.5, "peak_z={peak_z} should clear 1 cell");
        // After 1.5 s the player has landed.
        assert!(motion.grounded, "should be grounded after landing");
        assert!(
            (cam.pos.z - 1.5).abs() < 0.05,
            "pos.z={} should return to floor height 1.5",
            cam.pos.z
        );
    }

    #[test]
    fn jump_from_midair_denied() {
        // try_jump should do nothing if not grounded.
        let mut motion = PlayerMotion {
            vz: -3.0,
            grounded: false,
            ..Default::default()
        };
        let jumped = CameraPose::try_jump(&mut motion);
        assert!(!jumped, "mid-air jump must be denied");
        assert_eq!(motion.vz, -3.0, "vz must be unchanged");
    }

    #[test]
    fn grounded_flag_set_correctly() {
        // Simulate landing and check grounded transitions properly.
        let map = open_room();
        let mut cam = CameraPose {
            pos: Vec3::new(2.5, 2.5, 2.5),
            yaw: 0.0,
            pitch: 0.0,
        };
        let mut motion = PlayerMotion {
            vz: 0.0,
            grounded: false,
            ..Default::default()
        };

        // Initially falling — not grounded.
        cam.step_gravity(&mut motion, 0.01, &map);
        // On first step there's been no landing yet (just started falling).
        // We just need to ensure it doesn't claim grounded from thin air.

        // Simulate until grounded.
        for _ in 0..200 {
            if motion.grounded {
                break;
            }
            cam.step_gravity(&mut motion, 0.01, &map);
        }
        assert!(motion.grounded, "should land eventually");

        // One more step while on the floor — stays grounded.
        cam.step_gravity(&mut motion, 0.01, &map);
        assert!(motion.grounded, "should stay grounded on floor");
        // And pos.z stays at floor height.
        assert!(
            (cam.pos.z - 1.5).abs() < 0.02,
            "pos.z={} should be at 1.5",
            cam.pos.z
        );
    }

    #[test]
    fn gravity_deterministic() {
        // Same initial conditions → identical result every time.
        let run = || {
            let map = open_room();
            let mut cam = CameraPose {
                pos: Vec3::new(2.5, 2.5, 2.5),
                yaw: 0.0,
                pitch: 0.0,
            };
            let mut motion = PlayerMotion {
                vz: 0.0,
                grounded: false,
                ..Default::default()
            };
            for _ in 0..100 {
                cam.step_gravity(&mut motion, 0.016, &map);
            }
            (cam.pos.z, motion.vz, motion.grounded)
        };
        let a = run();
        let b = run();
        assert_eq!(a, b, "gravity must be deterministic");
    }

    /// Build a `5×5×sz` open room (floor at z=0, ceiling at z=sz-1, interior air
    /// between) directly — no `re_level` dependency. Used by the room-height jump
    /// sweep below to measure how much vertical rise each room height allows.
    fn open_room_sz(sz: usize) -> Map {
        let spec = GeometrySpec {
            size: (5, 5, sz),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[],
        };
        stamp_geometry(&spec)
    }

    /// REGRESSION (jump diagnosis, HYP-A/B): for each real-level room height
    /// (sz = 3, 4, 6) place the player grounded at the spawn eye height, prove
    /// `grounded == true` at spawn, jump, then integrate gravity at 1/60 s for
    /// ~1.5 s and record the PEAK eye z and peak rise. Asserts the measured rise
    /// matches the head-room clamp `eye_hi = (sz-1) - HEAD_ROOM` so a future
    /// change to JUMP_IMPULSE / clamp geometry is caught.
    ///
    /// The numbers this prints (via the assert messages on failure) feed the jump
    /// root-cause report: in a short room the ceiling clamp swallows almost all of
    /// the jump arc.
    #[test]
    fn jump_rise_per_room_height() {
        // start eye = 1.0 (floor top) + EYE_HEIGHT = 1.5 for every standard room.
        let start_z = 1.0 + PLAYER_EYE_HEIGHT;
        // The full free-flight apex of a 6.5 cell/s impulse under g=20:
        //   h = v² / (2g) = 6.5² / 40 ≈ 1.056 cells.
        let free_apex_rise = (JUMP_IMPULSE * JUMP_IMPULSE) / (2.0 * GRAVITY);

        for &sz in &[3usize, 4, 6] {
            let map = open_room_sz(sz);
            let mut cam = CameraPose {
                pos: Vec3::new(2.5, 2.5, start_z),
                yaw: 0.0,
                pitch: 0.0,
            };
            let mut motion = PlayerMotion {
                vz: 0.0,
                grounded: false,
                ..Default::default()
            };

            // Settle one frame so `grounded` is computed from contact (mirrors the
            // game: gravity_sys runs every frame, including the spawn frame).
            cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
            assert!(
                motion.grounded,
                "sz={sz}: player must be GROUNDED at spawn eye z={} (HYP-A check)",
                cam.pos.z
            );

            // Jump.
            let jumped = CameraPose::try_jump(&mut motion);
            assert!(jumped, "sz={sz}: try_jump must succeed when grounded");

            // Integrate ~1.5 s at 1/60 and record the peak.
            let mut peak_z = cam.pos.z;
            for _ in 0..90 {
                cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
                if cam.pos.z > peak_z {
                    peak_z = cam.pos.z;
                }
            }
            let peak_rise = peak_z - start_z;

            // The ceiling clamp caps the eye at eye_hi = (sz-1) - HEAD_ROOM.
            let eye_hi = (sz - 1) as f32 - PLAYER_HEAD_ROOM;
            let clamp_rise = eye_hi - start_z;
            // The achievable rise is the SMALLER of the free apex and the clamp.
            let expected = free_apex_rise.min(clamp_rise);

            // Peak must land at the expected ceiling-limited apex (±0.06 cell for the
            // 1/60 integration granularity), and must be returning to the floor by
            // the end of the window.
            assert!(
                (peak_rise - expected).abs() < 0.08,
                "sz={sz}: peak_rise={peak_rise:.3} (peak_z={peak_z:.3}); expected \
                 min(free_apex={free_apex_rise:.3}, clamp={clamp_rise:.3})={expected:.3}"
            );
            assert!(
                (cam.pos.z - start_z).abs() < 0.06,
                "sz={sz}: player should have landed back at {start_z}, got {}",
                cam.pos.z
            );
        }
    }

    // -----------------------------------------------------------------------
    // Coyote-time + jump-buffer tests (Swaggerfall-study item 5).
    // -----------------------------------------------------------------------

    /// A raised 1-cell-tall deck (top at world z=3.0, standing eye z=3.5)
    /// spanning x=1..=3 at y=2, with nothing beyond x=3 except open air all
    /// the way down to the real floor (top z=1.0, eye z=1.5) — a clean
    /// 2-cell drop off a ledge that needs no jump to survive.
    fn ledge_room() -> Map {
        let spec = GeometrySpec {
            size: (6, 5, 5),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[],
        };
        let mut map = stamp_geometry(&spec);
        for x in 1..=3 {
            map.set(x, 2, 2, CellKind::WallSolid);
        }
        map
    }

    #[test]
    fn walk_off_a_ledge_then_jump_within_coyote_window_still_succeeds() {
        let map = ledge_room();
        // Start standing on the deck's last cell (x=3, centre 3.5), eye z=3.5.
        let mut cam = CameraPose {
            pos: Vec3::new(3.5, 2.5, 3.5),
            yaw: 0.0,
            pitch: 0.0,
        };
        let mut motion = PlayerMotion {
            vz: 0.0,
            grounded: true,
            ..Default::default()
        };
        cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
        assert!(motion.grounded, "must start grounded on the deck");

        // Step off the ledge — teleport past the deck's edge (no collision
        // needed for this vertical-physics test; this is exactly where a
        // horizontal move off the last deck cell would land).
        cam.pos.x = 4.5;

        // One frame of gravity: the deck no longer supports this column, so
        // the player must now read as airborne, with the coyote clock just
        // starting.
        cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
        assert!(
            !motion.grounded,
            "must be airborne immediately after leaving the deck"
        );
        assert!(
            motion.since_grounded < COYOTE_TIME,
            "just left the ledge — still within the window"
        );

        // Press jump well within the coyote window: must still fire, exactly
        // like a grounded jump (same impulse, same arc — only the ALLOWED
        // press timing widened).
        let jumped = CameraPose::try_jump(&mut motion);
        assert!(jumped, "a press within the coyote window must still jump");
        assert_eq!(
            motion.vz, JUMP_IMPULSE,
            "coyote jump must set the FULL impulse"
        );
        assert!(
            motion.jumped_airborne,
            "coyote jump counts as this airborne phase's one jump"
        );
    }

    #[test]
    fn walk_off_a_ledge_then_jump_after_coyote_window_is_denied() {
        let map = ledge_room();
        let mut cam = CameraPose {
            pos: Vec3::new(3.5, 2.5, 3.5),
            yaw: 0.0,
            pitch: 0.0,
        };
        let mut motion = PlayerMotion {
            vz: 0.0,
            grounded: true,
            ..Default::default()
        };
        cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
        cam.pos.x = 4.5;
        cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
        assert!(!motion.grounded, "must be airborne after leaving the deck");

        // Let the coyote window fully lapse. The drop to the real floor
        // below takes ~0.45s (sqrt(2*2.0/GRAVITY)), so a bounded 20-frame
        // sweep (~0.33s) stops well short of landing while comfortably
        // clearing the 0.10s window.
        for _ in 0..20 {
            if motion.since_grounded >= COYOTE_TIME + 0.05 {
                break;
            }
            cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
        }
        assert!(
            motion.since_grounded >= COYOTE_TIME,
            "since_grounded={} should have cleared the window",
            motion.since_grounded
        );
        assert!(
            !motion.grounded,
            "still airborne mid-fall (drop is ~0.45s, window is 0.10s)"
        );

        let vz_before = motion.vz;
        let jumped = CameraPose::try_jump(&mut motion);
        assert!(!jumped, "a press after the coyote window must NOT jump");
        assert_eq!(
            motion.vz, vz_before,
            "a denied press must not touch velocity"
        );
        assert!(
            motion.jump_buffered.is_some(),
            "the late press is buffered instead of silently dropped"
        );
    }

    #[test]
    fn try_jump_no_double_jump_within_the_same_airborne_phase() {
        // A grounded jump fires normally...
        let mut motion = PlayerMotion {
            vz: 0.0,
            grounded: true,
            ..Default::default()
        };
        assert!(CameraPose::try_jump(&mut motion), "grounded jump must fire");
        assert!(!motion.grounded);
        assert!(motion.jumped_airborne);

        // ...but an immediate second press, even though `since_grounded` is
        // still comfortably inside the coyote window (it was just reset to
        // 0.0 by the jump itself), must NOT grant a second jump —
        // `jumped_airborne` blocks it. This is the "only once per airborne
        // phase" rule: coyote time forgives WHEN you pressed, never HOW
        // MANY TIMES.
        let vz_before = motion.vz;
        assert!(
            motion.since_grounded < COYOTE_TIME,
            "since_grounded is still 0.0 right after jumping"
        );
        let jumped_again = CameraPose::try_jump(&mut motion);
        assert!(
            !jumped_again,
            "a second press mid-flight must not double-jump"
        );
        assert_eq!(
            motion.vz, vz_before,
            "a denied press must not touch velocity"
        );
    }

    #[test]
    fn step_gravity_fires_a_buffered_jump_exactly_on_landing() {
        let map = open_room();
        // Positioned to land on the very next `step_gravity` call (same
        // one-call floor-snap `cannot_fall_through_floor` exercises): just
        // above the floor, falling.
        let mut cam = CameraPose {
            pos: Vec3::new(2.5, 2.5, 1.55),
            yaw: 0.0,
            pitch: 0.0,
        };
        let mut motion = PlayerMotion {
            vz: -5.0,
            grounded: false,
            since_grounded: 0.5, // long past the coyote window — irrelevant here
            jump_buffered: Some(0.02), // a recent press, well inside JUMP_BUFFER_TIME
            jumped_airborne: true,
        };
        cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
        assert!(
            (cam.pos.z - 1.5).abs() < 0.05,
            "must land at floor height, pos.z={}",
            cam.pos.z
        );
        assert_eq!(
            motion.vz, JUMP_IMPULSE,
            "a buffered press must fire the FULL jump impulse on landing"
        );
        assert!(
            !motion.grounded,
            "the auto-fired jump immediately un-grounds the player again"
        );
        assert!(
            motion.jumped_airborne,
            "the auto-fired jump counts as this (new) airborne phase's jump"
        );
        assert_eq!(
            motion.jump_buffered, None,
            "the buffer is consumed once it fires"
        );
    }

    #[test]
    fn step_gravity_drops_a_stale_buffered_jump_instead_of_firing_it() {
        let map = open_room();
        let mut cam = CameraPose {
            pos: Vec3::new(2.5, 2.5, 1.55),
            yaw: 0.0,
            pitch: 0.0,
        };
        let mut motion = PlayerMotion {
            vz: -5.0,
            grounded: false,
            since_grounded: 0.5,
            jump_buffered: Some(JUMP_BUFFER_TIME + 0.01), // just past the window
            jumped_airborne: true,
        };
        cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
        assert!(
            (cam.pos.z - 1.5).abs() < 0.05,
            "must land at floor height, pos.z={}",
            cam.pos.z
        );
        assert!(
            motion.vz.abs() < 0.1,
            "a stale buffered press must NOT fire — this is a normal landing"
        );
        assert!(motion.grounded, "a normal landing stays grounded");
        assert_eq!(
            motion.jump_buffered, None,
            "the stale buffer is dropped, not carried forward"
        );
    }

    #[test]
    fn coyote_and_buffer_timers_are_fixed_timestep_deterministic() {
        // Same scripted walk-off-the-ledge-then-press-jump sequence, run
        // twice, must produce byte-identical results — matches
        // `gravity_deterministic`'s pattern, extended to the new timer
        // fields.
        let run = || {
            let map = ledge_room();
            let mut cam = CameraPose {
                pos: Vec3::new(3.5, 2.5, 3.5),
                yaw: 0.0,
                pitch: 0.0,
            };
            let mut motion = PlayerMotion {
                vz: 0.0,
                grounded: true,
                ..Default::default()
            };
            cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
            cam.pos.x = 4.5;
            for _ in 0..10 {
                cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
            }
            let jumped = CameraPose::try_jump(&mut motion);
            for _ in 0..60 {
                cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
            }
            (
                cam.pos.z,
                motion.vz,
                motion.grounded,
                motion.since_grounded,
                motion.jump_buffered,
                motion.jumped_airborne,
                jumped,
            )
        };
        let a = run();
        let b = run();
        assert_eq!(
            a, b,
            "coyote/jump-buffer state must be deterministic under a fixed timestep"
        );
    }

    #[test]
    fn collision_deterministic() {
        // Same pose + move → same result every time.
        let run = || {
            let map = open_room();
            let mut cam = mid_pose();
            cam.move_with_collision(0.3, -0.2, &map, PLAYER_RADIUS);
            (cam.pos.x, cam.pos.y)
        };
        let a = run();
        let b = run();
        assert_eq!(a, b, "collision must be deterministic");
    }

    #[test]
    #[cfg(feature = "reflect")]
    fn camera_pose_is_reflectable() {
        fn assert_reflect<T: bevy::reflect::Reflect>() {}
        assert_reflect::<CameraPose>();
    }

    // -----------------------------------------------------------------------
    // Portal tests.
    // -----------------------------------------------------------------------

    /// (a) A default map has no portals and portal_for_cell returns None
    /// everywhere, including out-of-bounds coordinates.
    #[test]
    fn portal_default_map_has_no_portals() {
        let m = test_room();
        assert!(m.portals.is_empty(), "fresh map must have no portals");
        // Interior air cell.
        assert_eq!(m.portal_for_cell(3, 3, 1), None);
        // Out-of-bounds.
        assert_eq!(m.portal_for_cell(-1, 0, 0), None);
        assert_eq!(m.portal_for_cell(99, 99, 99), None);
        // A solid wall cell (not a portal).
        assert_eq!(m.portal_for_cell(0, 0, 1), None);
    }

    /// (b) After inject_demo_portal_pair():
    ///   - exactly 2 PortalLinks registered;
    ///   - stamped cells are CellKind::Portal with is_solid()==true and map to
    ///     SurfaceKind::Portal via the CellKind predicate;
    ///   - portal_for_cell returns Some on them and None on non-portal cells.
    #[test]
    fn portal_inject_stamps_cells_and_registers_links() {
        use re_core::SurfaceKind;

        // Build the 12×16×10 room that matches levels/92-fractal-portal.lvl.
        let spec = GeometrySpec {
            size: (12, 16, 10),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[],
        };
        let mut map = stamp_geometry(&spec);
        map.inject_demo_portal_pair();

        // Exactly two portal links.
        assert_eq!(map.portals.len(), 2, "expected exactly 2 PortalLinks");

        // --- North wall (portal A, idx=0): x∈[1,10], y=0, z∈[1,8] ---
        // A cell in the middle of the north face.
        let north_cell = map.cell_at(5, 0, 4);
        assert_eq!(north_cell, CellKind::Portal, "north face must be Portal");
        assert!(north_cell.is_solid(), "portal cell must be solid (Stage 1)");
        assert_eq!(map.portal_for_cell(5, 0, 4), Some(0));
        // Edge cells of the north face span.
        assert_eq!(map.portal_for_cell(1, 0, 1), Some(0));
        assert_eq!(map.portal_for_cell(10, 0, 8), Some(0));

        // --- West wall (portal B, idx=1): x=0, y∈[1,14], z∈[1,8] ---
        let west_cell = map.cell_at(0, 7, 4);
        assert_eq!(west_cell, CellKind::Portal, "west face must be Portal");
        assert!(west_cell.is_solid(), "portal cell must be solid (Stage 1)");
        assert_eq!(map.portal_for_cell(0, 7, 4), Some(1));
        assert_eq!(map.portal_for_cell(0, 1, 1), Some(1));
        assert_eq!(map.portal_for_cell(0, 14, 8), Some(1));

        // Non-portal cells return None.
        assert_eq!(
            map.portal_for_cell(5, 5, 1),
            None,
            "interior air is not a portal"
        );
        assert_eq!(
            map.portal_for_cell(11, 0, 1),
            None,
            "unpainted wall is not a portal"
        );

        // CellKind::Portal maps to SurfaceKind::Portal (the re_core enum value).
        // We verify this by checking the variant — re_world doesn't own the
        // mapping function (that lives in re_render), but we can confirm the
        // new enum variant compiles and is distinct.
        let _ = SurfaceKind::Portal; // compile-time proof the variant exists in re_core
    }

    /// `add_portal_pair` (the shared implementation) produces portals BIT-FOR-BIT
    /// identical to the refactored `inject_demo_portal_pair`, proving the parser
    /// path and the scaffold path cannot diverge.
    #[test]
    fn add_portal_pair_matches_demo() {
        let spec = GeometrySpec {
            size: (12, 16, 10),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[],
        };

        // Path 1: the scaffold (which now delegates to add_portal_pair).
        let mut demo = stamp_geometry(&spec);
        demo.inject_demo_portal_pair();

        // Path 2: call add_portal_pair directly with the same regions.
        let mut direct = stamp_geometry(&spec);
        direct.add_portal_pair(((1, 0, 1), (10, 0, 8)), ((0, 1, 1), (0, 14, 8)), 0.0);

        assert_eq!(demo.portals.len(), direct.portals.len());
        for (i, (pa, pb)) in demo.portals.iter().zip(direct.portals.iter()).enumerate() {
            assert_eq!(pa.plane_pos, pb.plane_pos, "portal {i} plane_pos");
            assert_eq!(pa.plane_normal, pb.plane_normal, "portal {i} plane_normal");
            assert_eq!(pa.half_u, pb.half_u, "portal {i} half_u");
            assert_eq!(pa.half_v, pb.half_v, "portal {i} half_v");
            assert_eq!(pa.delta, pb.delta, "portal {i} delta");
            assert_eq!(pa.delta_inv, pb.delta_inv, "portal {i} delta_inv");
        }
        // Stamped cells identical too.
        for z in 0..10 {
            for y in 0..16 {
                for x in 0..12 {
                    assert_eq!(
                        demo.portal_for_cell(x, y, z),
                        direct.portal_for_cell(x, y, z),
                        "portal cell mismatch at ({x},{y},{z})"
                    );
                }
            }
        }
    }

    /// (c) Round-trip: delta_inv is the inverse of delta (A→B→A within 1e-4),
    /// and a point on plane A maps near plane B under delta.
    #[test]
    fn portal_delta_round_trip() {
        let spec = GeometrySpec {
            size: (12, 16, 10),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[],
        };
        let mut map = stamp_geometry(&spec);
        map.inject_demo_portal_pair();

        assert_eq!(map.portals.len(), 2);
        let link_a = &map.portals[0];
        let link_b = &map.portals[1];

        // (i) delta_inv is the inverse of delta within numerical tolerance.
        //     delta * delta_inv should be (very close to) the identity matrix.
        let identity_ab = link_a.delta * link_a.delta_inv;
        for col in 0..4 {
            for row in 0..4 {
                let expected = if row == col { 1.0_f32 } else { 0.0_f32 };
                let got = identity_ab.col(col)[row];
                assert!(
                    (got - expected).abs() < 1e-4,
                    "delta*delta_inv[{row},{col}] = {got}, expected {expected}"
                );
            }
        }
        // Same for link B.
        let identity_ba = link_b.delta * link_b.delta_inv;
        for col in 0..4 {
            for row in 0..4 {
                let expected = if row == col { 1.0_f32 } else { 0.0_f32 };
                let got = identity_ba.col(col)[row];
                assert!(
                    (got - expected).abs() < 1e-4,
                    "B: delta*delta_inv[{row},{col}] = {got}, expected {expected}"
                );
            }
        }

        // (ii) A→B: a point on portal A's inner-face plane should map onto
        // portal B's inner-face plane.
        // portal A plane_pos = (5.5, 1.0, 4.5) — the inner face at y=1.
        // After delta it should arrive at portal B's plane_pos = (1.0, 7.5, 4.5).
        let a_point = link_a.plane_pos.extend(1.0); // homogeneous
        let mapped = link_a.delta * a_point;
        let b_pos = link_b.plane_pos;
        // The mapped point should be within a small tolerance of portal B's inner-face centre.
        assert!(
            (mapped.x - b_pos.x).abs() < 1e-4,
            "A→B x: got {}, expected {}",
            mapped.x,
            b_pos.x
        );
        assert!(
            (mapped.y - b_pos.y).abs() < 1e-4,
            "A→B y: got {}, expected {}",
            mapped.y,
            b_pos.y
        );
        assert!(
            (mapped.z - b_pos.z).abs() < 1e-4,
            "A→B z: got {}, expected {}",
            mapped.z,
            b_pos.z
        );

        // (iii) B→A: the inverse should map portal B's inner-face centre back near portal A's.
        let b_point = link_b.plane_pos.extend(1.0);
        let back = link_b.delta * b_point;
        let a_pos = link_a.plane_pos;
        assert!(
            (back.x - a_pos.x).abs() < 1e-4,
            "B→A x: got {}, expected {}",
            back.x,
            a_pos.x
        );
        assert!(
            (back.y - a_pos.y).abs() < 1e-4,
            "B→A y: got {}, expected {}",
            back.y,
            a_pos.y
        );
        assert!(
            (back.z - a_pos.z).abs() < 1e-4,
            "B→A z: got {}, expected {}",
            back.z,
            a_pos.z
        );
    }

    /// (d) Front-to-front flip: a ray arriving along portal A's inward normal
    /// direction (−normal_A in world space) must emerge from portal B traveling
    /// INTO B's room (dot with normal_B > 0), and vice versa.
    ///
    /// This is the regression guard for the 180° "front-to-front" flip baked
    /// into `inject_demo_portal_pair`. Without the flip the exiting ray would
    /// travel BACK INTO the wall (dot < 0), producing fallback slices in the
    /// --dump render instead of the destination room interior.
    #[test]
    fn portal_delta_exit_direction_is_into_room() {
        let spec = GeometrySpec {
            size: (12, 16, 10),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[],
        };
        let mut map = stamp_geometry(&spec);
        map.inject_demo_portal_pair();

        assert_eq!(map.portals.len(), 2);
        let link_a = &map.portals[0];
        let link_b = &map.portals[1];

        // Portal A: normal = +y (room-ward, into A's room from the north wall).
        // A ray traveling into the north wall moves in −y, i.e. the ray
        // direction approaching portal A is (0, −1, 0) in world space.
        let arriving_at_a = Vec3::new(0.0, -1.0, 0.0); // approaching A
                                                       // Transform the direction through delta_A→B (linear/homogeneous 0 part).
        let exiting_b = link_a.delta.transform_vector3(arriving_at_a).normalize();
        // The exit direction must travel INTO B's room: dot with B's inward
        // normal (+x) must be positive.
        assert!(
            exiting_b.dot(link_b.plane_normal) > 0.0,
            "A→B: exit direction {exiting_b:?} must have dot > 0 with B normal {:?}",
            link_b.plane_normal
        );

        // Portal B: normal = +x (room-ward, into B's room from the west wall).
        // A ray arriving at portal B travels in −x.
        let arriving_at_b = Vec3::new(-1.0, 0.0, 0.0); // approaching B
        let exiting_a = link_b.delta.transform_vector3(arriving_at_b).normalize();
        // The exit direction must travel INTO A's room: dot with A's inward
        // normal (+y) must be positive.
        assert!(
            exiting_a.dot(link_a.plane_normal) > 0.0,
            "B→A: exit direction {exiting_a:?} must have dot > 0 with A normal {:?}",
            link_a.plane_normal
        );
    }

    /// Portal cell is solid and NOT a hologram or a wall (is_wall returns false
    /// for Portal — the renderer uses a dedicated dispatch path, not the wall path).
    #[test]
    fn portal_cellkind_predicates() {
        assert!(CellKind::Portal.is_solid(), "portal must be solid");
        assert!(!CellKind::Portal.is_hologram(), "portal is not a hologram");
        assert!(
            !CellKind::Portal.is_wall(),
            "portal is not a conventional wall"
        );
    }

    // -----------------------------------------------------------------------
    // Stage 2: collision predicate split + seamless teleport tests.
    // -----------------------------------------------------------------------

    /// is_collidable() must diverge from is_solid() for Portal: Portal is
    /// solid (raycaster stops on it) but NOT collidable (player walks through).
    /// All other kinds must be unchanged.
    #[test]
    fn is_collidable_split_from_is_solid_for_portal() {
        // Portal: solid for raycaster, walkable for player.
        assert!(
            CellKind::Portal.is_solid(),
            "Portal must be solid (raycaster)"
        );
        assert!(
            !CellKind::Portal.is_collidable(),
            "Portal must NOT be collidable (player walks through)"
        );

        // Walls: both solid and collidable.
        assert!(CellKind::WallSolid.is_solid() && CellKind::WallSolid.is_collidable());
        assert!(CellKind::WallTrace.is_solid() && CellKind::WallTrace.is_collidable());
        assert!(CellKind::WallMatrix.is_solid() && CellKind::WallMatrix.is_collidable());
        assert!(CellKind::Object.is_solid() && CellKind::Object.is_collidable());

        // Non-solid / pass-through cells are not collidable either.
        assert!(!CellKind::Empty.is_solid() && !CellKind::Empty.is_collidable());
        assert!(!CellKind::Hologram.is_solid() && !CellKind::Hologram.is_collidable());
        assert!(!CellKind::Floor.is_solid() && !CellKind::Floor.is_collidable());
        assert!(!CellKind::Ceiling.is_solid() && !CellKind::Ceiling.is_collidable());
    }

    /// A move into a Portal cell must NOT be blocked by collision (the player
    /// passes through), while a move into a WallSolid IS blocked (regression).
    ///
    /// Setup: two 7×7×4 rooms. In the "wall" room the east interior wall at x=5
    /// is a WallSolid. In the "portal" room the same cell is a Portal. Compare:
    /// starting at x=2.5, a 2.8-unit move in +x:
    /// - wall room:   player must NOT enter the wall cell (x stops before x=5).
    /// - portal room: player CAN enter the portal cell (x reaches past 5−radius).
    #[test]
    fn portal_cell_is_walkable_wall_cell_is_not() {
        // 7×7×4 rooms — large enough that the east perimeter wall (x=6) is
        // well beyond the portal cell at x=5, so OOB doesn't interfere.
        let spec = GeometrySpec {
            size: (7, 7, 4),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[],
        };

        // WALL room: interior column at (5, 3, 1) is WallSolid.
        let mut wall_map = stamp_geometry(&spec);
        wall_map.set(5, 3, 1, CellKind::WallSolid);

        // PORTAL room: same column replaced with Portal.
        let mut portal_map = stamp_geometry(&spec);
        portal_map.set(5, 3, 1, CellKind::Portal);

        // Starting pose: x=2.5, y=3.5, eye z=1.5.
        let start = Vec3::new(2.5, 3.5, 1.5);

        // Move 2.8 units in +x — enough to reach cell x=5 from x=2.5.
        let dx = 2.8_f32;

        // Wall room: blocked before entering the WallSolid.
        let mut cam_wall = CameraPose {
            pos: start,
            yaw: 0.0,
            pitch: 0.0,
        };
        cam_wall.move_with_collision(dx, 0.0, &wall_map, PLAYER_RADIUS);
        assert!(
            cam_wall.pos.x + PLAYER_RADIUS <= 5.0 + 1e-4,
            "wall room: x={} must not enter the WallSolid at x=5",
            cam_wall.pos.x
        );

        // Portal room: NOT blocked — player passes into the portal cell.
        let mut cam_portal = CameraPose {
            pos: start,
            yaw: 0.0,
            pitch: 0.0,
        };
        cam_portal.move_with_collision(dx, 0.0, &portal_map, PLAYER_RADIUS);
        assert!(
            cam_portal.pos.x > cam_wall.pos.x,
            "portal room: x={} must be further than wall room x={} (portal is walkable)",
            cam_portal.pos.x,
            cam_wall.pos.x
        );
        // Specifically: the portal-room player should have entered cell x=5.
        assert!(
            cam_portal.pos.x > 5.0 - PLAYER_RADIUS,
            "portal room: x={} must have entered the Portal cell at x=5",
            cam_portal.pos.x
        );

        // WallSolid in +y direction still blocks (regression check).
        let mut cam_y = CameraPose {
            pos: start,
            yaw: 0.0,
            pitch: 0.0,
        };
        cam_y.move_with_collision(0.0, 10.0, &portal_map, PLAYER_RADIUS);
        assert!(
            cam_y.pos.y < 6.0,
            "y={} must not pass the south WallSolid perimeter at y=6",
            cam_y.pos.y
        );
    }

    /// Helper: build the 12×16×10 portal-demo map.
    fn portal_demo_map() -> Map {
        let spec = GeometrySpec {
            size: (12, 16, 10),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[],
        };
        let mut map = stamp_geometry(&spec);
        map.inject_demo_portal_pair();
        map
    }

    /// Plane-cross detection: stepping from one side of portal A's plane to the
    /// other (within bounds) triggers exactly one crossing. Stepping to the SAME
    /// side, or OUTSIDE the opening bounds, triggers none.
    #[test]
    fn portal_plane_cross_detection() {
        let map = portal_demo_map();
        // Portal A: inner-face plane at y=1.0, normal=+y (pointing into room, +y).
        // Opening cells: x∈[1,10], z∈[1,8]. Opening centre = (5.5, 1.0, 4.5).
        // The player approaches from the south (y > 1.0) and crosses to y < 1.0.

        // --- (1) Valid crossing: before is at y=1.3 (south/room side, da=+0.3),
        //     after is at y=0.7 (inside the portal cell row, db=−0.3). ---
        let pos_before_valid = Vec3::new(5.5, 1.3, 4.5); // room side (+y relative to plane)
        let pos_after_valid = Vec3::new(5.5, 0.7, 4.5); // past the inner face (into portal cell)

        let mut cam = CameraPose {
            pos: pos_after_valid,
            yaw: 0.0,
            pitch: 0.0,
        };
        let yaw_before = cam.yaw;
        let pos_before_snap = cam.pos; // will be overwritten by teleport
        cam.apply_portal_crossings(pos_before_valid, &map);

        // After a valid crossing, position must have changed (teleported).
        assert_ne!(
            cam.pos, pos_before_snap,
            "pose must be re-based after a valid plane crossing"
        );
        // Yaw must have rotated (delta includes a 90° turn A→B).
        let yaw_delta = (cam.yaw - yaw_before).abs();
        // The delta is a 90° rotation; normalised yaw difference should be ~π/2.
        assert!(
            yaw_delta > 0.1,
            "yaw must change after crossing portal A; got Δyaw={yaw_delta:.4}"
        );

        // --- (2) Same-side: before and after both south of the plane (y > 1.0) → no cross ---
        let mut cam2 = CameraPose {
            pos: Vec3::new(5.5, 1.5, 4.5),
            yaw: 0.0,
            pitch: 0.0,
        };
        let pos2_before = Vec3::new(5.5, 2.0, 4.5);
        let pos2_snap = cam2.pos;
        cam2.apply_portal_crossings(pos2_before, &map);
        assert_eq!(
            cam2.pos, pos2_snap,
            "same-side segment must not trigger teleport"
        );

        // --- (3) Outside bounds: cross the plane but x is out of the opening ---
        // Opening x∈[1,10]; x=11.5 is outside.
        let mut cam3 = CameraPose {
            pos: Vec3::new(11.5, 0.7, 4.5),
            yaw: 0.0,
            pitch: 0.0,
        };
        let pos3_before = Vec3::new(11.5, 1.3, 4.5);
        let pos3_snap = cam3.pos;
        cam3.apply_portal_crossings(pos3_before, &map);
        assert_eq!(
            cam3.pos, pos3_snap,
            "out-of-bounds crossing must not trigger teleport"
        );
    }

    /// Pose re-base correctness: after crossing portal A, the new pos equals
    /// `delta_A.transform_point3(pos_after)`, and the yaw has rotated by the
    /// delta's yaw component.
    #[test]
    fn portal_pose_rebase_correctness() {
        let map = portal_demo_map();
        let link_a = &map.portals[0];

        // A pose crossing portal A's inner-face plane at y=1.0: before at y=1.3
        // (room side), after at y=0.7 (into the portal cell row).
        let pos_before = Vec3::new(5.5, 1.3, 4.5);
        let pos_after = Vec3::new(5.5, 0.7, 4.5);
        let initial_yaw: f32 = 0.0;

        let mut cam = CameraPose {
            pos: pos_after,
            yaw: initial_yaw,
            pitch: 0.0,
        };
        cam.apply_portal_crossings(pos_before, &map);

        // Expected new position = delta_A applied to pos_after, plus the
        // anti-double-warp nudge (1e-3 world-units along the EXIT direction —
        // the room-side exit direction in the destination).
        // exit_dir = delta.transform_vector3(-plane_normal) pushes the player
        // INTO the destination room (away from the destination portal plane).
        let exit_dir = link_a.delta.transform_vector3(-link_a.plane_normal);
        let expected_pos = link_a.delta.transform_point3(pos_after) + exit_dir * 1e-3;
        assert!(
            (cam.pos - expected_pos).length() < 1e-4,
            "pos after teleport: got {:?}, expected {:?}",
            cam.pos,
            expected_pos
        );

        // Yaw re-base: the player's 3-D forward (sin yaw, cos yaw, 0) is
        // transformed by delta_A→B and the new yaw is read back via atan2.
        // This is the SAME transform the raycaster applies to a ray direction
        // so walk and view always agree (fixes additive-delta bug).
        let old_dir = Vec3::new(initial_yaw.sin(), initial_yaw.cos(), 0.0);
        let new_dir = link_a.delta.transform_vector3(old_dir);
        let expected_yaw = normalise_yaw(new_dir.x.atan2(new_dir.y));
        assert!(
            (cam.yaw - expected_yaw).abs() < 1e-4,
            "yaw after teleport: got {:.4}, expected {:.4}",
            cam.yaw,
            expected_yaw
        );
    }

    /// Round-trip: a cross through portal A → then a cross through portal B
    /// should (approximately) return the pose to its starting position and yaw.
    ///
    /// This tests the A→B→A reciprocity: the two deltas are inverses of each
    /// other, so two teleports compose to the identity.
    #[test]
    fn portal_round_trip_a_to_b_to_a() {
        let map = portal_demo_map();
        let link_a = &map.portals[0];
        let link_b = &map.portals[1];

        // --- Step 1: teleport A→B ---
        // Portal A inner-face plane at y=1.0: before at y=1.3, after at y=0.7.
        let pos_before_a = Vec3::new(5.5, 1.3, 4.5);
        let pos_after_a = Vec3::new(5.5, 0.7, 4.5);
        let initial_pos = pos_after_a;
        let initial_yaw: f32 = 0.0;

        let mut cam = CameraPose {
            pos: pos_after_a,
            yaw: initial_yaw,
            pitch: 0.0,
        };
        cam.apply_portal_crossings(pos_before_a, &map);

        let pos_at_b = cam.pos;
        let yaw_at_b = cam.yaw;

        // After A→B we must now be near portal B's inner-face plane (x≈1.0),
        // nudged INTO the room by the exit direction (1e-3 units).
        // exit_dir for portal A = delta_AB.transform_vector3(-plane_normal_A).
        let exit_dir_a = link_a.delta.transform_vector3(-link_a.plane_normal);
        let expected_at_b = link_a.delta.transform_point3(pos_after_a) + exit_dir_a * 1e-3;
        assert!(
            (pos_at_b - expected_at_b).length() < 1e-3,
            "after A→B: pos={:?} expected near {:?}",
            pos_at_b,
            expected_at_b
        );

        // --- Step 2: construct a crossing segment through portal B ---
        // Portal B: inner-face plane at x=1.0, normal=+x.
        // We need to cross from +x side to −x side (room → into portal cell).
        // pos_at_b is near x=1.3 (from the A→B re-base) plus the +X nudge,
        // so pos_at_b.x ≈ 1.301. +0.3 puts it deeper in the room; −0.5 crosses
        // the x=1.0 inner face (0.3 + 0.2 margin ensures x_after < 1.0).
        let pos_before_b = Vec3::new(pos_at_b.x + 0.3, pos_at_b.y, pos_at_b.z);
        let pos_after_b = Vec3::new(pos_at_b.x - 0.5, pos_at_b.y, pos_at_b.z);

        // Verify the crossing segment actually crosses portal B's plane.
        let n_b = link_b.plane_normal; // +x
        let p_b = link_b.plane_pos;
        let da_b = n_b.dot(pos_before_b - p_b);
        let db_b = n_b.dot(pos_after_b - p_b);
        assert!(
            da_b * db_b <= 0.0,
            "round-trip test: segment must cross portal B plane (da={da_b}, db={db_b})"
        );

        cam.pos = pos_after_b;
        cam.apply_portal_crossings(pos_before_b, &map);

        // After A→B→A the final position must be near the ORIGINAL position.
        // Tolerance is looser (~0.7 cell) because pos_at_b was offset ±0.3/0.5
        // cells for the second crossing construction, and the B→A nudge adds
        // another 1e-3 in the exit direction.
        assert!(
            (cam.pos - initial_pos).length() < 0.7,
            "round-trip A→B→A: final pos {:?} must be near start {:?}",
            cam.pos,
            initial_pos
        );

        // Yaw reciprocity (forward-vector re-base model):
        // A→B: transforms the player's forward through delta_AB.
        // B→A: transforms the result through delta_BA = inverse(delta_AB).
        // Two applications of a rigid transform and its inverse compose to the
        // identity — the final yaw must be near the original.
        //
        // Note: the old additive delta_yaw approach gave a net ~180° turn (not
        // identity) because additive composition of the same rotation angle
        // doubles rather than cancels. The forward-vector re-base is correct:
        // delta_BA(delta_AB(v)) == v.
        let yaw_round_trip_err = normalise_yaw(cam.yaw - initial_yaw).abs();
        assert!(
            yaw_round_trip_err < 0.05,
            "A→B→A round-trip yaw must return to initial {initial_yaw:.4}; \
             got {:.4}, err={yaw_round_trip_err:.4}",
            cam.yaw
        );
        // Intermediate (A→B) yaw must have actually changed — the portal is not
        // the identity transform.
        let delta1 = normalise_yaw(yaw_at_b - initial_yaw).abs();
        assert!(
            delta1 > 0.05,
            "A→B delta must be non-zero (portal has a rotation); Δ1={delta1:.4}"
        );
    }

    /// GAMEPLAY WALK TEST — proves the teleport actually FIRES during real movement.
    ///
    /// This is the gap that the unit tests missed: the portal plane was at y=0.0
    /// (outer grid boundary) so the player could never reach the far side; the
    /// fix moves the plane to y=1.0 (inner face of the portal cell row).
    ///
    /// Setup: a 12×16×10 demo-portal map. Spawn the camera near the south end
    /// of the room (y≈14.5, analogous to the level-92 spawn), facing north
    /// (yaw = π). Step forward (−y) in small increments using `move_with_collision`
    /// for 30 iterations. Assert that BY THE END:
    ///   (a) exactly one portal crossing occurred — yaw changed by ~π/2 (the
    ///       demo A→B corner turn) and is NOT a multiple of the original yaw,
    ///   (b) position is no longer "north of spawn" in a straight line — the
    ///       re-base has shifted x significantly (into the west region),
    ///   (c) a second 30-step walk from the post-teleport pose does NOT produce
    ///       another crossing (position and yaw are stable) — guards against
    ///       double-warp.
    #[test]
    fn portal_gameplay_walk_teleport_fires() {
        use std::f32::consts::PI;

        let map = portal_demo_map();

        // Spawn near the south wall facing north (−y direction).
        // Level-92-equivalent: player cell is around (6, 14, 1).
        // yaw = π → dir = (sin π, cos π) = (0, −1) → moving in −y.
        let spawn_pos = Vec3::new(6.5, 14.5, 1.5);
        let spawn_yaw: f32 = PI;
        let mut cam = CameraPose {
            pos: spawn_pos,
            yaw: spawn_yaw,
            pitch: 0.0,
        };

        // Step size: 0.5 cells per iteration in the facing direction.
        let step = 0.5_f32;

        // Walk north for 30 steps (~15 cells of travel, enough to cross y=1.0).
        for _ in 0..30 {
            let (dx, dy) = cam.dir();
            cam.move_with_collision(dx * step, dy * step, &map, PLAYER_RADIUS);
        }

        let pos_after = cam.pos;
        let yaw_after = cam.yaw;

        // (a) Yaw must have changed from π by approximately π/2.
        //     Spawn yaw = π (facing −y). Forward vector = (sin π, cos π) = (0, −1).
        //     delta_A→B maps (0,−1,0) → (+1,0,0), so new yaw = atan2(1,0) = π/2.
        //     We allow ±0.2 rad tolerance for floating-point accumulation.
        let yaw_change = normalise_yaw(yaw_after - spawn_yaw);
        assert!(
            yaw_change.abs() > 0.5 && yaw_change.abs() < PI - 0.2,
            "Expected yaw change of ~π/2 (one portal crossing); got Δyaw={yaw_change:.4} \
             (yaw_after={yaw_after:.4}, spawn_yaw={spawn_yaw:.4})"
        );

        // (a2) FACING TEST — the critical assertion the old code lacked.
        //      After crossing portal A (north wall, normal +y), the player emerges
        //      at portal B (west wall, normal +x). A SEAMLESS teleport must leave
        //      the player facing INTO the destination room — i.e. forward · (+X) > 0.
        //
        //      "Forward" in the dir() convention is (sin yaw, cos yaw).
        //      We compare only the planar forward against the west-wall normal (+x).
        //      This assertion is what would have caught the original yaw-additive bug.
        {
            let fwd_x = yaw_after.sin();
            let fwd_y = yaw_after.cos();
            // Portal B (west wall) has inward normal +x = (1, 0, 0).
            // Only the x component of forward matters for this dot product.
            let dot_dest_normal = fwd_x; // dot((fwd_x, fwd_y), (1, 0))
            assert!(
                dot_dest_normal > 0.0,
                "After portal A→B, player must face INTO the destination room \
                 (forward·dest_normal > 0). Got forward=({fwd_x:.4},{fwd_y:.4}), \
                 dot={dot_dest_normal:.4}, yaw={yaw_after:.4}. \
                 A negative dot means the player is facing the west wall — wrong exit facing."
            );
        }

        // (b) Position x must have shifted significantly — after the A→B re-base
        //     the player lands in the west region (x near 1.0), NOT at x≈6.5.
        //     A straight north walk with NO teleport would keep x≈6.5; a teleport
        //     shifts x away from spawn_x by more than 2 cells.
        let x_shift = (pos_after.x - spawn_pos.x).abs();
        assert!(
            x_shift > 2.0,
            "Expected x to shift by >2 cells after portal A→B re-base; \
             spawn_x={}, pos_x={}, shift={x_shift:.3}",
            spawn_pos.x,
            pos_after.x
        );

        // (c) A second 30-step walk from the teleported pose (facing the new yaw)
        //     must NOT trigger another crossing: yaw and position remain stable.
        let pos_before_2nd = cam.pos;
        let yaw_before_2nd = cam.yaw;
        for _ in 0..30 {
            let (dx, dy) = cam.dir();
            cam.move_with_collision(dx * step, dy * step, &map, PLAYER_RADIUS);
        }
        let yaw_change_2nd = normalise_yaw(cam.yaw - yaw_before_2nd);
        assert!(
            yaw_change_2nd.abs() < 0.3,
            "Second walk must NOT produce another crossing; Δyaw2={yaw_change_2nd:.4}"
        );
        // Position should not have jumped back or teleported again.
        let pos_drift = (cam.pos - pos_before_2nd).length();
        // Allow up to 15 cells of normal walking distance but disallow a sudden
        // multi-cell discontinuity that would indicate a second teleport.
        // A real teleport re-bases by many cells; a straight walk stays continuous.
        // We verify no sudden x-axis re-base (the portal is in x, and the second
        // walk is parallel to the new facing direction).
        let x_drift_2nd = (cam.pos.x - pos_before_2nd.x).abs();
        assert!(
            x_drift_2nd < 12.0,
            "x must not re-base during second walk (no second crossing); \
             Δx2={x_drift_2nd:.3}, total_drift={pos_drift:.3}"
        );
    }

    /// Non-portal collision unchanged: existing walls still block after Stage 2.
    /// Regression guard — all existing collision tests already cover this, but
    /// an explicit named Stage 2 test makes the requirement unambiguous.
    #[test]
    fn stage2_non_portal_collision_unchanged() {
        let map = open_room(); // 5×5×4, all walls are WallSolid
        let mut cam = mid_pose();

        // Large move in +x — blocked by east wall at x=4.
        cam.move_with_collision(10.0, 0.0, &map, PLAYER_RADIUS);
        assert!(
            cam.pos.x + PLAYER_RADIUS < 4.0 + 1e-4,
            "x={} must not pass WallSolid east wall (Stage 2 regression)",
            cam.pos.x
        );

        // Large move in +y — blocked by south wall at y=4.
        let mut cam2 = mid_pose();
        cam2.move_with_collision(0.0, 10.0, &map, PLAYER_RADIUS);
        assert!(
            cam2.pos.y + PLAYER_RADIUS < 4.0 + 1e-4,
            "y={} must not pass WallSolid south wall (Stage 2 regression)",
            cam2.pos.y
        );
    }

    // -----------------------------------------------------------------------
    // M3a: Shape storage + shape-aware collision tests.
    // -----------------------------------------------------------------------

    // --- Back-compat: all-Full world is byte-identical ---

    /// A freshly created Map has no non-Full shapes.
    #[test]
    fn m3a_shape_store_defaults_all_full() {
        let m = test_room();
        for z in 0..4_i32 {
            for y in 0..6_i32 {
                for x in 0..6_i32 {
                    assert_eq!(
                        m.shape_at(x, y, z),
                        Shape::Full,
                        "fresh map must have Full shapes at ({x},{y},{z})"
                    );
                }
            }
        }
    }

    /// Out-of-bounds shape_at returns Full (no panic, no corrupt result).
    #[test]
    fn m3a_shape_at_oob_is_full() {
        let m = test_room();
        assert_eq!(m.shape_at(-1, 0, 0), Shape::Full);
        assert_eq!(m.shape_at(99, 0, 0), Shape::Full);
        assert_eq!(m.shape_at(0, 0, 99), Shape::Full);
    }

    /// set_shape + shape_at round-trip for every Shape variant.
    #[test]
    fn m3a_shape_roundtrip_all_variants() {
        let mut m = test_room();
        let cases: &[(usize, usize, usize, Shape)] = &[
            (3, 3, 1, Shape::WedgeXY(DiagXY::NE)),
            (3, 3, 1, Shape::WedgeXY(DiagXY::NW)),
            (3, 3, 1, Shape::WedgeXY(DiagXY::SE)),
            (3, 3, 1, Shape::WedgeXY(DiagXY::SW)),
            (3, 3, 1, Shape::RampZ(RampDir::PosX)),
            (3, 3, 1, Shape::RampZ(RampDir::NegX)),
            (3, 3, 1, Shape::RampZ(RampDir::PosY)),
            (3, 3, 1, Shape::RampZ(RampDir::NegY)),
        ];
        for &(x, y, z, shape) in cases {
            m.set_shape(x, y, z, shape);
            assert_eq!(
                m.shape_at(x as i32, y as i32, z as i32),
                shape,
                "shape round-trip failed for {shape:?}"
            );
        }
        // Reset to Full removes the entry.
        m.set_shape(3, 3, 1, Shape::Full);
        assert_eq!(m.shape_at(3, 3, 1), Shape::Full);
    }

    /// set_shape out-of-bounds is silently ignored — no panic.
    #[test]
    fn m3a_set_shape_oob_ignored() {
        let mut m = test_room(); // 6×6×4
        m.set_shape(99, 99, 99, Shape::WedgeXY(DiagXY::NE)); // should not panic
        m.set_shape(6, 0, 0, Shape::RampZ(RampDir::PosX)); // exactly OOB
                                                           // The map is still readable.
        assert_eq!(m.shape_at(0, 0, 0), Shape::Full);
    }

    /// Back-compat: collision in an all-Full world works byte-identically to
    /// before M3a.  We run the same move-into-wall test that was added in Stage 2.
    #[test]
    fn m3a_backcompat_full_world_collision_unchanged() {
        // All cells are Full; no shape is set.  The collision must behave
        // identically to the pre-shape code — east wall blocks at x=4.
        let map = open_room();
        let mut cam = mid_pose(); // starts at x=2.5
        cam.move_with_collision(10.0, 0.0, &map, PLAYER_RADIUS);
        assert!(
            cam.pos.x + PLAYER_RADIUS <= 4.0 + 1e-4,
            "all-Full world: x={} must not enter east WallSolid at x=4",
            cam.pos.x
        );
    }

    /// Back-compat: gravity in an all-Full world works byte-identically to
    /// before M3a.
    #[test]
    fn m3a_backcompat_full_world_gravity_unchanged() {
        let map = open_room();
        let mut cam = CameraPose {
            pos: Vec3::new(2.5, 2.5, 2.5),
            yaw: 0.0,
            pitch: 0.0,
        };
        let mut motion = PlayerMotion {
            vz: 0.0,
            grounded: false,
            ..Default::default()
        };
        for _ in 0..200 {
            cam.step_gravity(&mut motion, 0.01, &map);
        }
        assert!(
            (cam.pos.z - 1.5).abs() < 0.01,
            "all-Full world: pos.z={} should be 1.5 after landing",
            cam.pos.z
        );
        assert!(
            motion.grounded,
            "should be grounded after landing on Full floor"
        );
    }

    // --- Helper: 7×7×4 room with an interior cell at (3,3,1) set to a given shape ---

    /// Build a 7×7×4 open room (perimeter walls, Floor z=0, Ceiling z=3, air z=1..=2).
    /// Set the interior cell at (3,3,1) to `kind` (a WallSolid) with `shape`.
    /// The player starts at x=1.5, y=3.5, z=1.5, facing +x.
    fn shape_room(kind: CellKind, shape: Shape) -> (Map, CameraPose) {
        let spec = GeometrySpec {
            size: (7, 7, 4),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[],
        };
        let mut map = stamp_geometry(&spec);
        // Place the interior wedge/ramp cell at (3,3,1).
        map.set(3, 3, 1, kind);
        map.set_shape(3, 3, 1, shape);
        let cam = CameraPose {
            pos: Vec3::new(1.5, 3.5, 1.5),
            yaw: 0.0,
            pitch: 0.0,
        };
        (map, cam)
    }

    // --- WedgeXY NE: solid where x+y≥1 (in local coords) ---
    //
    // Interior cell at (3,3,1).  The probe convention for move_with_collision:
    //   new_x + dx.signum()*radius = (new_x + RADIUS) at the leading X edge.
    // Starting at x=1.5, y=3.9, z=1.5.  Use dx=2.0 → new_x=3.5.
    //   Probe X at px=3.5+0.25=3.75 (cell x=3), py=3.9±0.25.
    //   local = (3.75-3, 3.9±0.25-3, ...) = (0.75, 0.65 or 0.90, ...).
    //   NE check: x+y=0.75+0.65=1.40 ≥ 1 → SOLID → blocked.
    //
    // For the pass-through test at y=3.1 (local y near 0.1):
    //   py=3.1-0.25=2.85 (floor=2, cell 2, not cell 3) or 3.1+0.25=3.35 (cell 3).
    //   Probe at cell 3, local y=0.35: x+y=0.75+0.35=1.10 ≥ 1 → SOLID still blocked!
    // Actually even at y=3.1 (local y≈0.1), the probe that samples y+RADIUS = 3.35
    // gives local y=0.35, and x+y=0.75+0.35=1.10 ≥ 1 → still solid!
    //
    // Correct strategy: to test the AIR half of NE (x+y < 1), we need local x + local y < 1.
    // With RADIUS=0.25 and the probe at local x=0.75:
    //   need local y < 0.25 → probe y-coords (y±RADIUS) must both be in cell 3 with local < 0.25.
    //   py−RADIUS must be ≥ 3.0 (in cell 3) and py+RADIUS's local y < 0.25 → py+RADIUS < 3.25.
    //   So probe range: [py-0.25, py+0.25] ⊂ [3.0, 3.25] → py ∈ (3.25, 3.25)? Impossible.
    //   The probe samples y at (self.pos.y + oy) for oy in [-RADIUS, +RADIUS]. Both probe points
    //   must be in the AIR half simultaneously. With local x=0.75:
    //   y-RADIUS probe: local_y_low = (pos_y - RADIUS) - 3 must satisfy 0.75 + local_y_low < 1.0
    //     → local_y_low < 0.25 → pos_y - 0.25 < 3.25 → pos_y < 3.5.
    //   y+RADIUS probe: local_y_high = (pos_y + RADIUS) - 3 must satisfy 0.75 + local_y_high < 1.0
    //     → local_y_high < 0.25 → pos_y + 0.25 < 3.25 → pos_y < 3.0.
    //   So BOTH probes only pass through when pos_y < 3.0 — but cell 3 starts at y=3.
    //   If pos_y < 3.0, the y+RADIUS probe at pos_y + 0.25 may be in cell 2 (if pos_y < 2.75),
    //   or cell 3 with local_y < 0.25 (if 2.75 ≤ pos_y < 3.0), but the -RADIUS probe is in cell 2.
    //
    // Let's put y=2.9: probe y-RADIUS=2.65 (cell 2, not cell 3) → cell_at(x=3,y=2,z=1) = Empty.
    //                   probe y+RADIUS=3.15 (cell 3, local_y=0.15): x+y=0.75+0.15=0.90 < 1 → AIR!
    // So with pos_y=2.9, the x-axis movement only needs BOTH probes clear:
    //   probe 1: cell (3, 2, 1) = Empty → not collidable → ok
    //   probe 2: cell (3, 3, 1) = WallSolid w/ WedgeXY(NE), local=(0.75, 0.15, ..) → 0.90 < 1 → AIR
    // → BOTH clear → x-axis move succeeds → player passes through!
    //
    // For the SOLID test: with pos_y=3.7:
    //   probe y-RADIUS=3.45 (cell 3, local_y=0.45): x+y=0.75+0.45=1.20 ≥ 1 → SOLID → blocked!

    #[test]
    fn m3a_wedge_ne_blocks_solid_half() {
        // Player at y=3.7 (probe hits solid half of NE: local x=0.75, local y≈0.45 → 1.20≥1).
        // Use dx=2.0 so new_x=1.5+2.0=3.5, probe at 3.75 (cell 3, local x=0.75).
        let (map, mut cam) = shape_room(CellKind::WallSolid, Shape::WedgeXY(DiagXY::NE));
        cam.pos.y = 3.7;
        let x_before = cam.pos.x; // 1.5
                                  // A medium move toward cell 3 — the probe should hit the solid half.
        cam.move_with_collision(2.0, 0.0, &map, PLAYER_RADIUS);
        // Must be blocked — must not reach cell 3 (probe was solid).
        assert!(
            cam.pos.x + PLAYER_RADIUS <= 3.0 + 1e-4,
            "WedgeXY NE solid half: x={} must not enter cell 3 at y=3.7",
            cam.pos.x
        );
        // The player must have NOT moved (collision blocked the entire step because
        // new_x=3.5 probe at cell 3 was solid).
        let _ = x_before; // not asserting partial move — move_with_collision is all-or-nothing
    }

    #[test]
    fn m3a_wedge_ne_passes_empty_half() {
        // Player at y=2.9. Probe y+RADIUS=3.15 → cell 3, local_y=0.15.
        // At leading X probe (local x=0.75): 0.75+0.15=0.90 < 1 → AIR.
        // Probe y-RADIUS=2.65 → cell 2 (Empty) → not blocked.
        // So BOTH probes are clear → player passes through cell 3.
        let (map, mut cam) = shape_room(CellKind::WallSolid, Shape::WedgeXY(DiagXY::NE));
        cam.pos.y = 2.9;
        // Move dx=2.0 → new_x=3.5 (probe at 3.75, cell 3, local_x=0.75).
        cam.move_with_collision(2.0, 0.0, &map, PLAYER_RADIUS);
        // The player should now be at x=3.5 (moved through the air half).
        assert!(
            cam.pos.x > 3.0,
            "WedgeXY NE empty half: x={} must pass through cell 3 at y=2.9",
            cam.pos.x
        );
    }

    // --- WedgeXY SW: complement of NE — solid where x+y≤1 ---
    // At y=2.9 (probe y+RADIUS→cell 3, local_y=0.15): x+y=0.75+0.15=0.90 ≤ 1 → SOLID.
    // At y=3.7 (probe y-RADIUS→cell 3, local_y=0.45): x+y=0.75+0.45=1.20 > 1 → AIR.

    #[test]
    fn m3a_wedge_sw_blocks_solid_half() {
        let (map, mut cam) = shape_room(CellKind::WallSolid, Shape::WedgeXY(DiagXY::SW));
        cam.pos.y = 2.9; // local y≈0.15 via probe → x+y=0.90 ≤ 1 → SOLID
        cam.move_with_collision(2.0, 0.0, &map, PLAYER_RADIUS);
        assert!(
            cam.pos.x + PLAYER_RADIUS <= 3.0 + 1e-4,
            "WedgeXY SW solid half: x={} must be blocked at y=2.9",
            cam.pos.x
        );
    }

    #[test]
    fn m3a_wedge_sw_passes_empty_half() {
        let (map, mut cam) = shape_room(CellKind::WallSolid, Shape::WedgeXY(DiagXY::SW));
        cam.pos.y = 3.7; // local y≈0.45 via probe → x+y=1.20 > 1 → AIR
        cam.move_with_collision(2.0, 0.0, &map, PLAYER_RADIUS);
        assert!(
            cam.pos.x > 3.0,
            "WedgeXY SW empty half: x={} must pass through cell 3 at y=3.7",
            cam.pos.x
        );
    }

    // --- RampZ horizontal collision: ramps are WALKABLE from ALL directions ---
    //
    // Cell (3,3,1). RampZ cells are NOT blocking walls — the player can enter
    // any ramp cell footprint from any horizontal direction, regardless of their
    // current eye-z vs the slope position. `step_gravity` (ramp-snap) handles
    // vertical positioning on the slope.
    //
    // This is the key fix that makes all-4-direction staircases and spiral
    // staircases traversable: previously the shape_solid_contains check would
    // block a player whose eye-level z was above the ramp surface at the probe
    // local_x/local_y, which is exactly what happens when approaching from any
    // direction other than the ramp's own ascending axis.

    /// RampZ(PosX) is NOT a horizontal wall — the player can step through both
    /// the "solid" half and the "air" half of the ramp footprint.
    #[test]
    fn m3a_ramp_posx_walkable_from_any_direction() {
        let spec = GeometrySpec {
            size: (7, 7, 4),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[],
        };
        let mut map = stamp_geometry(&spec);
        map.set(3, 3, 1, CellKind::WallSolid);
        map.set_shape(3, 3, 1, Shape::RampZ(RampDir::PosX));

        // Large +X move: would have hit the "solid half" under the old code.
        // Now must pass through — the ramp is walkable.
        let mut cam = CameraPose {
            pos: Vec3::new(1.5, 3.5, 1.5),
            yaw: 0.0,
            pitch: 0.0,
        };
        cam.move_with_collision(2.0, 0.0, &map, PLAYER_RADIUS);
        assert!(
            cam.pos.x > 3.0,
            "RampZ PosX: x={} must pass through ramp footprint (ramps are walkable)",
            cam.pos.x
        );
    }

    #[test]
    fn m3a_ramp_posx_small_step_also_walkable() {
        // dx=1.6 → new_x=3.1, probe in ramp cell: also walkable.
        let spec = GeometrySpec {
            size: (7, 7, 4),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[],
        };
        let mut map = stamp_geometry(&spec);
        map.set(3, 3, 1, CellKind::WallSolid);
        map.set_shape(3, 3, 1, Shape::RampZ(RampDir::PosX));
        let mut cam = CameraPose {
            pos: Vec3::new(1.5, 3.5, 1.5),
            yaw: 0.0,
            pitch: 0.0,
        };
        cam.move_with_collision(1.6, 0.0, &map, PLAYER_RADIUS);
        assert!(
            cam.pos.x > 3.0,
            "RampZ PosX small step: x={} must enter ramp cell",
            cam.pos.x
        );
    }

    // --- RampZ gravity: standing on the slope surface ---
    //
    // A RampZ(PosX) cell is WallSolid with the ramp shape.  If the player is
    // dropped onto it from above, step_gravity should land them at the slope
    // surface z at their x position, not at the cell top (which would be 1 cell
    // above the floor).

    #[test]
    fn m3a_ramp_gravity_snaps_to_slope_surface() {
        // Build a 7×7×4 room with a RampZ(PosX) cell at (3,3,1).
        // The slope is z=x in local coords.  At world x≈3.7, local x≈0.7:
        // the slope surface is at world z = 1.0 + 0.7 = 1.7.
        // The eye should rest at 1.7 + PLAYER_EYE_HEIGHT = 1.7 + 0.5 = 2.2.
        let spec = GeometrySpec {
            size: (7, 7, 4),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[],
        };
        let mut map = stamp_geometry(&spec);
        map.set(3, 3, 1, CellKind::WallSolid);
        map.set_shape(3, 3, 1, Shape::RampZ(RampDir::PosX));
        // Also place floors adjacent so the player stays in bounds.
        // (The room shell already has floor at z=0.)

        // Drop from above the ramp cell at x=3.7 (local x=0.7).
        let mut cam = CameraPose {
            pos: Vec3::new(3.7, 3.5, 2.8),
            yaw: 0.0,
            pitch: 0.0,
        };
        let mut motion = PlayerMotion {
            vz: 0.0,
            grounded: false,
            ..Default::default()
        };

        // Simulate enough time to fall and land.
        for _ in 0..300 {
            cam.step_gravity(&mut motion, 0.01, &map);
            if motion.grounded {
                break;
            }
        }

        assert!(motion.grounded, "player must land on the ramp surface");
        // Eye z should be approximately slope_z + PLAYER_EYE_HEIGHT = 1.7 + 0.5 = 2.2.
        let expected_eye_z = 1.0 + 0.7 + PLAYER_EYE_HEIGHT; // cell_base(1.0) + local_slope_z(0.7) + eye_height
        assert!(
            (cam.pos.z - expected_eye_z).abs() < 0.05,
            "RampZ PosX: eye z={} should rest at slope surface z≈{expected_eye_z}",
            cam.pos.z
        );
    }

    // --- Shape accessor: non-Full cells have the right shape, others stay Full ---

    #[test]
    fn m3a_shape_only_set_where_written() {
        let mut m = test_room(); // 6×6×4
                                 // Set a wedge at interior cell (3,3,1).
        m.set_shape(3, 3, 1, Shape::WedgeXY(DiagXY::NW));
        // Verify it's there.
        assert_eq!(m.shape_at(3, 3, 1), Shape::WedgeXY(DiagXY::NW));
        // All other interior air cells remain Full.
        assert_eq!(m.shape_at(3, 4, 1), Shape::Full);
        assert_eq!(m.shape_at(4, 3, 1), Shape::Full);
        assert_eq!(m.shape_at(3, 3, 2), Shape::Full);
        // Perimeter walls are unaffected.
        assert_eq!(m.shape_at(0, 0, 1), Shape::Full);
    }

    // -----------------------------------------------------------------------
    // Staircase / multi-floor ramp tests (spec §4 §5 §7, seam fix).
    // -----------------------------------------------------------------------

    /// Build a straight staircase map of `h` consecutive `RampZ(PosX)` cells
    /// rising along +X.
    ///
    /// Layout (y=2 is the staircase lane):
    ///   - Ramp cell `n` is at world position `(n, 2, n)` for `n = 1..=h`.
    ///   - Each ramp cell is stamped as `CellKind::WallSolid` (so it is solid)
    ///     with `Shape::RampZ(RampDir::PosX)`.
    ///   - Map size: `(h+3) × 5 × (h+3)`.
    ///     * Perimeter walls + floor (z=0) + ceiling (z=h+2) from `stamp_geometry`.
    ///     * Interior air surrounds the staircase.
    ///
    /// The player starts near `x=1.1, z≈1.6` (base of the first ramp) and is
    /// expected to rise to `z≈(h+1.0)+0.5` at `x=h+0.9` (top of the last ramp).
    fn staircase_map(h: usize) -> Map {
        let sz = h + 3;
        let spec = GeometrySpec {
            size: (h + 3, 5, sz),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[],
        };
        let mut map = stamp_geometry(&spec);
        // Stamp each ramp cell.
        for n in 1..=h {
            map.set(n, 2, n, CellKind::WallSolid);
            map.set_shape(n, 2, n, Shape::RampZ(RampDir::PosX));
        }
        map
    }

    /// Same as `staircase_map` but adds a solid wall cell directly above
    /// a given ramp cell `(cx, 2, cz)` to test headroom blocking.
    ///
    /// Uses `CellKind::WallSolid` (not `CellKind::Ceiling`) because `is_solid()`
    /// — the predicate the physics ceiling check uses — only matches the wall
    /// variants and Object/Portal, not the floor/ceiling slab variants.  A
    /// `WallSolid` block above the stairs acts as a low beam the player's head
    /// cannot enter.
    fn staircase_map_with_ceiling(h: usize, cx: usize, cz: usize) -> Map {
        let mut map = staircase_map(h);
        // Stamp a solid block one cell above the ramp.
        map.set(cx, 2, cz, CellKind::WallSolid);
        // (Shape stays Full — a flat solid block.)
        map
    }

    /// Walking a full straight run of H `RampZ(PosX)` cells raises the camera
    /// eye z by exactly H (within epsilon) and `grounded` stays `true` across
    /// every seam.
    ///
    /// Simulates: start at `(x=1.1, y=2.5)` on the base of ramp 1, then
    /// step in increments of 0.05 cells in +X, calling `step_gravity` at each
    /// step. Checks grounded at every seam crossing and measures total rise.
    #[test]
    fn staircase_ascending_raises_z_and_stays_grounded() {
        const H: usize = 4;
        let map = staircase_map(H);

        // Start near the base of the first ramp: x=1.1, surface_z = 1+0.1 = 1.1.
        // Eye starts at 1.1 + PLAYER_EYE_HEIGHT.
        let start_x: f32 = 1.1;
        let start_surface_z: f32 = 1.0 + (start_x - 1.0); // ramp cell 1: surface = 1 + local_x
        let start_eye_z = start_surface_z + PLAYER_EYE_HEIGHT;

        let mut cam = CameraPose {
            pos: Vec3::new(start_x, 2.5, start_eye_z),
            yaw: 0.0,
            pitch: 0.0,
        };
        let mut motion = PlayerMotion {
            vz: 0.0,
            grounded: false,
            ..Default::default()
        };

        // Settle on the first ramp surface.
        for _ in 0..10 {
            cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
        }
        assert!(motion.grounded, "must start grounded on ramp 1");

        // Walk in +X steps of 0.05, stepping gravity each time.
        // Track that grounded stays true after each seam crossing.
        let x_end: f32 = (H as f32) + 0.9; // near top of last ramp
        let step_x: f32 = 0.05;
        let mut seam_grounded_failures: Vec<f32> = Vec::new();

        let mut x = start_x;
        while x < x_end {
            x = (x + step_x).min(x_end);
            cam.pos.x = x;
            // Settle for 3 frames per horizontal step (physics dt = 1/60 s).
            for _ in 0..3 {
                cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
            }
            // Seam positions are at integer x values; check grounded near each seam.
            let local_x = x - x.floor();
            let near_seam = !(0.1..=0.9).contains(&local_x);
            if near_seam && !motion.grounded {
                seam_grounded_failures.push(x);
            }
        }

        assert!(
            seam_grounded_failures.is_empty(),
            "grounded flickered at seams: {seam_grounded_failures:?}"
        );
        assert!(
            motion.grounded,
            "must still be grounded at top of staircase"
        );

        // The expected eye z at x=4.9 (near top of ramp 4):
        //   surface = ramp_cell_z + local_x = 4 + 0.9 = 4.9
        //   eye     = 4.9 + PLAYER_EYE_HEIGHT
        let expected_eye_z = (H as f32) + 0.9 + PLAYER_EYE_HEIGHT;
        assert!(
            (cam.pos.z - expected_eye_z).abs() < 0.15,
            "eye z={:.4} should be ≈{expected_eye_z:.4} at top of staircase (H={H})",
            cam.pos.z
        );

        // Total rise ≈ H cells.
        let rise = cam.pos.z - start_eye_z;
        let expected_rise = (H as f32) - 0.2; // rough: from x=1.1 to x=4.9
        assert!(
            rise > expected_rise - 0.2 && rise < expected_rise + 0.2,
            "total rise={rise:.4} should be ≈{expected_rise:.4} (H={H})"
        );
    }

    /// Descending a straight run of H `RampZ(PosX)` cells lowers the camera
    /// eye z by ~H and `grounded` stays `true` across every seam.
    #[test]
    fn staircase_descending_lowers_z_and_stays_grounded() {
        const H: usize = 4;
        let map = staircase_map(H);

        // Start near the top of the last ramp: x = H+0.9.
        // Start near top of last ramp: ramp H is at (H, 2, H).
        // x=H+0.9 → cell H, local_x=0.9, surface = H+0.9.
        let start_x: f32 = (H as f32) + 0.9;
        let start_surface_z: f32 = (H as f32) + 0.9;
        let start_eye_z = start_surface_z + PLAYER_EYE_HEIGHT;

        let mut cam = CameraPose {
            pos: Vec3::new(start_x, 2.5, start_eye_z),
            yaw: 0.0,
            pitch: 0.0,
        };
        let mut motion = PlayerMotion {
            vz: 0.0,
            grounded: false,
            ..Default::default()
        };

        // Settle on the last ramp surface.
        for _ in 0..10 {
            cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
        }
        assert!(motion.grounded, "must start grounded on top ramp");

        // Walk in -X steps of 0.05.
        let x_end: f32 = 1.1; // near base of ramp 1
        let step_x: f32 = 0.05;
        let mut seam_grounded_failures: Vec<f32> = Vec::new();

        let mut x = start_x;
        while x > x_end {
            x = (x - step_x).max(x_end);
            cam.pos.x = x;
            for _ in 0..3 {
                cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
            }
            let local_x = x - x.floor();
            let near_seam = !(0.1..=0.9).contains(&local_x);
            if near_seam && !motion.grounded {
                seam_grounded_failures.push(x);
            }
        }

        assert!(
            seam_grounded_failures.is_empty(),
            "grounded flickered during descent at seams: {seam_grounded_failures:?}"
        );
        assert!(
            motion.grounded,
            "must still be grounded at base of staircase"
        );

        // Expected eye z near base of ramp 1: surface = 1 + 0.1 = 1.1, eye = 1.6.
        let expected_eye_z = 1.0 + 0.1 + PLAYER_EYE_HEIGHT;
        assert!(
            (cam.pos.z - expected_eye_z).abs() < 0.15,
            "eye z={:.4} should be ≈{expected_eye_z:.4} at base of staircase (H={H})",
            cam.pos.z
        );
    }

    /// A solid ceiling cell directly above a ramp blocks ascent at the correct
    /// height (headroom on stairs).
    ///
    /// Layout: 3-ramp staircase, solid ceiling at `(2, 2, 4)` (one cell above
    /// ramp 2 at `(2, 2, 2)`). When the player tries to walk up ramp 2 their
    /// head (eye + HEAD_ROOM) will hit the ceiling cell at z=4, clamping the
    /// eye to `4 - HEAD_ROOM = 3.7`.
    #[test]
    fn staircase_ascending_blocked_by_low_ceiling() {
        const H: usize = 3;
        // Place a ceiling block at (2, 2, H+1) — directly above ramp H (the top ramp).
        // Ramp H is at (H, 2, H). One cell above = (H, 2, H+1).
        let map = staircase_map_with_ceiling(H, H, H + 1);

        // Start at base of ramp 1.
        let start_x: f32 = 1.1;
        let start_eye_z = 1.0 + 0.1 + PLAYER_EYE_HEIGHT;
        let mut cam = CameraPose {
            pos: Vec3::new(start_x, 2.5, start_eye_z),
            yaw: 0.0,
            pitch: 0.0,
        };
        let mut motion = PlayerMotion {
            vz: 0.0,
            grounded: false,
            ..Default::default()
        };
        for _ in 0..10 {
            cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
        }
        assert!(motion.grounded, "must start grounded");

        // Walk up all the way to ramp H (the blocked ramp) at x = H + 0.5.
        // The ceiling is at z = H+1; head = eye + HEAD_ROOM; block when head ≥ H+1.
        // Maximum allowed eye z = (H+1) - HEAD_ROOM.
        let x_target: f32 = (H as f32) + 0.5;
        let mut x = start_x;
        while x < x_target {
            x = (x + 0.05_f32).min(x_target);
            cam.pos.x = x;
            for _ in 0..3 {
                cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
            }
        }

        // The ceiling is the cell at (H, 2, H+1) = solid.
        // head_z ceiling = (H+1) as f32. Max eye = (H+1) as f32 - HEAD_ROOM.
        let max_eye_z = (H + 1) as f32 - PLAYER_HEAD_ROOM;
        assert!(
            cam.pos.z <= max_eye_z + 0.05,
            "eye z={:.4} exceeded ceiling limit {max_eye_z:.4} (H={H})",
            cam.pos.z
        );
    }

    /// Regression: walking on a flat floor (not a ramp) stays grounded and at
    /// exactly the same eye z — the ramp-snap pass must not disturb flat-floor
    /// behaviour.
    #[test]
    fn flat_floor_stays_grounded_and_stable() {
        let map = open_room(); // 5×5×4, floor at z=0
        let mut cam = CameraPose {
            pos: Vec3::new(2.5, 2.5, 1.5), // eye at standard standing height
            yaw: 0.0,
            pitch: 0.0,
        };
        let mut motion = PlayerMotion {
            vz: 0.0,
            grounded: true,
            ..Default::default()
        };

        let initial_z = cam.pos.z;
        for step in 0..60 {
            cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
            assert!(
                motion.grounded,
                "flat floor must stay grounded at step {step}, pos.z={}",
                cam.pos.z
            );
            assert!(
                (cam.pos.z - initial_z).abs() < 0.02,
                "flat floor must not drift: pos.z={} vs initial {initial_z} at step {step}",
                cam.pos.z
            );
        }
    }

    // -----------------------------------------------------------------------
    // All-4-direction staircase tests: NegX, PosY, NegY (PosX covered above).
    // -----------------------------------------------------------------------

    /// Build a NegX staircase: H ramp cells where x DECREASES as z increases.
    /// Cell n (1-indexed) sits at (H+1-n, LANE, n).  The ramp surface is
    /// `NegX`: solid where `z ≤ 1−x`, meaning the high end is at local_x=0.
    /// The player walks in −X from x=H+0.9 toward x=1.1.
    fn staircase_map_negx(h: usize) -> Map {
        let sz = h + 3;
        let spec = GeometrySpec {
            size: (h + 3, 5, sz),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[],
        };
        let mut map = stamp_geometry(&spec);
        for n in 1..=h {
            // Ramp cell n: x decreases, z = n.
            let cx = h + 1 - n; // cx goes h, h-1, …, 1
            map.set(cx, 2, n, CellKind::WallSolid);
            map.set_shape(cx, 2, n, Shape::RampZ(RampDir::NegX));
        }
        map
    }

    /// Build a PosY staircase: H ramp cells along +Y axis.
    /// Cell n at (LANE, n, n).  Surface `PosY`: rises as y increases.
    /// Player walks in +Y from y=1.1 toward y=H+0.9.
    fn staircase_map_posy(h: usize) -> Map {
        let sz = h + 3;
        let spec = GeometrySpec {
            size: (5, h + 3, sz),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[],
        };
        let mut map = stamp_geometry(&spec);
        for n in 1..=h {
            map.set(2, n, n, CellKind::WallSolid);
            map.set_shape(2, n, n, Shape::RampZ(RampDir::PosY));
        }
        map
    }

    /// Build a NegY staircase: H ramp cells where y DECREASES as z increases.
    /// Cell n at (LANE, H+1-n, n).  Surface `NegY`: rises as y decreases.
    /// Player walks in −Y from y=H+0.9 toward y=1.1.
    fn staircase_map_negy(h: usize) -> Map {
        let sz = h + 3;
        let spec = GeometrySpec {
            size: (5, h + 3, sz),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[],
        };
        let mut map = stamp_geometry(&spec);
        for n in 1..=h {
            let cy = h + 1 - n;
            map.set(2, cy, n, CellKind::WallSolid);
            map.set_shape(2, cy, n, Shape::RampZ(RampDir::NegY));
        }
        map
    }

    /// Walk a straight NegX staircase ascending: eye z rises ~H, grounded.
    #[test]
    fn staircase_ascending_negx_raises_z_and_stays_grounded() {
        const H: usize = 4;
        let map = staircase_map_negx(H);

        // Start near the base of ramp 1: x = H+0.9, ramp cell (H, 2, 1).
        // NegX surface at local_x=0.9 → height = 1 + (1-0.9) = 1.1.
        let start_x: f32 = (H as f32) + 0.9;
        let start_surface_z: f32 = 1.0 + (1.0 - 0.9); // cell z=1, local_x=0.9
        let start_eye_z = start_surface_z + PLAYER_EYE_HEIGHT;

        let mut cam = CameraPose {
            pos: Vec3::new(start_x, 2.5, start_eye_z),
            yaw: 0.0,
            pitch: 0.0,
        };
        let mut motion = PlayerMotion {
            vz: 0.0,
            grounded: false,
            ..Default::default()
        };

        // Settle on first ramp.
        for _ in 0..10 {
            cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
        }
        assert!(motion.grounded, "NegX: must start grounded on ramp 1");

        // Walk in -X steps of 0.05.
        let x_end: f32 = 1.1; // near top of last ramp (x=1, local_x=0.1 → height=1.9)
        let step_x: f32 = 0.05;
        let mut x = start_x;
        let mut seam_grounded_failures: Vec<f32> = Vec::new();

        while x > x_end {
            x = (x - step_x).max(x_end);
            cam.pos.x = x;
            for _ in 0..3 {
                cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
            }
            let local_x = x - x.floor();
            let near_seam = !(0.1..=0.9).contains(&local_x);
            if near_seam && !motion.grounded {
                seam_grounded_failures.push(x);
            }
        }

        assert!(
            seam_grounded_failures.is_empty(),
            "NegX staircase grounded flickered at seams: {seam_grounded_failures:?}"
        );
        assert!(
            motion.grounded,
            "NegX: must be grounded at top of staircase"
        );

        // Expected eye z at x=1.1: ramp cell (1,2,H), local_x=0.1 → surface = H+(1-0.1) = H+0.9.
        let expected_eye_z = (H as f32) + 0.9 + PLAYER_EYE_HEIGHT;
        assert!(
            (cam.pos.z - expected_eye_z).abs() < 0.15,
            "NegX: eye z={:.4} should be ≈{expected_eye_z:.4} at top",
            cam.pos.z
        );
    }

    /// Walk a straight NegX staircase descending.
    #[test]
    fn staircase_descending_negx_lowers_z_and_stays_grounded() {
        const H: usize = 4;
        let map = staircase_map_negx(H);

        // Start at the top: x=1.1, ramp cell (1,2,H), local_x=0.1 → surface=H+0.9.
        let start_x: f32 = 1.1;
        let start_surface_z: f32 = (H as f32) + (1.0 - 0.1);
        let start_eye_z = start_surface_z + PLAYER_EYE_HEIGHT;

        let mut cam = CameraPose {
            pos: Vec3::new(start_x, 2.5, start_eye_z),
            yaw: 0.0,
            pitch: 0.0,
        };
        let mut motion = PlayerMotion {
            vz: 0.0,
            grounded: false,
            ..Default::default()
        };
        for _ in 0..10 {
            cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
        }
        assert!(motion.grounded, "NegX descent: must start grounded");

        let x_end: f32 = (H as f32) + 0.9;
        let step_x: f32 = 0.05;
        let mut x = start_x;
        let mut seam_grounded_failures: Vec<f32> = Vec::new();

        while x < x_end {
            x = (x + step_x).min(x_end);
            cam.pos.x = x;
            for _ in 0..3 {
                cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
            }
            let local_x = x - x.floor();
            let near_seam = !(0.1..=0.9).contains(&local_x);
            if near_seam && !motion.grounded {
                seam_grounded_failures.push(x);
            }
        }

        assert!(
            seam_grounded_failures.is_empty(),
            "NegX descent grounded flickered: {seam_grounded_failures:?}"
        );
        assert!(
            motion.grounded,
            "NegX descent: must still be grounded at base"
        );

        // At x=H+0.9: ramp cell (H,2,1), local_x=0.9 → surface=1+(1-0.9)=1.1.
        let expected_eye_z = 1.0 + (1.0 - 0.9) + PLAYER_EYE_HEIGHT;
        assert!(
            (cam.pos.z - expected_eye_z).abs() < 0.15,
            "NegX descent: eye z={:.4} should be ≈{expected_eye_z:.4} at base",
            cam.pos.z
        );
    }

    /// Walk a straight PosY staircase ascending.
    #[test]
    fn staircase_ascending_posy_raises_z_and_stays_grounded() {
        const H: usize = 4;
        let map = staircase_map_posy(H);

        // Start near y=1.1, ramp cell (2,1,1), local_y=0.1 → surface=1+0.1=1.1.
        let start_y: f32 = 1.1;
        let start_surface_z: f32 = 1.0 + (start_y - 1.0);
        let start_eye_z = start_surface_z + PLAYER_EYE_HEIGHT;

        let mut cam = CameraPose {
            pos: Vec3::new(2.5, start_y, start_eye_z),
            yaw: 0.0,
            pitch: 0.0,
        };
        let mut motion = PlayerMotion {
            vz: 0.0,
            grounded: false,
            ..Default::default()
        };
        for _ in 0..10 {
            cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
        }
        assert!(motion.grounded, "PosY: must start grounded on ramp 1");

        let y_end: f32 = (H as f32) + 0.9;
        let step_y: f32 = 0.05;
        let mut y = start_y;
        let mut seam_grounded_failures: Vec<f32> = Vec::new();

        while y < y_end {
            y = (y + step_y).min(y_end);
            cam.pos.y = y;
            for _ in 0..3 {
                cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
            }
            let local_y = y - y.floor();
            let near_seam = !(0.1..=0.9).contains(&local_y);
            if near_seam && !motion.grounded {
                seam_grounded_failures.push(y);
            }
        }

        assert!(
            seam_grounded_failures.is_empty(),
            "PosY staircase grounded flickered at seams: {seam_grounded_failures:?}"
        );
        assert!(
            motion.grounded,
            "PosY: must be grounded at top of staircase"
        );

        let expected_eye_z = (H as f32) + 0.9 + PLAYER_EYE_HEIGHT;
        assert!(
            (cam.pos.z - expected_eye_z).abs() < 0.15,
            "PosY: eye z={:.4} should be ≈{expected_eye_z:.4} at top",
            cam.pos.z
        );
    }

    /// Walk a straight PosY staircase descending.
    #[test]
    fn staircase_descending_posy_lowers_z_and_stays_grounded() {
        const H: usize = 4;
        let map = staircase_map_posy(H);

        let start_y: f32 = (H as f32) + 0.9;
        let start_surface_z: f32 = (H as f32) + 0.9;
        let start_eye_z = start_surface_z + PLAYER_EYE_HEIGHT;

        let mut cam = CameraPose {
            pos: Vec3::new(2.5, start_y, start_eye_z),
            yaw: 0.0,
            pitch: 0.0,
        };
        let mut motion = PlayerMotion {
            vz: 0.0,
            grounded: false,
            ..Default::default()
        };
        for _ in 0..10 {
            cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
        }
        assert!(motion.grounded, "PosY descent: must start grounded");

        let y_end: f32 = 1.1;
        let step_y: f32 = 0.05;
        let mut y = start_y;
        let mut seam_grounded_failures: Vec<f32> = Vec::new();

        while y > y_end {
            y = (y - step_y).max(y_end);
            cam.pos.y = y;
            for _ in 0..3 {
                cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
            }
            let local_y = y - y.floor();
            let near_seam = !(0.1..=0.9).contains(&local_y);
            if near_seam && !motion.grounded {
                seam_grounded_failures.push(y);
            }
        }

        assert!(
            seam_grounded_failures.is_empty(),
            "PosY descent grounded flickered: {seam_grounded_failures:?}"
        );
        assert!(
            motion.grounded,
            "PosY descent: must still be grounded at base"
        );

        let expected_eye_z = 1.0 + 0.1 + PLAYER_EYE_HEIGHT;
        assert!(
            (cam.pos.z - expected_eye_z).abs() < 0.15,
            "PosY descent: eye z={:.4} should be ≈{expected_eye_z:.4} at base",
            cam.pos.z
        );
    }

    /// Walk a straight NegY staircase ascending (y decreases as z increases).
    #[test]
    fn staircase_ascending_negy_raises_z_and_stays_grounded() {
        const H: usize = 4;
        let map = staircase_map_negy(H);

        // Start near y=H+0.9, ramp cell (2,H,1), local_y=0.9 → NegY surface=1+(1-0.9)=1.1.
        let start_y: f32 = (H as f32) + 0.9;
        let start_surface_z: f32 = 1.0 + (1.0 - 0.9);
        let start_eye_z = start_surface_z + PLAYER_EYE_HEIGHT;

        let mut cam = CameraPose {
            pos: Vec3::new(2.5, start_y, start_eye_z),
            yaw: 0.0,
            pitch: 0.0,
        };
        let mut motion = PlayerMotion {
            vz: 0.0,
            grounded: false,
            ..Default::default()
        };
        for _ in 0..10 {
            cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
        }
        assert!(motion.grounded, "NegY: must start grounded on ramp 1");

        let y_end: f32 = 1.1;
        let step_y: f32 = 0.05;
        let mut y = start_y;
        let mut seam_grounded_failures: Vec<f32> = Vec::new();

        while y > y_end {
            y = (y - step_y).max(y_end);
            cam.pos.y = y;
            for _ in 0..3 {
                cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
            }
            let local_y = y - y.floor();
            let near_seam = !(0.1..=0.9).contains(&local_y);
            if near_seam && !motion.grounded {
                seam_grounded_failures.push(y);
            }
        }

        assert!(
            seam_grounded_failures.is_empty(),
            "NegY staircase grounded flickered at seams: {seam_grounded_failures:?}"
        );
        assert!(
            motion.grounded,
            "NegY: must be grounded at top of staircase"
        );

        // At y=1.1: ramp cell (2,1,H), local_y=0.1 → NegY surface=H+(1-0.1)=H+0.9.
        let expected_eye_z = (H as f32) + 0.9 + PLAYER_EYE_HEIGHT;
        assert!(
            (cam.pos.z - expected_eye_z).abs() < 0.15,
            "NegY: eye z={:.4} should be ≈{expected_eye_z:.4} at top",
            cam.pos.z
        );
    }

    /// Walk a straight NegY staircase descending.
    #[test]
    fn staircase_descending_negy_lowers_z_and_stays_grounded() {
        const H: usize = 4;
        let map = staircase_map_negy(H);

        // Start at top: y=1.1, cell (2,1,H), local_y=0.1 → NegY surface=H+0.9.
        let start_y: f32 = 1.1;
        let start_surface_z: f32 = (H as f32) + (1.0 - 0.1);
        let start_eye_z = start_surface_z + PLAYER_EYE_HEIGHT;

        let mut cam = CameraPose {
            pos: Vec3::new(2.5, start_y, start_eye_z),
            yaw: 0.0,
            pitch: 0.0,
        };
        let mut motion = PlayerMotion {
            vz: 0.0,
            grounded: false,
            ..Default::default()
        };
        for _ in 0..10 {
            cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
        }
        assert!(motion.grounded, "NegY descent: must start grounded");

        let y_end: f32 = (H as f32) + 0.9;
        let step_y: f32 = 0.05;
        let mut y = start_y;
        let mut seam_grounded_failures: Vec<f32> = Vec::new();

        while y < y_end {
            y = (y + step_y).min(y_end);
            cam.pos.y = y;
            for _ in 0..3 {
                cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
            }
            let local_y = y - y.floor();
            let near_seam = !(0.1..=0.9).contains(&local_y);
            if near_seam && !motion.grounded {
                seam_grounded_failures.push(y);
            }
        }

        assert!(
            seam_grounded_failures.is_empty(),
            "NegY descent grounded flickered: {seam_grounded_failures:?}"
        );
        assert!(
            motion.grounded,
            "NegY descent: must still be grounded at base"
        );

        let expected_eye_z = 1.0 + (1.0 - 0.9) + PLAYER_EYE_HEIGHT;
        assert!(
            (cam.pos.z - expected_eye_z).abs() < 0.15,
            "NegY descent: eye z={:.4} should be ≈{expected_eye_z:.4} at base",
            cam.pos.z
        );
    }

    /// Ramps are horizontally non-blocking: a player can step ONTO a ramp cell
    /// from any approach direction using move_with_collision.
    #[test]
    fn ramp_walkable_from_all_approach_directions() {
        // Single ramp cell at (3, 3, 1) with each orientation.
        // Approach from the perpendicular direction at floor-level (eye_z = 1.5).
        // The perpendicular approach is the one that would have blocked before the fix:
        // e.g., PosY ramp approached from the X direction at height that would have
        // been "solid" according to the old shape_solid_contains check.
        let spec = GeometrySpec {
            size: (7, 7, 4),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[],
        };
        for dir in [RampDir::PosX, RampDir::NegX, RampDir::PosY, RampDir::NegY] {
            let mut map = stamp_geometry(&spec);
            map.set(3, 3, 1, CellKind::WallSolid);
            map.set_shape(3, 3, 1, Shape::RampZ(dir));

            // Approach in +X at y=3.5 (perpendicular to PosY/NegY ramps).
            // For PosX/NegX this is the normal walk-on direction; for PosY/NegY
            // this is the previously-blocked perpendicular approach.
            let mut cam_x = CameraPose {
                pos: Vec3::new(1.5, 3.5, 1.5),
                yaw: 0.0,
                pitch: 0.0,
            };
            cam_x.move_with_collision(2.0, 0.0, &map, PLAYER_RADIUS);
            assert!(
                cam_x.pos.x > 3.0,
                "dir={dir:?}: cannot enter ramp from +X direction (pos.x={:.3})",
                cam_x.pos.x
            );

            // Approach in +Y at x=3.5 (perpendicular to PosX/NegX ramps).
            let mut cam_y = CameraPose {
                pos: Vec3::new(3.5, 1.5, 1.5),
                yaw: 0.0,
                pitch: 0.0,
            };
            cam_y.move_with_collision(0.0, 2.0, &map, PLAYER_RADIUS);
            assert!(
                cam_y.pos.y > 3.0,
                "dir={dir:?}: cannot enter ramp from +Y direction (pos.y={:.3})",
                cam_y.pos.y
            );
        }
    }

    // -----------------------------------------------------------------------
    // Spiral staircase collision regression tests.
    //
    // ROOT CAUSE: `move_with_collision` probes at `cz = floor(eye_z)`. When
    // the player climbs tread 8 of a spiral (RampZ NegY) their eye rises to
    // ≈8.0, making `cz=8`. The final landing cell is `Shape::Full, WallSolid`
    // at z=8. `solid_or_oob` used to return `true` for any Full-collidable
    // cell — so the player was permanently blocked from entering the landing,
    // even though their feet (eye_z − 0.5 = 7.5) were below the landing floor
    // (8.0). The fix: treat a Full-solid cell as a walkable step-up surface
    // (horizontal probe returns `false`) when the player's feet are below the
    // cell floor — `step_gravity` then snaps them to the top. Otherwise
    // (feet at or above the cell floor) the cell remains a solid wall.
    // -----------------------------------------------------------------------

    /// Build the 21-stairs-spiral map from stamps, mirroring the live level.
    ///
    /// Spiral: center=(6,6,0), radius=1, turns=1.
    /// One full turn (8 ramp treads) rises 8 cells.
    /// The map has a 12×12×13 grid with perimeter walls, floor at z=0, ceiling at z=12.
    fn spiral_stairs_map() -> Map {
        let spec = GeometrySpec {
            size: (12, 12, 13),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[],
        };
        let mut map = stamp_geometry(&spec);

        // Central column: (6,6,z) for z=0..=8 — Shape::Full, WallSolid.
        for z in 0usize..=8 {
            map.set(6, 6, z, CellKind::WallSolid);
            // shape_at defaults to Full — no explicit set needed, but be explicit.
            map.set_shape(6, 6, z, Shape::Full);
        }

        // Ramp treads (match levels/21-stairs-spiral.lvl [shapes] section).
        // The spiral_staircase generator overwrites corners with ramps so only
        // the final corner (5,5,8) remains Shape::Full.
        let treads: &[(usize, usize, usize, Shape)] = &[
            (5, 5, 0, Shape::RampZ(RampDir::PosX)),
            (6, 5, 1, Shape::RampZ(RampDir::PosX)),
            (7, 5, 2, Shape::RampZ(RampDir::PosY)),
            (7, 6, 3, Shape::RampZ(RampDir::PosY)),
            (7, 7, 4, Shape::RampZ(RampDir::NegX)),
            (6, 7, 5, Shape::RampZ(RampDir::NegX)),
            (5, 7, 6, Shape::RampZ(RampDir::NegY)),
            (5, 6, 7, Shape::RampZ(RampDir::NegY)),
        ];
        for &(x, y, z, shape) in treads {
            map.set(x, y, z, CellKind::WallSolid);
            map.set_shape(x, y, z, shape);
        }

        // Final landing: (5,5,8) — Shape::Full, WallSolid (player steps onto top).
        map.set(5, 5, 8, CellKind::WallSolid);
        map.set_shape(5, 5, 8, Shape::Full);

        map
    }

    /// REPRO test (was FAILING before the fix).
    ///
    /// The player climbs all 8 ramp treads of the spiral and then tries to
    /// enter the final landing cell `(5,5,8)` by moving in −y. This is the
    /// exact move that was blocked: the probe at `cz=floor(eye_z)=8` hit the
    /// landing (Full/WallSolid) and returned solid.
    ///
    /// After the fix the player can enter the landing cell; `step_gravity`
    /// then snaps them to the top surface (eye_z ≈ 9.5 = 8+1+0.5).
    #[test]
    fn spiral_landing_is_reachable_from_final_tread() {
        let map = spiral_stairs_map();

        // Place the player at the END of tread 8 (5,6,7) = RampZ(NegY).
        // local_y ≈ 0.5 → surface_z = 7 + (1 − 0.5) = 7.5; eye_z = 8.0.
        // The player needs to move in −y to enter landing (5,5,8).
        let eye_z = 8.0_f32;
        let mut cam = CameraPose {
            pos: Vec3::new(5.5, 6.5, eye_z),
            yaw: std::f32::consts::PI, // facing −y (north)
            pitch: 0.0,
        };
        let mut motion = PlayerMotion {
            vz: 0.0,
            grounded: false,
            ..Default::default()
        };
        // Settle on tread 8 surface.
        for _ in 0..10 {
            cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
        }

        // Attempt to move −y (toward the landing).
        let (_, moved_y) = cam.move_with_collision(0.0, -0.5, &map, PLAYER_RADIUS);

        assert!(
            moved_y,
            "REPRO FAIL: player blocked entering landing (5,5,8) from tread-8; \
             pos.y={:.4} (should have moved from 6.5 toward 5.5). \
             Root cause: horizontal probe at cz=floor(eye_z)=8 hit Full/WallSolid landing \
             even though feet_z={:.4} < cz=8 (step-up, not a wall).",
            cam.pos.y,
            eye_z - PLAYER_EYE_HEIGHT,
        );

        // The player must have moved closer to the landing.
        assert!(
            cam.pos.y < 6.5,
            "player must have advanced toward landing: pos.y={:.4}",
            cam.pos.y,
        );
    }

    /// Regression: the full spiral ring is walkable end-to-end.
    ///
    /// The player follows the REAL loop order — `move_with_collision` THEN
    /// `step_gravity` — and completes all 8 ramp treads plus enters the final
    /// landing. Waypoints drive each segment in the correct direction; after
    /// each segment, `step_gravity` settles vertical physics.
    ///
    /// Verifies no segment is blocked AND that the walker actually ends up
    /// standing on the TRUE top landing cell `(5,5,8)` — not merely that
    /// `total_rise` cleared some threshold. `total_rise > 5.0` alone is a weak
    /// property: a walker that climbed partway up the ring and then fell back
    /// down past its start height by less than 5 cells, or one that stalled a
    /// few treads short of the top, can still show a rise over 5 (the ring's
    /// full climb is 8 cells, so plenty of partial/wrong outcomes clear a 5.0
    /// floor). The tightened check instead pins the walker's final (x, y, z)
    /// to the landing's own known surface: `(5,5,8)` is `Shape::Full`, so its
    /// top is at world z=9 and the grounded eye height is
    /// `9.0 + PLAYER_EYE_HEIGHT` — the exact value `spiral_landing_is_reachable_from_final_tread`
    /// already establishes for this same cell.
    #[test]
    fn spiral_ring_walkable_end_to_end() {
        let map = spiral_stairs_map();

        // Helper: move the camera toward a target (x, y) in small steps,
        // running move_with_collision then step_gravity at each step.
        // Returns `false` if any step is blocked (moved_x AND moved_y both
        // false when non-zero displacement requested).
        fn walk_toward(
            cam: &mut CameraPose,
            motion: &mut PlayerMotion,
            map: &Map,
            target_x: f32,
            target_y: f32,
            label: &'static str,
        ) -> Result<(), (&'static str, f32, f32, f32)> {
            let max_steps = 200;
            for _ in 0..max_steps {
                let dx = target_x - cam.pos.x;
                let dy = target_y - cam.pos.y;
                if dx.abs() < 0.02 && dy.abs() < 0.02 {
                    break; // arrived
                }
                let step = 0.08_f32;
                let sdx = (dx.signum() * step).clamp(-dx.abs(), dx.abs());
                let sdy = (dy.signum() * step).clamp(-dy.abs(), dy.abs());

                let (mx, my) = cam.move_with_collision(sdx, sdy, map, PLAYER_RADIUS);
                // Block detection: if we requested movement but got nothing.
                let requested = sdx.abs() > 1e-4 || sdy.abs() > 1e-4;
                if requested && !mx && !my {
                    return Err((label, cam.pos.x, cam.pos.y, cam.pos.z));
                }
                for _ in 0..3 {
                    cam.step_gravity(motion, 1.0 / 60.0, map);
                }
            }
            Ok(())
        }

        // Start on the floor near tread 1.
        let mut cam = CameraPose {
            pos: Vec3::new(4.5, 5.5, STANDING_EYE_Z),
            yaw: 0.0,
            pitch: 0.0,
        };
        let mut motion = PlayerMotion {
            vz: 0.0,
            grounded: false,
            ..Default::default()
        };
        for _ in 0..20 {
            cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
        }
        assert!(motion.grounded, "must start grounded");

        let start_eye_z = cam.pos.z;

        // Walk the ring: each waypoint is the centre of a ring cell.
        // The path mirrors the spiral: (5,5)→(6,5)→(7,5)→(7,6)→(7,7)→(6,7)→(5,7)→(5,6)→(5,5).
        let waypoints: &[(&str, f32, f32)] = &[
            ("enter tread1 (5,5)", 5.5, 5.5),
            ("tread2 (6,5)", 6.5, 5.5),
            ("tread3 (7,5)", 7.5, 5.5),
            ("tread4 (7,6)", 7.5, 6.5),
            ("tread5 (7,7)", 7.5, 7.5),
            ("tread6 (6,7)", 6.5, 7.5),
            ("tread7 (5,7)", 5.5, 7.5),
            ("tread8 (5,6)", 5.5, 6.5),
            ("landing (5,5)", 5.5, 5.5),
        ];

        for &(label, tx, ty) in waypoints {
            if let Err((lbl, x, y, z)) = walk_toward(&mut cam, &mut motion, &map, tx, ty, label) {
                panic!(
                    "spiral walk blocked at segment {lbl:?}: \
                     pos=({x:.3},{y:.3},{z:.3}) target=({tx},{ty})"
                );
            }
        }

        // Settle fully on the landing before checking the final pose — the
        // per-waypoint budget (3 gravity ticks) is enough to keep walking
        // unblocked but not guaranteed to fully converge vertically.
        for _ in 0..10 {
            cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
        }
        assert!(
            motion.grounded,
            "must end grounded, standing on the landing"
        );
        assert!(
            (cam.pos.x - 5.5).abs() < 0.15 && (cam.pos.y - 5.5).abs() < 0.15,
            "must end OVER the landing cell (5,5,8), not stalled partway around \
             the ring: pos=({:.3},{:.3})",
            cam.pos.x,
            cam.pos.y,
        );
        let expected_landing_eye_z = 8.0 + 1.0 + PLAYER_EYE_HEIGHT;
        assert!(
            (cam.pos.z - expected_landing_eye_z).abs() < 0.05,
            "must end standing on the TRUE top landing surface \
             (cell z=8, Shape::Full, top=9, eye={expected_landing_eye_z:.3}); \
             ended at z={:.3} instead — a partial climb (or a climb-then-fall) \
             would land lower than this, even though it can still clear a bare \
             `total_rise > 5.0` floor",
            cam.pos.z,
        );

        let total_rise = cam.pos.z - start_eye_z;
        assert!(
            total_rise > 5.0,
            "spiral must rise at least 5 cells; rose only {total_rise:.3} \
             (start={start_eye_z:.3}, end={:.3})",
            cam.pos.z,
        );
    }

    /// Full-solid landing cell `(5,5,8)` directly: can enter from -y.
    /// This isolates the exact blocked move in the repro.
    #[test]
    fn full_solid_step_up_landing_allows_horizontal_entry() {
        // Build a minimal map: 8×8×12 room.
        let spec = GeometrySpec {
            size: (8, 8, 12),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[],
        };
        let mut map = stamp_geometry(&spec);
        // Place a Full solid cell at (3, 2, 3) — the "landing" a step higher
        // than the floor.
        map.set(3, 2, 3, CellKind::WallSolid);
        // shape is Full by default.

        // Player standing on floor z=2 (air cell, eye_z = 3.0).
        // Their feet are at 3.0 − 0.5 = 2.5, just BELOW the landing floor (3.0).
        // This is the step-up scenario.
        let mut cam = CameraPose {
            pos: Vec3::new(3.5, 3.5, 3.0),
            yaw: 0.0,
            pitch: 0.0,
        };

        // Probe: try to move in −y toward the landing.
        let (_, moved) = cam.move_with_collision(0.0, -1.0, &map, PLAYER_RADIUS);
        assert!(
            moved,
            "step-up landing must not block horizontal entry: \
             eye_z=3.0 feet_z=2.5 < landing_cell_z=3 → should be allowed; \
             pos.y={:.4}",
            cam.pos.y,
        );
    }

    /// A Full-solid WALL at the same height as the player (feet inside the cell)
    /// still blocks — the fix must not weaken normal wall collision.
    #[test]
    fn full_solid_wall_at_eye_level_still_blocks() {
        let spec = GeometrySpec {
            size: (8, 8, 5),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[],
        };
        let mut map = stamp_geometry(&spec);
        // Wall at (3, 2, 1) — same z-cell as the player standing on floor.
        map.set(3, 2, 1, CellKind::WallSolid);

        // Player at floor level: eye_z=1.5, feet_z=1.0 = cz=1 → wall blocks.
        let mut cam = CameraPose {
            pos: Vec3::new(3.5, 3.5, 1.5),
            yaw: 0.0,
            pitch: 0.0,
        };
        cam.move_with_collision(0.0, -1.0, &map, PLAYER_RADIUS);
        assert!(
            cam.pos.y + PLAYER_RADIUS >= 2.0 - 1e-4,
            "wall at same z-level must block: pos.y={:.4} must not enter cell (3,2)",
            cam.pos.y,
        );
    }

    /// Wedge cells (WedgeXY) STILL block horizontally — regression guard.
    /// The fix to make RampZ walkable must not affect WedgeXY diagonal wall
    /// collision.
    #[test]
    fn wedge_cells_still_block_horizontally() {
        // WedgeXY(NE): solid where x+y≥1. Cell at (3,3,1).
        // Approach from x=2.5, y=3.5 in +X with dx=0.5:
        //   new_x = 3.0, probe_x = 3.0 + 0.25 = 3.25, cx=3.
        //   local_x = 0.25, local_y = 3.5-3 = 0.5 → x+y = 0.75 < 1 → AIR!
        // The NE wedge's SOLID half is x+y≥1. The probe at y≈3.5 (local_y=0.5)
        // needs local_x≥0.5 to be solid. Use dx=0.75 so probe is at 3.5+0.25=3.75:
        //   local_x=0.75, local_y=0.5 → x+y=1.25 ≥ 1 → SOLID → blocked.
        let spec = GeometrySpec {
            size: (7, 7, 4),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[],
        };

        // NE wedge: solid half is x+y≥1. Approaching from x=2.5, y=3.5 with
        // dx=0.75 makes probe_x=3.5 (local_x=0.5), local_y=0.5 → x+y=1.0 → SOLID.
        let mut map_ne = stamp_geometry(&spec);
        map_ne.set(3, 3, 1, CellKind::WallSolid);
        map_ne.set_shape(3, 3, 1, Shape::WedgeXY(DiagXY::NE));
        let mut cam_ne = CameraPose {
            pos: Vec3::new(2.5, 3.5, 1.5),
            yaw: 0.0,
            pitch: 0.0,
        };
        cam_ne.move_with_collision(0.75, 0.0, &map_ne, PLAYER_RADIUS);
        assert!(
            cam_ne.pos.x + PLAYER_RADIUS <= 3.0 + 1e-3,
            "WedgeXY NE solid half still blocks: x={:.3} should be blocked at y=3.5",
            cam_ne.pos.x
        );

        // NE wedge: air half is x+y<1. At y=2.5 (local_y≈0.5... wait, from 2.5
        // the probe is in cy=2, not cy=3). Use starting y=3.2 (local_y≈0.2 when in
        // cell 3): probe x=3.25 → local_x=0.25, local_y=0.2 → x+y=0.45 < 1 → AIR.
        let mut cam_ne_air = CameraPose {
            pos: Vec3::new(2.5, 3.2, 1.5),
            yaw: 0.0,
            pitch: 0.0,
        };
        cam_ne_air.move_with_collision(0.75, 0.0, &map_ne, PLAYER_RADIUS);
        assert!(
            cam_ne_air.pos.x > 3.0,
            "WedgeXY NE air half passes: x={:.3} should enter ramp cell at y=3.2",
            cam_ne_air.pos.x
        );

        // Full cell (no shape) still blocks entirely.
        let mut map_full = stamp_geometry(&spec);
        map_full.set(3, 3, 1, CellKind::WallSolid);
        // shape defaults to Full
        let mut cam_full = CameraPose {
            pos: Vec3::new(1.5, 3.5, 1.5),
            yaw: 0.0,
            pitch: 0.0,
        };
        cam_full.move_with_collision(2.0, 0.0, &map_full, PLAYER_RADIUS);
        assert!(
            cam_full.pos.x + PLAYER_RADIUS <= 3.0 + 1e-3,
            "Full cell still blocks: x={:.3}",
            cam_full.pos.x
        );
    }

    // -----------------------------------------------------------------------
    // Sky-image storage tests.
    // -----------------------------------------------------------------------

    /// Helper: build a minimal 1×1 TextureImage filled with a single RGBA colour.
    fn tiny_texture(r: u8, g: u8, b: u8) -> re_core::TextureImage {
        re_core::TextureImage::new(1, 1, vec![r, g, b, 255])
    }

    /// A fresh Map carries no sky images (all slots None) so procedural sky
    /// behaviour is byte-identical to today.
    #[test]
    fn sky_images_default_all_none() {
        let m = Map::new(4, 4, 4);
        let sky = m.sky_images();
        assert!(sky.cylinder.is_none(), "cylinder must default to None");
        assert!(sky.cap_top.is_none(), "cap_top must default to None");
        assert!(sky.cap_bottom.is_none(), "cap_bottom must default to None");
    }

    /// Setting the cylinder image is stored and read back by reference.
    #[test]
    fn sky_images_set_cylinder_round_trips() {
        let mut m = Map::new(4, 4, 4);
        let tex = std::sync::Arc::new(tiny_texture(0x11, 0x22, 0x33));
        m.set_sky_cylinder(tex.clone());

        let sky = m.sky_images();
        assert!(sky.cylinder.is_some(), "cylinder must be Some after set");
        let stored = sky.cylinder.as_ref().unwrap();
        assert_eq!(stored.width, 1);
        assert_eq!(stored.height, 1);
        assert_eq!(stored.rgba[0], 0x11);
        assert_eq!(stored.rgba[1], 0x22);
        assert_eq!(stored.rgba[2], 0x33);
        // cap slots remain None — only cylinder was set.
        assert!(sky.cap_top.is_none());
        assert!(sky.cap_bottom.is_none());
    }

    /// Setting the top cap image is stored and read back; other slots unaffected.
    #[test]
    fn sky_images_set_cap_top_round_trips() {
        let mut m = Map::new(4, 4, 4);
        m.set_sky_cap_top(std::sync::Arc::new(tiny_texture(0xAA, 0xBB, 0xCC)));

        let sky = m.sky_images();
        assert!(sky.cap_top.is_some(), "cap_top must be Some after set");
        assert_eq!(sky.cap_top.as_ref().unwrap().rgba[0], 0xAA);
        assert!(sky.cylinder.is_none());
        assert!(sky.cap_bottom.is_none());
    }

    /// Setting the bottom cap image is stored and read back; other slots unaffected.
    #[test]
    fn sky_images_set_cap_bottom_round_trips() {
        let mut m = Map::new(4, 4, 4);
        m.set_sky_cap_bottom(std::sync::Arc::new(tiny_texture(0xDE, 0xAD, 0xBE)));

        let sky = m.sky_images();
        assert!(
            sky.cap_bottom.is_some(),
            "cap_bottom must be Some after set"
        );
        assert_eq!(sky.cap_bottom.as_ref().unwrap().rgba[0], 0xDE);
        assert!(sky.cylinder.is_none());
        assert!(sky.cap_top.is_none());
    }

    /// All three sky slots can be set independently; all round-trip correctly.
    #[test]
    fn sky_images_all_three_slots_independent() {
        let mut m = Map::new(4, 4, 4);
        m.set_sky_cylinder(std::sync::Arc::new(tiny_texture(1, 2, 3)));
        m.set_sky_cap_top(std::sync::Arc::new(tiny_texture(4, 5, 6)));
        m.set_sky_cap_bottom(std::sync::Arc::new(tiny_texture(7, 8, 9)));

        let sky = m.sky_images();
        assert_eq!(sky.cylinder.as_ref().unwrap().rgba[0], 1);
        assert_eq!(sky.cap_top.as_ref().unwrap().rgba[0], 4);
        assert_eq!(sky.cap_bottom.as_ref().unwrap().rgba[0], 7);
    }

    /// Replacing a sky slot replaces the Arc; no old data leaks through.
    #[test]
    fn sky_images_cylinder_replacement() {
        let mut m = Map::new(4, 4, 4);
        m.set_sky_cylinder(std::sync::Arc::new(tiny_texture(0xFF, 0x00, 0x00)));
        m.set_sky_cylinder(std::sync::Arc::new(tiny_texture(0x00, 0xFF, 0x00)));

        let stored = m.sky_images().cylinder.as_ref().unwrap();
        // Must see the SECOND set (green), not the first (red).
        assert_eq!(
            stored.rgba[0], 0x00,
            "red channel must be 0 after replacement"
        );
        assert_eq!(
            stored.rgba[1], 0xFF,
            "green channel must be 0xFF after replacement"
        );
    }

    /// SkyImages can be cloned (Map is Clone); the clone owns independent Arcs
    /// that share the underlying pixel buffer.
    #[test]
    fn sky_images_map_clone_shares_buffer() {
        let mut m = Map::new(4, 4, 4);
        let tex = std::sync::Arc::new(tiny_texture(0x42, 0x43, 0x44));
        m.set_sky_cylinder(tex.clone());

        let m2 = m.clone();
        let sky = m2.sky_images();
        assert!(sky.cylinder.is_some());
        // The Arc inside the clone shares the same allocation.
        assert!(
            std::sync::Arc::ptr_eq(
                m.sky_images().cylinder.as_ref().unwrap(),
                sky.cylinder.as_ref().unwrap()
            ),
            "clone must share the Arc pointer"
        );
    }
}
