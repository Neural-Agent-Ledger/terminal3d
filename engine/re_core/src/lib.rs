// Test-module assert_eq! on f32 values are intentional exact-equal checks.
// Allow float_cmp only in test compilation units; lib code remains denied.
#![cfg_attr(test, allow(clippy::float_cmp))]
#![warn(missing_docs)]
//! `re_core` — shared types for the TERMINAL STATE engine.
//!
//! Holds the [`Framebuffer`] resource (the seam between the renderer and the
//! terminal) plus shared math re-exports. Engine-agnostic: no terminal, no
//! `bevy_render`. Everything here compiles without a terminal present.

pub use glam;
pub use glam::{Vec2, Vec3};

use bevy::prelude::Resource;

pub mod layer;
pub use layer::{CompositeStack, Layer, LayerBlend};

pub mod shape;
pub use shape::{shape_solid_contains, shape_surface, DiagXY, RampDir, Shape, SurfacePlane};

pub mod perf;
pub use perf::{FrameTimingMetrics, InputGate, PerfOverlay};

pub mod cell_state;
pub use cell_state::{
    decode_flow_state, decode_holo_style, decode_pipe_tag, encode_flow_state, encode_holo_rgb,
    encode_pipe_side_data, holo_style_named, CellStateMap, FlowDir, DOOR_OBJECT_TAG, HOLO_PRESETS,
    HOLO_STYLE_RGB, INERT_OBJECT_TAG, PIPE_TAG_FLAG, PIPE_TAG_MASK, RESERVED_SENTINELS,
    RESERVED_TAG_BASE,
};

pub mod text;

pub mod face;
pub use face::{FaceDir, FACE_MATCH_MIN_DOT};

pub mod pool;
pub use pool::{NodePool, PoolTable};

pub mod overlay;
pub use overlay::{
    Inset, InsetContent, InsetId, Line, LineMaterialId, LinePath, MaterialId, Overlay, Region,
    ResolvedInset, ScaleRule, SelfSimilarParams, Zone,
};

pub mod trigger;
pub use trigger::{Trigger, TriggerAction};

pub mod oklab;

/// Sentinel `underline` value meaning "no colored underline" — the cell's
/// underline color is left to the terminal default. A real underline color is a
/// packed `0x00RRGGBB`, so the high byte is always `0x00`; using `u32::MAX`
/// (high byte `0xFF`, impossible for a real packed color) as the sentinel keeps
/// every real color valid and unambiguous.
pub const NO_UNDERLINE: u32 = u32::MAX;

// Style flag bits for [`CellStyle::flags`]. A compact `u8` bitset — six terminal
// text attributes that gracefully degrade (a terminal that lacks one ignores its
// SGR). Kept as plain associated consts (no `bitflags` dep) so `re_core` stays
// dependency-light and `CellStyle` is trivially `Copy`/`Default`.
impl CellStyle {
    /// Bold / increased intensity (SGR 1).
    pub const BOLD: u8 = 1 << 0;
    /// Dim / faint (SGR 2).
    pub const DIM: u8 = 1 << 1;
    /// Italic (SGR 3).
    pub const ITALIC: u8 = 1 << 2;
    /// Underline (SGR 4).
    pub const UNDERLINE: u8 = 1 << 3;
    /// Reverse video — swap fg/bg (SGR 7).
    pub const REVERSE: u8 = 1 << 4;
    /// Strikethrough / crossed-out (SGR 9).
    pub const STRIKE: u8 = 1 << 5;
}

/// The terminal underline SUBSTYLE — a separate "smart-material" channel layered
/// over the existing [`CellStyle`] attributes + colored-underline color.
///
/// Terminals expose several underline shapes via the SGR `4:n` extension (and the
/// legacy `21` for double): a plain rule, a double rule, a curly/undercurl
/// squiggle, a dotted rule, and a dashed rule. They render INDEPENDENTLY of the
/// underline *colour* (`58:2:…`), so a material can pair, say, a curly substyle
/// with the hologram hue to get a glowing scan-glow under the line glyphs.
///
/// Additive + back-compatible: the default is [`None`](Self::None) (no underline
/// shape at all), and the legacy [`UNDERLINE`](CellStyle::UNDERLINE) flag maps to
/// [`Single`](Self::Single) — so a cell that never sets a substyle renders exactly
/// as before. The substyles beyond `Single` are emitted faithfully in OUR `--ansi`
/// dump (we own the bytes) and shown by a capable terminal; the live ratatui
/// backend can do `Single` + colour but not the curly/dotted/dashed shapes, so it
/// falls back to a single underline there (documented in the present layer).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UnderlineStyle {
    /// No underline shape (the default — renders identically to the pre-substyle
    /// cell). Note a colored underline still implies a `Single` rule at present
    /// time (see [`CellStyle::effective_underline_style`]).
    #[default]
    None,
    /// A single underline rule — SGR `4` / `4:1`. The shape the legacy
    /// [`UNDERLINE`](CellStyle::UNDERLINE) flag maps to.
    Single,
    /// A double underline rule — SGR `4:2` (legacy `21`).
    Double,
    /// A curly / undercurl squiggle — SGR `4:3`. The hologram scan-glow shape.
    Curly,
    /// A dotted underline — SGR `4:4`.
    Dotted,
    /// A dashed underline — SGR `4:5`.
    Dashed,
}

impl UnderlineStyle {
    /// The SGR underline-substyle parameter digit (the `n` in `4:n`): `Single`→1,
    /// `Double`→2, `Curly`→3, `Dotted`→4, `Dashed`→5. `None` has no shape and
    /// returns `0` (the `4:0` / `24` reset), so callers can treat it as "off".
    #[inline]
    pub fn sgr_param(self) -> u8 {
        match self {
            UnderlineStyle::None => 0,
            UnderlineStyle::Single => 1,
            UnderlineStyle::Double => 2,
            UnderlineStyle::Curly => 3,
            UnderlineStyle::Dotted => 4,
            UnderlineStyle::Dashed => 5,
        }
    }

    /// True for every shape that draws an underline (i.e. not [`None`](Self::None)).
    #[inline]
    pub fn is_some(self) -> bool {
        self != UnderlineStyle::None
    }
}

/// Additive per-[`Cell`] terminal text styling: a compact bitset of attribute
/// flags, an optional independent third color channel (the colored underline),
/// and an underline SUBSTYLE shape (single/double/curly/dotted/dashed).
///
/// This is purely additive over the existing `glyph`/`fg`/`bg` contract: a
/// [`Default`] `CellStyle` carries no flags, [`NO_UNDERLINE`], and
/// [`UnderlineStyle::None`], so a cell that never touches it renders exactly as
/// before. The present layer maps `flags` onto terminal SGR attributes, emits
/// `underline` as the colored-underline SGR (`58;2;r;g;b`) — a real, independent
/// color separate from fg/bg — and emits `underline_style` as the `4:n` substyle.
///
/// * `flags` — OR of `CellStyle::BOLD | DIM | ITALIC | UNDERLINE | REVERSE | STRIKE`.
/// * `underline` — packed `0x00RRGGBB` colored-underline color, or [`NO_UNDERLINE`].
///   A colored underline implies underlining; the present layer turns on the
///   underline attribute whenever a real underline color is present, so a glow
///   accent shows even if the [`UNDERLINE`](CellStyle::UNDERLINE) flag was not set.
/// * `underline_style` — the underline SHAPE (see [`UnderlineStyle`]). Additive:
///   [`UnderlineStyle::None`] (the default) is unchanged; a substyle is a
///   smart-material accent (e.g. a curly hologram scan-glow). The legacy
///   `UNDERLINE` flag / a colored underline imply a `Single` rule when no explicit
///   substyle is set — see [`effective_underline_style`](Self::effective_underline_style).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CellStyle {
    /// Bitset of attribute flags (see the `CellStyle::BOLD …` associated consts).
    pub flags: u8,
    /// Packed `0x00RRGGBB` colored-underline color, or [`NO_UNDERLINE`] for none.
    pub underline: u32,
    /// The underline SHAPE / substyle (single/double/curly/dotted/dashed).
    /// [`UnderlineStyle::None`] (the default) leaves the shape to the legacy
    /// underline attribute — see the struct docs.
    pub underline_style: UnderlineStyle,
}

impl Default for CellStyle {
    /// No attributes, no colored underline, no underline shape — renders
    /// identically to the pre-style cell.
    fn default() -> Self {
        CellStyle {
            flags: 0,
            underline: NO_UNDERLINE,
            underline_style: UnderlineStyle::None,
        }
    }
}

impl CellStyle {
    /// A style with no attributes, no colored underline, and no underline shape
    /// (same as [`Default`]). Const so it can seed other consts.
    pub const NONE: CellStyle = CellStyle {
        flags: 0,
        underline: NO_UNDERLINE,
        underline_style: UnderlineStyle::None,
    };

    /// True if no attribute flags are set AND there is no colored underline AND no
    /// underline shape — i.e. this style is a no-op the present layer can skip
    /// entirely.
    #[inline]
    pub fn is_empty(self) -> bool {
        self.flags == 0
            && self.underline == NO_UNDERLINE
            && self.underline_style == UnderlineStyle::None
    }

    /// True if the bold flag is set.
    #[inline]
    pub fn is_bold(self) -> bool {
        self.flags & CellStyle::BOLD != 0
    }

    /// True if the dim flag is set.
    #[inline]
    pub fn is_dim(self) -> bool {
        self.flags & CellStyle::DIM != 0
    }

    /// True if the italic flag is set.
    #[inline]
    pub fn is_italic(self) -> bool {
        self.flags & CellStyle::ITALIC != 0
    }

    /// True if the underline flag is set. Note a colored underline
    /// ([`has_underline_color`](Self::has_underline_color)) is rendered as
    /// underlined even if this flag is unset.
    #[inline]
    pub fn is_underline(self) -> bool {
        self.flags & CellStyle::UNDERLINE != 0
    }

    /// True if the reverse-video flag is set.
    #[inline]
    pub fn is_reverse(self) -> bool {
        self.flags & CellStyle::REVERSE != 0
    }

    /// True if the strikethrough flag is set.
    #[inline]
    pub fn is_strike(self) -> bool {
        self.flags & CellStyle::STRIKE != 0
    }

    /// True if a real colored-underline color is set (not [`NO_UNDERLINE`]).
    #[inline]
    pub fn has_underline_color(self) -> bool {
        self.underline != NO_UNDERLINE
    }

    /// The colored-underline color, if any (`None` ⇒ [`NO_UNDERLINE`]).
    #[inline]
    pub fn underline_color(self) -> Option<u32> {
        if self.underline == NO_UNDERLINE {
            None
        } else {
            Some(self.underline & 0x00FF_FFFF)
        }
    }

    /// The explicitly-set underline SHAPE (see [`UnderlineStyle`]). This is the
    /// raw stored value; [`UnderlineStyle::None`] means "no explicit shape" — use
    /// [`effective_underline_style`](Self::effective_underline_style) to also honour
    /// the legacy underline flag / colored underline.
    #[inline]
    pub fn underline_style(self) -> UnderlineStyle {
        self.underline_style
    }

    /// The underline shape the present layer should actually draw, resolving the
    /// back-compat rule: an explicit [`underline_style`](Self::underline_style)
    /// wins; otherwise a legacy underline attribute or a colored underline implies
    /// a [`Single`](UnderlineStyle::Single) rule; otherwise [`None`](UnderlineStyle::None).
    /// This keeps every pre-substyle cell (which only ever sets the `UNDERLINE`
    /// flag / a colored underline) rendering a single underline exactly as before.
    #[inline]
    pub fn effective_underline_style(self) -> UnderlineStyle {
        if self.underline_style.is_some() {
            self.underline_style
        } else if self.is_underline() || self.has_underline_color() {
            UnderlineStyle::Single
        } else {
            UnderlineStyle::None
        }
    }

    /// Set (OR-in) one or more flag bits, returning the modified style (builder).
    #[inline]
    #[must_use]
    pub fn with_flags(mut self, flags: u8) -> Self {
        self.flags |= flags;
        self
    }

    /// Turn on the bold attribute (builder).
    #[inline]
    #[must_use]
    pub fn with_bold(self) -> Self {
        self.with_flags(CellStyle::BOLD)
    }

    /// Turn on the underline attribute (builder).
    #[inline]
    #[must_use]
    pub fn with_underline(self) -> Self {
        self.with_flags(CellStyle::UNDERLINE)
    }

    /// Set the colored-underline color to a packed `0x00RRGGBB` (builder). Also
    /// turns on the underline attribute so the color is actually drawn.
    #[inline]
    #[must_use]
    pub fn with_underline_color(mut self, rgb: u32) -> Self {
        self.underline = rgb & 0x00FF_FFFF;
        self.flags |= CellStyle::UNDERLINE;
        self
    }

    /// Set the underline SHAPE / substyle (builder). A non-[`None`](UnderlineStyle::None)
    /// shape also turns on the legacy underline attribute so a backend that only
    /// understands a single underline (e.g. live ratatui) still draws *a* rule —
    /// the substyle degrades to single rather than vanishing. [`None`](UnderlineStyle::None)
    /// clears the explicit shape without touching the attribute flag.
    #[inline]
    #[must_use]
    pub fn with_underline_style(mut self, style: UnderlineStyle) -> Self {
        self.underline_style = style;
        if style.is_some() {
            self.flags |= CellStyle::UNDERLINE;
        }
        self
    }
}

/// What kind of surface a [`Cell`] looks at — the shader-dispatch key.
///
/// A Rust port of the legacy `SurfaceId` enum (`framebuffer.h`), trimmed to the
/// surfaces the M3 textured raycaster actually emits. The raycaster stamps this
/// per cell so the glyph_shade pass can dispatch the right look (wall texture vs
/// floor grain vs ceiling stars) without re-deriving geometry.
///
/// `Empty` is the zero value so a defaulted [`Cell`] is "sky / nothing", matching
/// the legacy `SURFACE_EMPTY == 0` contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SurfaceKind {
    /// No hit — open sky above the horizon. Shaded as ambient ceiling.
    #[default]
    Empty,
    /// Opaque flat-shaded wall (`CELL_WALL_SOLID`) — brick/panel texture.
    WallSolid,
    /// Circuit-trace decal wall (`CELL_WALL_TRACE`).
    WallTrace,
    /// Matrix-rain decal wall (`CELL_WALL_MATRIX`).
    WallMatrix,
    /// A non-wall solid hit (node / door): treated as a plain solid wall face.
    WallOther,
    /// Interior floor, reached by per-row floor casting.
    Floor,
    /// Interior ceiling (or sky), reached by per-row casting / no-hit.
    Ceiling,
    /// A non-euclidean warp-portal face — the DDA stops here and the raycaster
    /// re-bases the ray by the portal's `delta` transform into the destination
    /// region (Stage 1 render re-basing, spec §3.1). Carries the same geometric
    /// data as a wall hit (tex_u/side/hit) so the portal rim can be shaded; the
    /// re-based segment's colour is composited over it.
    ///
    /// In exhaustive `match surface { … }` arms, treat `Portal` like a solid
    /// wall surface where a plain default is acceptable — the renderer will add
    /// a dedicated arm in Stage 1 (`re_render::surface_material` dispatch).
    Portal,
}

impl SurfaceKind {
    /// True for the wall surfaces (any face the DDA stopped on).
    ///
    /// `Portal` is NOT included here: it is NOT a conventional wall surface —
    /// the renderer dispatches it separately for ray re-basing. Callers that
    /// use `is_wall()` to gate mirror / decal logic should not fire on a portal.
    #[inline]
    pub fn is_wall(self) -> bool {
        matches!(
            self,
            SurfaceKind::WallSolid
                | SurfaceKind::WallTrace
                | SurfaceKind::WallMatrix
                | SurfaceKind::WallOther
        )
    }
}

/// One terminal cell in the [`Framebuffer`].
///
/// The downstream-stable fields (`glyph`/`fg`/`bg`/`depth`) are the contract the
/// terminal-present layer depends on — they MUST keep their names, types, and
/// meaning. The M3 textured raycaster ADDS the richer per-cell geometry the
/// glyph_shade pass needs (`tex_u`, `side`, `hit`, `surface`, `material`); all
/// have zero/empty defaults so any code that ignores them still compiles.
///
/// * `glyph` — Unicode scalar value to draw.
/// * `fg`/`bg` — packed `0x00RRGGBB` colors.
/// * `depth` — perpendicular distance to the surface (for diffing / later LOD).
/// * `tex_u` — wall texture coordinate in `[0,1]` across the hit face (0 off-wall).
/// * `tex_v` — wall texture coordinate in `[0,1]` DOWN the hit face: `0` at the
///   top of the visible wall slice, `1` at the bottom (0 off-wall). Together with
///   `tex_u` it gives a full 2D surface coordinate, which a 2D wall texture (e.g.
///   text laid out in a grid — the `CodeWall` recipe) needs to know which glyph of
///   the source maps to this cell. Additive: floor/ceiling and pre-`CodeWall` code
///   leave it 0, so nothing that ignores it changes.
/// * `side` — wall face hit: `0` = NS face, `1` = EW face (directional shading).
/// * `hit` — exact world-space point this cell looks at (wall hit or floor point).
/// * `surface` — surface-kind dispatch key (see [`SurfaceKind`]).
/// * `material` — surface-specific extra tag (node/door id); `0` when unused.
/// * `style` — additive terminal text styling (bold/dim/italic/underline/reverse/
///   strike + an independent colored-underline channel). [`CellStyle::default`]
///   (no flags, no underline) so any cell that ignores it renders exactly as
///   before; the present layer maps it to SGR.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cell {
    /// Unicode scalar value to draw.
    pub glyph: u32,
    /// Foreground color, packed `0x00RRGGBB`.
    pub fg: u32,
    /// Background color, packed `0x00RRGGBB`.
    pub bg: u32,
    /// Perpendicular distance to the surface (for diffing / later LOD).
    pub depth: f32,
    /// Wall texture coordinate in `[0,1]` across the hit face (`0` off-wall).
    pub tex_u: f32,
    /// Vertical wall coordinate in `[0,1]` (0 = top of the wall slice, 1 = bottom).
    /// 0 for floor/ceiling cells. See the struct docs for why it exists.
    pub tex_v: f32,
    /// Wall face hit: `0` = NS face, `1` = EW face (directional shading).
    pub side: u8,
    /// Exact world-space point this cell looks at (wall hit or floor point).
    pub hit: Vec2,
    /// Surface-kind dispatch key (see [`SurfaceKind`]).
    pub surface: SurfaceKind,
    /// Surface-specific extra tag (node/door id); `0` when unused.
    pub material: u16,
    /// Additive per-cell terminal text style (SGR attributes + colored
    /// underline). [`CellStyle::default`] = no styling.
    pub style: CellStyle,
}

impl Default for Cell {
    fn default() -> Self {
        Cell {
            glyph: u32::from(' '),
            fg: 0x0000_0000,
            bg: 0x0000_0000,
            depth: f32::INFINITY,
            tex_u: 0.0,
            tex_v: 0.0,
            side: 0,
            hit: Vec2::ZERO,
            surface: SurfaceKind::Empty,
            material: 0,
            style: CellStyle::NONE,
        }
    }
}

/// The per-cell render target. A dense row-major grid of [`Cell`]s.
///
/// Owned by the renderer (`re_render` writes it) and consumed by the terminal
/// crate (`re_term` presents it). It is the ONLY contract between Layer A and
/// Layer B — exactly as in the C engine's `Framebuffer`.
#[derive(Resource, Clone, Debug)]
pub struct Framebuffer {
    cols: usize,
    rows: usize,
    cells: Vec<Cell>,
}

impl Framebuffer {
    /// Allocate a `cols`×`rows` framebuffer of default (empty) cells.
    pub fn new(cols: usize, rows: usize) -> Self {
        Framebuffer {
            cols,
            rows,
            cells: vec![Cell::default(); cols * rows],
        }
    }

    /// Grid width in cells.
    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Grid height in cells.
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Resize (and clear) the grid if the terminal dimensions changed.
    pub fn resize(&mut self, cols: usize, rows: usize) {
        if cols != self.cols || rows != self.rows {
            self.cols = cols;
            self.rows = rows;
            self.cells.clear();
            self.cells.resize(cols * rows, Cell::default());
        }
    }

    /// Reset every cell to the default (empty) value.
    pub fn clear(&mut self) {
        for c in &mut self.cells {
            *c = Cell::default();
        }
    }

    #[inline]
    fn index(&self, col: usize, row: usize) -> usize {
        row * self.cols + col
    }

    /// Read the cell at `(col, row)`. Returns `None` if out of bounds.
    #[inline]
    pub fn get(&self, col: usize, row: usize) -> Option<&Cell> {
        if col < self.cols && row < self.rows {
            Some(&self.cells[self.index(col, row)])
        } else {
            None
        }
    }

    /// Write the cell at `(col, row)`. Out-of-bounds writes are ignored.
    #[inline]
    pub fn set(&mut self, col: usize, row: usize, cell: Cell) {
        if col < self.cols && row < self.rows {
            let i = self.index(col, row);
            self.cells[i] = cell;
        }
    }

    /// All cells, row-major (for batched presentation).
    #[inline]
    pub fn cells(&self) -> &[Cell] {
        &self.cells
    }
}

/// An RGBA8 image (e.g. a decoded PNG) usable as a wall texture.
///
/// Row-major, **top-left origin**, 4 bytes per pixel (`R,G,B,A`). This is the
/// engine-agnostic, terminal-free pixel buffer the renderer samples to texture a
/// wall face. Decoding (`.png` → these bytes) is the job of a higher layer
/// (`re_level`) — `re_core` only owns the in-memory contract and the sampler so
/// every crate agrees on layout and UV convention.
///
/// UV convention (matches [`Cell::tex_u`] / [`Cell::tex_v`]): `u` runs `0→1`
/// left-to-right across the source, `v` runs `0→1` top-to-bottom (`v = 0` is the
/// top row). [`sample_nearest`](TextureImage::sample_nearest) clamps out-of-range
/// UVs and picks the nearest texel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextureImage {
    /// Pixel-row width. Not to be confused with `height_map`'s relief channel.
    pub width: u32,
    /// Pixel-row count (the PIXEL ROW COUNT — see [`height_map`](Self::height_map)
    /// for why the relief channel is not also called `height`).
    pub height: u32,
    /// `width * height * 4` bytes, row-major, top-left origin, `R,G,B,A`.
    pub rgba: Vec<u8>,
    /// OPTIONAL 8-bit RELIEF channel: `width * height` bytes, ONE per texel, in the
    /// same row-major top-left-origin order as [`rgba`](Self::rgba). `0` = the lowest
    /// point of the surface, `255` = the highest. `None` (every image decoded before
    /// the relief work, and every image whose `[texture]` block authors no
    /// `height = <path>`) means the image carries no relief and the renderer must not
    /// sample it — see [`sample_height_nearest`](Self::sample_height_nearest).
    ///
    /// # Why it is not called `height`
    ///
    /// [`height`](Self::height) is already taken, and it means the PIXEL ROW COUNT.
    /// A second field spelled `height` would make `img.height` ambiguous at every
    /// call site in the workspace — including the ~40 that divide by it. `height_map`
    /// is the graphics term for exactly this channel and cannot be misread as a
    /// dimension.
    ///
    /// # Why `Arc<[u8]>` and not `Vec<u8>`
    ///
    /// The relief channel is read once per shaded cell and is often megabytes. It is
    /// `Arc`-shared so that cloning a `TextureImage` (or handing the same source image
    /// to several materials) never duplicates it, and so the shader can hold a
    /// borrowed slice with no allocation in the hot path.
    pub height_map: Option<std::sync::Arc<[u8]>>,
}

impl TextureImage {
    /// Build an image from raw RGBA8 bytes.
    ///
    /// The buffer is normalized to exactly `width * height * 4` bytes: a short
    /// buffer is zero-padded and an over-long one is truncated, so an invalid
    /// caller can never make the sampler read out of bounds. A zero-area image
    /// (either dimension `0`) is allowed and samples as transparent black.
    pub fn new(width: u32, height: u32, mut rgba: Vec<u8>) -> Self {
        let needed = (width as usize)
            .saturating_mul(height as usize)
            .saturating_mul(4);
        if rgba.len() != needed {
            rgba.resize(needed, 0);
        }
        TextureImage {
            width,
            height,
            rgba,
            // A plain `new` never carries relief: the un-authored path must stay
            // byte-identical, and `height_at` in the shader keys off exactly this.
            height_map: None,
        }
    }

    /// The height a texel reports when this image has NO relief channel — a flat
    /// mid-grey surface. Chosen (rather than `0`) so that an accidental sample of a
    /// map-less image produces a *level* surface rather than a hole: every gradient
    /// tap around it returns the same value, so the perturbed normal equals the
    /// geometric face normal and the lighting is unchanged.
    pub const NEUTRAL_HEIGHT: f32 = 0.5;

    /// Attach an 8-bit relief channel (see [`height_map`](Self::height_map)),
    /// consuming and returning the image so it composes with
    /// [`new`](Self::new): `TextureImage::new(w, h, rgba).with_height_map(bytes)`.
    ///
    /// The buffer is normalized to exactly `width * height` bytes the same way
    /// [`new`](Self::new) normalizes `rgba` — short buffers are zero-padded, long
    /// ones truncated — so a caller that resampled the relief source to the wrong
    /// size can never make the sampler read out of bounds. A zero-area image gets an
    /// empty channel, which samples as [`NEUTRAL_HEIGHT`](Self::NEUTRAL_HEIGHT).
    #[must_use]
    pub fn with_height_map(mut self, mut bytes: Vec<u8>) -> Self {
        let needed = (self.width as usize).saturating_mul(self.height as usize);
        if bytes.len() != needed {
            bytes.resize(needed, 0);
        }
        self.height_map = Some(std::sync::Arc::from(bytes.into_boxed_slice()));
        self
    }

    /// Attach an ALREADY-SHARED relief channel — the frame-set form of
    /// [`with_height_map`](Self::with_height_map).
    ///
    /// An animated texture's frames all share one authored `height = <path>` map (the
    /// relief belongs to the SURFACE, not to the moment). Cloning the `Vec` into each
    /// of, say, 68 frames would cost 68 copies of a megabyte-scale buffer for a
    /// channel that is identical in every one of them; this takes the `Arc` and bumps
    /// a refcount instead.
    ///
    /// A buffer of the WRONG length cannot be shared as-is without breaking the
    /// sampler's bounds contract, so it falls back to the copying
    /// [`with_height_map`](Self::with_height_map) path, which normalizes it.
    #[must_use]
    pub fn with_shared_height_map(mut self, bytes: std::sync::Arc<[u8]>) -> Self {
        let needed = (self.width as usize).saturating_mul(self.height as usize);
        if bytes.len() == needed {
            self.height_map = Some(bytes);
            self
        } else {
            self.with_height_map(bytes.to_vec())
        }
    }

    /// Does this image carry a relief channel? Lets the renderer decide ONCE, at
    /// material-resolve time, whether a relief path is even possible, instead of
    /// testing per cell.
    #[inline]
    #[must_use]
    pub fn has_height_map(&self) -> bool {
        self.height_map.is_some()
    }

    /// Sample the relief channel at `(u, v)` in `[0,1]` (clamped, `v = 0` = top row),
    /// returning a height in `[0,1]`. Uses the SAME nearest-texel addressing as
    /// [`sample_nearest`](Self::sample_nearest), so the colour and the relief of a
    /// texel always come from the same pixel.
    ///
    /// Returns [`NEUTRAL_HEIGHT`](Self::NEUTRAL_HEIGHT) when there is no relief
    /// channel or the image is zero-area — every input shape has a defined output and
    /// nothing here can panic.
    #[inline]
    #[must_use]
    pub fn sample_height_nearest(&self, u: f32, v: f32) -> f32 {
        let Some(hm) = self.height_map.as_ref() else {
            return Self::NEUTRAL_HEIGHT;
        };
        if self.width == 0 || self.height == 0 {
            return Self::NEUTRAL_HEIGHT;
        }
        let uc = u.clamp(0.0, 1.0);
        let vc = v.clamp(0.0, 1.0);
        let px = ((uc * self.width as f32) as u32).min(self.width - 1);
        let py = ((vc * self.height as f32) as u32).min(self.height - 1);
        let idx = (py * self.width + px) as usize;
        // `with_height_map` normalizes the length, but a hand-built struct literal
        // could still be short: fall back to neutral rather than panicking.
        match hm.get(idx) {
            Some(&b) => f32::from(b) / 255.0,
            None => Self::NEUTRAL_HEIGHT,
        }
    }

    /// Sample the nearest texel for `(u, v)` in `[0,1]` (clamped). `v = 0` is the
    /// top row. Returns `[R,G,B,A]`. A zero-area image returns transparent black.
    #[inline]
    pub fn sample_nearest(&self, u: f32, v: f32) -> [u8; 4] {
        if self.width == 0 || self.height == 0 {
            return [0, 0, 0, 0];
        }
        // Clamp UVs into [0,1], map to pixel centers, then to the nearest index.
        let uc = u.clamp(0.0, 1.0);
        let vc = v.clamp(0.0, 1.0);
        let px = ((uc * self.width as f32) as u32).min(self.width - 1);
        let py = ((vc * self.height as f32) as u32).min(self.height - 1);
        let idx = ((py * self.width + px) as usize) * 4;
        [
            self.rgba[idx],
            self.rgba[idx + 1],
            self.rgba[idx + 2],
            self.rgba[idx + 3],
        ]
    }
}

/// The DEFAULT playback rate of an animated [`TextureSet`], in frames per second —
/// the value `[texture NAME] fps = …` falls back to when the author names a
/// `frames = clip.gif` but no rate. Chosen to match the spec's `fps = 10`.
pub const TEXTURE_SET_DEFAULT_FPS: f32 = 10.0;

/// A texture as the renderer binds it: **one or more** [`TextureImage`] frames plus
/// the rate they play at.
///
/// A still image is a set of ONE frame — [`is_animated`](Self::is_animated) is
/// `false`, [`frame_at`](Self::frame_at) ignores `time` and returns that frame, and
/// every consumer behaves exactly as it did when the field was a bare
/// `Arc<TextureImage>`. That is deliberate: the still path must stay byte-identical,
/// so "animated" is a property a texture OPTS INTO by carrying more than one frame,
/// never a branch a still image pays for.
///
/// # Invariants
///
/// * `frames` is **never empty** — every constructor substitutes a zero-area
///   placeholder frame for an empty input, so [`first`](Self::first) /
///   [`frame_at`](Self::frame_at) are total and can never panic or return `None`.
/// * `fps` is finite and `> 0` — a non-finite / non-positive input falls back to
///   [`TEXTURE_SET_DEFAULT_FPS`]. (The `.lvl` parser rejects such a value with a
///   line-numbered error; this is the belt-and-braces so the TYPE is total too.)
///
/// # Why `Vec<Arc<TextureImage>>` and not `Arc<[TextureImage]>`
///
/// The renderer clones INDIVIDUAL frames into a material (`Material::texture` holds
/// frame 0), and the material registry hands the same frame to several materials.
/// Per-frame `Arc`s make both a refcount bump instead of a megabyte memcpy; a
/// `Arc<[TextureImage]>` would force whole-set sharing and block the frame-0 alias
/// that keeps every existing still call site unchanged.
#[derive(Clone, Debug, PartialEq)]
pub struct TextureSet {
    /// The frames, in playback order. Never empty (see the type invariants).
    frames: Vec<std::sync::Arc<TextureImage>>,
    /// Playback rate in frames per second. Finite and `> 0`.
    fps: f32,
}

impl TextureSet {
    /// The zero-area frame every constructor substitutes for an empty frame list, so
    /// the "never empty" invariant holds for ALL inputs. It samples as transparent
    /// black (`TextureImage::sample_nearest` short-circuits on a zero dimension), so a
    /// degenerate set is *defined*, not garbage.
    fn placeholder() -> std::sync::Arc<TextureImage> {
        std::sync::Arc::new(TextureImage::new(0, 0, Vec::new()))
    }

    /// Normalize an authored `fps` to the invariant: finite and `> 0`, else
    /// [`TEXTURE_SET_DEFAULT_FPS`].
    #[inline]
    #[must_use]
    pub fn sanitize_fps(fps: f32) -> f32 {
        if fps.is_finite() && fps > 0.0 {
            fps
        } else {
            TEXTURE_SET_DEFAULT_FPS
        }
    }

    /// A ONE-frame (still) set. The historical shape: `Map::push_texture(img)` builds
    /// this, and it reports `is_animated() == false`.
    #[must_use]
    pub fn still(image: TextureImage) -> Self {
        Self::from_shared_frames(vec![std::sync::Arc::new(image)], TEXTURE_SET_DEFAULT_FPS)
    }

    /// A one-frame set around an ALREADY-shared image (no pixel copy).
    #[must_use]
    pub fn still_shared(image: std::sync::Arc<TextureImage>) -> Self {
        Self::from_shared_frames(vec![image], TEXTURE_SET_DEFAULT_FPS)
    }

    /// A set from owned frames (the GIF-decode path). `fps` is sanitized; an empty
    /// `frames` becomes the one-frame placeholder set.
    #[must_use]
    pub fn new(frames: Vec<TextureImage>, fps: f32) -> Self {
        Self::from_shared_frames(frames.into_iter().map(std::sync::Arc::new).collect(), fps)
    }

    /// A set from already-shared frames (no pixel copy). `fps` is sanitized; an
    /// empty `frames` becomes the one-frame placeholder set.
    #[must_use]
    pub fn from_shared_frames(mut frames: Vec<std::sync::Arc<TextureImage>>, fps: f32) -> Self {
        if frames.is_empty() {
            frames.push(Self::placeholder());
        }
        TextureSet {
            frames,
            fps: Self::sanitize_fps(fps),
        }
    }

    /// All frames, in playback order. Never empty.
    #[inline]
    #[must_use]
    pub fn frames(&self) -> &[std::sync::Arc<TextureImage>] {
        &self.frames
    }

    /// Playback rate, frames per second. Finite and `> 0`.
    #[inline]
    #[must_use]
    pub fn fps(&self) -> f32 {
        self.fps
    }

    /// Number of frames. Always `>= 1`.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Always `false` — the invariant guarantees at least one frame. Present because
    /// clippy asks for it beside [`len`](Self::len).
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Does this set carry MORE THAN ONE frame? The single predicate every consumer
    /// uses to decide whether the texture depends on `time` — the renderer folds it
    /// into `Material::is_animated`, which is what already bypasses the texture cache
    /// (whose key has no time term).
    #[inline]
    #[must_use]
    pub fn is_animated(&self) -> bool {
        self.frames.len() > 1
    }

    /// Frame 0 — what a still texture *is*, and what every pre-animation call site
    /// (aspect/fit derivation, the edge-field precompute, `Map::texture`) reads.
    #[inline]
    #[must_use]
    pub fn first(&self) -> &std::sync::Arc<TextureImage> {
        // The invariant guarantees index 0 exists; `first().expect` would be the same
        // thing with a panic branch the optimizer keeps.
        &self.frames[0]
    }

    /// The frame index for `time` seconds at `fps` over `n` frames — the spec's
    /// `frames[(time·fps) as usize % n]`, made total.
    ///
    /// PURE: the index is a function of exactly these three arguments. No wall clock,
    /// no RNG, no interior mutability — two casts at the same `time` pick the same
    /// frame, which is what keeps `--dump` reproducible and the golden corpus stable.
    ///
    /// Every input shape is defined: `n == 0` → `0`; a negative, NaN or infinite
    /// `time·fps` → frame `0` (Rust's float→int casts saturate, so this cannot be
    /// UB — the explicit guard makes "before the clock started" mean "the first
    /// frame" rather than "the last one").
    #[inline]
    #[must_use]
    pub fn frame_index(time: f32, fps: f32, n: usize) -> usize {
        if n <= 1 {
            return 0;
        }
        let t = time * fps;
        if !t.is_finite() || t <= 0.0 {
            return 0;
        }
        (t as usize) % n
    }

    /// This set's frame index at `time` seconds. See [`frame_index`](Self::frame_index).
    #[inline]
    #[must_use]
    pub fn index_at(&self, time: f32) -> usize {
        Self::frame_index(time, self.fps, self.frames.len())
    }

    /// The frame to show at `time` seconds. A one-frame (still) set returns that
    /// frame for EVERY `time`, so a still texture is time-independent by construction.
    #[inline]
    #[must_use]
    pub fn frame_at(&self, time: f32) -> &std::sync::Arc<TextureImage> {
        &self.frames[self.index_at(time)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fb_alloc_and_default() {
        let fb = Framebuffer::new(4, 3);
        assert_eq!(fb.cols(), 4);
        assert_eq!(fb.rows(), 3);
        assert_eq!(fb.cells().len(), 12);
        assert_eq!(fb.get(0, 0).unwrap().glyph, u32::from(' '));
    }

    #[test]
    fn cell_style_default_is_none() {
        let s = CellStyle::default();
        assert!(s.is_empty(), "default style must be a no-op");
        assert_eq!(s.flags, 0);
        assert_eq!(s.underline, NO_UNDERLINE);
        assert!(!s.is_bold());
        assert!(!s.has_underline_color());
        assert_eq!(s.underline_color(), None);
        // A default Cell carries the default (empty) style.
        assert!(Cell::default().style.is_empty());
    }

    #[test]
    fn cell_style_flag_helpers() {
        let s = CellStyle::default()
            .with_bold()
            .with_flags(CellStyle::ITALIC | CellStyle::STRIKE);
        assert!(s.is_bold());
        assert!(s.is_italic());
        assert!(s.is_strike());
        assert!(!s.is_dim());
        assert!(!s.is_reverse());
        assert!(!s.is_empty());
    }

    #[test]
    fn cell_style_colored_underline_implies_underline() {
        let s = CellStyle::default().with_underline_color(0x0012_3456);
        assert!(s.has_underline_color());
        assert_eq!(s.underline_color(), Some(0x0012_3456));
        // Setting a colored underline turns the underline attribute on so the
        // color is actually drawn.
        assert!(s.is_underline());
        // The colored underline is an INDEPENDENT third channel — it doesn't
        // alter fg/bg or any other flag.
        assert_eq!(s.flags, CellStyle::UNDERLINE);
    }

    #[test]
    fn underline_style_default_is_none_and_unchanged() {
        let s = CellStyle::default();
        // Default carries no shape and is a no-op.
        assert_eq!(s.underline_style(), UnderlineStyle::None);
        assert_eq!(s.effective_underline_style(), UnderlineStyle::None);
        assert!(s.is_empty());
        // A bare shape == None contributes nothing.
        assert_eq!(UnderlineStyle::None.sgr_param(), 0);
        assert!(!UnderlineStyle::None.is_some());
    }

    #[test]
    fn underline_style_sgr_params_match_spec() {
        // 4:n digits per the terminal SGR underline-substyle extension.
        assert_eq!(UnderlineStyle::Single.sgr_param(), 1);
        assert_eq!(UnderlineStyle::Double.sgr_param(), 2);
        assert_eq!(UnderlineStyle::Curly.sgr_param(), 3);
        assert_eq!(UnderlineStyle::Dotted.sgr_param(), 4);
        assert_eq!(UnderlineStyle::Dashed.sgr_param(), 5);
        for s in [
            UnderlineStyle::Single,
            UnderlineStyle::Double,
            UnderlineStyle::Curly,
            UnderlineStyle::Dotted,
            UnderlineStyle::Dashed,
        ] {
            assert!(s.is_some());
        }
    }

    #[test]
    fn with_underline_style_sets_shape_and_implies_underline() {
        // A curly underline accent in a holo hue (the hologram scan-glow case).
        let s = CellStyle::default()
            .with_underline_style(UnderlineStyle::Curly)
            .with_underline_color(0x0000_FFAA);
        assert_eq!(s.underline_style(), UnderlineStyle::Curly);
        assert_eq!(s.effective_underline_style(), UnderlineStyle::Curly);
        // A non-None shape turns on the underline attribute so single-underline
        // backends still draw a rule.
        assert!(s.is_underline());
        assert_eq!(s.underline_color(), Some(0x0000_FFAA));
        assert!(!s.is_empty());
    }

    #[test]
    fn legacy_underline_implies_single_effective_shape() {
        // Pre-substyle cells only ever set the UNDERLINE flag / a colored
        // underline; they must still resolve to a Single rule.
        let flag_only = CellStyle::default().with_underline();
        assert_eq!(flag_only.underline_style(), UnderlineStyle::None);
        assert_eq!(
            flag_only.effective_underline_style(),
            UnderlineStyle::Single
        );

        let color_only = CellStyle::default().with_underline_color(0x0012_3456);
        assert_eq!(
            color_only.effective_underline_style(),
            UnderlineStyle::Single
        );
    }

    #[test]
    fn cell_style_bold_plus_colored_underline_composes() {
        let s = CellStyle::default()
            .with_bold()
            .with_underline_color(0x00FF_8800);
        assert!(s.is_bold());
        assert!(s.is_underline());
        assert_eq!(s.underline_color(), Some(0x00FF_8800));
    }

    #[test]
    fn fb_set_get_roundtrip() {
        let mut fb = Framebuffer::new(2, 2);
        let c = Cell {
            glyph: u32::from('#'),
            fg: 0x00FF_FFFF,
            bg: 0,
            depth: 2.5,
            ..Cell::default()
        };
        fb.set(1, 1, c);
        assert_eq!(*fb.get(1, 1).unwrap(), c);
        // out of bounds is a no-op / None
        assert!(fb.get(5, 5).is_none());
    }

    #[test]
    fn texture_image_normalizes_buffer_len() {
        // Short buffer is zero-padded to w*h*4.
        let img = TextureImage::new(2, 2, vec![1, 2, 3]);
        assert_eq!(img.rgba.len(), 2 * 2 * 4);
        assert_eq!(&img.rgba[0..3], &[1, 2, 3]);
        // Over-long buffer is truncated.
        let img2 = TextureImage::new(1, 1, vec![9; 99]);
        assert_eq!(img2.rgba.len(), 4);
    }

    #[test]
    fn texture_image_sample_nearest_2x2_corners() {
        // 2x2 image, top-left origin, row-major RGBA:
        //   (0,0) red      (1,0) green
        //   (0,1) blue     (1,1) white
        let rgba = vec![
            255, 0, 0, 255, // (0,0) red
            0, 255, 0, 255, // (1,0) green
            0, 0, 255, 255, // (0,1) blue
            255, 255, 255, 255, // (1,1) white
        ];
        let img = TextureImage::new(2, 2, rgba);
        // Sample at the center of each texel: u,v in {0.25, 0.75}.
        assert_eq!(img.sample_nearest(0.25, 0.25), [255, 0, 0, 255]); // top-left red
        assert_eq!(img.sample_nearest(0.75, 0.25), [0, 255, 0, 255]); // top-right green
        assert_eq!(img.sample_nearest(0.25, 0.75), [0, 0, 255, 255]); // bottom-left blue
        assert_eq!(img.sample_nearest(0.75, 0.75), [255, 255, 255, 255]); // bottom-right white
                                                                          // Clamp at the edges: v=0 is the top row.
        assert_eq!(img.sample_nearest(0.0, 0.0), [255, 0, 0, 255]);
        assert_eq!(img.sample_nearest(1.0, 1.0), [255, 255, 255, 255]);
    }

    #[test]
    fn texture_image_zero_area_is_transparent() {
        let img = TextureImage::new(0, 0, Vec::new());
        assert_eq!(img.sample_nearest(0.5, 0.5), [0, 0, 0, 0]);
    }

    /// PROTECTS: an image decoded the ordinary way carries NO relief, and asking it
    /// for a height still yields a defined, FLAT value. This is the un-authored path
    /// the whole no-regression guarantee rests on - if `new` ever started attaching a
    /// channel, every image material in the shipped corpus would stop being self-lit.
    #[test]
    fn a_plain_texture_image_has_no_height_map_and_samples_flat() {
        let img = TextureImage::new(2, 1, vec![0; 8]);
        assert!(
            !img.has_height_map(),
            "TextureImage::new must not attach relief"
        );
        assert_eq!(img.height_map, None);
        assert_eq!(
            img.sample_height_nearest(0.0, 0.0),
            TextureImage::NEUTRAL_HEIGHT
        );
        assert_eq!(
            img.sample_height_nearest(0.9, 0.9),
            TextureImage::NEUTRAL_HEIGHT
        );
    }

    /// PROTECTS: the relief channel addresses the SAME texel as the colour channel.
    /// A row/column transposition here would light a photo with somebody else's
    /// bumps, which reads as noise rather than as a bug.
    #[test]
    fn height_map_samples_the_same_texel_as_the_colour() {
        // 2x2 colour: red / green / blue / white (as in the sampler test above).
        let rgba = vec![
            255, 0, 0, 255, // (0,0)
            0, 255, 0, 255, // (1,0)
            0, 0, 255, 255, // (0,1)
            255, 255, 255, 255, // (1,1)
        ];
        // Height, same row-major order: low, high, mid, max.
        let img = TextureImage::new(2, 2, rgba).with_height_map(vec![0, 255, 128, 255]);
        assert!(img.has_height_map());
        assert_eq!(img.sample_nearest(0.25, 0.25), [255, 0, 0, 255]);
        assert_eq!(img.sample_height_nearest(0.25, 0.25), 0.0);
        assert_eq!(img.sample_nearest(0.75, 0.25), [0, 255, 0, 255]);
        assert_eq!(img.sample_height_nearest(0.75, 0.25), 1.0);
        assert!((img.sample_height_nearest(0.25, 0.75) - 128.0 / 255.0).abs() < 1e-6);
        // Out-of-range UVs clamp, exactly like the colour sampler.
        assert_eq!(img.sample_height_nearest(-5.0, -5.0), 0.0);
        assert_eq!(img.sample_height_nearest(5.0, 5.0), 1.0);
    }

    /// PROTECTS: a mis-sized relief buffer can never make the sampler read out of
    /// bounds. A caller that resampled to the wrong dimensions gets padding or
    /// truncation - never a panic and never a neighbouring allocation's bytes.
    #[test]
    fn a_mis_sized_height_buffer_is_normalized_not_trusted() {
        let short = TextureImage::new(2, 2, vec![0; 16]).with_height_map(vec![200]);
        assert_eq!(short.height_map.as_ref().unwrap().len(), 4, "padded to w*h");
        assert!((short.sample_height_nearest(0.25, 0.25) - 200.0 / 255.0).abs() < 1e-6);
        assert_eq!(short.sample_height_nearest(0.75, 0.75), 0.0, "zero padding");

        let long = TextureImage::new(1, 1, vec![0; 4]).with_height_map(vec![9, 9, 9, 9, 9]);
        assert_eq!(
            long.height_map.as_ref().unwrap().len(),
            1,
            "truncated to w*h"
        );

        // Zero-area: an empty channel, sampled as neutral rather than panicking.
        let empty = TextureImage::new(0, 0, Vec::new()).with_height_map(vec![7, 7]);
        assert_eq!(empty.height_map.as_ref().unwrap().len(), 0);
        assert_eq!(
            empty.sample_height_nearest(0.5, 0.5),
            TextureImage::NEUTRAL_HEIGHT
        );
    }

    /// PROTECTS: the relief channel is shared, not copied. Cloning a `TextureImage`
    /// (which the renderer does when it hands the same source to several materials)
    /// must not duplicate what can be megabytes of height data.
    #[test]
    fn cloning_shares_the_height_channel_rather_than_duplicating_it() {
        let a = TextureImage::new(4, 4, vec![0; 64]).with_height_map(vec![1; 16]);
        let b = a.clone();
        let (ha, hb) = (
            a.height_map.as_ref().unwrap(),
            b.height_map.as_ref().unwrap(),
        );
        assert!(
            std::sync::Arc::ptr_eq(ha, hb),
            "clone must share the Arc, not allocate a second buffer"
        );
    }

    // ── TextureSet: still vs animated frame sets (spec 1.4) ──────────────────

    /// A 1x1 image whose single texel is `v` in every channel — a frame that is
    /// trivially identifiable by its pixel.
    fn frame(v: u8) -> TextureImage {
        TextureImage::new(1, 1, vec![v, v, v, 255])
    }

    /// PROTECTS: a STILL texture is a ONE-frame set — not animated, and its sampling
    /// is time-independent. This is the no-regression contract: everything that held
    /// when `Map::textures` was a bare `Arc<TextureImage>` still holds.
    #[test]
    fn a_still_texture_is_a_one_frame_set_that_ignores_time() {
        let set = TextureSet::still(frame(42));
        assert_eq!(set.len(), 1);
        assert!(!set.is_animated(), "one frame is not an animation");
        for t in [0.0, 0.5, 1.0, 7.25, 1_000.0, -3.0, f32::NAN, f32::INFINITY] {
            assert_eq!(set.index_at(t), 0, "a still set is frame 0 at time {t}");
            assert_eq!(set.frame_at(t).sample_nearest(0.5, 0.5), [42, 42, 42, 255]);
        }
    }

    /// PROTECTS: the frame index is exactly `(time * fps) as usize % n` — it ADVANCES
    /// with time and WRAPS at the end of the clip.
    #[test]
    fn frame_index_advances_with_time_and_wraps() {
        let set = TextureSet::new((0..4).map(|i| frame(i as u8 * 10)).collect(), 10.0);
        assert!(set.is_animated());
        assert_eq!(set.len(), 4);
        // 10 fps ⇒ one frame per 0.1 s.
        assert_eq!(set.index_at(0.00), 0);
        assert_eq!(set.index_at(0.09), 0);
        assert_eq!(set.index_at(0.10), 1);
        assert_eq!(set.index_at(0.25), 2);
        assert_eq!(set.index_at(0.35), 3);
        // Wrap: frame 4 does not exist, so 0.40 s is back to frame 0.
        assert_eq!(set.index_at(0.40), 0);
        assert_eq!(set.index_at(0.55), 1);
        // The pixel follows the index.
        assert_eq!(set.frame_at(0.25).sample_nearest(0.0, 0.0)[0], 20);
        assert_eq!(set.frame_at(0.40).sample_nearest(0.0, 0.0)[0], 0);
    }

    /// PROTECTS: DETERMINISM — the index is a pure function of (time, fps, n). No
    /// clock, no RNG: the same time always picks the same frame, which is what keeps
    /// `--dump` reproducible and the golden corpus stable.
    #[test]
    fn frame_index_is_a_pure_function_of_time_fps_and_count() {
        for &(t, fps, n, want) in &[
            (0.0f32, 10.0f32, 68usize, 0usize),
            (0.1, 10.0, 68, 1),
            (6.7, 10.0, 68, 67),
            (6.8, 10.0, 68, 0),  // exactly one loop
            (13.6, 10.0, 68, 0), // two loops
            (1.0, 1.0, 3, 1),
            (2.999, 1.0, 3, 2),
        ] {
            assert_eq!(
                TextureSet::frame_index(t, fps, n),
                want,
                "t={t} fps={fps} n={n}"
            );
            // Called twice, same answer — no interior state anywhere.
            assert_eq!(TextureSet::frame_index(t, fps, n), want);
        }
    }

    /// PROTECTS: every degenerate input has a DEFINED output — no panic, no UB, no
    /// out-of-range index.
    #[test]
    fn frame_index_is_total_for_degenerate_inputs() {
        assert_eq!(TextureSet::frame_index(1.0, 10.0, 0), 0, "zero frames");
        assert_eq!(TextureSet::frame_index(1.0, 10.0, 1), 0, "one frame");
        assert_eq!(TextureSet::frame_index(-5.0, 10.0, 4), 0, "negative time");
        assert_eq!(TextureSet::frame_index(f32::NAN, 10.0, 4), 0, "NaN time");
        assert_eq!(
            TextureSet::frame_index(f32::INFINITY, 10.0, 4),
            0,
            "infinite time"
        );
        // An empty frame list still yields a usable, samplable set.
        let empty = TextureSet::new(Vec::new(), 10.0);
        assert_eq!(empty.len(), 1);
        assert!(!empty.is_animated());
        assert_eq!(empty.frame_at(9.0).sample_nearest(0.5, 0.5), [0, 0, 0, 0]);
    }

    /// PROTECTS: `fps` is normalized to the type invariant even if a caller bypasses
    /// the parser's line-numbered rejection.
    #[test]
    fn fps_is_sanitized_to_finite_and_positive() {
        for bad in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let set = TextureSet::new(vec![frame(1), frame(2)], bad);
            assert_eq!(set.fps(), TEXTURE_SET_DEFAULT_FPS, "fps {bad} falls back");
        }
        assert_eq!(TextureSet::new(vec![frame(1)], 24.0).fps(), 24.0);
    }

    /// PROTECTS: `first()` is frame 0 for an animated set too — every time-less
    /// consumer (aspect/fit, the edge-field precompute, `Map::texture`) keeps reading
    /// the same image it read before frames existed.
    #[test]
    fn first_is_frame_zero_for_an_animated_set() {
        let set = TextureSet::new(vec![frame(7), frame(8), frame(9)], 12.0);
        assert_eq!(set.first().sample_nearest(0.0, 0.0)[0], 7);
        assert_eq!(set.frames().len(), 3);
    }

    /// PROTECTS: a shared relief channel attaches without copying, and a WRONG-length
    /// one is still normalized rather than corrupting the sampler's bounds.
    #[test]
    fn shared_height_map_attaches_and_normalizes() {
        let bytes: std::sync::Arc<[u8]> = std::sync::Arc::from(vec![200u8].into_boxed_slice());
        let img = frame(9).with_shared_height_map(std::sync::Arc::clone(&bytes));
        assert!(img.has_height_map());
        assert!((img.sample_height_nearest(0.5, 0.5) - 200.0 / 255.0).abs() < 1e-6);
        // The buffer really is SHARED (same allocation), not copied.
        assert!(std::sync::Arc::ptr_eq(
            &bytes,
            img.height_map.as_ref().unwrap()
        ));
        // A wrong-length buffer falls back to the normalizing path — still safe.
        let short: std::sync::Arc<[u8]> = std::sync::Arc::from(Vec::new().into_boxed_slice());
        let img2 = frame(9).with_shared_height_map(short);
        assert!(img2.has_height_map());
        assert!((img2.sample_height_nearest(0.5, 0.5) - 0.0).abs() < 1e-6);
    }
}
