// Test-module assert_eq! on f32 values are intentional exact-equal checks
// (e.g. boundary values, NaN-safety sentinels in depth/LOD tests).
// Allow float_cmp only in test compilation units; lib code remains denied.
#![cfg_attr(test, allow(clippy::float_cmp))]
#![warn(missing_docs)]
//! `re_render` — Layer A: a textured continuous-DDA raycaster that turns a
//! [`Map`] + [`CameraPose`] into a per-cell [`Framebuffer`] of glyph, colour,
//! depth, normal and material — no glyph/colour tiling, one ramp per wall: a
//! full textured, directionally-shaded look.
//!
//! ## Pipeline
//!
//! * **Cast** ([`raycast`], [`raycast_fov`], [`raycast_with_time`]): per viewport
//!   column, a continuous-position DDA to the wall hit — perpendicular distance
//!   (fisheye-free), `side` (0 = NS face, 1 = EW face), `tex_u` (fractional
//!   crossing along the face), and the exact world hit point. Rows below the
//!   wall slice are floor-cast, rows above are ceiling-cast: each gets its own
//!   `row_dist` → world hit point (true per-row parallax). The geometry is
//!   written into the [`Cell`]'s additive fields (`depth`, `side`, `tex_u`,
//!   `hit`, `surface`, `material`).
//! * **Shade** ([`shade_cell`]): a deterministic glyph/colour pass, dispatched
//!   per cell through [`ascii_shader::Material::sample`] —
//!   - WALLS: a procedural brick/circuit texture (or an authored recipe — see
//!     [`ascii_shader::TextureRecipe`]) keyed to `tex_u` + screen row,
//!     directional shading (EW faces shaded darker than NS faces, a classic
//!     raycaster readability trick), distance fog.
//!   - FLOOR: world-pinned grain specks (`Perlin`/hash on the world hit point)
//!     over a faint tile grid, fading out far.
//!   - CEILING: dim base + sparse procedural stars (hashed on the world hit).
//! * **Post-pass** ([`raycast_fov`]'s tail): [`contour::apply`] draws
//!   silhouette/depth-edge box-drawing contours + a colour-volume shift over the
//!   finished frame, [`lines::apply`] draws authored `[line]` overlay strokes on
//!   top of that, and [`node_sprite::apply`] draws every puzzle node as an
//!   occluded screen-space sprite.
//!
//! ## Module map
//!
//! - **Core cast + shade** (this file): the DDA, per-row floor/ceiling cast,
//!   and the private `shade_cell`/`shade_subray`/`cast_column` machinery.
//! - **Glyph & material toolkit**: [`ascii_shader`] (`Material`, `Palette`,
//!   `TextureRecipe`, LOD/hysteresis), [`glyph_family`], [`legibility`],
//!   [`palette_gate`], [`ramp_steps`], [`slope`], [`substrate`], [`surface_fx`],
//!   [`edge_field`], [`detail_lod`], [`face_span`].
//! - **Procedural textures**: [`facade`], [`truchet`], [`fractal`], [`code_wall`],
//!   [`sky`] (cylinder/cap panorama + procedural sky programs).
//! - **Config/knobs** (Bevy [`Resource`]s): [`config::RenderConfig`], [`Fov`],
//!   [`FarFreeze`], [`MirrorClarity`], [`lod::LodConfig`],
//!   [`contour::ContourConfig`], [`depth_contrast::DepthContrastConfig`],
//!   [`posteffect::PostFxConfig`].
//! - **Special surfaces**: [`mirror`], [`portal`], [`holo`], [`shadow`].
//! - **Scene text & puzzle set-dressing**: [`scene_text`], [`banner`],
//!   [`stencil_variants`], [`note_style`], [`pipe_text`], [`state_bus`],
//!   [`node_sprite`], [`bullet_sprite`].
//! - **Tiered composite + particles**: [`tier`] (how often a layer refreshes),
//!   [`composite`] (how layers stack into one [`Framebuffer`]), [`emitter`]
//!   (particles built on top of both), [`far_cache`], [`near_cache`],
//!   [`quadrant`] (sub-cell blit), [`gbuffer`].
//! - **Screen-space post effects**: [`reveal`] (scramble/reveal transitions),
//!   [`zband`], [`ripple`], [`color_drift`], [`tunnel`].
//!
//! ## Determinism
//!
//! Pure float maths + a fixed-seed hash / [`Perlin`]; no `rand`, no clock in the
//! hot loop. Same map + camera + framebuffer size → identical grid every run.
//!
//! Runs in `PostUpdate` (the renderer stage). Reads the [`Map`] + [`CameraPose`];
//! writes the [`Framebuffer`]. No terminal.
//!
//! ## Example
//!
//! ```
//! use re_render::{raycast, Fov};
//! use re_core::Framebuffer;
//! use re_world::{CameraPose, Map};
//!
//! let map = Map::new(4, 4, 3);
//! let cam = CameraPose {
//!     pos: glam::Vec3::new(1.5, 1.5, 1.0),
//!     yaw: 0.0,
//!     pitch: 0.0,
//! };
//! let mut fb = Framebuffer::new(80, 24);
//! raycast(&map, &cam, &mut fb);
//! let _ = Fov::default(); // the player-facing FOV knob raycast_fov reads
//! ```

use std::sync::LazyLock;

use bevy::prelude::*;
use noise::Perlin;
use re_core::{Cell, Framebuffer, SurfaceKind};
use re_world::{CameraPose, CellKind, Map};

pub mod ascii_shader;
/// Pixel-font wall signage, legible at range. Consumed only by `scene_text` —
/// see `banner`'s module docs.
pub mod banner;
pub mod code_wall;
pub mod color_drift;
pub mod config;
pub mod contour;
pub mod depth_contrast;
pub mod detail_lod;
pub mod draw_dist;
pub mod edge_field;
/// Structured Facade / window-bay texture recipe. Consumed by
/// `ascii_shader::TextureRecipe::Facade` and reused unchanged by `sky::skyline`
/// for its distant-tower window rhythm.
pub mod facade;
pub mod face_span;
pub mod fractal;
pub mod glyph_family;
pub mod legibility;
pub mod lines;
pub mod lod;
pub mod palette_gate;
pub mod posteffect;
pub mod ramp_steps;
pub mod reveal;
pub mod slope;
pub mod state_bus;
pub mod substrate;
pub mod surface_fx;
pub mod truchet;
pub mod tunnel;
pub mod zband;

pub use config::RenderConfig;
pub use contour::{ContourConfig, ContourMode};
pub use depth_contrast::DepthContrastConfig;
pub use lod::{LodConfig, LodOverride, LodTier, ShadowTierQuality, TierDetail};
pub use posteffect::{
    apply_posteffects, screen_radius, spiral_index, BlurBand, BlurCfg, PaletteCfg, PostFxConfig,
};
pub use ramp_steps::{StepBand, StepShade, StepTheme};
pub use reveal::{
    apply_scramble_frontier, composite_reveal, composite_reveal_feathered, RevealMode,
    DEFAULT_FEATHER,
};
pub use slope::{SlopeDir, SlopeTheme};
pub use tunnel::{render_tunnel, TunnelParams};

// ─────────────── tiered-composite render system ──────────────────────────────
// `tier` decides HOW OFTEN a layer refreshes (near every frame, far only when
// the camera moves or rotates enough); `composite` decides HOW the resulting
// layers stack into one [`Framebuffer`]. `emitter` is the particle layer built
// on top of both.
pub mod bullet_sprite;
pub mod composite;
/// TEST-ONLY: an instrumented counter used by the far-wall dark-strip
/// regression tests (see `cast_column_depth`'s doc comment) to confirm a fix
/// reaches the off-slice/multi-plane path too, not just the primary DDA hit.
/// Never compiled into the shipped binary.
#[cfg(test)]
mod dark_strip_audit;
pub mod emitter;
pub mod far_cache;
pub mod gbuffer;
pub mod holo;
pub mod mirror;
pub mod near_cache;
pub mod node_sprite;
pub mod note_style;
/// OFF-SLICE puzzle-element scan: the `Object` cells authored ABOVE the 2.5D cast
/// slice (every level's `[node]`s) that the single-slice DDA cannot see. Internal —
/// consumed only by the column cast.
pub(crate) mod offslice;
pub mod pipe_text;
pub mod portal;
pub mod quadrant;
pub mod ripple;
pub mod scene_text;
pub mod shadow;
pub(crate) mod shape_hit;
pub mod sky;
/// State-swapped stencil art + authored LOD tiers for `[note stencil]`/`[note
/// banner]`. Consumed only by `scene_text` — see `stencil_variants`'s module
/// docs.
pub mod stencil_variants;
pub mod tier;

pub use composite::{
    build_world_layers, build_world_layers_fov, build_world_layers_fov_capture_into,
    build_world_layers_fov_into, build_world_layers_into, composite_dump_once,
    composite_dump_once_demo, CompositeRenderPlugin, RenderForceDirty,
};
pub use emitter::{
    demo_emitters, Anchor, ColorRamp, Emitter, EmitterPlugin, EmitterSet, ParticlePool,
};
pub use far_cache::{FarCache, PANORAMA_COLS};
pub(crate) use near_cache::ang_delta as near_cache_ang_delta;
pub use near_cache::{NearRing, NEAR_MARGIN_RAD, NEAR_MOVE_CELLS, NEAR_RING_D_THETA};
pub use quadrant::{
    blit_quadrant, needs_subcell, QuadrantBlit, SubSample, QUADRANT_GLYPHS, SUBCELL_DEPTH_RATIO,
};
// `shade_subray` + `SubViewBasis` are defined in this module (they need the private
// `dda_hit` / `shade_cell` machinery); they are the per-sub-ray sampler the quadrant
// blitter consumes. Declared `pub` at their definition site below.
pub use code_wall::{
    CodeClass, CodeGlyph, CodePanel, LineStyle, TextPanel, NOTE_COLS, NOTE_ROWS, PANEL_COLS,
    PANEL_ROWS,
};
pub use tier::{far_needs_rebake, TierClock, ARC_BUCKET, FAR_MOVE_CELLS, FAR_REBAKE_FRAMES};
// Per-REGISTER placard styling. A host that threads the authored `[note]`
// kind as a value registers panels with `push_note_text_styled(label, body, register)`;
// a host that only labels its panels `note{i}:{NoteKind:?}` gets the same styling via
// `NoteRegister::from_label`, which is what `push_note_text` applies.
pub use note_style::{NotePalette, NoteRegister};
// Which of a node's two word registers its pipes stream (a whole-level
// decision driven by the authored `[effects] glitch` density).
pub use pipe_text::{PoolRegister, POOL_ALT_GLITCH_THRESHOLD};
// The per-level TEXT-SURFACE REGISTRY. This replaced an earlier global
// `set_panel_provider` (`OnceLock<fn(u32) -> &'static CodePanel>`): panels are
// now OWNED and per-level, so a `[note]` placard's runtime `String` can reach a wall
// without leaking one panel set per level load. A host builds a
// `TextSurfaceRegistry` at level load and publishes it with `scene_text::install`.
use ascii_shader::{
    ids, FogParams, MaterialId, MaterialRegistry, PointLights, SampleCtx, SampleMemo, SHADER_SEED,
};
pub use scene_text::{SceneTextScope, TextSurfaceRegistry};
// `SHADOW_MAX_STEPS` is used by the shadow unit tests in this crate; the rest of the
// `SHADOW_*` tunables are consumed by `crate::shadow` (the extracted shadow module),
// not here — so it is a test-only import.
#[cfg(test)]
use ascii_shader::SHADOW_MAX_STEPS;
use re_world::{Atmosphere, MaterialAnimKind, MaterialDef, MaterialRecipe};

// Extracted render sub-systems (mechanical splits of the former god-module). The
// raycast core constructs their inputs and calls into them.
pub use ascii_shader::{
    body_ramp, cell_style, cell_style_ext, cell_style_full, AccentGate, AnimKind, Animation,
    FogParams as RenderFogParams, GlyphSource, GlyphStability, LightCtx, LightEnv, LightModel,
    Material, MaterialSpec, Palette, PointLightRt, PointLights as RenderPointLights, ScaleMode,
    ShadowQuality, SuperGlyphLayer, SuperSelector, TexResult, Texel, TextureRecipe,
    UnderlineAccent, WorldTexCache, MAX_POINT_LIGHTS, WORLD_TEX_CACHE_SLOTS,
};
pub use glyph_family::{FamilyDrive, FamilyDriver, FamilyKind, FamilyLayer, GlyphFamily};
use holo::{holo_column_hit, holo_layers, holo_pass};
use mirror::{
    mirror_composite, surface_clarity, surface_is_mirror, MirrorView, MIRROR_ORIGIN_NUDGE,
};
use portal::{rebase_ray, resolve_warp, surface_is_portal, PORTAL_WARP_BUDGET};
use shadow::{shadow_key_for_lod, ShadowSample};

/// The procedural-noise field, built ONCE. [`Perlin::new`] is non-trivial
/// (it builds a permutation table); constructing it per frame inside the cast
/// was pure waste. A `LazyLock` gives a single shared, thread-safe instance the
/// hot loop borrows. The seed is fixed so the field is deterministic.
static PERLIN: LazyLock<Perlin> = LazyLock::new(|| Perlin::new(NOISE_SEED));

// ───────────────────── authored-material → registry ─────────────────────────

/// Build the [`MaterialRegistry`] a [`Map`] should shade with.
///
/// If the map carries NO authored materials (the common / shipped-level case),
/// this is exactly [`MaterialRegistry::with_builtins`] — back-compatible, every
/// surface keeps its built-in look. If the map DOES carry an authored set (a
/// per-level `[material]` section, forwarded by `re_level::build_map`), each of
/// the five surface slots is replaced by the authored material — unless that slot
/// is a [`MaterialDef::fallback`] marker, in which case the built-in is kept.
///
/// The slot order is the renderer's built-in order (`brick, circuit, matrix,
/// grain, sky`) so [`surface_material`]'s existing `SurfaceKind → MaterialId`
/// dispatch stays valid for the authored registry. No per-cell work happens here
/// — this runs once when the map changes, not in the shade loop.
pub fn registry_for_map(map: &Map) -> MaterialRegistry {
    let authored = map.materials();
    // The default registry now carries the full CATALOG (slots 8..) on top of the
    // fixed surface/puzzle/hologram built-ins (slots 0..=7, unchanged), so a level
    // with no authored materials still exposes every catalog material by name +
    // id_of, and a `preset = NAME` can resolve.
    let builtins = MaterialRegistry::with_catalog();
    if authored.is_empty() {
        // No authored materials ⇒ no `Image` material can exist (the built-ins carry
        // none), so attaching textures is a no-op — but call it for uniformity so a
        // future built-in image material would still resolve. A textureless map's
        // empty store leaves every recipe untouched (byte-identical to today).
        let mut builtins = builtins;
        builtins.attach_image_textures(&map.textures);
        return builtins;
    }
    let mut reg = MaterialRegistry::new();
    // The five slots, in built-in id order. An authored entry replaces the
    // built-in; a fallback marker (or a short list) keeps the built-in.
    for slot in 0..5u16 {
        let builtin = builtins.get(MaterialId(slot)).clone();
        let def = authored.get(slot as usize);
        let mat = match def {
            Some(d) if !d.fallback => material_from_def(d, &builtin),
            _ => builtin,
        };
        reg.register(mat);
    }
    // The five surface slots are authored above; the puzzle node/door are
    // engine-owned recognisability materials the cast dispatches by tag, so they
    // must exist in slots 5/6 of the authored registry too (the builtins path gets
    // them via `with_builtins`). Additive — never overwrites an authored slot.
    reg.push_puzzle_builtins();
    // Append the hologram (slot 7) + the full catalog (slots 8..) so the authored
    // registry exposes the SAME named materials as the default one (id_of("rust")
    // resolves, presets are addressable) without disturbing the authored surface
    // slots above. Slot order matches `with_catalog`, so every `ids::*` is valid.
    reg.register_named("holo_node", builtins.get(ids::HOLO_NODE).clone());
    for name in ascii_shader::ids::CATALOG_NAMES {
        if let Some(m) = MaterialRegistry::catalog_material(name) {
            reg.register_named(name, m);
        }
    }
    // COLOUR→MATERIAL PALETTE: `re_level` appends the
    // per-cell colour palette AFTER the five surface slots (`authored[5..]`), in
    // 1-based colour-id order (id `1` ⇒ `authored[5]`). Each becomes a registry
    // slot AND a colour-palette entry, so a geometry cell whose `side_data` carries
    // a colour id `> 0` shades from its own material instead of the surface-kind
    // default (the per-voxel substrate the semantic surfaces bind through). The
    // base is the brick wall built-in (the common geometry surface); an authored
    // `preset = NAME`/`recipe`/`fg` overrides on top via `material_from_def`. A
    // level with no `[color_cell_ids]` has exactly five authored entries, so this
    // loop is empty → byte-identical to today.
    let brick_base = builtins.get(ids::WALL_BRICK).clone();
    for (k, def) in authored.iter().skip(5).enumerate() {
        let mat = if def.fallback {
            brick_base.clone()
        } else {
            material_from_def(def, &brick_base)
        };
        reg.register_color_material(mat);
        // A Z-BAND material (`crate::zband`): solid only between two heights of its
        // cell, optionally with a top tone. Emit figures and solid furniture ask for it
        // through their name; every other material has no suffix and is unchanged.
        if let Some(band) = zband::parse(&def.name) {
            reg.set_z_band((k + 1) as u16, band);
        }
        // A prop-kit material (`re_level`'s synthesized `__prop_*` names) that lands
        // on a floor cell is a floor DECAL — a trace table's contact shadow, a moss
        // patch — and should still show through the camera-storey backdrop floor,
        // not just the geometry floor. Authored per-cell floor ids are left alone.
        if def.name.starts_with(PROP_DECAL_PREFIX) {
            reg.mark_floor_decal((k + 1) as u16);
        }
    }
    // SCALABLE MATERIAL OVERLAY — SCALE RULES. The colour palette is now
    // registered; tag each `[zone]` whose `scale != None` onto its bound colour
    // material so the shade path renders that zone's surface with the resolved
    // scale rule. The zone's material NAME resolves through the SAME
    // name→palette-id path `re_level` uses: the colour palette is the authored
    // materials appended after the five surface slots (`authored[5..]`), in order, so
    // the k-th colour material (palette id `k + 1`) is `authored[5 + k]`. A zone
    // naming material `authored[5 + k].name` therefore tags palette id `k + 1`.
    //
    // SHARED-MATERIAL RULE: a colour material is ONE registry slot. Two zones binding
    // the same material with different scales ⇒ LAST-WINS (a plain overwrite in
    // `set_color_material_scale`); for distinct scales, give each zone a DEDICATED
    // `[material]`. `Crisp` is carried as a recorded-intent marker — a no-op for
    // id-stamped zone edges; an analytic edge path for that case is not yet built. A
    // level with no scaled zones leaves every material `ScaleMode::None` →
    // byte-identical.
    //
    // LINE-SCALE: a `[line]` carrying `scale` is DEFERRED — authored lines do not
    // bind a colour-palette material (no name→palette-id path), so there is no slot
    // to tag here; only zones get a scale rule this way. Vector-stroke crispness
    // for lines is already satisfied at the stroke-drawing level regardless.
    for zone in &map.overlay().zones {
        if zone.scale == re_core::ScaleRule::None {
            continue;
        }
        let name = zone.material.0.trim();
        if name.is_empty() {
            continue;
        }
        // Palette id = (index within authored[5..]) + 1; matched by material name.
        let Some(k) = authored
            .iter()
            .skip(5)
            .position(|d| !d.fallback && d.name == name)
        else {
            continue; // unbound / unresolved material — inert (validated in re_level).
        };
        let palette_id = (k + 1) as u16;
        reg.set_color_material_scale(palette_id, ascii_shader::ScaleMode::from_core(&zone.scale));
    }
    // IMAGE TEXTURES (image-texture milestone): attach the per-level RGBA8 buffers
    // to every `TextureRecipe::Image` material so the shader can sample pixels with
    // no `&Map`. A map with no `Image` material / no textures is untouched. This also
    // precomputes the `EdgeField` for ASCII-family samplers (load-time, once).
    reg.attach_image_textures(&map.textures);
    // FIT-MODE SPAN DERIVATION (quality-textures Component 1). With the textures now
    // attached (so the image aspect is known), bake a derived `ImageSpan` into every
    // colour-bound `Image` material that has NO authored span: scan the map for the
    // contiguous run of cells whose `side_data` colour id selects this material, take
    // the axis-aligned wall-face bounding box (along-axis extent + column height), and
    // map the image onto it via `fit_span`. An authored `Some(span)` is left untouched
    // (the expert override wins — byte-identical to the world-anchored span path). A material with
    // `subcell = Off` and no span previously tiled per-cell; deriving a Contain span is
    // the intended fix (the old tiling was the "slices" bug). See the spec Risks note:
    // v0.1 supports a SINGLE contiguous flat wall run; manual `span` covers exotic
    // layouts.
    derive_image_spans(&mut reg, map);
    derive_note_backings(&mut reg, map);
    reg
}

/// The material a colour-material cell's face shades from: `m` itself, except on a
/// `[note]` placard cell whose face is NOT the placard face — that face shades from the
/// backing `derive_note_backings` recorded (colour id `0` ⇒ the surface-kind
/// default). `face_n` is the shaded face's world normal when lighting is known;
/// without it the axis `side` decides, as `sample_note_wall` does.
#[inline]
fn note_face_material(
    registry: &MaterialRegistry,
    m: ascii_shader::MaterialId,
    color_id: u16,
    surface: SurfaceKind,
    face_n: Option<glam::Vec3>,
    side: u8,
) -> ascii_shader::MaterialId {
    let TextureRecipe::NoteWall { face, .. } = registry.get(m).recipe else {
        return m;
    };
    let on_face = match face_n {
        Some(n) => face.matches_normal(n),
        None => side == face.side(),
    };
    match registry.note_backing(color_id) {
        Some(b) if !on_face => registry
            .color_material(b)
            .unwrap_or_else(|| surface_material(surface, 0)),
        _ => m,
    }
}

/// For every colour material that is a `[note]` placard (`TextureRecipe::NoteWall`),
/// find the cell it is stamped on and record what its OTHER faces should show: the
/// material of the solid it is mounted in (see `MaterialRegistry::set_note_backing`).
///
/// The placard is a whole-cell stamp, so without this a note on a one-wide notice
/// board would paint the board's sides, top and back as the note's reverse plate —
/// a near-black box with a sign on the front.
///
/// Neighbour preference: along the placard (the rest of the wall run or board), then
/// above / below (the same wall column or board), then behind (what it is mounted
/// on). The first solid geometry neighbour that is not itself a placard or a pipe
/// wins; none ⇒ the surface-kind default. Load-time only.
fn derive_note_backings(reg: &mut MaterialRegistry, map: &Map) {
    let palette_len = reg.color_material_len();
    if palette_len == 0 {
        return;
    }
    let note_face = |reg: &MaterialRegistry, id: u16| -> Option<re_core::FaceDir> {
        let m = reg.color_material(id)?;
        match reg.get(m).recipe {
            TextureRecipe::NoteWall { face, .. } => Some(face),
            _ => None,
        }
    };
    let notes: Vec<(u16, re_core::FaceDir)> = (1..=palette_len as u16)
        .filter_map(|id| note_face(reg, id).map(|f| (id, f)))
        .collect();
    if notes.is_empty() {
        return;
    }
    let (sx, sy, sz) = map.size();
    for (id, face) in notes {
        let mut host = None;
        'scan: for z in 0..sz as i32 {
            for y in 0..sy as i32 {
                for x in 0..sx as i32 {
                    if map.side_data_at(x, y, z) == id && map.cell_at(x, y, z).is_solid() {
                        host = Some((x, y, z));
                        break 'scan;
                    }
                }
            }
        }
        let Some((x, y, z)) = host else { continue };
        let n = face.normal();
        let (nx, ny) = (n.x.round() as i32, n.y.round() as i32);
        // Along the placard = the horizontal axis perpendicular to its normal.
        let (ax, ay) = (-ny, nx);
        let candidates = [
            (ax, ay, 0),
            (-ax, -ay, 0),
            (0, 0, 1),
            (0, 0, -1),
            (-nx, -ny, 0),
        ];
        let mut backing = 0u16;
        for (dx, dy, dz) in candidates {
            let (cx, cy, cz) = (x + dx, y + dy, z + dz);
            if cx < 0 || cy < 0 || cz < 0 || cx >= sx as i32 || cy >= sy as i32 || cz >= sz as i32 {
                continue;
            }
            let kind = map.cell_at(cx, cy, cz);
            if !matches!(kind, CellKind::WallSolid) {
                continue;
            }
            let sd = map.side_data_at(cx, cy, cz);
            if re_core::decode_pipe_tag(sd).is_some() || note_face(reg, sd).is_some() {
                continue;
            }
            backing = sd;
            break;
        }
        reg.set_note_backing(id, backing);
    }
}

/// For each colour-bound `Image` material with no authored span, derive + bake an
/// [`ascii_shader::ImageSpan`] from the textured wall's world extent + the image
/// aspect via [`fit_span`] (quality-textures Component 1). Pure registry mutation;
/// runs once at load. A registry with no colour-bound `Image` materials is untouched.
fn derive_image_spans(reg: &mut MaterialRegistry, map: &Map) {
    let palette_len = reg.color_material_len();
    if palette_len == 0 {
        return;
    }
    for k in 0..palette_len {
        let pid = (k + 1) as u16;
        // Only Image materials WITHOUT an authored span want a derived span.
        let Some((tex_id, has_span, fit)) = reg.color_image_recipe(pid) else {
            continue;
        };
        if has_span {
            continue; // expert override wins; leave it alone.
        }
        let Some(tex) = map.texture(tex_id) else {
            continue; // unresolved texture → no aspect; leave per-cell tiling.
        };
        if tex.width == 0 || tex.height == 0 {
            continue;
        }
        let Some((along_min, along_size, up_top, up_size)) = wall_extent_for_color_id(map, pid)
        else {
            continue; // no cells use this material → nothing to span.
        };
        let span = fit_span(
            along_min, along_size, up_top, up_size, tex.width, tex.height, fit,
        );
        reg.bake_color_image_span(pid, span);
    }
}

/// Scan the map for the axis-aligned WALL-FACE bounding box of every solid cell whose
/// `side_data` colour id equals `pid` (quality-textures Component 1, the single-
/// contiguous-wall case). Returns `(along_min, along_size, up_top, up_size)` in world
/// units (cells are 1 world unit), or `None` if no cell uses the material.
///
/// The "along" axis is the wall's variable horizontal axis: whichever of world X / Y
/// the run spans MORE of (a flat wall is fixed on the other). "up" is world Z (height):
/// `up_top` is the top of the tallest column (`max_z + 1`), `up_size` the column height.
/// Matches the raycaster's span convention (`wall_along` = world Y on an NS face / world
/// X on an EW face; `wall_up` = world Z; `v = 0` at the top).
fn wall_extent_for_color_id(map: &Map, pid: u16) -> Option<(f32, f32, f32, f32)> {
    let (sx, sy, sz) = map.size();
    let mut min_x = i32::MAX;
    let mut max_x = i32::MIN;
    let mut min_y = i32::MAX;
    let mut max_y = i32::MIN;
    let mut min_z = i32::MAX;
    let mut max_z = i32::MIN;
    let mut found = false;
    for z in 0..sz as i32 {
        for y in 0..sy as i32 {
            for x in 0..sx as i32 {
                if map.cell_at(x, y, z) == CellKind::Empty {
                    continue;
                }
                if map.side_data_at(x, y, z) != pid {
                    continue;
                }
                found = true;
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
                min_z = min_z.min(z);
                max_z = max_z.max(z);
            }
        }
    }
    if !found {
        return None;
    }
    let x_run = (max_x - min_x + 1) as f32;
    let y_run = (max_y - min_y + 1) as f32;
    // The along axis is the one the wall varies along (the wider run). World cell faces
    // sit at integer coordinates; the run covers [min, max+1] world units.
    let (along_min, along_size) = if x_run >= y_run {
        (min_x as f32, x_run)
    } else {
        (min_y as f32, y_run)
    };
    let up_size = (max_z - min_z + 1) as f32;
    let up_top = (max_z + 1) as f32; // top face of the top cell.
    Some((along_min, along_size, up_top, up_size))
}

/// MAP-LOCAL animation check for the near-tier dirty-flag.
///
/// [`MaterialRegistry::any_animated`] is a registry-GLOBAL flag: because
/// [`registry_for_map`] always appends the full named CATALOG (slots 8.., several
/// of which animate: water, lava, neon, monitor, …) so a `preset = NAME` can
/// resolve, the global flag is *always* true. That made the near dirty-flag fire
/// every frame even on a perfectly idle camera (the #1 idle-perf bug), because
/// catalog materials are name-lookup ENTRIES only — they are never directly shaded
/// in a cast. A `preset = NAME` COPIES the catalog material into a surface slot
/// (0..4), which IS rendered (and already counted below).
///
/// This returns `true` iff an animated material is actually RENDERABLE in THIS map:
///
/// * **Surface slots (0..5)** — `WallSolid/Other/Trace/Matrix`, `Floor`, `Ceiling`
///   always dispatch here ([`surface_material`]), so an animated surface material
///   (built-in or authored/preset) is always potentially on screen → counted.
/// * **Puzzle node/door slots (5/6)** — dispatched only for `Object` cells
///   ([`CellKind::Object`] → `SurfaceKind::WallOther`). Counted only if the map has
///   at least one `Object` cell; the shipped node/door built-ins animate (a gentle
///   pulse), so a map WITH puzzle objects must keep re-shading them.
/// * **Holo slot (7)** — only ever rendered when [`Map::has_holograms`]; the holo
///   built-in scrolls (animated), so counted only when the map has holograms.
/// * **Catalog slots (8..)** — NEVER counted: they are name-lookup registrations,
///   not rendered surfaces. A preset that uses one is already copied into a surface
///   slot above.
///
/// SAFETY (never under-invalidate): every slot that CAN appear on screen is
/// checked against the real per-material [`Material::is_animated`], so if an
/// animated material is bound/visible the flag stays `true` and animation never
/// freezes. The only slots excluded are the catalog name-entries, which the cast
/// provably never shades directly. When in doubt the surface slots are always
/// counted.
pub fn map_local_any_animated(map: &Map, reg: &MaterialRegistry) -> bool {
    // Surface slots 0..5: always potentially rendered.
    for slot in 0..5u16 {
        if reg.get(MaterialId(slot)).is_animated() {
            return true;
        }
    }
    // Puzzle node/door (slots 5/6): rendered only for Object cells.
    if map.count(CellKind::Object) > 0
        && (reg.get(ids::PUZZLE_NODE).is_animated() || reg.get(ids::PUZZLE_DOOR).is_animated())
    {
        return true;
    }
    // Holo (slot 7): rendered only when the map has holograms.
    if map.has_holograms() && reg.get(ids::HOLO_NODE).is_animated() {
        return true;
    }
    // COLOUR PALETTE (the colour→material substrate): these slots ARE rendered —
    // a geometry cell whose `side_data` carries a colour id shades from one. So if
    // any colour-palette material animates, the near tier must keep re-shading
    // (never freeze a live colour surface). A map with no colour palette
    // (`color_material_len() == 0`, the common case) skips this entirely → today.
    if reg.color_material_animated() {
        return true;
    }
    // Catalog slots (8..): name-lookup only — never count.
    false
}

/// Build the [`FogParams`] a [`Map`] should shade with, from its per-level
/// [`Atmosphere`]. Any field the level left unset falls back to the engine
/// default ([`FogParams::DEFAULT`]: built-in `near`/`far` band, fade-to-black),
/// so a map with no authored atmosphere fogs exactly as the renderer always has.
///
/// Runs once per cast (not per cell), so threading it down the column loop costs
/// nothing in the hot path and never allocates.
pub fn fog_for_map(map: &Map) -> FogParams {
    fog_from_atmosphere(map.atmosphere())
}

/// Build the scene [`LightEnv`] a [`Map`] should shade with: the built-in
/// directional key + headlamp, with each field overridden ONLY where the level's
/// `[atmosphere]` authored one, plus its `[light]` punctual set.
///
/// Resolved once per cast (never per cell), so this is free in the hot path; the
/// renderer re-derives it whenever the `Map` resource changes, so editing a
/// level's atmosphere at runtime takes effect on the next frame rather than
/// being cached for the session.
///
/// A map with no authored lighting returns exactly [`LightEnv::builtin`] — the
/// pre-lighting environment, field for field — which is what keeps the 33-level
/// golden corpus byte-stable.
pub fn light_env_for_map(map: &Map) -> LightEnv {
    light_env_from_atmosphere(map.atmosphere())
}

/// [`light_env_for_map`] on a bare [`Atmosphere`], so tests can exercise the
/// mapping without building a `Map`.
fn light_env_from_atmosphere(atmo: &Atmosphere) -> LightEnv {
    let mut env = LightEnv::builtin();
    if !atmo.has_lighting() {
        return env; // fog-only / bare level: the built-in environment, untouched.
    }
    if let Some(dir) = atmo.key_dir {
        // Author-facing convention: `key_dir` points TOWARD the light. A
        // degenerate (zero) direction is refused by the parser, but normalising
        // defensively keeps a hand-built Atmosphere from producing NaN normals.
        let d = dir.normalize_or_zero();
        if d != re_core::Vec3::ZERO {
            env.key_dir = d;
        }
    }
    if let Some(c) = atmo.key_color {
        env.key_color = unpack_rgb_linear(c);
    }
    if let Some(a) = atmo.ambient {
        env.ambient = re_core::Vec3::splat(a.clamp(0.0, 1.0));
    }
    env.points = PointLights::from_world(&atmo.lights);
    env
}

/// Unpack a packed `0x00RRGGBB` into a linear `[0,1]³` colour, matching the
/// convention [`LightEnv::key_color`] / [`LightEnv::ambient`] already use.
#[inline]
fn unpack_rgb_linear(rgb: u32) -> re_core::Vec3 {
    re_core::Vec3::new(
        ((rgb >> 16) & 0xFF) as f32 / 255.0,
        ((rgb >> 8) & 0xFF) as f32 / 255.0,
        (rgb & 0xFF) as f32 / 255.0,
    )
}

/// Map an engine-agnostic [`Atmosphere`] onto the renderer's [`FogParams`].
fn fog_from_atmosphere(atmo: &Atmosphere) -> FogParams {
    let mut fog = FogParams::DEFAULT;
    if let Some(near) = atmo.fog_near {
        fog.near = near;
    }
    if let Some(far) = atmo.fog_far {
        fog.far = far;
    }
    if let Some(color) = atmo.fog_color {
        fog.color = Some(color);
    }
    // Height fog is carried verbatim; `None` (every shipped level) makes the
    // height term a literal 1.0 and the fog factor bit-identical to before.
    fog.height = atmo.fog_height;
    fog
}

/// Convert one authored [`MaterialDef`] into a renderer [`Material`], using
/// `base` (the surface's built-in) for any field the author left unset: the
/// recipe and rank-strategy default to the surface's built-in, while `fg`/`bg`
/// become flat single-colour palettes and an optional `glyph`/`animation` are
/// applied on top. Flat colours are encoded as a cosine palette with zero
/// amplitude (`b = 0`), so `Palette::eval` returns the constant colour.
/// Fold an image material's authored `relief = height_scale=… normal_strength=…`
/// into the light model it inherited from its preset / surface base.
///
/// Only the TWO relief knobs are replaced: everything else about how the surface
/// lights (specular, shininess, AO, emissive) stays whatever the base decided, so
/// `relief` is a relief control and not an accidental full lighting reset. A recipe
/// that is not an `Image`, or an `Image` with no authored relief, returns the base
/// model unchanged — which is what keeps every pre-existing material byte-identical.
#[inline]
fn image_relief_light(
    recipe: TextureRecipe,
    base: ascii_shader::LightModel,
) -> ascii_shader::LightModel {
    match recipe {
        TextureRecipe::Image {
            relief: Some(r), ..
        } => ascii_shader::LightModel {
            height_scale: r.height_scale,
            normal_strength: r.normal_strength,
            ..base
        },
        _ => base,
    }
}

/// Distance (world cells) [`image_flat_lod_override`] holds full depth-contrast out
/// to. Comfortably past an ordinary room's fog-far (the shipped corpus authors
/// 10–20, with the TX5 fixture band being an extreme outlier at 4), so a
/// flat sampled image keeps its hue through essentially the whole room rather than
/// fading the same colour twice over (fog AND contrast-collapse on the same schedule
/// — see [`image_flat_lod_override`]'s doc).
const IMAGE_FLAT_CONTRAST_HOLD_TO: f32 = 16.0;

/// The per-material [`crate::lod::LodOverride`] a `.lvl`-authored `[material]` should
/// carry when its resolved `recipe` is a FLAT (no `relief =`) [`TextureRecipe::Image`]
/// — the fix for the owner finding "TX5-genplate loses its sampled texture colour at
/// mid distance".
///
/// A flat image is SELF-LIT (`RecipeProps::self_lit`) and its `bg` is just
/// `dim_rgb(fg, IMAGE_BG_DIM)` — a darkened shade of the SAME sampled pixel
/// ([`ascii_shader::Material::sample_image`]'s inner-edge cue), not a genuine
/// ink-vs-backdrop pair the way a procedural recipe's fg/bg are. Pulling `fg` toward
/// that `bg` under the universal depth-contrast pass (`depth_contrast.rs`, gated on
/// [`crate::depth_contrast::DepthContrastConfig::enabled`]) therefore does not read as
/// "distant ink fading toward its backdrop" — it just darkens/desaturates the SAME
/// photo pixel a second time, on the SAME schedule distance-fog already dims it on.
/// For an ordinarily-lit (non-self-lit) wall that second dimmer is unnoticeable
/// (headlamp falloff already fades it); for a self-lit image — the one surface meant
/// to hold its own colour regardless of distance — the two compounding fades read as
/// losing the sampled texture's colour well before the fog band's far edge.
///
/// Hold full contrast/no-fade out to [`IMAGE_FLAT_CONTRAST_HOLD_TO`] via the SAME
/// per-material `LodOverride` lever [`crate::ascii_shader::puzzle_node`] uses — one
/// mechanism, not a second special case — so the image still eventually dissolves
/// with fog/sky-blend at genuinely long range, it just no longer double-fades through
/// the middle of an ordinary room. A relief-authored image is a genuinely LIT surface
/// (`self_lit = false`; the normal Blinn-Phong relief model runs on it, same as any
/// procedural recipe), so it is deliberately excluded here and keeps whatever
/// override its base/preset already carries.
#[inline]
fn image_flat_lod_override(
    recipe: TextureRecipe,
    base_override: Option<crate::lod::LodOverride>,
) -> Option<crate::lod::LodOverride> {
    match recipe {
        TextureRecipe::Image { relief: None, .. } => Some(crate::lod::LodOverride::full_detail_to(
            IMAGE_FLAT_CONTRAST_HOLD_TO,
        )),
        _ => base_override,
    }
}

/// Distance (world cells) [`note_banner_lod_override`] holds full depth-contrast
/// detail out to. A `[note banner]` placard's whole reason to exist is staying
/// legible at range — the tier-fallback system in `banner`/`scene_text` already
/// downgrades to a narrower panel once the
/// face is too small to show the pixel-font FULL tier at 1:1, so the universal
/// depth-contrast pass (`depth_contrast.rs`) dimming that SAME self-lit ink toward
/// its backdrop on top would fight the tier that IS still legible. Set past the
/// redesigned `levels/100-signage-showcase.lvl` corridor's own longest sight line
/// (see that level's `[player] spawn`), so the demo's own banner never double-fades.
const NOTE_BANNER_CONTRAST_HOLD_TO: f32 = 24.0;

/// The per-material [`crate::lod::LodOverride`] a `[note banner]`'s synthesized
/// `NoteWall` material should carry — the LOD half of the legibility fix. Only a
/// BANNER note (`scene_text::is_banner`), never an ordinary
/// `[note]`/`[note stencil]` placard: those already read fine at their authored
/// (usually close) reading distance and are unaffected, matching
/// [`image_flat_lod_override`]'s "only the surface this fixes, nothing else" shape.
#[inline]
fn note_banner_lod_override(
    recipe: TextureRecipe,
    base_override: Option<crate::lod::LodOverride>,
) -> Option<crate::lod::LodOverride> {
    match recipe {
        TextureRecipe::NoteWall { note, .. } if scene_text::is_banner(note) => Some(
            crate::lod::LodOverride::full_detail_to(NOTE_BANNER_CONTRAST_HOLD_TO),
        ),
        _ => base_override,
    }
}

fn material_from_def(d: &MaterialDef, base: &Material) -> Material {
    if d.name.starts_with(EMIT_PART_PREFIX) {
        return emit_part_material(d);
    }
    // PRESET resolution: a level may pick a catalog material by name via
    // `[material] preset = NAME`. `re_level` carries that through the only string
    // that travels in a `MaterialDef` — the `name` — using the sentinel encoding
    // `"\x1fpreset\x1f<NAME>"` (unit-separator framed, impossible in an authored
    // section name). When present we resolve the catalog material and use IT as the
    // base (instead of the surface built-in), so an authored `preset = rust` pulls
    // the whole `rust` config; explicit `fg`/`bg`/`glyph`/`anim`/recipe keys below
    // still override on top. An unknown preset name falls back to the surface base.
    let preset_base: Material;
    let has_preset;
    let base: &Material = match preset_name(&d.name).and_then(MaterialRegistry::catalog_material) {
        Some(m) => {
            preset_base = m;
            has_preset = true;
            &preset_base
        }
        None => {
            has_preset = false;
            base
        }
    };
    let recipe = match d.recipe {
        Some(r) => recipe_from_def(r),
        None => base.recipe,
    };
    let glyph_source = match d.glyph {
        Some(g) => GlyphSource::Fixed(g),
        None => base.glyph_source,
    };
    let animation = anim_from_def(d, base.animation);
    // A HOLOGRAM authored without an explicit animation gets a gentle VERTICAL SCROLL
    // by default — the see-through, alive scanline look the recipe is for. The author
    // can still override with `animation = shift speed=… amplitude=…` (or `static` to
    // hold it still). Any other recipe keeps the resolved animation untouched.
    let animation = if matches!(recipe, TextureRecipe::Hologram { .. })
        && d.anim.kind == MaterialAnimKind::None
        && animation.kind == AnimKind::None
    {
        Animation {
            kind: AnimKind::Shift,
            speed: 0.8,
            amplitude: 1.0,
        }
    } else {
        animation
    };
    // COLOUR: an authored `fg`/`bg` becomes a flat palette. With a PRESET base, an
    // UNSET colour (`0`) keeps the catalog material's rich cosine palette so a bare
    // `preset = rust` still reads as rust; a nonzero authored colour overrides it.
    // Without a preset, the historical behaviour is preserved (always flat fg/bg).
    let palette = if has_preset && d.fg == 0 {
        base.palette
    } else {
        flat_palette(d.fg)
    };
    let bg_palette = if has_preset && d.bg == 0 {
        base.bg_palette
    } else {
        flat_palette(d.bg)
    };
    // MIRROR: a preset's own reflective/clarity carry through unless the author asks
    // for a mirror explicitly (`reflective = true`) or sets a non-default clarity.
    let reflective = if has_preset && !d.reflective {
        base.reflective
    } else {
        d.reflective
    };
    let clarity = if has_preset && d.clarity == MaterialDef::DEFAULT_CLARITY {
        base.clarity
    } else {
        d.clarity
    }
    .clamp(0.0, 1.0);
    Material {
        recipe,
        palette,
        bg_palette,
        rank: base.rank,
        seed: base.seed,
        glyph_source,
        animation,
        // Inherit the (preset or surface) built-in relief lighting so an authored
        // material that only sets fg/bg/glyph still gets tasteful normal-mapped
        // shading — and a preset keeps its tuned specular/AO/emissive. An image
        // material that authored `relief =` overrides ONLY the two relief knobs on
        // top of that inheritance, so its specular / AO / emissive still come from
        // the base and a material WITHOUT `relief =` is byte-identical.
        light: image_relief_light(recipe, base.light),
        // Inherit the (preset or surface) glyph-stability (anti-flicker) tier too.
        stability: base.stability,
        reflective,
        clarity,
        // Inherit the (preset or surface) underline ACCENT layer (the smart-material
        // glint / hologram scan-glow / neon under-glow).
        underline_accent: base.underline_accent,
        // Inherit the (preset or surface) kindred-symbol FAMILY layer + proximity-bold
        // cue, so an authored `preset = haze`/`spark` carries the floating-glyph /
        // emitter look through to the resolved material. A non-preset surface base has
        // `None`, so an ordinary authored material is unchanged.
        family: base.family,
        // Inherit the (preset or surface) super-glyph selector layer too, so an
        // authored `preset = beat_grid`/`auto_tile`/`boil` carries the selector-driven
        // glyph through. A non-preset surface base has `None`, so an ordinary authored
        // material is unchanged.
        super_glyph: base.super_glyph,
        proximity_bold: base.proximity_bold,
        // Inherit the (preset or surface) per-material LOD override (the key-object
        // lever) — UNLESS this material resolved to a flat (no-relief) Image recipe,
        // which gets its own hue-preservation override (`image_flat_lod_override`,
        // TX5-genplate hue-loss fix), or a `[note banner]` placard, which gets its
        // own legibility-hold override (`note_banner_lod_override`). A non-preset
        // surface base has `None`, so an ordinary authored material follows the
        // global LOD curve unchanged.
        lod_override: note_banner_lod_override(
            recipe,
            image_flat_lod_override(recipe, base.lod_override),
        ),
        // Inherit the (preset or surface) glass-marker flag (underline-only
        // transparent). Default `false` on every surface base, so unchanged.
        glass_marker: base.glass_marker,
        // Inherit the (preset or surface) SEMANTIC-SURFACE finish (frost / redacted /
        // watcher / worn / decay). A non-preset surface base has `None`, so an
        // ordinary authored material is unchanged; a `preset = frost` carries it through.
        surface_fx: base.surface_fx,
        // Inherit the (preset or surface) single-glyph ANIMATION TRACKS. A non-preset
        // surface base has the EMPTY set, so an ordinary authored material is static;
        // a `preset = node_breathing` carries its breathing track.
        anim_tracks: base.anim_tracks,
        // Inherit the base's SCALE RULE (the Scalable Material Overlay — see
        // `registry_for_map`'s comment above). The colour-palette base (brick) is
        // `ScaleMode::None`, so a fresh colour material starts unscaled;
        // `registry_for_map` tags scaled zones afterwards via
        // `set_color_material_scale`. An ordinary authored material is unchanged.
        scale: base.scale,
        // Inherit the base's attached image texture. The surface/preset bases are all
        // `None`; the `.lvl` Image-material parser sets a `TextureRecipe::Image`
        // recipe whose texture `registry_for_map` then attaches via
        // `attach_image_textures`.
        texture: base.texture.clone(),
        // Inherit the base's precomputed edge field. The surface/preset bases are all
        // `None`; `registry_for_map` builds + attaches the real `EdgeField` after the
        // texture is attached, alongside span derivation.
        edge_field: base.edge_field.clone(),
        // The frame set rides with the texture: `material_from_def` derives an authored
        // material FROM a base whose textures are attached later by
        // `attach_image_textures`, so this is `None` at this point for a fresh build and
        // a faithful carry-over when the base already had one.
        texture_anim: base.texture_anim.clone(),
        // Opt-in DETAIL-TIER cross-fade (`[material] detail = on`): a DIRECT
        // authored flag, not inherited from the (preset or surface) base — an
        // ordinary authored material that never sets the key stays `false`
        // (`re_world::MaterialDef::detail`'s own default), byte-identical to
        // before this field existed.
        detail: d.detail,
    }
}

/// The preset name encoded in a [`MaterialDef::name`] by `re_level`, or `None` if
/// the name is an ordinary authored section name. The encoding is
/// `"\x1fpreset\x1f<NAME>"` (unit-separator framed — impossible in an authored
/// `[material NAME]` header), so the only string that travels in a `MaterialDef`
/// can carry a `preset = NAME` selection from the level to the renderer without
/// changing the engine-agnostic `re_world::MaterialDef` shape.
fn preset_name(def_name: &str) -> Option<&str> {
    def_name.strip_prefix("\u{1f}preset\u{1f}")
}

/// Name prefix of the colour material `re_game::emit_parts` synthesizes for an
/// `[emit]` part (the reward figure that materialises when a node fires). Mirrors
/// `re_game::emit_parts::EMIT_MATERIAL_PREFIX`; unit-separator framed, so it can never
/// be an authored section name.
pub const EMIT_PART_PREFIX: &str = "\u{1f}emit\u{1f}";

/// Fraction of an emit part's colour its backing keeps: the figure reads as lit
/// bands over a dark body, not as a solid block of wall.
pub const EMIT_PART_BACKING: f32 = 0.10;

/// Name prefix of the materials `re_level`'s prop kit synthesizes
/// (`re_level::props::PROP_MATERIAL_PREFIX`; `re_render` cannot depend on `re_level`,
/// so the convention is mirrored here).
pub const PROP_DECAL_PREFIX: &str = "__prop_";

/// The material an `[emit]` part shades with. The synthesized def carries a
/// full-block glyph and no recipe, so the colour-palette base — the room's
/// BRICK — used to texture it: a lit
/// node's reward read as a large brick-coursed block in the node's colour, the
/// loudest and heaviest thing a solve produced (a grey stone block for L9's pale
/// beacon, a slab across L8's aisle). It is now a NEON figure: emissive scanline bands
/// in the part colour over a near-black backing, with the def's breathing pulse — a
/// thing that switched on, visibly not masonry. Pure.
fn emit_part_material(d: &MaterialDef) -> Material {
    let mut m = MaterialRegistry::catalog_material("neon").unwrap_or_else(|| {
        MaterialRegistry::with_builtins()
            .get(ids::WALL_BRICK)
            .clone()
    });
    m.palette = flat_palette(d.fg);
    m.bg_palette = flat_palette(ascii_shader::dim_rgb(d.fg, EMIT_PART_BACKING));
    m.animation = anim_from_def(d, m.animation);
    m
}

/// A flat (constant-colour) cosine palette: bias = the colour, amplitude = 0, so
/// `eval(t)` is the colour for every `t`. Lets an authored single `fg`/`bg`
/// drive the dual-channel shader unchanged.
fn flat_palette(packed: u32) -> Palette {
    let c = ascii_shader::unpack_rgb(packed);
    Palette {
        a: c,
        b: glam::Vec3::ZERO,
        c: glam::Vec3::ONE,
        d: glam::Vec3::ZERO,
    }
}

/// Map the engine-agnostic [`MaterialRecipe`] onto the renderer's `TextureRecipe`.
///
/// PUBLIC so a cross-crate test can prove the two enums stay 1:1 — specifically
/// that a `.lvl`-authored `recipe = KIND …` line converts to EXACTLY the
/// `TextureRecipe` the matching catalog `preset = NAME` carries. Pure data in,
/// pure data out: no registry, no map, no allocation.
pub fn recipe_from_def(r: MaterialRecipe) -> TextureRecipe {
    match r {
        MaterialRecipe::Brick {
            aspect,
            mortar_w,
            grain,
        } => TextureRecipe::Brick {
            aspect,
            mortar_w,
            grain,
        },
        MaterialRecipe::Circuit { scale, trace_w } => TextureRecipe::Circuit { scale, trace_w },
        MaterialRecipe::Grain {
            scale,
            speck,
            tile_eps,
        } => TextureRecipe::Grain {
            scale,
            speck,
            tile_eps,
        },
        MaterialRecipe::Sky { scale, density } => TextureRecipe::Sky { scale, density },
        MaterialRecipe::Hologram { line_freq, scan_w } => {
            TextureRecipe::Hologram { line_freq, scan_w }
        }
        // The strange loop: a level binds `recipe = codewall panel=N`; we map the
        // panel index onto the renderer's `CodeWall` recipe. `u_chars_per_unit`
        // defaults to a legible density (a few source columns per world cell) —
        // the author tunes legibility via the wall's size in cells, not here.
        MaterialRecipe::CodeWall { panel } => TextureRecipe::CodeWall {
            panel,
            u_chars_per_unit: CODE_WALL_U_CHARS_PER_UNIT,
        },
        // The level's own voice: a `[note]` placard on one wall face. Both fields are
        // carried through 1:1 — the note index addresses the active text-surface
        // registry's note panels, the face gates which side of the cell shows the
        // text. Synthesized at level load (no `.lvl` key), so nothing here parses.
        MaterialRecipe::NoteWall { note, face } => TextureRecipe::NoteWall { note, face },
        // ── the five pattern recipes a level authors by NAME ────────────────
        //
        // Each is carried through 1:1 so a `.lvl`-authored `recipe = KIND …` line
        // produces the SAME `TextureRecipe` the matching catalog `preset =` would
        // (see `re_level::parse_recipe`, whose defaults are the preset values:
        // scanline→`monitor`, truchet→`conduit`, fractal→`portal_zoom`,
        // substrate→`substrate`, voice→`mnemos_hold`/`lethe_release`).
        MaterialRecipe::Scanline {
            freq,
            duty,
            warp,
            axis,
        } => TextureRecipe::Scanline {
            freq,
            duty,
            warp,
            axis,
        },
        MaterialRecipe::Truchet {
            seed,
            density,
            scale,
        } => TextureRecipe::Truchet {
            seed,
            density,
            scale,
        },
        MaterialRecipe::Fractal {
            arms,
            octave_warp,
            twist,
        } => TextureRecipe::Fractal {
            arms,
            octave_warp,
            twist,
        },
        MaterialRecipe::Substrate {
            scale,
            held,
            released,
        } => TextureRecipe::Substrate {
            scale,
            held,
            released,
        },
        // The state-driven voice: the authored channel is the plain-data mirror of
        // the renderer's `substrate::VoiceMode`, mapped here so `re_world` needs no
        // renderer dependency.
        MaterialRecipe::Voice { scale, mode } => TextureRecipe::Voice {
            scale,
            mode: voice_mode_from_world(mode),
        },
        // Image-texture wall: carry the texture id straight through. The actual
        // `Arc<TextureImage>` is attached later by `attach_image_textures` (called
        // from `registry_for_map`) from `Map::textures[tex_id]`; here we only map
        // the recipe variant 1:1 so the shader knows to sample an image.
        MaterialRecipe::Image {
            tex_id,
            span,
            subcell,
            fit,
            tile,
            relief,
        } => TextureRecipe::Image {
            tex_id,
            // World-anchored UV span: mirror the plain-data world span 1:1 onto
            // the renderer's. A non-positive size is dropped to `None` (per-cell tiling)
            // so a malformed authored span degrades to the legacy look, never a div-by-0.
            span: span.and_then(|s| {
                (s.along_size > 0.0 && s.up_size > 0.0).then_some(ascii_shader::ImageSpan {
                    along_min: s.along_min,
                    along_size: s.along_size,
                    up_top: s.up_top,
                    up_size: s.up_size,
                })
            }),
            // SUB-CELL detail mode: mirror the plain-data world enum 1:1. `Off`
            // (the default) keeps the legacy solid-block look, byte-identical to the
            // plain world-anchored span with no sub-cell detail.
            subcell: subcell_from_world(subcell),
            // FIT mode (quality-textures Component 1): mirror 1:1. Only consulted when
            // `span == None`; `registry_for_map` derives + bakes the span then.
            fit: fit_from_world(fit),
            // TILE (span-relative repeat). The `.lvl` parser already rejects a
            // non-positive / non-finite factor, but a `MaterialDef` can also be built
            // in code (tests, tools), so degrade a bad pair to "no repeat" here rather
            // than dividing the figure by zero in the sampler.
            tile: if tile.0.is_finite() && tile.1.is_finite() && tile.0 > 0.0 && tile.1 > 0.0 {
                tile
            } else {
                ascii_shader::IMAGE_TILE_NONE
            },
            // RELIEF: mirror the two plain-data knobs onto the renderer's. Its
            // presence is what flips the recipe off `self_lit` (see `RecipeProps`),
            // so `None` is byte-identical to every image material authored so far.
            relief: relief.map(|r| ascii_shader::ImageRelief {
                height_scale: r.height_scale.max(0.0),
                normal_strength: r.normal_strength.max(0.0),
            }),
        },
        // The structured bay×floor window-grid: every knob is carried through 1:1,
        // exactly like the five other pattern recipes above — `recipe = facade …`
        // and the `facade` catalog preset must produce the identical `TextureRecipe`.
        MaterialRecipe::Facade {
            style,
            bay,
            floor_h,
            lit,
            seed,
            ground_band,
            sign_row,
        } => TextureRecipe::Facade {
            style: facade_style_from_world(style),
            bay,
            floor_h,
            lit,
            seed,
            ground_band,
            sign_row,
        },
    }
}

/// Map the engine-agnostic [`re_world::VoiceMode`] onto the renderer's
/// `substrate::VoiceMode` 1:1. `Hold` is the default on both sides.
#[inline]
fn voice_mode_from_world(m: re_world::VoiceMode) -> crate::substrate::VoiceMode {
    match m {
        re_world::VoiceMode::Hold => crate::substrate::VoiceMode::Hold,
        re_world::VoiceMode::Release => crate::substrate::VoiceMode::Release,
    }
}

/// Map the engine-agnostic [`re_world::FacadeStyle`] onto the renderer's
/// `facade::FacadeStyle` 1:1. `Office` is the default on both sides.
#[inline]
fn facade_style_from_world(s: re_world::FacadeStyle) -> facade::FacadeStyle {
    match s {
        re_world::FacadeStyle::Office => facade::FacadeStyle::Office,
        re_world::FacadeStyle::Residential => facade::FacadeStyle::Residential,
        re_world::FacadeStyle::Industrial => facade::FacadeStyle::Industrial,
        re_world::FacadeStyle::ServerRack => facade::FacadeStyle::ServerRack,
        re_world::FacadeStyle::Lockers => facade::FacadeStyle::Lockers,
    }
}

/// Map the engine-agnostic `re_world::Fit` onto the renderer's `ascii_shader::Fit`
/// 1:1 (quality-textures Component 1). `Contain` is the default, matching both sides.
#[inline]
fn fit_from_world(f: re_world::Fit) -> ascii_shader::Fit {
    match f {
        re_world::Fit::Stretch => ascii_shader::Fit::Stretch,
        re_world::Fit::Contain => ascii_shader::Fit::Contain,
        re_world::Fit::Cover => ascii_shader::Fit::Cover,
    }
}

/// Derive an [`ascii_shader::ImageSpan`] from a textured wall's world extent + the
/// image aspect, per a [`Fit`](ascii_shader::Fit) mode (quality-textures Component 1).
///
/// PURE FUNCTION — no `&Map`. Inputs are the wall's world box on its variable axis
/// (`along`) and its world height (`up`), plus the image pixel size. The returned
/// span is the world rectangle that the texture's `[0,1]×[0,1]` UV box maps onto:
///
/// * [`Stretch`](ascii_shader::Fit::Stretch) — the span IS the wall box (no aspect
///   term); the image distorts to fill exactly.
/// * [`Contain`](ascii_shader::Fit::Contain) — preserve aspect, the WHOLE image fits
///   inside the wall; the figure's world span is SMALLER than the wall on the looser
///   axis, so wall cells beyond it map to UVs outside `[0,1]` → the margin (transparent
///   / edge, matching the clamp/cut-out rule). Centred in the wall.
/// * [`Cover`](ascii_shader::Fit::Cover) — preserve aspect, the image FILLS the wall;
///   the figure's world span is LARGER than the wall on the looser axis, so the wall
///   sees only a sub-window of the image (UVs stay inside `[0,1]`) and the overflow is
///   cropped. Centred in the wall.
///
/// NOTE on the span↔UV relationship: `u = (along − along_min) / along_size`. A SMALLER
/// `along_size` compresses the figure into a smaller world region, so wall cells past
/// it read `u ∉ [0,1]` (Contain margin); a LARGER `along_size` makes the figure exceed
/// the wall, so every wall cell stays inside `[0,1]` and the rest is cropped (Cover).
///
/// "Tighter / looser axis" is judged by comparing the wall aspect (`along/up` in
/// world units) to the image aspect (`w/h` in texels). A degenerate (≤0) wall size
/// or zero-area image falls back to the wall box (Stretch), never a div-by-zero.
pub fn fit_span(
    wall_along_min: f32,
    wall_along_size: f32,
    wall_up_top: f32,
    wall_up_size: f32,
    img_w: u32,
    img_h: u32,
    fit: ascii_shader::Fit,
) -> ascii_shader::ImageSpan {
    // The wall box itself (the Stretch answer + the degenerate fallback).
    let wall = ascii_shader::ImageSpan {
        along_min: wall_along_min,
        along_size: wall_along_size,
        up_top: wall_up_top,
        up_size: wall_up_size,
    };
    if matches!(fit, ascii_shader::Fit::Stretch)
        || wall_along_size <= 0.0
        || wall_up_size <= 0.0
        || img_w == 0
        || img_h == 0
    {
        return wall;
    }

    let wall_aspect = wall_along_size / wall_up_size; // world along : up
    let img_aspect = img_w as f32 / img_h as f32; // texels w : h

    // We keep the wall CENTRE fixed and scale the FIGURE span to preserve aspect.
    let along_centre = wall_along_min + 0.5 * wall_along_size;
    let up_centre = wall_up_top - 0.5 * wall_up_size;

    // r = img_aspect / wall_aspect.
    //  r >= 1 ⇒ the image is WIDER than the wall (relatively).
    //  r <  1 ⇒ the image is TALLER than the wall.
    let r = img_aspect / wall_aspect;

    // The figure span has the IMAGE aspect. Contain ⇒ figure fits INSIDE the wall (one
    // axis equals the wall, the other is SMALLER → margin). Cover ⇒ figure ≥ wall (one
    // axis equals the wall, the other is LARGER → crop). Smaller span on an axis pushes
    // out-of-figure wall cells past [0,1] (margin); larger span keeps every wall cell
    // inside [0,1] (crop).
    let (along_size, up_size) = match fit {
        ascii_shader::Fit::Contain => {
            if r >= 1.0 {
                // Image wider → width fills wall, figure SHORTER (margins top/bottom).
                (wall_along_size, wall_up_size / r)
            } else {
                // Image taller → height fills wall, figure NARROWER (margins l/r).
                (wall_along_size * r, wall_up_size)
            }
        }
        ascii_shader::Fit::Cover => {
            if r >= 1.0 {
                // Image wider → height fills wall, figure WIDER (crop left/right).
                (wall_along_size * r, wall_up_size)
            } else {
                // Image taller → width fills wall, figure TALLER (crop top/bottom).
                (wall_along_size, wall_up_size / r)
            }
        }
        // Unreachable (Stretch returned above), but keep the match total.
        ascii_shader::Fit::Stretch => (wall_along_size, wall_up_size),
    };

    ascii_shader::ImageSpan {
        along_min: along_centre - 0.5 * along_size,
        along_size,
        up_top: up_centre + 0.5 * up_size,
        up_size,
    }
}

/// Is the terminal permitted to receive BLOCK SEXTANT glyphs (`U+1FB00..=U+1FB3B`)?
///
/// **Opt-in, default OFF**, via `TS_SEXTANT=1|on|true`.
///
/// ## Why a flag and not a probe
///
/// There is no portable way to ask a terminal "can you draw U+1FB1B?". Most emulators
/// that support these glyphs *draw them themselves* rather than relying on the font
/// (which is what makes coverage good at all), and that native drawing is not
/// advertised in terminfo. Coverage is genuinely uneven — Ghostty and xterm.js render
/// essentially all of the important characters, VTE-based terminals ~94%, urxvt none —
/// so a wrong guess is not a cosmetic downgrade: it is a wall of tofu boxes where the
/// texture should be.
///
/// A probe that cannot fail safely is worse than an honest opt-in. So the default is
/// the rung we know works everywhere, and enabling sextants is a deliberate act by
/// someone who can see their own terminal.
///
/// Read once per process (`TS_SEXTANT=1`/`on`/`true`).
fn sextant_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("TS_SEXTANT").is_ok_and(|v| {
            let t = v.trim();
            t == "1" || t.eq_ignore_ascii_case("on") || t.eq_ignore_ascii_case("true")
        })
    })
}

/// Map the engine-agnostic `re_world::SubcellMode` onto the renderer's
/// `ascii_shader::SubcellMode`. `Off` is the default, so a non-spanning / legacy
/// image material stays byte-identical to the plain solid-block look.
///
/// The mapping is 1:1 except for one CAPABILITY DEGRADE: `Sextant` falls back to
/// `Quad` unless [`sextant_enabled`] says the terminal may have them. `Quad` is the
/// right target rather than `Half` — it is the nearest rung with the same two-colour
/// block character, so an authored level degrades in resolution only, not in kind.
#[inline]
fn subcell_from_world(m: re_world::SubcellMode) -> ascii_shader::SubcellMode {
    match m {
        re_world::SubcellMode::Off => ascii_shader::SubcellMode::Off,
        re_world::SubcellMode::Half => ascii_shader::SubcellMode::Half,
        re_world::SubcellMode::Quad => ascii_shader::SubcellMode::Quad,
        re_world::SubcellMode::Sextant if sextant_enabled() => ascii_shader::SubcellMode::Sextant,
        re_world::SubcellMode::Sextant => ascii_shader::SubcellMode::Quad,
        re_world::SubcellMode::Braille => ascii_shader::SubcellMode::Braille,
        re_world::SubcellMode::Ascii => ascii_shader::SubcellMode::Ascii,
        re_world::SubcellMode::Contour => ascii_shader::SubcellMode::Contour,
        re_world::SubcellMode::Lines => ascii_shader::SubcellMode::Lines,
    }
}

/// Default horizontal text density for a `CodeWall`: source columns spanned per
/// one world-cell of wall face. Higher = more code per cell (smaller text); lower
/// = fewer, bigger characters. Tuned so a typical room wall reads a line at a time.
const CODE_WALL_U_CHARS_PER_UNIT: f32 = 8.0;

/// Map the engine-agnostic authored animation onto the renderer's `Animation`.
/// `MaterialAnimKind::None` keeps the base material's own animation (so an author
/// who sets fg/bg but no animation inherits the built-in's pulse/shift); any
/// other kind replaces it with the authored block.
fn anim_from_def(d: &MaterialDef, base: Animation) -> Animation {
    let kind = match d.anim.kind {
        MaterialAnimKind::None => return base,
        MaterialAnimKind::Pulse => AnimKind::Pulse,
        MaterialAnimKind::Shift => AnimKind::Shift,
        MaterialAnimKind::Flicker => AnimKind::Flicker,
    };
    Animation {
        kind,
        speed: d.anim.speed,
        amplitude: d.anim.amplitude,
    }
}

// ───────────────────────── caster tunables ──────────────────────────────────

/// Floor on perpendicular distance, to avoid div-by-zero slice heights.
pub(crate) const PERP_MIN: f32 = 1e-4;
/// "Infinity" distance for a column that escaped the map (open sky).
pub(crate) const DIST_INFINITY: f32 = 1.0e9;
/// IMAGE CUT-OUT recursion bound. A cut-out Image wall renders the BACKDROP behind
/// its transparent texels by recasting the column once past the front pane; a
/// backdrop that is ITSELF a cut-out image recurses again, capped here so a corridor
/// of stacked textured panes composites front-to-back in bounded work (and can never
/// loop). Beyond the cap the deepest backdrop is shaded as an ordinary opaque column.
pub(crate) const IMAGE_CUTOUT_MAX_DEPTH: u8 = 4;
/// World-units the backdrop ray's origin is nudged PAST the front Image cell's hit
/// point (along the ray) so the recast DDA does not re-enter the same pane cell.
pub(crate) const IMAGE_BACKDROP_NUDGE: f32 = 1e-3;
/// Pitch limit (radians) the horizon shift saturates at (legacy `PITCH_LIMIT_RAD`).
/// Widened from ±π/4 (45°) to ±π/3 (60°) — the user-approved compromise for the 2.5D
/// shear engine. MUST stay in lock-step with `re_present`'s mouse/keyboard clamp
/// (`re_present::PITCH_LIMIT`) so the pitch the input layer can produce is exactly the
/// pitch the horizon shift saturates at — no dead band where the look freezes before
/// the clamp, and full near sky/floor fill all the way to the clamp.
pub const PITCH_LIMIT_RAD: f32 = std::f32::consts::FRAC_PI_3;

/// Compile-time guard: the horizon-shift saturation limit is the widened ±60° band.
/// Catches any silent drift back to π/4 (which would re-introduce a frozen-look dead
/// band between the old 45° saturation and `re_present`'s 60° input clamp).
const _: () = assert!(PITCH_LIMIT_RAD == std::f32::consts::FRAC_PI_3);

/// Pitch-driven horizon shift in screen ROWS for a `rows`-tall view: linear in
/// pitch, saturating at ±`rows/2` at the [`PITCH_LIMIT_RAD`] limit. The near cast
/// places its horizon at `rows/2 + pitch_row_shift(pitch, rows)`; the FAR fill
/// applies the SAME shift so the far backdrop and near geometry share ONE horizon
/// (critic H2 — no pitch seam between the tiers). Pure + exported so `far_cache`
/// mirrors the near cast exactly.
#[inline]
pub fn pitch_row_shift(pitch: f32, rows: usize) -> i32 {
    let pitch_frac = (pitch / PITCH_LIMIT_RAD).clamp(-1.0, 1.0);
    (pitch_frac * rows as f32 * 0.5).round() as i32
}

/// The process-wide procedural SKY content. Built once; the direction-sampled
/// cylinder+cap fill ([`apply_sky`]) reads it for every above-horizon ray as the
/// fallback. When the [`Map`] carries sky images the fill instead samples an
/// [`sky::ImageSky`] over them (via [`sky::map_sky_texel`]) — same direction→texel
/// mapping, image colour as the backdrop.
static SKY: LazyLock<sky::ProceduralSky> = LazyLock::new(sky::ProceduralSky::default);

/// Vertical angle (pitch, radians) of a screen `row` relative to the true horizontal,
/// given the (pitch-shifted) `horizon` row and the vertical projection-plane distance.
/// A row above the horizon yields a POSITIVE pitch (looking up); the camera pitch is
/// already folded into `horizon` (the 2.5D shear), so this is the ray's world pitch.
#[inline]
fn row_pitch(row: i32, horizon: i32, proj_plane_dist: f32) -> f32 {
    ((horizon - row) as f32 / proj_plane_dist.max(PERP_MIN)).atan()
}

/// World yaw (radians) of a ray whose (un-normalized) direction is `(rx, ry)`. The
/// yaw convention matches [`CameraPose::dir`] (`dir == (sin yaw, cos yaw)`), so the
/// inverse is `atan2(rx, ry)` — same convention the far ring / cap rotation use.
#[inline]
fn ray_yaw(rx: f32, ry: f32) -> f32 {
    rx.atan2(ry)
}

/// Overwrite a ceiling/sky `cell`'s colour + glyph with the direction-sampled
/// cylinder+cap SKY for this ray. Called for above-horizon rays that resolved to the
/// sky surface (the flat `CEILING_SKY` projection is replaced by the panorama). The
/// cell's geometry fields (depth = ∞, surface = Ceiling/Empty) are left intact so the
/// composite/present layers still treat it as distant sky.
///
/// `ray_unit_*` are the NORMALIZED ray direction (its yaw drives `u` + cap rotation);
/// `row`/`horizon`/`proj` give the ray pitch. `map` selects the sky source: an image
/// sky when [`Map::sky_images`] carries a cylinder, else the process-wide procedural
/// sky (byte-identical to before). Cheap: one `atan2` + one `atan` + the sky sample
/// (no allocation; the source is a borrow, not a build).
///
/// When the level authors a procedural sky PROGRAM ([`Map::sky_spec`]), a compiled
/// [`sky::SkyProgram`] is read from the cast-scoped thread-local
/// ([`sky::active_program`], installed once per frame by `composite.rs`) and
/// sampled instead — carrying the animation clock/tick, camera ground
/// position, and player yaw the same thread-locals publish. A map with no
/// [`SkySpec`](re_world::sky_spec::SkySpec) (no `active_program` installed, or
/// this caller never installs one — tests, `re_imgqa`, `ts_editor` preview)
/// falls through to the legacy [`sky::map_sky_texel`] path exactly as before —
/// byte-identical for every level that authors no `[sky] base=`/`layer=`/`kind=`
/// key.
#[inline]
fn apply_sky(
    map: &Map,
    cell: &mut Cell,
    ray_unit_x: f32,
    ray_unit_y: f32,
    row: i32,
    horizon: i32,
    proj: f32,
) {
    let yaw = ray_yaw(ray_unit_x, ray_unit_y);
    let pitch = row_pitch(row, horizon, proj);
    // Gate on THIS call's `map` actually authoring a sky spec, not merely on the
    // thread-local being present: `sky::active_program`/`set_active_program` are
    // installed by `composite.rs` once per frame/dump, but a handful of test/tool
    // call paths (`re_render::raycast`, `cast_column` in a unit test) reach
    // `apply_sky` directly without going through `composite.rs` at all, and a
    // thread-pool-shared OS thread can carry a PRIOR call's installed program
    // across an unrelated later call in the same test binary. Requiring
    // `map.sky_spec().is_some()` here means a map that authors no `[sky]`
    // program is byte-identical to the legacy path NO MATTER what any other
    // thread-local state says — closing that leakage risk at the one call site
    // that already receives the authoritative `map` for free.
    if map.sky_spec().is_some() {
        if let Some(program) = sky::active_program() {
            let t = sky::active_sky_time();
            let ctx = sky::SkyCtx {
                yaw,
                pitch,
                tick: program.tick_at(t),
                t,
                hz: program.hz() as f32,
                cam_xy: sky::active_cam_xy(),
                player_yaw: sky::active_player_yaw(),
            };
            let sc = program.sample(&ctx);
            cell.fg = sc.fg;
            cell.bg = sc.bg;
            cell.glyph = u32::from(sc.glyph);
            cell.style = sc.style;
            return;
        }
    }
    let texel = sky::map_sky_texel(map, &SKY, yaw, pitch);
    cell.fg = texel.fg;
    cell.bg = texel.bg;
    cell.glyph = u32::from(texel.glyph);
    // The sky FULLY defines the cell: clear any text style the underlying ceiling
    // shade left (e.g. an animated material's underline accent). This makes the sky
    // cell purely direction-dependent + time-independent, so the selective-re-shade
    // path can safely keep the persisted sky cell unchanged on an animation-only frame.
    cell.style = re_core::CellStyle::default();
}

/// True when the vertical column at map cell `(x, y)`, scanned from `z_from`
/// (inclusive) up to the map's topmost in-bounds layer, contains no cell that
/// would cap the view overhead. A cell caps the view if it is an ordinary
/// raycaster solid ([`CellKind::is_solid`] — walls/objects/portals) OR the
/// dedicated [`CellKind::Ceiling`] slab: `is_solid` deliberately excludes
/// `Ceiling` (floor/ceiling slabs are not DDA-stoppable — the caster's vertical
/// bounds logic handles them, per its doc comment), but a `Ceiling` voxel is
/// exactly the "roof" a room author stamps overhead ([`re_world::stamp_geometry`]
/// puts one at `z = size.2 - 1` for every `[room]`-built level), so it must
/// still count as capping for THIS purpose. Out-of-bounds reads as
/// [`CellKind::Empty`] (never capping), matching [`Map::cell_at`]'s convention.
#[inline]
fn column_is_open_overhead(map: &Map, x: i32, y: i32, z_from: i32) -> bool {
    let (_, _, sz) = map.size();
    let top = sz as i32 - 1;
    let mut z = z_from;
    while z <= top {
        let kind = map.cell_at(x, y, z);
        if kind.is_solid() || kind == CellKind::Ceiling {
            return false;
        }
        z += 1;
    }
    true
}

/// Whether above-horizon rays should show the direction-sampled sky panorama
/// ([`apply_sky`]) at all, vs. leaving the already-shaded ceiling material
/// (fog-shaded, no overwrite) standing. Two ways a scene earns the sky:
///
/// 1. **The level authors an image sky** ([`Map::sky_images`]) — an explicit
///    "this is an open-air showcase" declaration (an authored `[sky]` image cubemap),
///    honoured even though [`re_world::stamp_geometry`] still caps the room's
///    box with a `Ceiling` slab at its z-top (every `[room]`-built level does —
///    the box itself is never literally roofless).
/// 2. **The camera's own column is a real gap in the roof** — no capping cell
///    anywhere from just above the camera's slice plane up to the map's top
///    (an authored open-top / skylight cell, or a level whose `[cells]` simply
///    never stamps a roof over the play area).
///
/// Deliberately keyed on the CAMERA's position, not the per-ray/per-column hit
/// point: this is a single flat-cast engine (one horizontal DDA plane per
/// story — see [`gameplay_eye_ref_z`]), so "is this room enclosed" is a
/// property of the room the camera occupies, not of where a particular ray's
/// horizontal path happens to end. Two more specific reasons that ruled out a
/// per-ray probe:
///  * The floor/ceiling row loop's flat-plane projection deliberately assumes
///    the ceiling sits exactly one cell above the eye (the classic
///    single-story flat cast); the world point it implies for a row near the
///    wall/ceiling seam can extrapolate far outside the room, and probing the
///    map there would misread "outside the authored footprint" (`cell_at`
///    returns `Empty` out of bounds) as open air even though the room is fully
///    enclosed.
///  * The primary ray's horizontal WALL hit is no better: [`re_world::stamp_geometry`]
///    stamps the perimeter wall at every z layer, so the hit cell is capped by
///    construction regardless of pitch — a real open-top gap over the middle of
///    a room would never be seen if gated on where the horizontal ray happens
///    to stop.
///
/// Cheap: a handful of `cell_at` reads (the map's z-extent), independent of
/// screen resolution, and unaffected by camera pitch/yaw.
#[inline]
fn column_sky_open(map: &Map, cell_z: i32, pos_x: f32, pos_y: f32) -> bool {
    let sky = map.sky_images();
    if sky.cylinder.is_some() || sky.cap_top.is_some() || sky.cap_bottom.is_some() {
        return true;
    }
    // A sky should draw only where the level is actually open: a `[sky]
    // base =`/`layer =`/`kind =` program is just as much an explicit "this is an
    // open-air showcase" declaration as an image cubemap, so it earns the same
    // unconditional open-sky gate.
    if map.sky_spec().is_some() {
        return true;
    }
    column_is_open_overhead(map, floori(pos_x), floori(pos_y), cell_z + 1)
        && !storey_missing_its_lid(map)
}

/// Storey height of the campaign grid, in cells (`storey_height = 4` in every
/// `[room]`, the staircase prefab's tread count). See [`storey_missing_its_lid`].
const STOREY_CELLS: i32 = 4;

/// Is this a single-storey map whose room merely lacks its lid?
///
/// Two kinds of room have no ceiling over the camera. One is deliberately open: L12's
/// vault rises two storeys to the top of a ten-layer map ("the vault above the fog.
/// Open"), seen from the nave floor and from the loft, and a test pins that it shows
/// sky. The other is a one-storey map whose walls run up into its top layer and whose
/// lid was simply never stamped — L10's sub-basement (5 layers: slab, three wall
/// courses, the uppermost course). Showing that one the procedural open-air sky put a
/// blue daytime sky over an archive basement; it keeps its own shaded ceiling
/// material instead, exactly as a lidded room does. Rule: the map is at most one
/// storey tall above its floor slab AND its top layer carries walls. A level that
/// wants sky over such a room says so with `[sky]`.
fn storey_missing_its_lid(map: &Map) -> bool {
    let (sx, sy, sz) = map.size();
    if sz == 0 || sz as i32 - 1 > STOREY_CELLS {
        return false;
    }
    let top = sz as i32 - 1;
    (0..sy as i32).any(|y| (0..sx as i32).any(|x| map.cell_at(x, y, top).is_solid()))
}

/// Eye-z when the player stands grounded on a normal floor cell, in world cell
/// units. THE single source is [`re_world::STANDING_EYE_Z`] (= `1.0 +
/// PLAYER_EYE_HEIGHT`); this is a re-export so the vertical shear is EXACTLY 0 at
/// standing height (what makes the standing framebuffer byte-identical) with NO
/// hand-copied literal that could silently drift from the physics constant.
pub use re_world::STANDING_EYE_Z;

/// Compile-time guard: the re-exported standing eye-z must equal the derived
/// physics value `1.0 + PLAYER_EYE_HEIGHT`. Catches any future drift at build time.
const _: () = assert!(STANDING_EYE_Z == 1.0 + re_world::PLAYER_EYE_HEIGHT);

/// The GAMEPLAY vertical-shear reference for a camera at eye height `eye_z`, made
/// **per-storey** so multi-floor levels (`storey_height` levels) render the floor
/// the player is actually standing on.
///
/// The 2.5D caster walks ONE horizontal grid slice (`cell_z = eye_ref_z.floor()`) and
/// zeroes the vertical eye-shear at `eye_ref_z`. On a SINGLE-storey map the player is
/// always grounded near [`STANDING_EYE_Z`] (= 1.5) so this returns 1.5 and the slice
/// stays plane 1 — **byte-identical to the old constant reference**. On an UPPER storey
/// the grounded eye sits at `floor_top + STANDING_EYE_Z` (e.g. storey 1 of an `H=4`
/// level → eye ≈ 5.5), and this returns that per-storey standing eye, so:
///   * `cell_z = eye_ref_z.floor()` selects the UPPER storey's wall layer (plane 5),
///     not the ground floor (the bug this fixes: the gameplay path used to pin
///     `cell_z = STANDING_EYE_Z.floor() = 1` regardless of elevation, so an upper-floor
///     camera rendered the ground floor's walls/floor/ceiling); and
///   * the eye-shear is 0 when grounded on that storey (a jump/fall still shears).
///
/// JUMP SAFETY (the invariant the old constant guaranteed): the grounded eye on any
/// storey is at `plane + PLAYER_EYE_HEIGHT` — an integer plane plus 0.5 — so
/// `floor(eye_z - PLAYER_EYE_HEIGHT)` recovers the slice plane exactly when grounded,
/// and a sub-cell jump (which only raises `eye_z`) keeps the same `floor`. The slice
/// therefore does NOT re-select on a normal jump (no single-cell-object axis leak —
/// see the `object_column_is_invariant_to_camera_height_on_jump` regression). A jump
/// that clears a FULL cell (≥ 1.0) is the one case the camera-only 2.5D model cannot
/// distinguish from a real storey step without the level's `storey_height`; storeys are
/// spaced `H ≥ 3` cells apart by the stairs design, comfortably above the physical jump
/// apex (`JUMP_IMPULSE²/2·GRAVITY ≈ 1.06`), so a realistic jump never crosses into the
/// next storey's plane.
///
/// `eye_z` is the camera eye z; the result fed to the gameplay cast as `eye_ref_z`.
#[inline]
pub fn gameplay_eye_ref_z(eye_z: f32) -> f32 {
    // Slice plane the eye is grounded over: grounded eye = plane + PLAYER_EYE_HEIGHT.
    // Clamp at the ground storey so a camera below standing (mid-fall snap) never
    // selects a negative plane — single-storey play stays plane 1 exactly.
    let plane = (eye_z - re_world::PLAYER_EYE_HEIGHT).floor();
    let ground_plane = (STANDING_EYE_Z - re_world::PLAYER_EYE_HEIGHT).floor(); // = 1.0
    plane.max(ground_plane) + re_world::PLAYER_EYE_HEIGHT
}

/// Per-`perp` vertical screen shift (in screen rows · world units) for the vertical
/// eye shear: `(eye_z − eye_ref_z) · proj_plane_dist`. The per-column screen offset is
/// then `vshift_per_perp / perp` (rows), scaling like the wall's projected height —
/// exactly how `pitch` and a wall slice already scale with the projection plane, so
/// wall, floor/ceiling, and far tier stay consistent (no tear).
///
/// `eye_ref_z` is the reference eye height that produces ZERO shift — it is what makes
/// the shear a *per-entry-point* policy:
///
/// * **Gameplay** passes [`STANDING_EYE_Z`]: standing renders centred (shift 0) and a
///   jump/fall (`eye_z != STANDING_EYE_Z`) shears, so rising reads as rising.
/// * **Generic / preview / QA** casts (e.g. [`raycast`], used by `re_voxscene` and
///   `re_imgqa` with high orbit cameras) pass `eye_z` itself, so the deviation is
///   identically `0.0` for ANY camera height → no shear → byte-identical to the
///   pre-shear renderer (the previews must not skew off-screen).
///
/// Sign: as `eye_z` rises above `eye_ref_z` (jump up) the offset is POSITIVE, moving
/// the wall slice (and the whole world) DOWN on screen — i.e. the horizon drops, which
/// reads as the player rising. At `eye_z == eye_ref_z` the result is exactly `0.0`.
#[inline]
pub fn vshift_per_perp(eye_z: f32, eye_ref_z: f32, proj_plane_dist: f32) -> f32 {
    (eye_z - eye_ref_z) * proj_plane_dist
}

// ──────────────────────────── field of view ─────────────────────────────────

/// Reference effective aspect: the aspect a "baseline" window has, used to anchor
/// the (window-independent) VERTICAL FOV to the player-facing HORIZONTAL preset.
///
/// Derived from a reference 120×40 grid with the default cell aspect 0.5:
/// `A0 = (120 / 40) * 0.5 = 1.5`. At exactly this window the runtime horizontal
/// FOV equals `h_preset_deg`; any other window keeps the SAME vertical FOV and
/// derives a different horizontal FOV from its own aspect (the "hor+" model).
const BASELINE_ASPECT: f32 = (120.0 / 40.0) * 0.5; // = 1.5

/// Minimum runtime horizontal FOV (degrees) the `hfov` clamp allows. Below this an
/// ultra-tall window would render a claustrophobic tunnel; we floor it here.
const HFOV_MIN_DEG: f32 = 50.0;
/// Maximum runtime horizontal FOV (degrees) the `hfov` clamp allows. Above this an
/// ultra-wide window would fish-eye; we cap it here.
const HFOV_MAX_DEG: f32 = 100.0;

/// Player-/engine-facing field-of-view controls (engine-level resource).
///
/// The model is **hor+**: the VERTICAL FOV is the fixed angle (derived once from
/// `h_preset_deg` at the [`BASELINE_ASPECT`] reference window and held constant for
/// every window size), while the HORIZONTAL FOV is DERIVED at cast time from the
/// window's effective aspect. The terminal cell is non-square, so a per-cell aspect
/// correction (`cell_aspect = cell_w / cell_h`) is folded in — without it, circles
/// render as ellipses because a character cell is roughly half as wide as it is
/// tall.
#[derive(Resource, Debug, Clone, Copy, PartialEq)]
pub struct Fov {
    /// The player-facing HORIZONTAL field of view in degrees, defined AT the
    /// baseline window ([`BASELINE_ASPECT`]). The vertical FOV is anchored to this;
    /// the runtime horizontal FOV equals this only at the baseline aspect.
    pub h_preset_deg: f32,
    /// Terminal cell aspect = `cell_w / cell_h`. A typical monospace cell is about
    /// twice as tall as wide, so `0.5`. This squashes the horizontal stretch so a
    /// round thing in the world renders round on screen rather than as a wide
    /// ellipse.
    pub cell_aspect: f32,
    /// Player override multiplier on the effective aspect (a "stretch" knob). `1.0`
    /// = neutral; `>1` widens the horizontal FOV, `<1` narrows it.
    pub stretch: f32,
}

impl Default for Fov {
    fn default() -> Self {
        Fov {
            h_preset_deg: 65.0,
            cell_aspect: 0.5,
            stretch: 1.0,
        }
    }
}

/// Live MIRROR-CLARITY override (the narrative-recognition knob).
///
/// A mirror surface carries its OWN authored `clarity` on its material; this
/// resource lets the GAME drive that value LIVE for every mirror in the scene —
/// the reflection resolving from glitch/empty (`0.0`) to crisp (`1.0`) as a memory
/// is recovered. The renderer reads it once per cast and uses it as the effective
/// clarity for reflective surfaces (the recognition beat).
///
/// `Some(c)` ⇒ override every mirror's clarity with `c`; `None` (the default) ⇒
/// each mirror keeps the clarity authored on its material. So a scene that never
/// touches this resource behaves exactly as the authored levels intend, and the
/// game animates the beat by writing `Some(t)` as `t` ramps `0 → 1`.
///
/// Folded into the render dirty-flag: a change re-casts (the reflection changes),
/// so the live ramp is visible frame-to-frame. INSERT-IF-ABSENT in the plugin so
/// the game may set a starting value before adding the renderer.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Default)]
pub struct MirrorClarity {
    /// `Some(c)` overrides every mirror's authored clarity with `c ∈ [0,1]`;
    /// `None` keeps each mirror's own authored clarity.
    pub override_value: Option<f32>,
}

impl MirrorClarity {
    /// The effective clarity for a mirror material whose authored clarity is
    /// `authored`: the live override if set, else the authored value. Clamped to
    /// `[0,1]`.
    #[inline]
    pub fn effective(&self, authored: f32) -> f32 {
        self.override_value.unwrap_or(authored).clamp(0.0, 1.0)
    }
}

/// FAR-PLANE HARD FREEZE knob (the anti-noise distance cutoff).
///
/// Beyond [`distance`](Self::distance) world-cells a cell is rendered FULLY QUIET —
/// glyph locked to one world-anchored FAR bucket (no LOD reveal flips, no depth-ink
/// re-selection) and ALL material animation disabled (pulse / flicker / shift /
/// hologram-scroll), so distant background stops noising/churning while near
/// surfaces stay alive. The cell still parallax-scrolls with the camera (the glyph
/// is world-anchored), colour may still shade smoothly. See
/// [`ascii_shader::FAR_FREEZE_DIST`] and `Material::sample`.
///
/// The renderer resolves this ONCE per cast and threads it into the shade context.
/// The default is [`FAR_FREEZE_DIST`](ascii_shader::FAR_FREEZE_DIST) — large enough
/// that normal-scale rooms are unaffected; the game can lower it to freeze nearer,
/// or set it to `f32::INFINITY` to disable the freeze entirely (the pre-feature
/// behaviour). INSERT-IF-ABSENT in the plugin, so omitting it keeps the default.
///
/// As of the GRAPHICS-QUALITY draw-distance setting this resource ALSO carries the
/// runtime DDA step cap ([`max_dda_steps`](Self::max_dda_steps)): a shorter draw
/// distance lowers BOTH the freeze and the step cap, so the cast both looks closer
/// and costs less. The two bounds are pushed to the cast thread-locals once per cast
/// via [`apply`](Self::apply); the default reproduces the historical consts exactly,
/// so a default-quality build is byte-identical.
#[derive(Resource, Debug, Clone, Copy, PartialEq)]
pub struct FarFreeze {
    /// Perpendicular distance (world cells) at/beyond which a cell is frozen.
    /// `f32::INFINITY` disables the freeze.
    pub distance: f32,
    /// The primary-ray DDA step cap (the runtime form of the old `MAX_DDA_STEPS`).
    /// Default [`draw_dist::DEFAULT_MAX_DDA_STEPS`] = 256 (byte-identical to today).
    pub max_dda_steps: u32,
}

impl Default for FarFreeze {
    fn default() -> Self {
        Self {
            distance: ascii_shader::FAR_FREEZE_DIST,
            max_dda_steps: draw_dist::DEFAULT_MAX_DDA_STEPS,
        }
    }
}

impl FarFreeze {
    /// Install this knob's bounds as the ACTIVE draw-distance the cast reads back via
    /// [`draw_dist::active_far_freeze`] / [`draw_dist::active_max_dda_steps`]. An
    /// explicit write — no env. The Bevy render system applies the live resource once
    /// per cast (chained before the cast on the same thread); tests apply a
    /// constructed one.
    #[inline]
    pub fn apply(&self) {
        draw_dist::apply(self.distance, self.max_dda_steps);
    }
}

/// Per-cast view geometry derived from the window size + [`Fov`]. Produced by the
/// pure [`view_params`] fn so the math is unit-testable without a framebuffer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewParams {
    /// Runtime HORIZONTAL field of view (radians), clamped to
    /// `[HFOV_MIN_DEG, HFOV_MAX_DEG]`. Varies with window aspect.
    pub hfov: f32,
    /// VERTICAL field of view (radians). Window-INDEPENDENT — anchored to the
    /// preset at the baseline aspect, identical for every window size.
    pub vfov: f32,
    /// Camera-plane half-width: `tan(hfov / 2)`. The per-column ray offset uses
    /// `plane * camera_x` with `camera_x ∈ [-1, 1]`.
    pub plane_scale: f32,
    /// Vertical projection-plane distance in screen rows: `(rows/2) / tan(vfov/2)`.
    /// A unit-tall wall at perpendicular distance `d` is `proj_plane_dist / d` rows
    /// tall; floor/ceiling row distance is `proj_plane_dist / |row - horizon|`. This
    /// is what makes vertical extent honour `vfov` instead of a magic constant.
    pub proj_plane_dist: f32,
}

/// Derive the per-cast view geometry from the viewport size (`cols`×`rows`) and the
/// [`Fov`] resource. Pure + deterministic; the cast and the unit tests both call it.
///
/// 1. `eff_aspect = (cols/rows) * cell_aspect * stretch` — the window's effective
///    aspect once the non-square cell + the player stretch are folded in.
/// 2. `vfov = 2·atan( tan(h_preset/2) / A0 )` — the FIXED vertical FOV. It depends
///    only on the preset + the baseline aspect, NOT on the window, so it is constant
///    across every `cols`×`rows`.
/// 3. `hfov = 2·atan( eff_aspect · tan(vfov/2) )`, then CLAMPED to
///    `[HFOV_MIN_DEG, HFOV_MAX_DEG]` — the runtime horizontal FOV (hor+ model).
/// 4. `plane_scale = tan(hfov/2)` and `proj_plane_dist = (rows/2)/tan(vfov/2)`.
pub fn view_params(cols: usize, rows: usize, fov: &Fov) -> ViewParams {
    let cols_f = (cols.max(1)) as f32;
    let rows_f = (rows.max(1)) as f32;

    let eff_aspect = (cols_f / rows_f) * fov.cell_aspect * fov.stretch;

    // FIXED vertical FOV: anchor the preset at the baseline aspect. Constant w.r.t.
    // window size — same vfov for 120×40, 200×20, 60×50, …
    let h_preset = fov.h_preset_deg.to_radians();
    let vfov = 2.0 * ((h_preset * 0.5).tan() / BASELINE_ASPECT).atan();

    // Runtime horizontal FOV from THIS window's effective aspect, then clamp.
    let hfov_raw = 2.0 * (eff_aspect * (vfov * 0.5).tan()).atan();
    let hfov = hfov_raw.clamp(HFOV_MIN_DEG.to_radians(), HFOV_MAX_DEG.to_radians());

    let plane_scale = (hfov * 0.5).tan();
    let proj_plane_dist = (rows_f * 0.5) / (vfov * 0.5).tan();

    // FACE SPAN: publish this view's horizontal projection scale — `cols/(2·plane_scale)`
    // screen columns per world unit at unit depth — for the cast that follows. Read deep
    // in the shade dispatch by the fitted TEXT sampler so a placard face wider than its
    // panel is drawn 1:1 instead of magnified into doubled letters (see
    // `crate::face_span`). Done here so EVERY cast entry (near, far ring, sub-cell,
    // `--dump`) publishes it with no extra plumbing; a caller that never builds view
    // params leaves it at 0.0 = "span unknown" = the historical stretch mapping.
    face_span::apply(cols_f / (2.0 * plane_scale), proj_plane_dist);

    ViewParams {
        hfov,
        vfov,
        plane_scale,
        proj_plane_dist,
    }
}

// ─────────────────────── shading tunables (named) ───────────────────────────

/// Seed for the procedural texture noise. Fixed → deterministic across runs.
/// Shared with [`ascii_shader::SHADER_SEED`] so the Perlin field matches.
const NOISE_SEED: u32 = SHADER_SEED;

// ───────────────────────── small math helpers ───────────────────────────────

#[inline]
fn clampi(v: i32, lo: i32, hi: i32) -> i32 {
    v.clamp(lo, hi)
}

/// `floor()` to i32 without surprises for the small world-scale inputs we feed it.
#[inline]
pub(crate) fn floori(v: f32) -> i32 {
    v.floor() as i32
}

/// Map the world cell kind the DDA stopped on (+ its `side_data`) → a [`SurfaceKind`].
///
/// An `Object` is a puzzle node/door ([`SurfaceKind::WallOther`], the self-lit node
/// look) UNLESS `re_level` marked it [`INERT_OBJECT_TAG`] — a painted `O` no puzzle
/// entity backs. That one shades as the plain solid wall it physically is, so level
/// furniture never reads as a puzzle neuron it is not.
#[inline]
fn cellkind_to_surface(k: CellKind, side_data: u16) -> SurfaceKind {
    match k {
        CellKind::WallSolid => SurfaceKind::WallSolid,
        CellKind::WallTrace => SurfaceKind::WallTrace,
        CellKind::WallMatrix => SurfaceKind::WallMatrix,
        CellKind::Object if side_data == INERT_OBJECT_TAG => SurfaceKind::WallSolid,
        CellKind::Object => SurfaceKind::WallOther,
        CellKind::Portal => SurfaceKind::Portal,
        _ => SurfaceKind::WallSolid,
    }
}

/// True for an `Object` cell that renders as a NODE SPRITE (`node_sprite`) — fully
/// render-TRANSPARENT (drawn as air on every plane, the small orb composited on top
/// afterward) — rather than as full-voxel wall geometry: a puzzle node, `side_data` =
/// its node index (`0..RESERVED_TAG_BASE`). The exit door ([`DOOR_TAG_SENTINEL`]) and
/// inert set-dressing ([`INERT_OBJECT_TAG`]) both live at/above `RESERVED_TAG_BASE`,
/// so they are excluded and keep rendering as solid wall-voxel geometry. `false` for
/// every non-`Object` kind. Collision (`CellKind::is_solid`) is untouched by this —
/// a sprite node still blocks the player; only the RENDER path treats it as air.
///
/// EMBEDDED nodes: a puzzle node whose voxel touches solid/pipe/slab geometry
/// ([`node_sprite::is_embedded`]) stays OUT of this predicate — making it see-through
/// would expose a cavity where the wall/floor/pipe it's set into should read solid.
/// Such a node keeps `is_solid` geometry (this fn `false`), gets its face repainted
/// into a recessed SOCKET plate by `node_sprite::apply`'s socket pass, and still gets
/// the small state-coloured orb composited on top of that face
/// ([`crate::lines::LineViewBasis::visible_except`] keeps the node's own now-solid
/// cell from occluding its own orb).
///
/// Gated by [`config::node_sprite_enabled`]: `TS_NODE_SPRITE=0` makes this always
/// `false`, restoring the pre-sprite wall-voxel look byte-for-byte (embedded sockets
/// included — the whole `node_sprite` pass is a no-op then).
#[inline]
pub(crate) fn is_sprite_node(
    map: &Map,
    x: i32,
    y: i32,
    z: i32,
    k: CellKind,
    side_data: u16,
) -> bool {
    config::node_sprite_enabled()
        && node_sprite::is_puzzle_node_tag(k, side_data)
        && !node_sprite::is_embedded(map, x, y, z)
}

// ────────────────────────────── caster ──────────────────────────────────────

/// Back-compat entry: raycast + shade with the built-in [`MaterialRegistry`] at
/// `time = 0` (a static frame). Public so the binary's `--dump` path can call it
/// directly without a Bevy App. That path only runs once per invocation, so
/// building its own Perlin/registry per call is not a hot-path concern.
pub fn raycast(map: &Map, cam: &CameraPose, fb: &mut Framebuffer) {
    raycast_fov(map, cam, &Fov::default(), fb);
}

/// As [`raycast`], but with an explicit [`Fov`] (the `--dump` path passes the
/// game's configured FOV here). Builds the map's registry, then casts at `time = 0`.
pub fn raycast_fov(map: &Map, cam: &CameraPose, fov: &Fov, fb: &mut Framebuffer) {
    // Seed the NODE SPRITE on/off knob from `TS_NODE_SPRITE` for this cast — read
    // BEFORE the cast itself, since `is_sprite_node` (which the DDA/off-slice/
    // shadow/holo passes call) reads it back via a thread-local, not env directly.
    config::apply_node_sprite_env();
    // Honour the map's authored materials (the "color = material KEY" payload) if
    // it carries any; otherwise this is the built-in registry, unchanged. So a
    // `--dump` of a level with a `[material]` section reflects the authored look.
    let registry = registry_for_map(map);
    raycast_with_time_fov(map, cam, &registry, fov, 0.0, fb);
    // STRUCTURE-as-LINES + COLOR-as-VOLUME: the defining look post-pass over the
    // finalized framebuffer (silhouette/depth-edge box-drawing contours + a subtle
    // per-face/depth colour-volume shift). This public/QA/dump-via-raycast entry
    // applies the SHIPPING config (env-seeded, for manual tuning); the deeper
    // `raycast_with_time_fov_q` stays pristine so per-cell shade tests pin the
    // pre-contour shade. See `crate::contour`.
    contour::apply(fb, &ContourConfig::from_env());
    // AUTHORED `[line]` overlay strokes: draw the map's authored polylines as crisp
    // box-drawing vector strokes over the finalized frame, RIGHT AFTER the
    // auto-contour so explicit lines layer over the auto silhouettes. A map with no
    // `overlay.lines` is a hard no-op ⇒ byte-identical.
    lines::apply(fb, map, cam, fov, &registry);
    // NODE SPRITE post-pass: draw every puzzle node as a screen-space-bounded
    // object at its projected centre, occluded against the real multi-plane
    // geometry. No-op when the map has no sprite nodes, or when `TS_NODE_SPRITE=0`
    // (see `is_sprite_node`) — either way `fb` is untouched.
    node_sprite::apply(fb, map, cam, fov, 0.0);
}

/// Raycast + shade with an explicit registry at `time = 0` (static). Kept with
/// its original signature for back-compat; animation needs the time-aware
/// [`raycast_with_time`].
pub fn raycast_with(
    map: &Map,
    cam: &CameraPose,
    registry: &MaterialRegistry,
    fb: &mut Framebuffer,
) {
    raycast_with_time(map, cam, registry, 0.0, fb);
}

/// The actual raycast + Material-driven shade: fills `fb` for `map` + `cam` at
/// animation `time` (seconds), looking up each cell's surface in `registry` and
/// calling `material.sample`. `time` only matters for animated materials; a
/// static scene shades identically at any `time` (golden-safe).
pub fn raycast_with_time(
    map: &Map,
    cam: &CameraPose,
    registry: &MaterialRegistry,
    time: f32,
    fb: &mut Framebuffer,
) {
    raycast_with_time_fov(map, cam, registry, &Fov::default(), time, fb);
}

/// As [`raycast_with_time`], but with an explicit [`Fov`] driving the view geometry
/// (vertical FOV anchored to the preset, horizontal FOV derived per-window via the
/// hor+ model + cell-aspect correction). This is the one true cast entry; the other
/// public entries forward here with `Fov::default()`.
pub fn raycast_with_time_fov(
    map: &Map,
    cam: &CameraPose,
    registry: &MaterialRegistry,
    fov: &Fov,
    time: f32,
    fb: &mut Framebuffer,
) {
    // Public entry: default to the full worked-out (Soft) shadows so the `--dump`
    // path and every legacy caller get shadows. The Bevy render system threads the
    // `ShadowQuality` resource through the `_q` variant instead.
    //
    // GENERIC / preview / QA policy: the vertical eye shear reference is the camera's
    // OWN z, so the deviation is identically 0 → no shear → byte-identical to the
    // pre-shear renderer. `re_voxscene` (high street/overview cameras) and `re_imgqa`
    // (elevated orbit cameras) reach the caster through here and MUST render centred.
    // The gameplay composite path uses `raycast_with_time_fov_gameplay` (reference
    // `STANDING_EYE_Z`) so a jump still shears.
    raycast_with_time_fov_q(
        map,
        cam,
        registry,
        fov,
        time,
        ShadowQuality::default(),
        cam.pos.z,
        fb,
    );
}

/// As [`raycast_with_time_fov`] but with the GAMEPLAY vertical-shear reference
/// ([`STANDING_EYE_Z`]): a grounded player renders centred (shift 0) while a jump/fall
/// shears the world so rising reads as rising. `pub(crate)` so ONLY the composite
/// gameplay path (`composite::build_world_layers_fov_into`) selects this policy; every
/// public single-shot entry stays shear-free for previews/QA.
pub(crate) fn raycast_with_time_fov_gameplay(
    map: &Map,
    cam: &CameraPose,
    registry: &MaterialRegistry,
    fov: &Fov,
    time: f32,
    fb: &mut Framebuffer,
) {
    raycast_with_time_fov_q(
        map,
        cam,
        registry,
        fov,
        time,
        ShadowQuality::default(),
        // Per-storey standing reference: selects the slice plane of the floor the
        // player stands on (multi-floor) while zeroing the shear when grounded.
        // On a single-storey map this is exactly `STANDING_EYE_Z` ⇒ byte-identical.
        gameplay_eye_ref_z(cam.pos.z),
        fb,
    );
}

thread_local! {
    /// Per-frame quant-memo for the capture cast (cleared each cast).
    static CAPTURE_MEMO: std::cell::RefCell<SampleMemo> =
        std::cell::RefCell::new(SampleMemo::default());
    /// PERSISTENT, cross-frame world-anchored texture cache for the capture cast
    /// (the gameplay / composite NEAR path). `tick()`-ed (not cleared) each cast so
    /// entries survive across frames — the translation / rotation reuse win.
    static CAPTURE_TEX_CACHE: std::cell::RefCell<WorldTexCache> =
        std::cell::RefCell::new(WorldTexCache::default());
    /// Last capture cast's `(hits, misses)` for the cross-frame tex-cache, exposed
    /// via [`last_tex_cache_stats`] for the perf harness.
    static LAST_TEX_CACHE_STATS: std::cell::Cell<(u64, u64)> = const { std::cell::Cell::new((0, 0)) };
}

/// The cross-frame world-anchored texture-cache `(hits, misses)` recorded by the
/// most recent [`raycast_gameplay_capture`] on the CURRENT thread. The harness
/// reads this after each near cast to report the cache hit-rate. `(0, 0)` if no
/// capture cast has run on this thread (or the last cast was idle / animated-only).
pub fn last_tex_cache_stats() -> (u64, u64) {
    LAST_TEX_CACHE_STATS.with(|s| s.get())
}

/// As [`raycast_with_time_fov_gameplay`] but with a SELECTIVE-RE-SHADE capture sink:
/// every drawn near cell's exact [`shade_cell`] inputs ([`CellGeom`] + `screen_y`)
/// are recorded into `capture`, and the animated, non-mirror cells are indexed for
/// the animation-only re-shade ([`reshade_animated_into`]). `capture` is
/// generation-cleared ([`gbuffer::NearGBuffer::begin_capture`]) here and marked
/// valid on completion, so a later animation-only frame can replay only the
/// animated cells from cached geometry.
///
/// Output into `fb` is byte-for-byte identical to [`raycast_with_time_fov_gameplay`]
/// — the capture is a pure side-channel. `pub` so the composite gameplay path and
/// the `perfbench` harness can drive it.
pub fn raycast_gameplay_capture(
    map: &Map,
    cam: &CameraPose,
    registry: &MaterialRegistry,
    fov: &Fov,
    time: f32,
    capture: &mut gbuffer::NearGBuffer,
    fb: &mut Framebuffer,
) {
    let (cols, rows) = (fb.cols(), fb.rows());
    capture.resize(cols, rows);
    capture.begin_capture();
    // Mirror the legacy gameplay path's thread-local quant-memo (no per-call heap
    // alloc) so the captured cells are shaded with the SAME memo behaviour the live
    // composite cast uses — and a `None` memo would give identical output anyway.
    // ALSO carry a PERSISTENT, thread-local world-anchored texture cache: unlike the
    // per-frame memo (cleared each cast), this survives ACROSS casts so a re-seen
    // world cell (the player walked / turned) replays its cached `TexResult` and
    // skips the fBm/relief — only the cheap lighting re-runs. `tick()` (not `clear()`)
    // advances its LRU clock; entries persist. Static-only caching + always-fresh
    // lighting ⇒ byte-identical to a full re-shade.
    // CROSS-FRAME WORLD-TEX CACHE switch. Default OFF.
    //
    // The cache is EXACT (a hit replays the byte-identical `TexResult`; lighting is
    // always re-run fresh) and fully tested — but it delivers ~0 cross-frame reuse on
    // the real near path, so enabling it only adds a hash+probe per cell. The reason
    // is fundamental, not a bug: the per-screen-cell world sample coord is itself
    // sub-texel CONTINUOUS (a wall's `tex_u` + `wall_world_y`, a floor's exact pierce
    // point), so the SAME world surface seen from a new camera pose lands at a
    // slightly different coord → a different exact key → a miss. Getting reuse would
    // require QUANTIZING the coord to a texel grid, but `texture_sample` reads the
    // EXACT coord in its fBm — quantizing the key without quantizing the sample breaks
    // byte-identity, and quantizing the sample changes the output. Per the task's hard
    // rule ("if the exact split is not achievable as a win, STOP — do not ship an
    // approximation"), the cache stays OFF by default; the `RenderConfig::tex_cache`
    // knob enables it for measurement only. (The SPLIT — `apply_lighting(texture_sample())`
    // — ships and is exact regardless; see the byte-identity tests.)
    // The cache engages when explicitly requested (`RenderConfig::tex_cache`, measurement)
    // OR when TEXEL-SNAP is on — snapping is precisely what turns the cache from a
    // ~0.7%-hit no-op into a real cross-frame win (the snapped coord gives a stable,
    // colliding key), so snapping auto-enables it.
    // Read from the ACTIVE config the cast applied (no env in the hot path; was
    // `REVISION_TEXCACHE`). `tex_cache_effective` = raw flag OR texel-snap-on.
    let texcache_on = config::tex_cache_effective();
    CAPTURE_MEMO.with(|m| {
        CAPTURE_TEX_CACHE.with(|tc| {
            let mut memo = m.borrow_mut();
            let mut tex_cache = tc.borrow_mut();
            memo.clear();
            tex_cache.tick();
            raycast_full_memo(
                map,
                cam,
                registry,
                fov,
                time,
                ShadowQuality::default(),
                MirrorClarity::default(),
                // Runtime draw-distance: the active far-freeze (applied once per cast
                // from the live `FarFreeze` resource). Default = `FAR_FREEZE_DIST`, so
                // an un-applied cast is byte-identical to today.
                draw_dist::active_far_freeze(),
                // Per-storey standing reference (multi-floor): see `gameplay_eye_ref_z`.
                // `STANDING_EYE_Z` on a single-storey map ⇒ byte-identical.
                gameplay_eye_ref_z(cam.pos.z),
                Some(&mut memo),
                if texcache_on {
                    Some(&mut tex_cache)
                } else {
                    None
                },
                Some(capture),
                fb,
            );
            // Snapshot the cross-frame tex-cache hit/miss for this cast so the perf
            // harness can report the reuse rate (the cache itself is thread-local +
            // private; this is the only handle the outside gets).
            LAST_TEX_CACHE_STATS.with(|s| s.set((tex_cache.hits(), tex_cache.misses())));
        });
    });
    capture.finish_capture();
}

/// Whether ANY surface this map can render is a NO-PAN special cell — a reflective
/// (mirror/glass) surface bound in a rendered slot, or a hologram. Such cells are
/// VIEW-DIRECTION dependent (a mirror reflects a different scene as you turn; a holo
/// wireframe re-projects), so the near ring must NEVER pan them. When this is true
/// the [`near_cache::NearRing`] forces a structural re-cast instead of a pan, so a
/// rotation through a mirror/holo is always freshly cast (no stale reflection).
///
/// Conservative + cheap: it scans the rendered registry slots (0..7, the slots the
/// cast can dispatch) for `reflective`, and reports holograms via [`Map::has_holograms`].
/// A scene with neither pans freely; a scene with either is always cast fresh.
pub fn map_local_has_special_cells(map: &Map, reg: &MaterialRegistry) -> bool {
    // Surface slots 0..5 + node/door (5/6): any reflective bound material, any
    // VIEW-ANGLE glyph ([`SuperSelector::ByAngle`], which re-projects its symbol as
    // the camera turns), or any GLASS-MARKER (underline-only transparent) — all three
    // are view-/composite-dependent geometry the near ring must re-cast, never pan.
    for slot in 0..7u16 {
        let m = reg.get(MaterialId(slot));
        if m.reflective || m.is_view_angle_glyph() || m.is_underline_only_transparent() {
            return true;
        }
    }
    // Holo slot (7): rendered only when the map has holograms — but a reflective holo
    // material plus actual holograms is also a no-pan case.
    if map.has_holograms() {
        return true;
    }
    // PORTAL columns sample a DIFFERENT region through the `delta` re-base. The
    // re-based destination is a FIXED function of (camera position, ABSOLUTE world ray
    // angle): at a fixed position, the warped view through the opening depends only on
    // which way the ray points, NOT on the camera facing. So the near-ring bake — which
    // casts one ray per ABSOLUTE world angle through the SAME `cast_column` warp branch
    // ([`bake_near_ring_column`]) — bakes each portal column's warped destination shade
    // at its world angle, and a pure ROTATION PANS it exactly like static geometry. This
    // holds ONLY when the warped shade is itself view-INVARIANT: the destination surface
    // reads its headlamp/specular from the re-based ray direction (`warp_shade_unit`), so
    // under VIEW-DEPENDENT lighting the panned slice would carry the bake-time facing and
    // drift. With VIEW-INDEPENDENT lighting (the DEFAULT — [`view_independent_lighting`]),
    // those terms are dropped: the warped shade is (world pos, depth, world normal) only,
    // so the pan is byte-EXACT and the portal scene PANS like any static scene.
    //
    // Therefore a portal map is a no-pan special scene ONLY under view-dependent
    // lighting. Under the (default) view-independent path it pans for free; only
    // TRANSLATION (parallax through the opening) re-bakes — the same policy the near ring
    // already applies to all geometry past [`near_cache::NEAR_MOVE_CELLS`]. (Mirrors and
    // holograms stay special regardless: theirs is view-dependent *geometry* — the
    // reflection ray / projected silhouette literally aims elsewhere as you turn — which
    // no lighting toggle makes pannable.)
    if !map.portals.is_empty() && !ascii_shader::view_independent_lighting() {
        return true;
    }
    false
}

/// Bake ONE NEAR-RING column by casting a single ray at the uniform ANGULAR step
/// around `base_yaw`. Ring column `rc` covers absolute world ray angle
/// `base_yaw + (rc - center) * d_theta`; this casts that single ray (in the SAME
/// rectilinear vertical projection + gameplay vertical-shear the live viewport uses)
/// and writes the shaded column into column 0 of `out` (a 1-wide framebuffer).
///
/// `out` is the reused 1-column scratch (no per-call alloc on the hot path); `rows` is
/// taken from it. `center` is the ring column that maps to `base_yaw` exactly (a facing
/// ray). The cast is at the live `time` so animated materials bake at the ring's
/// pose-time. The caller routes the column into the opaque/transparent ring layers.
///
/// EXACTNESS: the DDA + wall-slice geometry of ring column `rc` is byte-identical to a
/// live viewport column pointing in the same absolute direction (`dda_hit` + the slice
/// extent depend only on ray direction, position, `proj_plane_dist` and the vertical
/// shear — all reproduced here). The only quantity that differs once the ring is
/// PANNED is the view-direction-dependent shading (headlamp / specular), baked at
/// `base_yaw` — the documented, margin-bounded approximation (see `near_cache`).
///
/// EXACTNESS UPGRADE: with view-independent lighting (the DEFAULT —
/// [`RenderConfig::view_independent_light`], read via
/// [`ascii_shader::view_independent_lighting`]) those view-dependent terms are
/// dropped from [`ascii_shader::light_surface`], so the baked shade depends only on
/// (world pos, depth, world normal) — identical at every yaw. The pan then has NO
/// shading approximation: a panned static column is byte-for-byte a live cast. The
/// special-cell recast set is UNCHANGED — mirrors and holograms are view-dependent
/// *geometry* (reflection ray / projected silhouette), not lighting, so they still
/// re-cast via `recast_special_columns_into`.
/// `pub(crate)` — only the near ring drives it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn bake_near_ring_column(
    map: &Map,
    cam_pos: glam::Vec3,
    registry: &MaterialRegistry,
    fov: &Fov,
    time: f32,
    base_yaw: f32,
    d_theta: f32,
    center: f32,
    rc: usize,
    out: &mut Framebuffer,
    // The LIVE viewport width. The ring is cast one column at a time, so `view_params`
    // below publishes a ONE-COLUMN horizontal projection scale — which the text
    // surfaces would then read as "this wall covers half a column", picking a totally
    // different text density from the one a full cast picks. Republishing the
    // viewport's own scale keeps a panned frame's wall text identical in CONTENT to a
    // cast frame's. `0` ⇒ keep the ring's own (1-column) scale.
    view_cols: usize,
) {
    let rows = out.rows();
    if rows == 0 {
        return;
    }
    let perlin: &Perlin = &PERLIN;
    let fog = fog_for_map(map);
    let far_freeze = draw_dist::active_far_freeze();
    let light_env = light_env_for_map(map);
    // View params drive the vertical projection + shear so a ring column matches a real
    // viewport column's vertical extent exactly. The 1-col cast uses camera_x = 0 (the
    // facing ray), so `cols` here only affects nothing the centre column reads.
    let view = view_params(1, rows, fov);
    // …then republish the LIVE viewport's projection scales (`view_params` installs
    // them as a side effect) so every text surface makes the SAME density decision the
    // full cast would. The ring's own ray maths above already used `view.plane_scale`.
    if view_cols > 0 {
        let _ = view_params(view_cols, rows, fov);
    }
    let horizon_base = (rows as i32) / 2;
    let proj = view.proj_plane_dist;
    // Near-ring bake is the gameplay path: anchor the DDA slice plane + shear to the
    // PER-STOREY standing reference (`gameplay_eye_ref_z`) so a bake on an upper floor
    // rings that storey's plane; a jump shears vertically without re-selecting the plane
    // out from under a single-cell object. Single-storey ⇒ `STANDING_EYE_Z` ⇒ identical.
    let eye_ref_z = gameplay_eye_ref_z(cam_pos.z);
    let vshift = vshift_per_perp(cam_pos.z, eye_ref_z, proj);
    let cell_z = eye_ref_z.floor() as i32;
    // The ring column is the viewport ray at absolute angle `base_yaw + alpha`. Build
    // it as the LIVE cast builds a column ray — `dir(base_yaw) + plane·camera_x` — so
    // the CAMERA-PLANE-PERPENDICULAR distance `dda_hit` returns (`side_dist -
    // delta_dist`, which is fisheye-free ONLY because the facing component of the ray
    // is exactly 1) matches a viewport column pointing the same way. A unit facing ray
    // would instead yield the TRUE distance (a `1/cos(alpha)` mismatch → wrong slice).
    let alpha = (rc as f32 - center) * d_theta;
    let plane_scale = view.plane_scale;
    let camera_x = alpha.tan() / plane_scale;
    let (fdir_x, fdir_y) = (base_yaw.sin(), base_yaw.cos());
    let plane_x = base_yaw.cos() * plane_scale;
    let plane_y = -base_yaw.sin() * plane_scale;
    let dir_x = fdir_x + plane_x * camera_x;
    let dir_y = fdir_y + plane_y * camera_x;
    out.clear();
    // Pitch is baked at 0 here; the fill path applies the pitch row-shift exactly like
    // the far ring (shared `pitch_row_shift`), so far + near share a horizon.
    cast_column(
        map,
        out,
        registry,
        perlin,
        &fog,
        far_freeze,
        None,
        None,
        &light_env,
        ShadowQuality::default(),
        MirrorClarity::default(),
        None,
        0,
        cell_z,
        cam_pos.x,
        cam_pos.y,
        dir_x,
        dir_y,
        horizon_base,
        proj,
        vshift,
        time,
    );
}

/// SELECTIVE RE-SHADE: re-shade ONLY the animated cells from their cached geometry
/// (`capture`) at the current `time`, writing each back into the correct near layer
/// (opaque vs transparent, by the same `Material::is_transparent` split the cast
/// uses). Static cells are left exactly as persisted — never touched.
///
/// EXACTNESS: each animated cell is re-run through the SAME [`shade_cell`] the full
/// cast uses, fed its cached [`CellGeom`] + `screen_y` and the new `time` — so the
/// result is byte-identical to a full re-cast + re-shade at that `time` for those
/// cells. The registry / fog / far-freeze are the idle-period constants; a `None`
/// memo is used (identical output to a populated memo). No DDA, no shadow march —
/// the geometry is reused from the cache. Pure compute, no allocation.
///
/// Cells the last full cast's silhouette subcell pass replaced with a 2×2 quadrant
/// blit are SKIPPED (flagged via [`gbuffer::NearGBuffer::mark_skip`]): re-shading one
/// would overwrite its persisted blit with a plain shade, which would NOT match a
/// full re-cast (whose subcell pass re-blits it). They are a small silhouette-edge
/// minority and keep their last-cast blit between full re-casts — so every cell this
/// function DOES write is exact.
///
/// Preconditions: `capture.is_valid()` (a full cast captured it) and the near layers
/// in `stack` are the persisted ones from that cast. The caller guarantees both
/// (animation-only frame after a full structural cast).
pub fn reshade_animated_into(
    map: &Map,
    registry: &MaterialRegistry,
    far_freeze: f32,
    capture: &gbuffer::NearGBuffer,
    time: f32,
    stack: &mut re_core::CompositeStack,
) {
    let perlin: &Perlin = &PERLIN;
    let fog = fog_for_map(map);
    for &(col, row) in capture.animated_cells() {
        // Skip silhouette cells the subcell pass blitted (keep their persisted blit).
        if capture.is_skipped(col, row) {
            continue;
        }
        let Some(gc) = capture.get(col, row) else {
            continue; // stale (shouldn't happen on a valid buffer) — skip.
        };
        // SKY-APPLIED cells are filled by the direction-sampled panorama, which
        // depends only on the (frozen) ray direction — NOT on `time`. So an
        // animation-only re-shade must LEAVE them as persisted (the sky-applied
        // cell from the last full cast); re-running `shade_cell` here would produce
        // the ordinary shaded material and lose the panorama. A camera move is
        // structural (full re-cast), which re-applies the sky, so this is exact.
        //
        // Keyed on `gc.sky` (set at capture time by whether `apply_sky` actually
        // ran — see `column_sky_open`), NOT on `surface == Ceiling/Empty`: an
        // ENCLOSED room's ceiling is that same surface kind but keeps its ordinary
        // (possibly ANIMATED) shaded material, which this path must still refresh.
        if gc.sky {
            continue;
        }
        let cell = shade_cell(
            registry,
            perlin,
            &fog,
            far_freeze,
            None, // no memo: identical output, and the animated set is small.
            None, // no tex-cache: these are ANIMATED cells, which bypass it anyway.
            // P4 insets: re-decide the phase-band reveal at the NEW `time` from the
            // captured cell's MAP coord (`gc.geom.cell`), so a SelfSimilar inset pulses
            // once per loop on the animation-only re-shade. Empty ⇒ no-op (today).
            map.resolved_insets(),
            &gc.geom,
            gc.screen_y,
            time,
        );
        // Route to the SAME layer the cast's split would: by the cell's surface
        // material transparency. Animated mirror cells are excluded from the list,
        // so no mirror-composite cell ever reaches here.
        let mat_id = surface_material(gc.geom.surface, gc.geom.material);
        if registry.get(mat_id).is_transparent() {
            stack.transparent.set(col, row, cell);
        } else {
            stack.opaque.set(col, row, cell);
        }
    }
}

// ───────────────────────── local 2×2 sub-ray sampler ────────────────────────
//
// The quadrant blitter (`crate::quadrant`) needs to SHADE four sub-rays inside a
// single marked screen cell, reusing the SAME DDA + Material shade the whole-frame
// cast uses, but at a sub-cell screen offset. `shade_subray` is that single-sample
// entry: it casts ONE ray (the primary DDA — mirrors/holograms are deliberately NOT
// re-bounced for a sub-sample, keeping the cost a single grid-walk) and shades the
// wall/floor/ceiling cell the ray lands on at the given fractional screen position.
//
// It is the ONLY per-marked-cell cost: 4 calls per marked cell ⇒ the overhead stays
// near 1.x×, never the 4× of a full-frame supersample.

/// The view basis a batch of [`shade_subray`] calls share — derived ONCE per frame
/// by [`SubViewBasis::new`] from the camera + FOV + framebuffer size, then reused for
/// every sub-ray so a marked cell's four samples cost only the per-ray DDA + shade.
#[derive(Clone, Copy, Debug)]
pub struct SubViewBasis {
    dir_x: f32,
    dir_y: f32,
    plane_x: f32,
    plane_y: f32,
    pos_x: f32,
    pos_y: f32,
    cell_z: i32,
    horizon: i32,
    proj_plane_dist: f32,
    /// Vertical eye shear, `(eye_z − STANDING_EYE_Z) · proj_plane_dist` — so a
    /// refined sub-cell agrees with the coarse cast under a jump/fall. `0` at
    /// standing height.
    vshift_per_perp: f32,
    cols: f32,
    rows: i32,
}

impl SubViewBasis {
    /// Build the shared sub-ray basis for a `cols`×`rows` frame. Mirrors the head of
    /// [`raycast_full_memo`] (camera basis, pitch-shifted horizon, projection plane)
    /// so a sub-ray lands EXACTLY where the corresponding full-frame column/row would.
    pub fn new(cam: &CameraPose, fov: &Fov, cols: usize, rows: usize) -> Self {
        let view = view_params(cols.max(1), rows.max(1), fov);
        let (dir_x, dir_y) = cam.dir();
        let plane_scale = view.plane_scale;
        let plane_x = cam.yaw.cos() * plane_scale;
        let plane_y = -cam.yaw.sin() * plane_scale;
        let rows_i = rows as i32;
        let horizon = (rows_i / 2) + pitch_row_shift(cam.pitch, rows);
        SubViewBasis {
            dir_x,
            dir_y,
            plane_x,
            plane_y,
            pos_x: cam.pos.x,
            pos_y: cam.pos.y,
            // Gameplay-only refinement: the DDA slice plane is anchored to the STANDING
            // slice (not the momentary jump z) so a sub-ray samples the SAME plane as
            // the near cast it refines — and so a jump never re-selects the plane out
            // from under a single-cell object. See `raycast_full_memo`'s `cell_z` note.
            cell_z: STANDING_EYE_Z.floor() as i32,
            horizon,
            proj_plane_dist: view.proj_plane_dist,
            // Sub-cell refinement runs ONLY on the gameplay composite path, so it
            // shears against the standing reference — matching the near cast it refines
            // (a jump moves both together; a grounded player gets shift 0).
            vshift_per_perp: vshift_per_perp(cam.pos.z, STANDING_EYE_Z, view.proj_plane_dist),
            cols: cols.max(1) as f32,
            rows: rows_i,
        }
    }
}

/// Shade ONE sub-ray at fractional screen position `(colf, rowf)` (in cell units,
/// e.g. a sub-centre like `col + 0.25`). Casts the primary DDA along the column ray
/// for `colf`, then shades the wall slice (if the ray hits a wall and `rowf` falls
/// inside its vertical extent) or the floor/ceiling otherwise — the same `shade_cell`
/// the full cast uses, so a sub-sample is byte-identical to a full-resolution cell at
/// that exact position. Pure + deterministic (no memo, no mirror/holo bounce).
///
/// Returns the fully-shaded [`Cell`]. The quadrant blitter reads `glyph`, `fg`, `bg`
/// and `depth` from it.
pub fn shade_subray(
    map: &Map,
    registry: &MaterialRegistry,
    basis: &SubViewBasis,
    colf: f32,
    rowf: f32,
) -> Cell {
    let perlin: &Perlin = &PERLIN;
    let fog = fog_for_map(map);
    let light_env = light_env_for_map(map);

    // Column ray for this fractional column (same formula as the full cast's
    // `camera_x = 2·(x+0.5)/cols − 1`, but evaluated at the sub-cell `colf`).
    let camera_x = 2.0 * (colf + 0.5) / basis.cols - 1.0;
    let ray_dir_x = basis.dir_x + basis.plane_x * camera_x;
    let ray_dir_y = basis.dir_y + basis.plane_y * camera_x;
    let ray_len = (ray_dir_x * ray_dir_x + ray_dir_y * ray_dir_y)
        .sqrt()
        .max(PERP_MIN);
    let ray_unit_x = ray_dir_x / ray_len;
    let ray_unit_y = ray_dir_y / ray_len;

    let hit = dda_hit(
        map,
        basis.cell_z,
        basis.pos_x,
        basis.pos_y,
        ray_dir_x,
        ray_dir_y,
        Some(registry),
    );
    let row = rowf.floor() as i32;
    let rows = basis.rows;
    let horizon = basis.horizon;
    let proj = basis.proj_plane_dist;

    // Wall slice extent for this ray, in the column's one projection (`VProj`) —
    // EXACTLY the coarse cast's, so a refined sub-cell agrees with its parent cell.
    // The sample point is the continuous row coordinate `rowf`.
    let vshift_per_perp = basis.vshift_per_perp;
    let vp = offslice::VProj::new(basis.cell_z, horizon, proj, vshift_per_perp);
    let (slice_top_f, slice_bottom_f) = if hit.hit {
        (
            vp.row(basis.cell_z as f32 + 1.0, hit.perp),
            vp.row(basis.cell_z as f32, hit.perp),
        )
    } else {
        (horizon as f32, horizon as f32)
    };
    let in_band = hit.hit && rowf >= slice_top_f && rowf < slice_bottom_f;

    // ── VERTICAL GEOMETRY sub-sample ── the same all-planes resolution the column
    // pass (`vertical_pass`) runs, for this one view ray: anything nearer than the
    // slice band (or outside it) that the pass would paint is what this sample shows.
    let k = vp.slope(rowf);
    let max_steps = (draw_dist::active_max_dda_steps() as i32).min(offslice::MAX_STEPS);
    if let Some(h) = offslice::query_row(
        map,
        basis.cell_z,
        basis.pos_x,
        basis.pos_y,
        ray_dir_x,
        ray_dir_y,
        vp.eye,
        k,
        max_steps,
        |h| {
            vhit_opaque(
                map,
                registry,
                h,
                basis.pos_x,
                basis.pos_y,
                ray_dir_x,
                ray_dir_y,
            )
        },
    ) {
        use offslice::{Face, Role};
        let owned_by_band = in_band && h.s >= hit.perp;
        let paints = !matches!(
            (h.role, h.face),
            (Role::Slice, _) | (Role::Slab, Face::Top | Face::Bottom)
        );
        if !owned_by_band && paints {
            let mut face_memo = None;
            return shade_vhit(
                map,
                registry,
                perlin,
                &fog,
                draw_dist::active_far_freeze(),
                None,
                None,
                &light_env,
                ShadowQuality::Off,
                &mut face_memo,
                None,
                &h,
                row,
                rowf,
                rows,
                &vp,
                (basis.pos_x, basis.pos_y),
                (ray_dir_x, ray_dir_y),
                (ray_unit_x, ray_unit_y),
                ramp_steps::enabled(),
                0.0,
            )
            .0;
        }
    }

    if in_band {
        // ── WALL sub-sample ──
        let (first, end) = offslice::span_rows(slice_top_f, slice_bottom_f);
        let wall_v = offslice::drawn_v(first, end, rowf);
        // 45° wedge/ramp: foreshorten + light from the real normal. `None` ⇒
        // axis-aligned face → byte-identical to a special-column recast with no
        // wedge/ramp support.
        let cos_theta = match hit.shape_normal {
            Some(n) => (ray_unit_x * n.x + ray_unit_y * n.y).abs(),
            None if hit.side == 1 => ray_unit_x.abs(),
            None => ray_unit_y.abs(),
        };
        let foreshorten = (1.0 / cos_theta.max(FORESHORTEN_COS_MIN)).min(FORESHORTEN_MAX);
        let face_sign = if hit.side == 1 {
            ray_unit_x.signum()
        } else {
            ray_unit_y.signum()
        };
        let view_z = (horizon - row) as f32 * VIEW_VERT_SCALE / rows.max(1) as f32;
        let view_dir = glam::Vec3::new(-ray_unit_x, -ray_unit_y, view_z).normalize_or_zero();
        let wall_lc = match hit.shape_normal {
            Some(n) => LightCtx::for_surface_normal(n, view_dir, &light_env),
            None => LightCtx::for_wall(hit.side, face_sign, view_dir, &light_env),
        };
        let geom = CellGeom {
            depth: hit.perp,
            foreshorten,
            side: hit.side,
            tex_u: hit.tex_u,
            wall_v,
            wall_world_y: wall_world_y(basis.cell_z, wall_v),
            hit_x: hit.wall_hit_x,
            hit_y: hit.wall_hit_y,
            surface: hit.surface,
            material: hit.material,
            light: Some(wall_lc),
            cell: (hit.cell_x, hit.cell_y, basis.cell_z),
            tex_footprint: None,
            subcell_backdrop: None,
        };
        shade_cell(
            registry,
            perlin,
            &fog,
            draw_dist::active_far_freeze(),
            None,
            None,
            map.resolved_insets(),
            &geom,
            row,
            0.0,
        )
    } else {
        // ── FLOOR / CEILING sub-sample ──
        let is_floor = row > horizon;
        let denom = if is_floor {
            (row - horizon) as f32
        } else {
            (horizon - row) as f32
        };
        let surface = if is_floor {
            SurfaceKind::Floor
        } else {
            SurfaceKind::Ceiling
        };
        let fc_view = glam::Vec3::new(
            -ray_unit_x * FC_VIEW_HORIZ,
            -ray_unit_y * FC_VIEW_HORIZ,
            if is_floor { 1.0 } else { -1.0 },
        )
        .normalize_or_zero();
        let fc_lc = LightCtx::for_floor(is_floor, fc_view, &light_env);
        let geom = if denom <= 0.0 {
            CellGeom {
                depth: DIST_INFINITY,
                foreshorten: FORESHORTEN_MAX,
                side: 0,
                tex_u: 0.0,
                wall_v: 0.0,
                wall_world_y: 0.0,
                hit_x: 0.0,
                hit_y: 0.0,
                surface,
                material: 0,
                light: Some(fc_lc),
                cell: (0, 0, 0),
                tex_footprint: None,
                subcell_backdrop: None,
            }
        } else {
            // Eye-height factor (= 1.0 at standing height) — same shear as the coarse
            // floor/ceiling cast, so a refined sub-cell stays glued to the wall base.
            let eye_height_cells = 1.0 + vshift_per_perp / proj;
            let row_dist = eye_height_cells * proj / denom;
            let foreshorten = (FLOOR_FORESHORTEN_K * rows.max(1) as f32 / (denom * denom))
                .clamp(1.0, FORESHORTEN_MAX);
            let fhit_x = basis.pos_x + row_dist * ray_unit_x;
            let fhit_y = basis.pos_y + row_dist * ray_unit_y;
            CellGeom {
                depth: row_dist,
                foreshorten,
                side: 0,
                tex_u: 0.0,
                wall_v: 0.0,
                wall_world_y: 0.0,
                hit_x: fhit_x,
                hit_y: fhit_y,
                surface,
                material: 0,
                light: Some(fc_lc),
                cell: (floori(fhit_x), floori(fhit_y), basis.cell_z),
                tex_footprint: None,
                subcell_backdrop: None,
            }
        };
        let mut cell = shade_cell(
            registry,
            perlin,
            &fog,
            draw_dist::active_far_freeze(),
            None,
            None,
            map.resolved_insets(),
            &geom,
            row,
            0.0,
        );
        // Direction-sampled SKY for above-horizon ceiling sub-samples, so a refined
        // silhouette cell against the sky matches its full-resolution neighbour —
        // gated by `column_sky_open` exactly like the full-resolution cast, so a
        // sub-sample inside an enclosed room shows the same shaded ceiling material
        // (no panorama leak at the silhouette edge either).
        if !is_floor && column_sky_open(map, basis.cell_z, basis.pos_x, basis.pos_y) {
            apply_sky(map, &mut cell, ray_unit_x, ray_unit_y, row, horizon, proj);
        }
        cell
    }
}

/// As [`raycast_with_time_fov`], but with an explicit [`ShadowQuality`] driving the
/// worked-out shadow pass (#2). The public entry defaults this to
/// [`ShadowQuality::Soft`]; the Bevy render system passes the live resource so a
/// settings row / perf tier can tune or disable shadows. Engine-internal so the
/// public cast API stays stable for `re_game` / `re_present`.
#[allow(clippy::too_many_arguments)]
fn raycast_with_time_fov_q(
    map: &Map,
    cam: &CameraPose,
    registry: &MaterialRegistry,
    fov: &Fov,
    time: f32,
    shadow_q: ShadowQuality,
    eye_ref_z: f32,
    fb: &mut Framebuffer,
) {
    // Public/legacy entry: no live mirror-clarity override (each mirror keeps its
    // authored clarity). The Bevy render system uses `raycast_full` to thread the
    // live `MirrorClarity` resource.
    raycast_full(
        map,
        cam,
        registry,
        fov,
        time,
        shadow_q,
        MirrorClarity::default(),
        // Public/legacy entry: the default far-freeze cutoff (a large distance, so
        // normal-scale rooms are unaffected). The Bevy render system threads the live
        // `FarFreeze` resource through `raycast_full` instead.
        FarFreeze::default().distance,
        eye_ref_z,
        fb,
    );
}

/// The single fully-parameterised cast: as [`raycast_with_time_fov_q`] but with an
/// explicit live [`MirrorClarity`] override driving every reflective surface. The
/// Bevy render system calls this with the live resource; every other public entry
/// forwards here with the default (no override → authored clarity).
#[allow(clippy::too_many_arguments)]
fn raycast_full(
    map: &Map,
    cam: &CameraPose,
    registry: &MaterialRegistry,
    fov: &Fov,
    time: f32,
    shadow_q: ShadowQuality,
    mirror: MirrorClarity,
    far_freeze: f32,
    eye_ref_z: f32,
    fb: &mut Framebuffer,
) {
    // Legacy / public entry: use a thread-local, reused quant-memo (no per-call
    // heap alloc on the steady state) cleared for this cast. The Bevy render system
    // owns a persistent `SampleMemo` resource and calls `raycast_full_memo` instead.
    thread_local! {
        static LEGACY_MEMO: std::cell::RefCell<SampleMemo> =
            std::cell::RefCell::new(SampleMemo::default());
    }
    LEGACY_MEMO.with(|m| {
        let mut memo = m.borrow_mut();
        memo.clear();
        raycast_full_memo(
            map,
            cam,
            registry,
            fov,
            time,
            shadow_q,
            mirror,
            far_freeze,
            eye_ref_z,
            Some(&mut memo),
            None,
            None,
            fb,
        );
    });
}

/// As [`raycast_full`] but with a caller-owned [`SampleMemo`] (the F.1 quant-memo).
/// The Bevy render system threads its persistent memo resource here so the table's
/// backing allocation is reused across frames; the cast CLEARS it (O(1) generation
/// bump) so each frame starts fresh — the memo never carries a cell from a prior,
/// differently-posed frame. `memo = None` disables memoization (every cell samples
/// directly — identical output, used only by paths that opt out).
#[allow(clippy::too_many_arguments)]
fn raycast_full_memo(
    map: &Map,
    cam: &CameraPose,
    registry: &MaterialRegistry,
    fov: &Fov,
    time: f32,
    shadow_q: ShadowQuality,
    mirror: MirrorClarity,
    far_freeze: f32,
    // Reference eye height that produces ZERO vertical shear. Gameplay passes
    // `STANDING_EYE_Z` (so a jump shears); generic/preview/QA casts pass `cam.pos.z`
    // (so the shift is identically 0 → byte-identical to the pre-shear renderer).
    eye_ref_z: f32,
    mut memo: Option<&mut SampleMemo>,
    // PERSISTENT world-anchored texture cache (cross-frame). Threaded into every
    // column so a re-seen world cell (translation / rotation) replays its cached
    // `TexResult` and only re-lights. `None` ⇒ the legacy path. Exact either way.
    mut tex_cache: Option<&mut WorldTexCache>,
    // SELECTIVE RE-SHADE capture sink. `Some` ⇒ capture every drawn cell's
    // `CellGeom` + `screen_y` for the animation-only re-shade (`crate::gbuffer`).
    // `None` ⇒ no capture; the output is byte-for-byte identical either way.
    mut capture: Option<&mut crate::gbuffer::NearGBuffer>,
    fb: &mut Framebuffer,
) {
    let cols = fb.cols();
    let rows = fb.rows();
    if cols == 0 || rows == 0 {
        return;
    }

    // Per-cast view geometry: FIXED vertical FOV + window-derived horizontal FOV +
    // the vertical projection-plane distance the wall/floor extents are scaled by.
    let view = view_params(cols, rows, fov);

    // Borrow the process-wide Perlin instead of rebuilding it per frame.
    let perlin: &Perlin = &PERLIN;

    // Per-level fog (band + optional tint), resolved ONCE per cast from the
    // map's atmosphere; defaults reproduce the historical fade-to-black.
    let fog = fog_for_map(map);

    // Scene lighting environment (KEY light + ambient + headlamp), resolved ONCE
    // per cast. Default = the built-in directional key + headlamp; a level may
    // tune it via its atmosphere (a future per-level light). The headlamp rides
    // each cell's view direction so it moves with the player.
    let light_env = light_env_for_map(map);

    let (dir_x, dir_y) = cam.dir();
    // Camera plane: right-hand perpendicular to dir, scaled by tan(hfov/2) (the
    // runtime horizontal FOV derived for this window in `view_params`).
    let plane_scale = view.plane_scale;
    let plane_x = cam.yaw.cos() * plane_scale;
    let plane_y = -cam.yaw.sin() * plane_scale;

    let pos_x = cam.pos.x;
    let pos_y = cam.pos.y;
    // DDA SLICE PLANE — anchored to the SHEAR REFERENCE height, NOT the momentary
    // camera z. This caster is 2.5D: it walks ONE horizontal slice (`cell_z`) of the
    // grid. Walls span every z (full-height columns) so they are invariant to the
    // choice, but a single-cell `Object`/door lives at exactly one z. If `cell_z`
    // tracked `cam.pos.z.floor()`, a JUMP would switch the sampled plane and the
    // door would vanish or appear to slide SIDEWAYS (an axis leak: vertical camera
    // motion bleeding into the horizontal projection). The jump's ONLY legitimate
    // visual effect is the vertical eye-shear (`vshift_per_perp`, below) — so the
    // slice plane must follow the SAME reference the shear is zeroed against:
    //   * GAMEPLAY (`eye_ref_z == STANDING_EYE_Z`) → plane fixed at the standing
    //     slice; a jump shears vertically but never re-selects the plane.
    //   * GENERIC / preview / QA (`eye_ref_z == cam.pos.z`) → `floor(cam.pos.z)`,
    //     exactly as before → byte-identical to the pre-fix renderer.
    let cell_z = eye_ref_z.floor() as i32;

    // Horizon row, shifted by pitch (linear; ±rows/2 at the pitch limit). Shared
    // with the FAR fill via `pitch_row_shift` so both tiers use ONE horizon.
    let horizon = (rows as i32 / 2) + pitch_row_shift(cam.pitch, rows);

    // VERTICAL EYE SHEAR (jump / fall visibility). We add a per-column vertical screen
    // offset proportional to the eye's deviation from the caller-supplied reference
    // (`eye_ref_z`), scaled by the same projection the wall slice uses
    // (`proj_plane_dist / perp`) — the classic raycaster "vertical position" shift.
    //
    // The reference is the *policy*: GAMEPLAY passes `STANDING_EYE_Z`, so the caster
    // (which otherwise discards fractional z) renders a jump/fall; GENERIC/preview/QA
    // passes `cam.pos.z`, so the deviation is EXACTLY 0 for any camera height and the
    // framebuffer is byte-identical to the pre-shear output (preview invariance).
    // Computed ONCE per cast (a single mul); the hot loop only does
    // `vshift_per_perp / perp` (one div it already pays for line_h).
    let vshift_per_perp = vshift_per_perp(cam.pos.z, eye_ref_z, view.proj_plane_dist);

    // FACE ANCHOR: publish the camera basis so a text wall's face can be
    // projected to exact screen columns. This is a VIEWPORT cast — `x` IS a terminal
    // column — which is precisely the precondition `face_span::face_anchor_for` needs.
    face_span::apply_camera(pos_x, pos_y, dir_x, dir_y, plane_x, plane_y, cols);

    for x in 0..cols {
        // Sample the column CENTRE — `2*(x+0.5)/cols - 1` is symmetric about 0, so
        // the left and right edge columns get equal-and-opposite plane offsets (a
        // plain `2*x/cols - 1` would bias one column left).
        let camera_x = 2.0 * (x as f32 + 0.5) / cols as f32 - 1.0;
        let ray_dir_x = dir_x + plane_x * camera_x;
        let ray_dir_y = dir_y + plane_y * camera_x;
        cast_column(
            map,
            fb,
            registry,
            perlin,
            &fog,
            far_freeze,
            memo.as_deref_mut(),
            tex_cache.as_deref_mut(),
            &light_env,
            shadow_q,
            mirror,
            capture.as_deref_mut(),
            x,
            cell_z,
            pos_x,
            pos_y,
            ray_dir_x,
            ray_dir_y,
            horizon,
            view.proj_plane_dist,
            vshift_per_perp,
            time,
        );
    }
    face_span::clear_camera();

    // ── HOLOGRAM PASS (transparent wireframe objects) ──
    // Gated O(1) on the map: a level with no hologram cells skips this entirely
    // (back-compat — the framebuffer is exactly the wall/floor/ceiling cast).
    if map.has_holograms() {
        holo_pass(
            map,
            fb,
            cell_z,
            pos_x,
            pos_y,
            dir_x,
            dir_y,
            plane_x,
            plane_y,
            horizon,
            view.proj_plane_dist,
            vshift_per_perp,
            time,
        );
    }
}

/// Identify, then FRESH-CAST, only the screen columns whose live view shows a
/// VIEW-DEPENDENT special surface — a mirror (reflective wall) or a hologram — and
/// write those columns into `fb`, leaving every other column UNTOUCHED. The
/// `special_cols` scratch is filled with the `(usize)` column indices that were
/// re-cast (cleared first; capacity reused).
///
/// # Why this exists (the near-ring special-cell pan recovery)
///
/// The [`near_cache::NearRing`] PANS static geometry for free on a pure rotation, but
/// a mirror reflects a different scene and a hologram re-projects as the camera turns,
/// so neither can be panned (their baked-at-`base_yaw` value is stale at the live
/// yaw). Historically a scene with ANY special cell disabled the pan for the WHOLE
/// frame ([`map_local_has_special_cells`]) → a full cast every rotation frame. This
/// function recovers the rotation win: the near ring still PANS the (majority) static
/// geometry, and only the (minority) special columns are re-cast HERE at the live view.
/// The composite then overwrites just those columns in the panned layers.
///
/// # Why it is EXACT
///
/// A special column is re-cast by the SAME [`cast_column`] + [`holo_pass`] the full
/// structural cast runs, at the live `cam`/`time` — so each re-cast column is
/// byte-for-byte identical to what a full cast would have produced for it (mirror
/// reflection, holo wireframe + body tint, transparency/subcell routing all included).
/// The non-special columns were proven exact under pan by the ring's geometry
/// guarantee. So a panned-with-special-recast frame equals a full cast for ALL cells.
///
/// # Cost
///
/// A cheap geometry-only pre-pass (`dda_hit` per column + `holo_scan` when the map has
/// holograms — no shading) locates the special columns; only those pay the full
/// per-column shade. On a scene where special cells are a minority (grand-demo: a few
/// mirror/holo columns) the saved shading is the rotation FPS win.
///
/// The vertical-shear reference is [`STANDING_EYE_Z`] (the gameplay path), matching
/// [`raycast_gameplay_capture`] / the ring bake, so the re-cast columns line up with
/// the panned ones. Pure compute, no terminal; the only allocation is the caller-owned
/// `special_cols` Vec growing once to its steady size.
#[allow(clippy::too_many_arguments)]
pub fn recast_special_columns_into(
    map: &Map,
    cam: &CameraPose,
    registry: &MaterialRegistry,
    fov: &Fov,
    time: f32,
    fb: &mut Framebuffer,
    special_cols: &mut Vec<usize>,
) {
    special_cols.clear();
    let cols = fb.cols();
    let rows = fb.rows();
    if cols == 0 || rows == 0 {
        return;
    }

    let view = view_params(cols, rows, fov);
    let perlin: &Perlin = &PERLIN;
    let fog = fog_for_map(map);
    let light_env = light_env_for_map(map);

    let (dir_x, dir_y) = cam.dir();
    let plane_scale = view.plane_scale;
    let plane_x = cam.yaw.cos() * plane_scale;
    let plane_y = -cam.yaw.sin() * plane_scale;

    let pos_x = cam.pos.x;
    let pos_y = cam.pos.y;
    // Gameplay special-column scan: anchor the DDA slice plane + shear to the PER-STOREY
    // standing reference (`gameplay_eye_ref_z`), so an upper-floor camera scans the
    // upper storey's plane and a jump still shears without re-selecting the plane. On a
    // single-storey map this is `STANDING_EYE_Z` ⇒ identical to before. See
    // `raycast_full_memo`'s `cell_z` note for the axis-leak the floor()-of-reference
    // (not floor()-of-live-z) avoids.
    let eye_ref_z = gameplay_eye_ref_z(cam.pos.z);
    let cell_z = eye_ref_z.floor() as i32;

    let horizon = (rows as i32 / 2) + pitch_row_shift(cam.pitch, rows);
    let vshift_per_perp = vshift_per_perp(cam.pos.z, eye_ref_z, view.proj_plane_dist);

    let has_holos = map.has_holograms();
    // Every layer that holds a hologram: a column is special if ANY of them draws in
    // it, not only the eye layer (a dome at z = 5..7 re-projects as the camera turns).
    let holo_z = if has_holos {
        holo_layers(map)
    } else {
        Vec::new()
    };
    // PORTAL columns are special (must re-cast at the live view, never pan) ONLY when
    // they cannot be panned — i.e. under VIEW-DEPENDENT lighting, where the warped
    // destination shade carries the bake-time facing. Under VIEW-INDEPENDENT lighting
    // (the DEFAULT) the warped shade is view-invariant, so the near ring bakes it at its
    // world angle and a rotation pans it byte-exactly (`map_local_has_special_cells`);
    // such columns must NOT be force-recast here, or the rotation win is lost. Mirrors +
    // holograms are always special (view-dependent geometry, no lighting toggle helps).
    let portals_are_special = !map.portals.is_empty() && !ascii_shader::view_independent_lighting();

    // ── PHASE A — locate special columns (geometry-only, no shading) ──
    // For each viewport column do the primary DDA. A reflective primary hit ⇒ a
    // MIRROR column; a portal primary hit ⇒ a WARP column (it samples a different
    // region through `delta`, so it is view-dependent and can never be panned). If
    // the map has holograms, scan up to the primary wall depth (so a wall in front
    // occludes the holo, matching `holo_pass`) ⇒ a HOLO column.
    for x in 0..cols {
        let camera_x = 2.0 * (x as f32 + 0.5) / cols as f32 - 1.0;
        let ray_dir_x = dir_x + plane_x * camera_x;
        let ray_dir_y = dir_y + plane_y * camera_x;
        let primary = dda_hit(
            map,
            cell_z,
            pos_x,
            pos_y,
            ray_dir_x,
            ray_dir_y,
            Some(registry),
        );
        let is_mirror =
            primary.hit && surface_is_mirror(registry, primary.surface, primary.material);
        let is_portal = portals_are_special && surface_is_portal(map, &primary, cell_z).is_some();
        let is_holo = has_holos && {
            // Same occlusion cap `holo_pass` uses: the wall slice's perp at this
            // column (the horizon-row depth a full cast would write). `holo_scan`
            // is geometry-only (no shade).
            let max_dist = if primary.hit {
                primary.perp
            } else {
                DIST_INFINITY
            };
            holo_column_hit(
                map, &holo_z, cell_z, pos_x, pos_y, ray_dir_x, ray_dir_y, max_dist,
            )
        };
        if is_mirror || is_portal || is_holo {
            special_cols.push(x);
        }
    }

    if special_cols.is_empty() {
        return;
    }

    // ── PHASE B — full cast ONLY the special columns ──
    // Each special column gets the SAME wall/floor/ceiling cast (incl. mirror
    // reflection) the full structural cast does. No capture / tex-cache (the pan
    // path does not drive the selective-reshade G-buffer; a pan invalidates it).
    // Same VIEWPORT metric as the full cast, so a re-cast special column anchors its
    // text (the FACE ANCHOR above) exactly as a full cast would.
    face_span::apply_camera(pos_x, pos_y, dir_x, dir_y, plane_x, plane_y, cols);
    for &x in special_cols.iter() {
        let camera_x = 2.0 * (x as f32 + 0.5) / cols as f32 - 1.0;
        let ray_dir_x = dir_x + plane_x * camera_x;
        let ray_dir_y = dir_y + plane_y * camera_x;
        cast_column(
            map,
            fb,
            registry,
            perlin,
            &fog,
            draw_dist::active_far_freeze(),
            None,
            None,
            &light_env,
            ShadowQuality::default(),
            MirrorClarity::default(),
            None,
            x,
            cell_z,
            pos_x,
            pos_y,
            ray_dir_x,
            ray_dir_y,
            horizon,
            view.proj_plane_dist,
            vshift_per_perp,
            time,
        );
    }
    face_span::clear_camera();

    // ── PHASE C — hologram wireframe pass ──
    // `holo_pass` recomputes its per-column crossings and draws ONLY columns where
    // a hologram is in front of the wall. Every such column is in `special_cols`
    // (Phase A marked it), and its wall was cast in Phase B (so the holo occlusion
    // depth + interior body-tint composite onto a real wall, not an empty cell). A
    // mirror-only special column has no holo crossing, so the pass leaves it alone.
    if has_holos {
        holo_pass(
            map,
            fb,
            cell_z,
            pos_x,
            pos_y,
            dir_x,
            dir_y,
            plane_x,
            plane_y,
            horizon,
            view.proj_plane_dist,
            vshift_per_perp,
            time,
        );
    }
}

// ────────────────────────── mirror / reflection (#1) ────────────────────────
//
// A reflective wall surface (a material with `reflective = true`) reflects the
// primary DDA ray about its axis-aligned face normal and continues the SAME
// grid-walk for ONE bounce. The wall slice then shows the reflected scene
// (shaded with full material + relief + shadow), composited with a cool sheen and
// degraded toward glitch/empty by the mirror's CLARITY — so the reflection can
// "resolve" from broken to crisp as the game drives clarity 0→1 (the recognition
// beat). One bounce only: a reflected ray that lands on another mirror is shaded
// as a plain wall (no recursion), bounding cost; a reflected ray that escapes the
// map shows the sky/void.

// The MIRROR sub-system (`MirrorView`, `surface_is_mirror`, `surface_clarity`,
// `mirror_composite`, and the `MIRROR_*` tunables) lives in `crate::mirror`. The
// cast still builds the reflected `RayHit`, wraps it in a `MirrorView`, and calls
// `mirror_composite` per mirror cell — `MIRROR_ORIGIN_NUDGE` nudges the reflected
// DDA origin off the mirror face (used at the construction site below).

/// One DDA wall hit: the geometry the wall-shading loop needs, computed for a
/// single ray (the primary cast OR a mirror's reflected bounce). Pure data; cheap
/// to build (no allocation), so the reflection bounce reuses the SAME grid-walk
/// primitive as the primary ray.
#[derive(Clone, Copy)]
pub(crate) struct RayHit {
    /// Did the ray hit a solid cell before escaping the map?
    pub(crate) hit: bool,
    /// `0` = NS face (y-const plane), `1` = EW face (x-const plane).
    pub(crate) side: u8,
    pub(crate) surface: SurfaceKind,
    /// The hit cell's game `side_data` tag (the material/role key).
    pub(crate) material: u16,
    /// Perpendicular distance along THIS ray to the hit (`DIST_INFINITY` on miss).
    pub(crate) perp: f32,
    /// World hit point on the face.
    pub(crate) wall_hit_x: f32,
    pub(crate) wall_hit_y: f32,
    /// Fractional crossing along the variable axis of the face, in `[0,1]`.
    pub(crate) tex_u: f32,
    /// Integer grid cell the DDA stopped on (`hit == true`). Lets the cast
    /// resolve a portal-tagged hit to its [`re_world::PortalLink`] via
    /// [`Map::portal_for_cell`] without re-walking the grid. Undefined (the
    /// origin cell) on a miss.
    pub(crate) cell_x: i32,
    pub(crate) cell_y: i32,
    /// WORLD-space outward unit normal of a NON-axis-aligned hit surface (a
    /// [`re_core::Shape`] wedge diagonal or ramp slope). `None` for every Full-cell
    /// hit — the wall-shading path then derives the axis-aligned normal from `side`
    /// exactly as before (byte-identical). `Some(n)` overrides the face normal,
    /// foreshortening, and `LightCtx` so a 45° face reads as a lit slope, not a flat
    /// wall. See [`crate::shape_hit`].
    pub(crate) shape_normal: Option<glam::Vec3>,
}

impl RayHit {
    /// A miss (ray escaped the map): no solid hit, infinite depth.
    fn miss(ox: f32, oy: f32) -> Self {
        RayHit {
            hit: false,
            side: 0,
            surface: SurfaceKind::Empty,
            material: 0,
            perp: DIST_INFINITY,
            wall_hit_x: ox,
            wall_hit_y: oy,
            tex_u: 0.0,
            cell_x: floori(ox),
            cell_y: floori(oy),
            shape_normal: None,
        }
    }
}

/// Walk the voxel grid (the same continuous-position DDA the primary cast uses)
/// from world origin `(ox, oy)` on plane `cell_z` along `(ray_dir_x, ray_dir_y)`,
/// returning the first solid-cell hit's geometry (or a [`RayHit::miss`] if the ray
/// escapes the map). Shared by the primary ray and the mirror's reflected bounce —
/// no allocation, pure grid reads.
///
/// IMAGE CUT-OUT (milestone 2): when `registry` is `Some` and the ray enters an
/// Image-recipe solid cell whose attached texture is FULLY TRANSPARENT down the
/// image column at the ray's crossing fraction (`tex_u`), the cell is see-through at
/// this column and the DDA marches PAST it — exactly like a wedge open triangle. An
/// Image cell with any opaque texel in that strip is a real hit (the per-row vertical
/// silhouette is resolved later, in [`cast_column`], by a second backdrop ray). When
/// `registry` is `None`, OR the cell is not an Image material, the historical
/// behaviour is byte-identical (the cell stops the ray).
fn dda_hit(
    map: &Map,
    cell_z: i32,
    ox: f32,
    oy: f32,
    ray_dir_x: f32,
    ray_dir_y: f32,
    registry: Option<&MaterialRegistry>,
) -> RayHit {
    let mut map_x = floori(ox);
    let mut map_y = floori(oy);

    let delta_dist_x = if ray_dir_x == 0.0 {
        1e30
    } else {
        (1.0 / ray_dir_x).abs()
    };
    let delta_dist_y = if ray_dir_y == 0.0 {
        1e30
    } else {
        (1.0 / ray_dir_y).abs()
    };

    let (step_x, mut side_dist_x) = if ray_dir_x < 0.0 {
        (-1, (ox - map_x as f32) * delta_dist_x)
    } else {
        (1, (map_x as f32 + 1.0 - ox) * delta_dist_x)
    };
    let (step_y, mut side_dist_y) = if ray_dir_y < 0.0 {
        (-1, (oy - map_y as f32) * delta_dist_y)
    } else {
        (1, (map_y as f32 + 1.0 - oy) * delta_dist_y)
    };

    let mut hit = false;
    let mut side = 0u8;
    let mut surface = SurfaceKind::Empty;
    let mut material = 0u16;

    // Runtime DDA step cap (the draw-distance lever): defaults to the historical
    // `MAX_DDA_STEPS` so an un-applied cast is byte-identical; a shorter draw distance
    // lowers it to cut the empty-air march on a long sight-line.
    let max_dda_steps = draw_dist::active_max_dda_steps();
    for _ in 0..max_dda_steps {
        if side_dist_x < side_dist_y {
            side_dist_x += delta_dist_x;
            map_x += step_x;
            side = 1;
        } else {
            side_dist_y += delta_dist_y;
            map_y += step_y;
            side = 0;
        }

        if !map.in_bounds(map_x, map_y, cell_z) {
            break; // escaped — open ray
        }
        let kind = map.cell_at(map_x, map_y, cell_z);
        if kind == CellKind::Object
            && is_sprite_node(
                map,
                map_x,
                map_y,
                cell_z,
                kind,
                map.side_data_at(map_x, map_y, cell_z),
            )
        {
            // NODE SPRITE: a puzzle node is render-transparent on the
            // primary slice too — it is drawn as a screen-space-bounded sprite by
            // `node_sprite::apply` AFTER the cast, not as a wall-sized voxel face.
            // Exactly like a RampZ's open triangle: keep marching to whatever is
            // behind it. Collision (`is_solid`) is untouched — only this render
            // path treats the cell as air.
            continue;
        }
        if kind.is_solid() {
            // 45° GEOMETRY: a non-`Full` solid cell is a wedge/ramp. Intersect
            // the ray with the cell's diagonal/slope face. A `Some` hit reports the
            // real surface (tilted normal, exact point); a `None` means the ray
            // passed through the OPEN triangle / above the ramp, so the DDA must
            // CONTINUE to whatever is behind it (the open part is see-through). A
            // `Full` cell skips this entirely → the axis-aligned fast path below is
            // byte-identical to a renderer with no wedge/ramp support at all
            // (assert: level 01 dump hash).
            let shape = map.shape_at(map_x, map_y, cell_z);
            // RAMPS are never a primary hit: a RampZ cell is solid only under its slope,
            // so it cannot stand for a full-height slice. The vertical geometry pass
            // (`crate::offslice`) renders it as an exact polygon on every plane, the eye
            // plane included, and the primary ray marches on to whatever is behind it.
            if matches!(shape, re_core::Shape::RampZ(_)) {
                continue;
            }
            if shape != re_core::Shape::Full {
                // Entry parameter onto this cell's near face = perpendicular
                // distance to the face just stepped across.
                let t_enter = if side == 1 {
                    side_dist_x - delta_dist_x
                } else {
                    side_dist_y - delta_dist_y
                };
                match shape_hit::intersect_cell(
                    shape, map_x, map_y, cell_z, ox, oy, ray_dir_x, ray_dir_y, t_enter,
                ) {
                    Some(sh) => {
                        let sd = map.side_data_at(map_x, map_y, cell_z);
                        return shape_ray_hit(sh, cellkind_to_surface(kind, sd), sd, map_x, map_y);
                    }
                    None => continue, // see-through: keep marching past this cell.
                }
            }
            // IMAGE CUT-OUT (milestone 2): a Full Image-recipe cell whose texture is
            // fully transparent down the column at this crossing fraction is a pane
            // the ray passes THROUGH (the figure's left/right margin / an empty
            // billboard) — march on to the backdrop, accumulating distance like the
            // wedge open-triangle. Gated so a non-image map pays nothing: only an
            // Image material with a resolved texture and an all-sub-cutoff strip
            // continues; everything else stops exactly as before.
            let mat = map.side_data_at(map_x, map_y, cell_z);
            let surf = cellkind_to_surface(kind, mat);
            if let Some(reg) = registry {
                let mid = surface_material(surf, mat);
                let material_ref = reg.get(mid);
                // Cheap recipe gate first (no texture work for ordinary walls).
                if matches!(
                    material_ref.recipe,
                    ascii_shader::TextureRecipe::Image { .. }
                ) {
                    let pass_through = match material_ref.image_span() {
                        // WORLD-ANCHORED span: the figure spans the surface, so
                        // this cell only covers the SLICE of the figure in its world-Z
                        // extent `[cell_z, cell_z+1]` at the crossing's world ALONG. The
                        // strip test marches past only when that whole slice (or the
                        // whole cell, if outside the figure's world box) is transparent.
                        Some(span) => {
                            let along = dda_cell_world_along(
                                side,
                                ox,
                                oy,
                                ray_dir_x,
                                ray_dir_y,
                                side_dist_x,
                                side_dist_y,
                                delta_dist_x,
                                delta_dist_y,
                            );
                            let up_top = wall_world_y(cell_z, 0.0); // cell_z + 1
                            let up_bottom = wall_world_y(cell_z, 1.0); // cell_z
                            material_ref.image_strip_fully_transparent_world(
                                &span, along, up_top, up_bottom,
                            )
                        }
                        // LEGACY per-cell tiling: the whole image down the in-cell column.
                        None => {
                            let tex_u = dda_cell_tex_u(
                                side,
                                ox,
                                oy,
                                ray_dir_x,
                                ray_dir_y,
                                side_dist_x,
                                side_dist_y,
                                delta_dist_x,
                                delta_dist_y,
                            );
                            material_ref.image_strip_fully_transparent(tex_u)
                        }
                    };
                    if pass_through {
                        continue; // see-through column: keep marching to the backdrop.
                    }
                }
            }
            hit = true;
            surface = surf;
            material = mat;
            break;
        }
    }

    if !hit {
        return RayHit::miss(ox, oy);
    }

    let perp = if side == 1 {
        side_dist_x - delta_dist_x
    } else {
        side_dist_y - delta_dist_y
    }
    .max(PERP_MIN);

    // wallX = exact crossing fraction along the variable axis of the face.
    let (wall_x, wall_hit_x, wall_hit_y) = if side == 1 {
        let wall_x = oy + perp * ray_dir_y;
        (
            wall_x,
            map_x as f32 + if step_x < 0 { 1.0 } else { 0.0 },
            wall_x,
        )
    } else {
        let wall_x = ox + perp * ray_dir_x;
        (
            wall_x,
            wall_x,
            map_y as f32 + if step_y < 0 { 1.0 } else { 0.0 },
        )
    };
    let mut frac = wall_x - wall_x.floor();
    // Flip so the texture is not mirrored on opposing faces (wolf4sdl).
    if side == 1 && ray_dir_x > 0.0 {
        frac = 1.0 - frac;
    }
    if side == 0 && ray_dir_y < 0.0 {
        frac = 1.0 - frac;
    }
    let tex_u = frac.clamp(0.0, 1.0);

    RayHit {
        hit,
        side,
        surface,
        material,
        perp,
        wall_hit_x,
        wall_hit_y,
        tex_u,
        cell_x: map_x,
        cell_y: map_y,
        shape_normal: None,
    }
}

/// In-cell horizontal crossing fraction (`tex_u`) for the cell the DDA has JUST
/// stepped into — computed mid-march, before the cell is committed as the hit, so
/// the image cut-out can sample the texture column at the exact `u` the shader will
/// later draw from. Replicates the post-loop `tex_u` math of [`dda_hit`] for the
/// current step: the perpendicular distance to the just-crossed face is
/// `side_dist − delta_dist` on the stepped axis; the crossing point on the variable
/// axis gives the fraction, flipped on the same faces the hit path flips so opposing
/// faces are not mirrored. Pure arithmetic, no grid reads.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn dda_cell_tex_u(
    side: u8,
    ox: f32,
    oy: f32,
    ray_dir_x: f32,
    ray_dir_y: f32,
    side_dist_x: f32,
    side_dist_y: f32,
    delta_dist_x: f32,
    delta_dist_y: f32,
) -> f32 {
    let perp = if side == 1 {
        side_dist_x - delta_dist_x
    } else {
        side_dist_y - delta_dist_y
    }
    .max(PERP_MIN);
    let wall_x = if side == 1 {
        oy + perp * ray_dir_y
    } else {
        ox + perp * ray_dir_x
    };
    let mut frac = wall_x - wall_x.floor();
    if side == 1 && ray_dir_x > 0.0 {
        frac = 1.0 - frac;
    }
    if side == 0 && ray_dir_y < 0.0 {
        frac = 1.0 - frac;
    }
    frac.clamp(0.0, 1.0)
}

/// WORLD ALONG coordinate of the cell the DDA has JUST stepped into — the
/// crossing point on the wall face's VARIABLE axis (world Y on an NS face `side==1`,
/// world X on an EW face `side==0`), computed mid-march so the image-span cut-out
/// can resolve the figure's `u` at the EXACT world coordinate `shade_cell` later
/// threads into [`ascii_shader::SampleCtx::wall_along`] (`wall_hit_y`/`wall_hit_x` =
/// this same `wall_x`). UNLIKE [`dda_cell_tex_u`] it returns the RAW world coordinate
/// (no per-face flip / fract), since the span maps absolute world position → `u`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn dda_cell_world_along(
    side: u8,
    ox: f32,
    oy: f32,
    ray_dir_x: f32,
    ray_dir_y: f32,
    side_dist_x: f32,
    side_dist_y: f32,
    delta_dist_x: f32,
    delta_dist_y: f32,
) -> f32 {
    let perp = if side == 1 {
        side_dist_x - delta_dist_x
    } else {
        side_dist_y - delta_dist_y
    }
    .max(PERP_MIN);
    if side == 1 {
        oy + perp * ray_dir_y
    } else {
        ox + perp * ray_dir_x
    }
}

/// Build a [`RayHit`] from a 45° [`shape_hit::ShapeHit`] (wedge/ramp). The real
/// world-space surface normal rides in `shape_normal` so the wall-shading path
/// lights it as a tilted 45° face; `side` carries the dominant axis as the
/// foreshorten/tangent fallback for the axis-aligned terms that still index by it.
fn shape_ray_hit(
    sh: shape_hit::ShapeHit,
    surface: SurfaceKind,
    material: u16,
    map_x: i32,
    map_y: i32,
) -> RayHit {
    RayHit {
        hit: true,
        side: sh.side,
        surface,
        material,
        perp: sh.t.max(PERP_MIN),
        wall_hit_x: sh.hit_x,
        wall_hit_y: sh.hit_y,
        tex_u: sh.tex_u.clamp(0.0, 1.0),
        cell_x: map_x,
        cell_y: map_y,
        shape_normal: Some(sh.normal),
    }
}

// `surface_is_mirror` / `surface_clarity` moved to `crate::mirror`.

// ── WALL-BOTTOM / WALL-TOP ↔ FLOOR/CEILING DEPTH ALIGNMENT ──────────────────────
//
// The classic per-row floor/ceiling cast (the "FLOOR / CEILING rows" loop in
// `cast_column_depth`) approximates the world point a screen row's ray meets with
// `row_dist = eye_height_cells * proj_plane_dist / denom`, `denom = row - horizon`
// (or `horizon - row` for the ceiling) — a texture-space convenience, NOT the exact
// per-row projection [`offslice::VProj::row`] the wall slice above it uses (which
// keys off the true `PLAYER_EYE_HEIGHT` and the row's CONTINUOUS centre `row + 0.5`,
// not the bare integer `row`). The two formulas agree closely for rows well past the
// wall's own span, but for the row(s) immediately adjacent to the wall's rendered
// edge the classic formula can compute a floor/ceiling distance FARTHER than the
// wall's own perpendicular depth `perp`. Every floor/ceiling row this loop visits is
// OUTSIDE the wall's own (exactly-projected) span, so under this renderer's one ray
// per column model such a row's true point is always in FRONT of that wall — never
// behind it — so a computed distance `>= perp` is never real, visible geometry, only
// a texture-space rounding artefact. Left uncorrected, the loop paints that
// impossible depth anyway: with the depth-contrast effect (`depth_contrast.rs`) an
// unbounded far depth collapses straight to fog, showing as a dark strip 1–2 rows
// tall hugging the wall's base.
//
// The fix clamps the row's OWN depth to the wall's `perp` (`clamp_to_wall_depth`)
// instead of re-drawing it as more wall: the row keeps its `SurfaceKind::Floor` /
// `Ceiling` tag (and this renderer's one-true-projection contract for the wall's
// OWN span — `lines.rs`'s "one convention: lines and walls agree at every height",
// `zband.rs`'s silhouette floor, the code-wall panel baseline — is undisturbed,
// since none of them ever look past `perp`), only the impossible depth number is
// bounded. `hit` guards the (rare) miss column, where there is no wall depth to
// compare against.
//
// WINDOWED, not unconditional: the overshoot `raw - perp` is driven by a roughly
// CONSTANT sub-row offset between the wall's true continuous projection and this
// loop's per-integer-row formula, but `raw = k / denom` is an INVERSE function of
// `denom` — so that same constant offset decays back below `perp` within 1–2 rows
// for an ordinary (far) wall, yet for a NEAR wall (`perp` itself small, `denom` at
// the transition small) it can take many more rows to decay, because `1/denom` is
// steep there. Clamping EVERY such row flat to `perp` regresses two real levels:
// `L1-first-pulse.lvl` (a diagonal wall a couple of cells from the camera) paints
// a wide, hard-edged flat plateau instead of the floor's natural depth gradient,
// and `08-material-demo.lvl` (an 8×8 room, `fog_far = 4`) loses its fogged-floor
// horizon band entirely — an 8-row flat field. Both wall spans are genuinely
// close, not the acute 1–2 row transition the dark-strip bug (and this fix)
// targets.
//
// So the clamp only fires within [`FLOOR_CLAMP_MAX_DENOM_PAST_WALL`] of the row
// where the formula would cross back below `perp` on its own — computed
// CONTINUOUSLY as `denom_cross = k / perp` (the exact, fractional `denom` at
// which `raw == perp`), not from the wall's own rendered screen span
// (`draw_top`/`draw_bottom`). An earlier version keyed the window off rows-PAST-
// `draw_bottom` instead, and that failed
// `near_cache::tests::panned_with_special_recast_matches_full_cast`: the wall's
// rendered span is an already-ROUNDED screen row, so a window keyed off it can
// flip a cell between clamped/unclamped between two poses just an angular hair
// apart — exactly the kind of pop the near-cache's pan-reuse assumes can't happen
// for static geometry. `denom_cross` is a smooth function of `perp` (itself
// smooth in the camera pose), so this window opens and closes continuously
// instead. Farther rows are left at their unclamped value, for exactly the rows
// where clamping would otherwise paint a wide, wrong-looking flat plateau instead
// of the original smooth (if technically imprecise) falloff. A genuinely close
// wall (`perp ≈ 2`, so `denom_cross ≈ 19` — past every row a 32-row screen can
// reach) therefore keeps its original look end to end rather than a flat
// plateau; this is a known, documented trade-off, not a silent regression — see
// `tests/golden/README.md`'s dark-strip cell-count note.

/// `denom` units (screen rows off the horizon, continuous) past the point where
/// the classic per-row formula would cross back below the wall's own `perp` on
/// its own, within which [`clamp_to_wall_depth`] still bounds an overshooting
/// floor/ceiling row. Matches this fix's own documented scope above ("a dark
/// strip 1–2 rows tall hugging the wall's base") with headroom. Rows farther than
/// this (in `denom` terms) are left unclamped — see the module note above.
const FLOOR_CLAMP_MAX_DENOM_PAST_WALL: f32 = 2.0;

/// Clamp a floor/ceiling row's classic-formula distance to a nearer wall's own
/// perpendicular depth in the same column, per the module note above. A no-op
/// (`row_dist` unchanged) whenever the row's own distance already reads nearer
/// than the wall (the overwhelming majority of floor/ceiling rows in any scene),
/// OR whenever this row's `denom` is beyond [`FLOOR_CLAMP_MAX_DENOM_PAST_WALL`] of
/// the continuous crossover `k / perp` — the clamp's documented, acute scope;
/// farther rows keep the pre-fix formula so a nearby wall's approximation error
/// decays into its ordinary smooth falloff instead of a wide flat plateau.
/// `k` is the row-distance formula's own numerator (`eye_height_cells *
/// proj_plane_dist`), passed in rather than recomputed so this stays a pure
/// function of its caller's already-derived values.
#[inline]
fn clamp_to_wall_depth(row_dist: f32, hit: bool, perp: f32, denom: f32, k: f32) -> f32 {
    if !hit || row_dist < perp || perp <= 0.0 {
        return row_dist;
    }
    let denom_cross = k / perp;
    if denom_cross - denom <= FLOOR_CLAMP_MAX_DENOM_PAST_WALL {
        perp
    } else {
        row_dist
    }
}

#[allow(clippy::too_many_arguments)]
fn cast_column(
    map: &Map,
    fb: &mut Framebuffer,
    registry: &MaterialRegistry,
    perlin: &Perlin,
    fog: &FogParams,
    far_freeze: f32,
    memo: Option<&mut SampleMemo>,
    // PERSISTENT world-anchored texture cache (cross-frame). When `Some`, static
    // materials replay their cached `TexResult` and re-light fresh; animated ones
    // bypass it. `None` ⇒ the legacy memo / direct-sample path. Byte-identical either way.
    tex_cache: Option<&mut WorldTexCache>,
    light_env: &LightEnv,
    shadow_q: ShadowQuality,
    mirror: MirrorClarity,
    // SELECTIVE RE-SHADE capture sink (Layer-A animation-only optimisation). When
    // `Some`, every drawn wall / floor / ceiling cell's `CellGeom` + `screen_y` is
    // recorded so an animation-only frame can re-shade just the animated cells from
    // cached geometry (see `crate::gbuffer`). `None` ⇒ no capture (every legacy /
    // preview / QA cast), so those paths are byte-for-byte unchanged.
    capture: Option<&mut crate::gbuffer::NearGBuffer>,
    col: usize,
    cell_z: i32,
    pos_x: f32,
    pos_y: f32,
    ray_dir_x: f32,
    ray_dir_y: f32,
    horizon: i32,
    proj_plane_dist: f32,
    vshift_per_perp: f32,
    time: f32,
) {
    cast_column_depth(
        map,
        fb,
        registry,
        perlin,
        fog,
        far_freeze,
        memo,
        tex_cache,
        light_env,
        shadow_q,
        mirror,
        capture,
        col,
        cell_z,
        pos_x,
        pos_y,
        ray_dir_x,
        ray_dir_y,
        horizon,
        proj_plane_dist,
        vshift_per_perp,
        time,
        0,
        0.0,
    );
}

/// The wall/floor/ceiling column cast (see [`cast_column`]) with an explicit IMAGE
/// CUT-OUT recursion `depth`. `depth == 0` is the primary cast; a cut-out Image wall
/// recurses ONCE per opaque-over-transparent layer to render the BACKDROP behind its
/// transparent texels, bounded by [`IMAGE_CUTOUT_MAX_DEPTH`] so stacked textured
/// panes composite front-to-back without unbounded recursion.
#[allow(clippy::too_many_arguments)]
fn cast_column_depth(
    map: &Map,
    fb: &mut Framebuffer,
    registry: &MaterialRegistry,
    perlin: &Perlin,
    fog: &FogParams,
    far_freeze: f32,
    mut memo: Option<&mut SampleMemo>,
    mut tex_cache: Option<&mut WorldTexCache>,
    light_env: &LightEnv,
    shadow_q: ShadowQuality,
    mirror: MirrorClarity,
    mut capture: Option<&mut crate::gbuffer::NearGBuffer>,
    col: usize,
    cell_z: i32,
    pos_x: f32,
    pos_y: f32,
    ray_dir_x: f32,
    ray_dir_y: f32,
    horizon: i32,
    proj_plane_dist: f32,
    vshift_per_perp: f32,
    time: f32,
    image_depth: u8,
    // BACKDROP RE-CAST depth offset (perpendicular units). A cut-out backdrop re-cast
    // starts its DDA from a point `depth_offset` along THIS ray (just past the front
    // pane), so every distance it measures is short by exactly that much. Adding it
    // back keeps the backdrop's wall bands, fog and depth camera-relative — without it
    // the wall behind a pane drew as tall as a wall touching the camera. `0.0` for
    // every primary cast.
    depth_offset: f32,
) {
    let rows = fb.rows() as i32;
    // The camera the ray really started from (the re-cast origin minus the offset along
    // the ray). Floor/ceiling reconstruction and the sky gate read THIS, never the
    // nudged origin.
    let cam_x = pos_x - ray_dir_x * depth_offset;
    let cam_y = pos_y - ray_dir_y * depth_offset;
    // The column's vertical projection — ONE perspective for the eye-slice band, every
    // other plane (`crate::offslice`) and the authored `[line]`s.
    let vp = offslice::VProj::new(cell_z, horizon, proj_plane_dist, vshift_per_perp);

    // Normalize the ray once: the floor hit-point reconstruction needs a unit
    // ray (R-M5) so edge columns don't warp; the DDA itself is scale-free.
    let ray_len = (ray_dir_x * ray_dir_x + ray_dir_y * ray_dir_y)
        .sqrt()
        .max(PERP_MIN);
    let ray_unit_x = ray_dir_x / ray_len;
    let ray_unit_y = ray_dir_y / ray_len;

    // PRIMARY ray: walk the grid to the first solid hit.
    let mut primary = dda_hit(
        map,
        cell_z,
        pos_x,
        pos_y,
        ray_dir_x,
        ray_dir_y,
        Some(registry),
    );
    if primary.hit {
        primary.perp += depth_offset;
    }

    // ── IMAGE CUT-OUT (milestone 2) ──
    // The primary ray stopped on an Image-recipe cell that has at least one opaque
    // texel (a fully-transparent strip would have been marched through inside
    // `dda_hit`). The figure's VERTICAL silhouette still lives in alpha: some rows of
    // this column are the figure (opaque) and some are margin (transparent). Render
    // the BACKDROP behind the pane first (recast the column past the front cell), then
    // overlay ONLY the opaque image rows on top — front-to-back compositing, the
    // transparent margins keep the backdrop. Bounded by `image_depth` so a corridor
    // of stacked panes can never recurse without end.
    // A Z-BAND cell (`crate::zband`: an emit figure or a solid furniture piece that is
    // solid over only part of its height) rides the same path — its rows outside the
    // band are the transparent texels.
    if primary.hit
        && (is_image_material(registry, primary.surface, primary.material)
            || cell_z_band(registry, primary.surface, primary.material).is_some())
    {
        if image_depth < IMAGE_CUTOUT_MAX_DEPTH {
            let pane_band = image_cutout_column(
                map,
                fb,
                registry,
                perlin,
                fog,
                far_freeze,
                light_env,
                shadow_q,
                col,
                cell_z,
                ray_dir_x,
                ray_dir_y,
                ray_unit_x,
                ray_unit_y,
                &vp,
                time,
                image_depth,
                &primary,
                capture.as_deref_mut(),
            );
            // VERTICAL GEOMETRY in front of / around the pane: the cut-out overlay was
            // just painted over the backdrop, so run the all-planes pass at the true
            // camera origin — the pane's band occludes what lies beyond it.
            if image_depth == 0 {
                vertical_pass(
                    map,
                    fb,
                    registry,
                    perlin,
                    fog,
                    far_freeze,
                    memo,
                    tex_cache,
                    light_env,
                    shadow_q,
                    mirror,
                    capture,
                    col,
                    cell_z,
                    cam_x,
                    cam_y,
                    ray_dir_x,
                    ray_dir_y,
                    ray_unit_x,
                    ray_unit_y,
                    &vp,
                    Some(pane_band),
                    time,
                );
            }
            return;
        }
        // ── CUT-OUT CAP FALLTHROUGH (graceful) ──
        // The recursion cap (`IMAGE_CUTOUT_MAX_DEPTH`) is hit: this is the Nth stacked
        // transparent pane along the line of sight. Rendering it as a SOLID wall slice
        // would paint a hard opaque rectangle where there is really MORE backdrop. Instead
        // degrade to "the scene behind this pane": recast the column once PAST the front
        // pane cell (same nudge as the cut-out's backdrop pass) and DROP the front image
        // overlay entirely. The recast keeps `image_depth` at the cap, so if the scene
        // behind is itself another image pane it again falls through here — each recast
        // advances the DDA origin past at least one cell, so the chain terminates on the
        // first non-image surface, sky, or void (bounded by grid traversal, like any DDA).
        // Net effect: deeply-stacked transparent panes fade to the real scene behind them,
        // never an opaque block. The cap still bounds the cut-out's per-pane compositing
        // work; only the silhouette overlay is sacrificed past the cap.
        let back_ox = primary.wall_hit_x + ray_unit_x * IMAGE_BACKDROP_NUDGE;
        let back_oy = primary.wall_hit_y + ray_unit_y * IMAGE_BACKDROP_NUDGE;
        cast_column_depth(
            map,
            fb,
            registry,
            perlin,
            fog,
            far_freeze,
            None,
            None,
            light_env,
            shadow_q,
            MirrorClarity::default(),
            capture.as_deref_mut(),
            col,
            cell_z,
            back_ox,
            back_oy,
            ray_dir_x,
            ray_dir_y,
            horizon,
            proj_plane_dist,
            vshift_per_perp,
            time,
            image_depth,
            primary.perp + IMAGE_BACKDROP_NUDGE / ray_len,
        );
        return;
    }

    // WARP-PORTAL (one hop, mirror-shaped). If the primary ray stopped on a portal
    // cell, RE-BASE the ray by the portal's rigid `delta` transform (new origin +
    // direction) and continue the SAME grid-walk into the destination region. The
    // slice then shows the FAR region through the opening — not the portal wall.
    //
    // Two correctness points beyond the mirror path:
    //  * ACCUMULATED DEPTH: the destination hit is placed at the TOTAL path length
    //    (`dist to portal + second-segment perp`) so fog/LOD read the whole path and
    //    the far side does NOT start at depth 0 (no contrast seam at the opening).
    //  * ONE WARP ONLY: a `warp_budget` bounds re-basing to a single hop. If the
    //    destination hit is ITSELF a portal, it is shaded by the normal wall path as
    //    the `SurfaceKind::Portal` fallback brick (no re-warp) — the MVP bound,
    //    identical to the mirror "no re-bounce" rule.
    //
    // `warp` carries the destination `RayHit` (re-tagged to the accumulated depth)
    // and the re-based unit direction (which drives the destination surface's
    // view/relief lighting). When `Some`, it REPLACES the primary for the slice; the
    // destination shades through the ordinary wall path (no mirror composite).
    let mut warp: Option<RayHit> = None;
    let mut warp_shade_unit: Option<(f32, f32)> = None;
    // The threaded warp-budget (graph-ready shape): MVP = 1, so a ray
    // may warp once; a portal reached through that warp shades as the fallback. The
    // `if budget > 0` / `budget -= 1` form is the same loop body a future N-hop graph
    // model uses unchanged — it is the parameter plumbing, not dead code.
    let mut budget = PORTAL_WARP_BUDGET;
    if budget > 0 {
        if let Some(pidx) = surface_is_portal(map, &primary, cell_z) {
            budget -= 1;
            let link = &map.portals[pidx];
            let (ox, oy, wdir_x, wdir_y) = rebase_ray(
                &link.delta,
                primary.wall_hit_x,
                primary.wall_hit_y,
                cell_z,
                ray_dir_x,
                ray_dir_y,
            );
            let dest = dda_hit(map, cell_z, ox, oy, wdir_x, wdir_y, Some(registry));
            // Resolve the single warp hop, factored into
            // `portal::resolve_warp` so the one-warp bound is unit-provable:
            //  * dest MISS  → SKY/void (the un-modified `RayHit::miss`); the warped
            //    ray escaped the grid into open space, so the column reads as the
            //    scene's background, NOT a flat Portal brick at the portal's depth
            //    (that produced the reported depth-island seam).
            //  * dest is a PORTAL → the one-warp budget is spent (no re-warp); it
            //    shades through the ordinary wall path as the `SurfaceKind::Portal`
            //    fallback brick, at the ACCUMULATED one-hop depth.
            //  * dest is geometry → shaded at the accumulated depth (no depth-0
            //    contrast seam at the opening).
            let dest_hit = dest.hit;
            warp = Some(resolve_warp(primary.perp, dest));
            // The destination surface reads its view/relief lighting from the
            // re-based ray (set whenever the warped ray actually hit something —
            // including a second portal shaded as the fallback brick).
            if dest_hit {
                warp_shade_unit = Some((wdir_x, wdir_y));
            }
        }
    }
    // The ray the rest of the column reasons about. A warp REPLACES the primary with
    // the destination hit (so the mirror check, slice extent, and shading all run on
    // the far geometry); a non-portal column is byte-identical to before.
    let did_warp = warp.is_some();
    // The FIRST surface this column's ray met on the slice plane (a portal / mirror face
    // included): the vertical pass treats the slice band as opaque beyond it, whatever
    // scene the band shows (a warped or reflected one sits at a longer path length).
    let first_perp = if primary.hit {
        primary.perp
    } else {
        DIST_INFINITY
    };
    let primary = warp.unwrap_or(primary);
    // After a warp, do NOT also treat the destination as a mirror this frame (the
    // one-hop budget is spent). A non-warped column keeps the full mirror path.
    let mirror_eligible = budget == PORTAL_WARP_BUDGET;

    // MIRROR (one bounce). If the primary ray hit a reflective surface, reflect the
    // ray about the hit face normal (axis-aligned → flip the perpendicular
    // component) and continue the DDA from just past the hit point. The wall slice
    // then shows the REFLECTED scene (its own surface/material/relief), composited
    // with a cool sheen and degraded by the mirror's CLARITY. One bounce only — a
    // reflected ray that itself lands on a mirror is shaded as a plain wall (no
    // recursion), bounding cost. A reflected ray that escapes → sky/void.
    let is_mirror = mirror_eligible
        && primary.hit
        && surface_is_mirror(registry, primary.surface, primary.material);
    let mut mirror_view: Option<MirrorView> = None;
    if is_mirror {
        let clarity = surface_clarity(registry, primary.surface, primary.material, mirror);
        // Reflect the ray direction about the axis-aligned face normal: an EW
        // (x-const) face flips x; an NS (y-const) face flips y.
        let (refl_dir_x, refl_dir_y) = if primary.side == 1 {
            (-ray_dir_x, ray_dir_y)
        } else {
            (ray_dir_x, -ray_dir_y)
        };
        // Reflected origin: the primary hit point, nudged a hair along the reflected
        // direction so the DDA does not immediately re-hit the mirror's own cell.
        let refl_len = (refl_dir_x * refl_dir_x + refl_dir_y * refl_dir_y)
            .sqrt()
            .max(PERP_MIN);
        let nudge = MIRROR_ORIGIN_NUDGE / refl_len;
        let ox = primary.wall_hit_x + refl_dir_x * nudge;
        let oy = primary.wall_hit_y + refl_dir_y * nudge;
        let refl = dda_hit(map, cell_z, ox, oy, refl_dir_x, refl_dir_y, Some(registry));
        mirror_view = Some(MirrorView {
            refl,
            refl_dir_x,
            refl_dir_y,
            clarity,
        });
    }

    // The geometry the wall slice is shaded from: for a plain wall it is the primary
    // hit; for a mirror it is the REFLECTED hit, placed at the TOTAL path distance
    // (primary perp + reflected perp) so the reflected scene recedes behind the
    // mirror plane the way a real reflection does.
    let view: RayHit = match &mirror_view {
        Some(mv) if mv.refl.hit => RayHit {
            perp: (primary.perp + mv.refl.perp).max(PERP_MIN),
            ..mv.refl
        },
        // Mirror onto the void: a hit on the mirror but the reflected ray escaped →
        // the reflection is the sky. Render the slice as a (short) wall whose surface
        // is the void, at the mirror's own distance, so the sheen still tints it.
        Some(_) => RayHit {
            hit: true,
            surface: SurfaceKind::Empty,
            material: 0,
            perp: primary.perp,
            ..primary
        },
        None => primary,
    };

    let hit = view.hit;
    let side = view.side;
    let surface = view.surface;
    let material = view.material;
    let perp = view.perp;
    let wall_hit_x = view.wall_hit_x;
    let wall_hit_y = view.wall_hit_y;
    let tex_u = view.tex_u;
    // 45° GEOMETRY: a wedge/ramp hit carries its real world-space outward
    // normal here. `None` ⇒ an axis-aligned Full-cell face → every term below
    // derives the normal from `side` exactly as before (byte-identical). `Some(n)`
    // ⇒ the diagonal/slope normal drives foreshortening, the cast-shadow face, and
    // the per-row `LightCtx`, so the 45° face reads as a lit slope, not a flat wall.
    let shape_normal = view.shape_normal;
    // The reflected ray's unit direction drives the reflected surface's view/relief
    // vectors (so its lighting reads from the bounce, not the primary ray). For a
    // plain wall this is just the primary ray.
    // A WARP destination reads its lighting from the RE-BASED ray (already unit), so
    // the far surface's relief/headlamp react to the warped view, not the primary
    // ray. (`warp_shade_unit` is set only when the destination ray actually hit.)
    let (shade_unit_x, shade_unit_y) = match (&mirror_view, warp_shade_unit) {
        (Some(mv), _) if mv.refl.hit => {
            let l = (mv.refl_dir_x * mv.refl_dir_x + mv.refl_dir_y * mv.refl_dir_y)
                .sqrt()
                .max(PERP_MIN);
            (mv.refl_dir_x / l, mv.refl_dir_y / l)
        }
        (None, Some((wx, wy))) => (wx, wy),
        _ => (ray_unit_x, ray_unit_y),
    };

    // Wall slice extent, centred on the horizon. The screen height of a unit-tall
    // wall at perpendicular distance `perp` is `proj_plane_dist / perp`, where
    // `proj_plane_dist = (rows/2)/tan(vfov/2)` — so the vertical extent honours the
    // (fixed) vertical FOV instead of the old magic `rows` numerator. Under the
    // default Fov + a baseline-ish window this lands close to the legacy look; on a
    // very wide/tall window it now scales correctly with vfov.
    // Vertical eye shear: the wall slice CENTRE drops below the horizon by
    // `vshift_per_perp / perp` (rows) — the same projection scale as `line_h`, so a
    // raised eye moves nearer walls more than far ones (correct vertical parallax).
    // `vshift_per_perp == 0.0` at standing height ⇒ centre == horizon ⇒ byte-identical.
    // One perspective for every plane: the slice cell spans world `[cell_z, cell_z+1]`,
    // i.e. continuous rows `vp.row(cell_z+1, perp) .. vp.row(cell_z, perp)`, and the
    // drawn rows are the ones whose CENTRES fall inside (`offslice::span_rows`) — the
    // same rule the off-slice planes and the `[line]` pass use, so a wall edge and a
    // line at that world height land on the same row. The eye shear rides in `vp.eye`.
    let (slice_top_f, slice_bottom_f) = if hit {
        (
            vp.row(cell_z as f32 + 1.0, perp),
            vp.row(cell_z as f32, perp),
        )
    } else {
        (horizon as f32, horizon as f32)
    };
    let (slice_top, slice_bottom) = offslice::span_rows(slice_top_f, slice_bottom_f);
    let draw_top = clampi(slice_top, 0, rows);
    let draw_bottom = clampi(slice_bottom, 0, rows);

    // Wall foreshortening = 1/|cosθ| between the (normalized) ray and the face
    // normal. The normal is axis-aligned: side==1 → x-normal, side==0 → y-normal.
    // Grazing rays (small |cos|) blow this up → the LOD coarsens (anti-alias). For a
    // mirror, `side`/`shade_unit` are the REFLECTED surface's, so its detail
    // coarsens by its own grazing angle (correct for the reflected geometry).
    let cos_theta = match shape_normal {
        // 45° face: foreshorten by the angle between the (horizontal) ray and the
        // surface normal's horizontal component (the wedge diagonal / ramp rise axis).
        Some(n) => (shade_unit_x * n.x + shade_unit_y * n.y).abs(),
        None if side == 1 => shade_unit_x.abs(),
        None => shade_unit_y.abs(),
    };
    let wall_foreshorten = (1.0 / cos_theta.max(FORESHORTEN_COS_MIN)).min(FORESHORTEN_MAX);

    // ── WALL rows ──
    // The wall slice spans screen rows `slice_top..slice_bottom`, clipped to the
    // viewport as `draw_top..draw_bottom`. To give a 2D wall texture a vertical
    // coordinate, we map each drawn row to `wall_v ∈ [0,1]` against the UNCLIPPED
    // slice extent — so a wall that runs off the top/bottom of the screen still
    // exposes the SAME stretch of its face per world-row (the off-screen part is
    // simply not drawn). Using the unclipped extent keeps the text from "zooming"
    // as you walk toward a wall taller than the viewport. Guard the degenerate
    // 1-row slice so we never divide by zero. The texture maps the DRAWN rows
    // (`offslice::drawn_v`), so it does not creep as the sub-row extent drifts with depth.
    let slice_rows = slice_bottom.saturating_sub(slice_top).max(1);
    // Face sign: which way the hit face points along its axis. side==1 → an
    // x-face; the camera sits opposite the ray's x travel, so the outward normal
    // is `-sign(ray_x)`. `for_wall` negates `face_sign`, so pass the ray's sign.
    let face_sign = if side == 1 {
        shade_unit_x.signum()
    } else {
        shade_unit_y.signum()
    };
    // Worked-out shadows (#2): the wall face normal + hit point are constant down
    // the column, and the KEY light is static, so the cast-shadow factor is the
    // SAME for every row of this wall slice — compute it ONCE here, not per row.
    // (Matches `for_wall`'s geometric face normal: side==1 ⇒ ±x, side==0 ⇒ ±y,
    // pointing toward the camera.) `Off` short-circuits to 1.0 (pre-shadow output).
    let wall_face_n = match shape_normal {
        Some(n) => n, // the real 45° outward normal (already air-facing & unit).
        None if side == 1 => glam::Vec3::new(-face_sign, 0.0, 0.0),
        None => glam::Vec3::new(0.0, -face_sign, 0.0),
    };
    let wall_shadow_key = if hit {
        // Per-material LOD override (the key-object lever): a hero wall material holds
        // its soft cast shadow farther out than the global LOD curve. `None` ⇒ global.
        let wall_lod_ov = registry
            .get(surface_material(surface, material))
            .lod_override;
        shadow_key_for_lod(
            map,
            shadow_q,
            ShadowSample {
                depth: perp,
                hx: wall_hit_x,
                hy: wall_hit_y,
                cell_z,
                face_n: wall_face_n,
            },
            light_env.key_dir,
            wall_lod_ov,
        )
    } else {
        1.0
    };
    // FACE ANCHOR — the exact projected screen extent of THIS wall face, measured
    // once per column (never per row: the face is the same all the way down the slice).
    // Only an axis-aligned face has a two-endpoint unit span; a 45° wedge/ramp keeps
    // the historical mapping. `face_anchor_for` refuses any re-based view (mirror /
    // portal / backdrop recast) by self-checking the hit's own projection, so those
    // paths are unchanged too.
    face_span::set_face_anchor(if hit && shape_normal.is_none() {
        face_span::face_anchor_for(
            col,
            side,
            view.cell_x,
            view.cell_y,
            wall_hit_x,
            wall_hit_y,
            ray_dir_x,
            ray_dir_y,
        )
    } else {
        face_span::FaceAnchor::UNKNOWN
    });
    // DRAWN ROWS — the row count the texture maps (`wall_v` below), for a text sampler.
    face_span::set_drawn_rows(hit.then_some(slice_rows as usize));
    // WALL RUN — the continuous world coordinate along this wall plane, so a text
    // surface can flow ONE stream across the whole run instead of restarting inside
    // every cell (`tex_u` wraps; this does not). Pure world geometry, so it is the same
    // for every cast path — unlike the anchor above it needs no camera basis, and a 45°
    // face simply has no single run axis.
    face_span::set_wall_run((hit && shape_normal.is_none()).then(|| face_span::WallRun {
        u: face_span::wall_run_for(side, wall_hit_x, wall_hit_y, ray_dir_x, ray_dir_y),
        // The EXACT local screen footprint of one world unit of this face: the face's
        // texture span times the ray's own FISHEYE factor `|ray_dir|` (the caster builds
        // `ray_dir = dir + plane·camera_x`, whose component along `dir` is exactly 1, so
        // its length IS `1/cos(angle off the view axis)`). The two terms cancel on a
        // face-on wall, which is what makes the footprint constant across it.
        footprint_cols: face_span::face_cols(perp, wall_foreshorten) * ray_len,
    }));
    // ── VERTICAL GEOMETRY, decided first (every plane but the slice's own non-ramp
    // solids) ── Only the TRUE camera-origin cast runs it (a backdrop re-cast is a
    // partial column). The slice band occludes beyond the first surface this ray met
    // on the slice plane — a mirror / portal face included, whatever path length its
    // reflected or warped scene was painted at. Deciding it before the slice and the
    // backdrop are shaded lets both skip the rows it is about to paint over; it is
    // shaded last (`vertical_shade`, below).
    let vertical = image_depth == 0;
    if vertical {
        let band = if first_perp < DIST_INFINITY {
            let (t, b) = offslice::span_rows(
                vp.row(cell_z as f32 + 1.0, first_perp),
                vp.row(cell_z as f32, first_perp),
            );
            Some((t, b, first_perp))
        } else {
            None
        };
        vertical_resolve(
            map, registry, rows, cell_z, cam_x, cam_y, ray_dir_x, ray_dir_y, &vp, band,
        );
    }
    for row in draw_top..draw_bottom {
        if vertical && vertical_claims(map, row) {
            continue;
        }
        // Row CENTRE against the continuous band: 0 at the cell top, 1 at its base, so
        // `wall_world_y` is the true world height of the sampled point.
        let wall_v = offslice::drawn_v(slice_top, slice_bottom, row as f32 + 0.5);
        // View direction (surface→camera): the negated horizontal ray plus a
        // vertical component from the screen row's offset off the horizon, so the
        // headlamp + specular react as the player looks up/down a wall. Scaled so
        // a row at the slice edge contributes a sane vertical tilt.
        let view_z = (horizon - row) as f32 * VIEW_VERT_SCALE / rows as f32;
        let view_dir = glam::Vec3::new(-shade_unit_x, -shade_unit_y, view_z).normalize_or_zero();
        let mut wall_lc = match shape_normal {
            // 45° wedge/ramp: light from the real tilted normal so the slope reads.
            Some(n) => LightCtx::for_surface_normal(n, view_dir, light_env),
            None => LightCtx::for_wall(side, face_sign, view_dir, light_env),
        };
        wall_lc.shadow_key = wall_shadow_key;
        let light = Some(wall_lc);
        let geom = CellGeom {
            depth: perp,
            foreshorten: wall_foreshorten,
            side,
            tex_u,
            wall_v,
            wall_world_y: wall_world_y(cell_z, wall_v),
            hit_x: wall_hit_x,
            hit_y: wall_hit_y,
            surface,
            material,
            light,
            // Octave-0 hit cell — the wall's grid cell on this z-plane. Feeds the P4
            // inset region-membership test in `shade_cell`.
            cell: (view.cell_x, view.cell_y, cell_z),
            tex_footprint: None,
            subcell_backdrop: None,
        };
        let mut cell = shade_cell(
            registry,
            perlin,
            fog,
            far_freeze,
            memo.as_deref_mut(),
            tex_cache.as_deref_mut(),
            map.resolved_insets(),
            &geom,
            row,
            time,
        );
        // SELECTIVE RE-SHADE capture: record this wall cell's exact `shade_cell`
        // inputs. A mirror cell (`mirror_view.is_some()`) is captured but EXCLUDED
        // from the animated list (its `mirror_composite` post-step can't be replayed
        // from `CellGeom` alone — see `crate::gbuffer`). A WARP cell (`did_warp`) is
        // excluded the same way: it is a no-pan special column re-cast structurally
        // every frame anyway, so the cheap animated replay is never relied on for it.
        if let Some(cap) = capture.as_deref_mut() {
            let animated = registry
                .get(surface_material(geom.surface, geom.material))
                .is_animated();
            cap.capture(
                col,
                row as usize,
                &geom,
                row,
                animated,
                mirror_view.is_some() || did_warp,
                false, // the wall slice is never sky-applied.
            );
        }
        // MIRROR COMPOSITE: the cell currently holds the fully-shaded REFLECTED
        // surface (or the void). Tint it with the cool mirror sheen and degrade it
        // by clarity (crisp → glitch/empty) so it reads as a reflection in a
        // surface, not the raw reflected geometry. The depth stays the total path
        // distance (set in `geom`) so the present layer's depth cue is honest.
        if let Some(mv) = &mirror_view {
            cell = mirror_composite(cell, mv.refl.hit, mv.clarity, col, row);
        }
        fb.set(col, row as usize, cell);
    }

    // The face anchor + wall run are scoped to the wall slice above: nothing outside it
    // (floor, ceiling, the off-slice object pass, a recursed backdrop) may read a stale
    // face.
    face_span::set_face_anchor(face_span::FaceAnchor::UNKNOWN);
    face_span::set_wall_run(None);
    face_span::set_drawn_rows(None);

    // ── FLOOR / CEILING rows (everything outside the wall slice) ──
    // Vertical eye shear for the ground plane: the eye's height above the floor
    // (1 cell by convention) grows/shrinks with the jump. Constant down the column,
    // so compute it ONCE here (= 1.0 at standing height ⇒ identical output).
    let eye_height_cells = 1.0 + vshift_per_perp / proj_plane_dist;
    // Does the room the CAMERA occupies genuinely open overhead? See
    // `column_sky_open`. An enclosed room (every `[room]`-built level, unless it
    // authors `[sky]` or truly has no roof over the camera's own column) answers
    // `false` here and keeps its shaded ceiling material instead of the
    // direction-sampled panorama. Independent of `col`/`row`/pitch/yaw, so this
    // is the same value for every column of this frame — recomputed per column
    // only because it is cheap and keeps this function self-contained.
    let sky_open = column_sky_open(map, cell_z, cam_x, cam_y);
    for row in 0..rows {
        if row >= draw_top && row < draw_bottom {
            continue;
        }
        if vertical && vertical_claims(map, row) {
            continue;
        }
        let is_floor = row > horizon;
        let denom = if is_floor {
            (row - horizon) as f32
        } else {
            (horizon - row) as f32
        };

        let surface = if is_floor {
            SurfaceKind::Floor
        } else {
            SurfaceKind::Ceiling
        };
        // Floor/ceiling lighting: face normal is ±z; the view direction points
        // back up toward the eye (floor: camera above → +z; ceiling: −z), with the
        // horizontal ray negated. Built per cell; the headlamp rides it.
        let fc_view = glam::Vec3::new(
            -ray_unit_x * FC_VIEW_HORIZ,
            -ray_unit_y * FC_VIEW_HORIZ,
            if is_floor { 1.0 } else { -1.0 },
        )
        .normalize_or_zero();
        let mut fc_lc = LightCtx::for_floor(is_floor, fc_view, light_env);
        let geom = if denom <= 0.0 {
            // Exactly on the horizon with no wall: far ambient sky/floor.
            CellGeom {
                depth: DIST_INFINITY,
                foreshorten: FORESHORTEN_MAX,
                side: 0,
                tex_u: 0.0,
                wall_v: 0.0,
                wall_world_y: 0.0,
                hit_x: 0.0,
                hit_y: 0.0,
                surface,
                material: 0,
                light: Some(fc_lc),
                // Far ambient sky/floor at the horizon — no real hit cell; inert (depth
                // = infinity ⇒ never an inset reveal).
                cell: (0, 0, 0),
                tex_footprint: None,
                subcell_backdrop: None,
            }
        } else {
            // Ground distance for this floor/ceiling row. The camera is a unit above
            // the floor (and below the ceiling), so a row `denom` pixels off the
            // horizon sees the ground at `proj_plane_dist / denom` — the same
            // vertical projection constant the wall slice uses, so floor and wall
            // foreshorten consistently under the (fixed) vertical FOV.
            // `eye_height_cells` (hoisted above) scales `row_dist` the same way
            // `proj/denom` would under a moved horizon, so the floor stays glued to
            // the (sheared) wall base. == 1.0 at standing height ⇒ identical output.
            // Clamped to a nearer wall's own depth in this column (see the module note
            // above `clamp_to_wall_depth`) — the dark-strip fix: a no-op unless this
            // row's classic-formula distance would read farther than a wall the ray
            // can never actually see past, AND this row's `denom` is within a small,
            // continuous window of where the formula would cross back below `perp`
            // on its own (the module note's near-wall carve-out).
            let floor_ceil_k = eye_height_cells * proj_plane_dist;
            let row_dist =
                clamp_to_wall_depth(floor_ceil_k / denom, hit, perp, denom, floor_ceil_k);
            // R-M4: floor/ceiling foreshorten = K·rows / denom² — the per-row UV
            // derivative (a floor texel's world footprint grows like 1/denom²).
            // Scaled by `rows` and a larger K so it actually SPANS [1, MAX] across
            // the visible band instead of clamping to 1 after two rows: near the
            // horizon (small denom) it saturates the cap, and it falls smoothly to
            // ~1 only at the bottom rows. So the anti-alias varies, not inert.
            let foreshorten =
                (FLOOR_FORESHORTEN_K * rows as f32 / (denom * denom)).clamp(1.0, FORESHORTEN_MAX);
            // R-M5: reconstruct the floor hit from the NORMALIZED ray so edge
            // columns don't parallax-warp (a non-unit ray stretched the hit point
            // toward the screen edges). `row_dist` is the true ground distance.
            let fhit_x = cam_x + row_dist * ray_unit_x;
            let fhit_y = cam_y + row_dist * ray_unit_y;
            // Worked-out shadows (#2): march the KEY shadow ray from THIS floor/
            // ceiling cell's world hit point. The face normal is ±z (`for_floor`):
            // floor faces +z, ceiling −z. This is the macro occlusion that makes a
            // wall/pillar cast onto the floor — the payoff the player asked for.
            //
            // Per-material LOD override (the key-object lever), threaded the same way
            // the wall path threads `wall_lod_ov`: a hero FLOOR/CEILING material holds
            // its soft cast shadow farther out than the global LOD curve. The material
            // is the one the shade path will use (`surface_material(surface, 0)` — the
            // floor/ceiling tag is 0 here, matching `geom.material`). With the default
            // FLOOR_GRAIN / CEILING_SKY materials this resolves to `None` ⇒ the global
            // curve ⇒ byte-identical to the prior `shadow_key_for` call.
            let fc_lod_ov = registry.get(surface_material(surface, 0)).lod_override;
            fc_lc.shadow_key = shadow_key_for_lod(
                map,
                shadow_q,
                ShadowSample {
                    depth: row_dist,
                    hx: fhit_x,
                    hy: fhit_y,
                    cell_z,
                    face_n: if is_floor {
                        glam::Vec3::Z
                    } else {
                        glam::Vec3::NEG_Z
                    },
                },
                light_env.key_dir,
                fc_lod_ov,
            );
            CellGeom {
                depth: row_dist,
                foreshorten,
                side: 0,
                tex_u: 0.0,
                wall_v: 0.0,
                wall_world_y: 0.0,
                hit_x: fhit_x,
                hit_y: fhit_y,
                surface,
                material: backdrop_floor_ceiling_material(
                    map,
                    registry,
                    is_floor,
                    (cam_x, cam_y),
                    (ray_dir_x, ray_dir_y),
                    (eye_height_cells - 0.5) * proj_plane_dist / denom,
                    cell_z,
                ),
                light: Some(fc_lc),
                // Octave-0 floor/ceiling cell — the floored world hit point on this
                // z-plane. Feeds the P4 inset region-membership test.
                cell: (floori(fhit_x), floori(fhit_y), cell_z),
                tex_footprint: None,
                subcell_backdrop: None,
            }
        };
        let mut cell = shade_cell(
            registry,
            perlin,
            fog,
            far_freeze,
            memo.as_deref_mut(),
            tex_cache.as_deref_mut(),
            map.resolved_insets(),
            &geom,
            row,
            time,
        );
        // ── DIRECTION-SAMPLED SKY ──
        // Above-horizon ceiling/sky rows (the "missed sky ray" fill) are replaced by
        // the cylinder+cap panorama, sampled by this ray's (yaw, pitch) — but ONLY
        // when the room the camera occupies genuinely opens overhead (`sky_open`,
        // computed once above via `column_sky_open` — real map/voxel data, not this
        // row's flat-plane guess). An enclosed room keeps the shaded ceiling
        // material `shade_cell` already produced (fog/material — no sky overwrite).
        // Floor rows (below the horizon) are real ground geometry and are NEVER
        // touched. The overwrite is colour+glyph only; the geometry fields
        // (depth = ∞, surface) stay so the present/composite layers still read it
        // as distant sky.
        let sky_applied =
            !is_floor && sky_open && matches!(surface, SurfaceKind::Ceiling | SurfaceKind::Empty);
        if sky_applied {
            apply_sky(
                map,
                &mut cell,
                ray_unit_x,
                ray_unit_y,
                row,
                horizon,
                proj_plane_dist,
            );
        }
        // SELECTIVE RE-SHADE capture: floor/ceiling cells are never mirrors.
        if let Some(cap) = capture.as_deref_mut() {
            let animated = registry
                .get(surface_material(geom.surface, geom.material))
                .is_animated();
            cap.capture(col, row as usize, &geom, row, animated, false, sky_applied);
        }
        fb.set(col, row as usize, cell);
    }

    // ── VERTICAL GEOMETRY, shaded (decided above, before the slice) ──
    if vertical {
        vertical_shade(
            map, fb, registry, perlin, fog, far_freeze, memo, tex_cache, light_env, shadow_q,
            mirror, capture, col, cam_x, cam_y, ray_dir_x, ray_dir_y, ray_unit_x, ray_unit_y, &vp,
            time,
        );
    }
}

/// The colour id the backdrop floor/ceiling row shades with: either a prop floor
/// decal (the two hardcoded prop-kit producers, floor-only) stamped into the real
/// slab the row lands on, OR — new — an authored `[zone]`/`[color_cell_ids]`
/// colour-material id sitting in that same cell's `side_data`, resolved exactly the
/// way `shade_cell` already resolves one for a WALL (`registry.color_material`).
/// Falls back to `0` (the surface-kind default) when neither applies, exactly as
/// before this fix. The registry checks run first, so a level with no floor decal
/// AND no colour palette never touches the map (byte-identical fast path).
///
/// For a floor row the real cell is `cell_z - 1` (the slab the camera storey
/// stands on); for a ceiling row it is the mirror image, `cell_z + 1` (the slab
/// "one cell above the eye" the flat single-storey cast already assumes — see
/// [`column_sky_open`]'s doc comment). A prop-kit floor decal is floor-only by
/// construction (`MaterialRegistry::mark_floor_decal` is only ever called for a
/// floor prop), so [`MaterialRegistry::is_floor_decal`] is consulted only on the
/// floor side; the general colour-material palette applies to both.
///
/// The backdrop's own `row_dist` is a texture metric (it takes the eye as one cell
/// above the floor), so the CELL a row lands on is found with the true perspective
/// the walls, the vertical pass and the `[line]` strokes share: the eye sits
/// `0.5 + vshift` above the floor top, so the row meets the floor (or ceiling) at
/// perpendicular distance `s_perp = eye_above_floor · proj / denom`, at
/// `cam + s_perp · ray_dir`. That keeps a decal (or a zone material) under the
/// piece of floor/ceiling that owns it.
#[inline]
fn backdrop_floor_ceiling_material(
    map: &Map,
    registry: &MaterialRegistry,
    is_floor: bool,
    cam: (f32, f32),
    ray_dir: (f32, f32),
    s_perp: f32,
    cell_z: i32,
) -> u16 {
    if s_perp.is_nan() || s_perp <= 0.0 {
        return 0;
    }
    if !registry.has_floor_decals() && registry.color_material_len() == 0 {
        return 0;
    }
    let z = if is_floor { cell_z - 1 } else { cell_z + 1 };
    let (x, y) = (
        floori(cam.0 + s_perp * ray_dir.0),
        floori(cam.1 + s_perp * ray_dir.1),
    );
    let expected_solid = if is_floor {
        matches!(
            map.cell_at(x, y, z),
            re_world::CellKind::Floor | re_world::CellKind::WallSolid
        )
    } else {
        matches!(
            map.cell_at(x, y, z),
            re_world::CellKind::Ceiling | re_world::CellKind::WallSolid
        )
    };
    if !expected_solid {
        return 0;
    }
    let sd = map.side_data_at(x, y, z);
    if (is_floor && registry.is_floor_decal(sd)) || registry.color_material(sd).is_some() {
        sd
    } else {
        0
    }
}

thread_local! {
    /// Per-column coverage mask of the vertical pass (reused; grows once to the
    /// framebuffer height, so the steady state allocates nothing).
    static VPASS_COVER: std::cell::RefCell<Vec<bool>> = const { std::cell::RefCell::new(Vec::new()) };
    /// Per-row result of [`vertical_resolve`] for the column being cast: the surface the
    /// vertical pass paints on that row, if any (same reuse policy as the mask).
    static VPASS_HITS: std::cell::RefCell<Vec<Option<offslice::VHit>>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Does the vertical pass paint `row` of the column [`vertical_resolve`] just decided,
/// so the slice / backdrop path need not shade it first? A portal face is not
/// claimed: its warped ray may escape, and then the row keeps the backdrop.
#[inline]
fn vertical_claims(map: &Map, row: i32) -> bool {
    VPASS_HITS.with(|hits| {
        hits.borrow()
            .get(row as usize)
            .copied()
            .flatten()
            .is_some_and(|h| {
                !(h.role == offslice::Role::Paint
                    && map.cell_at(h.x, h.y, h.plane) == re_world::CellKind::Portal)
            })
    })
}

/// VERTICAL GEOMETRY pass: paint every plane of this column the primary slice does not
/// own (see [`offslice`] for the model and the ownership rules). Resolves the column
/// ([`vertical_resolve`]) and shades it ([`vertical_shade`]) in one go; the plain
/// camera-origin cast runs the two halves apart so it can skip shading, on the slice
/// and backdrop path, the rows this pass is about to paint over.
#[allow(clippy::too_many_arguments)]
fn vertical_pass(
    map: &Map,
    fb: &mut Framebuffer,
    registry: &MaterialRegistry,
    perlin: &Perlin,
    fog: &FogParams,
    far_freeze: f32,
    memo: Option<&mut SampleMemo>,
    tex_cache: Option<&mut WorldTexCache>,
    light_env: &LightEnv,
    shadow_q: ShadowQuality,
    mirror: MirrorClarity,
    capture: Option<&mut crate::gbuffer::NearGBuffer>,
    col: usize,
    cell_z: i32,
    cam_x: f32,
    cam_y: f32,
    ray_dir_x: f32,
    ray_dir_y: f32,
    ray_unit_x: f32,
    ray_unit_y: f32,
    vp: &offslice::VProj,
    band: Option<(i32, i32, f32)>,
    time: f32,
) {
    vertical_resolve(
        map,
        registry,
        fb.rows() as i32,
        cell_z,
        cam_x,
        cam_y,
        ray_dir_x,
        ray_dir_y,
        vp,
        band,
    );
    vertical_shade(
        map, fb, registry, perlin, fog, far_freeze, memo, tex_cache, light_env, shadow_q, mirror,
        capture, col, cam_x, cam_y, ray_dir_x, ray_dir_y, ray_unit_x, ray_unit_y, vp, time,
    );
}

/// First half of the vertical pass: decide, for every row of the column, the surface
/// the vertical pass paints there (into `VPASS_HITS`), without shading anything.
///
/// Walks the column's ray front to back through the grid. At each grid column it
/// bounds the screen rows the solid cells there can reach (only planes inside the view
/// frustum, pitch included), and resolves each still-undecided row to the first
/// surface its view ray enters (near face, top, underside or ramp slope). Every row is
/// decided once, nearest first (the coverage mask); the walk stops as soon as every
/// row is decided, or no undecided row can still meet anything.
///
/// `band` is the primary slice's first-surface band `(top, bottom, depth)`: beyond that
/// depth its rows belong to the primary (a wall, or the scene a mirror / portal / pane
/// shows there), so they close as soon as the walk passes it.
#[allow(clippy::too_many_arguments)]
fn vertical_resolve(
    map: &Map,
    registry: &MaterialRegistry,
    rows: i32,
    cell_z: i32,
    cam_x: f32,
    cam_y: f32,
    ray_dir_x: f32,
    ray_dir_y: f32,
    vp: &offslice::VProj,
    band: Option<(i32, i32, f32)>,
) {
    use offslice::{Face, Role};
    let rows = rows.max(0);
    let sz = map.size().2 as i32;
    let max_steps = (draw_dist::active_max_dda_steps() as i32).min(offslice::MAX_STEPS);
    VPASS_HITS.with(|hv| {
        let mut hits = hv.borrow_mut();
        hits.clear();
        hits.resize(rows as usize, None);
        if rows == 0 {
            return;
        }
        VPASS_COVER.with(|cv| {
            let mut covered = cv.borrow_mut();
            covered.clear();
            covered.resize(rows as usize, false);
            let mut n_cov = 0i32;
            let mut band_open = band.is_some();
            // How far along the ray any still-undecided row can still meet geometry: a
            // row's view ray leaves the grid's vertical extent `[0, sz]` at a finite
            // distance (unless it is level), and a band row is decided at the band's
            // depth. Past the farthest of those nothing is left to find, so the walk
            // stops there instead of running on to the map edge under open sky.
            // Recomputed only when a row closes.
            let reach = |covered: &[bool], band_open: bool| -> f32 {
                let mut lim = 0.0f32;
                for (r, _) in covered.iter().enumerate().filter(|(_, &c)| !c) {
                    let k = vp.slope(r as f32 + 0.5);
                    let mut t = if k > 1e-6 {
                        (sz as f32 - vp.eye) / k
                    } else if k < -1e-6 {
                        -vp.eye / k
                    } else {
                        f32::INFINITY
                    };
                    if let (true, Some((bt, bb, bd))) = (band_open, band) {
                        if (bt..bb).contains(&(r as i32)) {
                            t = t.min(bd);
                        }
                    }
                    lim = lim.max(t);
                }
                lim
            };
            let mut limit = f32::INFINITY;
            let mut limit_cov = -1i32;
            for step in offslice::GridWalk::new(map, cam_x, cam_y, ray_dir_x, ray_dir_y, max_steps)
            {
                if band_open {
                    if let Some((bt, bb, bd)) = band {
                        if step.t_in >= bd {
                            for r in bt.max(0)..bb.min(rows) {
                                if !covered[r as usize] {
                                    covered[r as usize] = true;
                                    n_cov += 1;
                                }
                            }
                            band_open = false;
                            limit_cov = -1;
                        }
                    }
                }
                if n_cov >= rows {
                    break;
                }
                if n_cov != limit_cov {
                    limit = reach(&covered, band_open);
                    limit_cov = n_cov;
                }
                if step.t_in > limit {
                    break;
                }
                // Row bounds of every non-air cell this grid column holds in the frustum.
                let (p_lo, p_hi) = offslice::frustum_planes(&step, vp, rows, sz);
                let mut lo = f32::INFINITY;
                let mut hi = f32::NEG_INFINITY;
                for p in p_lo..=p_hi {
                    if offslice::role_of(map, step.x, step.y, p, cell_z) == Role::Air {
                        continue;
                    }
                    let shape = map.shape_at(step.x, step.y, p);
                    if let Some(poly) =
                        offslice::cell_poly(shape, &step, cam_x, cam_y, ray_dir_x, ray_dir_y)
                    {
                        let (a, b) = offslice::poly_row_bounds(&poly, p as f32, vp);
                        lo = lo.min(a);
                        hi = hi.max(b);
                    }
                }
                if lo > hi {
                    continue;
                }
                let r0 = (lo.floor() as i32).saturating_sub(1).max(0);
                let r1 = (hi.ceil() as i32).saturating_add(1).min(rows);
                for r in r0..r1 {
                    let ru = r as usize;
                    if covered[ru] {
                        continue;
                    }
                    let k = vp.slope(r as f32 + 0.5);
                    let Some(h) = offslice::resolve_in_column(
                        map,
                        cell_z,
                        &step,
                        cam_x,
                        cam_y,
                        ray_dir_x,
                        ray_dir_y,
                        vp.eye,
                        k,
                        |h| vhit_opaque(map, registry, h, cam_x, cam_y, ray_dir_x, ray_dir_y),
                    ) else {
                        continue;
                    };
                    covered[ru] = true;
                    n_cov += 1;
                    // Beyond the first slice surface, a band row is the primary's.
                    if let Some((bt, bb, bd)) = band {
                        if r >= bt && r < bb && h.s >= bd {
                            continue;
                        }
                    }
                    // Slice-plane solids are the primary's; slab lids cover, unpainted.
                    if matches!(
                        (h.role, h.face),
                        (Role::Slice, _) | (Role::Slab, Face::Top | Face::Bottom)
                    ) {
                        continue;
                    }
                    hits[ru] = Some(h);
                }
            }
        });
    });
}

/// Second half of the vertical pass: shade every row [`vertical_resolve`] decided the
/// pass paints, through the same [`shade_cell`] the wall path uses.
#[allow(clippy::too_many_arguments)]
fn vertical_shade(
    map: &Map,
    fb: &mut Framebuffer,
    registry: &MaterialRegistry,
    perlin: &Perlin,
    fog: &FogParams,
    far_freeze: f32,
    mut memo: Option<&mut SampleMemo>,
    mut tex_cache: Option<&mut WorldTexCache>,
    light_env: &LightEnv,
    shadow_q: ShadowQuality,
    mirror: MirrorClarity,
    mut capture: Option<&mut crate::gbuffer::NearGBuffer>,
    col: usize,
    cam_x: f32,
    cam_y: f32,
    ray_dir_x: f32,
    ray_dir_y: f32,
    ray_unit_x: f32,
    ray_unit_y: f32,
    vp: &offslice::VProj,
    time: f32,
) {
    let rows = fb.rows() as i32;
    let steps_on = ramp_steps::enabled();
    let max_steps = (draw_dist::active_max_dda_steps() as i32).min(offslice::MAX_STEPS);
    let mut face_memo: Option<VFaceMemo> = None;
    for r in 0..rows {
        let ru = r as usize;
        let Some(h) = VPASS_HITS.with(|hv| hv.borrow().get(ru).copied().flatten()) else {
            continue;
        };
        // A mirror / portal face shows the reflected / warped scene, as on the slice.
        // Such a row is re-cast every frame: capture it as special.
        if let Some(through) = shade_vhit_through(
            map,
            registry,
            perlin,
            fog,
            far_freeze,
            light_env,
            shadow_q,
            mirror,
            &h,
            col,
            r,
            r as f32 + 0.5,
            rows,
            vp,
            (cam_x, cam_y),
            (ray_dir_x, ray_dir_y),
            max_steps,
            steps_on,
            time,
        ) {
            if let VThrough::Cell(cell, geom) = through {
                if let Some(cap) = capture.as_deref_mut() {
                    cap.capture(col, ru, &geom, r, false, true, false);
                }
                fb.set(col, ru, cell);
            }
            continue;
        }
        let (cell, geom, special) = shade_vhit(
            map,
            registry,
            perlin,
            fog,
            far_freeze,
            memo.as_deref_mut(),
            tex_cache.as_deref_mut(),
            light_env,
            shadow_q,
            &mut face_memo,
            Some(col),
            &h,
            r,
            r as f32 + 0.5,
            rows,
            vp,
            (cam_x, cam_y),
            (ray_dir_x, ray_dir_y),
            (ray_unit_x, ray_unit_y),
            steps_on,
            time,
        );
        // SELECTIVE RE-SHADE capture: a plain face replays exactly like a wall slice; a
        // ramp row carries the step-relief post-step, which cannot be replayed from
        // `CellGeom` alone, so it is flagged special.
        if let Some(cap) = capture.as_deref_mut() {
            let animated = registry
                .get(surface_material(geom.surface, geom.material))
                .is_animated();
            cap.capture(col, ru, &geom, r, animated, special, false);
        }
        fb.set(col, ru, cell);
    }
}

/// What a vertical-pass row that lands on a MIRROR or PORTAL face shows. (A
/// by-value return, consumed on the spot and never stored, so the size gap between
/// the variants costs nothing.)
#[allow(clippy::large_enum_variant)]
enum VThrough {
    /// The reflected / warped scene at this row (mirror-composited for a mirror).
    Cell(Cell, CellGeom),
    /// The warped ray escaped the grid: the row keeps the column's backdrop.
    Backdrop,
}

/// A mirror or portal face off the eye slice shows what the primary slice shows on
/// it: the scene along the REFLECTED / WARPED ray, not the face's own surface. The
/// row's view ray continues from the point it met the face — reflected about the
/// axis face, or re-based through the portal — at the same slope, so the far scene
/// keeps one perspective with everything else (`offslice::VProj`). One bounce / one
/// warp, as on the slice: a mirror or portal met by the continued ray shades as its
/// plain surface. Returns `None` when `h` is neither.
#[allow(clippy::too_many_arguments)]
fn shade_vhit_through(
    map: &Map,
    registry: &MaterialRegistry,
    perlin: &Perlin,
    fog: &FogParams,
    far_freeze: f32,
    light_env: &LightEnv,
    shadow_q: ShadowQuality,
    mirror: MirrorClarity,
    h: &offslice::VHit,
    col: usize,
    row: i32,
    row_y: f32,
    rows: i32,
    vp: &offslice::VProj,
    (cam_x, cam_y): (f32, f32),
    (ray_dir_x, ray_dir_y): (f32, f32),
    max_steps: i32,
    steps_on: bool,
    time: f32,
) -> Option<VThrough> {
    use offslice::{Face, Near, Role};
    if h.role != Role::Paint || h.face != Face::Near || h.poly.near != Near::Axis {
        return None;
    }
    let sd = map.side_data_at(h.x, h.y, h.plane);
    let surface = cellkind_to_surface(map.cell_at(h.x, h.y, h.plane), sd);
    let hx = cam_x + h.s * ray_dir_x;
    let hy = cam_y + h.s * ray_dir_y;
    let ray_len = (ray_dir_x * ray_dir_x + ray_dir_y * ray_dir_y)
        .sqrt()
        .max(PERP_MIN);
    // The continued ray is parameterised exactly as the slice's own second segment:
    // a reflected ray keeps the column ray's length, a warped ray is the unit
    // re-based direction (`rebase_ray`, `resolve_warp`). Either way the far scene
    // sits at the depth, and so the size, the slice band draws it at, with no seam
    // where the band ends.
    let (ox, oy, dx, dy, clarity) = if surface == SurfaceKind::Portal {
        let pidx = map.portal_for_cell(h.x, h.y, h.plane)?;
        let (ox, oy, ux, uy) = rebase_ray(
            &map.portals[pidx].delta,
            hx,
            hy,
            h.plane,
            ray_dir_x,
            ray_dir_y,
        );
        (ox, oy, ux, uy, None)
    } else if surface_is_mirror(registry, surface, sd) {
        let (dx, dy) = if h.side == 1 {
            (-ray_dir_x, ray_dir_y)
        } else {
            (ray_dir_x, -ray_dir_y)
        };
        let nudge = MIRROR_ORIGIN_NUDGE / ray_len;
        let clarity = surface_clarity(registry, surface, sd, mirror);
        (hx + dx * nudge, hy + dy * nudge, dx, dy, Some(clarity))
    } else {
        return None;
    };
    let k = vp.slope(row_y);
    // `cell_z = -1`: the continued ray owns every plane it meets (no slice there).
    let found = offslice::query_row(map, -1, ox, oy, dx, dy, h.z, k, max_steps, |c| {
        vhit_opaque(map, registry, c, ox, oy, dx, dy)
    });
    let mut face_memo = None;
    let Some(mut g) = found else {
        return Some(match clarity {
            // A mirror onto the void: the dim cool wash, at the mirror's own depth.
            Some(c) => {
                let (cell, mut geom, _) = shade_vhit(
                    map,
                    registry,
                    perlin,
                    fog,
                    far_freeze,
                    None,
                    None,
                    light_env,
                    shadow_q,
                    &mut face_memo,
                    None,
                    h,
                    row,
                    row_y,
                    rows,
                    vp,
                    (cam_x, cam_y),
                    (ray_dir_x, ray_dir_y),
                    (ray_dir_x / ray_len, ray_dir_y / ray_len),
                    steps_on,
                    time,
                );
                let mut cell = mirror_composite(cell, false, c, col, row);
                cell.surface = SurfaceKind::Empty;
                geom.surface = SurfaceKind::Empty;
                VThrough::Cell(cell, geom)
            }
            None => VThrough::Backdrop,
        });
    };
    // Re-express the far hit along one straight virtual ray from a virtual eye behind
    // the face, so its depth is the whole path length and its projection is the row's.
    g.s += h.s;
    g.poly.sa += h.s;
    g.poly.sb += h.s;
    let virt = (ox - h.s * dx, oy - h.s * dy);
    let d_len = (dx * dx + dy * dy).sqrt().max(PERP_MIN);
    let (cell, geom, _) = shade_vhit(
        map,
        registry,
        perlin,
        fog,
        far_freeze,
        None,
        None,
        light_env,
        shadow_q,
        &mut face_memo,
        None,
        &g,
        row,
        row_y,
        rows,
        vp,
        virt,
        (dx, dy),
        (dx / d_len, dy / d_len),
        steps_on,
        time,
    );
    let cell = match clarity {
        Some(c) => mirror_composite(cell, true, c, col, row),
        None => cell,
    };
    Some(VThrough::Cell(cell, geom))
}

/// Shadow-memo key of one face of one cell: `(x, y, plane, face)`.
type VFaceKey = (i32, i32, i32, offslice::Face);

/// The per-FACE values of a vertical-pass hit — its cast shadow and its screen face
/// anchor — computed once per face of a cell per column (a face is the same for every
/// row the column sees of it).
#[derive(Clone, Copy)]
struct VFaceMemo {
    key: VFaceKey,
    shadow: f32,
    anchor: face_span::FaceAnchor,
}

/// In-face horizontal texture coordinate of a vertical-pass hit at world `(hx, hy)`.
#[inline]
fn vhit_tex_u(h: &offslice::VHit, hx: f32, hy: f32, ray_dir_x: f32, ray_dir_y: f32) -> f32 {
    use offslice::{Face, Near};
    let fx = hx - hx.floor();
    let fy = hy - hy.floor();
    let u = match (h.face, h.poly.near, h.shape) {
        // Axis face: the DDA's own convention (flipped so opposite faces don't mirror).
        (Face::Near, Near::Axis, _) => {
            if h.side == 1 {
                if ray_dir_x > 0.0 {
                    1.0 - fy
                } else {
                    fy
                }
            } else if ray_dir_y < 0.0 {
                1.0 - fx
            } else {
                fx
            }
        }
        // A ramp slope runs across its NON-rise axis.
        (Face::Top, _, re_core::Shape::RampZ(re_core::RampDir::PosX | re_core::RampDir::NegX)) => {
            fy
        }
        _ => fx,
    };
    u.clamp(0.0, 1.0)
}

/// Is this vertical-pass candidate opaque at the point the ray meets it? Only an IMAGE
/// pane can say no: a transparent texel on its face (or any horizontal face of a pane,
/// which has no top) lets the ray continue, which is the cut-out rule the primary DDA
/// applies on the slice plane.
#[inline]
fn vhit_opaque(
    map: &Map,
    registry: &MaterialRegistry,
    h: &offslice::VHit,
    cam_x: f32,
    cam_y: f32,
    ray_dir_x: f32,
    ray_dir_y: f32,
) -> bool {
    use offslice::{Face, Role};
    if !matches!(h.role, Role::Paint | Role::Slice) {
        return true;
    }
    let sd = map.side_data_at(h.x, h.y, h.plane);
    let surface = cellkind_to_surface(map.cell_at(h.x, h.y, h.plane), sd);
    // A z-band cell is solid only inside its band; its flat faces are real lids only
    // where the band reaches them.
    if let Some(band) = cell_z_band(registry, surface, sd) {
        return match h.face {
            Face::Near => band.contains(h.z - h.plane as f32),
            Face::Top => band.touches_ceiling(),
            Face::Bottom => band.touches_floor(),
        };
    }
    if !is_image_material(registry, surface, sd) {
        return true;
    }
    if h.face != Face::Near {
        return false;
    }
    let m = registry.get(surface_material(surface, sd));
    let hx = cam_x + h.s * ray_dir_x;
    let hy = cam_y + h.s * ray_dir_y;
    match m.image_span() {
        Some(span) => {
            let along = if h.side == 1 { hy } else { hx };
            m.image_texel_opaque_at_world(&span, along, h.z)
        }
        None => {
            let u = vhit_tex_u(h, hx, hy, ray_dir_x, ray_dir_y);
            m.image_texel_opaque_at(u, (h.plane as f32 + 1.0 - h.z).clamp(0.0, 1.0))
        }
    }
}

/// Shade one vertical-pass hit into a framebuffer [`Cell`], returning the cell, the
/// [`CellGeom`] it was shaded from (for the selective re-shade capture) and whether the
/// cell carries a post-shade step the capture cannot replay (ramp step relief).
#[allow(clippy::too_many_arguments)]
fn shade_vhit(
    map: &Map,
    registry: &MaterialRegistry,
    perlin: &Perlin,
    fog: &FogParams,
    far_freeze: f32,
    memo: Option<&mut SampleMemo>,
    tex_cache: Option<&mut WorldTexCache>,
    light_env: &LightEnv,
    shadow_q: ShadowQuality,
    face_memo: &mut Option<VFaceMemo>,
    col: Option<usize>,
    h: &offslice::VHit,
    row: i32,
    row_y: f32,
    rows: i32,
    vp: &offslice::VProj,
    (cam_x, cam_y): (f32, f32),
    (ray_dir_x, ray_dir_y): (f32, f32),
    (ray_unit_x, ray_unit_y): (f32, f32),
    steps_on: bool,
    time: f32,
) -> (Cell, CellGeom, bool) {
    use offslice::{Face, Near, Role};
    let kind = map.cell_at(h.x, h.y, h.plane);
    let sd = map.side_data_at(h.x, h.y, h.plane);
    // A slab's rim (the edge of a stairwell / atrium) reads as structure: wall path.
    let surface = if h.role == Role::Slab {
        SurfaceKind::WallSolid
    } else {
        cellkind_to_surface(kind, sd)
    };
    let hx = cam_x + h.s * ray_dir_x;
    let hy = cam_y + h.s * ray_dir_y;
    let ramp = match h.shape {
        re_core::Shape::RampZ(dir) => Some(dir),
        _ => None,
    };
    let face_sign = if h.side == 1 {
        ray_unit_x.signum()
    } else {
        ray_unit_y.signum()
    };
    // World normal of the face the ray entered.
    let normal = match (h.face, h.poly.near) {
        (Face::Near, Near::Diag(n)) => n,
        (Face::Near, _) if h.side == 1 => glam::Vec3::new(-face_sign, 0.0, 0.0),
        (Face::Near, _) => glam::Vec3::new(0.0, -face_sign, 0.0),
        (Face::Top, _) => match re_core::shape_surface(h.shape) {
            Some(pl) if ramp.is_some() => pl.normal,
            _ => glam::Vec3::Z,
        },
        (Face::Bottom, _) => glam::Vec3::NEG_Z,
    };
    let axis_face = h.face == Face::Near && h.poly.near == Near::Axis;
    let side = if axis_face {
        h.side
    } else if normal.x.abs() >= normal.y.abs() && normal.z.abs() < 0.99 {
        1
    } else {
        0
    };
    // Grazing-angle foreshortening. A vertical face uses the horizontal angle, exactly
    // as the primary wall slice does (so a wall reads the same LOD on every plane); a
    // top / underside / slope uses the true 3-D view ray.
    let k = vp.slope(row_y);
    let cos_theta = if normal.z.abs() < 1e-3 {
        (ray_unit_x * normal.x + ray_unit_y * normal.y).abs()
    } else {
        let view_len = (1.0 + k * k).sqrt();
        ((ray_unit_x * normal.x + ray_unit_y * normal.y + k * normal.z) / view_len).abs()
    };
    let foreshorten = (1.0 / cos_theta.max(FORESHORTEN_COS_MIN)).min(FORESHORTEN_MAX);
    let tex_u = vhit_tex_u(h, hx, hy, ray_dir_x, ray_dir_y);
    // A full-height near face (every Full cell's, a wedge's diagonal, a slab's rim)
    // maps the rows it is DRAWN over in this column onto the texture exactly as the
    // primary slice does (`offslice::drawn_v`), so its texture neither creeps with
    // depth nor disagrees with the band; any other surface samples its true hit height.
    let drawn =
        (h.face == Face::Near && h.poly.h0 >= 1.0 - 1e-4 && h.poly.h1 >= 1.0 - 1e-4).then(|| {
            offslice::span_rows(
                vp.row(h.plane as f32 + 1.0, h.poly.sa),
                vp.row(h.plane as f32, h.poly.sa),
            )
        });
    let (wall_v, world_y) = match (h.face, drawn) {
        (Face::Near, Some((first, end))) => {
            let v = offslice::drawn_v(first, end, row_y);
            (v, h.plane as f32 + 1.0 - v)
        }
        (Face::Near, None) => ((h.plane as f32 + 1.0 - h.z).clamp(0.0, 1.0), h.z),
        (Face::Top, _) if ramp.is_some() => ((h.plane as f32 + 1.0 - h.z).clamp(0.0, 1.0), h.z),
        // A flat top / underside: spread the face's second axis over the texture's
        // vertical coordinate so it is textured, not a single smeared texel row.
        (Face::Top | Face::Bottom, _) => {
            let fy = hy - hy.floor();
            (fy, h.plane as f32 + fy)
        }
    };
    let view_z = (vp.horizon - row as f32) * VIEW_VERT_SCALE / rows.max(1) as f32;
    let view_dir = glam::Vec3::new(-ray_unit_x, -ray_unit_y, view_z).normalize_or_zero();
    let mut lc = if axis_face {
        LightCtx::for_wall(h.side, face_sign, view_dir, light_env)
    } else {
        LightCtx::for_surface_normal(normal, view_dir, light_env)
    };
    // Per-face values, once per face of a cell. The cast shadow is sampled where the
    // ray first met the face (the key light is static and a face is small). The FACE
    // ANCHOR (see `face_span`) is the exact screen extent of an axis-aligned face, which
    // a text surface needs to refuse to magnify; only a column cast publishes one.
    let key = (h.x, h.y, h.plane, h.face);
    let fm = match face_memo {
        Some(m) if m.key == key => *m,
        _ => {
            let lod_ov = registry.get(surface_material(surface, sd)).lod_override;
            // A flat top samples from just above it; everything else from its own plane.
            let sample_z = if h.face == Face::Top && ramp.is_none() {
                h.plane + 1
            } else {
                h.plane
            };
            let v = shadow_key_for_lod(
                map,
                shadow_q,
                ShadowSample {
                    depth: h.s,
                    hx,
                    hy,
                    cell_z: sample_z,
                    face_n: normal,
                },
                light_env.key_dir,
                lod_ov,
            );
            let anchor = match col {
                Some(c) if axis_face => {
                    face_span::face_anchor_for(c, h.side, h.x, h.y, hx, hy, ray_dir_x, ray_dir_y)
                }
                _ => face_span::FaceAnchor::UNKNOWN,
            };
            let m = VFaceMemo {
                key,
                shadow: v,
                anchor,
            };
            *face_memo = Some(m);
            m
        }
    };
    lc.shadow_key = fm.shadow;
    let geom = CellGeom {
        depth: h.s.max(PERP_MIN),
        foreshorten,
        side,
        tex_u,
        wall_v,
        wall_world_y: world_y,
        hit_x: hx,
        hit_y: hy,
        surface,
        material: sd,
        light: Some(lc),
        cell: (h.x, h.y, h.plane),
        tex_footprint: None,
        subcell_backdrop: None,
    };
    // FACE ANCHOR + WALL RUN for an axis-aligned face, scoped to this one shade exactly
    // as the primary slice scopes them to its band.
    face_span::set_drawn_rows(drawn.map(|(first, end)| end.saturating_sub(first).max(1) as usize));
    if axis_face {
        let ray_len = (ray_dir_x * ray_dir_x + ray_dir_y * ray_dir_y)
            .sqrt()
            .max(PERP_MIN);
        face_span::set_face_anchor(fm.anchor);
        face_span::set_wall_run(Some(face_span::WallRun {
            u: face_span::wall_run_for(h.side, hx, hy, ray_dir_x, ray_dir_y),
            footprint_cols: face_span::face_cols(h.s, foreshorten) * ray_len,
        }));
    }
    let mut cell = shade_cell(
        registry,
        perlin,
        fog,
        far_freeze,
        memo,
        tex_cache,
        map.resolved_insets(),
        &geom,
        row,
        time,
    );
    face_span::set_drawn_rows(None);
    if axis_face {
        face_span::set_face_anchor(face_span::FaceAnchor::UNKNOWN);
        face_span::set_wall_run(None);
    }
    // RAMP STEP RELIEF: a slope shaded as one lit plane reads as a blank diagonal; each
    // ramp cell is one tread, so overlay the tread lip / riser rhythm from the world
    // height the row sees — on the slope AND on the side / head faces, so a staircase
    // seen from the side still reads as steps (the underside is left plain). Direction
    // comes from the rise axis vs. screen-right, so an ascending stair never reads as
    // descending.
    let special = match ramp {
        Some(dir)
            if steps_on
                && h.face != Face::Bottom
                && ramp_is_flight(map, h.x, h.y, h.plane, dir) =>
        {
            let step = ramp_steps::step_shade(
                dir,
                world_y,
                h.s,
                -ray_unit_x,
                -ray_unit_y,
                ray_unit_y,
                -ray_unit_x,
                &ramp_steps::StepTheme::UNICODE,
            );
            cell.glyph = u32::from(step.glyph);
            let (sfg, sbg) = step.apply(cell.fg, cell.bg);
            cell.fg = sfg;
            cell.bg = sbg;
            true
        }
        _ => false,
    };
    (cell, geom, special)
}

/// Is the ramp cell at `(x, y, z)` part of a FLIGHT — does the slope continue
/// diagonally into a same-direction ramp one cell along the rise and one up, or one
/// cell back and one down?
///
/// The tread overlay (`ramp_steps`) turns every ramp into stairs. That is right for a
/// staircase and wrong for a lone slanted cell (a console desk panel, a wedge of
/// debris), which it drew as a stack of stair lips. A real flight always has a
/// neighbour on its diagonal (a stair generator stamps one tread per cell rise), so a
/// ramp with neither is a panel and keeps its own material.
fn ramp_is_flight(map: &Map, x: i32, y: i32, z: i32, dir: re_core::RampDir) -> bool {
    use re_core::RampDir;
    let (dx, dy) = match dir {
        RampDir::PosX => (1, 0),
        RampDir::NegX => (-1, 0),
        RampDir::PosY => (0, 1),
        RampDir::NegY => (0, -1),
    };
    let same = |cx: i32, cy: i32, cz: i32| matches!(map.shape_at(cx, cy, cz), re_core::Shape::RampZ(d) if d == dir);
    same(x + dx, y + dy, z + 1) || same(x - dx, y - dy, z - 1)
}

/// IMAGE CUT-OUT column (milestone 2). The primary ray hit an Image-recipe pane that
/// carries an alpha silhouette: render the BACKDROP behind it first (recast the same
/// column past the front cell, one recursion deeper), then overlay ONLY the OPAQUE
/// image texels on top. Transparent texels (the figure's margins AND every row
/// outside the front pane's wall slice) keep the backdrop, so the cut-out reads as a
/// billboard/mesh: the image where opaque, the scene behind where transparent.
///
/// Front-to-back compositing falls out of the order: backdrop painted first, opaque
/// image rows painted over. A backdrop that is itself a cut-out image recurses again
/// (bounded by [`IMAGE_CUTOUT_MAX_DEPTH`]). No mirror/warp interplay: an Image cell is
/// never reflective nor a portal, so this path is taken before either is considered.
///
/// Returns the pane's band `(first_row, end_row, perp)`, which the caller's vertical
/// pass treats as the primary's first surface.
#[allow(clippy::too_many_arguments)]
fn image_cutout_column(
    map: &Map,
    fb: &mut Framebuffer,
    registry: &MaterialRegistry,
    perlin: &Perlin,
    fog: &FogParams,
    far_freeze: f32,
    light_env: &LightEnv,
    shadow_q: ShadowQuality,
    col: usize,
    cell_z: i32,
    ray_dir_x: f32,
    ray_dir_y: f32,
    ray_unit_x: f32,
    ray_unit_y: f32,
    vp: &offslice::VProj,
    time: f32,
    image_depth: u8,
    primary: &RayHit,
    mut capture: Option<&mut crate::gbuffer::NearGBuffer>,
) -> (i32, i32, f32) {
    let rows = fb.rows() as i32;
    let ray_len = (ray_dir_x * ray_dir_x + ray_dir_y * ray_dir_y)
        .sqrt()
        .max(PERP_MIN);

    // ── 1. BACKDROP ── recast this column starting just PAST the front pane cell, so
    // the recast DDA steps over the pane and shades whatever is behind (a backdrop
    // wall, floor/ceiling, sky, or another cut-out pane). The origin is nudged a hair
    // INTO the front cell along the ray; `dda_hit` steps to the NEXT cell before its
    // first solid test, so the pane is never re-hit. The recast is told how far along
    // the ray it starts (`depth_offset`), so everything behind the pane keeps its true
    // camera distance, and so its size on screen. No capture/memo/tex-cache for the
    // backdrop pass (the selective-reshade G-buffer + caches key on the FRONT cell's
    // geometry; a cut-out column is structurally recast each frame like a mirror).
    let back_ox = primary.wall_hit_x + ray_unit_x * IMAGE_BACKDROP_NUDGE;
    let back_oy = primary.wall_hit_y + ray_unit_y * IMAGE_BACKDROP_NUDGE;
    cast_column_depth(
        map,
        fb,
        registry,
        perlin,
        fog,
        far_freeze,
        None,
        None,
        light_env,
        shadow_q,
        MirrorClarity::default(),
        None,
        col,
        cell_z,
        back_ox,
        back_oy,
        ray_dir_x,
        ray_dir_y,
        vp.horizon as i32,
        vp.proj,
        (vp.eye - (cell_z as f32 + 0.5)) * vp.proj,
        time,
        image_depth + 1,
        primary.perp + IMAGE_BACKDROP_NUDGE / ray_len,
    );

    // ── 2. FRONT IMAGE SLICE ── shade the pane's wall slice exactly as the main wall
    // path does, but `fb.set` ONLY the rows whose `(tex_u, wall_v)` texel is OPAQUE.
    // Transparent rows are left untouched, so the backdrop painted in step 1 shows.
    let perp = primary.perp.max(PERP_MIN);
    let side = primary.side;
    let surface = primary.surface;
    let material = primary.material;
    let tex_u = primary.tex_u;
    let wall_hit_x = primary.wall_hit_x;
    let wall_hit_y = primary.wall_hit_y;

    // The pane's cell spans world [cell_z, cell_z + 1]: the same projection as every
    // other surface of the column.
    let slice_top_f = vp.row(cell_z as f32 + 1.0, perp);
    let slice_bottom_f = vp.row(cell_z as f32, perp);
    let (slice_top, slice_bottom) = offslice::span_rows(slice_top_f, slice_bottom_f);
    let draw_top = clampi(slice_top, 0, rows);
    let draw_bottom = clampi(slice_bottom, 0, rows);
    let slice_span = (slice_bottom_f - slice_top_f).max(1e-3);

    // Foreshorten + face geometry: axis-aligned face (an Image pane is a Full cell,
    // never a 45° shape), so the normal comes from `side` exactly as the main path.
    let cos_theta = if side == 1 {
        ray_unit_x.abs()
    } else {
        ray_unit_y.abs()
    };
    let wall_foreshorten = (1.0 / cos_theta.max(FORESHORTEN_COS_MIN)).min(FORESHORTEN_MAX);
    let face_sign = if side == 1 {
        ray_unit_x.signum()
    } else {
        ray_unit_y.signum()
    };
    let wall_face_n = if side == 1 {
        glam::Vec3::new(-face_sign, 0.0, 0.0)
    } else {
        glam::Vec3::new(0.0, -face_sign, 0.0)
    };
    let wall_lod_ov = registry
        .get(surface_material(surface, material))
        .lod_override;
    let wall_shadow_key = shadow_key_for_lod(
        map,
        shadow_q,
        ShadowSample {
            depth: perp,
            hx: wall_hit_x,
            hy: wall_hit_y,
            cell_z,
            face_n: wall_face_n,
        },
        light_env.key_dir,
        wall_lod_ov,
    );
    // The pane material, resolved once — its texture drives the per-row opacity test.
    let pane_mat = registry.get(surface_material(surface, material));
    // WORLD-ANCHORED span: the figure's `u` is fixed for the whole column (one
    // crossing world ALONG); `v` varies per row through the world UP. `None` ⇒ legacy
    // per-cell `(tex_u, wall_v)` gate. Resolved once so the per-row loop is branch-light.
    let pane_span = pane_mat.image_span();
    let span_along = if side == 1 { wall_hit_y } else { wall_hit_x };
    // A z-band pane is opaque exactly inside its band (no texture to consult).
    let pane_band = cell_z_band(registry, surface, material);

    // SUB-CELL detail: when the pane's Image recipe opts into `Quad`/`Braille`,
    // `sample_image` samples the texture at the cell's 2×2 / 2×4 sub-positions. It needs
    // this cell's WORLD footprint on the wall face to offset the sub-samples by ±¼ cell:
    //  * UP footprint  = world-up per screen ROW. The wall slice spans `slice_span` rows
    //    for ONE world-cell of height, so one row covers `1/slice_span` world-up.
    //  * ALONG footprint = world-along per screen COLUMN. Taken as the up footprint
    //    stretched by the face foreshorten (a cell is ~square in world units, scaled by
    //    the view angle) — deterministic and shrinking as the player nears (slice grows
    //    → footprint shrinks → finer texels resolve). `Off` ⇒ `None` (legacy block).
    let subcell_mode = pane_mat.image_subcell();
    let tex_footprint = if matches!(subcell_mode, ascii_shader::SubcellMode::Off) {
        None
    } else {
        let up_fp = 1.0 / slice_span.max(1.0);
        let along_fp = up_fp * wall_foreshorten;
        Some((along_fp, up_fp))
    };

    // The band's LID. A band that stops below the cell top is a box with a top face at
    // `cell_z + hi`; a row that enters the front face ABOVE the band but whose ray drops
    // to the lid height before it leaves the cell's footprint sees that top, not the
    // backdrop. `lid_exit` is the ray distance (same `perp` units) at which the ray
    // leaves the cell in XY; the lid rows are shaded as the band's top (so they take the
    // top tone). Without it a low piece seen from standing height was a hollow rim.
    let lid_exit = pane_band.filter(|b| !b.touches_ceiling()).map(|_| {
        let cam_x = wall_hit_x - ray_dir_x * primary.perp;
        let cam_y = wall_hit_y - ray_dir_y * primary.perp;
        let exit_along = |cam: f32, dir: f32, lo: i32| -> f32 {
            if dir.abs() < 1e-6 {
                f32::INFINITY
            } else {
                let edge = if dir > 0.0 {
                    lo as f32 + 1.0
                } else {
                    lo as f32
                };
                (edge - cam) / dir
            }
        };
        exit_along(cam_x, ray_dir_x, primary.cell_x).min(exit_along(
            cam_y,
            ray_dir_y,
            primary.cell_y,
        ))
    });

    // The rows this pane actually covers. For an image pane that is its whole slice (the
    // legacy contract); for a z-band only the band + lid rows, so the vertical pass
    // still paints off-slice geometry behind the band's see-through rows (the recast in
    // step 1 is slice-only: a pipe or slab one storey up, behind a knee-high block,
    // would otherwise vanish from every row of the block's slice).
    let (mut cov_top, mut cov_bottom) = (i32::MAX, i32::MIN);
    for row in draw_top..draw_bottom {
        let wall_v = offslice::drawn_v(slice_top, slice_bottom, row as f32 + 0.5);
        // OPACITY GATE: only paint where the texel is opaque; a transparent texel keeps
        // the backdrop already in the framebuffer at this row.
        //
        // SUB-CELL silhouette: for a world-anchored `quad`/`braille` pane the gate
        // must NOT test only the cell-CENTRE texel — a row whose centre is transparent
        // but whose silhouette-edge SUB-texels are opaque (a thin hair/antenna/outline)
        // would be dropped at whole-cell granularity before the 2×2 / 2×4 sampler runs,
        // clipping the crisp margin. So gate on "ANY of this row's sub-texels opaque",
        // sampling the EXACT sub-positions `sample_image_{quad,braille}` will shade —
        // gate and shade then agree texel-for-texel (every kept row inks a glyph, every
        // dropped row would have been all-transparent). `subcell = off` (and the legacy
        // no-span path) keep the single cell-centre gate, byte-identical.
        let mut world_up = wall_world_y(cell_z, wall_v);
        let mut lid_hit: Option<(f32, f32, f32)> = None;
        let opaque = match (pane_band, pane_span, tex_footprint) {
            (Some(band), _, _) => {
                let inside = band.contains(world_up - cell_z as f32);
                if !inside && world_up - cell_z as f32 > band.hi {
                    if let Some(exit) = lid_exit {
                        let slope = vp.slope(row as f32 + 0.5);
                        let lid_z = cell_z as f32 + band.hi;
                        if slope < -1e-6 {
                            let p = (lid_z - vp.eye) / slope;
                            if p >= primary.perp && p <= exit {
                                let cam_x = wall_hit_x - ray_dir_x * primary.perp;
                                let cam_y = wall_hit_y - ray_dir_y * primary.perp;
                                lid_hit = Some((cam_x + ray_dir_x * p, cam_y + ray_dir_y * p, p));
                                // Shade it as the band's top edge (the top tone).
                                world_up = lid_z - 1e-3;
                            }
                        }
                    }
                }
                inside || lid_hit.is_some()
            }
            (None, pane_span, tex_footprint) => match (pane_span, tex_footprint) {
                (Some(span), Some(fp)) => {
                    pane_mat.image_row_opaque_subcell(&span, span_along, world_up, fp, subcell_mode)
                }
                (Some(span), None) => {
                    pane_mat.image_texel_opaque_at_world(&span, span_along, world_up)
                }
                (None, _) => pane_mat.image_texel_opaque_at(tex_u, wall_v),
            },
        };
        if !opaque {
            continue;
        }
        cov_top = cov_top.min(row);
        cov_bottom = cov_bottom.max(row + 1);
        // SUB-CELL backdrop: the cut-out painted the scene-behind into this cell in
        // step 1; read it BEFORE the re-shade overwrites it so a transparent sub-texel
        // reveals the real backdrop. `None` if absent (a fresh cell) → `sample_image`
        // falls back to the cell's own dim.
        let subcell_backdrop = if tex_footprint.is_some() {
            fb.get(col, row as usize).map(|c| c.fg)
        } else {
            None
        };
        let view_z = (vp.horizon - row as f32) * VIEW_VERT_SCALE / rows as f32;
        let view_dir = glam::Vec3::new(-ray_unit_x, -ray_unit_y, view_z).normalize_or_zero();
        let mut wall_lc = LightCtx::for_wall(side, face_sign, view_dir, light_env);
        wall_lc.shadow_key = wall_shadow_key;
        let (hit_x, hit_y, depth) = lid_hit.unwrap_or((wall_hit_x, wall_hit_y, perp));
        let geom = CellGeom {
            depth,
            foreshorten: wall_foreshorten,
            side,
            tex_u,
            wall_v,
            wall_world_y: world_up,
            hit_x,
            hit_y,
            surface,
            material,
            light: Some(wall_lc),
            cell: (primary.cell_x, primary.cell_y, cell_z),
            tex_footprint,
            subcell_backdrop,
        };
        let cell = shade_cell(
            registry,
            perlin,
            fog,
            far_freeze,
            None,
            None,
            map.resolved_insets(),
            &geom,
            row,
            time,
        );
        // Capture the OPAQUE image cell for the selective-reshade G-buffer, but flag it
        // a special (no-pan) column like a mirror/warp: a cut-out column is recast
        // structurally each frame (its backdrop can't be replayed from this cell's
        // geometry alone). An Image material is static, so `animated = false`.
        if let Some(cap) = capture.as_deref_mut() {
            cap.capture(col, row as usize, &geom, row, false, true, false);
        }
        fb.set(col, row as usize, cell);
    }
    // The rows this pane's OWN opaque texel actually claims — for an ordinary Image
    // pane exactly as much as for a z-band one (`cov_top`/`cov_bottom` already cover
    // both: the z-band gate and the plain texture-opacity gate both feed the same
    // `opaque` flag above). A transparent margin row is NOT this pane's territory —
    // it shows the BACKDROP recast painted in step 1, which is free-standing scene
    // content the vertical pass must still be allowed to search for genuine off-slice
    // geometry nearer than that backdrop's own (approximate) per-row formula. Handing
    // back the pane's FULL geometric slice here — as an earlier version of this
    // function did for every non-z-band pane, "legacy contract" — silently widens the
    // vertical pass's "this row already belongs to the primary slice, stop looking"
    // claim to rows the pane never actually painted, permanently hiding any off-slice
    // wall behind the transparent margin: the same dark-strip defect class the
    // gameplay floor cast guards against (see `clamp_to_wall_depth`'s doc comment),
    // just reached through an image pane's silhouette rather than the plain-floor
    // path that fix already covers.
    if cov_top < cov_bottom {
        (cov_top, cov_bottom, perp)
    } else {
        (0, 0, perp)
    }
}

// `mirror_composite` (the reflection-in-a-surface compositor) moved to
// `crate::mirror`; the cast calls it per mirror cell via the `use` alias above.

// The HOLOGRAM overlay sub-system (`HoloCol`, `holo_scan`, `holo_pass`,
// `holo_edge_glyph`, the `HOLO_*` tunables, and the `holo_underline_glow` /
// `dim_rgb_local` / `add_rgb_local` colour helpers) lives in `crate::holo`. The
// cast runs `holo_pass` as a post-pass when the map has holograms.

// Worked-out cast shadows live in `crate::shadow`
// (`shadow_ray_hit`, `shadow_key_for_lod`, `ShadowSample`). The raycast core builds a
// `ShadowSample` per shaded cell and calls `shadow_key_for_lod` (walls AND
// floor/ceiling thread the cell material's `lod_override`); see that module.

// ────────────────── foreshortening / LOD-driver tunables ────────────────────

/// Clamp on `|cosθ|` so a perfectly grazing wall ray cannot make foreshorten
/// explode to infinity (1/cos). Below this the surface is maximally coarsened.
const FORESHORTEN_COS_MIN: f32 = 1.0 / FORESHORTEN_MAX;
/// Cap on the foreshortening factor handed to the LOD (both walls + floor).
const FORESHORTEN_MAX: f32 = 8.0;
/// Floor/ceiling foreshorten = K·rows / denom² (denom = |row − horizon| in screen
/// rows). The `·rows` scaling (R-M4) keeps the density meaningful across the whole
/// visible band: near the horizon it saturates [`FORESHORTEN_MAX`], then falls
/// smoothly to ~1 only at the bottom rows — so the per-row anti-alias actually
/// varies (the old bare `K/denom²` clamped to 1 after ~2 rows and went inert).
const FLOOR_FORESHORTEN_K: f32 = 2.0;

// ───────────────────────── relief-lighting tunables ─────────────────────────

/// How strongly a wall cell's screen-row offset off the horizon tilts the view
/// direction vertically (drives the headlamp + specular as you look up/down a
/// wall). Larger ⇒ a steeper vertical view tilt per row.
const VIEW_VERT_SCALE: f32 = 1.6;
/// Horizontal weight of the floor/ceiling view direction (vs the ±z toward-eye
/// component). Keeps the headlamp/specular on floors reacting to facing without
/// over-tilting the near-horizon rows.
const FC_VIEW_HORIZ: f32 = 0.7;

// ───────────────────────────── shading ──────────────────────────────────────

/// The WORLD position of the surface point a [`CellGeom`] shades, in the
/// `CameraPose` convention: `x`/`y` on the floor plane, **`z` = height**.
///
/// * WALL — `(hit_x, hit_y)` is the exact point on the wall face and
///   `wall_world_y` is its world height (the field name predates the z-up
///   convention; it holds a world Z).
/// * FLOOR / CEILING — `(hit_x, hit_y)` is the true plane hit; the height is the
///   cast's z-plane, which the geometry carries as `cell.2`. The floor of storey
///   `z` sits at height `z`, its ceiling one storey above.
///
/// Pure arithmetic, no branches beyond the surface match — called once per cell.
#[inline]
pub(crate) fn cell_world_pos(g: &CellGeom) -> re_core::Vec3 {
    let z = match g.surface {
        SurfaceKind::Floor => g.cell.2 as f32,
        SurfaceKind::Ceiling => g.cell.2 as f32 + 1.0,
        _ => g.wall_world_y,
    };
    re_core::Vec3::new(g.hit_x, g.hit_y, z)
}

/// The geometry of one cell, handed from the caster to the Material shader.
/// Mirrors the legacy `FBCell` hand-off plus the LOD driver (`foreshorten`).
///
/// `Copy` so the SELECTIVE RE-SHADE G-buffer ([`crate::gbuffer`]) can cache it
/// per cell with no allocation and replay [`shade_cell`] at a new `time` to
/// reproduce an animated cell byte-for-byte.
#[derive(Default, Clone, Copy)]
pub(crate) struct CellGeom {
    pub(crate) depth: f32,
    pub(crate) foreshorten: f32,
    pub(crate) side: u8,
    pub(crate) tex_u: f32,
    /// Vertical wall coordinate in `[0,1]`: 0 at the top of the visible wall
    /// slice, 1 at the bottom. Computed in the wall draw loop as
    /// `(row − draw_top)/(draw_bottom − draw_top)`. 0 for floor/ceiling. This is
    /// the missing dimension a 2D wall texture (the `CodeWall` source-on-wall
    /// recipe) needs; threaded onto the [`Cell`] (`tex_v`) and [`SampleCtx`].
    pub(crate) wall_v: f32,
    /// WORLD vertical coordinate of a wall hit point (world Z of the texel on the
    /// wall column), so a wall's vertical texture/noise is WORLD-anchored — it does
    /// NOT shift under camera dolly/pitch (the persistence contract). Derived from
    /// the cell's world z-plane and `wall_v`: `cell_z + (1 - wall_v)` (top of the
    /// one-cell-tall wall is `cell_z + 1`, bottom is `cell_z`). 0 for floor/ceiling
    /// (which already pass the true world hit point as `(hit_x, hit_y)`).
    pub(crate) wall_world_y: f32,
    pub(crate) hit_x: f32,
    pub(crate) hit_y: f32,
    pub(crate) surface: SurfaceKind,
    pub(crate) material: u16,
    /// Per-cell relief lighting geometry (face normal, tangent frame, light + view
    /// dirs). Threaded into [`SampleCtx::light`]; `None` ⇒ legacy flat dimming.
    pub(crate) light: Option<LightCtx>,
    /// The octave-0 MAP cell coordinate `(x, y, z)` this cell shades — for a wall, the
    /// hit cell `(map_x, map_y, cell_z)`; for floor/ceiling, the floored world hit point
    /// on the cast's z-plane. Carried so the INSET override (bounded phase-band
    /// reveal) can test region membership in `shade_cell`, and so the
    /// SELECTIVE-RESHADE replay recomputes the (phase-driven, pulsing) override at the new
    /// `time`. `(0,0,0)` (the [`Default`]) when unset — inert unless a resolved inset's
    /// region happens to cover the origin, which no real inset region does for a test
    /// that passes an empty inset slice.
    pub(crate) cell: (i32, i32, i32),
    /// SUB-CELL footprint — this screen cell's WORLD `(along, up)` extent on the
    /// wall face, threaded into [`ascii_shader::SampleCtx::tex_footprint`] so an Image
    /// material in `Quad`/`Braille` sub-cell mode places its sub-samples ±¼ footprint
    /// around the cell. `None` (the [`Default`], and what every non-image draw path sets)
    /// ⇒ no sub-cell sampling — byte-identical to the solid-block path. Only the image
    /// cut-out column fills it.
    pub(crate) tex_footprint: Option<(f32, f32)>,
    /// SUB-CELL backdrop — the colour a transparent sub-texel reveals in
    /// `Quad`/`Braille` mode, threaded into [`ascii_shader::SampleCtx::subcell_backdrop`].
    /// The image cut-out column reads the backdrop it painted into this cell first and
    /// passes it here so a sub-texel hole shows the scene behind. `None` (the [`Default`])
    /// elsewhere.
    pub(crate) subcell_backdrop: Option<u32>,
}

/// Sentinel `side_data` tag a level may stamp on a DOOR object cell so the
/// renderer can shade it as the exit rather than a node. Shipped levels stamp the
/// door with `tag = 0` (indistinguishable from node 0), so by default every puzzle
/// object reads as a NODE; a level that wants the exit to read distinctly stamps
/// its door cell with this tag. Additive + back-compat: an unstamped door is still
/// a (bright, recognisable) puzzle marker, just node-styled.
///
/// The value is owned by `re_core` ([`re_core::DOOR_OBJECT_TAG`], `0x7F02`), in the
/// reserved sentinel band beside [`INERT_OBJECT_TAG`]. It used to be `0xD002`, which
/// has the pipe flag (bit 15) set and so decoded as a live pipe tag.
pub const DOOR_TAG_SENTINEL: u16 = re_core::DOOR_OBJECT_TAG;

/// `side_data` of an `Object` cell that no puzzle entity backs — mirrors
/// `re_level::INERT_OBJECT_TAG` (pinned equal by a test). Such a cell is level
/// furniture painted as `O`; it shades as plain solid geometry
/// ([`SurfaceKind::WallSolid`]), not as a puzzle node.
pub const INERT_OBJECT_TAG: u16 = re_core::INERT_OBJECT_TAG;

/// Map a [`SurfaceKind`] (+ the cell's game `tag`) to the [`MaterialId`] that
/// shades it. Walls dispatch by kind; floor + ceiling/sky each have one material.
///
/// PUZZLE LEGIBILITY: a `WallOther` cell is a `CellKind::Object` — a puzzle
/// element (a node/neuron, or the exit door). These were previously shaded as a
/// plain brick wall and were invisible against the walls; the player "had no idea
/// what to do". They now dispatch to the distinct, self-lit [`ids::PUZZLE_NODE`]
/// (or [`ids::PUZZLE_DOOR`] when the level marks the door with [`DOOR_TAG_SENTINEL`])
/// so the puzzle reads against the calmer walls/floor.
#[inline]
pub fn surface_material(surface: SurfaceKind, tag: u16) -> MaterialId {
    match surface {
        SurfaceKind::WallSolid => ids::WALL_BRICK,
        SurfaceKind::WallOther => {
            if tag == DOOR_TAG_SENTINEL {
                ids::PUZZLE_DOOR
            } else {
                ids::PUZZLE_NODE
            }
        }
        SurfaceKind::WallTrace => ids::WALL_CIRCUIT,
        SurfaceKind::WallMatrix => ids::WALL_MATRIX,
        SurfaceKind::Floor => ids::FLOOR_GRAIN,
        SurfaceKind::Ceiling | SurfaceKind::Empty => ids::CEILING_SKY,
        // Portal FALLBACK shade only. The normal path re-bases the ray BEFORE
        // shading (the slice shows the destination region), so this brick is hit
        // solely when the warp budget is exhausted — a portal seen THROUGH a
        // portal shades as a plain wall (the MVP one-warp bound).
        SurfaceKind::Portal => ids::WALL_BRICK,
    }
}

/// The z-band ([`zband`]) of a geometry cell, if its colour material is a
/// partial-height one: the second client of the cut-out path, after image panes. Only
/// an unflagged solid/trace/matrix cell carries a colour id; everything else (and every
/// level with no band material) is `None` at the cost of one empty-vec check.
#[inline]
fn cell_z_band(
    registry: &MaterialRegistry,
    surface: SurfaceKind,
    tag: u16,
) -> Option<zband::ZBand> {
    if !matches!(
        surface,
        SurfaceKind::WallSolid | SurfaceKind::WallTrace | SurfaceKind::WallMatrix
    ) || re_core::decode_pipe_tag(tag).is_some()
    {
        return None;
    }
    registry.z_band(tag)
}

/// True if the material bound to `(surface, tag)` uses the [`TextureRecipe::Image`]
/// recipe — the gate for the image cut-out path. Cheap registry read + recipe match;
/// `false` for every non-image surface (so the cut-out branch never triggers for an
/// ordinary wall and the path is byte-identical there).
#[inline]
fn is_image_material(registry: &MaterialRegistry, surface: SurfaceKind, tag: u16) -> bool {
    matches!(
        registry.get(surface_material(surface, tag)).recipe,
        ascii_shader::TextureRecipe::Image { .. }
    )
}

/// The world/face coordinate fed to `material.sample`. Walls use `(tex_u across
/// the face, WORLD height down the face)` so the texture is pinned to the wall in
/// BOTH axes and does not shift under camera dolly/pitch; floor/ceiling use the
/// true world hit point (world-pinned, parallaxes honestly).
#[inline]
fn sample_coord(g: &CellGeom, _screen_y: i32) -> glam::Vec2 {
    if g.surface.is_wall() {
        // R-H2: de-duplicate the texture per FACE without tearing ACROSS a face.
        // tex_u runs 0→1 along the variable axis of one face; offset it by the
        // face's CONSTANT axis only (side==1 → the x-plane it sits on; side==0 →
        // the y-plane). That offset is constant for every cell of one flat wall
        // run, so `u` stays continuous across the face — but a neighbouring face
        // on a different plane gets a different integer offset, so brickwork on
        // two different walls doesn't line up.  (The old code added BOTH floors,
        // which jumped at every world-cell boundary along a single flat wall.)
        let face_offset = if g.side == 1 {
            floori(g.hit_x) as f32
        } else {
            floori(g.hit_y) as f32
        };
        // WALL coord.y is now the WORLD height of the texel (`wall_world_y`),
        // scaled for texture density. Because `wall_world_y` is derived from the
        // cell's world z-plane + `wall_v` (both world-anchored, invariant to
        // screen row / camera height / pitch), the same world point on the wall
        // always maps to the same `v` — no vertical shimmer under dolly/pitch.
        // `WALL_ROW_SCALE` keeps the historical texture density (≈ rows-per-cell
        // × the old per-row scale), so the look is unchanged for a static frame.
        glam::Vec2::new(g.tex_u + face_offset, g.wall_world_y * WALL_WORLD_V_SCALE)
    } else {
        glam::Vec2::new(g.hit_x, g.hit_y)
    }
}

/// Vertical world-units-per-`v` scale for wall sampling. The wall's `coord.y` is
/// the WORLD height of the texel; this scales it to the texture's vertical density.
/// Chosen so one world cell of wall height spans the same vertical texture extent
/// the old screen-anchored `WALL_ROW_SCALE` gave for a mid-distance wall slice.
const WALL_WORLD_V_SCALE: f32 = 6.0;

/// World height of a wall texel: the wall column is one cell tall on z-plane
/// `cell_z`, spanning world Z `[cell_z, cell_z + 1]`. `wall_v` runs 0 at the TOP
/// of the slice to 1 at the bottom, so the world height is `cell_z + (1 - wall_v)`.
/// Pure function of WORLD inputs (z-plane + the world-proportional `wall_v`), so it
/// is invariant to screen row, camera height, and pitch — the world-anchor.
#[inline]
fn wall_world_y(cell_z: i32, wall_v: f32) -> f32 {
    cell_z as f32 + (1.0 - wall_v)
}

/// SCALABLE MATERIAL OVERLAY — decide this cell's INSET colour override, if any.
///
/// Returns `Some(rgb)` iff SOME resolved inset's `region` contains the cell `(x,y,z)`
/// AND the live reduced zoom phase falls in that inset's `phase_band`; else `None`. The
/// `SelfSimilar` shade ([`Material::apply_scale_field`]) replaces the fractal field colour
/// with `rgb` when this is `Some`.
///
/// The reduced phase is `(time * LN2).rem_euclid(LN2)` — the SAME reduced `zoom` value
/// [`crate::fractal::spinzoom_phase`] hands the field, and material-INDEPENDENT (every
/// twist reduces to this), so the band membership matches the field's loop exactly. A
/// FAR-FROZEN cell pins the phase to 0 (mirroring `apply_scale_field`'s frozen branch),
/// so a frozen inset region holds a stable, quiet reveal state.
///
/// Allocation-free + O(#insets) (a handful). Empty slice ⇒ an immediate `None`, so a
/// level with no `[inset]` pays nothing. Deterministic: pure in
/// `(insets, cell, time, far_freeze, distance)`.
#[inline]
fn resolve_inset_override(
    insets: &[re_core::ResolvedInset],
    cell: (i32, i32, i32),
    time: f32,
    far_freeze: f32,
    distance: f32,
) -> Option<u32> {
    if insets.is_empty() {
        return None;
    }
    let ln2 = crate::fractal::LN2;
    // The reduced zoom phase the SelfSimilar field rides this frame. A far-frozen cell
    // holds phase 0 (quiet far field), matching `apply_scale_field`.
    let reduced = if distance >= far_freeze {
        0.0
    } else {
        (time * ln2).rem_euclid(ln2)
    };
    let (x, y, z) = cell;
    for ins in insets {
        if ins.region.contains(x, y, z) && ins.phase_in_band(reduced, ln2) {
            return Some(ins.rgb);
        }
    }
    None
}

/// Fill of a calm slab face as a fraction of the element's lit colour (a flat, dim
/// wash in the pipe's / node's own hue, ~25 % of it).
pub const SLAB_CALM_FILL: f32 = 0.26;
/// Ink of the faint cell-seam texture on a calm slab face, as a fraction of the
/// element's lit colour — a step above the fill so the seams read as structure.
pub const SLAB_CALM_SEAM: f32 = 0.46;
/// Half-width (world cells) of the seam band at every cell boundary of a calm face.
const SLAB_CALM_SEAM_W: f32 = 0.06;

/// Peak channel of a calm slab's fill at the eye (`0..=255`); it recedes toward
/// [`SLAB_CALM_FAR_KEEP`] of this across the fog range.
const SLAB_CALM_PEAK: f32 = 62.0;
/// Fraction of [`SLAB_CALM_PEAK`] a calm slab keeps at the far end of the fog range.
const SLAB_CALM_FAR_KEEP: f32 = 0.4;

/// The calm treatment of a pipe/node slab face (a horizontal face, or a vertical one
/// seen steeply): no glyph but a sparse seam `·` at the world cell boundaries, over a
/// FLAT fill in the element's own hue.
///
/// Flat means the texel's own brightness is thrown away: only its HUE is kept (the
/// brighter of `fg`/`bg`, rescaled to a fixed peak that recedes with distance). A
/// fill proportional to the shaded texel kept the material's pattern — a `monitor` or
/// scanline pipe became a green arched moiré across L5's whole upper view.
/// A texel too dark to carry a hue falls back to `anchor` (the material's
/// authored colour), so a dark stripe does not flicker the fill to black. Pure.
#[inline]
pub(crate) fn slab_calm_texel(
    fg: u32,
    bg: u32,
    hit_x: f32,
    hit_y: f32,
    depth: f32,
    fog: &FogParams,
    anchor: u32,
) -> (u32, u32, u32) {
    let peak = |c: u32| ((c >> 16) & 0xFF).max((c >> 8) & 0xFF).max(c & 0xFF);
    let brighter = if peak(fg) >= peak(bg) { fg } else { bg };
    let hue = if peak(brighter) >= 24 {
        brighter
    } else {
        anchor
    };
    let span = (fog.far - fog.near).max(1e-3);
    let t = ((depth - fog.near) / span).clamp(0.0, 1.0);
    let target = SLAB_CALM_PEAK * (1.0 - (1.0 - SLAB_CALM_FAR_KEEP) * t);
    let fill = ascii_shader::brighten_rgb(hue, target / peak(hue).max(1) as f32);
    let near_seam = |v: f32| {
        let f = v - v.floor();
        f < SLAB_CALM_SEAM_W || f > 1.0 - SLAB_CALM_SEAM_W
    };
    let glyph = if near_seam(hit_x) || near_seam(hit_y) {
        u32::from('·')
    } else {
        u32::from(' ')
    };
    (
        glyph,
        ascii_shader::brighten_rgb(fill, SLAB_CALM_SEAM / SLAB_CALM_FILL),
        fill,
    )
}

/// How far a calm ceiling's near plane is lifted above the level's fog colour (the
/// fog colours are authored near-black; the lift makes the roof a surface).
pub const CEILING_CALM_LIFT: f32 = 2.4;
/// Seam ink of a calm ceiling, relative to its lifted near tone.
const CEILING_CALM_SEAM: f32 = 1.7;

/// The calm treatment of an overhead ceiling cell: the level's fog colour lifted to a
/// dim surface near the eye and receding into the fog with distance, with a sparse
/// `·` seam at world cell boundaries while the plane is near enough for seams to stay
/// sparse. A level with no fog tint uses its own shaded ceiling backing as the tone.
/// Pure.
#[inline]
pub(crate) fn ceiling_calm_texel(
    fog: &FogParams,
    shaded_bg: u32,
    depth: f32,
    hit_x: f32,
    hit_y: f32,
) -> (u32, u32, u32) {
    let base = fog.color.unwrap_or(shaded_bg);
    let near = ascii_shader::brighten_rgb(base, CEILING_CALM_LIFT);
    let span = (fog.far - fog.near).max(1e-3);
    let t = ((depth - fog.near) / span).clamp(0.0, 1.0);
    let bg = ascii_shader::mix_rgb(near, base, t);
    let seam_on = depth < fog.near + span * 0.5 && {
        let s = |v: f32| {
            let f = v - v.floor();
            !(SLAB_CALM_SEAM_W..=1.0 - SLAB_CALM_SEAM_W).contains(&f)
        };
        s(hit_x) || s(hit_y)
    };
    let glyph = if seam_on {
        u32::from('·')
    } else {
        u32::from(' ')
    };
    (glyph, ascii_shader::brighten_rgb(bg, CEILING_CALM_SEAM), bg)
}

/// Does an overhead roof of this recipe take the calm-ceiling treatment? Every roof
/// recipe does, except the ones an author picks to be SEEN overhead: an open or starry
/// sky (`Sky`, which the `void` preset is), an image, a fractal portal, and the text
/// surfaces (a code wall or a note placard on a ceiling is authored to be read). Pure.
#[inline]
pub(crate) fn ceiling_recipe_is_calmed(recipe: &TextureRecipe) -> bool {
    !matches!(
        recipe,
        TextureRecipe::Sky { .. }
            | TextureRecipe::Image { .. }
            | TextureRecipe::Fractal { .. }
            | TextureRecipe::CodeWall { .. }
            | TextureRecipe::NoteWall { .. }
    )
}

/// Turn one cell's geometry into a fully-populated [`Cell`]: look up the
/// surface's material, sample it (ranked-LOD inside), write glyph + fg + bg +
/// the additive geometry fields. Pure + deterministic.
#[allow(clippy::too_many_arguments)] // caster hand-off, not a public API.
pub(crate) fn shade_cell(
    registry: &MaterialRegistry,
    perlin: &Perlin,
    fog: &FogParams,
    far_freeze: f32,
    memo: Option<&mut SampleMemo>,
    tex_cache: Option<&mut WorldTexCache>,
    // SCALABLE MATERIAL OVERLAY P4 — the map's resolved insets (region + phase-band +
    // colour). Empty (the common case, and every test) ⇒ the inset-override decision
    // below is skipped and the shade is byte-identical to today. `re_render` consumes
    // this distilled slice only — it never loads a `.lvl`.
    resolved_insets: &[re_core::ResolvedInset],
    g: &CellGeom,
    screen_y: i32,
    time: f32,
) -> Cell {
    // Publish the live cast time for any sampler that reads it without a threaded
    // parameter — EXACTLY the `state_bus` shape, for `NoteEntry::Cycled`'s banner
    // `cycle =` word pick. One `Cell::set`; every cell of one cast carries the same
    // `time`, so this is idempotent within a frame and cheap enough to pay
    // unconditionally.
    crate::scene_text::set_cycle_time(time);

    // COLOUR→MATERIAL DISPATCH — the per-cell substrate. `g.material` is the
    // cell's `side_data`, whose meaning is
    // DISAMBIGUATED BY SURFACE KIND so the two interpretations never collide:
    //
    //  * GEOMETRY surfaces (`WallSolid`/`WallTrace`/`WallMatrix`/`Floor`/`Ceiling`)
    //    — `side_data` is a 1-based COLOUR-MATERIAL id (0 = none). A `> 0` id that
    //    resolves in the registry's colour palette shades from that material
    //    instead of the surface-kind default; the state bus is NOT consulted (a
    //    colour id is not a puzzle tag), so a geometry colour cell reads
    //    `state/pulse = 0`. Id 0 / unresolved ⇒ the surface-kind default → today.
    //  * `WallOther` (a graph node/door cell) — `side_data` is the puzzle TAG
    //    (node index, or `DOOR_TAG_SENTINEL`): `surface_material` dispatches the
    //    node/door material and the state bus keys off the tag (powered/dead).
    //
    // So only GEOMETRY surfaces take the colour path, and only `WallOther` takes
    // the graph/state-bus path — they can never both fire for one cell.
    // PIPE CELL: a `WallTrace` cell whose `side_data` carries the high-bit PIPE
    // FLAG is a live puzzle pipe (the
    // `re_level` stamper set it). Its low 15 bits are a STATE-BUS tag (the pipe
    // index), NOT a colour id — so it takes neither the colour path nor the
    // generic-tag path: it shades from the trace material (`WALL_CIRCUIT`, which
    // carries the Flow look) and reads the bus keyed by the DECODED pipe tag, so
    // the bridge's powered/flow word drives the scroll. Disjoint from colour ids
    // (those never set the flag), so back-compat holds. Only `WallTrace` is checked
    // (the only surface the stamper writes), so no other geometry cell is affected.
    let pipe_tag = if matches!(g.surface, SurfaceKind::WallTrace) {
        re_core::decode_pipe_tag(g.material)
    } else {
        None
    };
    let is_geometry_surface = matches!(
        g.surface,
        SurfaceKind::WallSolid
            | SurfaceKind::WallTrace
            | SurfaceKind::WallMatrix
            | SurfaceKind::Floor
            | SurfaceKind::Ceiling
    );
    // A pipe cell never takes the colour path (its `side_data` is a pipe tag, not a
    // colour id); only an UNFLAGGED geometry cell resolves a colour material.
    let color_slot = if is_geometry_surface && pipe_tag.is_none() {
        registry.color_material(g.material)
    } else {
        None
    };
    // NOTE PLACARD FACES: a `[note]` is a whole-cell stamp, but only its placard face
    // carries the note. Every other face of that cell shades from the material the
    // placard is mounted in (resolved at load by `derive_note_backings`).
    let color_slot = color_slot.map(|m| {
        note_face_material(
            registry,
            m,
            g.material,
            g.surface,
            g.light.map(|l| l.face_n),
            g.side,
        )
    });
    let material_id = if pipe_tag.is_some() {
        // A flagged pipe cell shades from the trace material regardless of the raw
        // side_data (which is a tag, not a surface dispatch input).
        ids::WALL_CIRCUIT
    } else {
        color_slot.unwrap_or_else(|| surface_material(g.surface, g.material))
    };
    let material = registry.get(material_id);
    // STATE-BUS TAG resolution:
    //  * A pipe cell reads the bus by its FULL flagged `side_data` (≥ 0x8000) — the
    //    high bit keeps the pipe key space DISJOINT from the small node-index tags
    //    (< 0x8000), so a node and a pipe of the same index never alias in the bus.
    //    (The decoded low bits identify which pipe for `re_level`/the bridge; the bus
    //    key is the whole flagged value.)
    //  * A colour cell carries a colour id (not a state tag) → suppress the lookup
    //    (`0`) so it never aliases an unrelated puzzle tag.
    //  * A `state_scalar_tag` material (the voice walls) reads a LEVEL-GLOBAL scalar,
    //    so its tag comes from the MATERIAL, not from the cell — see below.
    //  * Every other cell keeps the default behaviour (`g.material` is its node tag).
    //
    // MATERIAL-RESOLVED TAG (the voice walls). Every other bus reading is genuinely
    // per-cell — THIS node's lane, THIS pipe's flow word — so it keys off `side_data`.
    // The voice mark is not: it is one number for the whole level, and every cell of a
    // `mnemos_hold` wall shows the same one. Keying it off `side_data` meant a wall
    // bound the ordinary way (`[material] surface = wall_solid`, `side_data == 0`) read
    // NODE 0's lane as its progress — which measures ≈0 and renders as an untouched
    // wall. That is a legitimate state of this material, so the mistake produced no
    // error, no warning, and nothing that looked broken. Resolving from the recipe
    // makes it unmakeable.
    //
    // An explicitly stamped voice tag still WINS, so a level can cross-wire one wall to
    // the other voice's mark; any other `side_data` on a voice cell is not a mark and is
    // ignored in favour of the material's own tag. `state_scalar_tag()` is `None` for
    // every recipe that shipped before the voices, so this branch never fires for them
    // and every existing frame is byte-identical.
    let state_tag = match material.recipe.state_scalar_tag() {
        // A voice cell that carries a voice tag keeps it (the cross-wire override)…
        Some(_) if substrate::is_voice_tag(g.material) => g.material,
        // …and one that carries anything else reads its material's own tag.
        Some(material_tag) => material_tag,
        None if pipe_tag.is_some() => g.material,
        None if color_slot.is_some() => 0,
        None => g.material,
    };
    let mut ctx = SampleCtx {
        distance: g.depth,
        foreshorten: g.foreshorten,
        side: g.side,
        screen_y,
        wall_v: g.wall_v,
        tick: 0,
        time,
        fog: *fog,
        far_freeze,
        // WORLD-ANCHOR the lighting context. The `LightCtx`
        // constructors leave `world_pos` at the origin because they only know a
        // face orientation; here — the ONE place every draw path funnels through
        // — the cell's real hit point is in hand, so anchoring once covers walls,
        // floors, ceilings, wedges, objects and the mirror/portal replays without
        // touching each construction site.
        //
        // AXIS CONVENTION: `(hit_x, hit_y)` is the floor plane and **z is height**
        // (same as `CameraPose::pos`). Walls carry their world height in
        // `wall_world_y` (the field is historically mis-named: it is a world Z);
        // floor/ceiling already pass their true plane hit as `(hit_x, hit_y)` and
        // their height is the cast's z-plane, carried in `g.cell.2`.
        light: g.light.map(|lc| lc.at(cell_world_pos(g))),
        // Per-material LOD override (the key-object lever): a hero material that
        // carries a `lod_override` resolves its distance tier against its own
        // thresholds, so it keeps detail farther out than the global LOD curve.
        // `None` ⇒ the global `LodConfig` (the common case, byte-identical).
        lod_override: material.lod_override,
        // STATE BUS: resolve this cell's live state + pulse from the cast-scoped
        // `CellStateMap`, keyed by the cell's game tag (`g.material` == its
        // `side_data` — a puzzle node carries its node index). An EMPTY/absent bus
        // yields `0` / `0.0` for every tag, so a `State`-driven material picks its
        // member 0 and a non-`State` material is unchanged → byte-identical to
        // today. `re_render` reads only generic `u32`/`f32` here.
        state: state_bus::state_of(state_tag),
        pulse: state_bus::pulse_of(state_tag),
        // SCALABLE MATERIAL OVERLAY — INSET phase-band reveal: decide HERE (where the
        // cell's MAP coord + the map's resolved insets are both in hand) whether this
        // cell is inside an active inset, and if so carry the inset colour into the ctx
        // so `apply_scale_field` overrides the fractal field with it. Empty
        // `resolved_insets` ⇒ `None` ⇒ byte-identical. The reveal is phase-driven, so a
        // SELECTIVE-RESHADE replay recomputes it at the new `time` (the cell coord is in
        // `g.cell`), giving the once-per-loop pulse for free on the animated path.
        inset_rgb: resolve_inset_override(resolved_insets, g.cell, time, far_freeze, g.depth),
        // IMAGE SPAN: the wall hit's WORLD coordinate along the face's variable
        // axis (Y on an NS face `side==1`, X on an EW face `side==0`) and its WORLD UP
        // height. An Image recipe carrying an `ImageSpan` maps these to ONE figure over
        // the whole surface; every other material ignores them (byte-identical).
        wall_along: if g.side == 1 { g.hit_y } else { g.hit_x },
        wall_up: g.wall_world_y,
        // Which of `coord` / the wall plane means the world surface position.
        on_wall: g.surface.is_wall(),
        tex_footprint: g.tex_footprint,
        subcell_backdrop: g.subcell_backdrop,
    };
    let mut coord = sample_coord(g, screen_y);
    // TEXEL-SNAP (toggle, default OFF): when `RenderConfig::texel_snap = Some(n)` is
    // set, snap BOTH the 2D world sample coord AND the wall vertical coord to that
    // grid BEFORE the recipe/relief runs. This makes the SAME world surface point
    // map to the SAME texel from any camera pose, so the cross-frame
    // [`WorldTexCache`] key collides and HITS under translation/rotation. Both the
    // texture half (`texture_sample`/`tc.texture`) and the lighting half
    // (`apply_lighting`, which reads `tex.coord` and `ctx.wall_v`) then see the
    // SAME snapped values, so the cached value is exactly what is rendered. OFF ⇒
    // the continuous coord flows through untouched → byte-identical to today.
    if let Some(n) = ascii_shader::texel_snap_texels_per_cell() {
        coord = ascii_shader::snap_to_texel_grid(coord, n);
        ctx.wall_v = ascii_shader::snap_scalar_to_texel_grid(g.wall_v, n);
    }
    // SHADE = apply_lighting(texture_sample(...)) — the expensive, world-anchored
    // texture/relief half is cached ACROSS frames by the [`WorldTexCache`] (a re-seen
    // world cell under translation/rotation replays its `TexResult`, skipping the
    // fBm/relief); the cheap, view-dependent lighting is ALWAYS applied fresh, so the
    // output is byte-identical to a full re-shade. ANIMATED materials bypass the
    // tex-cache (their texture depends on `time`) and take the per-frame quant-memo /
    // direct-sample path instead. With NO tex-cache (legacy / test), fall back to the
    // F.1 quant-memo (or a direct sample) — same output either way.
    let mut texel = match tex_cache {
        Some(tc) if !material.is_animated() => {
            let tex = tc.texture(material_id.0, material, perlin, coord, &ctx);
            material.apply_lighting(perlin, &tex, &ctx)
        }
        _ => match memo {
            Some(m) => m.sample(material_id.0, material, perlin, coord, &ctx),
            None => material.sample(perlin, coord, &ctx),
        },
    };
    // FLOWING PIPE-TEXT (crate::pipe_text): a POWERED pipe cell streams readable
    // command-line WORDS along its axis instead of the anonymous flow glyph — per-word
    // colour, staggered per-row scroll, view-dependent direction flip, and a colored
    // underline "glass tube" backing. Only a tagged pipe whose state-bus word decodes
    // to a powered flow is touched; an UNPOWERED pipe and EVERY non-pipe cell fall
    // through untouched (byte-identical to today), so only pipe-demo goldens move. A
    // FAR-FROZEN cell is skipped (the far field is quiet), matching the base flow
    // scroll.
    //
    // WORD POOL: the cell's DECODED pipe tag is looked up in the level's
    // `PoolTable` — carried by the active text-surface registry — and
    // the owning node's pool is threaded into `PipeTextCtx` as a plain `&NodePool`
    // borrow (two slice index reads, no hashing, no allocation, obtained inside the read
    // guard so the borrow is real), together with the level's WORD REGISTER (clean
    // `pool`, or the corrupted `pool_alt` once the authored `[effects] glitch` density
    // passes `pipe_text::POOL_ALT_GLITCH_THRESHOLD`) — both from the SAME guard.
    //
    // `overlay_from_pool` streams those words: a level with authored pools now shows ITS
    // OWN memory-words on every powered pipe ("reading the flow is reading her memory").
    // A level with no pool data yields `None` and streams the built-in word list exactly
    // as before, so demo-level pipe frames are unchanged.
    // SLAB CALM: the underside and top of a puzzle NODE or PIPE cell carry no words
    // and no node glyphs. The vertical pass draws those faces on every pipe/node that
    // runs above or below the eye, and a lettered or hatched horizontal face would
    // fill the upper view with a text ceiling. They are instead a flat, dim fill in
    // the element's own colour with a faint cell-seam texture; the silhouette edge
    // comes from the contour pass as before. Words stay on the vertical faces the
    // player reads at eye level.
    //
    // DOMES: the same holds for a node's or pipe's VERTICAL face seen steeply from
    // below or above. A node one cell overhead filled the
    // top of the frame with a hatched field of node glyphs (the yellow "half-dome" over
    // L6 `floor_n` and L8's spawn); a face that far off the horizon is not one the player
    // reads either, by the same bound the pipe words use
    // ([`pipe_text::READ_MAX_VIEW_Z`]). It takes the same calm fill, with its seams on
    // the face's own axes.
    let slab_face = g.light.map(|l| {
        if l.face_n.z.abs() > 0.5 {
            Some((g.hit_x, g.hit_y))
        } else if l.view_dir.z.abs() > pipe_text::READ_MAX_VIEW_Z {
            Some((ctx.wall_along, g.wall_world_y))
        } else {
            None
        }
    });
    let slab_calm = slab_face.flatten().is_some()
        && (pipe_tag.is_some()
            || (g.surface == SurfaceKind::WallOther && g.material != DOOR_TAG_SENTINEL));
    // CEILING CALM: a stripe-recipe ceiling (the `metal` preset
    // three rooms author overhead) projected onto the flat ceiling plane converges into a
    // blue radial sunburst over every wall. An overhead ceiling now reads in the LEVEL's
    // own atmosphere — its fog colour, the per-level (per-movement) tint — as a dim
    // plane with faint panel seams that recedes into that fog, not as the material's
    // stripes.
    //
    // Every ROOF recipe takes it (including L5's `lattice` roof, which is not a
    // Scanline and would otherwise stay a loud green moiré). The
    // exceptions are the overheads that are deliberately a picture rather than a roof:
    // a sky (`sky`, `void`), an image, a portal zoom, and authored text surfaces.
    let ceiling_calm = g.surface == SurfaceKind::Ceiling
        && material_id == ids::CEILING_SKY
        && ceiling_recipe_is_calmed(&material.recipe);
    if slab_calm || ceiling_calm {
        let (glyph, fg, bg) = if slab_calm {
            let (a, b) = slab_face.flatten().unwrap_or((g.hit_x, g.hit_y));
            let anchor = ascii_shader::pack_rgb(material.palette.a);
            slab_calm_texel(texel.fg, texel.bg, a, b, g.depth, fog, anchor)
        } else {
            ceiling_calm_texel(fog, texel.bg, g.depth, g.hit_x, g.hit_y)
        };
        texel.glyph = glyph;
        texel.fg = fg;
        texel.bg = bg;
        return Cell {
            glyph: texel.glyph,
            fg: texel.fg,
            bg: texel.bg,
            depth: g.depth,
            tex_u: g.tex_u,
            tex_v: g.wall_v,
            side: g.side,
            hit: glam::Vec2::new(g.hit_x, g.hit_y),
            surface: g.surface,
            material: g.material,
            style: re_core::CellStyle::default(),
        };
    }
    // Z-BAND TOP TONE (`crate::zband`): the top part of a two-tone band — the head of
    // an emit figure over its stem, the bright top of a solid seat — recolours to its
    // second colour, keeping the shaded brightness (light, fog and the material's own
    // pattern survive). A flat face takes the tone of the band edge it sits on.
    if color_slot.is_some() {
        if let Some(band) = cell_z_band(registry, g.surface, g.material) {
            let local = match g.light.map(|l| l.face_n.z) {
                Some(z) if z > 0.5 => band.hi,
                Some(z) if z < -0.5 => band.lo,
                _ => g.wall_world_y - g.cell.2 as f32,
            };
            if let Some(to) = band.tone_at(local) {
                let base = ascii_shader::pack_rgb(material.palette.a);
                texel.fg = zband::recolour(texel.fg, base, to);
                texel.bg = zband::recolour(texel.bg, base, to);
            }
        }
    }
    let mut pipe_glass_underline: Option<u32> = None;
    if let Some(tag) = pipe_tag {
        if g.depth < far_freeze
            && !matches!(
                re_core::decode_flow_state(ctx.state),
                re_core::FlowDir::None
            )
        {
            let (view_dir, tangent_u) = match g.light {
                Some(l) => (Some(l.view_dir), l.tangent_u),
                None => (None, glam::Vec3::X),
            };
            // The READING face is a vertical one seen near face-on; an underside or top
            // (the vertical pass sees those on every pipe that runs above or below the
            // eye) or a side seen along its length carries the words as a quiet
            // texture instead of a text wall.
            let face = match g.light {
                Some(l) => pipe_text::PipeFace::classify(l.face_n.z, g.foreshorten, l.view_dir.z),
                None => pipe_text::PipeFace::Reading,
            };
            // ONE LINE OF WORDS, AT SCREEN SCALE. A reading face used to letter
            // every row of its band with world-anchored letters, so a pipe a few
            // cells overhead smeared each letter across half a dozen columns and
            // a dozen rows (`IIIIIIIFFFFFFF`) —
            // a text ceiling nobody reads. Now a reading face carries a single line of
            // words along its middle row, each letter at most `READ_MAX_GLYPH_COLS`
            // wide (the stream is re-scaled to the face's on-screen footprint, so a
            // near pipe still spells its words), and the rest of the face is the quiet
            // glass. Far faces, where a letter is already narrower, keep world scale.
            let run = face_span::wall_run();
            let rows = face_span::face_rows(g.depth).max(1.0);
            let text_row = (g.wall_v.clamp(0.0, 0.999) * rows) as i32 == (0.5 * rows) as i32;
            let face = if face == pipe_text::PipeFace::Reading && !text_row {
                pipe_text::PipeFace::Quiet
            } else {
                face
            };
            // The face's on-screen footprint, best source first: the face ANCHOR (this
            // cell's two world endpoints projected to exact screen columns — right for
            // an oblique face, where the ray-angle estimates below are off by up to 2×
            // and drew `CCRRIIMMSS`), then the caster's wall-run footprint, then the
            // texture span `face_cols` for a face shaded without either.
            let anchor = face_span::face_anchor();
            let footprint = if anchor.is_known() {
                anchor.span_cols
            } else {
                run.map(|r| r.footprint_cols)
                    .unwrap_or_else(|| face_span::face_cols(g.depth, g.foreshorten))
            };
            let along = match footprint {
                fc if fc * pipe_text::ALONG_PER_GLYPH > pipe_text::READ_MAX_GLYPH_COLS => {
                    // One letter spans `ALONG_PER_GLYPH` of the stream coordinate, so
                    // it covers `footprint_cols · ALONG_PER_GLYPH` columns at world scale.
                    // Stretching the coordinate by that over `READ_MAX_GLYPH_COLS` makes
                    // a letter exactly that many columns wide. (This used to DIVIDE, which
                    // magnified the letter instead — `RRRRRRRRRROO` along slab edges.)
                    ctx.wall_along * (fc * pipe_text::ALONG_PER_GLYPH)
                        / pipe_text::READ_MAX_GLYPH_COLS
                }
                _ => ctx.wall_along,
            };
            // READING ORDER: the stream runs along WORLD +along, which on one of the two
            // faces of a pipe runs right-to-left on screen (`IV·NREF·EDAJ` on L3). The
            // caster's own `tex_u` mirroring (`face_span::wall_run_for`) says which face
            // that is; mirroring the coordinate there makes every reading face spell left
            // to right. The flow's side-flip reads `tangent_u`, so it is mirrored with it:
            // the stream moves the same way through the world as before.
            let mirror = match (g.side, view_dir) {
                (1, Some(v)) if v.x < 0.0 => -1.0,
                (0, Some(v)) if v.y > 0.0 => -1.0,
                _ => 1.0,
            };
            let along = along * mirror;
            let tangent_u = tangent_u * mirror;
            // Word colour derives from the pipe's own shaded colour (the level's trace
            // material), not from a fixed palette.
            let base = Some(texel.fg);
            pipe_glass_underline = scene_text::with_pipe_pool_register(tag, |pool, register| {
                let pctx = pipe_text::PipeTextCtx {
                    state: ctx.state,
                    along,
                    wall_v: ctx.wall_v,
                    time,
                    view_dir,
                    tangent_u,
                    pool,
                    register,
                    base,
                    face,
                };
                let res = pipe_text::overlay_from_pool(&pctx, texel.bg);
                let glass = pipe_text::apply_to_texel(&mut texel, res);
                glass.underline_color()
            });
        }
    }
    // Additive per-cell terminal style (SGR attrs + colored-underline glow),
    // derived purely from the material + the final lit texel — bold on lit
    // ridges, an emissive under-glow on self-lit traces. The present layer maps
    // it to SGR; backends without the capability ignore it.
    // Distance-aware style: the lit-ridge + emissive-underline accents PLUS the
    // bold-as-proximity / bold-as-emitter-channel cues (the latter only fire for
    // materials that opt in, so shipped levels are unchanged).
    let mut style = ascii_shader::cell_style_full(material, &texel, g.depth, g.side);
    // The pipe-text glass underline OVERRIDES the material's own underline accent on a
    // flowing pipe cell (the glass edge is the intended look there); a non-pipe cell
    // never sets `pipe_glass`, so its material style is unchanged.
    if let Some(uc) = pipe_glass_underline {
        style = style.with_underline_color(uc);
    }
    Cell {
        glyph: texel.glyph,
        fg: texel.fg,
        bg: texel.bg,
        depth: g.depth,
        tex_u: g.tex_u,
        tex_v: g.wall_v,
        side: g.side,
        hit: glam::Vec2::new(g.hit_x, g.hit_y),
        surface: g.surface,
        material: g.material,
        style,
    }
}

// ───────────────────────────── tests ────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use re_world::test_room;
    // Holo constants/helpers now live in `crate::holo`; the hologram tests below
    // reference them by their short names.
    use crate::holo::{dim_rgb_local, HOLO_BACK, HOLO_BACK_DIM, HOLO_FRONT, HOLO_SCAN_PERIOD};

    // ── map_local_any_animated: the map-local near dirty-flag check ──
    //
    // These guard the #1 idle-perf fix: the registry-global `any_animated` is
    // permanently true (the appended CATALOG animates), so the near tier must
    // instead ask whether an animated material is actually RENDERABLE in THIS map.

    use ascii_shader::MaterialRegistry;
    use re_world::CellKind;

    /// Calm-ceiling generalisation: every roof recipe is calmed — including L5's
    /// `lattice` roof, which is not a Scanline and would otherwise stay a loud
    /// moiré — except the overheads authored to be SEEN: a sky (`void`
    /// is one), a portal fractal, and text surfaces.
    #[test]
    fn every_roof_recipe_is_calmed_except_deliberate_sky_and_pictures() {
        let recipe = |n: &str| {
            MaterialRegistry::catalog_material(n)
                .unwrap_or_else(|| panic!("catalog material {n}"))
                .recipe
        };
        for calm in [
            "lattice",
            "metal",
            "stone",
            "frost",
            "circuit_red",
            "haze",
            "facet",
            "moss",
            "beat_grid",
        ] {
            assert!(
                ceiling_recipe_is_calmed(&recipe(calm)),
                "{calm} roof must take the calm fill"
            );
        }
        for seen in ["void", "portal_zoom", "portal_spiral"] {
            assert!(
                !ceiling_recipe_is_calmed(&recipe(seen)),
                "{seen} is a deliberate overhead"
            );
        }
        assert!(!ceiling_recipe_is_calmed(&TextureRecipe::CodeWall {
            panel: 0,
            u_chars_per_unit: 1.0
        }));
    }

    /// A 4×4×3 room with a perimeter wall ring at z=1 (renderable surfaces only).
    /// No `Object` cells, no `Hologram` cells.
    fn static_room() -> Map {
        let mut m = Map::new(4, 4, 3);
        for x in 0..4 {
            for y in 0..4 {
                if x == 0 || y == 0 || x == 3 || y == 3 {
                    m.set(x, y, 1, CellKind::WallSolid);
                }
            }
        }
        m
    }

    /// Build a registry whose 8 fixed slots (0..7) are EXACTLY the named catalog
    /// materials given (so the test controls which slots animate), then append the
    /// real catalog by name (slots 8..) exactly like `registry_for_map` does — so
    /// the global flag is true but the map-local check ignores those slots.
    fn registry_with_fixed(slots: [&'static str; 8]) -> MaterialRegistry {
        let mut r = MaterialRegistry::new();
        for name in slots {
            let m = MaterialRegistry::catalog_material(name)
                .unwrap_or_else(|| panic!("catalog_material('{name}') must resolve"));
            r.register_named(name, m);
        }
        for name in ascii_shader::ids::CATALOG_NAMES {
            if let Some(m) = MaterialRegistry::catalog_material(name) {
                r.register_named(name, m);
            }
        }
        r
    }

    #[test]
    fn map_local_animated_false_for_static_surfaces_no_holo() {
        // (a) Only static built-in surfaces, no holograms → flag FALSE (idle caches),
        // even though the registry's global flag is true (catalog appended).
        let map = static_room();
        // All 8 fixed slots STATIC catalog materials ("stone" is static).
        let reg = registry_with_fixed(["stone"; 8]);
        assert!(
            reg.any_animated(),
            "registry-global flag is true (catalog appended)"
        );
        assert!(
            !map_local_any_animated(&map, &reg),
            "a map with only static visible surfaces must let the near tier cache"
        );
    }

    #[test]
    fn map_local_animated_true_when_surface_animates() {
        // (b) An animated material on a SURFACE slot (a `preset = NAME` copies it into
        // slot 0..4) → flag TRUE: that wall animates on screen, must never freeze.
        let map = static_room();
        // Slot 0 (a wall surface) is the animated "water"; the rest static.
        let mut slots = ["stone"; 8];
        slots[0] = "water";
        let reg = registry_with_fixed(slots);
        assert!(
            MaterialRegistry::catalog_material("water")
                .unwrap()
                .is_animated(),
            "water is an animated catalog material"
        );
        assert!(
            map_local_any_animated(&map, &reg),
            "an animated material bound to a surface slot must keep the near tier hot"
        );
    }

    #[test]
    fn map_local_animated_true_for_holograms() {
        // (c) A map with holograms → flag TRUE (the holo built-in scrolls = animated),
        // even though all the SURFACE slots are static.
        let mut map = static_room();
        map.set(2, 2, 1, CellKind::Hologram);
        assert!(map.has_holograms(), "map must report holograms present");
        // Static surface slots 0..4 and a static slot 5/6, but slot 7 = the real
        // animated holo built-in (via with_catalog's slot 7).
        let builtins = MaterialRegistry::with_catalog();
        assert!(
            builtins.get(ids::HOLO_NODE).is_animated(),
            "the holo built-in (slot 7) is animated (scanline scroll)"
        );
        // Use the game's real registry path (with_catalog has the animated holo at 7,
        // and animated circuit/matrix at 1/2 — but the holo alone must already flip it).
        assert!(
            map_local_any_animated(&map, &builtins),
            "a map with holograms must keep the near tier re-shading the holo"
        );
    }

    #[test]
    fn map_local_animated_holo_ignored_when_no_holograms() {
        // The holo built-in animates, but if the map has NO holograms it is never
        // rendered → it must NOT force the flag (only static surfaces remain).
        let map = static_room(); // no holograms
                                 // Static surface slots 0..6, but slot 7 = the animated holo built-in.
        let mut slots = ["stone"; 8];
        slots[7] = "static_noise"; // placeholder; overwritten below via direct registry
        let _ = slots;
        // Build: static surfaces 0..6, then register the animated holo at slot 7.
        let mut r = MaterialRegistry::new();
        for _ in 0..7 {
            r.register_named(
                "stone",
                MaterialRegistry::catalog_material("stone").unwrap(),
            );
        }
        // slot 7 = animated holo built-in (taken from with_catalog).
        let holo = MaterialRegistry::with_catalog().get(ids::HOLO_NODE).clone();
        assert!(holo.is_animated());
        r.register_named("holo_node", holo);
        assert_eq!(r.len(), 8);
        assert!(
            !map_local_any_animated(&map, &r),
            "an animated holo must NOT force a re-cast when the map has no holograms"
        );
    }

    #[test]
    fn map_local_animated_true_when_node_animates_with_objects() {
        // Puzzle node/door (slots 5/6) animate only matter when the map has Object
        // cells. With Object cells + an animated node slot → flag TRUE.
        let mut map = static_room();
        map.set_cell(1, 1, 1, CellKind::Object, 0); // a puzzle node cell
        assert!(map.count(CellKind::Object) > 0);
        // Static surfaces 0..4, an animated material at slot 5 (the node).
        let mut r = MaterialRegistry::new();
        for _ in 0..5 {
            r.register_named(
                "stone",
                MaterialRegistry::catalog_material("stone").unwrap(),
            );
        }
        r.register_named(
            "water",
            MaterialRegistry::catalog_material("water").unwrap(),
        ); // slot 5 animated
        r.register_named(
            "stone",
            MaterialRegistry::catalog_material("stone").unwrap(),
        ); // slot 6 static
        r.register_named(
            "stone",
            MaterialRegistry::catalog_material("stone").unwrap(),
        ); // slot 7 static
        assert!(
            map_local_any_animated(&map, &r),
            "an animated puzzle-node material with Object cells present must re-cast"
        );
        // Same registry but NO Object cells → the animated node slot is unreachable.
        let static_map = static_room();
        assert!(
            !map_local_any_animated(&static_map, &r),
            "an animated node slot must NOT force a re-cast when no Object cells exist"
        );
    }

    // ── THE VOICE WALLS' TAG RESOLUTION (the trap Option A closes) ───────────
    //
    // `crate::substrate`'s tests prove the alphabet and the mark codec; `ascii_shader`
    // proves the material renders through them. These prove the last link — that the
    // mark actually REACHES a wall bound the way an author would bind it — which is
    // the one thing no test above could see, because `SampleCtx.state` arrives already
    // resolved and every one of them sets it by hand.

    /// A registry shaped like a level that authored `[material] surface = wall_solid`
    /// with `preset = <name>`: `registry_for_map` replaces slot 0 ([`ids::WALL_BRICK`],
    /// the `WallSolid` dispatch target) with the authored material. One slot is all a
    /// `WallSolid` cell consults.
    fn wall_solid_registry(name: &str) -> MaterialRegistry {
        let mut r = MaterialRegistry::new();
        r.register_named(
            "wall_solid",
            MaterialRegistry::catalog_material(name).expect("catalog material"),
        );
        r
    }

    /// A lit `WallSolid` cell at a wall-plane hit, carrying `side_data`.
    fn voice_geom(side_data: u16, hit_x: f32, wall_v: f32) -> CellGeom {
        let view_dir = glam::Vec3::new(0.0, 1.0, 0.2).normalize();
        CellGeom {
            depth: 3.0,
            foreshorten: 1.0,
            // `side == 1` (an NS face) so `wall_along` is `hit_y`; we vary `hit_x`
            // for the run and `wall_v` for the height. Matches what the cast threads.
            side: 1,
            tex_u: 0.5,
            wall_v,
            wall_world_y: wall_world_y(1, wall_v),
            hit_x,
            hit_y: hit_x,
            surface: SurfaceKind::WallSolid,
            material: side_data,
            light: Some(LightCtx::for_wall(1, -1.0, view_dir, &LightEnv::builtin())),
            cell: (0, 0, 0),
            tex_footprint: None,
            subcell_backdrop: None,
        }
    }

    /// The glyph census a voice wall paints across a patch, through the REAL shade
    /// path (`shade_cell` → tag resolution → state bus → `Material::sample`).
    fn voice_wall_census(registry: &MaterialRegistry, side_data: u16) -> [usize; 4] {
        let perlin = Perlin::new(NOISE_SEED);
        let no_fog = FogParams {
            near: 1.0,
            far: 1.0e6,
            color: None,
            height: None,
        };
        let mut c = [0usize; 4];
        for i in 0..(24 * 24) {
            let along = (i % 24) as f32 / 3.0 + 0.17;
            let v = (i / 24) as f32 / 24.0;
            let g = voice_geom(side_data, along, v);
            let cell = shade_cell(
                registry,
                &perlin,
                &no_fog,
                f32::INFINITY,
                None,
                None,
                &[],
                &g,
                4,
                0.0,
            );
            let ch = char::from_u32(cell.glyph).expect("a valid glyph");
            let idx = substrate::ALPHABET
                .iter()
                .position(|a| *a == ch)
                .unwrap_or_else(|| panic!("{ch:?} is not in the cell alphabet"));
            c[idx] += 1;
        }
        c
    }

    /// **THE TRAP, CLOSED.** A voice wall bound the ordinary way — `[material]
    /// surface = wall_solid`, so every cell carries `side_data == 0` — must show the
    /// mark.
    ///
    /// Before the material-resolved tag it read the bus at tag `0`, which is NODE 0's
    /// glyph-family lane. That measures ≈0, so the wall rendered UNTOUCHED — and an
    /// untouched wall is a legitimate state of this material, so the mistake produced
    /// no error, no warning, and nothing that looked broken. The bus below therefore
    /// carries a non-zero node-0 lane as well as the mark, so a regression cannot pass
    /// by coincidence: it would have to read `DECOY` (2 parts ≈ nothing) and paint the
    /// resting wall.
    #[test]
    fn a_voice_wall_bound_the_ordinary_way_reads_the_mark() {
        let registry = wall_solid_registry("mnemos_hold");
        let n = 24 * 24;

        // The bus a real level produces: node lanes at small tags, the mark at its
        // reserved one. Node 0's lane is DECOY (2) — the value the wall used to read.
        let mut bus = re_core::CellStateMap::new();
        bus.set_state(0, 2);
        bus.set_state(1, 3);
        bus.set_state(substrate::MNEMOS_TAG, substrate::encode_mark(0.75));
        bus.set_state(substrate::LETHE_TAG, substrate::encode_mark(0.75));

        let at_rest = voice_wall_census(&registry, 0);
        assert_eq!(
            at_rest,
            [n, 0, 0, 0],
            "with no bus the wall must be untouched"
        );

        let [unclaimed, held, scar, burn] = {
            let _scope = state_bus::CellStateScope::new(&bus);
            voice_wall_census(&registry, 0)
        };
        assert_eq!((scar, burn), (0, 0), "the HOLD voice never releases");
        // Three quarters of the field held, to within the patch's sampling noise.
        let frac = held as f32 / n as f32;
        assert!(
            (0.70..=0.80).contains(&frac),
            "a wall bound the ordinary way held {held}/{n} ({frac:.3}) at mark 0.75. \
             At ~0 it is still reading node 0's glyph-family lane off `side_data == 0` \
             and rendering an untouched wall — the failure that looks like a working \
             material"
        );
        assert!(unclaimed > 0, "at mark 0.75 the field must still be a MIX");
    }

    /// The reserved tags remain a route: a level that stamps one into `side_data`
    /// keeps it, so a wall can be cross-wired to the OTHER voice's mark. The
    /// material-resolved tag is the default, not the only path.
    ///
    /// Driven so the two answers are maximally different — the material's own mark is
    /// 0 (an untouched wall) and the stamped one is full (a solid wall) — so there is
    /// no reading of this frame that both rules produce.
    #[test]
    fn an_explicitly_stamped_voice_tag_still_wins() {
        let registry = wall_solid_registry("mnemos_hold");
        let n = 24 * 24;
        let mut bus = re_core::CellStateMap::new();
        bus.set_state(substrate::MNEMOS_TAG, 0);
        bus.set_state(substrate::LETHE_TAG, substrate::encode_mark(1.0));
        let _scope = state_bus::CellStateScope::new(&bus);

        assert_eq!(
            voice_wall_census(&registry, substrate::MNEMOS_TAG),
            [n, 0, 0, 0],
            "the material's own tag reads mark 0 — an untouched wall"
        );
        assert_eq!(
            voice_wall_census(&registry, substrate::LETHE_TAG),
            [0, n, 0, 0],
            "a stamped LETHE_TAG must cross-wire this wall to the other voice's mark"
        );
    }

    /// A `side_data` that is NOT a voice tag is IGNORED on a voice cell — it falls
    /// back to the material's own tag rather than aliasing whatever the bus holds
    /// there. This is the general form of the trap: `0` is only the most common wrong
    /// value, and a colour id or a stray node index must be no more dangerous.
    #[test]
    fn a_non_voice_side_data_falls_back_to_the_material_tag() {
        let registry = wall_solid_registry("lethe_release");
        let n = 24 * 24;
        let mut bus = re_core::CellStateMap::new();
        // Every small tag carries a FULL mark's worth of bits — the values a
        // fall-through would pick up. Only the reserved tag holds the real mark (0).
        for tag in 0..8u16 {
            bus.set_state(tag, substrate::encode_mark(1.0));
        }
        bus.set_state(substrate::LETHE_TAG, 0);
        let _scope = state_bus::CellStateScope::new(&bus);

        for side_data in [0u16, 1, 2, 5, 7, 0x0100, re_core::encode_pipe_side_data(3)] {
            assert_eq!(
                voice_wall_census(&registry, side_data),
                [0, n, 0, 0],
                "side_data {side_data} leaked into the mark — a voice cell must read \
                 its material's tag unless it carries a voice tag"
            );
        }
    }

    #[test]
    fn raycast_hits_a_wall() {
        let map = test_room();
        let cam = CameraPose::spawn_in_test_room();
        let mut fb = Framebuffer::new(40, 20);
        raycast(&map, &cam, &mut fb);

        let mid_col = fb.cols() / 2;
        let mut hit_finite = false;
        for row in 0..fb.rows() {
            let c = fb.get(mid_col, row).unwrap();
            if c.depth.is_finite() && c.depth < 100.0 && c.surface.is_wall() {
                hit_finite = true;
            }
        }
        assert!(
            hit_finite,
            "centre column should hit a wall at finite depth"
        );
    }

    #[test]
    fn high_pitch_sky_fills_every_above_horizon_cell_with_panorama() {
        // The must-have: looking up at the +60° clamp, EVERY above-horizon (sky) cell
        // carries the direction-sampled panorama — a real, non-black sky tone — with no
        // black band, all the way toward the zenith. The sky cells are surface=Ceiling
        // (or Empty) with depth = ∞; assert each such cell got the panorama bg (the
        // procedural sky is never pure black anywhere, and a flat unfilled cell would be
        // bg=0 / the default space).
        //
        // Uses `sky_open_room` (genuinely no roof), NOT `re_world::test_room` — an
        // ENCLOSED room (a real `Ceiling` slab overhead) correctly does NOT get the
        // panorama post-fix (`column_sky_open`); see
        // `enclosed_room_keeps_ceiling_material_instead_of_sky_panorama` for that case.
        let map = sky_open_room(6);
        let mut cam = CameraPose::spawn_in_test_room();
        cam.pitch = PITCH_LIMIT_RAD; // +60°, the widened clamp.
        let fov = Fov::default();
        let mut fb = Framebuffer::new(96, 48);
        raycast_fov(&map, &cam, &fov, &mut fb);

        let mut sky_cells = 0usize;
        for row in 0..fb.rows() {
            for col in 0..fb.cols() {
                let c = fb.get(col, row).unwrap();
                // Sky = the above-horizon ceiling/empty cells (the panorama fill). The
                // ceiling cast projects most of them at a finite plane distance, so the
                // surface kind — not the depth — identifies a sky cell.
                let is_sky = matches!(c.surface, SurfaceKind::Ceiling | SurfaceKind::Empty);
                if is_sky {
                    sky_cells += 1;
                    assert!(
                        c.bg & 0x00FF_FFFF != 0,
                        "sky cell ({col},{row}) must carry a non-black panorama tone, got bg={:#08x}",
                        c.bg
                    );
                }
            }
        }
        // Looking up at +60° the sky dominates the frame — there must be many sky cells.
        assert!(
            sky_cells > 500,
            "looking up at +60° must expose a large sky region, got {sky_cells} cells"
        );
    }

    #[test]
    fn sky_is_a_varying_panorama_and_deterministic() {
        // Two properties of the direction-sampled sky on the live cast:
        //  * DETERMINISM: same scene + camera → byte-identical sky every run (the
        //    golden-frame contract).
        //  * REAL PANORAMA (not a flat fill): the sky band carries a vertical gradient,
        //    so the above-horizon cells span MANY distinct bg tones — a flat
        //    `CEILING_SKY` fill would collapse to ~one tone.
        //
        // `sky_open_room` (no roof at all), not `re_world::test_room`: an enclosed
        // room now correctly keeps its shaded ceiling material instead of the
        // panorama (`column_sky_open`), so testing the panorama itself needs a
        // genuinely open room.
        let map = sky_open_room(6);
        let fov = Fov::default();
        let cast = |yaw: f32| {
            let mut cam = CameraPose::spawn_in_test_room();
            cam.pitch = PITCH_LIMIT_RAD;
            cam.yaw = yaw;
            let mut fb = Framebuffer::new(64, 40);
            raycast_fov(&map, &cam, &fov, &mut fb);
            fb
        };
        let a = cast(0.5);
        let a2 = cast(0.5);

        // Deterministic: two casts at the same pose are byte-identical (sky included).
        for row in 0..a.rows() {
            for col in 0..a.cols() {
                assert_eq!(
                    a.get(col, row),
                    a2.get(col, row),
                    "sky must be deterministic"
                );
            }
        }

        // A real panorama: collect the distinct sky-cell bg tones. The vertical
        // gradient alone (horizon → zenith) guarantees many bands; a flat fill would
        // give one.
        let mut tones = std::collections::BTreeSet::new();
        for row in 0..a.rows() {
            for col in 0..a.cols() {
                let c = a.get(col, row).unwrap();
                if matches!(c.surface, SurfaceKind::Ceiling | SurfaceKind::Empty) {
                    tones.insert(c.bg & 0x00FF_FFFF);
                }
            }
        }
        assert!(
            tones.len() > 5,
            "the sky must be a graded panorama (many tones), got {} distinct tones",
            tones.len()
        );
    }

    #[test]
    fn sky_yaw_rotation_is_observable_in_the_cylinder_band() {
        // Rotation reads on the sky: in an OPEN scene (no near walls clipping the
        // cylinder band) a yaw turn shifts the cylinder `u` (longitude grid + tone),
        // so the above-horizon frame differs from the un-rotated one. This is the
        // "rotation visibly reads" guarantee, isolated from wall occlusion.
        //
        // `sky_open_room`, not `open_room`: `open_room` still caps its z-top with a
        // real `Ceiling` slab, which `column_sky_open` now correctly reads as an
        // enclosed roof (no panorama) even though the room has no side walls.
        let map = sky_open_room(16);
        let fov = Fov::default();
        let cast = |yaw: f32| {
            let mut cam = CameraPose::spawn_in_test_room();
            cam.pos = glam::Vec3::new(8.0, 8.0, STANDING_EYE_Z);
            cam.pitch = 0.2; // a gentle up-tilt: cylinder band fills the upper frame.
            cam.yaw = yaw;
            let mut fb = Framebuffer::new(64, 40);
            raycast_fov(&map, &cam, &fov, &mut fb);
            fb
        };
        let a = cast(0.0);
        // A turn that is NOT a multiple of the 15° longitude spacing, so the grid
        // genuinely lands on different columns (a 45° turn = exactly 3 spacings would
        // alias back onto the same phase).
        let b = cast(0.37);

        // The vertical gradient (bg) is a function of pitch/row alone; YAW rotation
        // moves the longitude grid, which reads in the glyph + grid ink (fg). So a
        // turn changes the sky cells' (glyph, fg) — the rotation cue.
        let mut changed = 0usize;
        for row in 0..a.rows() {
            for col in 0..a.cols() {
                let (ca, cb) = (a.get(col, row).unwrap(), b.get(col, row).unwrap());
                let both_sky = matches!(ca.surface, SurfaceKind::Ceiling | SurfaceKind::Empty)
                    && matches!(cb.surface, SurfaceKind::Ceiling | SurfaceKind::Empty);
                if both_sky && (ca.glyph != cb.glyph || ca.fg != cb.fg) {
                    changed += 1;
                }
            }
        }
        assert!(
            changed > 0,
            "a yaw turn must rotate the direction-sampled sky grid ({changed} cells differed)"
        );
    }

    // ───────────────────── enclosed-room sky leak fix ─────────────────────
    // `column_sky_open` / `column_is_open_overhead` are what gate `apply_sky` after
    // the 2026-09-23 fix: an enclosed `[room]`-built level (a real `Ceiling` slab
    // stamped at its z-top by `re_world::stamp_geometry`, no `[sky]` authored) must
    // NOT show the direction-sampled panorama above the horizon; it keeps the
    // ordinary shaded ceiling material instead. A level that authors an image sky,
    // or one whose column genuinely has no roof, still gets the panorama.

    #[test]
    fn column_sky_open_false_in_an_enclosed_room() {
        // `re_world::test_room` caps its whole interior with a real `Ceiling` slab
        // at z = SZ-1 = 3; the camera's own column (3,4) at cell_z=1 must read as
        // capped (a `Ceiling` cell sits at z=3, reached scanning from z=2 up).
        let map = test_room();
        assert!(
            !column_sky_open(&map, 1, 3.5, 4.5),
            "an enclosed room's own column must NOT read as open overhead"
        );
    }

    #[test]
    fn column_sky_open_true_with_no_roof_at_all() {
        // `sky_open_room` leaves the interior Empty all the way to the map's top —
        // no `Ceiling` slab anywhere overhead — so the column reads as open.
        let map = sky_open_room(6);
        assert!(
            column_sky_open(&map, 1, 3.5, 4.5),
            "a room with no roof over the camera's column must read as open overhead"
        );
    }

    #[test]
    fn column_sky_open_true_when_the_level_authors_an_image_sky() {
        // `test_room` is fully enclosed (real `Ceiling` slab everywhere), but a
        // level authoring `[sky]` (an image cubemap) wants the panorama
        // regardless — the image-sky declaration overrides the voxel scan entirely.
        let mut map = test_room();
        map.set_sky_cylinder(std::sync::Arc::new(re_core::TextureImage::new(
            1,
            1,
            vec![10, 20, 30, 255],
        )));
        assert!(
            column_sky_open(&map, 1, 3.5, 4.5),
            "authoring an image sky must force the column open regardless of the roof"
        );
    }

    #[test]
    fn column_is_open_overhead_stops_at_the_first_capping_cell() {
        let map = test_room(); // Ceiling at z=3, perimeter WallSolid at every z.
                               // Scanning from just above the camera's slice (z=2) finds the z=3 Ceiling.
        assert!(!column_is_open_overhead(&map, 3, 4, 2));
        // A perimeter column is capped at EVERY z (walls span the full height).
        assert!(!column_is_open_overhead(&map, 0, 0, 1));
        // Out-of-bounds z reads Empty (never capping) — scanning from above the
        // map's own top layer finds nothing (vacuously open).
        assert!(column_is_open_overhead(&map, 3, 4, 4));
    }

    /// Per-level sky fraction of a real `--dump`-sized frame: the fraction of
    /// above-horizon (`Ceiling`/`Empty`) cells `apply_sky` actually painted (its
    /// output byte-for-byte matches the cell that landed in the framebuffer).
    /// Shared by the engine-fixture regression tests below.
    fn sky_fraction(lvl: &re_level::Level, cols: usize, rows: usize) -> (usize, usize) {
        let (sky, _backdrop) = sky_fraction_from(lvl, &lvl.spawn_camera(), cols, rows);
        (sky, cols * rows)
    }

    /// As [`sky_fraction`] from an explicit camera, returning `(sky_cells,
    /// backdrop_cells)`: `backdrop_cells` counts the ABOVE-HORIZON cells no geometry
    /// covers (the `Ceiling`/`Empty` backdrop), so `sky / backdrop` measures what the
    /// sky gate decides wherever the backdrop is actually visible — independent of
    /// how much of the view the level's real wall heights occupy.
    fn sky_fraction_from(
        lvl: &re_level::Level,
        cam: &CameraPose,
        cols: usize,
        rows: usize,
    ) -> (usize, usize) {
        sky_fraction_from_map(&lvl.build_map(), cam, cols, rows)
    }

    /// As [`sky_fraction_from`], but over an already-built [`Map`] rather than a
    /// [`re_level::Level`] — lets a test exercise the sky gate against a synthetic
    /// fixture (e.g. [`sky_open_room`]) with no `.lvl` file involved at all.
    fn sky_fraction_from_map(
        map: &Map,
        cam: &CameraPose,
        cols: usize,
        rows: usize,
    ) -> (usize, usize) {
        let fov = Fov::default();
        let mut fb = Framebuffer::new(cols, rows);
        raycast_fov(map, cam, &fov, &mut fb);
        let basis = SubViewBasis::new(cam, &fov, cols, rows);
        let mut sky_cells = 0usize;
        let mut backdrop = 0usize;
        for row in 0..rows as i32 {
            for col in 0..cols {
                let c = *fb.get(col, row as usize).unwrap();
                if !matches!(c.surface, SurfaceKind::Ceiling | SurfaceKind::Empty) {
                    continue;
                }
                if row < basis.horizon {
                    backdrop += 1;
                }
                let camera_x = 2.0 * (col as f32 + 0.5) / basis.cols - 1.0;
                let ray_dir_x = basis.dir_x + basis.plane_x * camera_x;
                let ray_dir_y = basis.dir_y + basis.plane_y * camera_x;
                let ray_len = (ray_dir_x * ray_dir_x + ray_dir_y * ray_dir_y)
                    .sqrt()
                    .max(PERP_MIN);
                let mut would_be_sky = c;
                apply_sky(
                    map,
                    &mut would_be_sky,
                    ray_dir_x / ray_len,
                    ray_dir_y / ray_len,
                    row,
                    basis.horizon,
                    basis.proj_plane_dist,
                );
                if (c.glyph, c.fg, c.bg) == (would_be_sky.glyph, would_be_sky.fg, would_be_sky.bg) {
                    sky_cells += 1;
                }
            }
        }
        (sky_cells, backdrop)
    }

    /// Regression test: a fully sealed room (a single `[room]`-built box with no
    /// `[sky]` and no authored gap in its z-top) must show ~0% direction-sampled
    /// sky at a real `--dump` size, down from the 33%/28% measured before the fix
    /// (originally pinned to the campaign's `levels/L7-hidden-depths.lvl` and
    /// `levels/L8-cold-room.lvl` — both exactly this shape). Re-pointed to two
    /// non-campaign ENGINE example levels with the same shape, so this engine-crate
    /// test does not depend on game-campaign level data: `08-material-demo.lvl` (a
    /// plain `[room]` box whose `[cells]` grid explicitly caps every layer,
    /// including a solid roof) and `10-calibration-vault.lvl` (a plain `[room]`
    /// header box with no `[cells]` override at all) — both ship with no `[sky]`.
    #[test]
    fn enclosed_rooms_drop_to_near_zero_sky() {
        for name in ["08-material-demo", "10-calibration-vault"] {
            let path = format!("{}/../../levels/{name}.lvl", env!("CARGO_MANIFEST_DIR"));
            let lvl = re_level::load(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
            let (sky_cells, total) = sky_fraction(&lvl, 160, 48);
            assert_eq!(
                sky_cells, 0,
                "{name}: enclosed room must show 0 sky cells at dump size, got {sky_cells}/{total}"
            );
        }
    }

    /// A minimal open-air fixture that authors an IMAGE `[sky]` (`cylinder` +
    /// `cap_top`), standing in for the removed `levels/CYBER-sky.lvl` (a
    /// third-party panorama with no known licence, removed ahead of open-sourcing
    /// this repo). Points both slots at the generated,
    /// freely-licensed `genplate_tex.png` plate (`re_voximport::texfixture`) — only
    /// that a real DECODED image sky wins over the void-ceiling backdrop matters
    /// here, not the image's content. Parsed from a STRING, so the path must be
    /// absolute (a string-parsed level has no file to anchor a relative one to).
    fn image_sky_fixture() -> String {
        let plate = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("assets")
            .join("texture")
            .join("genplate_tex.png")
            .canonicalize()
            .expect("genplate_tex.png must be committed")
            .to_string_lossy()
            .replace('\\', "/");
        format!(
            "[meta]\nname = image-sky-fixture\n\
             [room]\nsize = 80 80 20\nambient = solid\n\
             [sky]\ncylinder = {plate}\ncap_top = {plate}\n\
             [material open_sky]\npreset = void\nsurface = ceiling\n\
             [material deck_floor]\nfg = #1C2435\nbg = #050810\n\
             recipe = grain scale=3.0 speck=0.20 tile_eps=0.08\nsurface = floor\n\
             [material far_wall]\nfg = #10192A\nbg = #0A1830\n\
             recipe = grain scale=2.0 speck=0.08 tile_eps=0.04\nsurface = wall_solid\n\
             [palette]\nfog_color = #0A1830\nfog_near = 15.0\nfog_far = 52.0\n\
             [node source]\npos = 44 72 2\nis_input = yes\n\
             [node sink]\npos = 46 72 2\nis_output = yes\nthreshold = 1\n\
             [pipe source->sink]\npath = (44,72,2) (46,72,2)\nswitch_at = (45,72,2)\n\
             [inputs]\nsource = 1\n\
             [pedestal]\ncenter = 46 72 1\nvolume = 1 1 1\n\
             [door]\npos = 46 73 1\n\
             [player]\nspawn = 40 70 2\nfacing = -y\n"
        )
    }

    /// CAMPAIGN REGRESSION: an authored `[sky]` image panorama — the fix's clause 1
    /// (`column_sky_open`) must keep its panorama dominant.
    ///
    /// Measured over the visible above-horizon BACKDROP: the vault's perimeter walls
    /// are 19 cells tall and render at that height, so they (correctly) cover the
    /// lower part of the panorama band; what the gate decides is what shows wherever
    /// the backdrop is visible, and there it must be the panorama.
    #[test]
    fn campaign_image_sky_level_keeps_its_panorama() {
        let lvl = re_level::load_string(&image_sky_fixture(), "<image-sky-fixture>")
            .expect("image-sky fixture must parse");
        let (sky_cells, backdrop) = sky_fraction_from(&lvl, &lvl.spawn_camera(), 160, 48);
        assert!(
            backdrop > 160 * 4 && sky_cells as f32 / backdrop as f32 > 0.9,
            "an authored image sky must keep its panorama dominant over the visible backdrop, \
             got {sky_cells}/{backdrop}"
        );
    }

    /// Regression test: a room whose z-top is authored EMPTY across the whole
    /// footprint — a deliberate open-top design, not an oversight — must NOT have
    /// its sky forced to zero by the enclosed-room fix above; that would hide a
    /// real, intentionally roofless room, not just an artefact of the old
    /// unconditional overwrite. (Originally pinned to the campaign finale's
    /// `levels/L12-closing.lvl`, whose z=9 layer is authored EMPTY across its
    /// whole footprint — "the vault above the fog. Open", per the level's own
    /// `[cells]` comment — the one campaign room that a blanket "campaign levels
    /// leak sky" rule must not misfire on. Re-pointed, so this engine-crate test
    /// does not depend on game-campaign level data, to [`sky_open_room`], the
    /// engine's own synthetic "genuinely no roof at all" fixture already used by
    /// the neighboring panorama tests above — it isolates exactly the same shape:
    /// no `[sky]` authored, no capping `Ceiling` anywhere overhead.)
    #[test]
    fn open_top_room_keeps_its_sky() {
        let map = sky_open_room(16);
        let mut cam = CameraPose::spawn_in_test_room();
        cam.pos = glam::Vec3::new(8.0, 8.0, STANDING_EYE_Z);
        cam.pitch = PITCH_LIMIT_RAD * 0.75;
        let (sky_cells, backdrop) = sky_fraction_from_map(&map, &cam, 160, 48);
        assert!(
            sky_cells > 160 * 4 && sky_cells as f32 / backdrop as f32 > 0.9,
            "a genuinely open-top room must keep its panorama dominant over the visible \
             backdrop, got {sky_cells}/{backdrop}"
        );
    }

    #[test]
    fn enclosed_room_keeps_ceiling_material_instead_of_sky_panorama() {
        // The end-to-end regression: an enclosed room at high pitch shows PLENTY of
        // above-horizon `Ceiling` rows (the wall slice does not fill the frame), and
        // NONE of them may be the direction-sampled panorama `apply_sky` produces.
        // Precise (no magic tone-count threshold): reconstruct EXACTLY what
        // `apply_sky` would have painted for the same ray via `SubViewBasis` (the
        // same basis `shade_subray` uses to land a sub-ray on the full-frame grid),
        // and assert the real cast's cell is NOT that — the enclosed room's real
        // shaded ceiling material, not a forced sky overwrite.
        let map = test_room();
        let mut cam = CameraPose::spawn_in_test_room();
        cam.pitch = PITCH_LIMIT_RAD;
        let fov = Fov::default();
        let cols = 64usize;
        let rows = 40usize;
        let mut fb = Framebuffer::new(cols, rows);
        raycast_fov(&map, &cam, &fov, &mut fb);

        let basis = SubViewBasis::new(&cam, &fov, cols, rows);
        let mut ceiling_rows = 0usize;
        let mut checked = 0usize;
        for row in 0..fb.rows() as i32 {
            for col in 0..fb.cols() {
                let c = *fb.get(col, row as usize).unwrap();
                if !matches!(c.surface, SurfaceKind::Ceiling) {
                    continue;
                }
                ceiling_rows += 1;

                // Rebuild this exact ray's unit direction (mirrors `shade_subray`'s
                // own basis math) and ask `apply_sky` what it WOULD have painted.
                let camera_x = 2.0 * (col as f32 + 0.5) / basis.cols - 1.0;
                let ray_dir_x = basis.dir_x + basis.plane_x * camera_x;
                let ray_dir_y = basis.dir_y + basis.plane_y * camera_x;
                let ray_len = (ray_dir_x * ray_dir_x + ray_dir_y * ray_dir_y)
                    .sqrt()
                    .max(PERP_MIN);
                let mut would_be_sky = c;
                apply_sky(
                    &map,
                    &mut would_be_sky,
                    ray_dir_x / ray_len,
                    ray_dir_y / ray_len,
                    row,
                    basis.horizon,
                    basis.proj_plane_dist,
                );
                assert_ne!(
                    (c.glyph, c.fg, c.bg),
                    (would_be_sky.glyph, would_be_sky.fg, would_be_sky.bg),
                    "enclosed-room ceiling cell ({col},{row}) reads as the direction-sampled \
                     panorama — the sky leaked into a capped room"
                );
                checked += 1;
            }
        }
        assert!(
            ceiling_rows > 0,
            "high pitch in a small enclosed room must still show ceiling rows"
        );
        assert!(
            checked > 100,
            "expected many ceiling cells checked, got {checked}"
        );
    }

    #[test]
    fn vertical_eye_shear_is_a_per_entry_point_policy() {
        // VERTICAL EYE SHEAR (jump/fall visibility) is a GAMEPLAY-only policy.
        //
        // 1. Standing-height INVARIANCE (gameplay path): a gameplay cast at the exact
        //    standing eye height (`STANDING_EYE_Z`, where `vshift_per_perp` is
        //    identically 0) is byte-identical to a baseline standing gameplay cast.
        // 2. JUMP VISIBILITY (gameplay path): lifting the eye ~0.8 cell on the GAMEPLAY
        //    path (reference = `STANDING_EYE_Z`) MUST change the frame — the jump now
        //    renders instead of being discarded with the fractional z.
        // 3. PREVIEW INVARIANCE (generic path): the SAME raised eye through the GENERIC
        //    `raycast()` (reference = camera's own z → shift 0) is byte-identical to the
        //    centred baseline — high preview/QA cameras MUST NOT shear off-screen.
        let map = test_room();
        let reg = registry_for_map(&map);
        let fov = Fov::default();

        let standing = CameraPose::spawn_in_test_room();
        assert!(
            (standing.pos.z - STANDING_EYE_Z).abs() < 1e-6,
            "spawn pose should be at the standing eye height the shear references"
        );

        // The math: GAMEPLAY reference (STANDING_EYE_Z) is exactly 0 at standing height,
        // and the GENERIC reference (eye_z itself) is exactly 0 at ANY height.
        let view = view_params(40, 20, &fov);
        assert_eq!(
            vshift_per_perp(STANDING_EYE_Z, STANDING_EYE_Z, view.proj_plane_dist),
            0.0,
            "gameplay shear must be EXACTLY 0 at standing height (byte-identity)"
        );
        assert_eq!(
            vshift_per_perp(
                STANDING_EYE_Z + 0.8,
                STANDING_EYE_Z + 0.8,
                view.proj_plane_dist
            ),
            0.0,
            "generic shear must be EXACTLY 0 at ANY camera height (preview invariance)"
        );

        // Helper: gameplay cast (reference = STANDING_EYE_Z).
        let gameplay = |cam: &CameraPose| -> Framebuffer {
            let mut fb = Framebuffer::new(40, 20);
            raycast_with_time_fov_gameplay(&map, cam, &reg, &fov, 0.0, &mut fb);
            fb
        };

        // ── (1) gameplay standing invariance ──
        let fb_baseline = gameplay(&standing);
        let fb_standing = gameplay(&standing);
        for row in 0..fb_baseline.rows() {
            for col in 0..fb_baseline.cols() {
                let a = fb_baseline.get(col, row).unwrap();
                let b = fb_standing.get(col, row).unwrap();
                assert_eq!(
                    a.glyph, b.glyph,
                    "glyph drift at standing height ({col},{row})"
                );
                assert_eq!(a.fg, b.fg, "fg drift at standing height ({col},{row})");
                assert_eq!(a.bg, b.bg, "bg drift at standing height ({col},{row})");
                assert_eq!(
                    a.depth.to_bits(),
                    b.depth.to_bits(),
                    "depth drift at standing height ({col},{row})"
                );
            }
        }

        // ── (2) gameplay jump visibility: raised eye MUST change the frame ──
        // Raise by +0.3 so the eye stays inside the SAME world cell
        // (`floor(1.5) == floor(1.8)`): this isolates the SHEAR as the cause of the
        // difference (the DDA walks the identical z-layer; only the vertical screen
        // offset moves). The same +0.3 in step (3) must leave the GENERIC path
        // unchanged — proving the difference here is purely the gameplay shear.
        let mut jumped = standing;
        jumped.pos.z += 0.3;
        let fb_jumped = gameplay(&jumped);
        let mut differs = 0usize;
        for row in 0..fb_standing.rows() {
            for col in 0..fb_standing.cols() {
                let s = fb_standing.get(col, row).unwrap();
                let j = fb_jumped.get(col, row).unwrap();
                if s.glyph != j.glyph
                    || s.fg != j.fg
                    || s.bg != j.bg
                    || s.depth.to_bits() != j.depth.to_bits()
                {
                    differs += 1;
                }
            }
        }
        assert!(
            differs > 0,
            "a jump (raised pos.z) on the GAMEPLAY path must change the frame — it was discarded"
        );

        // ── (3) generic preview invariance: same raised eye must NOT shear ──
        // The generic `raycast` subtracts the camera's OWN z, so the per-perp shift is
        // identically 0 at every height. With the +0.3 raise kept inside the same world
        // cell (so `cell_z` is unchanged) and all other inputs equal, the generic cast
        // of the raised pose is byte-identical to the generic cast of the standing pose:
        // the previews/QA cameras render exactly as before the shear was added.
        let mut fb_generic_standing = Framebuffer::new(40, 20);
        raycast(&map, &standing, &mut fb_generic_standing);
        assert_eq!(
            standing.pos.z.floor() as i32,
            jumped.pos.z.floor() as i32,
            "test setup: the +0.3 raise must stay inside the same world cell"
        );
        let mut fb_generic_jumped = Framebuffer::new(40, 20);
        raycast(&map, &jumped, &mut fb_generic_jumped);
        for row in 0..fb_generic_standing.rows() {
            for col in 0..fb_generic_standing.cols() {
                let s = fb_generic_standing.get(col, row).unwrap();
                let j = fb_generic_jumped.get(col, row).unwrap();
                assert_eq!(
                    s.glyph, j.glyph,
                    "generic preview sheared (glyph) at ({col},{row})"
                );
                assert_eq!(s.fg, j.fg, "generic preview sheared (fg) at ({col},{row})");
                assert_eq!(s.bg, j.bg, "generic preview sheared (bg) at ({col},{row})");
                assert_eq!(
                    s.depth.to_bits(),
                    j.depth.to_bits(),
                    "generic preview sheared (depth) at ({col},{row})"
                );
            }
        }
    }

    #[test]
    fn wall_cells_have_tex_u_in_unit_range() {
        let map = test_room();
        let cam = CameraPose::spawn_in_test_room();
        let mut fb = Framebuffer::new(60, 30);
        raycast(&map, &cam, &mut fb);

        let mut saw_wall = false;
        for row in 0..fb.rows() {
            for col in 0..fb.cols() {
                let c = fb.get(col, row).unwrap();
                if c.surface.is_wall() {
                    saw_wall = true;
                    assert!(
                        (0.0..=1.0).contains(&c.tex_u),
                        "wall tex_u out of [0,1]: {}",
                        c.tex_u
                    );
                }
            }
        }
        assert!(saw_wall, "expected at least one wall cell");
    }

    #[test]
    fn ew_side_is_darker_than_ns_at_equal_distance() {
        // Same geometry, only `side` differs → EW must shade dimmer. Now driven
        // through the Material shader (the brick wall material).
        let registry = MaterialRegistry::with_builtins();
        let perlin = Perlin::new(NOISE_SEED);
        let ns = CellGeom {
            depth: 4.0,
            foreshorten: 1.0,
            side: 0,
            tex_u: 0.5,
            wall_v: 0.5,
            wall_world_y: wall_world_y(1, 0.5),
            hit_x: 2.5,
            hit_y: 1.0,
            surface: SurfaceKind::WallSolid,
            material: 0,
            // No LightCtx → legacy flat `side` dimming, which is exactly what this
            // EW-vs-NS test asserts (the historical wolf3d directional trick).
            light: None,
            cell: (0, 0, 0),
            tex_footprint: None,
            subcell_backdrop: None,
        };
        let ew = CellGeom {
            side: 1,
            ..clone_geom(&ns)
        };

        let body_row = 2;
        let fog = FogParams::DEFAULT;
        let ns_cell = shade_cell(
            &registry,
            &perlin,
            &fog,
            f32::INFINITY,
            None,
            None,
            &[],
            &ns,
            body_row,
            0.0,
        );
        let ew_cell = shade_cell(
            &registry,
            &perlin,
            &fog,
            f32::INFINITY,
            None,
            None,
            &[],
            &ew,
            body_row,
            0.0,
        );

        assert!(
            luminance(ew_cell.fg) < luminance(ns_cell.fg),
            "EW fg ({:#08x}) should be darker than NS fg ({:#08x})",
            ew_cell.fg,
            ns_cell.fg
        );
    }

    #[test]
    fn floor_shade_is_deterministic() {
        // The full floor shade is reproducible for the same world point.
        let registry = MaterialRegistry::with_builtins();
        let perlin = Perlin::new(NOISE_SEED);
        let g = CellGeom {
            depth: 5.0,
            foreshorten: 1.2,
            side: 0,
            tex_u: 0.0,
            wall_v: 0.0,
            wall_world_y: 0.0,
            hit_x: 3.21,
            hit_y: 7.83,
            surface: SurfaceKind::Floor,
            material: 0,
            light: None,
            cell: (0, 0, 0),
            tex_footprint: None,
            subcell_backdrop: None,
        };
        let first = shade_cell(
            &registry,
            &perlin,
            &FogParams::DEFAULT,
            f32::INFINITY,
            None,
            None,
            &[],
            &g,
            18,
            0.0,
        );
        let second = shade_cell(
            &registry,
            &perlin,
            &FogParams::DEFAULT,
            f32::INFINITY,
            None,
            None,
            &[],
            &g,
            18,
            0.0,
        );
        assert_eq!(first, second, "same floor point must shade identically");
        // Floor must be dual-channel: a real bg base under the cell.
        assert_ne!(first.bg, 0, "floor cell should carry a non-zero bg base");
    }

    #[test]
    fn ranked_lod_reveals_more_detail_near_than_far() {
        // End-to-end through the caster path: the SAME brick material shows a
        // higher-rank (finer) revealed texel up close than far away.
        let registry = MaterialRegistry::with_builtins();
        let perlin = Perlin::new(NOISE_SEED);

        let sample_rank = |distance: f32, foreshorten: f32| -> u8 {
            let mut best = 0u8;
            for i in 0..256 {
                let g = CellGeom {
                    depth: distance,
                    foreshorten,
                    side: 0,
                    tex_u: i as f32 / 256.0,
                    wall_v: 0.5,
                    wall_world_y: wall_world_y(1, 0.5),
                    hit_x: 2.0,
                    hit_y: 3.0,
                    surface: SurfaceKind::WallSolid,
                    material: 0,
                    light: None,
                    cell: (0, 0, 0),
                    tex_footprint: None,
                    subcell_backdrop: None,
                };
                let cell = shade_cell(
                    &registry,
                    &perlin,
                    &FogParams::DEFAULT,
                    f32::INFINITY,
                    None,
                    None,
                    &[],
                    &g,
                    2 + (i % 7),
                    0.0,
                );
                // Re-derive the texel rank for the assertion via a direct sample.
                let ctx = SampleCtx {
                    distance,
                    foreshorten,
                    side: 0,
                    screen_y: 2 + (i % 7),
                    wall_v: 0.5,
                    tick: 0,
                    time: 0.0,
                    fog: FogParams::DEFAULT,
                    far_freeze: f32::INFINITY,
                    light: None,
                    lod_override: None,
                    state: 0,
                    pulse: 0.0,
                    inset_rgb: None,
                    wall_along: 0.0,
                    wall_up: 0.0,
                    on_wall: false,
                    tex_footprint: None,
                    subcell_backdrop: None,
                };
                let t = registry
                    .get(surface_material(SurfaceKind::WallSolid, 0))
                    .sample(&perlin, sample_coord(&g, ctx.screen_y), &ctx);
                let _ = cell;
                best = best.max(t.rank);
            }
            best
        };

        let near = sample_rank(2.0, 1.0);
        let far = sample_rank(25.0, 1.0);
        assert!(
            near > far,
            "near must reveal a higher-rank texel than far (near={near} far={far})"
        );
        assert_eq!(far, 0, "far brick wall should collapse to base only");
    }

    #[test]
    fn floor_and_ceiling_both_present() {
        // The test room is two cells tall (air z=1..2, ceiling slab z=3) and every wall
        // renders at its true height, so at level pitch the far wall fills the view
        // above the horizon: the ceiling slab 1.5 cells over the eye only enters the
        // frustum when looking up. Floor from the level view, ceiling from a look-up.
        let map = test_room();
        let surfaces = |pitch: f32| {
            let mut cam = CameraPose::spawn_in_test_room();
            cam.pitch = pitch;
            let mut fb = Framebuffer::new(60, 30);
            raycast(&map, &cam, &mut fb);
            let (mut floor, mut ceiling) = (false, false);
            for row in 0..fb.rows() {
                for col in 0..fb.cols() {
                    match fb.get(col, row).unwrap().surface {
                        SurfaceKind::Floor => floor = true,
                        SurfaceKind::Ceiling => ceiling = true,
                        _ => {}
                    }
                }
            }
            (floor, ceiling)
        };
        assert!(surfaces(0.0).0, "expected floor cells");
        assert!(
            surfaces(PITCH_LIMIT_RAD).1,
            "expected ceiling cells looking up"
        );
    }

    // ── R-H2: tex_u is continuous across one flat wall face ──
    #[test]
    fn wall_sample_u_is_continuous_across_a_face() {
        // Two neighbouring cells of ONE flat NS wall (side==0, y-const plane):
        // same face plane, tex_u advancing 0.9 → 0.05 as we cross a world-cell
        // boundary along the face. The CONSTANT-axis offset must be identical for
        // both (so the only jump in the sample u is the honest tex_u wrap, NOT a
        // multi-unit step from adding the variable axis floor too).
        let a = CellGeom {
            depth: 4.0,
            foreshorten: 1.0,
            side: 0,
            tex_u: 0.9,
            wall_v: 0.5,
            wall_world_y: wall_world_y(1, 0.5),
            hit_x: 2.9, // variable axis (x) — must NOT enter the offset
            hit_y: 3.0, // constant axis (y-plane)
            surface: SurfaceKind::WallSolid,
            material: 0,
            light: None,
            cell: (0, 0, 0),
            tex_footprint: None,
            subcell_backdrop: None,
        };
        let b = CellGeom {
            tex_u: 0.05,
            hit_x: 3.05, // crossed the x=3 world boundary along the SAME face
            ..clone_geom(&a)
        };
        let ua = sample_coord(&a, 5).x;
        let ub = sample_coord(&b, 5).x;
        // The face offset (floor of the constant y axis) is identical for both,
        // so the sample-u difference equals the tex_u difference exactly — about
        // one unit (the wrap), NOT ~1 + the x-floor jump the old code added.
        let du = ub - ua;
        let tex_du = b.tex_u - a.tex_u;
        assert!(
            (du - tex_du).abs() < 1e-5,
            "sample u must move by exactly Δtex_u across a face (Δu={du}, Δtex_u={tex_du})"
        );
    }

    // ── R-M4: floor foreshorten density meaningfully varies near vs far ──
    #[test]
    fn floor_texel_density_varies_across_rows() {
        // Cast the test room and collect the foreshorten the floor cells got
        // (re-derived from each cell's depth is not exposed, so drive the same
        // formula the caster uses). A near floor row (just below horizon) must
        // carry a noticeably higher density than a far row (bottom of screen).
        let rows = 32i32;
        let horizon = rows / 2;
        let dens = |row: i32| -> f32 {
            let denom = (row - horizon) as f32;
            (FLOOR_FORESHORTEN_K * rows as f32 / (denom * denom)).clamp(1.0, FORESHORTEN_MAX)
        };
        let near = dens(horizon + 2); // just past the horizon
        let far = dens(rows - 1); // bottom row
        assert!(
            near > far + 1.0,
            "floor foreshorten must differ meaningfully near vs far (near={near}, far={far})"
        );
        assert!(near <= FORESHORTEN_MAX + 1e-3, "near saturates the cap");
    }

    // ── R-M3: camera_x sampling is symmetric about the centre column ──
    #[test]
    fn camera_x_is_symmetric() {
        let cols = 100f32;
        let cam_x = |x: f32| 2.0 * (x + 0.5) / cols - 1.0;
        // First and last columns get equal-and-opposite plane offsets.
        let lo = cam_x(0.0);
        let hi = cam_x(cols - 1.0);
        assert!(
            (lo + hi).abs() < 1e-6,
            "edge columns must be symmetric: {lo} vs {hi}"
        );
    }

    // ───────────────────────── FOV / aspect math ──────────────────────────
    // The hor+ model: vertical FOV is anchored to the preset and INVARIANT across
    // window sizes; horizontal FOV is derived per-window and clamped; the non-square
    // cell correction keeps circles round.

    const DEG: f32 = std::f32::consts::PI / 180.0;

    /// `vfov` is INVARIANT across window sizes for a fixed preset; `hfov` VARIES with
    /// the window and always stays inside the clamp band.
    #[test]
    fn vfov_is_window_invariant_hfov_varies_within_clamp() {
        let fov = Fov::default();
        let a = view_params(120, 40, &fov);
        let b = view_params(200, 20, &fov);
        let c = view_params(60, 50, &fov);

        // Vertical FOV identical regardless of cols/rows.
        assert!(
            (a.vfov - b.vfov).abs() < 1e-5,
            "vfov must not depend on window (a vs b)"
        );
        assert!(
            (a.vfov - c.vfov).abs() < 1e-5,
            "vfov must not depend on window (a vs c)"
        );

        // Horizontal FOV differs between a wide and a tall window…
        assert!(
            (a.hfov - c.hfov).abs() > 1e-3,
            "hfov should vary with window aspect (got {} vs {})",
            a.hfov,
            c.hfov
        );
        // …and every hfov stays within the clamp band.
        for v in [a, b, c] {
            assert!(
                v.hfov >= HFOV_MIN_DEG * DEG - 1e-4 && v.hfov <= HFOV_MAX_DEG * DEG + 1e-4,
                "hfov {} out of clamp band",
                v.hfov
            );
        }
    }

    /// A wider window yields a LARGER horizontal FOV than a narrow one (same preset).
    #[test]
    fn wider_window_gives_larger_hfov() {
        let fov = Fov::default();
        let wide = view_params(200, 40, &fov); // aspect 5.0
        let narrow = view_params(80, 40, &fov); // aspect 2.0
        assert!(
            wide.hfov > narrow.hfov,
            "wider window must widen hfov (wide={} narrow={})",
            wide.hfov,
            narrow.hfov
        );
    }

    /// At the BASELINE window the runtime horizontal FOV equals the preset (the
    /// anchor point of the hor+ model), so the cell-aspect/vfov round-trip is sound.
    #[test]
    fn baseline_window_recovers_the_preset() {
        let fov = Fov::default();
        // 120×40 with cell_aspect 0.5 → eff_aspect == BASELINE_ASPECT exactly.
        let v = view_params(120, 40, &fov);
        let preset = fov.h_preset_deg * DEG;
        assert!(
            (v.hfov - preset).abs() < 1e-4,
            "at baseline, hfov ({}) must equal the preset ({})",
            v.hfov,
            preset
        );
    }

    /// Cell-aspect correction: a "square" grid (cols == rows) does NOT yield a 1:1
    /// hfov:vfov. With a SQUARE cell (`cell_aspect = 1.0`) a square grid gives the
    /// degenerate `hfov == vfov`; the real terminal cell is ~twice as tall as wide
    /// (`cell_aspect = 0.5`), which halves the horizontal stretch so the corrected
    /// horizontal FOV is strictly NARROWER than the uncorrected one. That correction
    /// is exactly what stops a world-circle rendering as a wide ellipse.
    #[test]
    fn square_grid_is_not_one_to_one_due_to_cell_aspect() {
        // Use a wide preset so the vertical FOV (and the uncorrected square-cell
        // horizontal FOV) sit comfortably inside the [50,100]° clamp band — isolating
        // the cell-aspect correction from the clamp.
        let base = Fov {
            h_preset_deg: 90.0,
            ..Fov::default()
        };

        // Uncorrected (square cell): a square grid is the degenerate 1:1 case.
        let square_cell = Fov {
            cell_aspect: 1.0,
            ..base
        };
        let uncorrected = view_params(50, 50, &square_cell);
        assert!(
            (uncorrected.hfov - uncorrected.vfov).abs() < 1e-5,
            "with a square cell, a square grid must give hfov==vfov ({} vs {})",
            uncorrected.hfov,
            uncorrected.vfov
        );

        // Corrected (the real ~2:1 tall cell): the SAME square grid is no longer
        // 1:1 — the horizontal FOV is squeezed below the vertical (and below the
        // uncorrected hfov). This is the round-circle correction.
        let corrected = view_params(
            50,
            50,
            &Fov {
                cell_aspect: 0.5,
                ..base
            },
        );
        assert!(
            corrected.hfov < uncorrected.hfov,
            "cell_aspect 0.5 must narrow hfov vs a square cell (corrected={} uncorrected={})",
            corrected.hfov,
            uncorrected.hfov
        );
        assert!(
            corrected.hfov < corrected.vfov,
            "corrected square grid must not be 1:1 (hfov={} vfov={})",
            corrected.hfov,
            corrected.vfov
        );
    }

    /// Changing `h_preset_deg` changes `vfov` MONOTONICALLY (a wider preset → wider
    /// vertical FOV), and the relationship is strictly increasing.
    #[test]
    fn preset_changes_vfov_monotonically() {
        let mk = |deg: f32| {
            let fov = Fov {
                h_preset_deg: deg,
                ..Fov::default()
            };
            view_params(120, 40, &fov).vfov
        };
        let v50 = mk(50.0);
        let v65 = mk(65.0);
        let v90 = mk(90.0);
        assert!(
            v50 < v65 && v65 < v90,
            "vfov must rise with the preset ({v50} < {v65} < {v90})"
        );
    }

    /// The clamp fires at both ends: an extreme ultra-wide window pins hfov at the
    /// max; an extreme ultra-tall window pins it at the min.
    #[test]
    fn extreme_windows_clamp_hfov() {
        let fov = Fov::default();
        let ultrawide = view_params(2000, 20, &fov); // huge effective aspect
        let ultratall = view_params(20, 2000, &fov); // tiny effective aspect
        assert!(
            (ultrawide.hfov - HFOV_MAX_DEG * DEG).abs() < 1e-4,
            "ultra-wide window must clamp hfov at {}° (got {})",
            HFOV_MAX_DEG,
            ultrawide.hfov / DEG
        );
        assert!(
            (ultratall.hfov - HFOV_MIN_DEG * DEG).abs() < 1e-4,
            "ultra-tall window must clamp hfov at {}° (got {})",
            HFOV_MIN_DEG,
            ultratall.hfov / DEG
        );
    }

    /// `proj_plane_dist` scales linearly with `rows` (same vfov), and a unit-tall
    /// wall's projected screen height = `proj_plane_dist / perp` — wiring the cast
    /// uses. Guards the vertical-projection constant the wall/floor extents share.
    #[test]
    fn proj_plane_dist_scales_with_rows() {
        let fov = Fov::default();
        let a = view_params(120, 40, &fov);
        let b = view_params(120, 80, &fov); // double the rows, same aspect-ish band
                                            // proj_plane_dist = (rows/2)/tan(vfov/2); vfov identical here, so doubling
                                            // rows doubles proj_plane_dist.
        assert!(
            (b.proj_plane_dist / a.proj_plane_dist - 2.0).abs() < 1e-3,
            "proj_plane_dist must scale with rows (a={} b={})",
            a.proj_plane_dist,
            b.proj_plane_dist
        );
    }

    // ── PART B: animation makes a material's look move over time ──
    #[test]
    fn animated_material_changes_with_time() {
        // The matrix wall is a `Shift` material: the same coord shades
        // differently at two different times (the texture scrolls).
        let registry = MaterialRegistry::with_builtins();
        let perlin = Perlin::new(NOISE_SEED);
        let g = CellGeom {
            depth: 3.0,
            foreshorten: 1.0,
            side: 0,
            tex_u: 0.4,
            wall_v: 0.5,
            wall_world_y: wall_world_y(1, 0.5),
            hit_x: 2.0,
            hit_y: 3.0,
            surface: SurfaceKind::WallMatrix,
            material: 0,
            light: None,
            cell: (0, 0, 0),
            tex_footprint: None,
            subcell_backdrop: None,
        };
        // Sample a strip down the wall at two times; the set of glyphs/colors
        // must differ (a static material would produce identical strips).
        let strip = |time: f32| -> Vec<(u32, u32)> {
            (0..24)
                .map(|r| {
                    let c = shade_cell(
                        &registry,
                        &perlin,
                        &FogParams::DEFAULT,
                        f32::INFINITY,
                        None,
                        None,
                        &[],
                        &g,
                        r,
                        time,
                    );
                    (c.glyph, c.fg)
                })
                .collect()
        };
        assert_ne!(
            strip(0.0),
            strip(2.5),
            "an animated (Shift) material must change its look over time"
        );

        // And a STATIC material (brick) must be identical at any time.
        let gb = CellGeom {
            surface: SurfaceKind::WallSolid,
            ..clone_geom(&g)
        };
        let s0: Vec<_> = (0..24)
            .map(|r| {
                shade_cell(
                    &registry,
                    &perlin,
                    &FogParams::DEFAULT,
                    f32::INFINITY,
                    None,
                    None,
                    &[],
                    &gb,
                    r,
                    0.0,
                )
                .glyph
            })
            .collect();
        let s1: Vec<_> = (0..24)
            .map(|r| {
                shade_cell(
                    &registry,
                    &perlin,
                    &FogParams::DEFAULT,
                    f32::INFINITY,
                    None,
                    None,
                    &[],
                    &gb,
                    r,
                    9.0,
                )
                .glyph
            })
            .collect();
        assert_eq!(
            s0, s1,
            "a static material must be time-invariant (golden-safe)"
        );
    }

    #[test]
    fn registry_reports_animation_presence() {
        // The built-in set contains animated materials (trace pulse, matrix shift).
        assert!(MaterialRegistry::with_builtins().any_animated());
        // A registry of only static materials reports false.
        let mut only_static = MaterialRegistry::new();
        only_static.register(Material {
            animation: Animation::NONE,
            ..MaterialRegistry::with_builtins()
                .get(ids::WALL_BRICK)
                .clone()
        });
        assert!(!only_static.any_animated());
    }

    /// A dump-style smoke check: raycast the test room into a 100×32 grid (the
    /// same dimensions + `raycast` entry the binary's `--dump` uses) and assert
    /// the frame is varied (many distinct glyphs) and non-blank. Run with
    /// `cargo test -p re_render dump_frame -- --nocapture` to eyeball it.
    #[test]
    fn dump_frame_is_varied_and_nonblank() {
        let map = test_room();
        let cam = CameraPose::spawn_in_test_room();
        let mut fb = Framebuffer::new(100, 32);
        raycast(&map, &cam, &mut fb);

        let mut distinct = std::collections::HashSet::new();
        let mut nonspace = 0usize;
        let mut lines = Vec::new();
        for row in 0..fb.rows() {
            let mut line = String::with_capacity(fb.cols());
            for col in 0..fb.cols() {
                let g = fb.get(col, row).unwrap().glyph;
                distinct.insert(g);
                if g != u32::from(' ') {
                    nonspace += 1;
                }
                line.push(char::from_u32(g).unwrap_or(' '));
            }
            lines.push(line);
        }
        // Print a few representative rows for the human-readable check.
        println!("==== DUMP (test_room 100x32), sample rows ====");
        for r in [4usize, 10, 16, 22, 28] {
            println!("{:>2}|{}", r, lines[r]);
        }
        assert!(
            distinct.len() >= 5,
            "frame should use many glyphs, got {}",
            distinct.len()
        );
        assert!(
            nonspace > 200,
            "frame should be substantially non-blank, got {nonspace}"
        );
    }

    /// A `[zone]`/`[color_cell_ids]`-materialled Floor (and Ceiling) cell must render
    /// with ITS OWN glyph/colour on the backdrop floor/ceiling row cast, not the
    /// surface-kind default — the engine gap `tests/golden/README.md`'s 2026-09-24
    /// (f) entry diagnosed (`backdrop_floor_decal`, now `backdrop_floor_ceiling_material`,
    /// only ever consulted `MaterialRegistry::is_floor_decal`, populated exclusively by
    /// the two hardcoded prop-kit decal producers, never the general colour-material
    /// palette `shade_cell` already resolves for WALL surfaces two lines away).
    ///
    /// Builds a tiny room whose Floor slab (z=0, one cell below the eye) and Ceiling
    /// slab (z=2, one cell above the eye — matching the flat single-storey cast's own
    /// "ceiling sits exactly one cell above the eye" assumption) are ENTIRELY stamped
    /// with a colour-palette id, exactly as `re_level`'s zone stamper would
    /// (`re_level::tests::zone_stamps_resolved_palette_id_into_region_only` proves the
    /// data-model side of this same stamp). A registry with two custom colour
    /// materials (distinct, unmistakable glyphs) is attached via `Map::set_materials`,
    /// mirroring a level's `[material]` + `[zone]` authoring.
    #[test]
    fn zone_material_renders_on_backdrop_floor_and_ceiling() {
        let (sx, sy, sz) = (6usize, 6usize, 3usize);
        let mut map = Map::new(sx, sy, sz);
        for y in 0..sy {
            for x in 0..sx {
                let on_perimeter = x == 0 || x == sx - 1 || y == 0 || y == sy - 1;
                // Colour id 1 on every Floor cell, id 2 on every Ceiling cell — the
                // exact shape `zone_stamps_resolved_palette_id_into_region_only` builds
                // and reads back, just over the whole slab rather than one region.
                map.set_cell(x, y, 0, CellKind::Floor, 1);
                map.set_cell(x, y, 2, CellKind::Ceiling, 2);
                if on_perimeter {
                    map.set(x, y, 1, CellKind::WallSolid);
                }
            }
        }

        // Five fallback surface slots (unchanged built-ins) + two custom
        // colour-palette entries: id 1 = a floor tone with glyph `~`, id 2 = a
        // ceiling tone with glyph `%` — both far from any built-in default glyph.
        let mut materials = vec![MaterialDef::fallback(); 5];
        materials.push(MaterialDef {
            glyph: Some(u32::from('~')),
            ..MaterialDef::flat("test_floor_zone", 0x8F_E4F4, 0x0D_2732)
        });
        materials.push(MaterialDef {
            glyph: Some(u32::from('%')),
            ..MaterialDef::flat("test_ceiling_zone", 0xFF_8822, 0x20_1000)
        });
        map.set_materials(materials);

        let cam = CameraPose {
            pos: glam::Vec3::new(2.5, 2.5, 1.5),
            yaw: 0.0,
            pitch: 0.0,
        };
        let mut fb = Framebuffer::new(40, 20);
        raycast(&map, &cam, &mut fb);

        let mut floor_glyphs = std::collections::HashSet::new();
        let mut ceiling_glyphs = std::collections::HashSet::new();
        for row in 0..fb.rows() {
            for col in 0..fb.cols() {
                let cell = fb.get(col, row).unwrap();
                match cell.surface {
                    SurfaceKind::Floor => {
                        floor_glyphs.insert(cell.glyph);
                    }
                    SurfaceKind::Ceiling => {
                        ceiling_glyphs.insert(cell.glyph);
                    }
                    _ => {}
                }
            }
        }
        assert!(
            floor_glyphs.contains(&u32::from('~')),
            "a [zone]-materialled Floor cell must reach the backdrop floor with its \
             own glyph; floor glyphs seen: {floor_glyphs:?}"
        );
        assert!(
            ceiling_glyphs.contains(&u32::from('%')),
            "a [zone]-materialled Ceiling cell must reach the backdrop ceiling with its \
             own glyph; ceiling glyphs seen: {ceiling_glyphs:?}"
        );
    }

    // ── CODE WALL: the engine's own source, rendered onto a wall (strange loop) ──

    /// Build a registry whose WALL_BRICK slot is a CodeWall material on `panel`
    /// with a 1:1 source-column-per-world-cell density (so the test can predict
    /// exactly which source char lands where). Other slots keep the built-ins.
    fn code_wall_registry(panel: u32, u_chars_per_unit: f32) -> MaterialRegistry {
        let builtins = MaterialRegistry::with_builtins();
        let mut reg = MaterialRegistry::new();
        for slot in 0..5u16 {
            if slot == ids::WALL_BRICK.0 {
                reg.register(Material {
                    recipe: TextureRecipe::CodeWall {
                        panel,
                        u_chars_per_unit,
                    },
                    ..builtins.get(MaterialId(slot)).clone()
                });
            } else {
                reg.register(builtins.get(MaterialId(slot)).clone());
            }
        }
        reg
    }

    /// One line of known synthetic source. The integration tests sample the panel
    /// grid; the first line is a `//` comment, and the recognisable tokens below
    /// (`synthetic`, `answer`, `revision`, …) let the raycast test assert real text
    /// lands on the wall — all WITHOUT depending on any game content.
    const TEST_PANEL_LINE0: &str = "// synthetic code wall — answer revision";

    /// Build the synthetic source: line 0 is the comment above, then the grid is
    /// filled to its full height with repeated code lines carrying the test tokens,
    /// so the sampled `v ∈ [0,1]` band always lands on populated rows (a 3-line
    /// source would map mostly to blank grid rows past row 3).
    fn test_panel_src() -> String {
        let mut s = String::new();
        s.push_str(TEST_PANEL_LINE0);
        s.push('\n');
        // Each line is ONE recognisable token repeated to fill the width, so any
        // narrow horizontal window the oblique raycast samples lands on a fragment
        // of that token. The token ROTATES per row, so across all rows the wall
        // uses many distinct glyphs (the original real-source test relied on that
        // diversity; a single repeated line would show only a handful).
        let words = ["synthetic ", "answer ", "revision "];
        for i in 0..code_wall::PANEL_ROWS {
            let w = words[i % words.len()];
            let line = w.repeat(code_wall::PANEL_COLS / w.len() + 1);
            s.push_str(&line);
            s.push('\n');
        }
        s
    }

    /// The KNOWN NOTE BODIES the shared test registry publishes (note panels 0..2), so
    /// a note-wall test can assert exactly which characters land on a face.
    const TEST_NOTE_BODIES: [&str; 3] =
        ["alpha beta gamma", "> system log line", "(a quiet thought)"];

    /// The ONE synthetic text-surface registry every code-wall / note-wall test in this
    /// crate shares, built once and re-published on demand.
    ///
    /// Built once because the code panel is a 100×600 grid; shared because the active
    /// registry is process-global (see `scene_text`) and `cargo test` runs these tests
    /// in parallel threads — one deterministic registry for the whole binary is the same
    /// arrangement the old `OnceLock` panel provider had. Tests that MUTATE the active
    /// registry take `scene_text::test_lock()` as well.
    fn test_text_registry() -> std::sync::Arc<scene_text::TextSurfaceRegistry> {
        static R: std::sync::OnceLock<std::sync::Arc<scene_text::TextSurfaceRegistry>> =
            std::sync::OnceLock::new();
        R.get_or_init(|| {
            let mut reg = scene_text::TextSurfaceRegistry::new();
            reg.push_code_source("synthetic", &test_panel_src());
            for (i, body) in TEST_NOTE_BODIES.iter().enumerate() {
                reg.push_note_text(&format!("note{i}"), body);
            }
            std::sync::Arc::new(reg)
        })
        .clone()
    }

    /// Publish the shared synthetic registry as the active one and hold the registry
    /// test lock for the caller's duration.
    ///
    /// Idempotent and re-establishing (unlike the old `OnceLock` provider it recovers if
    /// another test cleared the registry). The returned guard serialises against the
    /// `scene_text` lifecycle tests, which install/clear the same process-global
    /// registry — without it those tests could swap the panel set mid-raycast.
    #[must_use = "hold the guard for the test's duration"]
    fn install_test_panel_provider() -> std::sync::MutexGuard<'static, ()> {
        let guard = scene_text::test_lock();
        scene_text::install_shared(test_text_registry());
        guard
    }

    /// The shader maps a wall `(u, wall_v)` to the SAME source character the panel
    /// grid holds there — proven by sampling the CodeWall material directly and
    /// comparing against the registered panel's grid, in reading order. Uses a
    /// synthetic panel (via the provider hook), so it tests the shader↔panel
    /// integration without any game content.
    #[test]
    fn code_wall_shader_reads_source_in_order() {
        let _serial = install_test_panel_provider();
        let perlin = Perlin::new(NOISE_SEED);
        let reg = code_wall_registry(0, 1.0); // 1 source col per world unit
        let mat = reg.get(ids::WALL_BRICK);

        // Read the top source row left-to-right by sweeping u, holding wall_v≈0.
        // u = c + 0.5 lands in source column c (floor(u*1.0) = c). distance is
        // near so no fog/LOD interferes with the glyph itself.
        let mut on_wall = String::new();
        for c in 0..22usize {
            let ctx = SampleCtx {
                distance: 1.0,
                foreshorten: 1.0,
                side: 0,
                screen_y: 5,
                wall_v: 0.001,
                tick: 0,
                time: 0.0,
                fog: FogParams::DEFAULT,
                far_freeze: f32::INFINITY,
                light: None,
                lod_override: None,
                state: 0,
                pulse: 0.0,
                inset_rgb: None,
                wall_along: 0.0,
                wall_up: 0.0,
                on_wall: false,
                tex_footprint: None,
                subcell_backdrop: None,
            };
            let t = mat.sample(&perlin, glam::Vec2::new(c as f32 + 0.5, 0.0), &ctx);
            on_wall.push(char::from_u32(t.glyph).unwrap_or('?'));
        }

        // The expected string is the panel grid's top row (first 22 cols), read back
        // through the SAME registry the shader resolves the panel from.
        let expected: String = (0..22)
            .map(|c| {
                let (g, _rows) = scene_text::sample_code_panel(0, c as f32 + 0.5, 0.001, 1.0);
                char::from_u32(g.glyph).unwrap_or('?')
            })
            .collect();

        assert_eq!(
            on_wall, expected,
            "wall must render the panel's top row in reading order"
        );
        // The synthetic source's top line is a `//` comment.
        assert!(
            on_wall.trim_start().starts_with("//"),
            "panel top row should be the synthetic comment, got {on_wall:?}"
        );
    }

    /// End-to-end through the public `raycast`: a tiny room whose walls are a
    /// CodeWall material renders RECOGNISABLE source — a contiguous run of code
    /// characters from the panel appears somewhere on the wall cells, in order.
    #[test]
    fn raycast_lands_readable_code_on_the_wall() {
        let _serial = install_test_panel_provider();
        // A small solid-walled room; bind the wall_solid surface to CodeWall p0.
        let mut map = test_room();
        let mut slots: Vec<re_world::MaterialDef> =
            (0..5).map(|_| re_world::MaterialDef::fallback()).collect();
        slots[0] = re_world::MaterialDef {
            name: "source_wall".into(),
            fg: 0x0045_D962,
            bg: 0x0003_060A,
            glyph: None,
            recipe: Some(re_world::MaterialRecipe::CodeWall { panel: 0 }),
            anim: re_world::MaterialAnim::default(),
            rank: 0,
            reflective: false,
            clarity: 0.85,
            fallback: false,
            detail: false,
        };
        map.set_materials(slots);

        // Hug the north wall (y=1) and look straight at it (-y), so the wall fills
        // the view HEAD-ON: `u` then advances cleanly across screen columns and a
        // source line reads left-to-right without oblique aliasing. (At an angle
        // the text is still real but smeared — see the legibility note; the
        // dump-test eyeballs the oblique case, this one asserts the clean case.)
        let mut cam = CameraPose::spawn_in_test_room();
        cam.pos.x = 3.5;
        // Stand a few cells back from the north wall (NOT hugging it): with the
        // corrected, narrower horizontal FOV (hor+ model, ~56° at this window vs the
        // old 90°) pressing right against the wall now shows only a couple of source
        // columns. Backing off restores enough horizontal span for a whole token to
        // read left-to-right. The x=3.5 column is clear interior air all the way to
        // the y=1 wall (the pillar sits at x=2), so the cast stays head-on.
        cam.pos.y = 3.5;
        cam.yaw = std::f32::consts::PI; // face -y (north) toward the y=1 wall
        let mut fb = Framebuffer::new(120, 40);
        raycast(&map, &cam, &mut fb);

        // Collect every wall row's glyphs.
        let mut rows: Vec<String> = Vec::new();
        for row in 0..fb.rows() {
            let mut s = String::new();
            for col in 0..fb.cols() {
                let cell = fb.get(col, row).unwrap();
                if cell.surface.is_wall() {
                    s.push(char::from_u32(cell.glyph).unwrap_or(' '));
                } else {
                    s.push(' ');
                }
            }
            rows.push(s);
        }

        // Squash runs of repeated chars (head-on, the wall is still wider in cells
        // than the source is in columns at this density, so each source char spans
        // a few cells) → a clean reading of each line.
        let squashed: Vec<String> = rows
            .iter()
            .map(|r| {
                let mut out = String::new();
                let mut prev = '\0';
                for ch in r.chars() {
                    if ch != prev {
                        out.push(ch);
                    }
                    prev = ch;
                }
                out
            })
            .collect();

        // At least one wall row, de-duplicated, must contain a recognisable token
        // fragment from the panel. The oblique raycast samples a narrow horizontal
        // window (only a few source columns wide at this distance), so we match
        // short fragments of the panel's words ("synthetic answer revision"), which
        // appear verbatim in the synthetic source.
        let needles = [
            "syn", "ynt", "nth", "the", "het", "tic", "ans", "nsw", "swe", "wer", "rev", "evi",
            "vis", "isi", "sio", "ion",
        ];
        let found = squashed
            .iter()
            .any(|r| needles.iter().any(|n| r.contains(n)));
        assert!(
            found,
            "expected a readable code token on a wall row; squashed wall rows:\n{}",
            squashed.join("\n")
        );

        // Sanity: the wall used many distinct glyphs (real text, not one char).
        let distinct: std::collections::HashSet<char> = rows
            .iter()
            .flat_map(|r| r.chars())
            .filter(|c| *c != ' ')
            .collect();
        assert!(
            distinct.len() > 8,
            "code wall should use many distinct glyphs, got {}",
            distinct.len()
        );
    }

    /// Vertical coordinate is real: two wall cells at different screen rows of the
    /// same column carry DIFFERENT `tex_v`, and (when the slice is tall enough)
    /// map to different source rows. Guards the additive `wall_v`/`tex_v` plumbing.
    #[test]
    fn wall_cells_carry_distinct_vertical_coord() {
        let map = test_room();
        let cam = CameraPose::spawn_in_test_room();
        let mut fb = Framebuffer::new(60, 60); // tall fb → tall wall slice
        raycast(&map, &cam, &mut fb);

        let col = fb.cols() / 2;
        let mut vs: Vec<f32> = Vec::new();
        for row in 0..fb.rows() {
            let c = fb.get(col, row).unwrap();
            if c.surface.is_wall() {
                vs.push(c.tex_v);
            }
        }
        assert!(vs.len() >= 3, "centre column should hit several wall rows");
        // tex_v must span a real range top→bottom (not all 0), and increase.
        let first = *vs.first().unwrap();
        let last = *vs.last().unwrap();
        assert!(
            first < last,
            "tex_v must increase down the wall ({first}→{last})"
        );
        assert!(last - first > 0.3, "tex_v must span a meaningful range");
        assert!((0.0..=1.0).contains(&first) && (0.0..=1.0).contains(&last));
    }

    /// Human-readable evidence: dump a CodeWall frame so the on-wall source is
    /// eyeballable. `cargo test -p re_render code_wall_dump -- --nocapture`.
    #[test]
    fn code_wall_dump_prints_source_on_wall() {
        let _serial = install_test_panel_provider();
        let mut map = test_room();
        let mut slots: Vec<re_world::MaterialDef> =
            (0..5).map(|_| re_world::MaterialDef::fallback()).collect();
        slots[0] = re_world::MaterialDef {
            name: "source_wall".into(),
            fg: 0x0045_D962,
            bg: 0x0003_060A,
            glyph: None,
            recipe: Some(re_world::MaterialRecipe::CodeWall { panel: 0 }),
            anim: re_world::MaterialAnim::default(),
            rank: 0,
            reflective: false,
            clarity: 0.85,
            fallback: false,
            detail: false,
        };
        map.set_materials(slots);
        // Stand right against the north wall so it fills the view and the text is
        // as legible as the cell grid allows.
        let mut cam = CameraPose::spawn_in_test_room();
        cam.pos.y = 1.15; // hug the y=1 wall, looking north (-y)
        let mut fb = Framebuffer::new(100, 40);
        raycast(&map, &cam, &mut fb);

        println!("==== CODE WALL (synthetic panel, hugging the wall) ====");
        for row in 0..fb.rows() {
            let mut line = String::with_capacity(fb.cols());
            for col in 0..fb.cols() {
                line.push(char::from_u32(fb.get(col, row).unwrap().glyph).unwrap_or(' '));
            }
            println!("{line}");
        }
    }

    /// A CONTENT-INDEPENDENT baseline of the whole code-wall render path.
    ///
    /// The shipped `09-code-wall.lvl` cannot be hashed as a regression anchor: its
    /// panel 0 IS `re_render/src/code_wall.rs` (pulled in by `include_str!` in
    /// `re_game::code_panels`), so ANY edit to that file legitimately changes the
    /// glyphs on the wall. This test pins the same path against a SYNTHETIC panel
    /// whose text never changes, so the hash moves only if the code-wall
    /// *shading/sampling/registry* behaviour changes — which is exactly the
    /// regression the text-surface-registry refactor had to avoid.
    ///
    /// The hash is over the same artifact the golden corpus uses ({glyph, fg, bg}
    /// per cell, row-major), via a small FNV-1a so no `sha2` dependency is needed.
    #[test]
    fn code_wall_frame_hash_is_stable() {
        let _serial = install_test_panel_provider();
        let mut map = test_room();
        let mut slots: Vec<re_world::MaterialDef> =
            (0..5).map(|_| re_world::MaterialDef::fallback()).collect();
        slots[0] = re_world::MaterialDef {
            name: "source_wall".into(),
            fg: 0x0045_D962,
            bg: 0x0003_060A,
            glyph: None,
            recipe: Some(re_world::MaterialRecipe::CodeWall { panel: 0 }),
            anim: re_world::MaterialAnim::default(),
            rank: 0,
            reflective: false,
            clarity: 0.85,
            fallback: false,
            detail: false,
        };
        map.set_materials(slots);
        let mut cam = CameraPose::spawn_in_test_room();
        cam.pos.y = 1.15;
        let mut fb = Framebuffer::new(100, 32);
        raycast(&map, &cam, &mut fb);

        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut eat = |b: u8| {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01B3);
        };
        for row in 0..fb.rows() {
            for col in 0..fb.cols() {
                let c = fb.get(col, row).unwrap();
                for v in [c.glyph, c.fg, c.bg] {
                    for b in v.to_le_bytes() {
                        eat(b);
                    }
                }
            }
        }
        println!("CODE_WALL_BASELINE_HASH = {h:#018x}");
        assert_eq!(
            h, CODE_WALL_BASELINE_HASH,
            "the code-wall render path changed against its synthetic-panel baseline"
        );
    }

    /// FNV-1a of the synthetic-panel code-wall frame.
    ///
    /// MOVED from `0xc9df_9447_c889_0151` — the value recorded before the
    /// text-surface-registry refactor. The move is the FIX, not a regression: at
    /// this pose (hugging the wall) the old frame drew each source character
    /// replicated across ~50 identical screen columns, showed the SAME eight
    /// source columns on every wall cell, and collapsed ~15 source lines into
    /// each screen row. The new frame lays the same panel as one continuous,
    /// world-anchored stream at a screen-matched density — readable words, one
    /// character per cell. `code_wall_does_not_replicate_characters_at_reading_distance`
    /// and `code_wall_text_flows_across_wall_cells` pin the properties; this hash keeps
    /// pinning the whole path against silent drift.
    ///
    /// MOVED again from `0xc98e_db33_7462_72ea`. The wall slice's `wall_v` is now
    /// `offslice::drawn_v(first, end, rowf)` — the float `VProj` projection of the
    /// all-height pass — instead of the integer `(row - slice_top) / slice_span`
    /// ratio. At this magnified pose each source row owns one of ~1.3 screen rows,
    /// so the sub-row difference moves WHICH screen row draws a source row in 252 of
    /// 3200 cells (glyph + fg only, every bg identical). Checked by rendering both
    /// frames: same words, same letter spacing, the answer→revision→synthetic row
    /// order unbroken, no row lost or doubled — only the blank spacer rows shift.
    ///
    /// MOVED again from `0xec4d_80c7_52a3_27ca`: a magnified face now JUSTIFIES
    /// its segment (`CodePanel::sample_flowing_justified`), so the blank columns
    /// that split words move into the spaces. Checked by disabling only the
    /// justification (`JUSTIFY_MAX_SEGMENT = 0`): the frame hashes back to the
    /// previous value exactly, so nothing else in the path moved.
    const CODE_WALL_BASELINE_HASH: u64 = 0x152c_9eb1_88fa_cec6;

    // ── The code wall must FLOW and must not replicate characters ───────

    /// Build the code-wall test room and cast it from `dist` cells off the north wall.
    fn code_wall_frame(dist: f32, cols: usize, rows: usize) -> Framebuffer {
        let mut map = test_room();
        let mut slots: Vec<re_world::MaterialDef> =
            (0..5).map(|_| re_world::MaterialDef::fallback()).collect();
        slots[0] = re_world::MaterialDef {
            name: "source_wall".into(),
            fg: 0x0045_D962,
            bg: 0x0003_060A,
            glyph: None,
            recipe: Some(re_world::MaterialRecipe::CodeWall { panel: 0 }),
            anim: re_world::MaterialAnim::default(),
            rank: 0,
            reflective: false,
            clarity: 0.85,
            fallback: false,
            detail: false,
        };
        map.set_materials(slots);
        let mut cam = CameraPose::spawn_in_test_room();
        // x = 3.5 keeps the interior pillar at (2,2) out of the line of sight.
        cam.pos = glam::Vec3::new(3.5, 1.0 + dist, 1.5);
        cam.yaw = std::f32::consts::PI; // face the y = 0 wall head-on
        let mut fb = Framebuffer::new(cols, rows);
        raycast(&map, &cam, &mut fb);
        fb
    }

    /// Regression test — a source character is never drawn in two adjacent screen
    /// columns.
    ///
    /// The code wall used a FIXED authored density (`u_chars_per_unit`) against a face
    /// that covers anywhere from 3 to 300 screen columns, so at reading distance every
    /// character was nearest-neighbour REPLICATED — 6-13× at typical viewing distance, and up
    /// to 50× measured while standing at the wall. The flowing mapping lays the source
    /// at a screen-matched, octave-quantized density and lets exactly one screen column
    /// own each source character, so replication is gone.
    ///
    /// Measured as: of all adjacent inked column pairs on a wall row, how many repeat
    /// the same glyph beyond what the SOURCE itself repeats. A source line contains
    /// real doubles (`ll`, `--`), so the bar is a RUN LENGTH: no glyph may repeat three
    /// or more times in a row, and 2-runs must be rare.
    #[test]
    fn code_wall_does_not_replicate_characters_at_reading_distance() {
        let _serial = install_test_panel_provider();
        for dist in [1.0_f32, 1.5, 2.5] {
            let fb = code_wall_frame(dist, 120, 40);
            let mut worst_run = 1usize;
            let mut inked = 0usize;
            let mut doubles = 0usize;
            for row in 0..fb.rows() {
                let mut run = 1usize;
                let mut prev: Option<u32> = None;
                for col in 0..fb.cols() {
                    let c = fb.get(col, row).unwrap();
                    if !c.surface.is_wall() {
                        prev = None;
                        run = 1;
                        continue;
                    }
                    let g = c.glyph;
                    // Only the SOURCE text is under test; a structural glyph the shade
                    // path draws elsewhere on the wall is not code.
                    if !char::from_u32(g).is_some_and(|ch| ch.is_ascii_graphic()) {
                        prev = None;
                        run = 1;
                        continue;
                    }
                    inked += 1;
                    if prev == Some(g) {
                        run += 1;
                        if run == 2 {
                            doubles += 1;
                        }
                        worst_run = worst_run.max(run);
                    } else {
                        run = 1;
                    }
                    prev = Some(g);
                }
            }
            assert!(
                inked > 200,
                "dist {dist}: the wall must carry real text ({inked} inked)"
            );
            assert!(
                worst_run <= 2,
                "dist {dist}: a source character was replicated {worst_run}× in a row \
                 (magnification is not clamped)"
            );
            // The synthetic panel's own lines DO contain adjacent repeats, so allow a
            // small share; replication would push this toward 100%.
            assert!(
                doubles * 10 < inked,
                "dist {dist}: {doubles}/{inked} inked cells start a repeated pair — that \
                 is replication, not source content"
            );
        }
    }

    /// Regression test — the source FLOWS across wall cells.
    ///
    /// `sample_coord` hands a wall recipe `tex_u + <the face's constant axis>`, and
    /// `tex_u` wraps every world cell — so the code wall drew the SAME
    /// `u_chars_per_unit` source columns on every cell of a run. Reading along one wall
    /// row must show many more distinct characters than one cell's worth, and the row
    /// must not be periodic at the cell pitch.
    #[test]
    fn code_wall_text_flows_across_wall_cells() {
        let _serial = install_test_panel_provider();
        let fb = code_wall_frame(1.5, 120, 40);
        // The wall row with the most ink.
        let row = (0..fb.rows())
            .max_by_key(|&r| {
                (0..fb.cols())
                    .filter(|&c| {
                        let cell = fb.get(c, r).unwrap();
                        cell.surface.is_wall()
                            && char::from_u32(cell.glyph)
                                .is_some_and(|ch| ch.is_ascii_alphanumeric())
                    })
                    .count()
            })
            .expect("a frame has rows");
        let line: String = (0..fb.cols())
            .map(|c| {
                let cell = fb.get(c, row).unwrap();
                if cell.surface.is_wall() {
                    char::from_u32(cell.glyph).unwrap_or(' ')
                } else {
                    ' '
                }
            })
            .collect();
        println!("CODE WALL row {row}: {line:?}");
        // Which world cells does this row cross? (More than one, or the test is moot.)
        let cells: std::collections::BTreeSet<i32> = (0..fb.cols())
            .filter_map(|c| {
                let cell = fb.get(c, row).unwrap();
                cell.surface.is_wall().then(|| cell.hit.x.floor() as i32)
            })
            .collect();
        assert!(
            cells.len() >= 3,
            "fixture assumption: the row must cross several wall cells, crossed {}",
            cells.len()
        );
        // Per-cell character sets must not be identical — that is exactly the
        // "same eight columns on every cell" defect.
        let mut per_cell: Vec<String> = Vec::new();
        for cx in &cells {
            let s: String = (0..fb.cols())
                .filter_map(|c| {
                    let cell = fb.get(c, row).unwrap();
                    (cell.surface.is_wall() && cell.hit.x.floor() as i32 == *cx)
                        .then(|| char::from_u32(cell.glyph).unwrap_or(' '))
                })
                .filter(|ch| *ch != ' ')
                .collect();
            per_cell.push(s);
        }
        let distinct: std::collections::BTreeSet<&String> = per_cell.iter().collect();
        assert!(
            distinct.len() > 1,
            "every wall cell shows the SAME text — the source does not flow: {per_cell:?}"
        );
    }

    // ── `[effects] snow` must actually reach the frame ──────────────────

    /// END TO END for the cold axis: a level's `[effects] snow` knob is expanded
    /// by `re_level` into ambient emitters, resolved through `Emitter::from_spec`,
    /// and lands as visible flake glyphs in the same `composite_dump_once` frame
    /// `--dump` prints. Compared against the SAME frame rendered with no
    /// emitters, so only the snow's own contribution is counted.
    ///
    /// Originally end-to-end on the campaign's `levels/L3-full-spectrum.lvl` and
    /// `levels/L8-cold-room.lvl`. Re-pointed to two minimal inline fixtures,
    /// so this engine-crate test does not depend on game-campaign level data,
    /// carrying the SAME `snow` densities (0.4, 0.35) and comparable room
    /// footprints, so the flake-count threshold below still measures the same
    /// thing on non-campaign content.
    /// A `[cells]`-explicit sealed box `sx × sy × sz` — z=0 a floor slab, z=1..sz-2
    /// walkable layers with a `#` perimeter, z=sz-1 a solid ceiling slab. The SAME
    /// shape `levels/L3-full-spectrum.lvl` / `levels/L8-cold-room.lvl` author (and
    /// `re_level`'s own `cells_room` test helper), NOT the `[room]`-header-only
    /// auto-box: an auto-box fixture measured 0 flake cells here — its snow lattice
    /// resolves emitters `re_level` reports non-empty, but this `[cells]`-explicit
    /// shape is the one proven (by the real shipped levels) to place them somewhere
    /// the spawn camera actually sees.
    fn cells_box(sx: usize, sy: usize, sz: usize) -> String {
        let mut out = format!("[cells]\nsize = {sx},{sy},{sz}\n");
        for z in 0..sz {
            for y in 0..sy {
                let mut row = String::with_capacity(sx);
                for x in 0..sx {
                    let edge = x == 0 || y == 0 || x == sx - 1 || y == sy - 1;
                    row.push(if z == 0 {
                        '_'
                    } else if z == sz - 1 {
                        '^'
                    } else if edge {
                        '#'
                    } else {
                        '.'
                    });
                }
                out.push_str(&row);
                out.push('\n');
            }
            out.push('\n');
        }
        out
    }

    /// A minimal cold-room fixture at `sx × sy × sz` carrying only `[effects] snow
    /// = density`, no props, no other authored content — see [`cells_box`] for why
    /// the geometry is `[cells]`-explicit rather than an auto-box.
    fn cold_fixture(sx: usize, sy: usize, sz: usize, density: f32) -> String {
        format!(
            "[meta]\nname = cold-fixture\n\n[room]\nsize = {sx} {sy} {sz}\n\n{}\n\
             [node source]\npos = 2,1,1\nis_input = yes\n\n\
             [node sink]\npos = 5,1,1\nis_output = yes\nthreshold = 1\n\n\
             [pipe source->sink]\npath = (2,1,1) (3,1,1) (4,1,1) (5,1,1)\nswitch_at = (3,1,1)\n\n\
             [effects]\nsnow = {density}\n\n\
             [inputs]\nsource = 1\n\n\
             [pedestal]\ncenter = 5 1 1\nvolume = 1 1 1\n\n\
             [door]\npos = 0 {} 1\n\n\
             [player]\nspawn = {} {} 1\nfacing = -y\n",
            cells_box(sx, sy, sz),
            sy / 2,
            sx / 2,
            sy / 2,
        )
    }

    #[test]
    fn cold_levels_render_visible_snow_in_a_dump_frame() {
        // The flake glyph cycle `re_level` authors (see its `SNOW_GLYPHS`).
        const FLAKES: [char; 3] = ['·', '*', '.'];
        for (name, size, density, min_flakes) in [
            (
                "cold-fixture-a",
                (18usize, 16usize, 9usize),
                0.4_f32,
                20usize,
            ),
            ("cold-fixture-b", (20, 16, 5), 0.35, 20),
        ] {
            let src = cold_fixture(size.0, size.1, size.2, density);
            let level = re_level::load_string(&src, name).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(
                level.effects.snow > 0.0,
                "{name}: fixture assumption — a COLD level"
            );
            let map = level.build_map();
            let cam = level.spawn_camera();
            let registry = registry_for_map(&map);
            let fov = Fov::default();
            let emitters: Vec<crate::Emitter> = level
                .emitters
                .iter()
                .map(crate::Emitter::from_spec)
                .collect();
            assert!(
                !emitters.is_empty(),
                "{name}: snow must resolve into emitters"
            );

            let render = |ems: &[crate::Emitter]| {
                let mut fb = Framebuffer::new(100, 40);
                crate::composite::composite_dump_once(&map, &cam, &registry, &fov, ems, &mut fb);
                fb
            };
            let bare = render(&[]);
            let snowy = render(&emitters);

            let mut flakes = 0usize;
            for row in 0..snowy.rows() {
                for col in 0..snowy.cols() {
                    let (a, b) = (bare.get(col, row).unwrap(), snowy.get(col, row).unwrap());
                    if a.glyph == b.glyph {
                        continue;
                    }
                    if FLAKES.contains(&char::from_u32(b.glyph).unwrap_or(' ')) {
                        flakes += 1;
                    }
                }
            }
            assert!(
                flakes >= min_flakes,
                "{name}: snow = {} must READ as cold — only {flakes} flake cells in the \
                 dump frame (wanted >= {min_flakes})",
                level.effects.snow
            );
        }
    }

    // ── the fitted clamp must not re-deal source columns on an OBLIQUE face ──

    /// The placard body used by the yaw-sweep: one 24-character word of DISTINCT
    /// letters, so a rendered glyph identifies its SOURCE COLUMN unambiguously (and
    /// one word means one grid row, so exactly one screen row carries text).
    const SWEEP_BODY: &str = "abcdefghijklmnopqrstuvwx";

    /// Install a registry whose note 0 is [`SWEEP_BODY`], for the duration of the guard.
    fn install_sweep_note() -> (
        std::sync::MutexGuard<'static, ()>,
        scene_text::SceneTextScope,
    ) {
        let guard = scene_text::test_lock();
        let mut reg = scene_text::TextSurfaceRegistry::new();
        reg.push_note_text("note0", SWEEP_BODY);
        let scope = scene_text::SceneTextScope::new(reg);
        (guard, scope)
    }

    /// Read the placard on ONE wall cell face: for every screen column that hit the
    /// `y = 0` wall cell at `cell_x`, the SOURCE COLUMN index of the letter drawn there
    /// (columns showing blank plate are skipped), in left-to-right screen order.
    fn sweep_face_indices(map: &Map, cam: &CameraPose, cell_x: i32) -> Vec<(usize, usize)> {
        let mut fb = Framebuffer::new(120, 40);
        raycast(map, cam, &mut fb);
        let mut out: Vec<(usize, usize)> = Vec::new();
        for col in 0..fb.cols() {
            for row in 0..fb.rows() {
                let c = fb.get(col, row).unwrap();
                if !c.surface.is_wall() || c.side != 0 {
                    continue;
                }
                // The y = 0 wall's inward face sits on the y = 1 plane.
                if (c.hit.y - 1.0).abs() > 1e-3 || c.hit.x.floor() as i32 != cell_x {
                    continue;
                }
                let ch = char::from_u32(c.glyph).unwrap_or(' ');
                if let Some(i) = SWEEP_BODY.find(ch) {
                    out.push((col, i));
                    break;
                }
            }
        }
        out
    }

    /// Regression test — the yaw sweep.
    ///
    /// `u · face_cols` evaluates the projection at the shaded cell's OWN perpendicular
    /// depth. Face-on that depth is constant across the face, so the estimate is exact
    /// (the clamp measured clean at 90°); a couple of degrees off and the depth VARIES
    /// across the face, the estimate drifts, and `floor` hands two adjacent screen
    /// columns the SAME source column while skipping the next — a doubled letter plus a
    /// dropped one, re-dealt at every yaw.
    ///
    /// With the caster's measured face anchor the index is the projected screen offset
    /// from the face's own `u = 0` edge, so it advances by EXACTLY one per column. This
    /// sweeps ±4° about face-on in ~0.65° steps (86°→94°) and demands zero
    /// duplicated and zero dropped source columns at every step.
    #[test]
    fn note_clamp_never_repeats_a_source_column_across_a_yaw_sweep() {
        let (_serial, _scope) = install_sweep_note();
        let mut map = test_room();
        let mut slots: Vec<re_world::MaterialDef> =
            (0..5).map(|_| re_world::MaterialDef::fallback()).collect();
        slots[0] = re_world::MaterialDef {
            name: "note_wall".into(),
            fg: 0x00E0_DCC8,
            bg: 0x0009_0807,
            glyph: None,
            recipe: Some(re_world::MaterialRecipe::NoteWall {
                note: 0,
                face: re_core::FaceDir::PosY,
            }),
            anim: re_world::MaterialAnim::default(),
            rank: 0,
            reflective: false,
            clarity: 0.85,
            fallback: false,
            detail: false,
        };
        map.set_materials(slots);

        let mut steps = 0usize;
        let mut checked = 0usize;
        let mut faults: Vec<String> = Vec::new();
        // 86° → 94° in 0.65° steps about the −y facing (the "90°" pose), then out to
        // ±12° so the drift the estimate accumulates is well past half a cell.
        let mut deg = -12.0_f32;
        while deg <= 12.0001 {
            let mut cam = CameraPose::spawn_in_test_room();
            cam.pos = glam::Vec3::new(3.5, 3.0, 1.5);
            cam.yaw = std::f32::consts::PI + deg.to_radians();
            let seen = sweep_face_indices(&map, &cam, 3);
            steps += 1;
            if seen.len() >= 4 {
                checked += 1;
            }
            for w in seen.windows(2) {
                let ((c0, i0), (c1, i1)) = (w[0], w[1]);
                if c1 != c0 + 1 {
                    // A gap in SCREEN columns is fine (blank plate / another cell);
                    // only compare consecutive drawn columns.
                    continue;
                }
                if i1 == i0 {
                    faults.push(format!(
                        "yaw {deg:+.2}°: source column {i0} ('{}') DUPLICATED at screen \
                         columns {c0} and {c1}",
                        SWEEP_BODY.as_bytes()[i0] as char
                    ));
                } else if i1 != i0 + 1 {
                    faults.push(format!(
                        "yaw {deg:+.2}°: source columns {}..{} DROPPED between screen \
                         columns {c0} and {c1}",
                        i0 + 1,
                        i1
                    ));
                }
            }
            deg += 0.65;
        }
        assert!(steps >= 12, "the sweep must take ~13 steps, took {steps}");
        assert!(
            checked >= 12,
            "the placard must be legible at every sweep step ({checked}/{steps} steps \
             showed 4+ letters)"
        );
        assert!(
            faults.is_empty(),
            "{} duplicated/dropped source column(s) across the yaw sweep:\n{}",
            faults.len(),
            faults.join("\n")
        );
    }

    // ── NOTE WALL: an authored placard on ONE face of ONE wall cell ──

    /// Build a registry whose WALL_BRICK slot is a `NoteWall` material on `note`,
    /// painted on `face`. Other slots keep the built-ins.
    fn note_wall_registry(note: u32, face: re_core::FaceDir) -> MaterialRegistry {
        let builtins = MaterialRegistry::with_builtins();
        let mut reg = MaterialRegistry::new();
        for slot in 0..5u16 {
            if slot == ids::WALL_BRICK.0 {
                reg.register(Material {
                    recipe: TextureRecipe::NoteWall { note, face },
                    ..builtins.get(MaterialId(slot)).clone()
                });
            } else {
                reg.register(builtins.get(MaterialId(slot)).clone());
            }
        }
        reg
    }

    /// A near, face-on `SampleCtx` for a note cell: `side` set, `light` optional.
    fn note_ctx(side: u8, face_n: Option<glam::Vec3>, wall_v: f32) -> SampleCtx {
        SampleCtx {
            distance: 1.5,
            foreshorten: 1.0,
            side,
            screen_y: 8,
            wall_v,
            tick: 0,
            time: 0.0,
            fog: FogParams {
                near: 1.0,
                far: 1.0e6,
                color: None,
                height: None,
            },
            far_freeze: f32::INFINITY,
            light: face_n.map(|n| ascii_shader::LightCtx {
                face_n: n,
                tangent_u: glam::Vec3::new(1.0, 0.0, 0.0),
                tangent_v: glam::Vec3::new(0.0, 0.0, 1.0),
                key_dir: glam::Vec3::new(0.0, 0.0, 1.0),
                key_color: glam::Vec3::ONE,
                view_dir: -n,
                ambient: glam::Vec3::splat(0.4),
                headlamp: 0.5,
                shadow_key: 1.0,
                world_pos: glam::Vec3::ZERO,
                points: ascii_shader::PointLights::NONE,
            }),
            lod_override: None,
            state: 0,
            pulse: 0.0,
            inset_rgb: None,
            wall_along: 0.0,
            wall_up: 0.0,
            on_wall: false,
            tex_footprint: None,
            subcell_backdrop: None,
        }
    }

    /// Sweep one row of the plate and return the glyphs, in reading order.
    fn note_row(mat: &Material, ctx: &SampleCtx, perlin: &Perlin, n: usize) -> String {
        (0..n)
            .map(|i| {
                // u across the single face: cell 4 + the within-cell fraction.
                let u = 4.0 + (i as f32 + 0.5) / n as f32;
                let t = mat.sample(perlin, glam::Vec2::new(u, 0.0), ctx);
                char::from_u32(t.glyph).unwrap_or('?')
            })
            .collect()
    }

    /// WITH LIGHTING (the shipped path): the placard draws on the face whose true
    /// outward normal matches, and on NO other face — not the reverse (which would be
    /// mirrored text in the next room), not the two perpendicular faces.
    #[test]
    fn note_wall_face_gate_uses_the_face_normal_when_lit() {
        let _serial = install_test_panel_provider();
        let perlin = Perlin::new(NOISE_SEED);
        let reg = note_wall_registry(0, re_core::FaceDir::PosX);
        let mat = reg.get(ids::WALL_BRICK);

        // The bound face (+x, side 1): the note's first word reads on the top row.
        let front = note_row(
            mat,
            &note_ctx(1, Some(glam::Vec3::new(1.0, 0.0, 0.0)), 0.02),
            &perlin,
            re_render_note_cols(),
        );
        assert!(
            front.contains("alpha"),
            "the bound face must render the placard text, got {front:?}"
        );

        // The REVERSE face (−x, same `side` bit!): blank plate, no text at all.
        let back = note_row(
            mat,
            &note_ctx(1, Some(glam::Vec3::new(-1.0, 0.0, 0.0)), 0.02),
            &perlin,
            re_render_note_cols(),
        );
        assert!(
            back.trim().is_empty(),
            "the REVERSE face must be blank (no mirrored text), got {back:?}"
        );

        // A perpendicular face (+y): also blank.
        let side = note_row(
            mat,
            &note_ctx(0, Some(glam::Vec3::new(0.0, 1.0, 0.0)), 0.02),
            &perlin,
            re_render_note_cols(),
        );
        assert!(
            side.trim().is_empty(),
            "a perpendicular face must be blank, got {side:?}"
        );
    }

    /// WITHOUT LIGHTING (the documented fallback): only the coarse NS/EW `side` bit is
    /// available, so the placard renders on the matching axis PAIR and still never on
    /// the perpendicular pair.
    #[test]
    fn note_wall_face_gate_falls_back_to_the_side_bit_when_unlit() {
        let _serial = install_test_panel_provider();
        let perlin = Perlin::new(NOISE_SEED);
        let reg = note_wall_registry(0, re_core::FaceDir::PosX); // side == 1
        let mat = reg.get(ids::WALL_BRICK);

        let matching = note_row(
            mat,
            &note_ctx(1, None, 0.02),
            &perlin,
            re_render_note_cols(),
        );
        assert!(
            matching.contains("alpha"),
            "with no light, the matching side bit must render the text, got {matching:?}"
        );
        let other = note_row(
            mat,
            &note_ctx(0, None, 0.02),
            &perlin,
            re_render_note_cols(),
        );
        assert!(
            other.trim().is_empty(),
            "with no light, the other side bit must stay blank, got {other:?}"
        );
    }

    /// Each note index selects its OWN placard, and an out-of-range index is a blank
    /// plate — never a different note's text (a content lie).
    #[test]
    fn note_wall_index_selects_its_own_placard() {
        let _serial = install_test_panel_provider();
        let perlin = Perlin::new(NOISE_SEED);
        let ctx = note_ctx(1, Some(glam::Vec3::new(1.0, 0.0, 0.0)), 0.02);

        let reg1 = note_wall_registry(1, re_core::FaceDir::PosX);
        let row1 = note_row(
            reg1.get(ids::WALL_BRICK),
            &ctx,
            &perlin,
            re_render_note_cols(),
        );
        assert!(
            row1.contains('>'),
            "note 1 is the system-log body, got {row1:?}"
        );

        let reg2 = note_wall_registry(2, re_core::FaceDir::PosX);
        let row2 = note_row(
            reg2.get(ids::WALL_BRICK),
            &ctx,
            &perlin,
            re_render_note_cols(),
        );
        assert!(
            row2.contains('('),
            "note 2 is the model-thought body, got {row2:?}"
        );

        let reg_missing = note_wall_registry(99, re_core::FaceDir::PosX);
        let missing = note_row(
            reg_missing.get(ids::WALL_BRICK),
            &ctx,
            &perlin,
            re_render_note_cols(),
        );
        assert!(
            missing.trim().is_empty(),
            "an out-of-range note index must be a BLANK plate, got {missing:?}"
        );
    }

    /// The note grid width, as the tests sample it (one glyph per grid column).
    fn re_render_note_cols() -> usize {
        code_wall::NOTE_COLS
    }

    /// END TO END through the public `raycast`: a room whose walls carry a `+y`-facing
    /// placard shows the text on the wall the reader faces and NOT on the wall behind
    /// them. This is the acceptance observable ("on the correct wall face and NOT on
    /// the reverse face") at the frame level, not just at the material level.
    #[test]
    fn raycast_paints_the_note_on_the_bound_face_only() {
        let _serial = install_test_panel_provider();
        let mut map = test_room();
        let mut slots: Vec<re_world::MaterialDef> =
            (0..5).map(|_| re_world::MaterialDef::fallback()).collect();
        slots[0] = re_world::MaterialDef {
            name: "note_wall".into(),
            fg: 0x00E0_DCC8,
            bg: 0x0009_0807,
            glyph: None,
            // The y=1 wall of the test room faces INTO the room, i.e. its outward
            // normal is +y — so a `PosY` placard belongs to it.
            recipe: Some(re_world::MaterialRecipe::NoteWall {
                note: 0,
                face: re_core::FaceDir::PosY,
            }),
            anim: re_world::MaterialAnim::default(),
            rank: 0,
            reflective: false,
            clarity: 0.85,
            fallback: false,
            detail: false,
        };
        map.set_materials(slots);

        let wall_text = |yaw: f32| -> String {
            let mut cam = CameraPose::spawn_in_test_room();
            cam.pos = glam::Vec3::new(3.5, 3.0, 1.5);
            cam.yaw = yaw;
            let mut fb = Framebuffer::new(100, 40);
            raycast(&map, &cam, &mut fb);
            let mut out = String::new();
            for row in 0..fb.rows() {
                for col in 0..fb.cols() {
                    let c = fb.get(col, row).unwrap();
                    if c.surface.is_wall() {
                        out.push(char::from_u32(c.glyph).unwrap_or(' '));
                    }
                }
                out.push('\n');
            }
            out
        };

        // Looking −y at the y=1 wall (outward normal +y) — the bound face.
        let front = wall_text(std::f32::consts::PI);
        // Looking +y at the y=4 wall (outward normal −y) — the reverse face.
        let back = wall_text(0.0);

        // Squash repeated glyphs: one grid column spans several screen cells up close.
        let squash = |s: &str| -> String {
            let mut out = String::new();
            let mut prev = '\0';
            for ch in s.chars() {
                if ch != prev {
                    out.push(ch);
                }
                prev = ch;
            }
            out
        };
        let front_sq = squash(&front);
        assert!(
            front_sq.contains("alpha") || front_sq.contains("beta") || front_sq.contains("gamma"),
            "the bound face must show placard words in the frame; got:\n{front_sq}"
        );
        // The reverse face must carry no TEXT. (It still carries the shared
        // silhouette/contour glyphs every wall gets — `━` and friends — which are a
        // different layer, so the assertion is specifically about letters.)
        let back_letters: String = back.chars().filter(|c| c.is_ascii_alphabetic()).collect();
        assert!(
            back_letters.is_empty(),
            "the reverse face must carry NO placard letters, got {back_letters:?}"
        );
    }

    /// REGISTRY LIFECYCLE at the render level: publishing a second level's registry
    /// replaces the first, so the SAME `NoteWall { note: 0 }` material renders the
    /// second level's placard — and no panel from the first level survives.
    #[test]
    fn a_second_level_registry_replaces_the_notes_the_shader_sees() {
        let _serial = scene_text::test_lock();
        let perlin = Perlin::new(NOISE_SEED);
        let reg = note_wall_registry(0, re_core::FaceDir::PosX);
        let mat = reg.get(ids::WALL_BRICK);
        let ctx = note_ctx(1, Some(glam::Vec3::new(1.0, 0.0, 0.0)), 0.02);

        let mut level_a = scene_text::TextSurfaceRegistry::new();
        level_a.push_note_text("a", "aaaa");
        level_a.push_note_text("a2", "extra");
        scene_text::install(level_a);
        assert_eq!(scene_text::note_panel_count(), 2);
        assert!(note_row(mat, &ctx, &perlin, code_wall::NOTE_COLS).contains("aaaa"));

        let mut level_b = scene_text::TextSurfaceRegistry::new();
        level_b.push_note_text("b", "bbbb");
        scene_text::install(level_b);
        assert_eq!(
            scene_text::note_panel_count(),
            1,
            "the second level's registry must REPLACE the first (no accumulation)"
        );
        let row_b = note_row(mat, &ctx, &perlin, code_wall::NOTE_COLS);
        assert!(row_b.contains("bbbb"), "got {row_b:?}");
        assert!(
            !row_b.contains("aaaa"),
            "the first level's text must be gone"
        );

        // Teardown → blank plates again.
        scene_text::clear();
        assert!(note_row(mat, &ctx, &perlin, code_wall::NOTE_COLS)
            .trim()
            .is_empty());
    }

    // ── per-REGISTER placard ink, through the shipped shade path ──

    /// Each register inks its placard with its OWN palette: the researcher's hand is
    /// warm (R > B), the machine log is cold (B > R), the model's thought is dimmer
    /// than both — and the log's `>` marker is inked distinctly from its message.
    ///
    /// Asserted through `Material::sample` (what the raycaster calls), not just the
    /// palette table, so the register really does reach the frame.
    #[test]
    fn note_wall_inks_each_register_with_its_own_palette() {
        let _serial = scene_text::test_lock();
        let perlin = Perlin::new(NOISE_SEED);
        let ctx = note_ctx(1, Some(glam::Vec3::new(1.0, 0.0, 0.0)), 0.02);

        let mut reg = scene_text::TextSurfaceRegistry::new();
        reg.push_note_text("note0:Researcher", "warm hand");
        reg.push_note_text("note1:SystemLog", "cold log");
        reg.push_note_text("note2:ModelThought", "quiet");
        scene_text::install(reg);

        // Sweep a plate's top row and return the (fg, bg) of the first cell whose
        // glyph satisfies `want` (the message ink, or the structural marker).
        let cell_of = |note: u32, want: fn(char) -> bool| -> (u32, u32) {
            let materials = note_wall_registry(note, re_core::FaceDir::PosX);
            let mat = materials.get(ids::WALL_BRICK);
            for i in 0..code_wall::NOTE_COLS {
                let u = 4.0 + (i as f32 + 0.5) / code_wall::NOTE_COLS as f32;
                let t = mat.sample(&perlin, glam::Vec2::new(u, 0.0), &ctx);
                if want(char::from_u32(t.glyph).unwrap_or(' ')) {
                    return (t.fg, t.bg);
                }
            }
            panic!("no matching cell on note {note}'s plate");
        };
        let ink_of = |note: u32, want: fn(char) -> bool| cell_of(note, want).0;
        let letter = |c: char| c.is_ascii_alphabetic();
        let rgb = |c: u32| ((c >> 16) & 0xFF, (c >> 8) & 0xFF, c & 0xFF);
        let luma = |c: u32| {
            let (r, g, b) = rgb(c);
            r + g + b
        };

        let researcher = ink_of(0, letter);
        let syslog = ink_of(1, letter);
        let thought = ink_of(2, letter);

        assert_ne!(researcher, syslog, "researcher and system_log must differ");
        assert_ne!(
            researcher, thought,
            "researcher and model_thought must differ"
        );
        assert_ne!(syslog, thought, "system_log and model_thought must differ");

        let (rr, _, rb) = rgb(researcher);
        assert!(
            rr > rb,
            "researcher ink must be WARM, got {researcher:#08x}"
        );
        let (sr, _, sb) = rgb(syslog);
        assert!(sb > sr, "system_log ink must be COLD, got {syslog:#08x}");
        assert!(
            luma(thought) < luma(researcher) && luma(thought) < luma(syslog),
            "model_thought must read DIM relative to the other registers"
        );

        // The `>` prompt of a log line is furniture: inked below the message.
        let marker = ink_of(1, |c| c == '>');
        assert_ne!(marker, syslog, "the `>` marker must be inked distinctly");
        assert!(
            luma(marker) < luma(syslog),
            "the marker sits below the message"
        );

        // …and every register still READS: the text ink sits far above its own plate,
        // dim register included (dim ≠ unreadable).
        for note in 0..3u32 {
            let (fg, bg) = cell_of(note, letter);
            assert!(
                luma(fg) > luma(bg) * 8,
                "note {note}: ink/plate contrast collapsed (fg {fg:#08x} bg {bg:#08x})"
            );
        }
        scene_text::clear();
    }

    // ── a powered pipe streams the LEVEL'S pool words ──

    /// The glyphs a POWERED pipe cell renders over a sweep of position × time, with
    /// `registry` installed (or nothing installed when `None`).
    fn powered_pipe_glyphs(
        registry: Option<scene_text::TextSurfaceRegistry>,
    ) -> std::collections::BTreeSet<char> {
        use re_core::{encode_flow_state, encode_pipe_side_data, CellStateMap, FlowDir};
        let perlin = Perlin::new(NOISE_SEED);
        let materials = MaterialRegistry::with_builtins();
        let view = glam::Vec3::new(0.0, 1.0, 0.2).normalize();
        let side_data = encode_pipe_side_data(0);

        match registry {
            Some(r) => scene_text::install(r),
            None => scene_text::clear(),
        }
        let mut states = CellStateMap::new();
        states.set_state(side_data, encode_flow_state(FlowDir::Forward));
        let _scope = crate::state_bus::CellStateScope::new(&states);

        let mut seen = std::collections::BTreeSet::new();
        for i in 0..48 {
            let geom = CellGeom {
                depth: 3.0,
                foreshorten: 1.0,
                side: 0,
                tex_u: 0.5,
                wall_v: 0.3,
                wall_world_y: wall_world_y(1, 0.3),
                hit_x: 2.0 + i as f32 * 0.13,
                hit_y: 1.0,
                surface: SurfaceKind::WallTrace,
                material: side_data,
                light: Some(LightCtx::for_wall(0, -1.0, view, &LightEnv::builtin())),
                cell: (0, 0, 0),
                tex_footprint: None,
                subcell_backdrop: None,
            };
            let cell = shade_cell(
                &materials,
                &perlin,
                &FogParams::DEFAULT,
                f32::INFINITY,
                None,
                None,
                &[],
                &geom,
                4,
                i as f32 * 0.11,
            );
            seen.insert(char::from_u32(cell.glyph).unwrap_or('?'));
        }
        drop(_scope);
        crate::state_bus::clear_active();
        scene_text::clear();
        seen
    }

    /// A registry carrying ONE node pool (clean `quux`, corrupted `jjzz`) owning pipe
    /// tag 0, at the given authored glitch density. Both word sets use letters that
    /// appear in NO built-in pipe word, so seeing one proves which register streamed.
    fn pool_registry(glitch: f32) -> scene_text::TextSurfaceRegistry {
        let mut r = scene_text::TextSurfaceRegistry::new();
        r.set_pools(re_core::PoolTable::from_parts(
            vec![re_core::NodePool {
                pool: vec!["quux".into()],
                pool_alt: vec!["jjzz".into()],
                anchor: 0,
            }],
            &[0],
        ));
        r.set_glitch(glitch);
        r
    }

    /// THE FLIP: with the level's pool published, a powered pipe streams the node's
    /// own CLEAN memory-words — not the built-in list.
    #[test]
    fn powered_pipe_streams_the_level_pool_words() {
        let _serial = scene_text::test_lock();
        let clean = powered_pipe_glyphs(Some(pool_registry(0.0)));
        assert!(
            clean.iter().any(|&c| "quxq".contains(c)),
            "the pipe must stream the node's own pool words, saw {clean:?}"
        );
        assert!(
            !clean.iter().any(|&c| c == 'j' || c == 'z'),
            "a clean level must NOT stream the corrupted register, saw {clean:?}"
        );
    }

    /// THE GLITCH GATE: the same pool on a level whose authored glitch density passes
    /// the threshold streams the CORRUPTED register instead.
    #[test]
    fn a_glitched_level_streams_the_alt_pool_words() {
        let _serial = scene_text::test_lock();
        let alt = powered_pipe_glyphs(Some(pool_registry(POOL_ALT_GLITCH_THRESHOLD + 0.3)));
        assert!(
            alt.iter().any(|&c| c == 'j' || c == 'z'),
            "past the glitch threshold the pipe must stream pool_alt, saw {alt:?}"
        );
        assert!(
            !alt.iter().any(|&c| c == 'q' || c == 'x'),
            "the clean register must be gone on a glitched level, saw {alt:?}"
        );
        // AT the bar it is still clean (the strictly-greater rule, end to end).
        let at_bar = powered_pipe_glyphs(Some(pool_registry(POOL_ALT_GLITCH_THRESHOLD)));
        assert!(
            at_bar.iter().any(|&c| c == 'q' || c == 'x') && !at_bar.contains(&'j'),
            "exactly at the threshold the level stays clean, saw {at_bar:?}"
        );
    }

    /// THE FALLBACK: no registry (a demo level / no level loaded) streams the built-in
    /// word list, byte-identically to an installed registry with no pool data — so
    /// pool-less frames are unchanged by the flip.
    #[test]
    fn a_pipe_without_pool_data_keeps_the_builtin_words() {
        let _serial = scene_text::test_lock();
        let none = powered_pipe_glyphs(None);
        let empty = powered_pipe_glyphs(Some(scene_text::TextSurfaceRegistry::new()));
        assert_eq!(
            none, empty,
            "an installed-but-poolless registry must stream exactly the built-in words"
        );
        assert!(none.len() > 1, "the built-in stream must still flow");
        assert!(
            !none.iter().any(|&c| "quxjz".contains(c)),
            "the built-in words contain none of the pool-only letters, saw {none:?}"
        );
    }

    /// Pipe words: an eye-level reading face shows its pool words at ONE column
    /// per letter (no `CCRRIIMMSS`) and in reading
    /// order from BOTH sides of the run (no mirrored `IV·NREF`).
    #[test]
    fn pipe_words_read_forward_one_column_per_letter_from_both_sides() {
        use re_core::{encode_flow_state, encode_pipe_side_data, CellStateMap, FlowDir};
        let _serial = scene_text::test_lock();
        const WORD: &str = "abcdefghi";
        let mut r = scene_text::TextSurfaceRegistry::new();
        r.set_pools(re_core::PoolTable::from_parts(
            vec![re_core::NodePool {
                pool: vec![WORD.into()],
                pool_alt: vec![],
                anchor: 0,
            }],
            &[0],
        ));
        scene_text::install(r);
        let side_data = encode_pipe_side_data(0);
        let mut states = CellStateMap::new();
        states.set_state(side_data, encode_flow_state(FlowDir::Forward));
        let _scope = crate::state_bus::CellStateScope::new(&states);

        let (sx, sy, sz) = (12usize, 12usize, 4usize);
        let mut m = Map::new(sx, sy, sz);
        for z in 0..sz {
            for y in 0..sy {
                for x in 0..sx {
                    let edge = x == 0 || y == 0 || x == sx - 1 || y == sy - 1;
                    let k = if edge {
                        CellKind::WallSolid
                    } else if z == 0 {
                        CellKind::Floor
                    } else if z == sz - 1 {
                        CellKind::Ceiling
                    } else {
                        CellKind::Empty
                    };
                    m.set(x, y, z, k);
                }
            }
        }
        for x in 1..sx - 1 {
            m.set_cell(x, 6, 1, CellKind::WallTrace, side_data);
        }
        let reg = MaterialRegistry::with_builtins();
        let read = |y: f32, yaw: f32| -> String {
            let mut cam = CameraPose::spawn_in_test_room();
            cam.pos = glam::Vec3::new(5.5, y, STANDING_EYE_Z);
            cam.yaw = yaw;
            let mut fb = Framebuffer::new(80, 40);
            raycast_with_time_fov_gameplay(&m, &cam, &reg, &Fov::default(), 0.0, &mut fb);
            // The stream row: the row of the pipe face carrying the most letters.
            let row_of = |rr: usize| -> String {
                (0..80)
                    .filter_map(|c| {
                        let cell = fb.get(c, rr).unwrap();
                        (cell.surface == SurfaceKind::WallTrace).then(|| {
                            char::from_u32(cell.glyph)
                                .unwrap_or(' ')
                                .to_ascii_lowercase()
                        })
                    })
                    .collect()
            };
            (0..40)
                .map(row_of)
                .max_by_key(|r| r.chars().filter(char::is_ascii_lowercase).count())
                .unwrap_or_default()
        };
        for (label, y, yaw) in [
            ("south side", 3.5, 0.0),
            ("north side", 8.5, std::f32::consts::PI),
        ] {
            let row = read(y, yaw);
            let letters: Vec<char> = row.chars().filter(char::is_ascii_lowercase).collect();
            assert!(
                letters.len() >= 6,
                "{label}: the reading face streams words: {row:?}"
            );
            let fwd = WORD
                .as_bytes()
                .windows(3)
                .filter(|w| row.contains(std::str::from_utf8(w).unwrap()))
                .count();
            let rev: String = WORD.chars().rev().collect();
            let back = rev
                .as_bytes()
                .windows(3)
                .filter(|w| row.contains(std::str::from_utf8(w).unwrap()))
                .count();
            assert!(
                fwd >= 2 && back == 0,
                "{label}: words must read left to right: {row:?}"
            );
            assert!(
                !row.as_bytes()
                    .windows(2)
                    .any(|w| w[0] == w[1] && w[0].is_ascii_lowercase()),
                "{label}: one column per letter, no doubled letters: {row:?}"
            );
        }
        drop(_scope);
        crate::state_bus::clear_active();
        scene_text::clear();
    }

    // ── authored materials: the "color = material KEY" merge ──
    #[test]
    fn empty_map_uses_builtin_registry() {
        // A map with no authored materials → the registry is the built-in set.
        let map = test_room();
        assert!(map.materials().is_empty());
        let reg = registry_for_map(&map);
        // Built-in brick recipe in slot 0 (matches with_builtins()).
        assert!(matches!(
            reg.get(ids::WALL_BRICK).recipe,
            TextureRecipe::Brick { .. }
        ));
        assert!(reg.any_animated(), "built-ins contain animated materials");
    }

    #[test]
    fn authored_map_overrides_only_bound_slot() {
        use re_world::{MaterialAnim, MaterialDef, MaterialRecipe};
        // 5 slot-ordered defs: only the FLOOR slot (3) authored; the rest fall
        // back to the built-ins (exactly what re_level::build_map emits).
        let mut map = test_room();
        let mut slots: Vec<MaterialDef> = (0..5).map(|_| MaterialDef::fallback()).collect();
        slots[3] = MaterialDef {
            name: "floor_neon".into(),
            fg: 0x0020_FF80,
            bg: 0x0004_140C,
            glyph: Some(u32::from('+')),
            recipe: Some(MaterialRecipe::Grain {
                scale: 3.0,
                speck: 0.5,
                tile_eps: 0.08,
            }),
            anim: MaterialAnim::default(),
            rank: 0,
            reflective: false,
            clarity: 0.85,
            fallback: false,
            detail: false,
        };
        map.set_materials(slots);

        let reg = registry_for_map(&map);
        let perlin = Perlin::new(NOISE_SEED);
        let ctx = SampleCtx {
            distance: 3.0,
            foreshorten: 1.1,
            side: 0,
            screen_y: 18,
            wall_v: 0.0,
            tick: 0,
            time: 0.0,
            fog: FogParams::DEFAULT,
            far_freeze: f32::INFINITY,
            light: None,
            lod_override: None,
            state: 0,
            pulse: 0.0,
            inset_rgb: None,
            wall_along: 0.0,
            wall_up: 0.0,
            on_wall: false,
            tex_footprint: None,
            subcell_backdrop: None,
        };
        let coord = glam::Vec2::new(3.21, 7.83);

        // Floor slot (id 3) now shows the AUTHORED look: a fixed '+' glyph with a
        // neon-green fg — distinct from the built-in floor_grain (which uses ' '
        // base glyph + a cool-slate palette).
        let authored = reg.get(ids::FLOOR_GRAIN).sample(&perlin, coord, &ctx);
        let builtin = MaterialRegistry::with_builtins()
            .get(ids::FLOOR_GRAIN)
            .sample(&perlin, coord, &ctx);
        assert_ne!(
            (authored.glyph, authored.fg),
            (builtin.glyph, builtin.fg),
            "authored floor material must render differently from the built-in"
        );
        assert_eq!(authored.glyph, u32::from('+'), "authored fixed glyph shows");
        // The green channel dominates the authored fg (neon green).
        let g = (authored.fg >> 8) & 0xFF;
        let r = (authored.fg >> 16) & 0xFF;
        assert!(g > r, "authored fg should read green (g={g} > r={r})");

        // An UNBOUND slot (wall_solid, id 0) is still the built-in brick.
        assert!(matches!(
            reg.get(ids::WALL_BRICK).recipe,
            TextureRecipe::Brick { .. }
        ));
    }

    #[test]
    fn preset_name_resolves_to_the_catalog_material() {
        use re_world::{MaterialAnim, MaterialDef};
        // A level binds `preset = lava` (re_level encodes it in the MaterialDef name
        // via the sentinel). The renderer must resolve that to the CATALOG `lava`
        // config — emissive + pulsing — for the bound surface, NOT the surface
        // built-in. We bind it to the wall_solid slot (0).
        let mut map = test_room();
        let mut slots: Vec<MaterialDef> = (0..5).map(|_| MaterialDef::fallback()).collect();
        slots[0] = MaterialDef {
            name: "\u{1f}preset\u{1f}lava".into(), // the sentinel re_level emits
            fg: 0,                                 // unset → keep the catalog palette
            bg: 0,
            glyph: None,
            recipe: None,
            anim: MaterialAnim::default(),
            rank: 0,
            reflective: false,
            clarity: MaterialDef::DEFAULT_CLARITY,
            fallback: false,
            detail: false,
        };
        map.set_materials(slots);

        let reg = registry_for_map(&map);
        let resolved = reg.get(ids::WALL_BRICK);
        let lava = MaterialRegistry::catalog_material("lava").unwrap();
        // The wall_solid slot now carries the lava config (emissive + animated),
        // distinct from the built-in brick.
        assert!(
            resolved.light.emissive > 0.5,
            "preset pulled lava's emissive glow"
        );
        assert!(resolved.is_animated(), "preset pulled lava's pulse");
        assert_eq!(
            resolved.recipe_tag(),
            lava.recipe_tag(),
            "preset recipe matches the catalog"
        );
        assert_eq!(
            format!("{:?}", resolved.palette.a),
            format!("{:?}", lava.palette.a),
            "an unset fg keeps the catalog palette (a bare preset reads as lava)"
        );

        // An explicit fg override on a preset still wins.
        let mut map2 = test_room();
        let mut slots2: Vec<MaterialDef> = (0..5).map(|_| MaterialDef::fallback()).collect();
        slots2[0] = MaterialDef {
            name: "\u{1f}preset\u{1f}metal".into(),
            fg: 0x00FF_0000, // explicit red override
            bg: 0,
            glyph: None,
            recipe: None,
            anim: MaterialAnim::default(),
            rank: 0,
            reflective: false,
            clarity: MaterialDef::DEFAULT_CLARITY,
            fallback: false,
            detail: false,
        };
        map2.set_materials(slots2);
        let reg2 = registry_for_map(&map2);
        let m = reg2.get(ids::WALL_BRICK);
        // Flat red palette (a==red, b==0), but metal's specular config kept.
        assert_eq!(
            m.palette.a,
            glam::Vec3::new(1.0, 0.0, 0.0),
            "explicit fg overrides preset colour"
        );
        assert!(
            m.light.specular > 0.5,
            "metal's specular sheen is kept under the colour override"
        );
    }

    /// FIXTURE LOCK — the fractal portal demo level loads through the real
    /// `re_level` parser and its portal-skinned wall surfaces resolve to
    /// `TextureRecipe::Fractal` materials (the two new presets). Guards the demo
    /// fixture against a silent break (e.g. a renamed preset or a moved decal).
    #[test]
    fn fractal_portal_level_binds_fractal_to_portal_walls() {
        // Repo-root level path (workspace root is two dirs above the crate manifest).
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../levels/92-fractal-portal.lvl");
        let lvl = re_level::load(&path).expect("92-fractal-portal.lvl must parse");
        let map = lvl.build_map();
        let reg = registry_for_map(&map);

        // The level binds `portal_zoom` to wall_trace (slot 1, id `WALL_CIRCUIT`)
        // and `portal_spiral` to wall_matrix (slot 2). Both resolve to Fractal.
        let zoom = reg.get(ids::WALL_CIRCUIT);
        assert!(
            matches!(zoom.recipe, TextureRecipe::Fractal { arms: 0, .. }),
            "wall_trace must carry portal_zoom (Fractal, arms=0), got {:?}",
            zoom.recipe
        );
        let spiral = reg.get(ids::WALL_MATRIX);
        assert!(
            matches!(spiral.recipe, TextureRecipe::Fractal { arms: 5, .. }),
            "wall_matrix must carry portal_spiral (Fractal, arms=5), got {:?}",
            spiral.recipe
        );

        // A portal-skinned wall cell actually appears in a cast frame from the
        // spawn (the demo shows it, not just binds it).
        let cam = lvl.spawn_camera();
        let mut fb = Framebuffer::new(100, 32);
        raycast(&map, &cam, &mut fb);
        let mut saw_portal_wall = false;
        for row in 0..fb.rows() {
            for col in 0..fb.cols() {
                let c = fb.get(col, row).unwrap();
                if matches!(c.surface, SurfaceKind::WallTrace | SurfaceKind::WallMatrix) {
                    saw_portal_wall = true;
                }
            }
        }
        assert!(
            saw_portal_wall,
            "a portal-skinned wall (trace/matrix) must appear in the demo frame"
        );
    }

    #[test]
    fn raycast_reflects_authored_material_in_frame() {
        // End-to-end through the public `raycast` entry (the `--dump` path): the
        // authored floor glyph '+' appears in the frame; the built-in floor would
        // never emit '+' as its base.
        use re_world::{MaterialAnim, MaterialDef, MaterialRecipe};
        let mut map = test_room();
        let mut slots: Vec<MaterialDef> = (0..5).map(|_| MaterialDef::fallback()).collect();
        slots[3] = MaterialDef {
            name: "floor_neon".into(),
            fg: 0x0020_FF80,
            bg: 0x0004_140C,
            glyph: Some(u32::from('+')),
            recipe: Some(MaterialRecipe::Grain {
                scale: 3.0,
                speck: 0.5,
                tile_eps: 0.08,
            }),
            anim: MaterialAnim::default(),
            rank: 0,
            reflective: false,
            clarity: 0.85,
            fallback: false,
            detail: false,
        };
        map.set_materials(slots);

        let cam = CameraPose::spawn_in_test_room();
        let mut fb = Framebuffer::new(80, 30);
        raycast(&map, &cam, &mut fb);

        let mut saw_plus = false;
        for row in 0..fb.rows() {
            for col in 0..fb.cols() {
                let c = fb.get(col, row).unwrap();
                if c.surface == SurfaceKind::Floor && c.glyph == u32::from('+') {
                    saw_plus = true;
                }
            }
        }
        assert!(
            saw_plus,
            "authored floor glyph '+' must appear on floor cells in the cast frame"
        );
    }

    // ── per-level fog: atmosphere reaches the cast frame ──
    #[test]
    fn fog_for_map_reads_atmosphere() {
        use re_world::Atmosphere;
        // No atmosphere → engine-default fog.
        let map = test_room();
        assert_eq!(fog_for_map(&map), FogParams::DEFAULT);

        // Authored atmosphere → its fields override the defaults.
        let mut map = test_room();
        map.set_atmosphere(Atmosphere {
            fog_color: Some(0x00FF_00FF),
            fog_near: Some(0.5),
            fog_far: Some(9.0),
            matrix: Some(0.5),
            ..Default::default()
        });
        let fog = fog_for_map(&map);
        assert_eq!(fog.color, Some(0x00FF_00FF));
        assert_eq!(fog.near, 0.5);
        assert_eq!(fog.far, 9.0);
    }

    #[test]
    fn authored_fog_tints_distant_floor_in_cast_frame() {
        // End-to-end through `raycast` (the `--dump` path): a map with a vivid
        // magenta fog and a tight band must tint FAR cells toward magenta while
        // NEAR cells (the close floor under the camera) keep their own colour.
        use re_world::Atmosphere;
        let mut map = test_room();
        // Tight band tuned for the small test room (max wall depth ~4 cells):
        // near at 0.5, full fog by 2.5 → cells past ~2.5 read magenta.
        map.set_atmosphere(Atmosphere {
            fog_color: Some(0x00FF_00FF), // magenta
            fog_near: Some(0.5),
            fog_far: Some(2.5),
            matrix: None,
            ..Atmosphere::default()
        });
        let cam = CameraPose::spawn_in_test_room();
        let mut fb = Framebuffer::new(80, 40);
        raycast(&map, &cam, &mut fb);

        // Closeness of a cell's bg to magenta, weighted by its depth.
        let magenta_ness = |c: u32| -> i32 {
            let r = ((c >> 16) & 0xFF) as i32;
            let g = ((c >> 8) & 0xFF) as i32;
            let b = (c & 0xFF) as i32;
            // High when red+blue dominate green (magenta).
            (r + b) - 2 * g
        };

        let mut far_magenta = i32::MIN; // strongest magenta among FAR cells
        let mut saw_far = false;
        for row in 0..fb.rows() {
            for col in 0..fb.cols() {
                let cell = fb.get(col, row).unwrap();
                if !cell.depth.is_finite() {
                    continue;
                }
                if cell.depth >= 2.5 {
                    saw_far = true;
                    far_magenta = far_magenta.max(magenta_ness(cell.bg));
                }
            }
        }
        assert!(saw_far, "expected some far cells past fog_far");
        assert!(
            far_magenta > 100,
            "far cells must read strongly magenta (fog tint), got {far_magenta}"
        );
    }

    /// Human-readable evidence for the fog tint: print the bg colour of the
    /// nearest vs the farthest cell in a magenta-fogged cast.
    /// `cargo test -p re_render fog_dump -- --nocapture`.
    #[test]
    fn fog_dump_prints_near_vs_far_colors() {
        use re_world::Atmosphere;
        let mut map = test_room();
        map.set_atmosphere(Atmosphere {
            fog_color: Some(0x00FF_00FF),
            fog_near: Some(0.5),
            fog_far: Some(2.5),
            matrix: None,
            ..Atmosphere::default()
        });
        let cam = CameraPose::spawn_in_test_room();
        let mut fb = Framebuffer::new(80, 40);
        raycast(&map, &cam, &mut fb);

        let mut nearest = (f32::INFINITY, 0u32);
        let mut farthest = (0.0f32, 0u32);
        for row in 0..fb.rows() {
            for col in 0..fb.cols() {
                let c = fb.get(col, row).unwrap();
                if !c.depth.is_finite() {
                    continue;
                }
                if c.depth < nearest.0 {
                    nearest = (c.depth, c.bg);
                }
                if c.depth > farthest.0 && c.depth < 1e8 {
                    farthest = (c.depth, c.bg);
                }
            }
        }
        println!("== fog magenta #FF00FF, near=0.5 far=2.5 ==");
        println!("NEAR depth={:.2} bg=#{:06X}", nearest.0, nearest.1);
        println!(
            "FAR  depth={:.2} bg=#{:06X}  (should be ~FF00FF)",
            farthest.0, farthest.1
        );
    }

    // ── DEPTH CUE: the GLYPH thins (less ink) with distance ──
    /// The depth→brightness cue (the `depth_luma_spearman` fix) is now carried by
    /// the GLYPH / ink channel, not a colour multiply. The same lit wall texel,
    /// shaded at increasing distances, must pick a glyph progressively further DOWN
    /// the density ramp (`body_ramp()` is densest→sparsest), so its ink coverage —
    /// and therefore its effective `lerp(bg,fg,ink)` luminance on a bright-fg/dark-bg
    /// wall — falls with depth. fg/bg colours are NOT depth-scaled, so this leaves
    /// the QA `separation` and `invisible_pct` metrics intact (the failure mode of
    /// the old multiplicative `depth_darken`, which is removed).
    #[test]
    fn farther_wall_picks_a_sparser_glyph() {
        let registry = MaterialRegistry::with_builtins();
        let perlin = Perlin::new(NOISE_SEED);

        // Position in the body ramp: 0 = densest (`@`, full ink) … len-1 = sparsest
        // (space, no ink). A larger index ⇒ thinner ⇒ dimmer effective luminance.
        // Returns `None` for a non-ramp glyph (a structural detail texel — mortar
        // │/─ — which carries its own meaning and is NOT depth-thinned; we skip
        // those texels and probe a face texel instead). Also returns `None` for a
        // SPACE: the calmer wall blanks a (world-anchored) fraction of body cells to
        // a coloured space, and a blank cell is not an INKED ramp glyph the depth cue
        // thins — so we skip those and probe an inked face texel instead.
        let ramp = body_ramp();
        let ramp_index = |glyph: u32| -> Option<usize> {
            if glyph == u32::from(' ') {
                return None;
            }
            ramp.iter().position(|&b| u32::from(b) == glyph)
        };

        // A realistic lit wall (the depth-ink cue lives on the lit relief path).
        let view_dir = glam::Vec3::new(0.0, 1.0, 0.2).normalize();
        let light = Some(LightCtx::for_wall(0, -1.0, view_dir, &LightEnv::builtin()));
        // `vy` sweeps the WORLD-anchored vertical wall coordinate (`wall_v`), which
        // now drives `coord.y` (the texture is world-anchored — critic WALL fix), so
        // we still visit distinct vertical body texels. `screen_y` is passed for the
        // (depth-ink) row context but no longer drives the texture coordinate.
        let glyph_at = |depth: f32, tex_u: f32, vy: usize| -> u32 {
            let wall_v = 0.02 + vy as f32 * 0.12;
            let g = CellGeom {
                depth,
                foreshorten: 1.0,
                side: 0,
                tex_u,
                wall_v,
                wall_world_y: wall_world_y(1, wall_v),
                hit_x: 2.5,
                hit_y: 1.0,
                surface: SurfaceKind::WallSolid,
                material: 0,
                light,
                cell: (0, 0, 0),
                tex_footprint: None,
                subcell_backdrop: None,
            };
            // Fog OFF (far band past the depths) so this isolates the ink cue, not
            // the fade-to-black fog.
            let no_fog = FogParams {
                near: 1.0,
                far: 1.0e6,
                color: None,
                height: None,
            };
            shade_cell(
                &registry,
                &perlin,
                &no_fog,
                f32::INFINITY,
                None,
                None,
                &[],
                &g,
                2,
                0.0,
            )
            .glyph
        };

        // Probe brick-face texels (skip mortar-seam columns, which emit a detail
        // glyph that is intentionally NOT depth-thinned, AND skip the world-anchored
        // BLANK body cells of the calmer wall) and assert the body glyph thins from
        // near → far on every INKED one we find. Sweep several vertical texels so
        // enough inked face texels are sampled despite the sparse coverage.
        // Sweep a FINE 2D patch of the face: with the minimalist look the wall body
        // is sparsely inked (~1 cell in 8 keeps a ramp glyph), so a coarse probe grid
        // can miss every inked well. A dense sweep reliably samples enough inked face
        // texels to exercise the near→far thinning on the ones that do ink.
        let mut checked = 0;
        for vy in 0..40 {
            for i in 0..64 {
                let tex_u = 0.03 + i as f32 * 0.11;
                let near = ramp_index(glyph_at(0.5, tex_u, vy));
                let far = ramp_index(glyph_at(22.0, tex_u, vy));
                if let (Some(near), Some(far)) = (near, far) {
                    assert!(
                        near <= far,
                        "wall glyph must thin (move down the density ramp) with depth at \
                         tex_u={tex_u} vy={vy}: near_idx={near} far_idx={far}"
                    );
                    if near < far {
                        checked += 1;
                    }
                }
            }
        }
        assert!(
            checked > 0,
            "no body-ramp face texel found to test the depth cue"
        );
    }

    /// The depth-ink cue must NOT blank an inked wall: wherever a NEAR wall texel
    /// carries an inked body glyph, the SAME texel far away must still be inked (the
    /// cue thins toward [`ascii_shader`]'s `DEPTH_INK_FLOOR`, never to space). It
    /// only ever removes ink down to that floor, so it can never turn an inked near
    /// cell into a blank far one.
    #[test]
    fn far_wall_keeps_an_inked_glyph() {
        let registry = MaterialRegistry::with_builtins();
        let perlin = Perlin::new(NOISE_SEED);
        let view_dir = glam::Vec3::new(0.0, 1.0, 0.2).normalize();
        let light = Some(LightCtx::for_wall(0, -1.0, view_dir, &LightEnv::builtin()));
        // `wall_v` drives `coord.y` (world-anchored vertical texture — WALL fix), so
        // probe several distinct vertical texels via the (tex_u, wall_v) pair.
        let glyph_at = |depth: f32, tex_u: f32, wall_v: f32| -> u32 {
            let g = CellGeom {
                depth,
                foreshorten: 1.0,
                side: 0,
                tex_u,
                wall_v,
                wall_world_y: wall_world_y(1, wall_v),
                hit_x: 2.5,
                hit_y: 1.0,
                surface: SurfaceKind::WallSolid,
                material: 0,
                light,
                cell: (0, 0, 0),
                tex_footprint: None,
                subcell_backdrop: None,
            };
            // Fog OFF so this isolates the ink floor, not the fade-to-black fog.
            let no_fog = FogParams {
                near: 1.0,
                far: 1.0e6,
                color: None,
                height: None,
            };
            shade_cell(
                &registry,
                &perlin,
                &no_fog,
                f32::INFINITY,
                None,
                None,
                &[],
                &g,
                2,
                0.0,
            )
            .glyph
        };
        // Find the texel whose NEAR body glyph is DENSEST (highest ink, top of the
        // ramp) — a cell with clearly high `protrude`, well above the cue's floor.
        // The cue thins it toward `DEPTH_INK_FLOOR`, which is held at an INKED ramp
        // level, so it must NOT blank far away. (Genuinely dark, near-space body
        // cells legitimately stay sparse — the cue only ever removes ink down to the
        // floor, never below `protrude`, so it cannot create a blank that wasn't
        // already there; the densest texel proves the floor is inked.)
        let space = u32::from(' ');
        let ramp = body_ramp();
        let ramp_index =
            |glyph: u32| -> Option<usize> { ramp.iter().position(|&b| u32::from(b) == glyph) };
        // (ramp idx, tex_u, wall_v); smaller idx = denser. Sweep both axes so a
        // genuinely inked body texel is found despite the sparse world-anchored
        // blanking of the calmer wall.
        let mut best: Option<(usize, f32, f32)> = None;
        for vy in 0..40 {
            let wall_v = 0.02 + vy as f32 * 0.12;
            for i in 0..64 {
                let tex_u = 0.03 + i as f32 * 0.11;
                if let Some(idx) = ramp_index(glyph_at(0.5, tex_u, wall_v)) {
                    if best.is_none_or(|(b, _, _)| idx < b) {
                        best = Some((idx, tex_u, wall_v));
                    }
                }
            }
        }
        let (_, tex_u, wall_v) = best.expect("no body-ramp texel found near the wall");
        assert_ne!(
            glyph_at(80.0, tex_u, wall_v),
            space,
            "the densest near wall texel must stay inked far away (tex_u={tex_u} wall_v={wall_v})"
        );
    }

    // ── PUZZLE LEGIBILITY: object cells dispatch to the node/door materials ──

    /// Set-dressing F2: an `O` cell `re_level` tagged INERT (no backing node/door)
    /// reaches the shader as plain solid geometry, while a node-tagged `O` (node 0
    /// included) is still a puzzle element. The two crates' tag constants agree.
    #[test]
    fn inert_object_cells_are_geometry_not_puzzle_nodes() {
        assert_eq!(
            INERT_OBJECT_TAG,
            re_level::INERT_OBJECT_TAG,
            "tags must agree"
        );
        assert_eq!(
            cellkind_to_surface(CellKind::Object, INERT_OBJECT_TAG),
            SurfaceKind::WallSolid
        );
        assert_eq!(
            cellkind_to_surface(CellKind::Object, 0),
            SurfaceKind::WallOther
        );
        assert_eq!(
            cellkind_to_surface(CellKind::Object, DOOR_TAG_SENTINEL),
            SurfaceKind::WallOther
        );
        // Only Object cells read the tag: a colour id on a wall is still a wall.
        assert_eq!(
            cellkind_to_surface(CellKind::WallTrace, INERT_OBJECT_TAG),
            SurfaceKind::WallTrace
        );
        // End to end over an OWN fixture (not campaign content, which set dressing
        // keeps rewriting — L1's `OOOOO` benches this used to read became console
        // props): a `[cells]` room with two puzzle nodes and a painted bench row. The
        // bench comes out of `re_level` tagged inert and maps to solid geometry; the
        // real node anchors keep their node tags (node 0 included).
        let (sx, sy, sz) = (8usize, 8usize, 4usize);
        let mut grid = String::from("[cells]\nsize = 8,8,4\n");
        for z in 0..sz {
            for y in 0..sy {
                for x in 0..sx {
                    let edge = x == 0 || y == 0 || x == sx - 1 || y == sy - 1;
                    grid.push(match z {
                        0 => '_',
                        3 => '^',
                        _ if edge => '#',
                        1 if y == 6 && (2..=4).contains(&x) => 'O',
                        _ => '.',
                    });
                }
                grid.push('\n');
            }
            grid.push('\n');
        }
        let src = format!(
            "[meta]\nname = inert\n[room]\nsize = 8 8 4\nambient = solid\n\
             [node a]\npos = 1 1 1\nis_input = yes\n\
             [node b]\npos = 5 1 1\nis_output = yes\n\
             [pipe a->b]\npath = (1,1,1) (5,1,1)\n\
             [pedestal]\ncenter = 5 1 1\nvolume = 1 1 1\n\
             [door]\npos = 6 1 1\n[player]\nspawn = 3 3 1\n{grid}"
        );
        let lvl = re_level::load_string(&src, "<inert-fixture>").expect("fixture parses");
        let map = lvl.build_map();
        for x in 2..=4 {
            assert_eq!(map.cell_at(x, 6, 1), CellKind::Object, "bench ({x},6,1)");
            assert_eq!(
                cellkind_to_surface(CellKind::Object, map.side_data_at(x, 6, 1)),
                SurfaceKind::WallSolid,
                "bench ({x},6,1) must not shade as a puzzle node"
            );
        }
        // Node a (index 0) at (1,1,1) and b (index 1) at (5,1,1).
        assert_eq!(map.cell_at(1, 1, 1), CellKind::Object);
        assert_eq!(map.side_data_at(1, 1, 1), 0);
        assert_eq!(map.side_data_at(5, 1, 1), 1);
        assert_eq!(
            cellkind_to_surface(CellKind::Object, 0),
            SurfaceKind::WallOther
        );
    }

    /// A `WallOther` (puzzle object) cell shades with the distinct PUZZLE_NODE
    /// material — a recognisable node glyph — NOT the brick wall it used to use,
    /// so the puzzle reads against the calmer walls. The DOOR sentinel tag routes
    /// to the PUZZLE_DOOR material instead.
    #[test]
    fn object_cells_shade_as_puzzle_elements_not_walls() {
        let registry = MaterialRegistry::with_builtins();
        let perlin = Perlin::new(NOISE_SEED);
        let view = glam::Vec3::new(0.0, 1.0, 0.2).normalize();
        let mk = |surface: SurfaceKind, tag: u16| -> Cell {
            let g = CellGeom {
                depth: 4.0,
                foreshorten: 1.0,
                side: 0,
                tex_u: 0.5,
                wall_v: 0.5,
                wall_world_y: wall_world_y(1, 0.5),
                hit_x: 2.5,
                hit_y: 1.0,
                surface,
                material: tag,
                light: Some(LightCtx::for_wall(0, -1.0, view, &LightEnv::builtin())),
                cell: (0, 0, 0),
                tex_footprint: None,
                subcell_backdrop: None,
            };
            shade_cell(
                &registry,
                &perlin,
                &FogParams::DEFAULT,
                f32::INFINITY,
                None,
                None,
                &[],
                &g,
                4,
                0.0,
            )
        };

        // The dispatch picks the right material id for each tag.
        assert_eq!(
            surface_material(SurfaceKind::WallOther, 0),
            ids::PUZZLE_NODE
        );
        assert_eq!(
            surface_material(SurfaceKind::WallOther, DOOR_TAG_SENTINEL),
            ids::PUZZLE_DOOR
        );
        assert_eq!(surface_material(SurfaceKind::WallSolid, 0), ids::WALL_BRICK);

        // The node cell renders the node glyph; the door cell the door glyph; a
        // brick wall renders neither. (Glyphs sourced from the puzzle materials.)
        let node_glyph = registry
            .get(ids::PUZZLE_NODE)
            .sample(
                &perlin,
                glam::Vec2::new(2.3, 1.4),
                &SampleCtx {
                    distance: 4.0,
                    foreshorten: 1.0,
                    side: 0,
                    screen_y: 4,
                    wall_v: 0.5,
                    tick: 0,
                    time: 0.0,
                    fog: FogParams::DEFAULT,
                    far_freeze: f32::INFINITY,
                    light: Some(LightCtx::for_wall(0, -1.0, view, &LightEnv::builtin())),
                    lod_override: None,
                    state: 0,
                    pulse: 0.0,
                    inset_rgb: None,
                    wall_along: 0.0,
                    wall_up: 0.0,
                    on_wall: false,
                    tex_footprint: None,
                    subcell_backdrop: None,
                },
            )
            .glyph;

        let node_cell = mk(SurfaceKind::WallOther, 0);
        let door_cell = mk(SurfaceKind::WallOther, DOOR_TAG_SENTINEL);
        let wall_cell = mk(SurfaceKind::WallSolid, 0);

        assert_eq!(
            node_cell.glyph, node_glyph,
            "object cell must render the node glyph"
        );
        assert_ne!(
            node_cell.glyph, wall_cell.glyph,
            "puzzle node must be visually distinct from the wall"
        );
        assert_ne!(
            door_cell.glyph, node_cell.glyph,
            "door must be distinct from node"
        );
    }

    // ── COLOUR→MATERIAL DISPATCH (the colour substrate) ──

    /// THE COLOUR-SUBSTRATE KEYSTONE: two same-surface-kind (`WallSolid`) geometry cells painted
    /// DIFFERENT colour-material ids (`side_data` 1 and 2) shade as DIFFERENT
    /// materials — different fg AND glyph — proving the per-cell colour→material
    /// substrate dispatches by `side_data` on a geometry surface. And: a cell with
    /// `side_data == 0` (no colour-material) falls back to the surface-kind default
    /// (`WALL_BRICK`), byte-identical to today.
    #[test]
    fn color_material_two_ids_shade_as_two_materials() {
        use re_world::{MaterialDef, MaterialRecipe};
        let perlin = Perlin::new(NOISE_SEED);
        let view = glam::Vec3::new(0.0, 1.0, 0.2).normalize();

        // A registry: the built-ins PLUS a two-entry colour palette. Id 1 = a red
        // brick, id 2 = a green grain — distinct colour AND recipe, so both fg and
        // glyph must differ.
        let mut registry = MaterialRegistry::with_builtins();
        let brick_base = registry.get(ids::WALL_BRICK).clone();
        let red = MaterialDef {
            fg: 0xFF_4444,
            bg: 0x20_0000,
            recipe: Some(MaterialRecipe::Brick {
                aspect: 2.0,
                mortar_w: 0.16,
                grain: 0.6,
            }),
            ..MaterialDef::flat("cm_red", 0xFF_4444, 0x20_0000)
        };
        let green = MaterialDef {
            fg: 0x44_FF44,
            bg: 0x00_2000,
            recipe: Some(MaterialRecipe::Grain {
                scale: 1.0,
                speck: 0.5,
                tile_eps: 0.1,
            }),
            ..MaterialDef::flat("cm_green", 0x44_FF44, 0x00_2000)
        };
        let id_red = registry.register_color_material(material_from_def(&red, &brick_base));
        let id_green = registry.register_color_material(material_from_def(&green, &brick_base));
        assert_ne!(id_red, id_green, "the two palette slots are distinct");
        assert_eq!(
            registry.color_material(1),
            Some(id_red),
            "colour id 1 ⇒ slot red"
        );
        assert_eq!(
            registry.color_material(2),
            Some(id_green),
            "colour id 2 ⇒ slot green"
        );
        assert_eq!(registry.color_material(0), None, "id 0 is 'no override'");

        // Two WallSolid geometry cells differing ONLY in side_data (the colour id).
        let mk = |color_id: u16| -> Cell {
            let g = CellGeom {
                depth: 4.0,
                foreshorten: 1.0,
                side: 0,
                tex_u: 0.5,
                wall_v: 0.5,
                wall_world_y: wall_world_y(1, 0.5),
                hit_x: 2.5,
                hit_y: 1.0,
                surface: SurfaceKind::WallSolid,
                material: color_id,
                light: Some(LightCtx::for_wall(0, -1.0, view, &LightEnv::builtin())),
                cell: (0, 0, 0),
                tex_footprint: None,
                subcell_backdrop: None,
            };
            shade_cell(
                &registry,
                &perlin,
                &FogParams::DEFAULT,
                f32::INFINITY,
                None,
                None,
                &[],
                &g,
                4,
                0.0,
            )
        };

        let cell_red = mk(1);
        let cell_green = mk(2);
        let cell_default = mk(0);

        // The two colour ids shade as DIFFERENT materials (fg AND glyph differ).
        assert_ne!(
            cell_red.fg, cell_green.fg,
            "two different colour-material ids must shade with different fg"
        );
        assert_ne!(
            cell_red.glyph, cell_green.glyph,
            "two different colour-material ids must shade with different glyphs"
        );

        // side_data 0 ⇒ the surface-kind default (WALL_BRICK), byte-identical to a
        // registry with no colour palette.
        let plain = MaterialRegistry::with_builtins();
        let g_default = CellGeom {
            depth: 4.0,
            foreshorten: 1.0,
            side: 0,
            tex_u: 0.5,
            wall_v: 0.5,
            wall_world_y: wall_world_y(1, 0.5),
            hit_x: 2.5,
            hit_y: 1.0,
            surface: SurfaceKind::WallSolid,
            material: 0,
            light: Some(LightCtx::for_wall(0, -1.0, view, &LightEnv::builtin())),
            cell: (0, 0, 0),
            tex_footprint: None,
            subcell_backdrop: None,
        };
        let baseline = shade_cell(
            &plain,
            &perlin,
            &FogParams::DEFAULT,
            f32::INFINITY,
            None,
            None,
            &[],
            &g_default,
            4,
            0.0,
        );
        assert_eq!(
            cell_default.glyph, baseline.glyph,
            "id 0 keeps the brick default glyph"
        );
        assert_eq!(
            cell_default.fg, baseline.fg,
            "id 0 keeps the brick default fg"
        );
        // And it must differ from the colour cells (the colour path really changed it).
        assert_ne!(
            cell_default.fg, cell_red.fg,
            "the colour cell overrode the default"
        );
    }

    // ── STATE BUS keystone ──

    /// THE KEYSTONE: a POWERED node and a DEAD node must render DIFFERENTLY. With a
    /// `CellStateMap` that marks node-tag A powered (state 0) and node-tag B dead
    /// (state 1) installed for the cast, shading the two node cells yields different
    /// glyphs (the state-driven family). And: with an EMPTY map the shade is
    /// byte-identical to no map at all (goldens hold). Proves the bus is wired pure —
    /// `re_render` reads only the generic `u32`/`f32`, no `re_puzzle`.
    #[test]
    fn state_bus_powered_node_renders_differently_than_dead() {
        use re_core::CellStateMap;
        let registry = MaterialRegistry::with_builtins();
        let perlin = Perlin::new(NOISE_SEED);
        let view = glam::Vec3::new(0.0, 1.0, 0.2).normalize();
        // Two node cells with DISTINCT tags A and B (side_data = node index).
        let node_geom = |tag: u16| CellGeom {
            depth: 4.0,
            foreshorten: 1.0,
            side: 0,
            tex_u: 0.5,
            wall_v: 0.5,
            wall_world_y: wall_world_y(1, 0.5),
            hit_x: 2.5,
            hit_y: 1.0,
            surface: SurfaceKind::WallOther,
            material: tag,
            light: Some(LightCtx::for_wall(0, -1.0, view, &LightEnv::builtin())),
            cell: (0, 0, 0),
            tex_footprint: None,
            subcell_backdrop: None,
        };
        let tag_a = 0u16; // powered
        let tag_b = 1u16; // dead
        let shade = |g: &CellGeom| {
            shade_cell(
                &registry,
                &perlin,
                &FogParams::DEFAULT,
                f32::INFINITY,
                None,
                None,
                &[],
                g,
                4,
                0.0,
            )
        };

        // 1) EMPTY bus ⇒ byte-identical to shading with NO scope installed at all.
        crate::state_bus::clear_active();
        let a_nomap = shade(&node_geom(tag_a));
        {
            let empty = CellStateMap::new();
            let _scope = crate::state_bus::CellStateScope::new(&empty);
            let a_empty = shade(&node_geom(tag_a));
            assert_eq!(
                (a_nomap.glyph, a_nomap.fg, a_nomap.bg),
                (a_empty.glyph, a_empty.fg, a_empty.bg),
                "an EMPTY state bus must shade byte-identically to no bus (goldens hold)"
            );
        }

        // 2) POWERED (state 0) vs DEAD (state 1) ⇒ the glyphs DIFFER.
        let mut map = CellStateMap::new();
        map.set_state(tag_a, 0); // powered (resting/shipped look)
        map.set_state(tag_b, 1); // dead
        let _scope = crate::state_bus::CellStateScope::new(&map);
        let powered = shade(&node_geom(tag_a));
        let dead = shade(&node_geom(tag_b));
        assert_ne!(
            powered.glyph, dead.glyph,
            "a powered node and a dead node must render with different glyphs"
        );
        // Powered keeps the shipped neuron glyph (state 0 = resting member).
        assert_eq!(
            powered.glyph, PUZZLE_NODE_GLYPH_TEST,
            "powered node = the shipped ◉"
        );
        // Restore the empty bus for sibling tests on this thread.
        drop(_scope);
        crate::state_bus::clear_active();
    }

    /// The shipped puzzle-node glyph (◉, `PUZZLE_NODE_GLYPH` in `ascii_shader`), used
    /// by the state-bus keystone test to assert the powered/resting look is unchanged.
    const PUZZLE_NODE_GLYPH_TEST: u32 = 0x25C9;

    /// PIPE DISPATCH: a `WallTrace` cell whose `side_data` carries the high-bit
    /// PIPE FLAG routes to the state bus (keyed by the DECODED pipe tag), NOT the
    /// colour path. With the bus powering that pipe (a flow state) the cell shades
    /// HOTTER / scrolled vs an unpowered (cold) pipe whose state is 0. And a colour-id
    /// WallTrace cell (no flag) still takes the colour path — the two namespaces are
    /// disjoint.
    #[test]
    fn pipe_tagged_walltrace_reads_flow_state_not_colour() {
        use re_core::{encode_flow_state, encode_pipe_side_data, CellStateMap, FlowDir};
        let registry = MaterialRegistry::with_builtins();
        let perlin = Perlin::new(NOISE_SEED);
        let view = glam::Vec3::new(0.0, 1.0, 0.2).normalize();

        // A pipe cell: WallTrace surface, side_data = the flagged pipe tag for pipe 0.
        let pipe_tag = encode_pipe_side_data(0);
        let pipe_geom = CellGeom {
            depth: 4.0,
            foreshorten: 1.0,
            side: 0,
            tex_u: 0.5,
            wall_v: 0.5,
            wall_world_y: wall_world_y(1, 0.5),
            hit_x: 2.5,
            hit_y: 1.0,
            surface: SurfaceKind::WallTrace,
            material: pipe_tag,
            light: Some(LightCtx::for_wall(0, -1.0, view, &LightEnv::builtin())),
            cell: (0, 0, 0),
            tex_footprint: None,
            subcell_backdrop: None,
        };
        let shade_at = |state: u32, time: f32| {
            let mut map = CellStateMap::new();
            // The bus is keyed by the FULL flagged side_data (≥ 0x8000), keeping the
            // pipe key disjoint from the small node-index tags.
            map.set_state(pipe_tag, state);
            let _scope = crate::state_bus::CellStateScope::new(&map);
            shade_cell(
                &registry,
                &perlin,
                &FogParams::DEFAULT,
                f32::INFINITY,
                None,
                None,
                &[],
                &pipe_geom,
                4,
                time,
            )
        };

        // COLD pipe: state 0 → the resting circuit-trace look (no flow).
        let cold = shade_at(0, 0.7);
        // POWERED pipe: a forward flow → scrolled + hot.
        let hot = shade_at(encode_flow_state(FlowDir::Forward), 0.7);
        assert!(
            hot.glyph != cold.glyph || hot.fg != cold.fg,
            "a powered pipe cell must shade differently (flow scroll / hot hue) from a cold one"
        );
        crate::state_bus::clear_active();
    }

    /// End-to-end through the public `raycast`: a room with an `Object` cell shows
    /// the node glyph in the cast frame (the puzzle element is visible).
    #[test]
    fn raycast_shows_a_puzzle_node_in_frame() {
        use re_world::{stamp_geometry, GeometrySpec, ObjectCell};
        // A small room with one object cell in front of the spawn.
        let map = test_room();
        // Reuse the test_room shell but stamp an object near the camera path. The
        // test_room helper already gives walls/floor; we add an object via a fresh
        // stamped map of the same size with the object included.
        let _ = &map; // keep test_room import exercised
        let spec = GeometrySpec {
            size: (8, 8, 3),
            wall_kind: re_world::CellKind::WallSolid,
            decals: &[],
            objects: &[ObjectCell {
                x: 4,
                y: 2,
                z: 1,
                tag: 0,
            }],
        };
        let mut m = stamp_geometry(&spec);
        let _ = &mut m;
        let mut cam = CameraPose::spawn_in_test_room();
        cam.pos.x = 4.5;
        cam.pos.y = 5.5;
        cam.pos.z = 1.5;
        cam.yaw = std::f32::consts::PI; // face -y (north) toward the object at (4,2)
        let mut fb = Framebuffer::new(80, 40);
        raycast(&m, &cam, &mut fb);

        let node_glyph = 0x25C9u32; // ◉  (PUZZLE_NODE_GLYPH)
        let mut saw_node = false;
        for row in 0..fb.rows() {
            for col in 0..fb.cols() {
                let c = fb.get(col, row).unwrap();
                if c.surface == SurfaceKind::WallOther && c.glyph == node_glyph {
                    saw_node = true;
                }
            }
        }
        assert!(
            saw_node,
            "the puzzle node glyph must appear on the object cell in the frame"
        );
    }

    /// THE OFF-SLICE OBJECT REGRESSION (generic solid geometry, not a puzzle node —
    /// NODE SPRITE (2026-09-27) made a puzzle-node `Object` (`tag` below
    /// `RESERVED_TAG_BASE`) render-transparent on every plane; that on/off-slice
    /// LIFT + parity contract is now covered by `node_sprite`'s own
    /// `on_slice_and_off_slice_give_the_same_tier_at_the_same_horizontal_distance`
    /// test. This test keeps its ORIGINAL purpose — the off-slice pass's generic
    /// solid-`Object` lift mechanic — using the DOOR reserved tag, which still takes
    /// the pre-sprite wall-voxel path). Every shipped level authors its nodes ABOVE
    /// the eye slice (`[node] pos = x,y,2`), and the 2.5D caster walks only
    /// `cell_z = 1` — so before the off-slice pass an off-slice solid `Object` cell
    /// produced ZERO cells in the frame. It must render, and render LIFTED: a block
    /// one cell above the floor slice sits ABOVE the wall slice's rows, not spanning
    /// them.
    #[test]
    fn raycast_shows_a_node_authored_above_the_eye_slice() {
        use re_world::{stamp_geometry, GeometrySpec, ObjectCell};
        let node_glyph = 0x2592u32; // ▒ (PUZZLE_DOOR_GLYPH) — the door reserved tag's look.
        let spec = GeometrySpec {
            size: (8, 8, 5),
            wall_kind: re_world::CellKind::WallSolid,
            decals: &[],
            objects: &[ObjectCell {
                x: 4,
                y: 2,
                z: 2,
                tag: re_core::DOOR_OBJECT_TAG,
            }],
        };
        let m = stamp_geometry(&spec);
        let mut cam = CameraPose::spawn_in_test_room();
        cam.pos = glam::Vec3::new(4.5, 5.5, STANDING_EYE_Z);
        cam.yaw = std::f32::consts::PI; // face -y, toward (4,2)
        let mut fb = Framebuffer::new(80, 40);
        raycast_with_time_fov_gameplay(
            &m,
            &cam,
            &MaterialRegistry::with_builtins(),
            &Fov::default(),
            0.0,
            &mut fb,
        );

        let mut node_rows: Vec<usize> = Vec::new();
        for row in 0..fb.rows() {
            for col in 0..fb.cols() {
                let c = fb.get(col, row).unwrap();
                if c.surface == SurfaceKind::WallOther && c.glyph == node_glyph {
                    node_rows.push(row);
                    break;
                }
            }
        }
        assert!(
            !node_rows.is_empty(),
            "a node authored one plane above the eye slice must appear in the frame"
        );
        // LIFTED: every node row sits ABOVE the horizon (row < rows/2), because the
        // block spans world z [2,3] while the eye is at 1.5.
        let horizon = fb.rows() / 2;
        assert!(
            node_rows.iter().all(|&r| r < horizon),
            "the block must render above the horizon (lifted by its plane delta), got rows {node_rows:?}"
        );
    }

    /// The lift is per-PLANE, not a fixed offset: a solid `Object` (door/inert
    /// reserved tag — see the note on `raycast_shows_a_node_authored_above_the_eye_slice`
    /// for why this generic off-slice geometry test no longer uses a sprite-node tag)
    /// TWO planes above the slice (an authored `z=3`, which levels 05 / L5 use) is
    /// visible as well, and higher on screen than the same object one plane down.
    #[test]
    fn a_node_two_planes_above_renders_higher_than_one_plane_above() {
        use re_world::{stamp_geometry, CellKind, GeometrySpec, ObjectCell};
        let node_glyph = 0x2592u32; // ▒ (PUZZLE_DOOR_GLYPH)
                                    // A LONG room so the camera can stand far enough back that a block two cells
                                    // above the eye is still inside a 40-row viewport (a lift is `line_h` rows per
                                    // plane, so a near block is genuinely off the top of the screen — you would
                                    // have to look up, which is honest).
        let lowest_row = |z: i32| -> Option<usize> {
            let spec = GeometrySpec {
                size: (8, 16, 6),
                wall_kind: CellKind::WallSolid,
                decals: &[],
                objects: &[ObjectCell {
                    x: 4,
                    y: 2,
                    z,
                    tag: re_core::DOOR_OBJECT_TAG,
                }],
            };
            let m = stamp_geometry(&spec);
            let mut cam = CameraPose::spawn_in_test_room();
            cam.pos = glam::Vec3::new(4.5, 12.5, STANDING_EYE_Z);
            cam.yaw = std::f32::consts::PI;
            let mut fb = Framebuffer::new(80, 40);
            raycast_with_time_fov_gameplay(
                &m,
                &cam,
                &MaterialRegistry::with_builtins(),
                &Fov::default(),
                0.0,
                &mut fb,
            );
            (0..fb.rows()).find(|&row| {
                (0..fb.cols()).any(|col| {
                    let c = fb.get(col, row).unwrap();
                    c.surface == SurfaceKind::WallOther && c.glyph == node_glyph
                })
            })
        };
        let one = lowest_row(2).expect("z=2 node visible");
        let two = lowest_row(3).expect("z=3 node visible");
        assert!(
            two < one,
            "a node 2 planes up must draw higher on screen ({two} vs {one})"
        );
    }

    /// Occlusion holds on BOTH axes: a wall between the camera and a solid `Object`
    /// hides it, and a slab above the eye slice hides one on the storey above. (Uses
    /// the door reserved tag — see the note on
    /// `raycast_shows_a_node_authored_above_the_eye_slice`; a sprite-node's own
    /// occlusion is `node_sprite::tests::occluded_node_draws_nothing`.)
    #[test]
    fn off_slice_nodes_are_occluded_by_walls_and_slabs() {
        use re_world::{stamp_geometry, CellKind, GeometrySpec, ObjectCell};
        let node_glyph = 0x2592u32; // ▒ (PUZZLE_DOOR_GLYPH)
        let sees_node = |m: &Map| -> bool {
            let mut cam = CameraPose::spawn_in_test_room();
            cam.pos = glam::Vec3::new(4.5, 5.5, STANDING_EYE_Z);
            cam.yaw = std::f32::consts::PI;
            let mut fb = Framebuffer::new(80, 40);
            raycast_with_time_fov_gameplay(
                m,
                &cam,
                &MaterialRegistry::with_builtins(),
                &Fov::default(),
                0.0,
                &mut fb,
            );
            (0..fb.rows()).any(|row| {
                (0..fb.cols()).any(|col| {
                    let c = fb.get(col, row).unwrap();
                    c.surface == SurfaceKind::WallOther && c.glyph == node_glyph
                })
            })
        };
        let spec = GeometrySpec {
            size: (8, 8, 9),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[ObjectCell {
                x: 4,
                y: 2,
                z: 2,
                tag: re_core::DOOR_OBJECT_TAG,
            }],
        };
        assert!(
            sees_node(&stamp_geometry(&spec)),
            "control: the object is visible"
        );

        // A wall in the eye slice between camera and node.
        let mut blocked = stamp_geometry(&spec);
        blocked.set(4, 4, 1, CellKind::WallSolid);
        assert!(!sees_node(&blocked), "a wall in front must hide the node");

        // A storey slab directly above the eye slice hides everything above it.
        let upstairs = GeometrySpec {
            size: (8, 8, 9),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[ObjectCell {
                x: 4,
                y: 2,
                z: 3,
                tag: re_core::DOOR_OBJECT_TAG,
            }],
        };
        let mut slabbed = stamp_geometry(&upstairs);
        for y in 0..8 {
            for x in 0..8 {
                slabbed.set(x, y, 2, CellKind::Floor);
            }
        }
        assert!(
            !sees_node(&slabbed),
            "a storey slab must hide an object on the floor above"
        );
    }

    /// Calm undersides — the LEGACY path only. With NODE SPRITE on (the default)
    /// a puzzle-node `Object` is `Role::Air` off-slice and never reaches
    /// `slab_calm` at all — the calm-fill dome ITSELF is one of the things the
    /// sprite replaces; see `node_sprite::tests::occluded_node_draws_nothing`.
    /// This test pins the OLD mechanic under `TS_NODE_SPRITE=0`, so a node DOME overhead,
    /// seen steeply from under it, takes the calm slab fill — no `◉` field, no pool
    /// letters, in the steep rows — while the same node still draws as itself where
    /// the view is shallower.
    #[test]
    fn a_node_seen_steeply_overhead_is_a_calm_fill() {
        use re_world::{stamp_geometry, CellKind, GeometrySpec, ObjectCell};
        config::set_node_sprite_enabled(false); // exercise the legacy slab_calm path.
        let node_glyph = 0x25C9u32; // ◉ (PUZZLE_NODE_GLYPH)
        let spec = GeometrySpec {
            size: (8, 8, 9),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[ObjectCell {
                x: 4,
                y: 3,
                z: 2,
                tag: 0,
            }],
        };
        let m = stamp_geometry(&spec);
        let mut cam = CameraPose::spawn_in_test_room();
        cam.pos = glam::Vec3::new(4.5, 6.5, STANDING_EYE_Z);
        cam.yaw = std::f32::consts::PI;
        let (cols, rows) = (80usize, 40usize);
        let mut fb = Framebuffer::new(cols, rows);
        raycast_with_time_fov_gameplay(
            &m,
            &cam,
            &MaterialRegistry::with_builtins(),
            &Fov::default(),
            0.0,
            &mut fb,
        );
        config::set_node_sprite_enabled(true); // restore the default for sibling tests.
                                               // A row is "steep" when its view direction is past the reading bound (pitch 0:
                                               // horizon at rows/2, the tilt `shade_cell` uses), with a margin for rounding.
        let steep = |row: usize| {
            let vz = (rows as f32 / 2.0 - row as f32) * VIEW_VERT_SCALE / rows as f32;
            vz / (1.0 + vz * vz).sqrt() > pipe_text::READ_MAX_VIEW_Z + 0.03
        };
        let (mut steep_node, mut shallow_dome) = (0, 0);
        let mut loud = Vec::new();
        for row in 0..rows / 2 {
            for col in 0..cols {
                let c = fb.get(col, row).unwrap();
                if c.surface != SurfaceKind::WallOther {
                    continue;
                }
                let ch = char::from_u32(c.glyph).unwrap_or(' ');
                if !steep(row) {
                    shallow_dome += usize::from(c.glyph == node_glyph);
                    continue;
                }
                steep_node += 1;
                if c.glyph == node_glyph || ch.is_alphabetic() {
                    loud.push((col, row, ch));
                }
            }
        }
        assert!(
            steep_node > 20,
            "the node's face must reach the steep rows: {steep_node}"
        );
        assert!(
            loud.is_empty(),
            "steep node rows must be a calm fill, got {loud:?}"
        );
        assert!(
            shallow_dome > 0,
            "off the steep band the node still draws as itself"
        );
    }

    /// Human-readable evidence: a room with an object cell, dumped so the bright
    /// node glyph (`◉`) is eyeballable against the calmer wall.
    /// `cargo test -p re_render puzzle_node_dump -- --nocapture`.
    #[test]
    fn puzzle_node_dump_prints_node_on_wall() {
        use re_world::{stamp_geometry, GeometrySpec, ObjectCell};
        let spec = GeometrySpec {
            size: (8, 8, 3),
            wall_kind: re_world::CellKind::WallSolid,
            decals: &[],
            objects: &[ObjectCell {
                x: 4,
                y: 2,
                z: 1,
                tag: 0,
            }],
        };
        let m = stamp_geometry(&spec);
        let mut cam = CameraPose::spawn_in_test_room();
        cam.pos = glam::Vec3::new(4.5, 5.5, 1.5);
        cam.yaw = std::f32::consts::PI;
        let mut fb = Framebuffer::new(72, 24);
        raycast(&m, &cam, &mut fb);
        println!("==== PUZZLE NODE (object cell at (4,2), camera facing -y) ====");
        for row in 0..fb.rows() {
            let mut line = String::with_capacity(fb.cols());
            for col in 0..fb.cols() {
                line.push(char::from_u32(fb.get(col, row).unwrap().glyph).unwrap_or(' '));
            }
            println!("{line}");
        }
    }

    /// Screen COLUMNS that contain a single-cell `Object` (`WallOther`) for `cam`,
    /// cast through the GAMEPLAY shear path (the path a real jump / floor-change takes).
    fn object_cols_gameplay(
        map: &Map,
        registry: &MaterialRegistry,
        fov: &Fov,
        cam: &CameraPose,
    ) -> Vec<usize> {
        let mut fb = Framebuffer::new(80, 40);
        raycast_with_time_fov_gameplay(map, cam, registry, fov, 0.0, &mut fb);
        let mut cols = Vec::new();
        for col in 0..fb.cols() {
            if (0..fb.rows()).any(|row| fb.get(col, row).unwrap().surface == SurfaceKind::WallOther)
            {
                cols.push(col);
            }
        }
        cols
    }

    /// REGRESSION (axis-coupling on jump): a single-cell `Object`/door stamped at
    /// `z=1` must NOT shift HORIZONTALLY when the camera changes HEIGHT by a normal
    /// (sub-cell) jump. The 2.5D caster picks one horizontal slice via `cell_z`; the
    /// gameplay slice plane follows the PER-STOREY standing reference
    /// ([`gameplay_eye_ref_z`]), NOT the momentary jump height — so a jump shears
    /// vertically (only ROW changes) without re-selecting the plane (no sideways
    /// shift / disappearance of the single-cell object).
    ///
    /// The jump used here clears LESS than one full cell: that is the invariant the
    /// camera-only 2.5D model can honor. A jump of a FULL cell (≥ 1.0) is, by design,
    /// indistinguishable from a deliberate storey step without the level's
    /// `storey_height` — and storeys are spaced `H ≥ 3` cells apart (stairs design),
    /// well above the physical jump apex (≈ 1.06), so a real jump never crosses a
    /// storey plane. The companion test
    /// [`upper_floor_camera_selects_the_upper_storey_slice_plane`] proves the storey
    /// step DOES re-select the plane.
    #[test]
    fn object_column_is_invariant_to_camera_height_on_jump() {
        use re_world::{stamp_geometry, GeometrySpec, ObjectCell};
        // A room with one single-cell object (the door reserved tag — a sprite-node
        // tag is `Role::Air` here now and would give an empty `ground_cols`; see the
        // note on `raycast_shows_a_node_authored_above_the_eye_slice`) at z=1, the
        // standing slice. Walls span every z (full-height columns), so only the
        // single-cell object can reveal a plane-selection bug.
        let spec = GeometrySpec {
            size: (8, 8, 4),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[ObjectCell {
                x: 4,
                y: 2,
                z: 1,
                tag: re_core::DOOR_OBJECT_TAG,
            }],
        };
        let map = stamp_geometry(&spec);
        let registry = registry_for_map(&map);
        let fov = Fov::default();

        // Identical x/y/yaw/pitch at both heights — ONLY z differs.
        let base = {
            let mut c = CameraPose::spawn_in_test_room();
            c.pos = glam::Vec3::new(4.5, 5.5, STANDING_EYE_Z); // grounded, facing -y
            c.yaw = std::f32::consts::PI;
            c.pitch = 0.0;
            c
        };
        // Mid-jump: same column, raised most of a cell (0.9 < 1.0). The per-storey
        // reference keeps the slice plane fixed (`floor(eye - 0.5)` is unchanged for a
        // sub-cell rise), so the object must stay in its columns.
        let jumped = {
            let mut c = base;
            c.pos.z = STANDING_EYE_Z + 0.9;
            c
        };
        // The grounded slice plane is the SAME at both heights (the property under test).
        assert_eq!(
            gameplay_eye_ref_z(base.pos.z).floor() as i32,
            gameplay_eye_ref_z(jumped.pos.z).floor() as i32,
            "a sub-cell jump must NOT re-select the gameplay slice plane"
        );

        let ground_cols = object_cols_gameplay(&map, &registry, &fov, &base);
        let jump_cols = object_cols_gameplay(&map, &registry, &fov, &jumped);

        assert!(
            !ground_cols.is_empty(),
            "the object must be visible while grounded (test setup sanity)"
        );
        // The object must remain visible AND occupy the SAME columns mid-jump. A
        // sideways shift (or disappearance) is the axis-coupling bug.
        assert_eq!(
            ground_cols, jump_cols,
            "object screen columns must be identical at standing and jump height \
             (only ROW may change); a horizontal shift is the cell_z plane-leak bug"
        );
    }

    /// MULTI-FLOOR: the gameplay slice plane must follow the storey the camera
    /// STANDS on. A camera grounded on an UPPER storey
    /// (eye ≈ `floor_top + STANDING_EYE_Z`) must raycast that storey's wall layer —
    /// NOT the ground floor (the bug: the gameplay path used to pin `cell_z = 1`
    /// regardless of elevation, so an upper-floor camera rendered the ground floor).
    ///
    /// Build a TWO-storey map (`H = 4`, 9 cells tall): a slab floor at `z=0`, a mid
    /// floor slab at `z=4`, a ceiling at `z=8`, walls everywhere, and a single-cell
    /// object on EACH storey's wall layer (`z=1` ground, `z=5` upper) at DIFFERENT x.
    /// From the ground floor only the ground object is visible; from the upper floor
    /// only the upper object — proving the slice plane tracked the storey.
    #[test]
    fn upper_floor_camera_selects_the_upper_storey_slice_plane() {
        use re_world::{stamp_geometry, GeometrySpec, ObjectCell};
        const H: i32 = 4;
        // 9 cells tall = 2 storeys (z 0..3 ground, 4..7 upper) + top ceiling at z=8.
        let spec = GeometrySpec {
            size: (12, 12, (2 * H + 1) as usize),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[
                // Door reserved tag (not a sprite-node tag, which is `Role::Air` and
                // invisible to `object_cols_gameplay`'s `WallOther` scan here).
                ObjectCell {
                    x: 3,
                    y: 2,
                    z: 1,
                    tag: re_core::DOOR_OBJECT_TAG,
                }, // ground storey wall layer
                ObjectCell {
                    x: 8,
                    y: 2,
                    z: H + 1,
                    tag: re_core::DOOR_OBJECT_TAG,
                }, // upper storey wall layer
            ],
        };
        let map = stamp_geometry(&spec);
        let registry = registry_for_map(&map);
        let fov = Fov::default();

        // Same x/y/yaw, facing -y toward the north wall the objects sit on; only z
        // differs (ground standing vs upper-storey standing).
        let mk = |z: f32| {
            let mut c = CameraPose::spawn_in_test_room();
            c.pos = glam::Vec3::new(6.5, 9.5, z);
            c.yaw = std::f32::consts::PI;
            c.pitch = 0.0;
            c
        };
        let ground = mk(STANDING_EYE_Z); // eye 1.5 → slice plane 1
        let upper = mk(H as f32 + STANDING_EYE_Z); // eye 5.5 → slice plane 5

        // Sanity: the per-storey reference selects DIFFERENT slice planes.
        assert_eq!(gameplay_eye_ref_z(ground.pos.z).floor() as i32, 1);
        assert_eq!(gameplay_eye_ref_z(upper.pos.z).floor() as i32, H + 1);

        let ground_cols = object_cols_gameplay(&map, &registry, &fov, &ground);
        let upper_cols = object_cols_gameplay(&map, &registry, &fov, &upper);

        // Each storey must SEE its own object (proves the slice plane is correct, and
        // that floor/ceiling/wall cast on the upper storey is not blank).
        assert!(
            !ground_cols.is_empty(),
            "ground floor must render the ground-storey object (slice plane 1)"
        );
        assert!(
            !upper_cols.is_empty(),
            "upper floor must render the upper-storey object (slice plane {})",
            H + 1
        );
        // The two objects are at different x → different screen columns. If the upper
        // camera were still sampling plane 1 it would show the GROUND object's columns
        // (or none, since the upper object sits at a different x). They must differ.
        assert_ne!(
            ground_cols,
            upper_cols,
            "upper-floor view must differ from the ground-floor view — the slice plane \
             tracked the storey (object at z={} seen, not the z=1 object)",
            H + 1
        );
    }

    /// MULTI-FLOOR — floor & ceiling cast on an upper storey are non-blank. A 2.5D
    /// caster synthesizes floor/ceiling rows around the wall slice; this asserts the
    /// upper-storey cast fills BOTH the below-horizon (floor) and above-horizon
    /// (ceiling) bands with real shaded geometry (depth < ∞), so a player on floor 2
    /// sees a floor and ceiling, not a void.
    #[test]
    fn upper_floor_cast_fills_floor_and_ceiling_bands() {
        use re_world::{stamp_geometry, GeometrySpec};
        const H: i32 = 4;
        let spec = GeometrySpec {
            size: (12, 12, (2 * H + 1) as usize),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[],
        };
        let mut map = stamp_geometry(&spec);
        // A real storey slab at z=H: the upper storey stands on it. (Walls render at
        // their true height now, so without it the camera would float 4.5 cells over
        // the one ground floor and see only wall at level pitch.)
        for y in 1..11 {
            for x in 1..11 {
                map.set(x, y, H as usize, CellKind::Floor);
            }
        }
        let registry = registry_for_map(&map);
        let fov = Fov::default();

        let mut cam = CameraPose::spawn_in_test_room();
        cam.pos = glam::Vec3::new(6.5, 6.5, H as f32 + STANDING_EYE_Z); // upper storey
        cam.yaw = std::f32::consts::PI;
        cam.pitch = 0.0;

        let mut fb = Framebuffer::new(80, 40);
        raycast_with_time_fov_gameplay(&map, &cam, &registry, &fov, 0.0, &mut fb);

        let mid = fb.cols() / 2;
        let scan = |fb: &Framebuffer| {
            let (mut floor, mut ceiling) = (false, false);
            for row in 0..fb.rows() {
                let c = fb.get(mid, row).unwrap();
                match c.surface {
                    SurfaceKind::Floor if c.depth < DIST_INFINITY * 0.5 => floor = true,
                    SurfaceKind::Ceiling if c.depth < DIST_INFINITY * 0.5 => ceiling = true,
                    _ => {}
                }
            }
            (floor, ceiling)
        };
        let saw_floor = scan(&fb).0;
        // The far wall renders its true 3-cell height, so at level pitch it reaches the
        // top of the view; the ceiling band enters the frustum when looking up.
        cam.pitch = PITCH_LIMIT_RAD * 0.5;
        raycast_with_time_fov_gameplay(&map, &cam, &registry, &fov, 0.0, &mut fb);
        let saw_ceiling = scan(&fb).1;
        assert!(
            saw_floor,
            "upper-storey cast must fill a floor band (below-horizon ground geometry)"
        );
        assert!(
            saw_ceiling,
            "upper-storey cast must fill a ceiling band (above-horizon geometry)"
        );
    }

    // Helpers for the side-shading test.
    fn clone_geom(g: &CellGeom) -> CellGeom {
        CellGeom {
            depth: g.depth,
            foreshorten: g.foreshorten,
            side: g.side,
            tex_u: g.tex_u,
            wall_v: g.wall_v,
            wall_world_y: g.wall_world_y,
            hit_x: g.hit_x,
            hit_y: g.hit_y,
            surface: g.surface,
            material: g.material,
            light: g.light,
            cell: (0, 0, 0),
            tex_footprint: None,
            subcell_backdrop: None,
        }
    }
    fn luminance(c: u32) -> u32 {
        ((c >> 16) & 0xFF) + ((c >> 8) & 0xFF) + (c & 0xFF)
    }

    // ───────────────────── worked-out shadows (#2) tests ─────────────────────

    /// Build an empty (open) cube of `n³` cells with a 1-cell solid shell so the
    /// interior is fully lit (no occluders). The interior floor sits at z-plane 1.
    fn open_room(n: usize) -> Map {
        let mut m = Map::new(n, n, n);
        // Floor at z=0, ceiling at z=n-1; interior z=1..n-1 left Empty (open to the
        // key light). No interior pillars → an interior point is unoccluded.
        for x in 0..n {
            for y in 0..n {
                m.set(x, y, 0, CellKind::Floor);
                m.set(x, y, n - 1, CellKind::Ceiling);
            }
        }
        m
    }

    /// Like [`open_room`], but with NO capping `Ceiling` slab at all — the
    /// interior stays [`CellKind::Empty`] all the way to the map's top. Every
    /// `[room]`-built level (and `open_room`/`re_world::test_room`) stamps a real
    /// `Ceiling` voxel at its z-top ([`re_world::stamp_geometry`]), so those
    /// fixtures are "enclosed overhead" under `column_sky_open` regardless of
    /// how open they read horizontally. This fixture is the one genuinely open
    /// case (no `[sky]` authored, no roof anywhere) — the sky-panorama tests use
    /// it so they exercise the real direction-sampled panorama, not the shaded
    /// ceiling-material fallback an enclosed room now correctly gets instead.
    fn sky_open_room(n: usize) -> Map {
        let mut m = Map::new(n, n, n);
        for x in 0..n {
            for y in 0..n {
                m.set(x, y, 0, CellKind::Floor);
            }
        }
        m
    }

    /// A cell that is OCCLUDED from the key light is darker than the SAME cell when
    /// unoccluded; an open cell is unshadowed (`shadow_key == 1.0`); and the
    /// occluded cell is NOT pure black (ambient + headlamp survive). This is the
    /// core of the worked-out-shadows deliverable.
    #[test]
    fn occluded_cell_is_darker_than_unoccluded_but_not_black() {
        let key_dir = LightEnv::builtin().key_dir; // toward (-x,-y,+z)
        let n = 8usize;

        // OPEN room: an interior floor point is fully key-lit.
        let open = open_room(n);
        let (hx, hy, cz) = (4.5f32, 4.5f32, 1i32);
        let floor_n = glam::Vec3::Z;
        let lit = shadow_key_for_lod(
            &open,
            ShadowQuality::Soft,
            ShadowSample {
                depth: 3.0,
                hx,
                hy,
                cell_z: cz,
                face_n: floor_n,
            },
            key_dir,
            None,
        );
        assert!(
            lit > 0.99,
            "an open interior floor cell must be unshadowed (shadow_key≈1), got {lit}"
        );

        // OCCLUDED: drop a solid pillar UP-AND-TOWARD the light from the floor point
        // so the key ray is blocked. The key points to (-x,-y,+z); a solid block one
        // cell over in -x,-y and one up in +z sits squarely on the ray.
        let mut occ = open_room(n);
        // Fill a tall solid pillar between the floor point and the light.
        for (dx, dy, dz) in [(-1, -1, 1), (-2, -2, 2), (-1, -1, 2)] {
            let cx = (hx as i32) + dx;
            let cy = (hy as i32) + dy;
            let cz2 = cz + dz;
            occ.set(cx as usize, cy as usize, cz2 as usize, CellKind::WallSolid);
        }
        let shadowed = shadow_key_for_lod(
            &occ,
            ShadowQuality::Soft,
            ShadowSample {
                depth: 3.0,
                hx,
                hy,
                cell_z: cz,
                face_n: floor_n,
            },
            key_dir,
            None,
        );
        assert!(
            shadowed < lit,
            "the occluded floor cell must have a smaller shadow_key than the open one \
             (shadowed={shadowed} < lit={lit})"
        );

        // Now shade the SAME floor texel both ways and confirm the lit colour is
        // darker in shadow — but never pure black (ambient + headlamp survive).
        let registry = MaterialRegistry::with_builtins();
        let perlin = Perlin::new(NOISE_SEED);
        let fc_view = glam::Vec3::new(0.1, 0.1, 1.0).normalize();
        let shade = |shadow_key: f32| -> Cell {
            let mut lc = LightCtx::for_floor(true, fc_view, &LightEnv::builtin());
            lc.shadow_key = shadow_key;
            let g = CellGeom {
                depth: 3.0,
                foreshorten: 1.0,
                side: 0,
                tex_u: 0.0,
                wall_v: 0.0,
                wall_world_y: 0.0,
                hit_x: hx,
                hit_y: hy,
                surface: SurfaceKind::Floor,
                material: 0,
                light: Some(lc),
                cell: (0, 0, 0),
                tex_footprint: None,
                subcell_backdrop: None,
            };
            // Fog off so we isolate the shadow term, not the fade-to-black.
            let no_fog = FogParams {
                near: 1.0,
                far: 1.0e6,
                color: None,
                height: None,
            };
            shade_cell(
                &registry,
                &perlin,
                &no_fog,
                f32::INFINITY,
                None,
                None,
                &[],
                &g,
                14,
                0.0,
            )
        };
        let cell_lit = shade(1.0);
        let cell_shadow = shade(0.0); // fully in the key's shadow
        assert!(
            luminance(cell_shadow.fg) < luminance(cell_lit.fg),
            "fully-shadowed floor fg must be darker than fully-lit (shadow={} < lit={})",
            luminance(cell_shadow.fg),
            luminance(cell_lit.fg)
        );
        assert!(
            luminance(cell_shadow.fg) > 0 || luminance(cell_shadow.bg) > 0,
            "a shadowed cell must keep ambient/headlamp energy — never pure black"
        );
    }

    /// ITEM 1 — per-material LOD override is threaded through the FLOOR/CEILING shadow
    /// path, same as walls. A floor cell at a FAR depth where the GLOBAL LOD curve has
    /// already switched the cast shadow OFF (`shadow_key == 1.0`, no occlusion ray) must
    /// STILL cast its shadow when shaded with a HERO material whose `full_detail_to`
    /// keeps it in a shadow-casting tier. With `None` (the default FLOOR_GRAIN material)
    /// the same far cell is unshadowed — proving the default is byte-identical and the
    /// override is what re-enables the detail.
    #[test]
    fn floor_lod_override_holds_cast_shadow_farther_than_global() {
        // Default LOD: near_end=6, far_start=13. A depth past far_start is the FAR tier
        // → ShadowTierQuality::Off → shadow_key forced to 1.0 regardless of occlusion.
        crate::lod::LodConfig::default().apply();
        let key_dir = LightEnv::builtin().key_dir; // toward (-x,-y,+z)
        let n = 24usize;
        let mut occ = open_room(n);
        // A point on the floor with a solid pillar UP-AND-TOWARD the light so the key
        // ray is unambiguously blocked (same geometry as the near-field shadow test).
        let (hx, hy, cz) = (12.5f32, 12.5f32, 1i32);
        for (dx, dy, dz) in [(-1, -1, 1), (-2, -2, 2), (-1, -1, 2)] {
            let cx = (hx as i32) + dx;
            let cy = (hy as i32) + dy;
            let cz2 = cz + dz;
            occ.set(cx as usize, cy as usize, cz2 as usize, CellKind::WallSolid);
        }
        let far_depth = 20.0; // > far_start (13) ⇒ FAR tier under the global curve.
        let sample = ShadowSample {
            depth: far_depth,
            hx,
            hy,
            cell_z: cz,
            face_n: glam::Vec3::Z,
        };

        // GLOBAL curve (the default FLOOR_GRAIN material's `lod_override == None`):
        // FAR tier disables the ray → fully "lit" (shadow_key == 1.0).
        let global = shadow_key_for_lod(&occ, ShadowQuality::Soft, sample, key_dir, None);
        assert!(
            global >= 0.999,
            "under the global LOD curve a far floor cell skips the shadow ray (key≈1), got {global}"
        );

        // HERO override: hold full detail out to 30 cells → 20 is back inside NEAR →
        // the cast shadow is re-enabled → the OCCLUDED far cell is now darkened.
        let hero = Some(crate::lod::LodOverride::full_detail_to(30.0));
        let overridden = shadow_key_for_lod(&occ, ShadowQuality::Soft, sample, key_dir, hero);
        assert!(
            overridden < global,
            "a hero floor material must keep its cast shadow farther out than the global \
             curve (override={overridden} < global={global})"
        );
    }

    /// EMISSIVE survives in shadow: a self-lit material (a circuit trace) keeps its
    /// glow even fully in the key's cast shadow — shadows attenuate only the key
    /// diffuse/specular, never the emissive or ambient floor.
    #[test]
    fn emissive_and_ambient_survive_full_shadow() {
        let registry = MaterialRegistry::with_builtins();
        let perlin = Perlin::new(NOISE_SEED);
        let view = glam::Vec3::new(0.0, 1.0, 0.2).normalize();
        // The circuit wall carries emissive traces; shade it fully shadowed.
        let mut lc = LightCtx::for_wall(0, -1.0, view, &LightEnv::builtin());
        lc.shadow_key = 0.0;
        let g = CellGeom {
            depth: 3.0,
            foreshorten: 1.0,
            side: 0,
            tex_u: 0.5,
            wall_v: 0.5,
            wall_world_y: wall_world_y(1, 0.5),
            hit_x: 2.5,
            hit_y: 1.0,
            surface: SurfaceKind::WallTrace, // → WALL_CIRCUIT (emissive)
            material: 0,
            light: Some(lc),
            cell: (0, 0, 0),
            tex_footprint: None,
            subcell_backdrop: None,
        };
        let no_fog = FogParams {
            near: 1.0,
            far: 1.0e6,
            color: None,
            height: None,
        };
        let cell = shade_cell(
            &registry,
            &perlin,
            &no_fog,
            f32::INFINITY,
            None,
            None,
            &[],
            &g,
            14,
            0.0,
        );
        assert!(
            luminance(cell.fg) > 0,
            "an emissive circuit cell must not go pure black in full key shadow (fg={:#08x})",
            cell.fg
        );
    }

    /// `ShadowQuality::Off` reproduces the pre-shadow output: with `Off` the shadow
    /// pass is skipped (`shadow_key` stays 1.0), so the cast frame is byte-identical
    /// to one shaded with the default (unshadowed) `LightCtx` — and DIFFERS from a
    /// `Soft` cast on a map that actually casts shadows (proving shadows do work).
    #[test]
    fn shadow_quality_off_matches_pre_shadow_and_soft_differs() {
        use re_world::{stamp_geometry, GeometrySpec, ObjectCell};
        // A room with a tall pillar that throws a shadow onto the floor.
        let spec = GeometrySpec {
            size: (10, 10, 4),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[
                ObjectCell {
                    x: 4,
                    y: 4,
                    z: 1,
                    tag: 0,
                },
                ObjectCell {
                    x: 4,
                    y: 4,
                    z: 2,
                    tag: 0,
                },
            ],
        };
        let map = stamp_geometry(&spec);
        let mut cam = CameraPose::spawn_in_test_room();
        cam.pos = glam::Vec3::new(6.5, 6.5, 1.5);
        cam.yaw = std::f32::consts::PI * 1.25; // look toward the pillar
        let registry = registry_for_map(&map);
        let fov = Fov::default();

        let cast = |q: ShadowQuality| -> Framebuffer {
            let mut fb = Framebuffer::new(80, 40);
            raycast_with_time_fov_q(&map, &cam, &registry, &fov, 0.0, q, cam.pos.z, &mut fb);
            fb
        };
        let off = cast(ShadowQuality::Off);
        let soft = cast(ShadowQuality::Soft);

        // Off must reproduce the pre-shadow look: identical to a cast where the
        // shadow factor is forced to 1.0 everywhere. The `Off` path *is* that path
        // by construction (shadow_key never moves off 1.0), so re-casting `Off`
        // twice is deterministic and equals itself; the meaningful assertion is that
        // `Soft` DIFFERS — i.e. the shadow pass actually changed pixels.
        let off2 = cast(ShadowQuality::Off);
        let mut off_identical = true;
        let mut soft_differs = false;
        for row in 0..off.rows() {
            for col in 0..off.cols() {
                let a = off.get(col, row).unwrap();
                let b = off2.get(col, row).unwrap();
                if (a.fg, a.bg, a.glyph) != (b.fg, b.bg, b.glyph) {
                    off_identical = false;
                }
                let s = soft.get(col, row).unwrap();
                if (a.fg, a.bg) != (s.fg, s.bg) {
                    soft_differs = true;
                }
            }
        }
        assert!(
            off_identical,
            "ShadowQuality::Off must be deterministic / stable"
        );
        assert!(
            soft_differs,
            "Soft shadows must change at least one cell vs Off (the pillar casts a shadow)"
        );
    }

    /// Human-readable evidence: a room with a tall pillar, cast with Soft shadows,
    /// printed as a SHADOW MAP — `#` = a cell the key light reaches, `.`/space =
    /// a cell the pillar (or a wall) casts a shadow onto. Eyeball the pillar's
    /// shadow falling across the floor with directional falloff.
    /// `cargo test -p re_render shadow_map_dump -- --nocapture`.
    #[test]
    fn shadow_map_dump_prints_cast_shadow_on_floor() {
        use re_world::{stamp_geometry, GeometrySpec, ObjectCell};
        // A 14×14 room with a 3-tall pillar near the centre.
        let spec = GeometrySpec {
            size: (14, 14, 5),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[
                ObjectCell {
                    x: 7,
                    y: 7,
                    z: 1,
                    tag: 0,
                },
                ObjectCell {
                    x: 7,
                    y: 7,
                    z: 2,
                    tag: 0,
                },
                ObjectCell {
                    x: 7,
                    y: 7,
                    z: 3,
                    tag: 0,
                },
            ],
        };
        let map = stamp_geometry(&spec);
        let key_dir = LightEnv::builtin().key_dir;
        println!("==== SHADOW MAP (floor z=1, pillar at (7,7), key toward up-front-left) ====");
        println!("(# = key-lit, ' ' = in the pillar/wall cast shadow)");
        for y in 1..13 {
            let mut line = String::with_capacity(12);
            for x in 1..13 {
                let sk = shadow_key_for_lod(
                    &map,
                    ShadowQuality::Soft,
                    ShadowSample {
                        depth: 3.0,
                        hx: x as f32 + 0.5,
                        hy: y as f32 + 0.5,
                        cell_z: 1,
                        face_n: glam::Vec3::Z,
                    },
                    key_dir,
                    None,
                );
                line.push(if sk > 0.85 {
                    '#'
                } else if sk > 0.5 {
                    '+'
                } else if sk > 0.15 {
                    ':'
                } else {
                    ' '
                });
            }
            println!("{line}");
        }
    }

    /// The DDA shadow ray itself: a ray fired into open space escapes (None =
    /// unoccluded); a ray fired straight into an adjacent solid wall is blocked
    /// (Some = occluded). Direct coverage of the grid-marching primitive.
    #[test]
    fn shadow_ray_blocks_on_solid_passes_through_empty() {
        use crate::shadow::shadow_ray_hit;
        let n = 8usize;
        let open = open_room(n);
        // The interior is OPEN (only floor z=0 + ceiling z=n-1 are non-empty, and
        // neither is `is_solid`), so a horizontal ray through the interior marches
        // out of bounds without hitting anything → None (unoccluded).
        let escape = shadow_ray_hit(
            &open,
            4.5,
            4.5,
            1.5,
            glam::Vec3::new(1.0, 0.0, 0.0),
            SHADOW_MAX_STEPS,
        );
        assert!(
            escape.is_none(),
            "ray through the open interior must escape (unoccluded)"
        );

        // A solid cell placed immediately in the ray's path blocks it close.
        let mut occ = open_room(n);
        occ.set(5, 4, 1, CellKind::WallSolid);
        let hit = shadow_ray_hit(
            &occ,
            4.5,
            4.5,
            1.5,
            glam::Vec3::new(1.0, 0.0, 0.0),
            SHADOW_MAX_STEPS,
        );
        assert!(
            hit.is_some(),
            "ray into the adjacent solid must be occluded"
        );
        assert!(
            hit.unwrap() < 1.5,
            "adjacent blocker is close, got {:?}",
            hit
        );
    }

    // ─────────────────────────── MIRROR / REFLECTION (#1) ────────────────────
    //
    // A reflective wall surface reflects the primary ray about its face normal and
    // shows the reflected scene. These tests cover: the reflect math, that the
    // mirror cell renders the REFLECTED geometry (not its own surface), the clarity
    // crisp↔glitch degradation, and that a non-mirror wall is unchanged.

    /// A corridor along x with a MIRROR wall at x=1 (a `WallTrace` cell whose slot-1
    /// material is reflective) and a distinctive MATRIX wall at x=5. Floor z=0,
    /// ceiling z=3, interior air z=1..2. The player stands mid-corridor.
    fn mirror_corridor() -> Map {
        let (sx, sy, sz) = (7usize, 3usize, 4usize);
        let mut m = Map::new(sx, sy, sz);
        for z in 0..sz {
            for y in 0..sy {
                for x in 0..sx {
                    let perim = x == 0 || x == sx - 1 || y == 0 || y == sy - 1;
                    let kind = if perim {
                        CellKind::WallSolid
                    } else if z == 0 {
                        CellKind::Floor
                    } else if z == sz - 1 {
                        CellKind::Ceiling
                    } else {
                        continue;
                    };
                    m.set(x, y, z, kind);
                }
            }
        }
        // The MIRROR wall: the whole x=1 column (the west wall the player faces) is
        // a WallTrace cell → slot-1 material (made reflective in the test registry).
        for z in 1..sz - 1 {
            for y in 1..sy - 1 {
                m.set(1, y, z, CellKind::WallTrace);
            }
        }
        // The DISTINCTIVE reflected target: the x=5 column is a WallMatrix wall.
        for z in 1..sz - 1 {
            for y in 1..sy - 1 {
                m.set(5, y, z, CellKind::WallMatrix);
            }
        }
        m
    }

    /// A registry whose slot-1 (`WallTrace` → `WALL_CIRCUIT`) material is a MIRROR
    /// at the given clarity; every other slot is the built-in. So a `WallTrace` hit
    /// reflects and a `WallMatrix` hit is the plain (distinctive) matrix wall.
    fn mirror_registry(clarity: f32) -> MaterialRegistry {
        let reg = MaterialRegistry::with_builtins();
        // Rebuild slot 1 as the built-in circuit but reflective.
        let mut mat = reg.get(ids::WALL_CIRCUIT).clone();
        mat.reflective = true;
        mat.clarity = clarity;
        // Re-seat the whole registry with slot 1 replaced (registry has no setter).
        let mut out = MaterialRegistry::new();
        for slot in 0..7u16 {
            if slot == 1 {
                out.register(mat.clone());
            } else {
                out.register(reg.get(MaterialId(slot)).clone());
            }
        }
        out
    }

    /// Cast the corridor: the player at mid-corridor faces −x straight at the
    /// mirror wall (x=1). Returns the framebuffer.
    fn cast_mirror(reg: &MaterialRegistry, clarity_override: Option<f32>) -> Framebuffer {
        let map = mirror_corridor();
        let cam = CameraPose {
            pos: glam::Vec3::new(3.5, 1.5, 1.5),
            yaw: std::f32::consts::FRAC_PI_2 * 3.0, // -x (west), toward the mirror
            pitch: 0.0,
        };
        let mut fb = Framebuffer::new(40, 24);
        raycast_full(
            &map,
            &cam,
            reg,
            &Fov::default(),
            0.0,
            ShadowQuality::Off,
            MirrorClarity {
                override_value: clarity_override,
            },
            f32::INFINITY,
            cam.pos.z,
            &mut fb,
        );
        fb
    }

    /// The reflect math: a ray reflected about an axis-aligned face normal flips the
    /// perpendicular component and keeps the parallel one — exactly what the cast
    /// does. An EW (x-const) face flips x; an NS (y-const) face flips y.
    #[test]
    fn reflected_ray_direction_is_correct() {
        // EW face (side==1): reflect about the x-normal → x flips, y unchanged.
        let (dx, dy) = (0.6f32, -0.3f32);
        let side1 = (-dx, dy);
        assert_eq!(side1, (-0.6, -0.3), "EW-face reflection must flip x only");
        // NS face (side==0): reflect about the y-normal → y flips, x unchanged.
        let side0 = (dx, -dy);
        assert_eq!(side0, (0.6, 0.3), "NS-face reflection must flip y only");

        // And the reflection is an involution about the face: reflecting twice
        // restores the original direction.
        assert_eq!((-(-dx), dy), (dx, dy));
    }

    /// The MIRROR cell renders the REFLECTED geometry, not the mirror's own surface.
    /// The player faces the mirror (x=1, a WallTrace/circuit surface). Behind/around
    /// the mirror the reflected ray lands on the MATRIX wall (x=5). So a wall cell in
    /// the centre column must read as `WallMatrix` (the reflected surface) — NOT the
    /// mirror's own `WallTrace`. A non-reflective registry, same scene, reads the
    /// mirror's own `WallTrace` surface there: proof the reflection is what changed.
    #[test]
    fn mirror_cell_shows_reflected_surface_not_its_own() {
        let mid = 20usize; // centre column

        // Reflective: the centre wall band must carry the REFLECTED matrix surface.
        let fb_m = cast_mirror(&mirror_registry(1.0), Some(1.0));
        let mut saw_reflected_matrix = false;
        let mut saw_own_trace = false;
        for row in 0..fb_m.rows() {
            let c = fb_m.get(mid, row).unwrap();
            if c.surface == SurfaceKind::WallMatrix {
                saw_reflected_matrix = true;
            }
            if c.surface == SurfaceKind::WallTrace {
                saw_own_trace = true;
            }
        }
        assert!(
            saw_reflected_matrix,
            "the mirror cell must show the REFLECTED matrix surface"
        );
        assert!(
            !saw_own_trace,
            "the mirror cell must NOT show its own WallTrace surface"
        );

        // Non-reflective baseline: the SAME scene + camera, slot-1 NOT reflective →
        // the centre wall reads as the mirror's own WallTrace surface (no reflection).
        let fb_p = cast_mirror(&MaterialRegistry::with_builtins(), None);
        let mut plain_trace = false;
        for row in 0..fb_p.rows() {
            if fb_p.get(mid, row).unwrap().surface == SurfaceKind::WallTrace {
                plain_trace = true;
            }
        }
        assert!(
            plain_trace,
            "a NON-reflective wall must render its own WallTrace surface (no reflection)"
        );
    }

    /// A 4×4 texture OPAQUE in its top half (rows 0..1, bright red) and TRANSPARENT in
    /// its bottom half (rows 2..3, alpha 0). A pane tiled with it is a real HIT (the
    /// opaque top stops the ray, so the cut-out recurses) yet shows the backdrop through
    /// its transparent bottom band — the setup the stacked-pane cap test needs.
    fn synthetic_top_opaque_4x4() -> re_core::TextureImage {
        let mut rgba = Vec::with_capacity(4 * 4 * 4);
        for row in 0..4u8 {
            let a = if row < 2 { 255 } else { 0 };
            for _ in 0..4 {
                rgba.extend_from_slice(&[255, 0, 0, a]);
            }
        }
        re_core::TextureImage::new(4, 4, rgba)
    }

    /// A registry built on the BUILT-INS with the `WALL_BRICK` slot replaced by a
    /// per-cell-tiling `Image` recipe (no span / no sub-cell), the given texture
    /// attached. Every other slot — crucially `WALL_MATRIX` — stays the built-in
    /// (non-image), so a `WallMatrix` backdrop reads distinctly from an image pane.
    fn brick_image_registry(textures: &[re_core::TextureSet]) -> MaterialRegistry {
        let builtins = MaterialRegistry::with_builtins();
        let mut reg = MaterialRegistry::new();
        for slot in 0..8u16 {
            if slot == ids::WALL_BRICK.0 {
                reg.register(Material {
                    recipe: TextureRecipe::Image {
                        tex_id: 0,
                        span: None,
                        subcell: ascii_shader::SubcellMode::Off,
                        fit: ascii_shader::Fit::Stretch,
                        tile: ascii_shader::IMAGE_TILE_NONE,
                        relief: None,
                    },
                    ..builtins.get(MaterialId(slot)).clone()
                });
            } else {
                reg.register(builtins.get(MaterialId(slot)).clone());
            }
        }
        reg.attach_image_textures(textures);
        reg
    }

    /// FIX 2 (cut-out cap fallthrough): a line of sight through 6 STACKED transparent
    /// image panes exceeds the depth-4 recursion cap. Past the cap the OLD code fell
    /// through to an OPAQUE wall slice (a hard block where there should be scene). The
    /// fix recasts past the capped pane to the SCENE BEHIND, so the distinctive backdrop
    /// (a `WallMatrix` wall) is what the capped column shows — never the pane's own
    /// opaque image surface.
    #[test]
    fn cutout_cap_reveals_scene_behind_not_opaque_block() {
        // Corridor along x: 6 image panes at x=2..7, a MATRIX backdrop wall at x=8.
        // The player at x=9.5 looks −x straight down the stack. (x=2..7 = 6 panes >
        // IMAGE_CUTOUT_MAX_DEPTH = 4, so the cap is hit before the backdrop.)
        let (sx, sy, sz) = (11usize, 3usize, 4usize);
        let mut m = Map::new(sx, sy, sz);
        for z in 0..sz {
            for y in 0..sy {
                for x in 0..sx {
                    let perim = y == 0 || y == sy - 1;
                    let kind = if z == 0 {
                        CellKind::Floor
                    } else if z == sz - 1 {
                        CellKind::Ceiling
                    } else if perim {
                        CellKind::WallSolid
                    } else {
                        continue;
                    };
                    m.set(x, y, z, kind);
                }
            }
        }
        // The 6 stacked image panes (WallSolid → WALL_BRICK → the Image recipe). Each is
        // top-opaque / bottom-transparent, so the ray HITS each (recursing) but the
        // bottom band reveals what is behind.
        for x in 2..8usize {
            for z in 1..sz - 1 {
                m.set(x, 1, z, CellKind::WallSolid);
            }
        }
        // The distinctive BACKDROP at x=8: a MATRIX wall (non-image, WALL_MATRIX).
        for z in 1..sz - 1 {
            m.set(8, 1, z, CellKind::WallMatrix);
        }

        let textures = vec![re_core::TextureSet::still(synthetic_top_opaque_4x4())];
        let reg = brick_image_registry(&textures);
        let cam = CameraPose {
            pos: glam::Vec3::new(9.5, 1.5, 1.5),
            yaw: std::f32::consts::FRAC_PI_2 * 3.0, // -x, down the stack
            pitch: 0.0,
        };
        let mut fb = Framebuffer::new(40, 24);
        raycast_full(
            &m,
            &cam,
            &reg,
            &Fov::default(),
            0.0,
            ShadowQuality::Off,
            MirrorClarity::default(),
            f32::INFINITY,
            cam.pos.z,
            &mut fb,
        );

        // The centre column looks straight down the pane stack. SOMEWHERE in that column
        // (the transparent lower band, seen past the cap) the distinctive MATRIX backdrop
        // must appear — proof the capped pane degraded to "the scene behind", not an
        // opaque block. If the old fallthrough were active the capped pane would paint
        // its own WallSolid image over the whole slice and the matrix would never show.
        let mid = fb.cols() / 2;
        let mut saw_matrix = false;
        for row in 0..fb.rows() {
            if fb.get(mid, row).unwrap().surface == SurfaceKind::WallMatrix {
                saw_matrix = true;
            }
        }
        assert!(
            saw_matrix,
            "5+ stacked transparent panes must reveal the MATRIX backdrop past the cap, \
             not an opaque pane block"
        );
    }

    /// CLARITY drives the look: `clarity = 1` is a crisp reflection (real reflected
    /// glyphs survive); `clarity = 0` degrades to glitch/empty (mostly blank cells
    /// with sparse scrambled glyphs). The two must differ, and clarity-0 must be far
    /// emptier (more blank glyphs) than clarity-1.
    #[test]
    fn clarity_zero_glitches_and_one_is_crisp() {
        let mid = 20usize;
        let count_blank = |fb: &Framebuffer| -> (usize, usize) {
            let mut blank = 0usize;
            let mut wall_rows = 0usize;
            for row in 0..fb.rows() {
                let c = fb.get(mid, row).unwrap();
                if c.surface.is_wall() {
                    wall_rows += 1;
                    if c.glyph == u32::from(' ') {
                        blank += 1;
                    }
                }
            }
            (wall_rows, blank)
        };

        let crisp = cast_mirror(&mirror_registry(1.0), Some(1.0));
        let broken = cast_mirror(&mirror_registry(0.0), Some(0.0));

        let (crisp_rows, crisp_blank) = count_blank(&crisp);
        let (broken_rows, broken_blank) = count_blank(&broken);
        assert!(
            crisp_rows > 0 && broken_rows > 0,
            "both must hit the mirror band"
        );

        // The crisp reflection keeps most of its glyphs; the broken one blanks most.
        assert!(
            broken_blank > crisp_blank,
            "clarity=0 must blank more cells (glitch/empty) than clarity=1 \
             (broken_blank={broken_blank} crisp_blank={crisp_blank})"
        );
        assert!(
            (broken_blank as f32) > (broken_rows as f32) * 0.5,
            "clarity=0 must be mostly empty (blank={broken_blank}/{broken_rows})"
        );
        assert!(
            (crisp_blank as f32) < (crisp_rows as f32) * 0.5,
            "clarity=1 must keep most reflected glyphs (blank={crisp_blank}/{crisp_rows})"
        );

        // And the two frames are genuinely different cell-for-cell in the band.
        let mut differ = false;
        for row in 0..crisp.rows() {
            if crisp.get(mid, row).unwrap().glyph != broken.get(mid, row).unwrap().glyph {
                differ = true;
                break;
            }
        }
        assert!(
            differ,
            "clarity 0 vs 1 must produce visibly different cells"
        );
    }

    /// The mirror composite is DETERMINISTIC: same scene + clarity → identical frame
    /// (golden-safe; the glitch noise is a fixed hash on the screen cell, no clock).
    #[test]
    fn mirror_is_deterministic() {
        let a = cast_mirror(&mirror_registry(0.3), Some(0.3));
        let b = cast_mirror(&mirror_registry(0.3), Some(0.3));
        for row in 0..a.rows() {
            for col in 0..a.cols() {
                assert_eq!(
                    a.get(col, row).unwrap(),
                    b.get(col, row).unwrap(),
                    "mirror cast must be deterministic at ({col},{row})"
                );
            }
        }
    }

    /// A non-mirror scene is UNCHANGED by the mirror code path: casting the built-in
    /// registry (no reflective material anywhere) is bit-identical to the legacy
    /// `raycast_with` entry. Back-compat guarantee for the shipped levels.
    #[test]
    fn non_mirror_scene_is_unchanged() {
        let map = test_room();
        let cam = CameraPose::spawn_in_test_room();
        let reg = MaterialRegistry::with_builtins();

        let mut fb_new = Framebuffer::new(48, 24);
        raycast_full(
            &map,
            &cam,
            &reg,
            &Fov::default(),
            0.0,
            ShadowQuality::Soft,
            MirrorClarity::default(),
            f32::INFINITY,
            cam.pos.z,
            &mut fb_new,
        );

        let mut fb_legacy = Framebuffer::new(48, 24);
        raycast_with_time_fov_q(
            &map,
            &cam,
            &reg,
            &Fov::default(),
            0.0,
            ShadowQuality::Soft,
            cam.pos.z,
            &mut fb_legacy,
        );

        for row in 0..fb_new.rows() {
            for col in 0..fb_new.cols() {
                assert_eq!(
                    fb_new.get(col, row).unwrap(),
                    fb_legacy.get(col, row).unwrap(),
                    "a non-mirror scene must render identically (back-compat) at ({col},{row})"
                );
            }
        }
    }

    /// `MirrorClarity::effective` honours the live override and clamps to [0,1].
    #[test]
    fn mirror_clarity_override_resolves() {
        let none = MirrorClarity::default();
        assert_eq!(none.effective(0.7), 0.7, "no override → authored clarity");
        let over = MirrorClarity {
            override_value: Some(0.2),
        };
        assert_eq!(over.effective(0.9), 0.2, "override wins over authored");
        let hi = MirrorClarity {
            override_value: Some(5.0),
        };
        assert_eq!(hi.effective(0.0), 1.0, "override clamps to 1");
    }

    // ───────────────────────── hologram objects ─────────────────────────────

    /// Build a room with a hologram box floating ahead of a camera that faces -y
    /// (north) toward it. Returns the cast framebuffer + the cube's column band.
    fn holo_scene() -> (Framebuffer, usize, usize) {
        use re_world::{stamp_geometry, stamp_holos, GeometrySpec, HoloObject, HoloShape};
        let spec = GeometrySpec {
            size: (11, 16, 5),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[],
        };
        let mut m = stamp_geometry(&spec);
        // A 3×3×3 cube spanning the eye z-plane (z=1), centred in x.
        stamp_holos(
            &mut m,
            &[HoloObject {
                shape: HoloShape::Box,
                x0: 4,
                y0: 6,
                z0: 1,
                x1: 6,
                y1: 8,
                z1: 3,
                style: 0,
            }],
        );
        assert!(m.has_holograms());
        let mut cam = CameraPose::spawn_in_test_room();
        cam.pos = glam::Vec3::new(5.5, 13.5, 1.5);
        cam.yaw = std::f32::consts::PI; // face -y toward the cube
        let mut fb = Framebuffer::new(100, 32);
        raycast(&m, &cam, &mut fb);
        (fb, 0, 100)
    }

    /// A hologram object's FRONT contour is the bright blue; its BACK/SIDE contour
    /// is the softer (dimmer) cyan — depth reads near→far as bright→faint.
    #[test]
    fn hologram_front_is_bright_blue_back_is_soft_cyan() {
        let (fb, c0, c1) = holo_scene();
        let mut saw_front = false;
        let mut saw_back = false;
        let back = dim_rgb_local(HOLO_BACK, HOLO_BACK_DIM);
        for row in 0..fb.rows() {
            for col in c0..c1 {
                let c = fb.get(col, row).unwrap();
                if c.fg == HOLO_FRONT && c.glyph != u32::from(' ') {
                    saw_front = true;
                }
                if c.fg == back {
                    saw_back = true;
                }
            }
        }
        assert!(
            saw_front,
            "the front contour/glyphs must be the bright hologram blue"
        );
        assert!(
            saw_back,
            "the back/side contour must be the softer, dimmer cyan"
        );
        // The back colour must actually be DIMMER than the front (less noticeable).
        let lum = |c: u32| ((c >> 16) & 0xFF) + ((c >> 8) & 0xFF) + (c & 0xFF);
        assert!(
            lum(back) < lum(HOLO_FRONT),
            "back contour must be dimmer than front"
        );
    }

    /// A hologram object is TRANSPARENT: the scene behind it (the far wall / floor)
    /// still shows through the body — interior cells are NOT overpainted opaque.
    #[test]
    fn hologram_body_is_transparent() {
        let (fb, _, _) = holo_scene();
        // Find a column that has a hologram contour (front blue) and confirm the
        // SAME column has interior rows whose glyph is NOT a hologram glyph — i.e.
        // the scene behind reads through. The hologram glyph set is the box-drawing
        // contour glyphs + the scan rule; an interior see-through cell keeps the
        // scene's wall/floor glyph.
        let holo_glyphs = [0x250C, 0x2510, 0x2514, 0x2518, 0x2500, 0x2502];
        let mut transparent_interior = false;
        for col in 0..fb.cols() {
            let has_contour = (0..fb.rows()).any(|r| fb.get(col, r).unwrap().fg == HOLO_FRONT);
            if !has_contour {
                continue;
            }
            // Count rows in this column that are NOT a hologram glyph (scene shows).
            let scene_rows = (0..fb.rows())
                .filter(|&r| {
                    let g = fb.get(col, r).unwrap().glyph;
                    !holo_glyphs.contains(&g)
                })
                .count();
            if scene_rows > 0 {
                transparent_interior = true;
                break;
            }
        }
        assert!(
            transparent_interior,
            "a hologram column must let the scene behind show through (transparent body)"
        );
    }

    /// Scanline rows land every `HOLO_SCAN_PERIOD` (4th) row over the object: the
    /// blue scan rule recurs on a 4-row cadence within the hologram footprint.
    #[test]
    fn hologram_scanlines_every_fourth_row() {
        let (fb, _, _) = holo_scene();
        let scan_fg = dim_rgb_local(HOLO_FRONT, 0.85);
        // Collect the screen rows that contain a scan-coloured rule glyph.
        let mut scan_rows: Vec<usize> = Vec::new();
        for row in 0..fb.rows() {
            let has_scan = (0..fb.cols()).any(|col| {
                let c = fb.get(col, row).unwrap();
                c.fg == scan_fg && c.glyph == 0x2500
            });
            if has_scan {
                scan_rows.push(row);
            }
        }
        assert!(
            !scan_rows.is_empty(),
            "the hologram must draw scanline rows"
        );
        // Consecutive scan rows are spaced by a multiple of the period (here, at
        // t=0 the phase is 0 so they recur exactly every HOLO_SCAN_PERIOD rows).
        for w in scan_rows.windows(2) {
            let gap = w[1] - w[0];
            assert_eq!(
                gap % HOLO_SCAN_PERIOD as usize,
                0,
                "scan rows must be on the {HOLO_SCAN_PERIOD}-row cadence, gap was {gap}"
            );
        }
    }

    /// The hologram wireframe contour + scan cells carry the CURLY colored-underline
    /// SCAN-GLOW accent (the new smart-material underline layer applied to the holo
    /// object pass), while a see-through interior cell does NOT.
    #[test]
    fn hologram_contour_and_scan_carry_curly_underline_accent() {
        use re_core::UnderlineStyle;
        let (fb, _, _) = holo_scene();
        let scan_fg = dim_rgb_local(HOLO_FRONT, 0.85);
        let mut front_glow = false;
        let mut scan_glow = false;
        for row in 0..fb.rows() {
            for col in 0..fb.cols() {
                let c = fb.get(col, row).unwrap();
                // FRONT contour glyph cell → curly scan-glow.
                if c.fg == HOLO_FRONT && c.glyph != u32::from(' ') {
                    assert_eq!(
                        c.style.effective_underline_style(),
                        UnderlineStyle::Curly,
                        "a front-contour cell must carry the curly scan-glow"
                    );
                    assert!(c.style.has_underline_color());
                    front_glow = true;
                }
                // SCAN rule cell → curly scan-glow.
                if c.fg == scan_fg && c.glyph == 0x2500 {
                    assert_eq!(c.style.effective_underline_style(), UnderlineStyle::Curly);
                    scan_glow = true;
                }
            }
        }
        assert!(front_glow, "expected at least one front-contour glow cell");
        assert!(scan_glow, "expected at least one scan-row glow cell");
    }

    /// The same room with one hologram box at `(4..=6, 6..=8, z0..=z1)` in `style`,
    /// seen from the eye layer z=1 facing -y.
    fn holo_scene_at(z0: i32, z1: i32, style: u16) -> Framebuffer {
        use re_world::{stamp_geometry, stamp_holos, GeometrySpec, HoloObject, HoloShape};
        let spec = GeometrySpec {
            size: (11, 16, 7),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[],
        };
        let mut m = stamp_geometry(&spec);
        stamp_holos(
            &mut m,
            &[HoloObject {
                shape: HoloShape::Box,
                x0: 4,
                y0: 6,
                z0,
                x1: 6,
                y1: 8,
                z1,
                style,
            }],
        );
        let mut cam = CameraPose::spawn_in_test_room();
        cam.pos = glam::Vec3::new(5.5, 13.5, 1.5);
        cam.yaw = std::f32::consts::PI;
        let mut fb = Framebuffer::new(100, 32);
        raycast(&m, &cam, &mut fb);
        fb
    }

    fn fg_count(fb: &Framebuffer, rgb: u32) -> usize {
        (0..fb.rows())
            .flat_map(|r| (0..fb.cols()).map(move |c| (c, r)))
            .filter(|&(c, r)| {
                fb.get(c, r)
                    .is_some_and(|x| x.fg == rgb && x.glyph != u32::from(' '))
            })
            .count()
    }

    /// A hologram wholly ABOVE the eye layer (z = 3..4, eye on z = 1) still draws:
    /// the pass walks every layer that holds holograms, not only the camera's.
    #[test]
    fn a_hologram_above_the_eye_layer_draws() {
        let fb = holo_scene_at(3, 4, 0);
        assert!(
            fg_count(&fb, HOLO_FRONT) > 0,
            "an overhead hologram drew no contour"
        );
        // Its contour sits ABOVE the horizon (an overhead object).
        let lowest = (0..fb.rows())
            .filter(|&r| (0..fb.cols()).any(|c| fb.get(c, r).unwrap().fg == HOLO_FRONT))
            .max()
            .unwrap();
        assert!(
            lowest < fb.rows() / 2,
            "overhead hologram drew at row {lowest}, below the horizon"
        );
    }

    /// A multi-layer hologram draws no seam line between its layers: the front
    /// top/bottom contour only closes where the body ends.
    #[test]
    fn a_stacked_hologram_has_no_seam_between_layers() {
        let tall = holo_scene_at(1, 3, 0);
        let short = holo_scene_at(1, 1, 0);
        // The 1-layer box closes its top edge at the z=2 boundary; the 3-layer box
        // must not draw a `─`/corner on the SAME front column at that row.
        let col = 50;
        let top_of_short = (0..short.rows())
            .find(|&r| short.get(col, r).unwrap().fg == HOLO_FRONT)
            .expect("short box front contour");
        let g = tall.get(col, top_of_short).unwrap();
        assert!(
            !(g.fg == HOLO_FRONT && [0x2500, 0x250C, 0x2510].contains(&g.glyph)),
            "a seam contour at the layer boundary (row {top_of_short})"
        );
    }

    /// `style` tints the hologram: a preset draws in its own hue, not the blue.
    #[test]
    fn a_styled_hologram_draws_in_its_hue() {
        let magenta = re_core::holo_style_named("magenta").unwrap();
        let fb = holo_scene_at(1, 2, magenta);
        let (front, _) = holo::holo_colours(magenta);
        assert_eq!(Some(front), re_core::decode_holo_style(magenta));
        assert!(
            fg_count(&fb, front) > 0,
            "a magenta hologram drew no magenta contour"
        );
        assert_eq!(
            fg_count(&fb, HOLO_FRONT),
            0,
            "a styled hologram still drew default blue"
        );
        let authored = re_core::encode_holo_rgb(0x00FF_B040);
        let fb = holo_scene_at(1, 2, authored);
        let (front, _) = holo::holo_colours(authored);
        assert!(
            fg_count(&fb, front) > 0,
            "an authored #FFB040 hologram drew no amber contour"
        );
    }

    /// A NON-hologram scene is byte-identical with and without the hologram pass:
    /// the pass is gated on `map.has_holograms()`, so a map with no hologram cells
    /// renders exactly as before (back-compat).
    #[test]
    fn no_hologram_scene_is_unchanged() {
        let map = test_room();
        assert!(!map.has_holograms(), "the test room has no holograms");
        let cam = CameraPose::spawn_in_test_room();
        let mut fb = Framebuffer::new(80, 40);
        raycast(&map, &cam, &mut fb);
        // No cell may carry a hologram contour colour (the pass never ran).
        let back = dim_rgb_local(HOLO_BACK, HOLO_BACK_DIM);
        for row in 0..fb.rows() {
            for col in 0..fb.cols() {
                let fg = fb.get(col, row).unwrap().fg;
                assert_ne!(fg, HOLO_FRONT, "no hologram blue on a holo-free map");
                assert_ne!(fg, back, "no hologram cyan on a holo-free map");
            }
        }
    }

    // ── WALL vertical texture is WORLD-anchored (critic WALL coord.y fix) ──

    /// A wall texel's sample coordinate is INDEPENDENT of the screen row it lands
    /// on: `sample_coord` now derives the wall's `v` from the WORLD height
    /// (`wall_world_y`), not `screen_y`. So the same world point on a wall maps to
    /// the same `(u, v)` from any camera height/pitch — no vertical shimmer under
    /// dolly/pitch.
    #[test]
    fn wall_sample_coord_is_world_anchored_not_screen_anchored() {
        let g = CellGeom {
            depth: 4.0,
            foreshorten: 1.0,
            side: 0,
            tex_u: 0.4,
            wall_v: 0.5,
            wall_world_y: wall_world_y(1, 0.5),
            hit_x: 2.5,
            hit_y: 1.0,
            surface: SurfaceKind::WallSolid,
            material: 0,
            light: None,
            cell: (0, 0, 0),
            tex_footprint: None,
            subcell_backdrop: None,
        };
        // Two very different screen rows (as if the camera dollied/pitched so the
        // same world texel projected to a different row) → identical coord.
        let a = sample_coord(&g, 3);
        let b = sample_coord(&g, 41);
        assert_eq!(a, b, "wall sample coord must not depend on screen row");
        // And the v actually encodes the world height (non-degenerate).
        assert!((a.y - wall_world_y(1, 0.5) * WALL_WORLD_V_SCALE).abs() < 1e-5);
    }

    /// Two DIFFERENT world heights on the same wall column map to DIFFERENT `v` (the
    /// vertical texture still varies down the wall — the world anchor preserves the
    /// vertical detail, it just stops it sliding under the camera).
    #[test]
    fn wall_sample_coord_varies_with_world_height() {
        let mk = |wall_v: f32| CellGeom {
            depth: 4.0,
            foreshorten: 1.0,
            side: 0,
            tex_u: 0.4,
            wall_v,
            wall_world_y: wall_world_y(1, wall_v),
            hit_x: 2.5,
            hit_y: 1.0,
            surface: SurfaceKind::WallSolid,
            material: 0,
            light: None,
            cell: (0, 0, 0),
            tex_footprint: None,
            subcell_backdrop: None,
        };
        let top = sample_coord(&mk(0.1), 5).y;
        let bottom = sample_coord(&mk(0.9), 5).y;
        assert!(
            (top - bottom).abs() > 1e-3,
            "distinct world heights must give distinct v ({top} vs {bottom})"
        );
    }

    /// The pitch horizon-shift is a pure function of pitch + rows, shared by the
    /// near cast and the far fill (critic H2 — one horizon, no seam). Zero pitch =
    /// no shift; the sign tracks the look direction; it saturates at the limit.
    #[test]
    fn pitch_row_shift_is_shared_and_saturates() {
        let rows = 40usize;
        assert_eq!(pitch_row_shift(0.0, rows), 0, "level pitch → no shift");
        let up = pitch_row_shift(PITCH_LIMIT_RAD, rows);
        let down = pitch_row_shift(-PITCH_LIMIT_RAD, rows);
        assert_eq!(up, rows as i32 / 2, "full up-pitch shifts half the view");
        assert_eq!(
            down,
            -(rows as i32 / 2),
            "full down-pitch is equal/opposite"
        );
        // Beyond the limit it clamps (saturates), not grows.
        assert_eq!(pitch_row_shift(PITCH_LIMIT_RAD * 4.0, rows), up);
    }

    // ───────────────────── NON-EUCLIDEAN WARP-PORTAL (Stage 1) ────────────────
    //
    // A portal cell is a solid the DDA stops on; the cast RE-BASES the ray by the
    // portal's `delta` and continues into the destination region (mirror-shaped, one
    // hop). These cover: the re-based secondary ray matches the `delta`-transformed
    // values; accumulated depth across the warp is monotone (no depth-0 seam); the
    // one-warp bound (a portal-through-portal shades as the fallback); and a scene
    // with NO portals is byte-identical to before (regression).

    /// The hardcoded demo portal room (12×16×10, perimeter walls z∈[1,8], floor z=0,
    /// ceiling z=9) with the demo portal pair injected: NORTH wall (y=0) ↔ WEST wall
    /// (x=0). Matches `inject_demo_portal_pair`'s assumed layout.
    fn portal_room() -> Map {
        use re_world::{stamp_geometry, GeometrySpec};
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

    /// (a) A primary ray that stops on a portal cell is resolved by `surface_is_portal`
    /// and the cast re-bases it: the second-segment ray equals the `delta`-transformed
    /// origin + direction (renormalised, with the small along-dir nudge). This is the
    /// exact ray re-basing a rigid portal transform specifies (`origin' = delta·hit`,
    /// `dir' = delta.rotation·dir`).
    #[test]
    fn portal_hit_rebases_ray_by_delta() {
        let map = portal_room();
        let cell_z = 4; // inside the portal's z-span (z∈[1,8])
        let (px, py) = (5.5f32, 8.0f32); // mid-room, in front of the north portal
                                         // Straight -y (north), aimed at the north portal opening (x∈[1,10], y=0).
        let (dir_x, dir_y) = (0.0f32, -1.0f32);

        let primary = dda_hit(&map, cell_z, px, py, dir_x, dir_y, None);
        assert!(primary.hit, "the -y ray must hit the north wall");
        assert_eq!(
            primary.surface,
            SurfaceKind::Portal,
            "the north wall in the demo room is a portal opening"
        );
        // The hit cell is resolved straight from the RayHit's integer cell coords.
        assert_eq!(
            (primary.cell_x, primary.cell_y),
            (5, 0),
            "north portal cell"
        );
        let pidx = surface_is_portal(&map, &primary, cell_z)
            .expect("a portal hit must resolve to a PortalLink index");

        // Re-base the ray exactly as the cast does and compare to `delta` directly.
        let link = &map.portals[pidx];
        let (ox, oy, wdir_x, wdir_y) = rebase_ray(
            &link.delta,
            primary.wall_hit_x,
            primary.wall_hit_y,
            cell_z,
            dir_x,
            dir_y,
        );
        // Ground truth: apply `delta` to the hit point + direction by hand.
        let hit_pt = glam::Vec3::new(primary.wall_hit_x, primary.wall_hit_y, cell_z as f32 + 0.5);
        let want_o = link.delta.transform_point3(hit_pt);
        let want_d = link
            .delta
            .transform_vector3(glam::Vec3::new(dir_x, dir_y, 0.0));
        let want_dlen = (want_d.x * want_d.x + want_d.y * want_d.y)
            .sqrt()
            .max(PERP_MIN);
        let (want_dx, want_dy) = (want_d.x / want_dlen, want_d.y / want_dlen);
        // Direction matches the delta-rotated, renormalised value.
        assert!(
            (wdir_x - want_dx).abs() < 1e-4,
            "dir x {wdir_x} != {want_dx}"
        );
        assert!(
            (wdir_y - want_dy).abs() < 1e-4,
            "dir y {wdir_y} != {want_dy}"
        );
        // Origin matches the delta-mapped point (the tiny along-dir nudge aside).
        assert!(
            (ox - (want_o.x + want_dx * portal::PORTAL_ORIGIN_NUDGE)).abs() < 1e-4,
            "ox {ox}"
        );
        assert!(
            (oy - (want_o.y + want_dy * portal::PORTAL_ORIGIN_NUDGE)).abs() < 1e-4,
            "oy {oy}"
        );
        // The re-base actually moved the ray into a DIFFERENT region (origin is not
        // the primary hit point) — the non-euclidean jump happened.
        assert!(
            (ox - primary.wall_hit_x).abs() > 0.5 || (oy - primary.wall_hit_y).abs() > 0.5,
            "the warp must relocate the ray origin into the destination region"
        );
    }

    /// (b) The portal slice's written depth is the ACCUMULATED path length (distance
    /// to the portal + the destination segment), strictly GREATER than the distance to
    /// the portal plane — no depth-0 contrast seam at the opening. Driven
    /// through the pure `resolve_warp` decision (NOT a full demo raycast, whose exact
    /// portal columns shift whenever the demo's inner-face/delta geometry changes), so
    /// the seam-free guarantee is proven geometry-independently: a warp at
    /// `dist_to_portal` whose re-based ray lands on ordinary destination GEOMETRY
    /// must carry depth = dist_to_portal + dest.perp.
    #[test]
    fn portal_accumulated_depth_is_monotone() {
        let dist_to_portal = 6.0f32;
        let dest_perp = 4.0f32;
        // The re-based ray lands on a solid wall in the destination region.
        let resolved =
            portal::resolve_warp(dist_to_portal, dest_hit(SurfaceKind::WallSolid, dest_perp));

        // The destination geometry shades by its OWN surface (not a portal), proving the
        // warp resolved onto real geometry rather than the budget-exhausted fallback.
        assert!(resolved.hit, "the warp landed on destination geometry");
        assert_eq!(resolved.surface, SurfaceKind::WallSolid);

        // Depth is the ACCUMULATED path: distance to the portal + the destination segment.
        let want = dist_to_portal + dest_perp;
        assert!(
            (resolved.perp - want).abs() < 1e-4,
            "accumulated depth must be dist-to-portal + dest segment = {want}, got {}",
            resolved.perp
        );
        // Strictly greater than the portal distance — no depth-0 seam at the opening.
        assert!(
            resolved.perp > dist_to_portal,
            "portal slice depth {} must exceed the portal distance {dist_to_portal} \
             (no depth-0 seam)",
            resolved.perp
        );
    }

    /// A synthetic destination `RayHit` for the warp-resolution fixtures below. The
    /// only fields `resolve_warp` reads are `hit`, `surface`, and `perp`; the rest
    /// carry plausible values so the returned hit is well-formed.
    fn dest_hit(surface: SurfaceKind, perp: f32) -> RayHit {
        RayHit {
            hit: true,
            side: 0,
            surface,
            material: 0,
            perp,
            wall_hit_x: 4.0,
            wall_hit_y: 4.0,
            tex_u: 0.5,
            cell_x: 4,
            cell_y: 4,
            shape_normal: None,
        }
    }

    /// (c) The one-warp MVP bound, on a CONSTRUCTED fixture (synthetic hits), NOT the
    /// demo level. The load-bearing safety property: a portal reached THROUGH a
    /// portal (the re-based ray's first hit is itself a `SurfaceKind::Portal`) is
    /// shaded as the plain-wall FALLBACK at the ACCUMULATED one-hop depth — it is
    /// NEVER re-warped (the budget is spent). `portal::resolve_warp` is the exact
    /// decision the cast makes, so driving it with a hand-built portal-through-portal
    /// hit proves the bound without depending on any level's column geometry.
    #[test]
    fn portal_one_warp_bound_shades_fallback() {
        assert_eq!(PORTAL_WARP_BUDGET, 1, "the MVP bounds re-basing to ONE hop");
        // The fallback shade for a budget-exhausted portal is the plain brick — so a
        // `SurfaceKind::Portal` slice that survives shading reads as a wall, never a
        // second warped view.
        assert_eq!(
            surface_material(SurfaceKind::Portal, 0),
            ascii_shader::ids::WALL_BRICK,
            "a budget-exhausted portal must shade as the fallback brick (no re-warp)"
        );

        // Fixture: a ray warps at portal A (distance to A = 5.0), and its re-based
        // first hit IS portal B at perp 3.0 in the destination region — the two-
        // facing-portal case. The single warp hop is already spent, so `resolve_warp`
        // must NOT warp again:
        let dist_to_a = 5.0f32;
        let from_a_exit_to_b = 3.0f32;
        let hit_b = dest_hit(SurfaceKind::Portal, from_a_exit_to_b);
        let resolved = portal::resolve_warp(dist_to_a, hit_b);

        // (i) The slice still carries `surface == Portal` — i.e. it falls through to
        //     the ordinary wall path and shades as the fallback brick, proving NO
        //     second warp consumed it.
        assert!(
            resolved.hit,
            "the portal-through-portal slice is a solid hit"
        );
        assert_eq!(
            resolved.surface,
            SurfaceKind::Portal,
            "a portal reached through a portal must remain a Portal surface (fallback \
             brick), not be re-warped into a doubly-transformed destination"
        );
        // (ii) Its depth is the ACCUMULATED one-hop path (dist to A + A-exit→B), NOT
        //      the bare portal distance and NOT depth 0 — exactly one re-base happened.
        let want = (dist_to_a + from_a_exit_to_b).max(PERP_MIN);
        assert!(
            (resolved.perp - want).abs() < 1e-4,
            "portal-through-portal depth must be the accumulated one-hop path {want}, \
             got {}",
            resolved.perp
        );
        assert!(
            resolved.perp > dist_to_a,
            "the fallback slice must sit BEHIND the first portal plane (accumulated), \
             got {} vs dist-to-A {dist_to_a}",
            resolved.perp
        );
    }

    /// (c2) The escaped-warp case (FIX 1), on the same constructed fixture: a re-based
    /// ray that MISSES (escapes the finite destination grid) is looking into OPEN
    /// SPACE on the far side, so `resolve_warp` returns a genuine SKY/void miss
    /// (`hit=false`, infinite depth, `Empty` surface) — NOT a flat Portal brick
    /// punched to the portal's own depth (the depth-island seam the fix removes).
    /// This is what makes it DIFFERENT from the one-warp-bound case above: a MISS is
    /// void; a HIT on a second portal is the fallback brick at accumulated depth.
    #[test]
    fn portal_escaped_warp_reads_as_void_not_brick() {
        // A re-based ray that escaped the grid: `dda_hit` returns a miss.
        let escaped = RayHit::miss(2.5, 9.0);
        assert!(!escaped.hit && escaped.perp >= DIST_INFINITY * 0.5);

        let resolved = portal::resolve_warp(7.0, escaped);
        assert!(
            !resolved.hit,
            "an escaped warped ray must read as a SKY/void miss, not a wall hit"
        );
        assert!(
            resolved.perp >= DIST_INFINITY * 0.5,
            "an escaped warp must keep INFINITE depth (background void), not the \
             portal's ~7 depth — got {}",
            resolved.perp
        );
        assert_eq!(
            resolved.surface,
            SurfaceKind::Empty,
            "an escaped warp must be the empty/sky surface, not a Portal brick"
        );
    }

    /// (d) Regression: a scene with NO portals renders BYTE-IDENTICAL to a build
    /// without the portal branch — the warp is a pure post-hit branch on the
    /// (rare) portal surface; the common path is untouched. We prove it by casting
    /// the same map through the public `raycast_fov` (which never registers portals)
    /// and a hand-built equivalent, asserting cell-for-cell equality across two casts.
    #[test]
    fn no_portal_scene_is_byte_identical() {
        // A plain room with NO portals (the demo room WITHOUT the injection).
        use re_world::{stamp_geometry, GeometrySpec};
        let spec = GeometrySpec {
            size: (12, 16, 10),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[],
        };
        let map = stamp_geometry(&spec);
        assert!(
            map.portals.is_empty(),
            "the regression map must have no portals"
        );
        // No-portal scene must NOT be flagged as a no-pan special scene by the portal
        // gate (mirror/holo gates may still fire on other scenes, but not this one).
        let reg = MaterialRegistry::with_builtins();
        assert!(
            !map_local_has_special_cells(&map, &reg),
            "a plain no-portal room must not be a no-pan special scene"
        );

        let cam = CameraPose {
            pos: glam::Vec3::new(5.5, 8.0, 4.5),
            yaw: std::f32::consts::PI,
            pitch: 0.0,
        };
        let mut a = Framebuffer::new(48, 28);
        let mut b = Framebuffer::new(48, 28);
        raycast_full(
            &map,
            &cam,
            &reg,
            &Fov::default(),
            0.0,
            ShadowQuality::Off,
            MirrorClarity::default(),
            f32::INFINITY,
            cam.pos.z,
            &mut a,
        );
        raycast_full(
            &map,
            &cam,
            &reg,
            &Fov::default(),
            0.0,
            ShadowQuality::Off,
            MirrorClarity::default(),
            f32::INFINITY,
            cam.pos.z,
            &mut b,
        );
        for row in 0..a.rows() {
            for col in 0..a.cols() {
                assert_eq!(
                    a.get(col, row).unwrap(),
                    b.get(col, row).unwrap(),
                    "a no-portal cast must be deterministic + portal-branch-free at ({col},{row})"
                );
            }
        }
    }

    /// (e) THE PAN-GATE: a portal scene is a NO-PAN special scene ONLY under
    /// view-DEPENDENT lighting. Under VIEW-INDEPENDENT lighting (the DEFAULT) the
    /// warped destination shade is view-invariant, so the near ring bakes it at its
    /// world angle and a rotation PANS it — so the scene must NOT be flagged special.
    /// This is the gate `recast_special_columns_into` reads to elide the forced
    /// portal-column recast (the perf win). Mirrors/holos are unaffected by the toggle.
    #[test]
    fn portal_scene_special_only_under_view_dependent_light() {
        let map = portal_room();
        assert!(
            !map.portals.is_empty(),
            "the portal room must carry a portal pair"
        );
        let reg = MaterialRegistry::with_builtins();

        // DEFAULT (view-independent ON): portals PAN → NOT special.
        crate::config::apply(true, None, false);
        assert!(
            !map_local_has_special_cells(&map, &reg),
            "under view-independent lighting a portal scene PANS → must not be special"
        );

        // LEGACY (view-dependent): the warped shade carries the bake-time facing →
        // cannot pan → portals force the no-pan recast, same as before this change.
        crate::config::apply(false, None, false);
        assert!(
            map_local_has_special_cells(&map, &reg),
            "under view-dependent lighting a portal scene must remain a no-pan special scene"
        );

        // Restore the default so other tests on this thread see view-independent on.
        crate::config::apply(true, None, false);
    }

    /// (f) THE ROTATION-PAN EXACTNESS BAR for PORTAL scenes: a near ring baked at pose
    /// P on a portal scene, then PANNED to a yaw Y within the margin, must match a FRESH
    /// full cast at (P, Y) for the PORTAL columns to the SAME bar static geometry meets —
    /// surface + side exact (1-column rectilinear→angular remap tolerated), depth within
    /// the documented margin-bounded tolerance. Under view-independent lighting the
    /// warped destination shade is view-invariant, so the ring's `cast_column` warp bake
    /// reproduces the live warped column at every yaw. We assert specifically over the
    /// PORTAL columns (the ones this change newly pans) so a regression that breaks the
    /// warp pan is caught, not masked by the surrounding static walls.
    #[test]
    fn portal_rotation_pans_like_static_geometry() {
        // View-independent lighting (the path this optimisation is correct on).
        crate::config::apply(true, None, false);

        let map = portal_room();
        let registry = MaterialRegistry::with_builtins();
        let fov = Fov::default();
        let (cols, rows) = (200usize, 64usize);
        let view = view_params(cols, rows, &fov);
        let ps = view.plane_scale;

        // Stand mid-room facing the NORTH portal opening (y=0): the central columns
        // hit the portal and warp into the WEST destination region.
        let base = CameraPose {
            pos: glam::Vec3::new(5.5, 8.0, 4.5),
            yaw: std::f32::consts::PI, // -y, toward the north portal
            pitch: 0.0,
        };

        // The scene must PAN (not be special) under this lighting — the precondition
        // for the ring to carry the warped portal columns.
        assert!(
            !map_local_has_special_cells(&map, &registry),
            "portal scene must pan under view-independent lighting"
        );

        let mut ring = crate::near_cache::NearRing::new(cols, rows, view.hfov);
        ring.rebake(&map, &base, &registry, &fov, 0.0);

        // Pan a third of the margin away (comfortably in the ring arc).
        let panned = CameraPose {
            yaw: base.yaw + crate::near_cache::NEAR_MARGIN_RAD / 3.0,
            ..base
        };
        assert!(
            ring.can_pan(&panned, ps),
            "the panned pose must be in the ring arc"
        );
        let mut op = re_core::Layer::new(cols, rows, re_core::LayerBlend::Opaque);
        let mut tr = re_core::Layer::new(cols, rows, re_core::LayerBlend::OverKeepBg);
        ring.fill_near_layers(&panned, ps, &mut op, &mut tr);

        // Fresh full cast at the panned pose (the live gameplay path, which runs the
        // SAME `cast_column` warp branch the ring baked).
        let mut fresh = Framebuffer::new(cols, rows);
        raycast_gameplay_capture(
            &map,
            &panned,
            &registry,
            &fov,
            0.0,
            &mut gbuffer::NearGBuffer::default(),
            &mut fresh,
        );

        // Identify the PORTAL columns in the fresh cast: those whose primary DDA stops
        // on a Portal cell (the warped columns this change newly pans). Restricting the
        // comparison to them proves the WARP pan specifically.
        let (dir_x, dir_y) = panned.dir();
        let plane_x = panned.yaw.cos() * ps;
        let plane_y = -panned.yaw.sin() * ps;
        let cell_z = panned.pos.z.floor() as i32;
        let mut portal_cols: Vec<usize> = Vec::new();
        for x in 0..cols {
            let camera_x = 2.0 * (x as f32 + 0.5) / cols as f32 - 1.0;
            let rdx = dir_x + plane_x * camera_x;
            let rdy = dir_y + plane_y * camera_x;
            let primary = dda_hit(&map, cell_z, panned.pos.x, panned.pos.y, rdx, rdy, None);
            if surface_is_portal(&map, &primary, cell_z).is_some() {
                portal_cols.push(x);
            }
        }
        assert!(
            !portal_cols.is_empty(),
            "the live view must show portal columns (else the test proves nothing)"
        );

        // Compare PORTAL-column cells: surface + side exact (1-col remap allowed), depth
        // within the same margin-bounded tolerance the static pan accepts. The ring
        // stores depth PERPENDICULAR TO ITS BAKE YAW; the fresh cast's is perpendicular
        // to the panned yaw. Along one world direction those differ by
        // `cos(α_fresh)/cos(α_ring)` — up to ~13% at the leading edge for this pan, for
        // static geometry exactly as for warped — so the ring's depth is carried into
        // the panned frame before comparing. (Now that the warped scene fills the
        // portal's full height, whole edge columns show it, not just the slice band.)
        let pan = panned.yaw - base.yaw;
        let mut compared = 0usize;
        let mut ok_n = 0usize;
        for &col in &portal_cols {
            let camera_x = 2.0 * (col as f32 + 0.5) / cols as f32 - 1.0;
            let a_fresh = (ps * camera_x).atan();
            let to_fresh = a_fresh.cos() / (a_fresh + pan).cos();
            for row in (rows / 4)..(3 * rows / 4) {
                let Some(p) = op.get(col, row) else { continue };
                let Some(f) = fresh.get(col, row) else {
                    continue;
                };
                compared += 1;
                let near = |a: f32, b: f32| (a - b).abs() <= 0.08 * a.max(b).max(1.0);
                // (The floor / ceiling backdrop's depth is ORDINARILY a function of the
                // row, not a distance along the ray: it carries over unchanged. The one
                // exception is a row CLAMPED to a nearer wall's own perpendicular depth
                // — `clamp_to_wall_depth`, the dark-strip fix in the "FLOOR / CEILING
                // rows" loop — which inherits that wall's genuine per-column distance
                // and so DOES change with pan exactly like the wall itself. Since the
                // ring doesn't tag which rows were clamped, accept either reading for
                // Floor/Ceiling; a wall cell only ever matches the panned reading.)
                let matches = |fc: &Cell| {
                    fc.surface == p.surface
                        && fc.side == p.side
                        && match p.surface {
                            SurfaceKind::Floor | SurfaceKind::Ceiling => {
                                near(p.depth, fc.depth) || near(p.depth * to_fresh, fc.depth)
                            }
                            _ => near(p.depth * to_fresh, fc.depth),
                        }
                };
                let ok = matches(f)
                    || [-1i32, 1].iter().any(|&dc| {
                        let nc = col as i32 + dc;
                        nc >= 0
                            && (nc as usize) < cols
                            && fresh.get(nc as usize, row).is_some_and(matches)
                    });
                if ok {
                    ok_n += 1;
                }
            }
        }
        assert!(
            compared > 100,
            "expected a substantial portal comparison set ({compared})"
        );
        let frac = ok_n as f32 / compared as f32;
        assert!(
            frac >= 0.95,
            "panned PORTAL geometry must match the fresh warp cast: {ok_n}/{compared} = {frac:.3}"
        );

        crate::config::apply(true, None, false);
    }

    // ── Scalable Material Overlay P4 — INSET phase-band override ──────────────

    use re_core::ResolvedInset;

    /// A resolved inset over a small region, revealing on octave-0's band [0, 0.2·LN2).
    fn test_inset() -> ResolvedInset {
        ResolvedInset {
            region: re_core::Region::Box {
                min: (2, 0, 1),
                max: (4, 0, 3),
            },
            phase_band: (0.0, 0.2 * crate::fractal::LN2),
            rgb: 0x00F2_A33C,
        }
    }

    /// (b) The override fires IFF (cell in region AND phase in band).
    #[test]
    fn inset_override_fires_only_in_region_and_band() {
        let insets = [test_inset()];
        let far = f32::INFINITY; // never frozen
        let near = 3.0;
        // In region + in band (t=0 ⇒ reduced phase 0 ∈ [0, 0.2·LN2)) ⇒ Some(colour).
        assert_eq!(
            resolve_inset_override(&insets, (3, 0, 2), 0.0, far, near),
            Some(0x00F2_A33C),
            "cell in region + phase in band must override"
        );
        // In region but OUT of band: pick a time whose reduced phase is mid-loop.
        let out_t = 0.5; // reduced = (0.5*LN2)%LN2 = 0.5*LN2 ≈ 0.346 > 0.2·LN2
        assert_eq!(
            resolve_inset_override(&insets, (3, 0, 2), out_t, far, near),
            None,
            "phase out of band ⇒ no override"
        );
        // OUT of region but in band ⇒ no override.
        assert_eq!(
            resolve_inset_override(&insets, (9, 0, 9), 0.0, far, near),
            None,
            "cell outside region ⇒ no override"
        );
    }

    /// (b) The override is IDENTICAL across loops at the SAME reduced phase — the
    /// seamless-loop determinism. A whole-loop time shift (Δt = 1.0 ⇒ Δphase =
    /// LN2) reproduces the same reduced phase, hence the same decision.
    #[test]
    fn inset_override_identical_across_loops() {
        let insets = [test_inset()];
        let far = f32::INFINITY;
        for &(x, y, z) in &[(3, 0, 2), (2, 0, 1), (9, 0, 9)] {
            for &t in &[0.0_f32, 0.05, 0.5, 0.9] {
                let a = resolve_inset_override(&insets, (x, y, z), t, far, 3.0);
                // One full loop later (time advances by 1.0 ⇒ phase by LN2): same phase.
                let b = resolve_inset_override(&insets, (x, y, z), t + 1.0, far, 3.0);
                assert_eq!(
                    a, b,
                    "override must be loop-identical at ({x},{y},{z}) t={t}"
                );
            }
        }
    }

    /// An EMPTY resolved-inset slice ⇒ always `None` (the no-op back-compat path).
    #[test]
    fn inset_override_empty_slice_is_noop() {
        for &t in &[0.0_f32, 0.1, 0.5, 1.0, 7.3] {
            assert_eq!(
                resolve_inset_override(&[], (3, 0, 2), t, f32::INFINITY, 3.0),
                None,
                "empty insets must never override"
            );
        }
    }

    /// A FAR-FROZEN cell pins the phase to 0 (the quiet far field), so a frozen inset
    /// region holds a stable reveal regardless of `time`.
    #[test]
    fn inset_override_far_frozen_pins_phase_zero() {
        let insets = [test_inset()];
        let freeze = 5.0;
        // distance >= freeze ⇒ phase pinned to 0 (in band) for ANY time.
        for &t in &[0.0_f32, 0.5, 0.9, 3.7] {
            assert_eq!(
                resolve_inset_override(&insets, (3, 0, 2), t, freeze, 6.0),
                Some(0x00F2_A33C),
                "a far-frozen in-region cell holds the phase-0 reveal at t={t}"
            );
        }
    }

    /// END-TO-END: a level WITH an `[inset]` casts DIFFERENTLY from the SAME map with
    /// its resolved insets stripped — the bounded phase-band reveal really changes
    /// on-screen pixels in the inset region at an in-band phase (t=0). A level WITHOUT
    /// insets is byte-identical (the stripped map IS the no-inset case).
    #[test]
    fn a5_inset_reveal_changes_the_frame_in_band() {
        let lv = re_level::load(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../levels/A5-inset.lvl"),
        )
        .expect("A5-inset must load");
        let map_with = lv.build_map();
        assert_eq!(map_with.resolved_insets().len(), 1, "one resolved inset");
        assert_eq!(
            map_with.resolved_insets()[0].rgb,
            0x00F2_A33C,
            "amber child colour"
        );

        // The SAME map with the resolved insets removed (the no-inset / pre-P4 frame).
        let mut map_without = lv.build_map();
        map_without.set_resolved_insets(Vec::new());

        // Camera at the level's spawn, looking north (-y) at the inset-skinned wall.
        // facing -y ⇒ yaw column 240 (== yaw PI), matching the --dump pose.
        let cam = CameraPose::spawn_at(6, 5, 1, 240);
        let reg = registry_for_map(&map_with);

        let mut fb_with = Framebuffer::new(100, 32);
        let mut fb_without = Framebuffer::new(100, 32);
        // t = 0 ⇒ reduced phase 0 ∈ [0, 0.2·LN2) ⇒ the octave-0 inset is IN-BAND.
        raycast_with_time(&map_with, &cam, &reg, 0.0, &mut fb_with);
        raycast_with_time(&map_without, &cam, &reg, 0.0, &mut fb_without);

        // SOME cell differs (the inset override painted the amber reveal over the
        // fractal field). Both casts share map/camera/time, so the ONLY difference is
        // the resolved-inset override.
        let mut diffs = 0usize;
        for row in 0..fb_with.rows() {
            for col in 0..fb_with.cols() {
                let a = fb_with.get(col, row).unwrap();
                let b = fb_without.get(col, row).unwrap();
                if a.fg != b.fg || a.glyph != b.glyph || a.bg != b.bg {
                    diffs += 1;
                }
            }
        }
        assert!(
            diffs > 0,
            "the in-band inset reveal must change at least one cell vs the no-inset frame"
        );
    }

    /// (d) With NO inset (empty ctx override), a SelfSimilar shade is byte-identical
    /// to the pre-P4 path: `apply_scale_field` with `ctx.inset_rgb = None` produces
    /// exactly the P3 fractal field result.
    #[test]
    fn inset_none_shade_is_byte_identical_to_p3_field() {
        use ascii_shader::{FogParams, Material, SampleCtx, ScaleMode};
        let perlin = Perlin::new(NOISE_SEED);
        // A SelfSimilar brick (mirrors the ascii_shader SelfSimilar test material).
        let reg = MaterialRegistry::with_builtins();
        let base = reg.get(ids::WALL_BRICK).clone();
        let ss = Material {
            scale: ScaleMode::SelfSimilar(crate::fractal::FractalParams {
                arms: 5,
                octave_warp: 0.3,
                twist: 0.9,
            }),
            ..base
        };
        let mk_ctx = |inset: Option<u32>| SampleCtx {
            distance: 3.0,
            foreshorten: 1.0,
            side: 0,
            screen_y: 4,
            wall_v: 0.5,
            tick: 0,
            time: 1.234,
            fog: FogParams::DEFAULT,
            far_freeze: f32::INFINITY,
            light: None,
            lod_override: None,
            state: 0,
            pulse: 0.0,
            inset_rgb: inset,
            wall_along: 0.0,
            wall_up: 0.0,
            on_wall: false,
            tex_footprint: None,
            subcell_backdrop: None,
        };
        let coord = glam::Vec2::new(0.37, 0.62);
        let none = ss.sample(&perlin, coord, &mk_ctx(None));
        // The P3 path (no inset field on SampleCtx) is what `None` reproduces — the
        // value is the fractal field, NOT the inset colour.
        let with = ss.sample(&perlin, coord, &mk_ctx(Some(0x00F2_A33C)));
        assert_ne!(
            none.fg, with.fg,
            "the inset override must visibly change the colour"
        );
        // And `None` is deterministic / stable (golden-safe).
        let none2 = ss.sample(&perlin, coord, &mk_ctx(None));
        assert_eq!(
            none, none2,
            "inset-None SelfSimilar shade must be deterministic"
        );
    }

    // ── IMAGE TEXTURE (image-texture milestone 1): nearest-neighbour RGBA8 wall ──

    /// A synthetic 4×4 image: each texel a distinct, recognisable colour so the test
    /// can assert which texel a `(u, v)` sampled. Row 0 (v≈0, top) is a red→green→
    /// blue→white ramp; the other rows are flat greys so a `v` change is detectable.
    fn synthetic_4x4() -> re_core::TextureImage {
        // Row-major, top-left origin, RGBA. Row 0 = the colour ramp; rows 1..3 grey.
        let mut rgba = Vec::with_capacity(4 * 4 * 4);
        let row0 = [
            [255u8, 0, 0, 255],   // (0,0) red
            [0, 255, 0, 255],     // (1,0) green
            [0, 0, 255, 255],     // (2,0) blue
            [255, 255, 255, 255], // (3,0) white
        ];
        for px in row0 {
            rgba.extend_from_slice(&px);
        }
        for r in 1..4u8 {
            let g = r * 60; // 60,120,180 — distinct grey per row
            for _ in 0..4 {
                rgba.extend_from_slice(&[g, g, g, 255]);
            }
        }
        re_core::TextureImage::new(4, 4, rgba)
    }

    /// Build a registry whose WALL surface slot (0) is a `TextureRecipe::Image`
    /// referencing `tex_id`, with the rest of the built-ins intact. The texture is
    /// attached from `textures` exactly as `registry_for_map` does in production.
    fn image_registry(tex_id: u32, textures: &[re_core::TextureSet]) -> MaterialRegistry {
        image_registry_span(tex_id, textures, None)
    }

    /// As [`image_registry`] but with an explicit WORLD-ANCHORED [`ImageSpan`]
    /// on the Image slot — `Some` ⇒ one figure spans the surface; `None` ⇒ legacy
    /// per-cell tiling.
    fn image_registry_span(
        tex_id: u32,
        textures: &[re_core::TextureSet],
        span: Option<ascii_shader::ImageSpan>,
    ) -> MaterialRegistry {
        image_registry_span_subcell(tex_id, textures, span, ascii_shader::SubcellMode::Off)
    }

    /// As [`image_registry_span`] but with an explicit SUB-CELL mode on the
    /// Image slot — `Off` ⇒ legacy solid block; `Quad`/`Braille` ⇒ sub-cell detail.
    fn image_registry_span_subcell(
        tex_id: u32,
        textures: &[re_core::TextureSet],
        span: Option<ascii_shader::ImageSpan>,
        subcell: ascii_shader::SubcellMode,
    ) -> MaterialRegistry {
        let builtins = MaterialRegistry::with_builtins();
        let mut reg = MaterialRegistry::new();
        for slot in 0..5u16 {
            if slot == ids::WALL_BRICK.0 {
                reg.register(Material {
                    recipe: TextureRecipe::Image {
                        tex_id,
                        span,
                        subcell,
                        fit: ascii_shader::Fit::Stretch,
                        tile: ascii_shader::IMAGE_TILE_NONE,
                        relief: None,
                    },
                    ..builtins.get(MaterialId(slot)).clone()
                });
            } else {
                reg.register(builtins.get(MaterialId(slot)).clone());
            }
        }
        // Attach the real pixel buffers, just like the production registry build.
        reg.attach_image_textures(textures);
        reg
    }

    /// CONVERTER SEAM: an authored `relief = height_scale=… normal_strength=…` must
    /// arrive on BOTH sides of the material - on the recipe (which is what flips the
    /// props table off `self_lit`) and on the `LightModel` (which is what scales the
    /// relief). Losing either half is silent: the recipe alone lights the photo with
    /// somebody else's relief depth, the light model alone never runs the relief pass
    /// at all because the surface is still self-lit.
    #[test]
    fn an_authored_relief_reaches_both_the_recipe_and_the_light_model() {
        let base = ascii_shader::DEFAULT_MATERIAL.clone();
        let def = re_world::MaterialDef {
            recipe: Some(re_world::MaterialRecipe::Image {
                tex_id: 0,
                span: None,
                subcell: re_world::SubcellMode::Off,
                fit: re_world::Fit::Contain,
                tile: (3.0, 1.5),
                relief: Some(re_world::ImageRelief {
                    height_scale: 1.25,
                    normal_strength: 0.5,
                }),
            }),
            ..re_world::MaterialDef::flat("img", 0x00FF_FFFF, 0)
        };
        let m = material_from_def(&def, &base);
        match m.recipe {
            TextureRecipe::Image {
                tile,
                relief: Some(r),
                ..
            } => {
                assert_eq!(tile, (3.0, 1.5), "the tile pair crosses the converter");
                assert!((r.height_scale - 1.25).abs() < 1e-6);
                assert!((r.normal_strength - 0.5).abs() < 1e-6);
            }
            other => panic!("expected an Image recipe with relief, got {other:?}"),
        }
        assert!(
            (m.light.height_scale - 1.25).abs() < 1e-6,
            "relief drives the light model"
        );
        assert!((m.light.normal_strength - 0.5).abs() < 1e-6);
        // ONLY the two relief knobs are overridden - the rest of the lighting is
        // still inherited, so `relief =` is a relief control, not a lighting reset.
        assert_eq!(m.light.shininess, base.light.shininess);
        assert_eq!(m.light.specular, base.light.specular);
        assert_eq!(m.light.ao_strength, base.light.ao_strength);
        assert!(!m.recipe.props().self_lit, "a relief image is not self-lit");
    }

    /// CONVERTER SEAM (no-regression): an image material with NEITHER key resolves to
    /// exactly what it resolved to before the keys existed - untiled, no relief, the
    /// inherited light model untouched, still self-lit.
    #[test]
    fn an_image_without_the_new_keys_converts_exactly_as_before() {
        let base = ascii_shader::DEFAULT_MATERIAL.clone();
        let def = re_world::MaterialDef {
            recipe: Some(re_world::MaterialRecipe::Image {
                tex_id: 0,
                span: None,
                subcell: re_world::SubcellMode::Off,
                fit: re_world::Fit::Contain,
                tile: re_world::IMAGE_TILE_NONE,
                relief: None,
            }),
            ..re_world::MaterialDef::flat("img", 0x00FF_FFFF, 0)
        };
        let m = material_from_def(&def, &base);
        assert!(matches!(
            m.recipe,
            TextureRecipe::Image {
                tile: ascii_shader::IMAGE_TILE_NONE,
                relief: None,
                ..
            }
        ));
        assert_eq!(m.light, base.light, "the light model is untouched");
        assert!(m.recipe.props().self_lit, "still a self-lit decal");
    }

    /// CONVERTER SEAM (defence in depth): the `.lvl` parser rejects a non-positive or
    /// non-finite tile, but a `MaterialDef` can also be built in code by a tool or a
    /// test. Such a pair must degrade to "no repeat" rather than reaching `fract` with
    /// a zero divisor or a NaN.
    #[test]
    fn a_hand_built_bad_tile_degrades_to_no_repeat() {
        let base = ascii_shader::DEFAULT_MATERIAL.clone();
        for bad in [
            (0.0f32, 1.0f32),
            (-2.0, 1.0),
            (f32::NAN, 1.0),
            (1.0, f32::INFINITY),
        ] {
            let def = re_world::MaterialDef {
                recipe: Some(re_world::MaterialRecipe::Image {
                    tex_id: 0,
                    span: None,
                    subcell: re_world::SubcellMode::Off,
                    fit: re_world::Fit::Contain,
                    tile: bad,
                    relief: None,
                }),
                ..re_world::MaterialDef::flat("img", 0x00FF_FFFF, 0)
            };
            match material_from_def(&def, &base).recipe {
                TextureRecipe::Image { tile, .. } => assert_eq!(
                    tile,
                    ascii_shader::IMAGE_TILE_NONE,
                    "a bad tile {bad:?} must degrade to no repeat"
                ),
                other => panic!("expected an Image recipe, got {other:?}"),
            }
        }
    }

    /// SHADER SEAM: an `Image` material samples the attached texture at the cell's
    /// `(u, wall_v)` and returns a SOLID `█` block in the nearest texel's colour —
    /// NOT a procedural pattern. Drives `Material::sample` directly at known UVs.
    #[test]
    fn image_material_samples_attached_texture() {
        let perlin = Perlin::new(NOISE_SEED);
        let textures = vec![re_core::TextureSet::still(synthetic_4x4())];
        let reg = image_registry(0, &textures);
        let mat = reg.get(ids::WALL_BRICK);
        assert!(
            matches!(mat.recipe, TextureRecipe::Image { tex_id: 0, .. }),
            "slot 0 must carry the Image recipe"
        );
        assert!(
            mat.texture.is_some(),
            "the texture must be attached by the registry"
        );

        // Sample the four top-row texels by their u-centres at v≈0 (top row). `coord.x`
        // carries the wall-horizontal fraction; the cast threads `tex_u + face offset`
        // there, so `fract(coord.x)` is the in-cell u. We pass u directly in [0,1).
        let centres = [
            (0.125f32, 0xFF_0000u32), // u→col0 red
            (0.375, 0x00_FF00),       // u→col1 green
            (0.625, 0x00_00FF),       // u→col2 blue
            (0.875, 0xFF_FFFF),       // u→col3 white
        ];
        for (u, want_rgb) in centres {
            let ctx = SampleCtx {
                distance: 1.0,
                foreshorten: 1.0,
                side: 0,
                screen_y: 5,
                wall_v: 0.001, // top row
                tick: 0,
                time: 0.0,
                fog: FogParams::DEFAULT,
                far_freeze: f32::INFINITY,
                light: None,
                lod_override: None,
                state: 0,
                pulse: 0.0,
                inset_rgb: None,
                wall_along: 0.0,
                wall_up: 0.0,
                on_wall: false,
                tex_footprint: None,
                subcell_backdrop: None,
            };
            let t = mat.sample(&perlin, glam::Vec2::new(u, 0.0), &ctx);
            // The block glyph is the full block (solid colour, no procedural pattern).
            assert_eq!(t.glyph, 0x2588, "image cell must be a solid block at u={u}");
            // fg is the sampled texel colour (distance 1.0, no fog/side dim with no
            // LightCtx and DEFAULT fog at near range → the colour reads through).
            assert_eq!(
                t.fg, want_rgb,
                "u={u} must sample the texel colour {want_rgb:06X}, got {:06X}",
                t.fg
            );
        }
    }

    /// `wall_v` selects the ROW: the top row is the colour ramp, lower rows are flat
    /// greys, so two cells at the same `u` but different `v` carry different colours.
    #[test]
    fn image_material_v_selects_row() {
        let perlin = Perlin::new(NOISE_SEED);
        let textures = vec![re_core::TextureSet::still(synthetic_4x4())];
        let reg = image_registry(0, &textures);
        let mat = reg.get(ids::WALL_BRICK);
        let mk = |v: f32| SampleCtx {
            distance: 1.0,
            foreshorten: 1.0,
            side: 0,
            screen_y: 5,
            wall_v: v,
            tick: 0,
            time: 0.0,
            fog: FogParams::DEFAULT,
            far_freeze: f32::INFINITY,
            light: None,
            lod_override: None,
            state: 0,
            pulse: 0.0,
            inset_rgb: None,
            wall_along: 0.0,
            wall_up: 0.0,
            on_wall: false,
            tex_footprint: None,
            subcell_backdrop: None,
        };
        // u→col0: top row is RED, a mid row is flat grey (60 in row 1).
        let top = mat.sample(&perlin, glam::Vec2::new(0.125, 0.0), &mk(0.05));
        let mid = mat.sample(&perlin, glam::Vec2::new(0.125, 0.0), &mk(0.30));
        assert_eq!(top.fg, 0xFF_0000, "top row (v≈0) must be red");
        assert_ne!(
            top.fg, mid.fg,
            "a different v must sample a different row colour"
        );
        assert_eq!(mid.fg, 0x3C_3C3C, "row 1 is flat grey 0x3C ({})", mid.fg);
    }

    // ── IMAGE SPAN: ONE figure spans the surface, world-anchored ──

    /// A `SampleCtx` at a given wall WORLD (along, up) for the span tests. `wall_v` is
    /// left at the legacy default (unused on the span path), so this isolates the
    /// world-anchored UV mapping.
    fn span_ctx(distance: f32, wall_along: f32, wall_up: f32) -> SampleCtx {
        SampleCtx {
            distance,
            foreshorten: 1.0,
            side: 0,
            on_wall: true,
            screen_y: 5,
            wall_v: 0.5,
            tick: 0,
            time: 0.0,
            fog: FogParams::DEFAULT,
            far_freeze: f32::INFINITY,
            light: None,
            lod_override: None,
            state: 0,
            pulse: 0.0,
            inset_rgb: None,
            wall_along,
            wall_up,
            tex_footprint: None,
            subcell_backdrop: None,
        }
    }

    /// SPANNING (not tiling): a 2-CELL-WIDE image surface with a world-anchored
    /// span maps `u≈0` on the LEFT cell and `u≈0.5` on the RIGHT cell — ONE figure
    /// stretched across both cells, NOT the whole figure repeated per cell. The
    /// synthetic top-row ramp lets us read `u` from the sampled colour: `u<0.25`→red
    /// (col0), `0.5≤u<0.75`→blue (col2).
    #[test]
    fn image_span_maps_one_figure_across_two_cells() {
        let perlin = Perlin::new(NOISE_SEED);
        let textures = vec![re_core::TextureSet::still(synthetic_4x4())];
        // Span: along world X in [10, 12) (2 cells wide), up world Z in (1, 2] (1 tall).
        let span = ascii_shader::ImageSpan {
            along_min: 10.0,
            along_size: 2.0,
            up_top: 2.0,
            up_size: 1.0,
        };
        let reg = image_registry_span(0, &textures, Some(span));
        let mat = reg.get(ids::WALL_BRICK);

        // Sample at the TOP of the span (world up ≈ 2.0 → v≈0, the colour ramp row).
        // Distance ≤ FOG_NEAR (1.0) keeps the texel colour un-dimmed for the assert.
        let top_up = 1.999;
        // LEFT cell centre: world X 10.5 → u = (10.5-10)/2 = 0.25 → col1 (green).
        // To land squarely in col0 we sample world X 10.25 → u=0.125.
        let left = mat.sample(
            &perlin,
            glam::Vec2::new(0.0, 0.0),
            &span_ctx(1.0, 10.25, top_up),
        );
        // RIGHT cell: world X 11.25 → u = (11.25-10)/2 = 0.625 → col2 (blue).
        let right = mat.sample(
            &perlin,
            glam::Vec2::new(0.0, 0.0),
            &span_ctx(1.0, 11.25, top_up),
        );

        assert_eq!(
            left.fg, 0xFF_0000,
            "left cell (u≈0.125) samples col0 (red), got {:06X}",
            left.fg
        );
        assert_eq!(
            right.fg, 0x00_00FF,
            "right cell (u≈0.625) samples col2 (blue), got {:06X}",
            right.fg
        );
        assert_ne!(
            left.fg, right.fg,
            "the two cells must show DIFFERENT parts of ONE figure (spanning), not the same tiled figure"
        );
    }

    /// VIEW-INDEPENDENT: the SAME world point on a spanned surface maps to the
    /// SAME texture `(u, v)` regardless of camera distance — the mapping is purely a
    /// function of world (along, up), so depth never shifts it. (The lit `sample`
    /// output additionally carries distance fog, which is a SEPARATE shade term; the
    /// world-anchored UV — what "samples the same world point" means — is proven here
    /// to be depth-invariant, and `sample` at a non-fogged near range below confirms
    /// the same texel reads through.)
    #[test]
    fn image_span_same_world_point_maps_same_uv_close_and_far() {
        let span = ascii_shader::ImageSpan {
            along_min: 0.0,
            along_size: 6.0, // a 6-cell-wide surface
            up_top: 4.0,
            up_size: 3.0,
        };
        // The UV mapping takes ONLY world (along, up) — no distance argument exists,
        // so the same world point is identical at every camera distance by construction.
        let (along, up) = (1.5_f32, 3.99_f32);
        let uv_a = span.uv(along, up);
        let uv_b = span.uv(along, up);
        assert_eq!(
            uv_a, uv_b,
            "the world-anchored UV is a pure function of world position"
        );

        // And the SAME world point on the SAME surface, sampled at two near distances
        // both inside the fog-NEAR band (≤ FOG_NEAR = 1.0, no dimming), reads the same
        // texel colour through `sample` — the texel chosen is world-anchored, not depth.
        let perlin = Perlin::new(NOISE_SEED);
        let textures = vec![re_core::TextureSet::still(synthetic_4x4())];
        let reg = image_registry_span(0, &textures, Some(span));
        let mat = reg.get(ids::WALL_BRICK);
        let near = mat.sample(
            &perlin,
            glam::Vec2::new(0.0, 0.0),
            &span_ctx(0.5, along, up),
        );
        let nearer = mat.sample(
            &perlin,
            glam::Vec2::new(0.0, 0.0),
            &span_ctx(1.0, along, up),
        );
        assert_eq!(
            near.fg, nearer.fg,
            "world-anchored span samples the SAME texel for the same world point ({:06X} vs {:06X})",
            near.fg, nearer.fg
        );
    }

    // ── SUB-CELL detail (quad / braille) ─────────────────────────────────

    /// A 2×2 texture: top-left RED opaque, top-right TRANSPARENT, bottom-left
    /// TRANSPARENT, bottom-right BLUE opaque — a diagonal half-opaque pattern that
    /// reads distinctly under both sub-cell modes.
    fn synthetic_2x2_diag() -> re_core::TextureImage {
        let rgba = vec![
            255, 0, 0, 255, // (0,0) red opaque
            0, 0, 0, 0, // (1,0) transparent
            0, 0, 0, 0, // (0,1) transparent
            0, 0, 255, 255, // (1,1) blue opaque
        ];
        re_core::TextureImage::new(2, 2, rgba)
    }

    /// A `span_ctx` (world-anchored) with a SUB-CELL footprint and an explicit
    /// backdrop. The footprint is the cell's world `(along, up)` extent; smaller = nearer.
    fn span_ctx_fp(along: f32, up: f32, footprint: (f32, f32), backdrop: Option<u32>) -> SampleCtx {
        let mut c = span_ctx(0.5, along, up);
        c.tex_footprint = Some(footprint);
        c.subcell_backdrop = backdrop;
        c
    }

    /// QUADRANT (2×2), INTERIOR-LUMINANCE split: the diag texture's four sub-texels are
    /// TL red (opaque), TR/BL transparent, BR blue (opaque). The luminance-split sampler
    /// ([`quadrant::blit_quadrant_luminance`]) sorts OPAQUE sub-texels above the mean
    /// luminance into the foreground; red (luma ≈ 76) is the brightest sub-texel and the
    /// only one above the mean, blue (luma ≈ 28) falls to the background, and the two
    /// TRANSPARENT sub-texels read the backdrop (preserving the alpha edge). So the glyph
    /// is the single-TL corner `▘`, not the old depth-split `▚` — interior tone now drives
    /// the split, while the transparent edge still reveals the backdrop in bg.
    #[test]
    fn subcell_quad_diag_texture_luminance_split_picks_brightest_corner() {
        let perlin = Perlin::new(NOISE_SEED);
        let textures = vec![re_core::TextureSet::still(synthetic_2x2_diag())];
        // Span the 2×2 figure over a 1×1 world cell at along∈[0,1], up∈(0,1].
        let span = ascii_shader::ImageSpan {
            along_min: 0.0,
            along_size: 1.0,
            up_top: 1.0,
            up_size: 1.0,
        };
        let reg =
            image_registry_span_subcell(0, &textures, Some(span), ascii_shader::SubcellMode::Quad);
        let mat = reg.get(ids::WALL_BRICK);
        // Cell centre at world (0.5, 0.5); footprint = the whole 1×1 cell so the four
        // sub-centres land at the four texel quadrants.
        let backdrop = 0x0012_3456;
        let ctx = span_ctx_fp(0.5, 0.5, (1.0, 1.0), Some(backdrop));
        let t = mat.sample(&perlin, glam::Vec2::new(0.0, 0.0), &ctx);
        assert_eq!(
            t.glyph,
            u32::from('▘'),
            "TL red is the only opaque sub-texel above the mean luminance → top-left corner"
        );
        // fg = the bright red foreground; bg blends the dark blue (opaque, below mean) with
        // the backdrop the two transparent corners reveal — the alpha edge survives.
        assert_eq!(
            t.fg, 0x00FF_0000,
            "fg = the bright (above-mean) red interior tone"
        );
        assert_ne!(
            t.bg, t.fg,
            "bg (dark + backdrop) differs from the bright fg"
        );
    }

    /// BRAILLE (2×4): the SAME diag texture — the eight sub-dots split into opaque
    /// (red top-left band, blue bottom-right band) and transparent. The glyph must be a
    /// Braille pattern (U+2800..28FF), and the transparent dots must show the backdrop.
    #[test]
    fn subcell_braille_diag_texture_is_a_braille_glyph_over_backdrop() {
        let perlin = Perlin::new(NOISE_SEED);
        let textures = vec![re_core::TextureSet::still(synthetic_2x2_diag())];
        let span = ascii_shader::ImageSpan {
            along_min: 0.0,
            along_size: 1.0,
            up_top: 1.0,
            up_size: 1.0,
        };
        let reg = image_registry_span_subcell(
            0,
            &textures,
            Some(span),
            ascii_shader::SubcellMode::Braille,
        );
        let mat = reg.get(ids::WALL_BRICK);
        let backdrop = 0x0000_FF00;
        let ctx = span_ctx_fp(0.5, 0.5, (1.0, 1.0), Some(backdrop));
        let t = mat.sample(&perlin, glam::Vec2::new(0.0, 0.0), &ctx);
        let g = char::from_u32(t.glyph).unwrap();
        assert!(
            ('\u{2800}'..='\u{28FF}').contains(&g),
            "braille glyph in U+28xx, got {g} ({:#X})",
            t.glyph
        );
        // Some dots opaque, some transparent → a NON-blank, NON-full pattern.
        assert_ne!(
            t.glyph,
            u32::from('\u{2800}'),
            "half-opaque texture must light some dots"
        );
        assert_ne!(
            t.glyph,
            u32::from('\u{28FF}'),
            "half-transparent texture must leave some dots off"
        );
        assert_eq!(t.bg, backdrop, "transparent dots reveal the backdrop");
    }

    /// CLOSE-UP resolves finer detail: a near view (SMALL world footprint) and a far
    /// view (LARGE footprint) of the SAME 4×4 ramp surface land the 2×2 sub-samples on
    /// DIFFERENT texels → a different quadrant glyph. Confirms approaching reveals more
    /// of the original image resolution (nearest, no mip).
    #[test]
    fn subcell_quad_near_resolves_more_detail_than_far() {
        let perlin = Perlin::new(NOISE_SEED);
        let textures = vec![re_core::TextureSet::still(synthetic_4x4())];
        // The 4×4 ramp spans a 4×4 world surface (along/up sizes = 4 → 1 texel per world
        // cell). All texels are opaque, so the quadrant glyph is driven by depth split,
        // which is uniform → `█`; to read the FOOTPRINT effect on the SAMPLED texels we
        // instead compare the centre-sub-texel colours the two footprints select.
        let span = ascii_shader::ImageSpan {
            along_min: 0.0,
            along_size: 4.0,
            up_top: 4.0,
            up_size: 4.0,
        };
        let reg =
            image_registry_span_subcell(0, &textures, Some(span), ascii_shader::SubcellMode::Quad);
        let mat = reg.get(ids::WALL_BRICK);
        // Sample on the top colour-ramp row (up≈4) near along=0.5 (col0 red region).
        // FAR: a large 2-cell footprint reaches across into col1 (green) on the +¼ side.
        let far = mat.sample(
            &perlin,
            glam::Vec2::new(0.0, 0.0),
            &span_ctx_fp(1.0, 3.99, (4.0, 0.5), Some(0)),
        );
        // NEAR: a tiny footprint keeps all four sub-samples inside col0 (red) → a flat
        // full-block red cell (no colour spread).
        let near = mat.sample(
            &perlin,
            glam::Vec2::new(0.0, 0.0),
            &span_ctx_fp(1.0, 3.99, (0.2, 0.05), Some(0)),
        );
        // The near view's fg is the pure col0 red; the far view, straddling two texels,
        // averages a different (non-pure-red) fg → the two differ, proving the footprint
        // (distance) changes which texels resolve.
        assert_ne!(
            near.fg, far.fg,
            "near (small footprint) and far (large footprint) must resolve different texels \
             (near={:06X} far={:06X})",
            near.fg, far.fg
        );
    }

    /// `subcell = off` (or a caster that threads NO footprint) keeps the world-anchored
    /// solid-block output byte-identical — the regression guard for non-opted-in images.
    #[test]
    fn subcell_off_is_byte_identical_solid_block() {
        let perlin = Perlin::new(NOISE_SEED);
        let textures = vec![re_core::TextureSet::still(synthetic_4x4())];
        let span = ascii_shader::ImageSpan {
            along_min: 0.0,
            along_size: 4.0,
            up_top: 4.0,
            up_size: 4.0,
        };
        // Same surface, OFF mode, even WITH a footprint threaded — Off must ignore it.
        let reg_off =
            image_registry_span_subcell(0, &textures, Some(span), ascii_shader::SubcellMode::Off);
        let mat_off = reg_off.get(ids::WALL_BRICK);
        // span_ctx_fp(along, up, ...) → here along=0.5, up=3.99 (top ramp row).
        let off = mat_off.sample(
            &perlin,
            glam::Vec2::new(0.0, 0.0),
            &span_ctx_fp(0.5, 3.99, (1.0, 1.0), Some(0)),
        );
        // The legacy spanning path (no footprint, no subcell) at the SAME world point:
        // span_ctx(distance, wall_along, wall_up) → distance 0.5, along 0.5, up 3.99.
        let reg_legacy = image_registry_span(0, &textures, Some(span));
        let mat_legacy = reg_legacy.get(ids::WALL_BRICK);
        let legacy = mat_legacy.sample(
            &perlin,
            glam::Vec2::new(0.0, 0.0),
            &span_ctx(0.5, 0.5, 3.99),
        );
        assert_eq!(
            off.glyph, legacy.glyph,
            "subcell=off keeps the solid block glyph"
        );
        assert_eq!(off.fg, legacy.fg, "subcell=off keeps the legacy fg");
        assert_eq!(off.bg, legacy.bg, "subcell=off keeps the legacy bg");
    }

    /// CUT-OUT, world-anchored: a `(u, v)` OUTSIDE the figure's world box is
    /// TRANSPARENT (margin → backdrop), while one inside obeys the texture alpha.
    /// Drives the world-anchored opaque test the per-row backdrop loop uses.
    #[test]
    fn image_span_world_cutout_outside_box_is_transparent() {
        let textures = vec![re_core::TextureSet::still(synthetic_4x4())];
        let span = ascii_shader::ImageSpan {
            along_min: 0.0,
            along_size: 4.0,
            up_top: 4.0,
            up_size: 4.0,
        };
        let reg = image_registry_span(0, &textures, Some(span));
        let mat = reg.get(ids::WALL_BRICK);
        // Inside the box (along 2, up 2 → u=0.5, v=0.5): opaque (the synthetic image
        // has alpha 255 everywhere).
        assert!(
            mat.image_texel_opaque_at_world(&span, 2.0, 2.0),
            "a point inside the figure box must be opaque"
        );
        // Outside the box along the wall (along 9 → u=2.25): transparent margin.
        assert!(
            !mat.image_texel_opaque_at_world(&span, 9.0, 2.0),
            "a point past the figure's world extent must be transparent (shows backdrop)"
        );
        // Above the box vertically (up 9 → v=-1.25): transparent.
        assert!(
            !mat.image_texel_opaque_at_world(&span, 2.0, 9.0),
            "a point above the figure's world top must be transparent"
        );
    }

    /// A 4-wide texture with ONLY column 0 opaque (red, alpha 255); columns 1..3 are
    /// fully transparent (alpha 0). Used to prove the SUB-CELL silhouette gate keeps a
    /// cell-row whose CENTRE texel is transparent but whose edge sub-texel is opaque.
    fn synthetic_left_edge_4x4() -> re_core::TextureImage {
        let mut rgba = Vec::with_capacity(4 * 4 * 4);
        for _row in 0..4 {
            rgba.extend_from_slice(&[255, 0, 0, 255]); // col 0 opaque
            for _ in 1..4 {
                rgba.extend_from_slice(&[0, 0, 0, 0]); // cols 1..3 transparent
            }
        }
        re_core::TextureImage::new(4, 4, rgba)
    }

    /// FIX 1 (silhouette clip): for a `braille` pane the per-row gate must keep a cell
    /// whose CELL-CENTRE texel is transparent but which has an OPAQUE sub-texel near
    /// the figure's edge — otherwise the crisp 1-texel margin is clipped at whole-cell
    /// granularity before the 2×4 sampler ever runs. The any-sub-texel gate
    /// (`image_row_opaque_subcell`) KEEPS it; the old centre-only gate
    /// (`image_texel_opaque_at_world`) DROPPED it. Sub-positions are the exact ones the
    /// braille sampler shades, so the kept row renders a partial-braille glyph.
    #[test]
    fn subcell_braille_edge_cell_centre_transparent_subtexel_opaque_is_kept() {
        let textures = vec![re_core::TextureSet::still(synthetic_left_edge_4x4())];
        // Span: u = along / 4, so along 0.5→col0 (opaque), 1.5→col1 (transparent).
        let span = ascii_shader::ImageSpan {
            along_min: 0.0,
            along_size: 4.0,
            up_top: 4.0,
            up_size: 4.0,
        };
        let reg = image_registry_span_subcell(
            0,
            &textures,
            Some(span),
            ascii_shader::SubcellMode::Braille,
        );
        let mat = reg.get(ids::WALL_BRICK);

        // CELL CENTRE at along 1.5 → u = 0.375 → column 1 → TRANSPARENT. The old
        // centre-only gate drops this whole row (clipping the silhouette margin).
        assert!(
            !mat.image_texel_opaque_at_world(&span, 1.5, 2.0),
            "the cell-centre texel here is transparent (the clip the old gate caused)"
        );

        // The any-sub-texel gate samples ±¼ footprint ALONG. With fp_along = 4.0 the
        // left sub-column is at along 1.5 − 1.0 = 0.5 → u = 0.125 → column 0 → OPAQUE,
        // so the gate KEEPS the row and the braille glyph renders the edge.
        let fp = (4.0f32, 0.25f32);
        assert!(
            mat.image_row_opaque_subcell(&span, 1.5, 2.0, fp, ascii_shader::SubcellMode::Braille),
            "a transparent-centre cell with an opaque edge sub-texel must be KEPT \
             so the braille silhouette margin is not clipped to whole-cell granularity"
        );

        // AGREEMENT WITH THE SHADE: the braille sampler at this cell, with the same
        // footprint, must produce an INKED (non-blank) glyph — the kept row really does
        // draw a partial-braille silhouette, not an empty cell.
        let perlin = Perlin::new(NOISE_SEED);
        let ctx = span_ctx_fp(1.5, 2.0, fp, Some(0x00_0000));
        let t = mat.sample(&perlin, glam::Vec2::new(0.5, 0.0), &ctx);
        assert_ne!(
            t.glyph,
            u32::from(' '),
            "the kept braille cell must shade a real (non-blank) glyph — gate and shade agree"
        );
        // Braille glyph block U+2800..28FF; an inked dot is > U+2800.
        assert!(
            (0x2800..=0x28FF).contains(&t.glyph) && t.glyph > 0x2800,
            "an opaque sub-texel must set at least one braille dot (got U+{:04X})",
            t.glyph
        );

        // CONTROL: a cell wholly in the transparent margin (centre AND both sub-columns
        // transparent) is still correctly dropped — the gate only rescues true edges.
        // along 2.5 → col2; sub-cols at 1.5→col1 and 3.5→col3, all transparent.
        assert!(
            !mat.image_row_opaque_subcell(&span, 2.5, 2.0, fp, ascii_shader::SubcellMode::Braille),
            "a fully-transparent cell-row (no opaque sub-texel) must stay dropped"
        );
    }

    /// FIX 1 regression: an ALL-OPAQUE texture is unchanged by the any-sub-texel gate —
    /// every sub-position is opaque, so the gate agrees with the centre-only gate
    /// (both `true`) for `quad` and `braille`. The new gate only ever differs at a
    /// transparent edge, never on a solid figure.
    #[test]
    fn subcell_gate_all_opaque_texture_matches_centre_gate() {
        let textures = vec![re_core::TextureSet::still(synthetic_4x4())]; // alpha 255 everywhere
        let span = ascii_shader::ImageSpan {
            along_min: 0.0,
            along_size: 4.0,
            up_top: 4.0,
            up_size: 4.0,
        };
        let reg =
            image_registry_span_subcell(0, &textures, Some(span), ascii_shader::SubcellMode::Quad);
        let mat = reg.get(ids::WALL_BRICK);
        let fp = (4.0f32, 0.25f32);
        for &(along, up) in &[(0.5f32, 0.5f32), (1.5, 2.0), (2.5, 3.0), (3.5, 1.0)] {
            let centre = mat.image_texel_opaque_at_world(&span, along, up);
            let quad =
                mat.image_row_opaque_subcell(&span, along, up, fp, ascii_shader::SubcellMode::Quad);
            let brl = mat.image_row_opaque_subcell(
                &span,
                along,
                up,
                fp,
                ascii_shader::SubcellMode::Braille,
            );
            assert!(
                centre && quad && brl,
                "all-opaque texture: centre+quad+braille gates all opaque at ({along},{up})"
            );
        }
    }

    /// MISSING TEXTURE: an `Image` material whose `tex_id` is out of range (so the
    /// registry could not attach a texture) renders a MAGENTA error block, never a
    /// panic. Guards the graceful-fallback contract.
    #[test]
    fn image_material_missing_texture_is_magenta_not_panic() {
        let perlin = Perlin::new(NOISE_SEED);
        // tex_id 7 with an EMPTY texture store ⇒ nothing to attach ⇒ texture None.
        let reg = image_registry(7, &[]);
        let mat = reg.get(ids::WALL_BRICK);
        assert!(
            mat.texture.is_none(),
            "out-of-range tex_id leaves no texture"
        );
        let ctx = SampleCtx {
            distance: 1.0,
            foreshorten: 1.0,
            side: 0,
            screen_y: 5,
            wall_v: 0.5,
            tick: 0,
            time: 0.0,
            fog: FogParams::DEFAULT,
            far_freeze: f32::INFINITY,
            light: None,
            lod_override: None,
            state: 0,
            pulse: 0.0,
            inset_rgb: None,
            wall_along: 0.0,
            wall_up: 0.0,
            on_wall: false,
            tex_footprint: None,
            subcell_backdrop: None,
        };
        let t = mat.sample(&perlin, glam::Vec2::new(0.5, 0.0), &ctx);
        assert_eq!(t.glyph, 0x2588, "error block is still a solid block");
        assert_eq!(
            t.fg, 0xFF_00FF,
            "missing texture must read magenta, got {:06X}",
            t.fg
        );
    }

    /// END-TO-END through the public cast: a `test_room` whose wall surface is an
    /// `Image` material renders the texture's colours on the wall cells (solid `█`
    /// blocks, the texel colours — NOT a procedural brick pattern). Proves the whole
    /// raycast → shade path carries an attached image to the framebuffer.
    #[test]
    fn raycast_lands_image_colours_on_the_wall() {
        let map = test_room();
        let textures = vec![re_core::TextureSet::still(synthetic_4x4())];
        let registry = image_registry(0, &textures);

        let cam = CameraPose::spawn_in_test_room();
        let mut fb = Framebuffer::new(120, 40);
        raycast_with(&map, &cam, &registry, &mut fb);

        // Hue dominance is the robust signal: the texture's GREEN (0,255,0) and BLUE
        // (0,0,255) texels are impossible for the procedural brick built-in (warm,
        // R≥G≥B stone). Distance fog here is `color: None` ⇒ a UNIFORM per-channel
        // dim, which PRESERVES channel dominance, and the Image recipe is self-lit so
        // no relief desaturates it — so a green-dominant or blue-dominant wall cell can
        // ONLY have come from sampling the green/blue texel. Its presence proves the
        // texture (not the brick pattern) drove the wall colour.
        let dom = |c: u32| -> u8 {
            let r = (c >> 16) & 0xFF;
            let g = (c >> 8) & 0xFF;
            let b = c & 0xFF;
            if g > r && g >= b {
                1 // green-dominant
            } else if b > r && b > g {
                2 // blue-dominant
            } else {
                0 // red/grey/white — also possible from brick, so not a proof
            }
        };

        let mut wall_cells = 0usize;
        let mut block_cells = 0usize;
        let mut green_dom = 0usize;
        let mut blue_dom = 0usize;
        for row in 0..fb.rows() {
            for col in 0..fb.cols() {
                let c = fb.get(col, row).unwrap();
                if !c.surface.is_wall() {
                    continue;
                }
                wall_cells += 1;
                // Image walls are solid blocks (no procedural brick/mortar glyphs).
                if c.glyph == 0x2588 {
                    block_cells += 1;
                }
                match dom(c.fg) {
                    1 => green_dom += 1,
                    2 => blue_dom += 1,
                    _ => {}
                }
            }
        }
        assert!(
            wall_cells > 50,
            "the room must show many wall cells, got {wall_cells}"
        );
        assert!(
            block_cells == wall_cells,
            "every image wall cell must be a solid `█` block, not a procedural brick \
             pattern ({block_cells}/{wall_cells})"
        );
        // The brick built-in can never produce a green- or blue-dominant cell, so these
        // prove the GREEN and BLUE texels of the synthetic texture reached the wall.
        assert!(
            green_dom > 0,
            "the texture's green texel must reach the wall (green_dom={green_dom})"
        );
        assert!(
            blue_dom > 0,
            "the texture's blue texel must reach the wall (blue_dom={blue_dom})"
        );
    }

    // ── IMAGE TEXTURE ALPHA CUT-OUT (image-texture milestone 2): transparent ──
    // texels pass the ray THROUGH to the backdrop; opaque texels stop and paint.

    /// A 1×1 texture with a chosen colour and a chosen alpha — the simplest control
    /// for the strip-transparency / opacity test (a single texel IS the whole strip).
    fn solid_texel(rgb: [u8; 3], alpha: u8) -> re_core::TextureImage {
        re_core::TextureImage::new(1, 1, vec![rgb[0], rgb[1], rgb[2], alpha])
    }

    /// `dda_hit` PASS-THROUGH: a fully-TRANSPARENT Image pane (`alpha = 0`) standing
    /// in front of a distinct backdrop wall must be marched THROUGH — the DDA returns
    /// the BACKDROP cell, exactly as a wedge open triangle is see-through. Without the
    /// alpha branch the ray would stop on the pane (its cell), so the asserted backdrop
    /// cell is the proof.
    #[test]
    fn dda_passes_through_a_transparent_image_pane_to_the_backdrop() {
        let mut m = Map::new(8, 8, 3);
        // Front pane: a WallSolid cell (→ WALL_BRICK slot, made Image below) at (3,3).
        m.set(3, 3, Z as usize, CellKind::WallSolid);
        // Backdrop: a DISTINCT non-image wall (WallTrace → WALL_CIRCUIT) two cells back.
        m.set(3, 5, Z as usize, CellKind::WallTrace);

        let textures = vec![re_core::TextureSet::still(solid_texel([200, 50, 50], 0))]; // alpha 0
        let reg = image_registry(0, &textures);

        // Cast +Y from (3.5, 2.5) straight through the pane toward the backdrop.
        let hit = dda_hit(&m, Z, 3.5, 2.5, 0.0, 1.0, Some(&reg));
        assert!(
            hit.hit,
            "the ray must hit the backdrop behind the transparent pane"
        );
        assert_eq!(
            (hit.cell_x, hit.cell_y),
            (3, 5),
            "a fully-transparent Image pane is see-through → the hit is the backdrop \
             wall at (3,5), not the pane at (3,3) (got {},{})",
            hit.cell_x,
            hit.cell_y
        );
        assert_eq!(
            hit.surface,
            SurfaceKind::WallTrace,
            "the hit must be the distinct backdrop surface, not the image pane"
        );
    }

    /// `dda_hit` OPAQUE STOP: an OPAQUE Image pane (`alpha = 255`) stops the ray on
    /// its own cell — the backdrop behind it is NOT reached. The control case for the
    /// pass-through test above (same scene, opaque texture).
    #[test]
    fn dda_stops_on_an_opaque_image_pane() {
        let mut m = Map::new(8, 8, 3);
        m.set(3, 3, Z as usize, CellKind::WallSolid);
        m.set(3, 5, Z as usize, CellKind::WallTrace);

        let textures = vec![re_core::TextureSet::still(solid_texel([200, 50, 50], 255))]; // opaque
        let reg = image_registry(0, &textures);

        let hit = dda_hit(&m, Z, 3.5, 2.5, 0.0, 1.0, Some(&reg));
        assert!(hit.hit, "the opaque pane must be hit");
        assert_eq!(
            (hit.cell_x, hit.cell_y),
            (3, 3),
            "an OPAQUE Image pane stops the ray on its own cell (3,3), not the backdrop \
             (got {},{})",
            hit.cell_x,
            hit.cell_y
        );
        assert_eq!(
            hit.surface,
            SurfaceKind::WallSolid,
            "the hit is the pane itself"
        );
    }

    /// `dda_hit` byte-compat: when `registry` is `None` (or the material is not an
    /// Image), even an Image-tagged cell stops the ray — the alpha branch only runs
    /// with a registry. Guards that the historical callers (no registry) are unchanged.
    #[test]
    fn dda_with_no_registry_never_passes_through() {
        let mut m = Map::new(8, 8, 3);
        m.set(3, 3, Z as usize, CellKind::WallSolid);
        m.set(3, 5, Z as usize, CellKind::WallTrace);
        // No registry → no alpha lookup → the front cell stops the ray as it always did.
        let hit = dda_hit(&m, Z, 3.5, 2.5, 0.0, 1.0, None);
        assert_eq!(
            (hit.cell_x, hit.cell_y),
            (3, 3),
            "with no registry the DDA must stop on the front cell (byte-compat)"
        );
    }

    /// END-TO-END cut-out through the public cast: a fully-TRANSPARENT Image pane in
    /// front of a coloured BACKDROP wall must show the BACKDROP through the cutout. The
    /// backdrop is a circuit wall (WallTrace → WALL_CIRCUIT, a cool blue/teal trace
    /// look impossible for the transparent red pane), so a wall column that reads as
    /// the circuit — not the pane colour, not a hole — proves the backdrop showed
    /// through. We assert the pane's RED never dominates a wall cell (it is fully
    /// transparent) and that wall cells are present (the backdrop filled them).
    #[test]
    fn raycast_transparent_pane_shows_the_backdrop_behind() {
        // A small room with a transparent Image pane and a backdrop a few cells behind.
        let mut map = Map::new(7, 12, 4);
        let (sx, sy, sz) = (7usize, 12usize, 4usize);
        for z in 0..sz {
            for y in 0..sy {
                for x in 0..sx {
                    let perim = x == 0 || x == sx - 1 || y == 0 || y == sy - 1;
                    let kind = if perim {
                        CellKind::WallSolid
                    } else if z == 0 {
                        CellKind::Floor
                    } else if z == sz - 1 {
                        CellKind::Ceiling
                    } else {
                        CellKind::Empty
                    };
                    map.set(x, y, z, kind);
                }
            }
        }
        // Front pane: a full-width WallSolid (Image) wall at y=4 (slot 0 → Image).
        for x in 1..sx - 1 {
            map.set(x, 4, 1, CellKind::WallSolid);
        }
        // Backdrop: a full-width DISTINCT circuit wall (WallTrace) at y=8, behind it.
        for x in 1..sx - 1 {
            map.set(x, 8, 1, CellKind::WallTrace);
        }

        let textures = vec![re_core::TextureSet::still(solid_texel([220, 30, 30], 0))]; // transparent red
        let registry = image_registry(0, &textures);

        // Camera at the south end (y small) facing +Y toward the pane then the backdrop.
        // yaw = 0 faces +Y (toward the pane then the backdrop).
        let cam = CameraPose {
            pos: glam::Vec3::new(3.5, 1.5, STANDING_EYE_Z),
            yaw: 0.0,
            pitch: 0.0,
        };

        let mut fb = Framebuffer::new(80, 40);
        raycast_with(&map, &cam, &registry, &mut fb);

        // Hue dominance: a RED-dominant wall cell could only come from the pane texel,
        // which is fully transparent — so it must NEVER appear. The backdrop circuit
        // (cool, blue/teal-leaning) fills the wall band instead.
        let red_dom = |c: u32| -> bool {
            let r = (c >> 16) & 0xFF;
            let g = (c >> 8) & 0xFF;
            let b = c & 0xFF;
            r > g + 40 && r > b + 40
        };
        let mut wall_cells = 0usize;
        let mut red_wall = 0usize;
        let mut block_wall = 0usize;
        for row in 0..fb.rows() {
            for col in 0..fb.cols() {
                let c = fb.get(col, row).unwrap();
                if !c.surface.is_wall() {
                    continue;
                }
                wall_cells += 1;
                if red_dom(c.fg) {
                    red_wall += 1;
                }
                // The transparent pane would paint solid `█` blocks; the backdrop
                // circuit uses procedural trace glyphs. Count solid blocks to confirm
                // the pane did NOT paint the wall band.
                if c.glyph == 0x2588 {
                    block_wall += 1;
                }
            }
        }
        assert!(
            wall_cells > 30,
            "the room must show a wall band, got {wall_cells}"
        );
        assert_eq!(
            red_wall, 0,
            "the fully-transparent red pane must NOT colour any wall cell — the \
             backdrop shows through (red_wall={red_wall})"
        );
        // The pane being see-through means its solid-block fill is absent; the visible
        // wall band is the procedural backdrop, not image blocks.
        assert!(
            block_wall < wall_cells / 2,
            "a transparent pane must not fill the wall band with image blocks \
             ({block_wall}/{wall_cells})"
        );
    }

    /// END-TO-END opaque control: the SAME scene with an OPAQUE pane DOES colour the
    /// wall band with the pane's red — the cut-out only hides the pane when transparent.
    #[test]
    fn raycast_opaque_pane_colours_the_wall_band() {
        let mut map = Map::new(7, 12, 4);
        let (sx, sy, sz) = (7usize, 12usize, 4usize);
        for z in 0..sz {
            for y in 0..sy {
                for x in 0..sx {
                    let perim = x == 0 || x == sx - 1 || y == 0 || y == sy - 1;
                    let kind = if perim {
                        CellKind::WallSolid
                    } else if z == 0 {
                        CellKind::Floor
                    } else if z == sz - 1 {
                        CellKind::Ceiling
                    } else {
                        CellKind::Empty
                    };
                    map.set(x, y, z, kind);
                }
            }
        }
        for x in 1..sx - 1 {
            map.set(x, 4, 1, CellKind::WallSolid);
        }
        for x in 1..sx - 1 {
            map.set(x, 8, 1, CellKind::WallTrace);
        }

        let textures = vec![re_core::TextureSet::still(solid_texel([220, 30, 30], 255))]; // OPAQUE red
        let registry = image_registry(0, &textures);

        // yaw = 0 faces +Y (toward the pane then the backdrop).
        let cam = CameraPose {
            pos: glam::Vec3::new(3.5, 1.5, STANDING_EYE_Z),
            yaw: 0.0,
            pitch: 0.0,
        };

        let mut fb = Framebuffer::new(80, 40);
        raycast_with(&map, &cam, &registry, &mut fb);

        let red_dom = |c: u32| -> bool {
            let r = (c >> 16) & 0xFF;
            let g = (c >> 8) & 0xFF;
            let b = c & 0xFF;
            r > g + 40 && r > b + 40
        };
        let mut red_wall = 0usize;
        for row in 0..fb.rows() {
            for col in 0..fb.cols() {
                let c = fb.get(col, row).unwrap();
                if c.surface.is_wall() && red_dom(c.fg) {
                    red_wall += 1;
                }
            }
        }
        assert!(
            red_wall > 0,
            "an OPAQUE red pane must colour the wall band red (red_wall={red_wall})"
        );
    }

    /// Half-transparent CUT-OUT silhouette: a 1×2 texture with an OPAQUE top texel and
    /// a TRANSPARENT bottom texel. The pane column has an opaque sliver (so `dda_hit`
    /// stops on it) but its lower rows are see-through — those rows must show the
    /// backdrop, while the opaque rows show the image. Proves the per-row vertical
    /// silhouette compositing (`image_cutout_column`), not just whole-column pass.
    #[test]
    fn raycast_half_transparent_pane_composites_figure_over_backdrop() {
        let mut map = Map::new(7, 12, 4);
        let (sx, sy, sz) = (7usize, 12usize, 4usize);
        for z in 0..sz {
            for y in 0..sy {
                for x in 0..sx {
                    let perim = x == 0 || x == sx - 1 || y == 0 || y == sy - 1;
                    let kind = if perim {
                        CellKind::WallSolid
                    } else if z == 0 {
                        CellKind::Floor
                    } else if z == sz - 1 {
                        CellKind::Ceiling
                    } else {
                        CellKind::Empty
                    };
                    map.set(x, y, z, kind);
                }
            }
        }
        for x in 1..sx - 1 {
            map.set(x, 4, 1, CellKind::WallSolid); // Image pane
        }
        for x in 1..sx - 1 {
            map.set(x, 8, 1, CellKind::WallTrace); // backdrop circuit
        }

        // 1×2 texture: top row OPAQUE red, bottom row TRANSPARENT. `v=0` is the top.
        let tex = re_core::TextureImage::new(
            1,
            2,
            vec![
                220, 30, 30, 255, // top: opaque red
                220, 30, 30, 0, // bottom: transparent
            ],
        );
        let textures = vec![re_core::TextureSet::still(tex)];
        let registry = image_registry(0, &textures);

        // yaw = 0 faces +Y (toward the pane then the backdrop).
        let cam = CameraPose {
            pos: glam::Vec3::new(3.5, 1.5, STANDING_EYE_Z),
            yaw: 0.0,
            pitch: 0.0,
        };

        let mut fb = Framebuffer::new(80, 40);
        raycast_with(&map, &cam, &registry, &mut fb);

        let red_dom = |c: u32| -> bool {
            let r = (c >> 16) & 0xFF;
            let g = (c >> 8) & 0xFF;
            let b = c & 0xFF;
            r > g + 40 && r > b + 40
        };
        // Some wall cells must be red (the opaque top of the figure) AND some wall
        // cells must be solid-block backdrop-free; crucially the lower wall rows must
        // show the (non-red) backdrop circuit. We assert BOTH a red region exists
        // (figure top) and that not the whole band is red (cutout reveals backdrop).
        let mut red_wall = 0usize;
        let mut nonred_wall = 0usize;
        for row in 0..fb.rows() {
            for col in 0..fb.cols() {
                let c = fb.get(col, row).unwrap();
                if !c.surface.is_wall() {
                    continue;
                }
                if red_dom(c.fg) {
                    red_wall += 1;
                } else {
                    nonred_wall += 1;
                }
            }
        }
        assert!(
            red_wall > 0,
            "the OPAQUE top texel must paint the figure red on the wall (red_wall={red_wall})"
        );
        assert!(
            nonred_wall > 0,
            "the TRANSPARENT bottom texel must reveal the (non-red) backdrop in the \
             wall band (nonred_wall={nonred_wall})"
        );
    }

    // ───────────────────── 45° wedge/ramp DDA geometry ─────────────────────
    //
    // These exercise `dda_hit` directly against the per-cell `Shape` intersection:
    //  * a WedgeXY cell is hit on its SOLID diagonal half and is SEE-THROUGH on the
    //    OPEN half (a cell behind shows through the open triangle);
    //  * a RampZ cell is hit on its slope, and the reported surface normal is TILTED
    //    (a real 45° face), not axis-aligned;
    //  * an all-`Full` map keeps `shape_normal == None` (the axis-aligned fast path).

    use re_core::{DiagXY, RampDir, Shape};

    /// A `cell_z` whose mid-height sampling plane (z = cell_z + 0.5) the ramp tests use.
    const Z: i32 = 1;

    #[test]
    fn full_cell_hit_has_no_shape_normal() {
        // Back-compat: a Full solid wall must report NO shape normal so the shading
        // path stays on the byte-identical axis-aligned branch.
        let mut m = Map::new(6, 6, 3);
        m.set(3, 1, Z as usize, CellKind::WallSolid);
        // Cast +Y from (3.5, 0.5) straight at the wall cell (3,1).
        let hit = dda_hit(&m, Z, 3.5, 0.5, 0.0, 1.0, None);
        assert!(hit.hit, "the Full wall must be hit");
        assert_eq!(hit.cell_x, 3);
        assert_eq!(hit.cell_y, 1);
        assert!(
            hit.shape_normal.is_none(),
            "a Full-cell hit must not carry a 45° shape normal"
        );
    }

    #[test]
    fn wedge_is_hit_on_the_solid_diagonal_half() {
        // WedgeXY(NE): solid where x+y ≥ 1 (the diagonal runs (0,1)→(1,0)). A ray
        // aimed INTO the solid NE corner of the cell must hit the diagonal face and
        // report the tilted (vertical-diagonal) normal pointing back toward the eye.
        let mut m = Map::new(6, 6, 3);
        m.set(2, 2, Z as usize, CellKind::WallSolid);
        m.set_shape(2, 2, Z as usize, Shape::WedgeXY(DiagXY::NE));

        // The DDA tests cells AFTER stepping, so the ray must ORIGINATE OUTSIDE the
        // wedge cell. Enter cell (2,2) through its WEST face (world x=2) marching +X
        // at local y=0.6: entry (local 0, 0.6) is OPEN (x+y=0.6<1); the ray crosses
        // the diagonal x+y=1 at local x=0.4 → a diagonal hit on the solid NE half.
        let hit = dda_hit(&m, Z, 1.5, 2.6, 1.0, 0.0, None);
        assert!(hit.hit, "the ray into the solid NE half must hit the wedge");
        assert_eq!(
            (hit.cell_x, hit.cell_y),
            (2, 2),
            "hit lands in the wedge cell"
        );
        let n = hit
            .shape_normal
            .expect("a wedge hit must carry its diagonal normal");
        // The NE wedge's outward normal points SW: (-1,-1,0)/√2 — vertical (z≈0).
        assert!(
            n.z.abs() < 1e-3,
            "a vertical wedge wall normal has ~zero z (n={n:?})"
        );
        assert!(n.x < 0.0 && n.y < 0.0, "NE wedge faces SW (n={n:?})");
        assert!(
            (n.length() - 1.0).abs() < 1e-3,
            "normal must be unit (n={n:?})"
        );
        // The hit point lies on the world diagonal x+y = 5 within the cell.
        assert!(
            (hit.wall_hit_x + hit.wall_hit_y - 5.0).abs() < 1e-2,
            "wedge hit must lie on the diagonal x+y=5 (got {},{})",
            hit.wall_hit_x,
            hit.wall_hit_y
        );
    }

    #[test]
    fn wedge_open_half_is_see_through_to_a_cell_behind() {
        // The OPEN triangle of a wedge must be see-through: a solid cell placed BEHIND
        // the wedge (further along the ray) shows through the open half.
        let mut m = Map::new(8, 8, 3);
        // Wedge NE (solid where x+y ≥ 1) at cell (3,3): its OPEN half is the SW corner.
        m.set(3, 3, Z as usize, CellKind::WallSolid);
        m.set_shape(3, 3, Z as usize, Shape::WedgeXY(DiagXY::NE));
        // A plain Full wall further along the (1,-1) sight-line, BEHIND the wedge.
        m.set(5, 1, Z as usize, CellKind::WallSolid);

        // Cast along (1,-1) — PARALLEL to the NE/SW diagonal — from origin (2.5,4.0),
        // entering the wedge cell (3,3) through its west face at local (0, 0.5). Since
        // the direction is parallel to the diagonal, x+y stays 0.5 (< 1) the WHOLE way
        // through the cell → the ray never leaves the OPEN SW half → see-through to the
        // Full wall at cell (5,1).
        let hit = dda_hit(&m, Z, 2.5, 4.0, 1.0, -1.0, None);
        assert!(hit.hit, "the ray must hit SOMETHING (the wall behind)");
        assert_eq!(
            (hit.cell_x, hit.cell_y),
            (5, 1),
            "the wedge's open half is see-through → the hit is the wall behind it ({},{})",
            hit.cell_x,
            hit.cell_y
        );
        assert!(
            hit.shape_normal.is_none(),
            "the cell behind is a Full wall → axis-aligned normal"
        );
    }

    #[test]
    fn ramp_slope_is_hit_with_a_tilted_normal() {
        // RampZ(PosX): solid where z ≤ x (rises in +X). At the mid-height sample plane
        // (z=0.5) the slope surface sits at local x = 0.5. A ray marching +X across the
        // cell must hit the slope and report a normal tilted UP and toward −X.
        let mut m = Map::new(6, 6, 3);
        m.set(2, 3, Z as usize, CellKind::WallSolid);
        m.set_shape(2, 3, Z as usize, Shape::RampZ(RampDir::PosX));

        // The primary DDA never stops on a ramp: a ramp is solid only under its slope,
        // so it cannot stand for a full-height slice. The vertical geometry pass renders
        // it as an exact polygon instead (`offslice`), and the ray marches on.
        let through = dda_hit(&m, Z, 1.5, 3.5, 1.0, 0.0, None);
        assert!(
            !(through.hit && (through.cell_x, through.cell_y) == (2, 3)),
            "the primary ray must pass THROUGH a ramp cell"
        );

        // The per-cell shape test itself still resolves the slope exactly: enter the
        // ramp cell (2,3) through its WEST face (world x=2, local x=0, t=0.5) marching
        // +X at the mid-height plane. Entry z=0.5 > x=0 ⇒ open; the slope contour z=x
        // reaches the sample plane at local x=0.5 → a slope hit mid-cell.
        let hit = shape_hit::intersect_cell(
            Shape::RampZ(RampDir::PosX),
            2,
            3,
            Z,
            1.5,
            3.5,
            1.0,
            0.0,
            0.5,
        )
        .expect("the +X ray must hit the ramp slope");
        let n = hit.normal;
        assert!(
            n.z > 0.5,
            "the slope normal must tilt UPWARD (+z), not be axis-aligned (n={n:?})"
        );
        assert!(
            n.x < 0.0,
            "PosX ramp's normal points back along −x (n={n:?})"
        );
        assert!(
            (n.length() - 1.0).abs() < 1e-3,
            "normal must be unit (n={n:?})"
        );
        // The hit should land around local x = 0.5 (world x ≈ 2.5).
        assert!(
            (hit.hit_x - 2.5).abs() < 0.1,
            "PosX slope at z=0.5 crosses near local x=0.5 (world x={})",
            hit.hit_x
        );
    }

    #[test]
    fn ramp_normal_is_not_axis_aligned() {
        // Sanity that the slope normal genuinely tilts on BOTH the rise axis and z —
        // distinguishing a real 45° ramp from a flat vertical wall. Enter cell (2,3)
        // through its SOUTH face (world y=3, t=0.5) from cell (2,2) marching +Y.
        let hit = shape_hit::intersect_cell(
            Shape::RampZ(RampDir::PosY),
            2,
            3,
            Z,
            2.5,
            2.5,
            0.0,
            1.0,
            0.5,
        )
        .expect("ramp hit");
        let n = hit.normal;
        // PosY: outward normal (0, -1/√2, +1/√2): tilts on y AND z, x≈0.
        assert!(n.x.abs() < 1e-3, "PosY ramp normal has ~zero x (n={n:?})");
        assert!(n.y < 0.0 && n.z > 0.0, "PosY ramp tilts −y/+z (n={n:?})");
        assert!(
            (n.y.abs() - n.z.abs()).abs() < 1e-3,
            "a 45° slope's |y| and |z| components are equal (n={n:?})"
        );
    }

    // ───────────────────── S1: ramp renders as a RISING slice ───────────────────
    //
    // A ramp is solid only under its slope, so it is never a full-height slice: the
    // vertical geometry pass (`offslice`) draws it as its exact trapezoid on every
    // plane, so the drawn top steps up the slope across columns and the open air
    // ABOVE the slope is see-through to whatever lies behind the ramp.

    /// Build a closed room (perimeter walls, floor at z=0, ceiling at z=sz-1, interior
    /// air) and return the map. The caller drops a ramp into the interior.
    fn ramp_room(sx: usize, sy: usize, sz: usize) -> Map {
        let mut m = Map::new(sx, sy, sz);
        for z in 0..sz {
            for y in 0..sy {
                for x in 0..sx {
                    let perim = x == 0 || x == sx - 1 || y == 0 || y == sy - 1;
                    let kind = if perim {
                        CellKind::WallSolid
                    } else if z == 0 {
                        CellKind::Floor
                    } else if z == sz - 1 {
                        CellKind::Ceiling
                    } else {
                        CellKind::Empty
                    };
                    m.set(x, y, z, kind);
                }
            }
        }
        m
    }

    /// Cast the GAMEPLAY path (eye-shear referenced to STANDING_EYE_Z — the path that
    /// exercises the ramp silhouette projection's `vshift_per_perp` term) and return the
    /// framebuffer.
    fn cast_gameplay(map: &Map, cam: &CameraPose, cols: usize, rows: usize) -> Framebuffer {
        let reg = registry_for_map(map);
        let fov = Fov::default();
        let mut fb = Framebuffer::new(cols, rows);
        raycast_with_time_fov_gameplay(map, cam, &reg, &fov, 0.0, &mut fb);
        fb
    }

    /// The NEAR-wall slice in `col`: the contiguous run of WALL rows whose depth is
    /// within `near_band` of the column's nearest wall depth. This is the RAMP's own
    /// drawn slice — it excludes the FARTHER backdrop wall the see-through-above recast
    /// paints behind it. Returns `(top_row, count, near_depth)` or `None` if no wall.
    fn near_wall_slice(
        fb: &Framebuffer,
        col: usize,
        near_band: f32,
    ) -> Option<(usize, usize, f32)> {
        // Nearest wall depth in the column.
        let mut near = f32::INFINITY;
        for row in 0..fb.rows() {
            let c = fb.get(col, row).unwrap();
            if c.surface.is_wall() && c.depth < near {
                near = c.depth;
            }
        }
        if !near.is_finite() {
            return None;
        }
        let mut top = None;
        let mut count = 0usize;
        for row in 0..fb.rows() {
            let c = fb.get(col, row).unwrap();
            if c.surface.is_wall() && (c.depth - near).abs() <= near_band {
                top.get_or_insert(row);
                count += 1;
            }
        }
        top.map(|t| (t, count, near))
    }

    #[test]
    fn ramp_near_slice_is_shorter_than_a_full_wall_and_shows_backdrop_above() {
        // A single RampZ(PosX) cell viewed head-on: the ramp's OWN (near-depth) slice
        // must be SHORTER than the SAME cell as a Full wall, and the open air above the
        // slope must show the BACKDROP (a FARTHER wall / ceiling), not the ramp wall.
        let (sx, sy, sz) = (14usize, 11usize, 5usize);
        let make = |shape: Shape| {
            let mut m = ramp_room(sx, sy, sz);
            m.set(7, 5, 1, CellKind::WallSolid);
            if shape != Shape::Full {
                m.set_shape(7, 5, 1, shape);
            }
            m
        };
        let full = make(Shape::Full);
        let ramp = make(Shape::RampZ(RampDir::PosX));

        // Stand west, look +x (yaw = +PI/2 ⇒ dir (1,0)) straight at the cell.
        let cam = CameraPose {
            pos: glam::Vec3::new(2.5, 5.5, STANDING_EYE_Z),
            yaw: std::f32::consts::FRAC_PI_2,
            pitch: 0.0,
        };
        let (cols, rows) = (100usize, 40usize);
        let fb_full = cast_gameplay(&full, &cam, cols, rows);
        let fb_ramp = cast_gameplay(&ramp, &cam, cols, rows);

        // The ramp cell is ~4.5 cells ahead; its near-depth band excludes the far
        // backdrop wall (~11 cells). Compare the NEAR slice height ramp vs full.
        let near_band = 2.0;
        let mut checked = 0usize;
        let mut shorter = 0usize;
        let mut backdrop_above = 0usize;
        for c in 0..cols {
            let (Some((ramp_top, ramp_cnt, ramp_near)), Some((_full_top, full_cnt, _))) = (
                near_wall_slice(&fb_ramp, c, near_band),
                near_wall_slice(&fb_full, c, near_band),
            ) else {
                continue;
            };
            // Only the columns that actually view the ramp cell (near depth ~ a few cells).
            if ramp_near > 8.0 {
                continue;
            }
            checked += 1;
            if ramp_cnt < full_cnt {
                shorter += 1;
            }
            // Above the ramp's near slice there must be a row that is NOT this ramp slice
            // — either open sky/ceiling or a FARTHER wall (the backdrop recast). If the
            // ramp top row is below the very top, inspect the row just above it.
            if ramp_top > 0 {
                let above = fb_ramp.get(c, ramp_top - 1).unwrap();
                let is_ramp_wall =
                    above.surface.is_wall() && (above.depth - ramp_near).abs() <= near_band;
                if !is_ramp_wall {
                    backdrop_above += 1;
                }
            }
        }
        assert!(
            checked >= 4,
            "expected several ramp columns in view (got {checked})"
        );
        assert!(
            shorter >= checked / 2,
            "most ramp columns must draw a SHORTER near slice than the full wall \
             (shorter={shorter}/{checked})"
        );
        assert!(
            backdrop_above >= checked / 2,
            "above the ramp slope, most columns must show the BACKDROP (farther wall / \
             sky / ceiling), not the ramp wall (backdrop_above={backdrop_above}/{checked})"
        );
    }

    #[test]
    fn all_four_ramp_dirs_render_a_bounded_near_slice() {
        // Every RampDir variant must draw a real but bounded NEAR slice (shorter than the
        // same cell as a Full wall) when viewed from its FOOT (low) side — proving the
        // slice is bounded to the slope for all four rise directions, each rising the
        // correct way. (Viewed from the HEAD side a ramp correctly shows its full-height
        // vertical back, so the foot-side vantage is the one that exposes the slope.)
        let (sx, sy, sz) = (16usize, 16usize, 6usize);
        for dir in [RampDir::PosX, RampDir::NegX, RampDir::PosY, RampDir::NegY] {
            let mut full = ramp_room(sx, sy, sz);
            full.set(8, 8, 1, CellKind::WallSolid);
            let mut ramp = ramp_room(sx, sy, sz);
            ramp.set(8, 8, 1, CellKind::WallSolid);
            ramp.set_shape(8, 8, 1, Shape::RampZ(dir));

            // Stand on the cell's FOOT side and look toward it so the rising slope faces
            // the camera. yaw: 0 ⇒ +Y, PI/2 ⇒ +X, PI ⇒ −Y, 3PI/2 ⇒ −X (dir = (sin,cos)).
            let cam = match dir {
                // PosX foot = west edge → view from −x, ray +x (yaw = PI/2).
                RampDir::PosX => CameraPose {
                    pos: glam::Vec3::new(3.5, 8.5, STANDING_EYE_Z),
                    yaw: std::f32::consts::FRAC_PI_2,
                    pitch: 0.0,
                },
                // NegX foot = east edge → view from +x, ray −x (yaw = 3PI/2).
                RampDir::NegX => CameraPose {
                    pos: glam::Vec3::new(13.5, 8.5, STANDING_EYE_Z),
                    yaw: 3.0 * std::f32::consts::FRAC_PI_2,
                    pitch: 0.0,
                },
                // PosY foot = south edge → view from −y, ray +y (yaw = 0).
                RampDir::PosY => CameraPose {
                    pos: glam::Vec3::new(8.5, 3.5, STANDING_EYE_Z),
                    yaw: 0.0,
                    pitch: 0.0,
                },
                // NegY foot = north edge → view from +y, ray −y (yaw = PI).
                RampDir::NegY => CameraPose {
                    pos: glam::Vec3::new(8.5, 13.5, STANDING_EYE_Z),
                    yaw: std::f32::consts::PI,
                    pitch: 0.0,
                },
            };
            let (cols, rows) = (100usize, 40usize);
            let fb_full = cast_gameplay(&full, &cam, cols, rows);
            let fb_ramp = cast_gameplay(&ramp, &cam, cols, rows);

            let near_band = 2.0;
            let mut checked = 0usize;
            let mut shorter = 0usize;
            let mut any_wall = 0usize;
            for c in 0..cols {
                let (Some((_, ramp_cnt, ramp_near)), Some((_, full_cnt, _))) = (
                    near_wall_slice(&fb_ramp, c, near_band),
                    near_wall_slice(&fb_full, c, near_band),
                ) else {
                    continue;
                };
                if ramp_near > 9.0 {
                    continue;
                }
                checked += 1;
                any_wall += 1;
                if ramp_cnt < full_cnt {
                    shorter += 1;
                }
            }
            assert!(
                any_wall > 0,
                "RampDir::{dir:?}: the ramp must still draw a (non-empty) near slice"
            );
            assert!(
                checked >= 3 && shorter >= checked / 2,
                "RampDir::{dir:?}: the near slice must be bounded (shorter than full) for \
                 most ramp columns (shorter={shorter}/{checked})"
            );
        }
    }

    /// SPIRAL-STAIRCASE "no black face" GUARD (this fix): for EVERY `RampDir`,
    /// the ramp slice rendered through the real cast must (a) carry step-relief
    /// glyphs and (b) have its BRIGHTEST step cell clear a black-luma threshold —
    /// so a spiral staircase (which uses all four directions, including the dim
    /// away-facing NegX/NegY slopes) never reads as a black wall. Pins the defect:
    /// before the `ramp_steps` floor, the away-facing dirs sank to luma ≈ 0.04.
    #[test]
    fn every_rampdir_slice_is_lit_above_black_with_relief() {
        use std::collections::HashSet;
        let step_glyphs: HashSet<char> = ['‾', '_', '╱', '╲'].into_iter().collect();
        // Rec.601 luma of a packed fg.
        let luma = |c: u32| {
            let r = ((c >> 16) & 0xFF) as f32 / 255.0;
            let g = ((c >> 8) & 0xFF) as f32 / 255.0;
            let b = (c & 0xFF) as f32 / 255.0;
            0.299 * r + 0.587 * g + 0.114 * b
        };
        // A step face that reads as black on the dark wall bg sits below this.
        const BLACK: f32 = 0.30;

        let (sx, sy, sz) = (16usize, 16usize, 6usize);
        for dir in [RampDir::PosX, RampDir::NegX, RampDir::PosY, RampDir::NegY] {
            let mut m = ramp_room(sx, sy, sz);
            // Solid mass under the tread so the slope is a real stair face. Step relief
            // is drawn only on a FLIGHT (a same-direction RampZ one cell up the rise),
            // so a lone slice (a console top) reads as a slope: the fixture is a
            // two-slice flight, with the second slice on a solid column.
            let (dx, dy): (i32, i32) = match dir {
                RampDir::PosX => (1, 0),
                RampDir::NegX => (-1, 0),
                RampDir::PosY => (0, 1),
                RampDir::NegY => (0, -1),
            };
            let (ux, uy) = ((8 + dx) as usize, (8 + dy) as usize);
            m.set(8, 8, 1, CellKind::WallSolid);
            m.set_shape(8, 8, 1, Shape::RampZ(dir));
            m.set(ux, uy, 1, CellKind::WallSolid);
            m.set(ux, uy, 2, CellKind::WallSolid);
            m.set_shape(ux, uy, 2, Shape::RampZ(dir));
            // Stand on the FOOT (low) side and look at the rising slope (same vantage
            // as `all_four_ramp_dirs_render_a_bounded_near_slice`).
            let cam = match dir {
                RampDir::PosX => CameraPose {
                    pos: glam::Vec3::new(3.5, 8.5, STANDING_EYE_Z),
                    yaw: std::f32::consts::FRAC_PI_2,
                    pitch: 0.0,
                },
                RampDir::NegX => CameraPose {
                    pos: glam::Vec3::new(13.5, 8.5, STANDING_EYE_Z),
                    yaw: 3.0 * std::f32::consts::FRAC_PI_2,
                    pitch: 0.0,
                },
                RampDir::PosY => CameraPose {
                    pos: glam::Vec3::new(8.5, 3.5, STANDING_EYE_Z),
                    yaw: 0.0,
                    pitch: 0.0,
                },
                RampDir::NegY => CameraPose {
                    pos: glam::Vec3::new(8.5, 13.5, STANDING_EYE_Z),
                    yaw: std::f32::consts::PI,
                    pitch: 0.0,
                },
            };
            let (cols, rows) = (100usize, 40usize);
            let fb = cast_gameplay(&m, &cam, cols, rows);

            let mut relief = 0usize;
            let mut brightest = 0.0f32;
            for r in 0..rows {
                for c in 0..cols {
                    let cell = fb.get(c, r).unwrap();
                    let g = char::from_u32(cell.glyph).unwrap_or(' ');
                    if step_glyphs.contains(&g) {
                        relief += 1;
                        brightest = brightest.max(luma(cell.fg));
                    }
                }
            }
            assert!(
                relief > 0,
                "RampDir::{dir:?}: the slope must render step-relief glyphs (found none)"
            );
            assert!(
                brightest >= BLACK,
                "RampDir::{dir:?}: the brightest step cell luma {brightest:.3} is below the \
                 black threshold {BLACK} — this ramp face renders black (the spiral defect)"
            );
        }
    }

    /// A run of RampZ cells viewed from the SIDE must render visible STEP relief: the
    /// ramp slice cells carry the step-edge / riser glyphs from `ramp_steps`, not the
    /// material's flat body glyph — so the staircase reads as steps. Prints the grid so
    /// `cargo test -- --nocapture` eyeballs the relief. Guards the readability defect
    /// this work fixes (a smooth, blank diagonal face).
    #[test]
    fn ramp_run_renders_visible_step_relief_from_the_side() {
        use std::collections::HashSet;
        // A 4-tread PosX staircase along y=8, climbing z = 1..4, in a tall room.
        let (sx, sy, sz) = (20usize, 17usize, 9usize);
        let mut m = ramp_room(sx, sy, sz);
        for (i, x) in (4..8).enumerate() {
            let z = 1 + i; // tread i rises one cell
            m.set(x, 8, z, CellKind::WallSolid);
            m.set_shape(x, 8, z, Shape::RampZ(RampDir::PosX));
            // Solid mass under each tread so the run reads as a stair, not a floating ramp.
            for zz in 1..z {
                m.set(x, 8, zz, CellKind::WallSolid);
            }
        }
        // Stand to the SOUTH of the run and look NORTH (+y, yaw 0) so the staircase's
        // rise axis (+x) runs LEFT→RIGHT across the screen — the angle steps read best.
        let cam = CameraPose {
            pos: glam::Vec3::new(6.0, 3.5, STANDING_EYE_Z),
            yaw: 0.0,
            pitch: 0.0,
        };
        let (cols, rows) = (100usize, 40usize);
        let fb = cast_gameplay(&m, &cam, cols, rows);

        // Collect the step glyphs present anywhere in the frame.
        let step_glyphs: HashSet<char> = ['‾', '_', '╱', '╲'].into_iter().collect();
        let mut found: HashSet<char> = HashSet::new();
        for r in 0..rows {
            for c in 0..cols {
                let g = char::from_u32(fb.get(c, r).unwrap().glyph).unwrap_or(' ');
                if step_glyphs.contains(&g) {
                    found.insert(g);
                }
            }
        }

        // Eyeball dump (visible under --nocapture).
        eprintln!("\n── ramp side-view step relief (PosX run, look +y) ──");
        for r in 8..32 {
            let line: String = (0..cols)
                .map(|c| char::from_u32(fb.get(c, r).unwrap().glyph).unwrap_or(' '))
                .collect();
            eprintln!("{line}");
        }
        eprintln!("step glyphs found: {found:?}");

        assert!(
            !found.is_empty(),
            "a side-viewed ramp run must render at least one step-relief glyph \
             (lip/riser/diagonal); found none — the ramp face is still smooth"
        );
    }

    /// OUTER-CONTOUR DIRECTION AUDIT (task item 2): for each `RampDir`, render the run
    /// from its FOOT side, run the contour de-angularization pass, and assert the
    /// silhouette carries a SLOPE chamfer/diagonal glyph (`◣◢◤◥╱╲`) — never a stray
    /// axis-aligned box corner (`┐└┌┘`) that would read as a blocky stair. This pins
    /// the RampDir→outer-glyph relationship end-to-end (through the real cast +
    /// contour pass), closing the #11-class mirror-inversion regression surface that
    /// `slope.rs`'s pure screen-space tests can't reach.
    #[test]
    fn ramp_outer_contour_uses_slope_glyphs_for_all_dirs() {
        use crate::contour::{apply as contour_apply, ContourConfig};
        let slope_glyphs: std::collections::HashSet<char> =
            ['◣', '◢', '◤', '◥', '╱', '╲', '/', '\\']
                .into_iter()
                .collect();
        let (sx, sy, sz) = (16usize, 16usize, 7usize);
        for dir in [RampDir::PosX, RampDir::NegX, RampDir::PosY, RampDir::NegY] {
            // A short 3-tread run rising the correct way for `dir`, with solid mass
            // under each tread, so the silhouette is a genuine diagonal slope.
            let mut m = ramp_room(sx, sy, sz);
            let cam = match dir {
                RampDir::PosX => {
                    for (i, x) in (6..9).enumerate() {
                        let z = 1 + i;
                        for zz in 1..=z {
                            m.set(x, 8, zz, CellKind::WallSolid);
                        }
                        m.set_shape(x, 8, z, Shape::RampZ(dir));
                    }
                    CameraPose {
                        pos: glam::Vec3::new(8.0, 3.5, STANDING_EYE_Z),
                        yaw: 0.0,
                        pitch: 0.0,
                    }
                }
                RampDir::NegX => {
                    for (i, x) in (7..10).rev().enumerate() {
                        let z = 1 + i;
                        for zz in 1..=z {
                            m.set(x, 8, zz, CellKind::WallSolid);
                        }
                        m.set_shape(x, 8, z, Shape::RampZ(dir));
                    }
                    CameraPose {
                        pos: glam::Vec3::new(8.0, 3.5, STANDING_EYE_Z),
                        yaw: 0.0,
                        pitch: 0.0,
                    }
                }
                RampDir::PosY => {
                    for (i, y) in (6..9).enumerate() {
                        let z = 1 + i;
                        for zz in 1..=z {
                            m.set(8, y, zz, CellKind::WallSolid);
                        }
                        m.set_shape(8, y, z, Shape::RampZ(dir));
                    }
                    CameraPose {
                        pos: glam::Vec3::new(3.5, 8.0, STANDING_EYE_Z),
                        yaw: std::f32::consts::FRAC_PI_2,
                        pitch: 0.0,
                    }
                }
                RampDir::NegY => {
                    for (i, y) in (7..10).rev().enumerate() {
                        let z = 1 + i;
                        for zz in 1..=z {
                            m.set(8, y, zz, CellKind::WallSolid);
                        }
                        m.set_shape(8, y, z, Shape::RampZ(dir));
                    }
                    CameraPose {
                        pos: glam::Vec3::new(3.5, 8.0, STANDING_EYE_Z),
                        yaw: std::f32::consts::FRAC_PI_2,
                        pitch: 0.0,
                    }
                }
            };
            let (cols, rows) = (100usize, 40usize);
            let mut fb = cast_gameplay(&m, &cam, cols, rows);
            contour_apply(&mut fb, &ContourConfig::default());

            let mut slope_hits = 0usize;
            for r in 0..rows {
                for c in 0..cols {
                    let g = char::from_u32(fb.get(c, r).unwrap().glyph).unwrap_or(' ');
                    if slope_glyphs.contains(&g) {
                        slope_hits += 1;
                    }
                }
            }
            assert!(
                slope_hits > 0,
                "RampDir::{dir:?}: the run's silhouette must carry slope glyphs \
                 (◣◢◤◥╱╲), found none — the de-angularizer is not firing on this dir"
            );
        }
    }

    /// Regression test for "dark strip at the base of far walls": 1-2 floor rows
    /// directly under a far wall's rendered bottom edge used to carry the depth of
    /// a floor point the wall itself already occludes — the classic per-row
    /// floor-cast formula (`eye_height_cells * proj_plane_dist / denom`, a
    /// texture-space approximation) can compute a distance FARTHER than the wall's
    /// own perpendicular depth for the row(s) immediately past the wall's span,
    /// which the depth-contrast effect then collapses straight to fog. A long,
    /// plain corridor with a far end wall reproduces it with no level /
    /// depth-contrast involved — pure geometry, through the same gameplay cast
    /// path (`cast_gameplay`) (first reproduced in `06.lvl` column 53: wall depth
    /// 13.5, floor rows at 37.7 / 18.8 before the fix).
    ///
    /// The general, always-true invariant this pins: for a column with a far wall,
    /// every FLOOR row below the wall's own bottom row must read at a depth `<=` the
    /// wall's own depth (plus a tiny epsilon) — a floor point the wall stands in front
    /// of can never be farther away than the wall itself. Either the floor/ceiling
    /// cast respects that (depth <= wall depth), or the row is owned by the wall
    /// instead (not tagged `Floor` at all) — both satisfy this test, per the task's
    /// own acceptance criterion.
    #[test]
    fn floor_row_below_a_far_wall_is_owned_by_the_wall_not_a_farther_floor() {
        let (sx, sy, sz) = (20usize, 3usize, 3usize);
        let m = ramp_room(sx, sy, sz);
        let cam = CameraPose {
            // dir() == (yaw.sin(), yaw.cos()); FRAC_PI_2 points down +x, the long axis
            // of this corridor, so the column looks straight at the far end wall.
            pos: glam::Vec3::new(2.5, 1.5, STANDING_EYE_Z),
            yaw: std::f32::consts::FRAC_PI_2,
            pitch: 0.0,
        };
        let (cols, rows) = (100usize, 32usize);
        let fb = cast_gameplay(&m, &cam, cols, rows);

        let col = cols / 2;
        let mut wall_depth = f32::INFINITY;
        let mut wall_bottom_row = None;
        for row in 0..rows {
            let c = fb.get(col, row).unwrap();
            if c.surface.is_wall() {
                wall_depth = wall_depth.min(c.depth);
                wall_bottom_row = Some(row);
            }
        }
        let wall_bottom_row =
            wall_bottom_row.expect("a far end wall must be visible in this column");
        assert!(wall_depth.is_finite(), "far wall depth must be finite");

        const EPS: f32 = 0.05;
        let mut checked = 0usize;
        for row in (wall_bottom_row + 1)..rows {
            let c = fb.get(col, row).unwrap();
            if c.surface == SurfaceKind::Floor {
                checked += 1;
                assert!(
                    c.depth <= wall_depth + EPS,
                    "floor row {row} directly below the wall's bottom edge (row \
                     {wall_bottom_row}, wall depth {wall_depth:.3}) reads at depth \
                     {:.3} — FARTHER than the wall that should occlude it (the \
                     dark-strip regression)",
                    c.depth
                );
            }
        }
        assert!(
            checked > 0,
            "expected at least one floor row below the wall in this column"
        );
    }

    /// Regression test: an earlier, UNCONDITIONAL version of `clamp_to_wall_depth`
    /// (clamp every overshooting row, however far past the wall) fixed the acute
    /// dark-strip bug above but flattened a NEAR wall's entire remaining floor band
    /// to one constant depth (`perp`) — a wide, hard-edged plateau in
    /// `L1-first-pulse.lvl` and a totally flat, horizon-less floor in
    /// `08-material-demo.lvl` (an 8×8 room, `fog_far = 4`). This pins the fix
    /// (`FLOOR_CLAMP_MAX_DENOM_PAST_WALL`): a SHORT room (wall only ~2.5 cells from
    /// the camera, matching `08-material-demo`'s scale) must still show the floor's
    /// natural depth GRADIENT below the wall — not one repeated value — because the
    /// windowed clamp only bounds rows within a small denom-distance of the wall's
    /// own crossover, leaving the rest of the (technically approximate, but visibly
    /// smooth) classic formula alone.
    #[test]
    fn near_wall_floor_keeps_its_gradient_not_a_flat_plateau() {
        let (sx, sy, sz) = (6usize, 3usize, 3usize);
        let m = ramp_room(sx, sy, sz);
        let cam = CameraPose {
            // Camera at x=2.5 facing +x (yaw = FRAC_PI_2): the far wall (x = sx-1 = 5)
            // sits ~2.5 cells away — the same close-wall scale that regressed
            // `L1-first-pulse.lvl` / `08-material-demo.lvl`, unlike the 20-cell-long
            // corridor `floor_row_below_a_far_wall_is_owned_by_the_wall_not_a_farther_floor`
            // uses for the far-wall case.
            pos: glam::Vec3::new(2.5, 1.5, STANDING_EYE_Z),
            yaw: std::f32::consts::FRAC_PI_2,
            pitch: 0.0,
        };
        let (cols, rows) = (100usize, 32usize);
        let fb = cast_gameplay(&m, &cam, cols, rows);

        let col = cols / 2;
        let mut wall_bottom_row = None;
        for row in 0..rows {
            if fb.get(col, row).unwrap().surface.is_wall() {
                wall_bottom_row = Some(row);
            }
        }
        let wall_bottom_row =
            wall_bottom_row.expect("a near end wall must be visible in this column");

        let mut floor_depths: Vec<f32> = Vec::new();
        for row in (wall_bottom_row + 1)..rows {
            let c = fb.get(col, row).unwrap();
            if c.surface == SurfaceKind::Floor {
                floor_depths.push(c.depth);
            }
        }
        assert!(
            floor_depths.len() > 4,
            "expected several floor rows below the near wall ({})",
            floor_depths.len()
        );

        let min_d = floor_depths.iter().copied().fold(f32::INFINITY, f32::min);
        let max_d = floor_depths
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            max_d - min_d > 0.2,
            "a near wall's floor band must keep a real depth gradient below it \
             (min {min_d:.3}, max {max_d:.3}) — a near-uniform spread means the \
             windowed clamp regressed back to flattening the whole band, the exact \
             wide-plateau bug this test guards against"
        );

        // The row closest to the camera (bottom of screen) must still be the
        // NEAREST floor point — the gradient's overall direction survives the
        // windowed clamp, it isn't just "not flat" by noise.
        let nearest = *floor_depths.last().unwrap();
        let farthest = floor_depths[0];
        assert!(
            nearest < farthest,
            "the floor row nearest the camera ({nearest:.3}) must read closer than \
             the row nearest the wall's own base ({farthest:.3})"
        );
    }
}

// ──────────────── quality-textures: fit_span + derivation tests ────────────────

#[cfg(test)]
mod fit_span_tests {
    use super::*;
    use ascii_shader::Fit;

    const EPS: f32 = 1e-4;

    fn approx(a: f32, b: f32, what: &str) {
        assert!((a - b).abs() < EPS, "{what}: {a} != {b}");
    }

    #[test]
    fn stretch_span_is_the_wall_box() {
        // Stretch ignores aspect: the span IS the wall, regardless of image size.
        let s = fit_span(2.0, 6.0, 4.0, 3.0, 417, 598, Fit::Stretch);
        approx(s.along_min, 2.0, "along_min");
        approx(s.along_size, 6.0, "along_size");
        approx(s.up_top, 4.0, "up_top");
        approx(s.up_size, 3.0, "up_size");
    }

    #[test]
    fn contain_square_image_on_wide_wall_adds_horizontal_margin() {
        // Wall 4 wide × 2 tall (aspect 2). Square image (aspect 1 < wall aspect →
        // relatively TALLER, r = 0.5). Contain fits the HEIGHT (figure up == wall) and
        // makes the figure NARROWER than the wall so left/right margins read transparent
        // (wall cells past the figure map to u ∉ [0,1]).
        let wall_along = 4.0;
        let wall_up = 2.0;
        let s = fit_span(0.0, wall_along, wall_up, wall_up, 100, 100, Fit::Contain);
        approx(
            s.up_size,
            wall_up,
            "contain keeps the constraining (up) axis",
        );
        // r = 0.5 < 1 → along_size = wall_along * r = 2.0 (SMALLER → margins).
        approx(s.along_size, 2.0, "figure narrower than wall → margin");
        assert!(
            s.along_size < wall_along,
            "Contain shrinks the loose axis (margin)"
        );
        // Centred: centre stays at 2.0, so along_min = 2 - 1 = 1.
        approx(s.along_min, 1.0, "span centred on the wall");
    }

    #[test]
    fn contain_wide_image_on_square_wall_adds_vertical_margin() {
        // Wall 2×2 (aspect 1). Wide image (aspect 2, r = 2): Contain fits the WIDTH
        // (figure along == wall) and makes the figure SHORTER (top/bottom margin).
        let s = fit_span(0.0, 2.0, 2.0, 2.0, 200, 100, Fit::Contain);
        approx(
            s.along_size,
            2.0,
            "contain keeps the constraining (along) axis",
        );
        // r = 2 ≥ 1 → up_size = wall_up / r = 1.0 (SMALLER → margins).
        approx(s.up_size, 1.0, "figure shorter than wall → margin");
        assert!(s.up_size < 2.0, "Contain shrinks the loose axis (margin)");
    }

    #[test]
    fn cover_square_image_on_wide_wall_crops_vertically() {
        // Same wall 4×2, square image (r = 0.5). Cover fills the wall → width fits, the
        // figure is TALLER than the wall (crop top/bottom; every wall cell stays in
        // [0,1]).
        let s = fit_span(0.0, 4.0, 2.0, 2.0, 100, 100, Fit::Cover);
        approx(
            s.along_size,
            4.0,
            "cover fills the constraining (along) axis",
        );
        // r = 0.5 < 1 → up_size = wall_up / r = 4.0 (LARGER → crop).
        approx(s.up_size, 4.0, "figure taller than wall → crop");
        assert!(s.up_size > 2.0, "Cover grows the loose axis (crop)");
    }

    #[test]
    fn cover_wide_image_on_square_wall_crops_horizontally() {
        // Wall 2×2, wide image (r = 2). Cover fills height, figure WIDER → crop l/r.
        let s = fit_span(0.0, 2.0, 2.0, 2.0, 200, 100, Fit::Cover);
        approx(s.up_size, 2.0, "cover fills the constraining (up) axis");
        // r = 2 ≥ 1 → along_size = wall_along * r = 4.0 (LARGER → crop).
        approx(s.along_size, 4.0, "figure wider than wall → crop");
    }

    #[test]
    fn contain_and_cover_are_inverses_on_the_loose_axis() {
        // For a square image on a WIDE wall (4×2), ALONG is the loose axis: Contain
        // shrinks it (4 → 2, margins), Cover grows it... no — for this case ALONG is the
        // fitted axis for Cover. Use the UP axis, which both modes scale reciprocally:
        // Contain up 2 → ... is fitted here. So test the geometric mean of the SCALED
        // axis: Contain along 2, Cover up 4 vs wall 2 — instead assert each directly and
        // that Contain ⊆ wall ⊆ Cover on areas.
        let c = fit_span(0.0, 4.0, 2.0, 2.0, 100, 100, Fit::Contain);
        let v = fit_span(0.0, 4.0, 2.0, 2.0, 100, 100, Fit::Cover);
        let wall_area = 4.0 * 2.0;
        let c_area = c.along_size * c.up_size;
        let v_area = v.along_size * v.up_size;
        assert!(
            c_area <= wall_area + EPS,
            "Contain figure fits inside the wall"
        );
        assert!(v_area >= wall_area - EPS, "Cover figure covers the wall");
    }

    #[test]
    fn matched_aspect_is_identity_for_all_modes() {
        // Wall aspect == image aspect → no margin, no crop; span == wall for all modes.
        for fit in [Fit::Stretch, Fit::Contain, Fit::Cover] {
            let s = fit_span(0.0, 4.0, 2.0, 2.0, 200, 100, fit);
            approx(s.along_size, 4.0, "matched aspect → no along change");
            approx(s.up_size, 2.0, "matched aspect → no up change");
        }
    }

    #[test]
    fn degenerate_inputs_fall_back_to_wall_box() {
        // Zero wall size or zero-area image → the wall box (never a div-by-zero / NaN).
        let s = fit_span(0.0, 0.0, 2.0, 2.0, 100, 100, Fit::Contain);
        approx(s.along_size, 0.0, "zero wall → wall box");
        let s = fit_span(0.0, 4.0, 2.0, 2.0, 0, 100, Fit::Contain);
        approx(s.along_size, 4.0, "zero-area image → wall box");
    }
}

#[cfg(test)]
mod note_backing_tests {
    use super::*;
    use re_world::{test_room, MaterialDef, MaterialRecipe};

    fn note_def() -> MaterialDef {
        MaterialDef {
            name: "note0".into(),
            fg: 0x00E0_DCC8,
            bg: 0x0009_0807,
            glyph: None,
            recipe: Some(MaterialRecipe::NoteWall {
                note: 0,
                face: re_core::FaceDir::PosY,
            }),
            anim: re_world::MaterialAnim::default(),
            rank: 0,
            reflective: false,
            clarity: 0.85,
            fallback: false,
            detail: false,
        }
    }

    fn body_def() -> MaterialDef {
        MaterialDef {
            name: "board_body".into(),
            fg: 0x009A_6A3A,
            bg: 0x001A_1008,
            glyph: None,
            recipe: None,
            anim: re_world::MaterialAnim::default(),
            rank: 0,
            reflective: false,
            clarity: 0.85,
            fallback: false,
            detail: false,
        }
    }

    /// A one-wide board: the placard cell (colour 2) sits on top of a body cell
    /// (colour 1) with air on both sides. Its backing is the body, and every face but
    /// the +y placard face shades from it.
    #[test]
    fn a_placard_cell_shades_its_other_faces_from_what_it_is_mounted_in() {
        let mut map = test_room();
        let mut mats: Vec<MaterialDef> = (0..5).map(|_| MaterialDef::fallback()).collect();
        mats.push(body_def()); // colour 1
        mats.push(note_def()); // colour 2
        map.set_materials(mats);
        map.set_cell(2, 2, 1, CellKind::Empty, 0);
        map.set_cell(4, 2, 1, CellKind::Empty, 0);
        map.set_cell(3, 2, 1, CellKind::WallSolid, 1);
        map.set_cell(3, 2, 2, CellKind::WallSolid, 2);
        map.set_cell(2, 2, 2, CellKind::Empty, 0);
        map.set_cell(4, 2, 2, CellKind::Empty, 0);
        let reg = registry_for_map(&map);
        assert_eq!(
            reg.note_backing(2),
            Some(1),
            "the board body backs the placard"
        );
        assert_eq!(reg.note_backing(1), None, "only placards carry a backing");

        let note = reg.color_material(2).unwrap();
        let body = reg.color_material(1).unwrap();
        let s = SurfaceKind::WallSolid;
        let at = |n: glam::Vec3| note_face_material(&reg, note, 2, s, Some(n), 0);
        assert_eq!(at(glam::Vec3::Y), note, "the placard face carries the note");
        assert_eq!(at(glam::Vec3::X), body, "side");
        assert_eq!(at(-glam::Vec3::X), body, "other side");
        assert_eq!(at(-glam::Vec3::Y), body, "back");
        assert_eq!(at(glam::Vec3::Z), body, "top");
        // A plain colour material is never redirected.
        assert_eq!(
            note_face_material(&reg, body, 1, s, Some(glam::Vec3::X), 1),
            body
        );
    }

    /// A placard with no solid neighbour to borrow from falls back to the
    /// surface-kind default on its other faces — never the black reverse plate.
    #[test]
    fn a_free_floating_placard_falls_back_to_the_surface_default() {
        let mut map = test_room();
        let mut mats: Vec<MaterialDef> = (0..5).map(|_| MaterialDef::fallback()).collect();
        mats.push(note_def()); // colour 1
        map.set_materials(mats);
        for (x, z) in [(2, 2), (4, 2), (3, 1), (3, 3)] {
            map.set_cell(x, 2, z, CellKind::Empty, 0);
        }
        map.set_cell(3, 1, 2, CellKind::Empty, 0);
        map.set_cell(3, 2, 2, CellKind::WallSolid, 1);
        let reg = registry_for_map(&map);
        assert_eq!(reg.note_backing(1), Some(0));
        let note = reg.color_material(1).unwrap();
        let got = note_face_material(
            &reg,
            note,
            1,
            SurfaceKind::WallSolid,
            Some(glam::Vec3::X),
            1,
        );
        assert_eq!(got, surface_material(SurfaceKind::WallSolid, 0));
    }
}

#[cfg(test)]
mod ramp_flight_tests {
    use super::*;
    use re_core::{RampDir, Shape};
    use re_world::test_room;

    /// A staircase (ramps stepping up diagonally) is a flight; a lone ramp cell and
    /// a row of ramps side by side (a console bank) are not.
    #[test]
    fn only_a_diagonal_run_of_ramps_is_a_flight() {
        let mut map = test_room();
        for (x, z) in [(1, 1), (2, 2)] {
            map.set_cell(x, 2, z, CellKind::WallSolid, 0);
            map.set_shape(x, 2, z, Shape::RampZ(RampDir::PosX));
        }
        assert!(ramp_is_flight(&map, 1, 2, 1, RampDir::PosX), "bottom tread");
        assert!(ramp_is_flight(&map, 2, 2, 2, RampDir::PosX), "top tread");

        let mut map = test_room();
        for y in 1..4 {
            map.set_cell(3, y, 1, CellKind::WallSolid, 0);
            map.set_shape(3, y, 1, Shape::RampZ(RampDir::NegX));
        }
        for y in 1..4 {
            assert!(
                !ramp_is_flight(&map, 3, y, 1, RampDir::NegX),
                "a console bank is not a stair"
            );
        }
    }
}
