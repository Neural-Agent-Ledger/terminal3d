//! MATERIAL authoring (M5b): select a `[material]`, paint its colour-palette id
//! onto cells, and edit its appearance (fg/bg/glyph/recipe) — all pure functions
//! over [`EditorState`], no I/O, no terminal.
//!
//! ## Model
//! A level's authored materials live in `level.materials`. A material at index
//! `i` resolves to the **1-based** colour-palette id `i + 1` (the SAME id space
//! `[color_cell_ids]` and overlay zones use — see `re_level::push_material` /
//! `zone_material_id`). Painting a material onto a cell stamps `i + 1` into that
//! cell's `side_data`; the renderer reads `side_data` back and dispatches the
//! cell's material from the colour palette. Clearing resets `side_data` to `0`
//! (the surface-kind default). Persistence: materials round-trip via the
//! `[material]` section and per-cell ids via `[cells] material_ids` — both already
//! handled by `re_voximport::to_lvl_text` / `re_level`'s parser.
//!
//! ## Keys (documented in `chrome.rs` help + `main.rs` routing)
//! * `m` — enter material mode / cycle the selected material FORWARD.
//! * `M` (Shift+m) — cycle the selected material BACKWARD.
//! * `e` — create a NEW material (unique name, default look) and select it.
//! * `b` — cycle which material-def FIELD the `+`/`-` keys edit (Fg→Bg→Glyph→Recipe).
//! * `+` / `-` — adjust the focused field (Fg/Bg: step the palette; Glyph: cycle a
//!   candidate set; Recipe: cycle the known recipe variants incl. "none").
//! * `space` — stamp the selected material id into the cursor cell (paint).
//! * `x` — clear the cursor cell's material id back to default.
use crate::state::EditorState;
use re_world::{
    CellKind, Fit, ImageSpan, MaterialAnimKind, MaterialRecipe, SubcellMode, VoiceMode,
};

/// Which appearance field of the selected material the `+`/`-` (or `c` for enums)
/// keys edit. Cycled by `b`.
///
/// EXHAUSTIVE: every field of [`re_level::LevelMaterial`] is reachable — the flat
/// colours, glyph, recipe variant AND its per-variant params, the animation block
/// (kind/speed/amplitude), the LOD rank, the surface binding, the mirror flags
/// (reflective/clarity), and the catalog `preset`. The three `RecipeP*` slots map
/// to the CURRENT recipe variant's params by position (Brick's aspect/mortar_w/
/// grain, Circuit's scale/trace_w, …); a slot with no param for the active recipe
/// is a hinted no-op so the cycle stays uniform.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MaterialField {
    /// Foreground colour — `+`/`-` step the [`PALETTE`] swatches.
    Fg,
    /// Background colour — `+`/`-` step the [`PALETTE`] swatches.
    Bg,
    /// Fixed glyph (`Option<u32>`) — `+`/`-` cycle [`GLYPHS`] (incl. "none").
    Glyph,
    /// Procedural recipe VARIANT (`Option<MaterialRecipe>`) — `+`/`-` cycle.
    Recipe,
    /// Recipe param slot 0 (e.g. Brick.aspect / Circuit.scale) — `+`/`-` step.
    RecipeP0,
    /// Recipe param slot 1 (e.g. Brick.mortar_w / Circuit.trace_w) — `+`/`-` step.
    RecipeP1,
    /// Recipe param slot 2 (e.g. Brick.grain / Grain.tile_eps) — `+`/`-` step.
    RecipeP2,
    /// Animation kind — `c` cycles None/Pulse/Shift/Flicker.
    AnimKind,
    /// Animation speed — `+`/`-` step (numeric).
    AnimSpeed,
    /// Animation amplitude — `+`/`-` step (numeric).
    AnimAmplitude,
    /// LOD detail rank `[0,127]` — `+`/`-` step.
    Rank,
    /// Surface binding — `c` cycles none/wall_solid/wall_trace/wall_matrix/floor/ceiling.
    Surface,
    /// Mirror flag — `c` toggles reflective on/off.
    Reflective,
    /// Mirror clarity `[0,1]` — `+`/`-` step (only meaningful when reflective).
    Clarity,
    /// Catalog preset name — `c` cycles none + the known catalog presets.
    Preset,
    /// Image-texture binding — `c` cycles `(none)` + the level's `[texture]` names.
    /// Sets the material's `texture = NAME` key (only meaningful with `recipe = image`).
    Texture,
    /// Image sub-cell detail mode — `c` cycles off/quad/braille/ascii/contour/lines
    /// (`SubcellMode`). Only meaningful with `recipe = image`.
    Subcell,
    /// Image fit mode — `c` cycles stretch/contain/cover (`Fit`). Only meaningful with
    /// `recipe = image` AND no explicit `span` (an authored span overrides `fit`).
    Fit,
    /// Image world-anchored span `along_min` — `+`/`-` step. Authoring ANY span
    /// component turns the `Option<ImageSpan>` on (seeded from a sane default).
    SpanAlongMin,
    /// Image span `along_size` — `+`/`-` step, clamped `> 0`.
    SpanAlongSize,
    /// Image span `up_top` — `+`/`-` step.
    SpanUpTop,
    /// Image span `up_size` — `+`/`-` step, clamped `> 0`.
    SpanUpSize,
}

/// The material-def field cycle order (`b` steps through it, wrapping). The final
/// entry is a padding duplicate of `Fg` — [`MaterialField::next`] skips it so the
/// cycle visits each logical field exactly once.
pub const MATERIAL_FIELDS: [MaterialField; 23] = [
    MaterialField::Fg,
    MaterialField::Bg,
    MaterialField::Glyph,
    MaterialField::Recipe,
    MaterialField::RecipeP0,
    MaterialField::RecipeP1,
    MaterialField::RecipeP2,
    MaterialField::AnimKind,
    MaterialField::AnimSpeed,
    MaterialField::AnimAmplitude,
    MaterialField::Rank,
    MaterialField::Surface,
    MaterialField::Reflective,
    MaterialField::Clarity,
    MaterialField::Preset,
    // Image-texture fields (only meaningful when recipe = image; hinted no-ops else).
    MaterialField::Texture,
    MaterialField::Subcell,
    MaterialField::Fit,
    MaterialField::SpanAlongMin,
    MaterialField::SpanAlongSize,
    MaterialField::SpanUpTop,
    MaterialField::SpanUpSize,
    // (final slot is padding skipped by `next`.)
    MaterialField::Fg,
];

impl MaterialField {
    /// Step to the next field in [`MATERIAL_FIELDS`], wrapping. Skips the trailing
    /// duplicate `Fg` padding entry so the cycle visits each logical field once.
    pub fn next(self) -> MaterialField {
        // Search only the unique prefix (all but the final padding entry).
        let unique = &MATERIAL_FIELDS[..MATERIAL_FIELDS.len() - 1];
        let i = unique.iter().position(|&f| f == self).unwrap_or(0);
        unique[(i + 1) % unique.len()]
    }

    /// Short panel label.
    pub fn label(self) -> &'static str {
        match self {
            MaterialField::Fg => "fg",
            MaterialField::Bg => "bg",
            MaterialField::Glyph => "glyph",
            MaterialField::Recipe => "recipe",
            MaterialField::RecipeP0 => "recipe.p0",
            MaterialField::RecipeP1 => "recipe.p1",
            MaterialField::RecipeP2 => "recipe.p2",
            MaterialField::AnimKind => "anim",
            MaterialField::AnimSpeed => "anim.speed",
            MaterialField::AnimAmplitude => "anim.amp",
            MaterialField::Rank => "rank",
            MaterialField::Surface => "surface",
            MaterialField::Reflective => "reflective",
            MaterialField::Clarity => "clarity",
            MaterialField::Preset => "preset",
            MaterialField::Texture => "texture",
            MaterialField::Subcell => "subcell",
            MaterialField::Fit => "fit",
            MaterialField::SpanAlongMin => "span.along_min",
            MaterialField::SpanAlongSize => "span.along_size",
            MaterialField::SpanUpTop => "span.up_top",
            MaterialField::SpanUpSize => "span.up_size",
        }
    }

    /// True when `c` (enum-cycle) drives this field instead of `+`/`-`.
    ///
    /// NOTE: Glyph and Recipe are deliberately NOT enum fields — they "cycle a
    /// candidate set" via `+`/`-` (their long-standing behaviour), so authors keep
    /// one consistent stepping key for the appearance basics. The `c` key is for
    /// the fields that have no natural numeric step: animation KIND, surface
    /// binding, the reflective toggle, and the catalog preset.
    pub fn is_enum(self) -> bool {
        matches!(
            self,
            MaterialField::AnimKind
                | MaterialField::Surface
                | MaterialField::Reflective
                | MaterialField::Preset
                | MaterialField::Texture
                | MaterialField::Subcell
                | MaterialField::Fit
        )
    }
}

/// Per-recipe param NAMES for the three `RecipeP*` slots (or `None` for a slot the
/// variant doesn't use). Used for the panel readout AND to know which slots are
/// editable for the current recipe.
pub fn recipe_param_names(r: Option<MaterialRecipe>) -> [Option<&'static str>; 3] {
    match r {
        None => [None, None, None],
        Some(MaterialRecipe::Brick { .. }) => [Some("aspect"), Some("mortar_w"), Some("grain")],
        Some(MaterialRecipe::Circuit { .. }) => [Some("scale"), Some("trace_w"), None],
        Some(MaterialRecipe::Grain { .. }) => [Some("scale"), Some("speck"), Some("tile_eps")],
        Some(MaterialRecipe::Sky { .. }) => [Some("scale"), Some("density"), None],
        Some(MaterialRecipe::Hologram { .. }) => [Some("line_freq"), Some("scan_w"), None],
        Some(MaterialRecipe::CodeWall { .. }) => [Some("panel"), None, None],
        // `NoteWall` is NOT AUTHORABLE: it is synthesized at level load from an
        // existing `[note]`'s pos/facing (there is no `.lvl` recipe syntax for it, and
        // it is never present in a parsed Level), so the editor never offers it and
        // exposes no params for it. Listed only for exhaustiveness.
        Some(MaterialRecipe::NoteWall { .. }) => [None, None, None],
        // The five pattern recipes. `Scanline` has FOUR
        // knobs and the inspector has three slots, so `axis` (0 h / 1 v / 2 woven)
        // is authored in the `.lvl` text and preserved untouched by the editor —
        // stepping freq/duty/warp never rewrites it.
        Some(MaterialRecipe::Scanline { .. }) => [Some("freq"), Some("duty"), Some("warp")],
        Some(MaterialRecipe::Truchet { .. }) => [Some("seed"), Some("density"), Some("scale")],
        Some(MaterialRecipe::Fractal { .. }) => [Some("arms"), Some("octave_warp"), Some("twist")],
        Some(MaterialRecipe::Substrate { .. }) => [Some("scale"), Some("held"), Some("released")],
        // The voice's CHANNEL is not a param — it is what separates the two cycle
        // entries (`voice_hold` / `voice_release`), so only the density is a slot.
        Some(MaterialRecipe::Voice { .. }) => [Some("scale"), None, None],
        // Image-texture wall: the only param is the integer texture id (the index
        // into the level's decoded `[texture]` table). Authoring the binding by
        // NAME lives in the `[material] texture = NAME` key, not these float slots,
        // so the editor exposes just the id for completeness.
        Some(MaterialRecipe::Image { .. }) => [Some("tex_id"), None, None],
        // The bay×floor window grid has SEVEN knobs against three slots — the same
        // shape as `Scanline`'s fourth knob (`axis`, above). `style` is the variant
        // (five separate `RECIPES` cycle entries, `facade_office`/… — see
        // `recipe_label`), and `seed`/`ground_band`/`sign_row` are authored in the
        // `.lvl` text and preserved untouched by the editor. The three slots expose
        // the knobs an author actually tunes by eye: the bay/floor rhythm and the
        // lit fraction.
        Some(MaterialRecipe::Facade { .. }) => [Some("bay"), Some("floor"), Some("lit")],
    }
}

/// Read the recipe's param value at slot `i` (0..3), or `None` if the variant has
/// no param there.
pub fn recipe_param_get(r: Option<MaterialRecipe>, i: usize) -> Option<f32> {
    let vals: [Option<f32>; 3] = match r {
        None => [None, None, None],
        Some(MaterialRecipe::Brick {
            aspect,
            mortar_w,
            grain,
        }) => [Some(aspect), Some(mortar_w), Some(grain)],
        Some(MaterialRecipe::Circuit { scale, trace_w }) => [Some(scale), Some(trace_w), None],
        Some(MaterialRecipe::Grain {
            scale,
            speck,
            tile_eps,
        }) => [Some(scale), Some(speck), Some(tile_eps)],
        Some(MaterialRecipe::Sky { scale, density }) => [Some(scale), Some(density), None],
        Some(MaterialRecipe::Hologram { line_freq, scan_w }) => {
            [Some(line_freq), Some(scan_w), None]
        }
        Some(MaterialRecipe::CodeWall { panel }) => [Some(panel as f32), None, None],
        // Synthesized-only (see `recipe_param_names`): no editable params.
        Some(MaterialRecipe::NoteWall { .. }) => [None, None, None],
        // `axis` is deliberately absent — see `recipe_param_names`.
        Some(MaterialRecipe::Scanline {
            freq, duty, warp, ..
        }) => [Some(freq), Some(duty), Some(warp)],
        Some(MaterialRecipe::Truchet {
            seed,
            density,
            scale,
        }) => [Some(seed as f32), Some(density), Some(scale)],
        Some(MaterialRecipe::Fractal {
            arms,
            octave_warp,
            twist,
        }) => [Some(f32::from(arms)), Some(octave_warp), Some(twist)],
        Some(MaterialRecipe::Substrate {
            scale,
            held,
            released,
        }) => [Some(scale), Some(held), Some(released)],
        Some(MaterialRecipe::Voice { scale, .. }) => [Some(scale), None, None],
        Some(MaterialRecipe::Image { tex_id, .. }) => [Some(tex_id as f32), None, None],
        Some(MaterialRecipe::Facade {
            bay, floor_h, lit, ..
        }) => [Some(bay), Some(floor_h), Some(lit)],
    };
    vals.get(i).copied().flatten()
}

/// Return a recipe with its slot-`i` param nudged by `delta * step` (clamped to a
/// sensible per-param range). A slot with no param for this variant is unchanged.
/// `panel` (CodeWall) steps by whole integers, never below 0.
pub fn recipe_param_step(r: MaterialRecipe, i: usize, delta: i32) -> MaterialRecipe {
    let d = delta as f32;
    match r {
        MaterialRecipe::Brick {
            mut aspect,
            mut mortar_w,
            mut grain,
        } => {
            match i {
                0 => aspect = (aspect + d * 0.25).clamp(0.25, 8.0),
                1 => mortar_w = (mortar_w + d * 0.02).clamp(0.0, 0.9),
                _ => grain = (grain + d * 0.05).clamp(0.0, 2.0),
            }
            MaterialRecipe::Brick {
                aspect,
                mortar_w,
                grain,
            }
        }
        MaterialRecipe::Circuit {
            mut scale,
            mut trace_w,
        } => {
            match i {
                0 => scale = (scale + d * 0.2).clamp(0.1, 16.0),
                1 => trace_w = (trace_w + d * 0.02).clamp(0.0, 0.9),
                _ => {}
            }
            MaterialRecipe::Circuit { scale, trace_w }
        }
        MaterialRecipe::Grain {
            mut scale,
            mut speck,
            mut tile_eps,
        } => {
            match i {
                0 => scale = (scale + d * 0.25).clamp(0.1, 32.0),
                1 => speck = (speck + d * 0.05).clamp(0.0, 1.0),
                _ => tile_eps = (tile_eps + d * 0.01).clamp(0.0, 0.5),
            }
            MaterialRecipe::Grain {
                scale,
                speck,
                tile_eps,
            }
        }
        MaterialRecipe::Sky {
            mut scale,
            mut density,
        } => {
            match i {
                0 => scale = (scale + d * 0.5).clamp(0.5, 32.0),
                1 => density = (density + d * 0.02).clamp(0.0, 1.0),
                _ => {}
            }
            MaterialRecipe::Sky { scale, density }
        }
        MaterialRecipe::Hologram {
            mut line_freq,
            mut scan_w,
        } => {
            match i {
                0 => line_freq = (line_freq + d * 0.5).clamp(0.5, 32.0),
                1 => scan_w = (scan_w + d * 0.05).clamp(0.0, 1.0),
                _ => {}
            }
            MaterialRecipe::Hologram { line_freq, scan_w }
        }
        MaterialRecipe::CodeWall { panel } => {
            let next = panel as i64 + delta as i64;
            MaterialRecipe::CodeWall {
                panel: next.max(0) as u32,
            }
        }
        // Synthesized-only (see `recipe_param_names`): nothing to step.
        MaterialRecipe::NoteWall { note, face } => MaterialRecipe::NoteWall { note, face },
        // ── the five pattern recipes ─────────────────────────────────────────
        MaterialRecipe::Scanline {
            mut freq,
            mut duty,
            mut warp,
            axis,
        } => {
            match i {
                0 => freq = (freq + d * 0.5).clamp(0.25, 32.0),
                1 => duty = (duty + d * 0.05).clamp(0.0, 1.0),
                _ => warp = (warp + d * 0.05).clamp(0.0, 1.0),
            }
            // `axis` rides through untouched: the inspector has three slots and
            // this recipe has four knobs (see `recipe_param_names`).
            MaterialRecipe::Scanline {
                freq,
                duty,
                warp,
                axis,
            }
        }
        MaterialRecipe::Truchet {
            seed,
            mut density,
            mut scale,
        } => {
            // `seed` is an INTEGER pattern selector: it steps by whole numbers and
            // may be negative (any i32 is a valid deal).
            let mut seed = seed;
            match i {
                0 => seed = seed.saturating_add(delta),
                1 => density = (density + d * 0.05).clamp(0.0, 1.0),
                _ => scale = (scale + d * 0.5).clamp(0.25, 32.0),
            }
            MaterialRecipe::Truchet {
                seed,
                density,
                scale,
            }
        }
        MaterialRecipe::Fractal {
            arms,
            mut octave_warp,
            mut twist,
        } => {
            let mut arms = arms;
            match i {
                0 => arms = (i32::from(arms) + delta).clamp(0, 32) as u8,
                1 => octave_warp = (octave_warp + d * 0.05).clamp(0.0, 1.0),
                _ => twist = (twist + d * 0.1).clamp(-4.0, 4.0),
            }
            MaterialRecipe::Fractal {
                arms,
                octave_warp,
                twist,
            }
        }
        MaterialRecipe::Substrate {
            mut scale,
            mut held,
            mut released,
        } => {
            match i {
                0 => scale = (scale + d * 0.5).clamp(0.25, 32.0),
                1 => held = (held + d * 0.05).clamp(0.0, 1.0),
                _ => released = (released + d * 0.05).clamp(0.0, 1.0),
            }
            MaterialRecipe::Substrate {
                scale,
                held,
                released,
            }
        }
        // Only the density is a slot; the CHANNEL is the variant (see
        // `recipe_param_names`), stepped by the Recipe field, not by `+`/`-` here.
        MaterialRecipe::Voice { mut scale, mode } => {
            if i == 0 {
                scale = (scale + d * 0.5).clamp(0.25, 32.0);
            }
            MaterialRecipe::Voice { scale, mode }
        }
        MaterialRecipe::Image {
            tex_id,
            span,
            subcell,
            fit,
            tile,
            relief,
        } => {
            // Step the integer texture id (never below 0). Only slot 0 is meaningful;
            // the world-anchored span + sub-cell mode + fit are preserved untouched.
            let next = tex_id as i64 + delta as i64;
            MaterialRecipe::Image {
                tex_id: next.max(0) as u32,
                span,
                subcell,
                fit,
                tile,
                relief,
            }
        }
        MaterialRecipe::Facade {
            style,
            mut bay,
            mut floor_h,
            mut lit,
            seed,
            ground_band,
            sign_row,
        } => {
            match i {
                0 => bay = (bay + d * 0.5).clamp(0.5, 24.0),
                1 => floor_h = (floor_h + d * 0.5).clamp(0.5, 24.0),
                _ => lit = (lit + d * 0.05).clamp(0.0, 1.0),
            }
            MaterialRecipe::Facade {
                style,
                bay,
                floor_h,
                lit,
                seed,
                ground_band,
                sign_row,
            }
        }
    }
}

/// The list a `preset` `c` press walks: `None` ("no preset") followed by
/// `re_render`'s OWN material catalog (`ascii_shader::ids::CATALOG_NAMES`), then
/// every off-catalog preset this session has seen, then `current`.
///
/// The editor used to hardcode 7 entries. Shipped levels author 21 distinct
/// presets across 116 instances, so 66 of them — `void`, `decay`, `watcher`,
/// `gold`, `frost`, `ice`, … including all four "voice" materials — were
/// destroyed by a single `c`, and none of them was reachable from the editor at
/// all. One of the 7 hardcoded names, `concrete`, exists nowhere in the renderer:
/// choosing it wrote a `.lvl` naming a material that silently falls back to the
/// surface base. Enumerating the catalog is what makes that drift impossible;
/// remembering the rest is what makes `facet` / `relief_vol` (real demo materials
/// the catalog carries no NAME entry for) survive. See [`crate::vocab`].
pub fn preset_options(state: &EditorState, current: Option<&str>) -> Vec<Option<String>> {
    crate::vocab::options_at(
        &crate::vocab::preset_catalog(),
        &state.vocab.preset,
        &current.map(ToString::to_string),
    )
    .0
}

/// Surface-binding slot names the `Surface` field cycles. Index 0 is `none`
/// (unbound). Matches `re_level`'s `surface =` keywords / `SurfaceMaterials`.
pub const SURFACES: [&str; 6] = [
    "none",
    "wall_solid",
    "wall_trace",
    "wall_matrix",
    "floor",
    "ceiling",
];

/// A small, readable colour palette the Fg/Bg fields step through. Saturated,
/// distinct hues so cycling is visibly different each step (`0x00RRGGBB`).
pub const PALETTE: [u32; 12] = [
    0x00_FFFFFF, // white
    0x00_B0B0B0, // grey
    0x00_303840, // near-black
    0x00_FF4040, // red
    0x00_FF9020, // orange
    0x00_FFE040, // yellow
    0x00_40FF60, // green
    0x00_20FF80, // neon green
    0x00_40D0FF, // cyan
    0x00_4060FF, // blue
    0x00_C080FF, // violet
    0x00_FF44AA, // magenta
];

/// Candidate fixed glyphs the Glyph field cycles. The first entry is "no fixed
/// glyph" (`None` → the recipe/surface chooses), then a spread of distinct marks.
///
/// NOTE: `#` is deliberately EXCLUDED — `re_level`'s `[material]` parser strips a
/// ` #…` comment unless what follows is a 6-hex colour, so `glyph = #` would be
/// eaten on reload (the value becomes empty → `None`). The candidate set is
/// limited to comment-safe glyphs so every Glyph edit round-trips.
pub const GLYPHS: [Option<char>; 8] = [
    None,
    Some('+'),
    Some('*'),
    Some('%'),
    Some('='),
    Some('.'),
    Some('o'),
    Some('@'),
];

/// The recipe variants the Recipe field cycles, with sensible defaults (matching
/// `re_level::parse_recipe` / `re_voximport::recipe_to_text`). The first entry is
/// `None` ("no recipe" — a flat fg/bg/glyph material).
pub const RECIPES: [Option<MaterialRecipe>; 19] = [
    None,
    Some(MaterialRecipe::Brick {
        aspect: 2.0,
        mortar_w: 0.16,
        grain: 0.6,
    }),
    Some(MaterialRecipe::Circuit {
        scale: 1.6,
        trace_w: 0.12,
    }),
    Some(MaterialRecipe::Grain {
        scale: 3.0,
        speck: 0.30,
        tile_eps: 0.06,
    }),
    Some(MaterialRecipe::Sky {
        scale: 6.0,
        density: 0.10,
    }),
    Some(MaterialRecipe::Hologram {
        line_freq: 6.0,
        scan_w: 0.5,
    }),
    Some(MaterialRecipe::CodeWall { panel: 0 }),
    // The five pattern recipes. Their
    // defaults are `re_level::parse_recipe`'s, which are in turn the matching
    // catalog preset's knobs — so cycling to `scanline` gives the `monitor`
    // raster, `truchet` the `conduit` lattice, and so on.
    Some(MaterialRecipe::Scanline {
        freq: 9.0,
        duty: 0.55,
        warp: 0.05,
        axis: 0,
    }),
    Some(MaterialRecipe::Truchet {
        seed: 0,
        density: 0.5,
        scale: 4.0,
    }),
    Some(MaterialRecipe::Fractal {
        arms: 0,
        octave_warp: 0.30,
        twist: 0.0,
    }),
    Some(MaterialRecipe::Substrate {
        scale: 3.0,
        held: 0.55,
        released: 0.30,
    }),
    // The two VOICE channels are separate cycle entries on purpose: they say
    // opposite things (one fills, one erodes), so they must both be reachable and
    // each must keep its own tuned instance in the session vocab.
    Some(MaterialRecipe::Voice {
        scale: 3.0,
        mode: VoiceMode::Hold,
    }),
    Some(MaterialRecipe::Voice {
        scale: 3.0,
        mode: VoiceMode::Release,
    }),
    // The five FACADE styles are separate cycle entries, exactly like the two
    // VOICE channels above: `recipe_label` gives each its own name, so each keeps
    // its own tuned instance in the session vocab and a `c`-lap never collapses
    // one style into another. Defaults match `re_level::parse_recipe`'s bare
    // `facade style=<name>` line for that style. Placed BEFORE Image so Image
    // stays the final RECIPES entry — several tests (`recipe_field_cycles_variants`,
    // `io::create_image_material_bound_to_texture_save_reload_preserves_every_field`)
    // reach Image by wrapping backward from `None`, which lands on the last entry.
    Some(MaterialRecipe::Facade {
        style: re_world::FacadeStyle::Office,
        bay: 3.0,
        floor_h: 4.0,
        lit: 0.35,
        seed: 0,
        ground_band: false,
        sign_row: false,
    }),
    Some(MaterialRecipe::Facade {
        style: re_world::FacadeStyle::Residential,
        bay: 4.0,
        floor_h: 6.0,
        lit: 0.4,
        seed: 0,
        ground_band: false,
        sign_row: false,
    }),
    Some(MaterialRecipe::Facade {
        style: re_world::FacadeStyle::Industrial,
        bay: 2.0,
        floor_h: 2.0,
        lit: 0.25,
        seed: 0,
        ground_band: true,
        sign_row: false,
    }),
    Some(MaterialRecipe::Facade {
        style: re_world::FacadeStyle::ServerRack,
        bay: 6.0,
        floor_h: 10.0,
        lit: 0.5,
        seed: 0,
        ground_band: false,
        sign_row: false,
    }),
    Some(MaterialRecipe::Facade {
        style: re_world::FacadeStyle::Lockers,
        bay: 3.0,
        floor_h: 3.0,
        lit: 0.3,
        seed: 0,
        ground_band: false,
        sign_row: false,
    }),
    // Image-texture wall. The `tex_id` here is a build-time index placeholder (the
    // binding the renderer actually uses comes from the material's `texture = NAME`
    // key, re-resolved on load); `span`/`subcell`/`fit` are authored via the
    // dedicated Texture/Subcell/Fit/Span* fields on the SAME material. MUST stay the
    // final entry (see the FACADE comment above).
    Some(MaterialRecipe::Image {
        tex_id: 0,
        span: None,
        subcell: SubcellMode::Off,
        fit: Fit::Contain,
        tile: re_world::IMAGE_TILE_NONE,
        relief: None,
    }),
];

/// Short label for a recipe value (panel readout).
pub fn recipe_label(r: Option<MaterialRecipe>) -> &'static str {
    match r {
        None => "none",
        Some(MaterialRecipe::Brick { .. }) => "brick",
        Some(MaterialRecipe::Circuit { .. }) => "circuit",
        Some(MaterialRecipe::Grain { .. }) => "grain",
        Some(MaterialRecipe::Sky { .. }) => "sky",
        Some(MaterialRecipe::Hologram { .. }) => "hologram",
        Some(MaterialRecipe::CodeWall { .. }) => "codewall",
        // Synthesized-only, but label it so a material dump is never blank.
        Some(MaterialRecipe::NoteWall { .. }) => "notewall",
        Some(MaterialRecipe::Scanline { .. }) => "scanline",
        Some(MaterialRecipe::Truchet { .. }) => "truchet",
        Some(MaterialRecipe::Fractal { .. }) => "fractal",
        Some(MaterialRecipe::Substrate { .. }) => "substrate",
        // The two VOICE channels get DISTINCT labels, and that is load-bearing:
        // `recipe_variant_eq` is label equality, so distinct labels give the hold
        // and release walls separate cycle entries AND separate tuned-instance
        // slots in the session vocab. One shared "voice" label would make a level
        // that authors both lose one of them on a cycle lap.
        Some(MaterialRecipe::Voice {
            mode: VoiceMode::Hold,
            ..
        }) => "voice_hold",
        Some(MaterialRecipe::Voice {
            mode: VoiceMode::Release,
            ..
        }) => "voice_release",
        Some(MaterialRecipe::Image { .. }) => "image",
        // The five FACADE styles get DISTINCT labels for the same reason the two
        // VOICE channels do (see the comment above): `recipe_variant_eq` is label
        // equality, so distinct labels keep all five reachable and separately
        // tuned rather than collapsing onto one shared "facade" entry.
        Some(MaterialRecipe::Facade {
            style: re_world::FacadeStyle::Office,
            ..
        }) => "facade_office",
        Some(MaterialRecipe::Facade {
            style: re_world::FacadeStyle::Residential,
            ..
        }) => "facade_residential",
        Some(MaterialRecipe::Facade {
            style: re_world::FacadeStyle::Industrial,
            ..
        }) => "facade_industrial",
        Some(MaterialRecipe::Facade {
            style: re_world::FacadeStyle::ServerRack,
            ..
        }) => "facade_server_rack",
        Some(MaterialRecipe::Facade {
            style: re_world::FacadeStyle::Lockers,
            ..
        }) => "facade_lockers",
    }
}

/// True when two recipe values are the SAME variant (ignoring params), so the
/// cycle can find the current entry in [`RECIPES`] regardless of edited params.
fn recipe_variant_eq(a: Option<MaterialRecipe>, b: Option<MaterialRecipe>) -> bool {
    recipe_label(a) == recipe_label(b)
}

/// The sub-cell detail modes the `Subcell` field cycles through, paired with the
/// keyword the `re_level` parser / `re_voximport` writer use. Matches the
/// `[material] subcell = off|half|quad|sextant|braille|ascii|contour|lines`
/// vocabulary. Ordered along the sub-cell resolution LADDER (off → half → quad →
/// sextant → braille) before the ASCII-family edge samplers, so cycling the field
/// walks detail levels in a meaningful order rather than alphabetically.
pub const SUBCELL_MODES: [(SubcellMode, &str); 8] = [
    (SubcellMode::Off, "off"),
    (SubcellMode::Half, "half"),
    (SubcellMode::Quad, "quad"),
    (SubcellMode::Sextant, "sextant"),
    (SubcellMode::Braille, "braille"),
    (SubcellMode::Ascii, "ascii"),
    (SubcellMode::Contour, "contour"),
    (SubcellMode::Lines, "lines"),
];

/// A readable label for a [`SubcellMode`] (`off`/`quad`/…), matching the `.lvl` key.
pub fn subcell_label(m: SubcellMode) -> &'static str {
    SUBCELL_MODES
        .iter()
        .find(|&&(k, _)| k == m)
        .map(|&(_, l)| l)
        .unwrap_or("off")
}

/// Step a [`SubcellMode`] to the next/prev entry in [`SUBCELL_MODES`], wrapping.
fn step_subcell(m: SubcellMode, delta: i32) -> SubcellMode {
    let n = SUBCELL_MODES.len() as i32;
    let i = SUBCELL_MODES.iter().position(|&(k, _)| k == m).unwrap_or(0) as i32;
    SUBCELL_MODES[(i + delta).rem_euclid(n) as usize].0
}

/// The fit modes the `Fit` field cycles through, paired with the `.lvl` keyword
/// (`fit = stretch|contain|cover`).
pub const FIT_MODES: [(Fit, &str); 3] = [
    (Fit::Stretch, "stretch"),
    (Fit::Contain, "contain"),
    (Fit::Cover, "cover"),
];

/// A readable label for a [`Fit`] (`stretch`/`contain`/`cover`), matching the key.
pub fn fit_label(f: Fit) -> &'static str {
    FIT_MODES
        .iter()
        .find(|&&(k, _)| k == f)
        .map(|&(_, l)| l)
        .unwrap_or("contain")
}

/// Step a [`Fit`] to the next/prev entry in [`FIT_MODES`], wrapping.
fn step_fit(f: Fit, delta: i32) -> Fit {
    let n = FIT_MODES.len() as i32;
    let i = FIT_MODES.iter().position(|&(k, _)| k == f).unwrap_or(0) as i32;
    FIT_MODES[(i + delta).rem_euclid(n) as usize].0
}

/// The default [`ImageSpan`] a freshly-authored span starts from (a 1×1 world box
/// anchored at the origin). Editing any span component first seeds this when the
/// material had no span yet, then steps the chosen component.
fn default_image_span() -> ImageSpan {
    ImageSpan {
        along_min: 0.0,
        along_size: 1.0,
        up_top: 1.0,
        up_size: 1.0,
    }
}

/// Step the selected-material index by `delta` and ENTER material mode. Entering
/// from off (`material_mode == false`) selects the first material (index 0)
/// without stepping, so the first `m` press shows material 0. Subsequent `m`/`M`
/// presses step. A level with no materials reports the empty state (no panic).
pub fn cycle_material(state: &mut EditorState, delta: i32) {
    let n = state.level.materials.len();
    if n == 0 {
        state.material_mode = true;
        state.status = "no materials — press e to create one".into();
        return;
    }
    if !state.material_mode {
        // First entry: just turn the mode on and keep the current selection.
        state.material_mode = true;
        state.material_sel = state.material_sel.min(n - 1);
    } else {
        let i = state.material_sel as i32;
        state.material_sel = (i + delta).rem_euclid(n as i32) as usize;
    }
    state.status = material_summary(state);
}

/// Create a new material (unique name, default look) and select it for editing.
/// Marks dirty. Leaves the editor in material mode.
pub fn new_material(state: &mut EditorState) {
    state.schedule_preview_rebuild(); // material defs feed map.set_materials
    let idx = state.level.push_material();
    state.material_sel = idx;
    state.material_mode = true;
    state.mark_dirty();
    state.status = format!("new material '{}'", state.level.materials[idx].name);
}

/// Cycle which material-def field `+`/`-` edit (`b`). No-op label when there is
/// no material to edit.
pub fn cycle_field(state: &mut EditorState) {
    state.material_mode = true;
    state.material_field = state.material_field.next();
    state.status = format!(
        "field: {} = {}",
        state.material_field.label(),
        field_value(state)
    );
}

/// Adjust the focused material-def field by `delta` (±1) — the `+`/`-` path. Fg/Bg
/// step [`PALETTE`]; Glyph cycles [`GLYPHS`]; Recipe cycles [`RECIPES`]; the
/// `RecipeP*` slots step the active recipe's params; anim speed/amplitude, rank,
/// and clarity step their scalars. Enum fields (AnimKind/Surface/Reflective/Preset)
/// hint to use `c`. A no-op (with a hint) when the level has no materials. Marks
/// dirty on a real change.
pub fn adjust_field(state: &mut EditorState, delta: i32) {
    let Some(idx) = sel_index(state) else {
        state.status = "no material — press e to create".into();
        return;
    };
    state.schedule_preview_rebuild(); // material defs feed map.set_materials

    let field = state.material_field;
    // Enum fields are driven by `c`, not `+`/`-`.
    if field.is_enum() {
        state.status = format!("'{}' is an enum: use c", field.label());
        return;
    }
    // The glyph step REMEMBERS the value it steps off (`state.vocab`), so it is
    // resolved before the material itself is borrowed mutably.
    // The recipe step REMEMBERS the tuned instance it steps off (`state.vocab`),
    // so it too is resolved before the material is borrowed mutably.
    if field == MaterialField::Recipe {
        let next = step_recipe(state, state.level.materials[idx].recipe, delta);
        state.level.materials[idx].recipe = next;
        state.mark_dirty();
        state.status = format!("recipe {}", recipe_label(next));
        return;
    }
    if field == MaterialField::Glyph {
        let next = step_glyph(state, state.level.materials[idx].glyph, delta);
        state.level.materials[idx].glyph = next;
        state.mark_dirty();
        state.status = match next.and_then(char::from_u32) {
            Some(c) => format!("glyph {c}"),
            None => "glyph none".to_string(),
        };
        return;
    }
    let m = &mut state.level.materials[idx];
    let msg = match field {
        MaterialField::Fg => {
            m.fg = step_palette(m.fg, delta);
            format!("fg #{:06X}", m.fg & 0xFF_FFFF)
        }
        MaterialField::Bg => {
            m.bg = step_palette(m.bg, delta);
            format!("bg #{:06X}", m.bg & 0xFF_FFFF)
        }
        MaterialField::Glyph => unreachable!("glyph handled above"),
        MaterialField::Recipe => unreachable!("recipe handled above"),
        MaterialField::RecipeP0 | MaterialField::RecipeP1 | MaterialField::RecipeP2 => {
            let slot = match field {
                MaterialField::RecipeP0 => 0,
                MaterialField::RecipeP1 => 1,
                _ => 2,
            };
            match m.recipe {
                Some(r) if recipe_param_names(Some(r))[slot].is_some() => {
                    m.recipe = Some(recipe_param_step(r, slot, delta));
                    let name = recipe_param_names(m.recipe)[slot].unwrap_or("p");
                    let val = recipe_param_get(m.recipe, slot).unwrap_or(0.0);
                    format!("{name} {val:.3}")
                }
                _ => {
                    state.status = "recipe has no param here".into();
                    return;
                }
            }
        }
        MaterialField::AnimSpeed => {
            m.anim.speed = (m.anim.speed + delta as f32 * 0.1).max(0.0);
            format!("anim.speed {:.2}", m.anim.speed)
        }
        MaterialField::AnimAmplitude => {
            m.anim.amplitude = (m.anim.amplitude + delta as f32 * 0.05).clamp(0.0, 4.0);
            format!("anim.amp {:.2}", m.anim.amplitude)
        }
        MaterialField::Rank => {
            let next = m.rank as i32 + delta;
            m.rank = next.clamp(0, 127) as u8;
            format!("rank {}", m.rank)
        }
        MaterialField::Clarity => {
            m.clarity = (m.clarity + delta as f32 * 0.05).clamp(0.0, 1.0);
            format!("clarity {:.2}", m.clarity)
        }
        // Image world-anchored span components. Editing ANY of them turns the
        // `Option<ImageSpan>` on (seeded from a sane 1×1 default) so the author can
        // start from nothing; the two SIZE axes clamp `> 0` (a non-positive size
        // falls back to per-cell tiling, per `ImageSpan`'s contract).
        MaterialField::SpanAlongMin
        | MaterialField::SpanAlongSize
        | MaterialField::SpanUpTop
        | MaterialField::SpanUpSize => {
            let mut span = m.image_span.unwrap_or_else(default_image_span);
            let d = delta as f32 * 0.5;
            match field {
                MaterialField::SpanAlongMin => span.along_min += d,
                MaterialField::SpanAlongSize => span.along_size = (span.along_size + d).max(0.5),
                MaterialField::SpanUpTop => span.up_top += d,
                _ => span.up_size = (span.up_size + d).max(0.5),
            }
            m.image_span = Some(span);
            format!(
                "span {},{} {},{}",
                span.along_min, span.along_size, span.up_top, span.up_size
            )
        }
        // Enum fields handled above; anything else is unreachable.
        _ => unreachable!("enum field guarded by is_enum"),
    };
    state.mark_dirty();
    state.status = msg;
}

/// Cycle the focused material ENUM field (`c`): Glyph candidates, Recipe variants,
/// AnimKind, Surface binding, Reflective toggle, or Preset name. A no-op (with a
/// hint) when the focused field is numeric (use `+`/`-`) or no material exists.
pub fn cycle_enum(state: &mut EditorState, delta: i32) {
    let Some(idx) = sel_index(state) else {
        state.status = "no material — press e to create".into();
        return;
    };
    state.schedule_preview_rebuild(); // material defs feed map.set_materials

    let field = state.material_field;
    if !field.is_enum() {
        state.status = format!("'{}' is numeric: use +/-", field.label());
        return;
    }
    // Surface binding mutates `level.surface_materials` (keyed by the material's
    // NAME), so resolve the name up front before borrowing the material mutably.
    if field == MaterialField::Surface {
        let name = state.level.materials[idx].name.clone();
        let cur = current_surface(state, &name);
        let i = SURFACES.iter().position(|&s| s == cur).unwrap_or(0) as i32;
        let next = SURFACES[(i + delta).rem_euclid(SURFACES.len() as i32) as usize];
        set_surface_binding(state, &name, next);
        state.mark_dirty();
        state.status = format!("surface {next}");
        return;
    }
    // The preset cycle REMEMBERS an off-catalog preset (`state.vocab`), so it is
    // resolved before the material is borrowed mutably.
    if field == MaterialField::Preset {
        let cur = state.level.materials[idx].preset.clone();
        let canon = crate::vocab::preset_catalog();
        crate::vocab::remember(&canon, &mut state.vocab.preset, &cur);
        let next = crate::vocab::step(&canon, &state.vocab.preset, &cur, delta);
        state.level.materials[idx].preset.clone_from(&next);
        state.mark_dirty();
        state.status = format!("preset {}", next.as_deref().unwrap_or("none"));
        return;
    }
    // The texture binding cycles the level's `[texture]` NAMES (plus `(none)`), so
    // resolve that snapshot up front before borrowing the material mutably.
    if field == MaterialField::Texture {
        let names: Vec<String> = state
            .level
            .textures
            .iter()
            .map(|t| t.name.clone())
            .collect();
        let m = &mut state.level.materials[idx];
        // Candidate ring: index 0 = `(none)`, then each texture name in order.
        let cur = m.texture.as_deref();
        let cur_i = match cur {
            None => 0,
            Some(name) => names.iter().position(|n| n == name).map_or(0, |p| p + 1),
        } as i32;
        let ring_len = names.len() as i32 + 1;
        let next_i = (cur_i + delta).rem_euclid(ring_len);
        m.texture = if next_i == 0 {
            None
        } else {
            Some(names[(next_i - 1) as usize].clone())
        };
        let shown = m.texture.clone().unwrap_or_else(|| "(none)".to_string());
        state.mark_dirty();
        state.status = if names.is_empty() {
            "no textures — add one in Level Settings".into()
        } else {
            format!("texture {shown}")
        };
        return;
    }
    let m = &mut state.level.materials[idx];
    let msg = match field {
        MaterialField::Subcell => {
            m.image_subcell = step_subcell(m.image_subcell, delta);
            format!("subcell {}", subcell_label(m.image_subcell))
        }
        MaterialField::Fit => {
            m.image_fit = step_fit(m.image_fit, delta);
            format!("fit {}", fit_label(m.image_fit))
        }
        MaterialField::AnimKind => {
            m.anim.kind = step_anim_kind(m.anim.kind, delta);
            // A non-None anim with a zero speed never animates; seed a sane speed.
            if m.anim.kind != MaterialAnimKind::None && m.anim.speed <= 0.0 {
                m.anim.speed = 1.0;
            }
            format!("anim {}", anim_kind_label(m.anim.kind))
        }
        MaterialField::Reflective => {
            m.reflective = !m.reflective;
            format!("reflective {}", m.reflective)
        }
        // `Preset` is handled before the mutable borrow (it writes `state.vocab`).
        MaterialField::Preset => unreachable!("preset handled above"),
        _ => unreachable!("non-enum field guarded above"),
    };
    state.mark_dirty();
    state.status = msg;
}

/// Step a [`MaterialAnimKind`] through None→Pulse→Shift→Flicker (wrapping).
fn step_anim_kind(k: MaterialAnimKind, delta: i32) -> MaterialAnimKind {
    const KINDS: [MaterialAnimKind; 4] = [
        MaterialAnimKind::None,
        MaterialAnimKind::Pulse,
        MaterialAnimKind::Shift,
        MaterialAnimKind::Flicker,
    ];
    let i = KINDS.iter().position(|&x| x == k).unwrap_or(0) as i32;
    KINDS[(i + delta).rem_euclid(KINDS.len() as i32) as usize]
}

/// Short label for a [`MaterialAnimKind`].
pub fn anim_kind_label(k: MaterialAnimKind) -> &'static str {
    match k {
        MaterialAnimKind::None => "none",
        MaterialAnimKind::Pulse => "pulse",
        MaterialAnimKind::Shift => "shift",
        MaterialAnimKind::Flicker => "flicker",
    }
}

/// Which surface (if any) the named material is currently bound to. Returns the
/// `SURFACES` keyword (`"none"` when unbound) — the inverse of
/// [`set_surface_binding`].
pub fn current_surface(state: &EditorState, name: &str) -> &'static str {
    let sm = &state.level.surface_materials;
    let hit = |b: &Option<String>| b.as_deref() == Some(name);
    if hit(&sm.wall_solid) {
        "wall_solid"
    } else if hit(&sm.wall_trace) {
        "wall_trace"
    } else if hit(&sm.wall_matrix) {
        "wall_matrix"
    } else if hit(&sm.floor) {
        "floor"
    } else if hit(&sm.ceiling) {
        "ceiling"
    } else {
        "none"
    }
}

/// Bind the named material to `surface` (a `SURFACES` keyword). FIRST clears the
/// material's name from EVERY slot (a material binds at most one surface), then
/// sets the chosen slot (unless `"none"`). Idempotent + validate-before-mutate:
/// an unknown keyword leaves the bindings untouched.
fn set_surface_binding(state: &mut EditorState, name: &str, surface: &str) {
    let sm = &mut state.level.surface_materials;
    // Clear this material's name from every slot it currently occupies.
    for slot in [
        &mut sm.wall_solid,
        &mut sm.wall_trace,
        &mut sm.wall_matrix,
        &mut sm.floor,
        &mut sm.ceiling,
    ] {
        if slot.as_deref() == Some(name) {
            *slot = None;
        }
    }
    let target = match surface {
        "wall_solid" => &mut sm.wall_solid,
        "wall_trace" => &mut sm.wall_trace,
        "wall_matrix" => &mut sm.wall_matrix,
        "floor" => &mut sm.floor,
        "ceiling" => &mut sm.ceiling,
        _ => return, // "none" / unknown ⇒ leave unbound.
    };
    *target = Some(name.to_string());
}

/// Paint the SELECTED material's 1-based palette id into the cursor cell's
/// `side_data`. If the cell is empty it is made solid (`WallSolid`) so the
/// material has a surface to render on; an already-solid cell keeps its kind. A
/// no-op (with a hint) when there is no material selected. Marks dirty.
pub fn paint_id(state: &mut EditorState) {
    let Some(idx) = sel_index(state) else {
        state.status = "no material — press e to create".into();
        return;
    };
    let id = (idx + 1) as u16; // 1-based palette id
    let (x, y, z) = (state.cursor.0, state.cursor.1, state.z);
    let kind = state.map.cell_at(x as i32, y as i32, z as i32);
    let kind = if kind == CellKind::Empty {
        CellKind::WallSolid
    } else {
        kind
    };
    state.map.set_cell(x, y, z, kind, id);
    state.mark_dirty();
    let name = state.level.materials[idx].name.clone();
    state.status = format!("painted material '{name}' (id {id}) @ {x},{y},{z}");
}

/// Clear the cursor cell's material id back to the default (`side_data = 0`),
/// keeping the cell's geometry kind. Marks dirty.
pub fn clear_id(state: &mut EditorState) {
    let (x, y, z) = (state.cursor.0, state.cursor.1, state.z);
    let kind = state.map.cell_at(x as i32, y as i32, z as i32);
    state.map.set_cell(x, y, z, kind, 0);
    state.mark_dirty();
    state.status = format!("cleared material @ {x},{y},{z}");
}

/// The valid selected-material index, or `None` when the level has no materials
/// (or the selection drifted out of range — clamped defensively).
pub fn sel_index(state: &EditorState) -> Option<usize> {
    let n = state.level.materials.len();
    if n == 0 {
        None
    } else {
        Some(state.material_sel.min(n - 1))
    }
}

/// A compact one-line summary of the selected material (panel / status line).
pub fn material_summary(state: &EditorState) -> String {
    match sel_index(state) {
        None => "no materials (e: new)".to_string(),
        Some(i) => {
            let m = &state.level.materials[i];
            let g = m
                .glyph
                .and_then(char::from_u32)
                .map(|c| c.to_string())
                .unwrap_or_else(|| "-".to_string());
            format!(
                "[{}/{}] {} fg#{:06X} g:{} {}",
                i + 1,
                state.level.materials.len(),
                m.name,
                m.fg & 0xFF_FFFF,
                g,
                recipe_label(m.recipe),
            )
        }
    }
}

/// The focused field's current value rendered for the panel.
pub fn field_value(state: &EditorState) -> String {
    field_value_of(state, state.material_field)
}

/// Render an ARBITRARY field of the selected material (the row model: the
/// inspector lists every field, not just the focused one). `"-"` when no
/// material is selected.
pub fn field_value_of(state: &EditorState, field: MaterialField) -> String {
    let Some(i) = sel_index(state) else {
        return "-".into();
    };
    let m = &state.level.materials[i];
    match field {
        MaterialField::Fg => format!("#{:06X}", m.fg & 0xFF_FFFF),
        MaterialField::Bg => format!("#{:06X}", m.bg & 0xFF_FFFF),
        MaterialField::Glyph => m
            .glyph
            .and_then(char::from_u32)
            .map(|c| c.to_string())
            .unwrap_or_else(|| "none".to_string()),
        MaterialField::Recipe => recipe_label(m.recipe).to_string(),
        MaterialField::RecipeP0 | MaterialField::RecipeP1 | MaterialField::RecipeP2 => {
            let slot = match field {
                MaterialField::RecipeP0 => 0,
                MaterialField::RecipeP1 => 1,
                _ => 2,
            };
            match (
                recipe_param_names(m.recipe)[slot],
                recipe_param_get(m.recipe, slot),
            ) {
                (Some(name), Some(v)) => format!("{name}={v:.3}"),
                _ => "-".to_string(),
            }
        }
        MaterialField::AnimKind => anim_kind_label(m.anim.kind).to_string(),
        MaterialField::AnimSpeed => format!("{:.2}", m.anim.speed),
        MaterialField::AnimAmplitude => format!("{:.2}", m.anim.amplitude),
        MaterialField::Rank => m.rank.to_string(),
        MaterialField::Surface => current_surface(state, &m.name).to_string(),
        MaterialField::Reflective => m.reflective.to_string(),
        MaterialField::Clarity => format!("{:.2}", m.clarity),
        MaterialField::Preset => m.preset.clone().unwrap_or_else(|| "none".to_string()),
        MaterialField::Texture => m.texture.clone().unwrap_or_else(|| "(none)".to_string()),
        MaterialField::Subcell => subcell_label(m.image_subcell).to_string(),
        MaterialField::Fit => fit_label(m.image_fit).to_string(),
        MaterialField::SpanAlongMin => match m.image_span {
            Some(s) => format!("{:.2}", s.along_min),
            None => "(none)".to_string(),
        },
        MaterialField::SpanAlongSize => match m.image_span {
            Some(s) => format!("{:.2}", s.along_size),
            None => "(none)".to_string(),
        },
        MaterialField::SpanUpTop => match m.image_span {
            Some(s) => format!("{:.2}", s.up_top),
            None => "(none)".to_string(),
        },
        MaterialField::SpanUpSize => match m.image_span {
            Some(s) => format!("{:.2}", s.up_size),
            None => "(none)".to_string(),
        },
    }
}

/// The selected material's `(glyph, fg)` for tinting the top-down — falls back to
/// a block glyph when the material has no fixed glyph. `None` when no materials.
pub fn material_tint(state: &EditorState, id: u16) -> Option<(char, u32)> {
    if id == 0 {
        return None;
    }
    let idx = (id - 1) as usize;
    let m = state.level.materials.get(idx)?;
    let glyph = m.glyph.and_then(char::from_u32).unwrap_or('\u{2592}'); // ▒ medium shade
    Some((glyph, m.fg))
}

// ── field steppers (pure) ────────────────────────────────────────────────────

/// Squared RGB distance between two `0x00RRGGBB` colours — the sum of the
/// squared per-component differences. Pure; the metric [`nearest_palette`] uses
/// to decide which swatch an off-palette colour is closest to.
#[must_use]
fn rgb_dist2(a: u32, b: u32) -> u32 {
    let comp = |c: u32, sh: u32| ((c >> sh) & 0xFF) as i32;
    let d = |sh: u32| {
        let x = comp(a, sh) - comp(b, sh);
        (x * x) as u32
    };
    d(16) + d(8) + d(0)
}

/// Index of the [`PALETTE`] swatch CLOSEST to `color` by [`rgb_dist2`]. Ties are
/// broken in the direction of travel (`delta >= 0` keeps the later swatch, a
/// negative delta the earlier one) so `+` and `-` never disagree about which of
/// two equidistant swatches they meant.
#[must_use]
pub fn nearest_palette(color: u32, delta: i32) -> usize {
    let c = color & 0xFF_FFFF;
    let mut best = 0usize;
    let mut best_d = u32::MAX;
    for (i, &p) in PALETTE.iter().enumerate() {
        let d = rgb_dist2(p & 0xFF_FFFF, c);
        let better = if delta >= 0 { d <= best_d } else { d < best_d };
        if better {
            best = i;
            best_d = d;
        }
    }
    best
}

/// Step `color` to the next/prev [`PALETTE`] swatch.
///
/// On-palette values wrap ±1 through the swatch list, exactly as they always
/// have. An OFF-palette value — which is almost every colour in the shipped
/// corpus: 234 distinct colours are authored and only 3 of them are swatches —
/// lands on the swatch NEAREST it by [`rgb_dist2`], never on index 0 (white).
/// The old behaviour snapped every off-palette colour to white on the first
/// `+`/`-` press, which silently destroyed the authored hue of nearly every
/// object in nearly every level; landing on the nearest swatch keeps the press
/// inside the same corner of colour space. (Recovering the EXACT original value
/// is the colour overlay's job — hex entry plus the level's own used-colour row
/// — not the stepper's.)
#[must_use]
pub fn step_palette(color: u32, delta: i32) -> u32 {
    let c = color & 0xFF_FFFF;
    let n = PALETTE.len() as i32;
    match PALETTE.iter().position(|&p| (p & 0xFF_FFFF) == c) {
        Some(i) => PALETTE[((i as i32 + delta).rem_euclid(n)) as usize],
        None => PALETTE[nearest_palette(c, delta)],
    }
}

/// The list a material-`glyph` step walks: [`GLYPHS`] (which includes `None`)
/// plus every off-list glyph this session has seen, plus `current`.
pub fn glyph_options(state: &EditorState, current: Option<char>) -> Vec<Option<char>> {
    crate::vocab::options_at(&GLYPHS, &state.vocab.material_glyph, &current).0
}

/// Step an optional glyph `delta` places along [`glyph_options`], wrapping.
///
/// The candidate set can never be complete — the field's real domain is "any
/// char" — so an off-list glyph is REMEMBERED rather than snapped to index 0.
/// `levels/12-mirror.lvl` authors `glyph = #` and `GODRAYS-demo.lvl` authors
/// `glyph = ^`; both used to be erased by one `+`/`-` press.
fn step_glyph(state: &mut EditorState, glyph: Option<u32>, delta: i32) -> Option<u32> {
    let cur = glyph.and_then(char::from_u32);
    crate::vocab::remember(&GLYPHS, &mut state.vocab.material_glyph, &cur);
    crate::vocab::step(&GLYPHS, &state.vocab.material_glyph, &cur, delta).map(|c| c as u32)
}

/// Remember a recipe's TUNED PARAMETERS, one instance per variant (the newest
/// wins). See [`step_recipe`] for why.
pub fn remember_recipe(extras: &mut Vec<MaterialRecipe>, r: MaterialRecipe) {
    match extras
        .iter_mut()
        .find(|e| recipe_variant_eq(Some(**e), Some(r)))
    {
        Some(slot) => *slot = r,
        None => extras.push(r),
    }
}

/// The recipe VARIANTS a `+`/`-` press walks: [`RECIPES`] plus any variant this
/// session has seen that `RECIPES` does not carry (`notewall` is synthesized-only
/// and is not on the list, so an authored one used to snap to `none`).
fn recipe_options(
    state: &EditorState,
    current: Option<MaterialRecipe>,
) -> Vec<Option<MaterialRecipe>> {
    let extras: Vec<Option<MaterialRecipe>> =
        state.vocab.recipe.iter().copied().map(Some).collect();
    crate::vocab::options_at_by(&RECIPES, &extras, &current, |a, b| {
        recipe_variant_eq(*a, *b)
    })
    .0
}

/// The recipe option list for the SELECTED material, for the inspector row.
pub fn recipe_variant_options(state: &EditorState) -> Vec<Option<MaterialRecipe>> {
    let cur = sel_index(state).and_then(|i| state.level.materials[i].recipe);
    recipe_options(state, cur)
}

/// Step an optional recipe through [`recipe_options`] (which includes `None`).
/// Matching is by VARIANT so an edited-param recipe still finds its place.
///
/// TWO losses are repaired here, both of the [`crate::vocab`] family:
///
/// * a variant `RECIPES` does not list (`notewall`) used to snap to `none`;
/// * and — the one that bites real content — the cycle used to hand back
///   `RECIPES`' DEFAULT parameters, so stepping off `grain scale=2.5 speck=0.6`
///   (`levels/12-mirror.lvl`) and back returned a DIFFERENT grain. The tuned
///   instance is remembered per variant and restored when the cycle lands on that
///   variant again, so a full lap is the identity in the data, not just in the
///   label.
fn step_recipe(
    state: &mut EditorState,
    recipe: Option<MaterialRecipe>,
    delta: i32,
) -> Option<MaterialRecipe> {
    if let Some(r) = recipe {
        remember_recipe(&mut state.vocab.recipe, r);
    }
    let opts = recipe_options(state, recipe);
    let next = crate::vocab::step_by(&opts, &[], &recipe, delta, |a, b| recipe_variant_eq(*a, *b));
    // Restore the tuned instance of whatever variant we landed on.
    next.map(|r| {
        state
            .vocab
            .recipe
            .iter()
            .copied()
            .find(|e| recipe_variant_eq(Some(*e), Some(r)))
            .unwrap_or(r)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn level_with_two_materials() -> EditorState {
        let mut s = EditorState::new_empty(6, 6, 4);
        new_material(&mut s); // mat0
        new_material(&mut s); // mat1
        s.material_mode = false; // reset so cycle's "first entry" path is exercised
        s.material_sel = 0;
        s
    }

    #[test]
    fn cycle_selects_and_wraps() {
        let mut s = level_with_two_materials();
        // First `m` enters mode, keeps selection 0.
        cycle_material(&mut s, 1);
        assert!(s.material_mode);
        assert_eq!(s.material_sel, 0);
        // Next steps advance and wrap (2 materials).
        cycle_material(&mut s, 1);
        assert_eq!(s.material_sel, 1);
        cycle_material(&mut s, 1);
        assert_eq!(s.material_sel, 0, "wraps");
        // Backward wraps too.
        cycle_material(&mut s, -1);
        assert_eq!(s.material_sel, 1);
    }

    #[test]
    fn cycle_with_no_materials_is_a_hinted_noop() {
        let mut s = EditorState::new_empty(4, 4, 2);
        cycle_material(&mut s, 1);
        assert!(s.material_mode);
        assert!(sel_index(&s).is_none());
    }

    #[test]
    fn new_material_unique_name_and_selected() {
        let mut s = EditorState::new_empty(4, 4, 2);
        new_material(&mut s);
        new_material(&mut s);
        assert_eq!(s.level.materials.len(), 2);
        assert_ne!(s.level.materials[0].name, s.level.materials[1].name);
        assert_eq!(s.material_sel, 1, "selects the newest");
        assert!(s.dirty);
    }

    #[test]
    fn paint_stamps_palette_id_and_makes_solid() {
        let mut s = level_with_two_materials();
        s.material_sel = 1; // → id 2
        s.cursor = (2, 3);
        s.z = 1;
        paint_id(&mut s);
        assert_eq!(s.map.cell_at(2, 3, 1), CellKind::WallSolid, "empty → solid");
        assert_eq!(
            s.map.side_data_at(2, 3, 1),
            2,
            "1-based id of material idx 1"
        );
        assert!(s.dirty);
    }

    #[test]
    fn paint_keeps_existing_kind() {
        let mut s = level_with_two_materials();
        s.material_sel = 0; // → id 1
        s.cursor = (1, 1);
        s.z = 0;
        s.map.set(1, 1, 0, CellKind::Floor);
        paint_id(&mut s);
        assert_eq!(s.map.cell_at(1, 1, 0), CellKind::Floor, "kind preserved");
        assert_eq!(s.map.side_data_at(1, 1, 0), 1);
    }

    #[test]
    fn clear_resets_id_keeps_kind() {
        let mut s = level_with_two_materials();
        s.material_sel = 0;
        s.cursor = (2, 2);
        s.z = 0;
        paint_id(&mut s);
        assert_eq!(s.map.side_data_at(2, 2, 0), 1);
        clear_id(&mut s);
        assert_eq!(s.map.side_data_at(2, 2, 0), 0, "id cleared");
        assert_eq!(s.map.cell_at(2, 2, 0), CellKind::WallSolid, "kind kept");
    }

    #[test]
    fn paint_with_no_material_is_a_noop() {
        let mut s = EditorState::new_empty(4, 4, 2);
        s.cursor = (1, 1);
        paint_id(&mut s);
        assert_eq!(s.map.cell_at(1, 1, 0), CellKind::Empty);
        assert!(!s.dirty);
    }

    #[test]
    fn field_cycle_and_fg_edit() {
        let mut s = level_with_two_materials();
        s.material_sel = 0;
        // Field starts at Fg.
        assert_eq!(s.material_field, MaterialField::Fg);
        let fg0 = s.level.materials[0].fg;
        adjust_field(&mut s, 1);
        assert_ne!(
            s.level.materials[0].fg, fg0,
            "fg stepped to a palette swatch"
        );
        assert!(PALETTE.contains(&(s.level.materials[0].fg & 0xFF_FFFF)));
        assert!(s.dirty);
        // `b` advances the field through the (now longer) cycle.
        cycle_field(&mut s);
        assert_eq!(s.material_field, MaterialField::Bg);
        cycle_field(&mut s);
        assert_eq!(s.material_field, MaterialField::Glyph);
        cycle_field(&mut s);
        assert_eq!(s.material_field, MaterialField::Recipe);
        // Stepping the full unique cycle returns to Fg (wrap).
        let unique_len = MATERIAL_FIELDS.len() - 1;
        s.material_field = MaterialField::Fg;
        for _ in 0..unique_len {
            cycle_field(&mut s);
        }
        assert_eq!(
            s.material_field,
            MaterialField::Fg,
            "wraps after full cycle"
        );
    }

    #[test]
    fn glyph_field_cycles_through_candidates_incl_none() {
        let mut s = level_with_two_materials();
        s.material_sel = 0;
        s.material_field = MaterialField::Glyph;
        // Default glyph is None; stepping forward lands on the first candidate.
        assert!(s.level.materials[0].glyph.is_none());
        adjust_field(&mut s, 1);
        assert_eq!(s.level.materials[0].glyph, Some('+' as u32));
        // Stepping back from index 1 returns to None.
        adjust_field(&mut s, -1);
        assert!(s.level.materials[0].glyph.is_none());
    }

    #[test]
    fn recipe_field_cycles_variants() {
        let mut s = level_with_two_materials();
        s.material_sel = 0;
        s.material_field = MaterialField::Recipe;
        assert!(s.level.materials[0].recipe.is_none());
        adjust_field(&mut s, 1);
        assert!(matches!(
            s.level.materials[0].recipe,
            Some(MaterialRecipe::Brick { .. })
        ));
        // Cycling backward from Brick (index 1) returns to None (index 0).
        adjust_field(&mut s, -1);
        assert!(s.level.materials[0].recipe.is_none());
        // Wrap: backward from None lands on the last recipe (Image, the final entry).
        adjust_field(&mut s, -1);
        assert!(matches!(
            s.level.materials[0].recipe,
            Some(MaterialRecipe::Image { .. })
        ));
    }

    // ── EXHAUSTIVE material field coverage ───────────────────────────────────

    /// `b` steps through EVERY field in the unique cycle once, then wraps. Guards
    /// against a declared-but-unreachable field.
    #[test]
    fn field_cycle_visits_every_field_once() {
        let mut s = level_with_two_materials();
        s.material_sel = 0;
        s.material_field = MaterialField::Fg;
        let unique = &MATERIAL_FIELDS[..MATERIAL_FIELDS.len() - 1];
        let mut seen = std::collections::HashSet::new();
        for _ in 0..unique.len() {
            seen.insert(s.material_field);
            cycle_field(&mut s);
        }
        assert_eq!(seen.len(), unique.len(), "all material fields reachable");
        assert_eq!(s.material_field, MaterialField::Fg, "cycle wraps");
    }

    /// Recipe-param slots switch their meaning + editable count by the active
    /// recipe variant: Brick exposes 3 params, Circuit 2 (slot 2 is a no-op).
    #[test]
    fn recipe_params_switch_by_variant() {
        let mut s = level_with_two_materials();
        s.material_sel = 0;

        // Set recipe = Brick (Recipe is a +/- candidate-cycle), then edit aspect.
        s.material_field = MaterialField::Recipe;
        adjust_field(&mut s, 1); // None → Brick
        assert!(matches!(
            s.level.materials[0].recipe,
            Some(MaterialRecipe::Brick { .. })
        ));
        assert_eq!(
            recipe_param_names(s.level.materials[0].recipe),
            [Some("aspect"), Some("mortar_w"), Some("grain")]
        );
        s.material_field = MaterialField::RecipeP0;
        let before = recipe_param_get(s.level.materials[0].recipe, 0).unwrap();
        adjust_field(&mut s, 1);
        let after = recipe_param_get(s.level.materials[0].recipe, 0).unwrap();
        assert!(
            (after - (before + 0.25)).abs() < 1e-4,
            "brick aspect stepped"
        );

        // Switch to Circuit: slot 2 has no param ⇒ editing it is a hinted no-op.
        s.material_field = MaterialField::Recipe;
        adjust_field(&mut s, 1); // Brick → Circuit
        assert!(matches!(
            s.level.materials[0].recipe,
            Some(MaterialRecipe::Circuit { .. })
        ));
        assert_eq!(recipe_param_names(s.level.materials[0].recipe)[2], None);
        s.material_field = MaterialField::RecipeP2;
        let recipe_before = s.level.materials[0].recipe;
        s.dirty = false;
        adjust_field(&mut s, 1);
        assert_eq!(
            s.level.materials[0].recipe, recipe_before,
            "no slot-2 param ⇒ unchanged"
        );
        assert!(!s.dirty);

        // Editing Circuit slot 1 (trace_w) DOES change.
        s.material_field = MaterialField::RecipeP1;
        let tw0 = recipe_param_get(s.level.materials[0].recipe, 1).unwrap();
        adjust_field(&mut s, 1);
        let tw1 = recipe_param_get(s.level.materials[0].recipe, 1).unwrap();
        assert!((tw1 - tw0).abs() > 1e-6, "circuit trace_w stepped");
    }

    /// Animation kind cycles via `c`; speed + amplitude step via `+`/`-`.
    #[test]
    fn animation_kind_cycles_and_scalars_step() {
        use re_world::MaterialAnimKind;
        let mut s = level_with_two_materials();
        s.material_sel = 0;
        s.material_field = MaterialField::AnimKind;
        assert_eq!(s.level.materials[0].anim.kind, MaterialAnimKind::None);
        cycle_enum(&mut s, 1);
        assert_eq!(s.level.materials[0].anim.kind, MaterialAnimKind::Pulse);
        // A non-None kind seeds a sane speed so it actually animates.
        assert!(s.level.materials[0].anim.speed > 0.0);

        s.material_field = MaterialField::AnimSpeed;
        let sp0 = s.level.materials[0].anim.speed;
        adjust_field(&mut s, 1);
        assert!(s.level.materials[0].anim.speed > sp0);

        s.material_field = MaterialField::AnimAmplitude;
        adjust_field(&mut s, 1);
        assert!(s.level.materials[0].anim.amplitude > 0.0);
    }

    /// Rank steps and clamps to [0,127].
    #[test]
    fn rank_steps_and_clamps() {
        let mut s = level_with_two_materials();
        s.material_sel = 0;
        s.material_field = MaterialField::Rank;
        adjust_field(&mut s, 5);
        assert_eq!(s.level.materials[0].rank, 5);
        for _ in 0..200 {
            adjust_field(&mut s, 1);
        }
        assert_eq!(s.level.materials[0].rank, 127, "rank clamps at 127");
        for _ in 0..200 {
            adjust_field(&mut s, -1);
        }
        assert_eq!(s.level.materials[0].rank, 0, "rank clamps at 0");
    }

    /// Surface binding cycles and binds AT MOST one surface per material.
    #[test]
    fn surface_binding_cycles_and_is_exclusive() {
        let mut s = level_with_two_materials();
        s.material_sel = 0;
        let name = s.level.materials[0].name.clone();
        s.material_field = MaterialField::Surface;
        assert_eq!(current_surface(&s, &name), "none");
        cycle_enum(&mut s, 1); // none → wall_solid
        assert_eq!(current_surface(&s, &name), "wall_solid");
        assert_eq!(
            s.level.surface_materials.wall_solid.as_deref(),
            Some(name.as_str())
        );
        cycle_enum(&mut s, 1); // wall_solid → wall_trace
        assert_eq!(current_surface(&s, &name), "wall_trace");
        // The old binding was cleared (exclusive).
        assert!(s.level.surface_materials.wall_solid.is_none());
    }

    /// Reflective toggles + clarity steps via `c`/`+`/`-`.
    #[test]
    fn reflective_toggle_and_clarity_step() {
        let mut s = level_with_two_materials();
        s.material_sel = 0;
        s.material_field = MaterialField::Reflective;
        assert!(!s.level.materials[0].reflective);
        cycle_enum(&mut s, 1);
        assert!(s.level.materials[0].reflective);
        s.material_field = MaterialField::Clarity;
        let c0 = s.level.materials[0].clarity;
        adjust_field(&mut s, -1);
        assert!(s.level.materials[0].clarity < c0);
        for _ in 0..100 {
            adjust_field(&mut s, -1);
        }
        assert_eq!(s.level.materials[0].clarity, 0.0, "clarity clamps at 0");
    }

    /// Preset cycles through the RENDERER's catalog (`re_render`'s
    /// `CATALOG_NAMES`), starting at `None`.
    #[test]
    fn preset_cycles_through_catalog() {
        let mut s = level_with_two_materials();
        s.material_sel = 0;
        s.material_field = MaterialField::Preset;
        assert!(s.level.materials[0].preset.is_none());
        cycle_enum(&mut s, 1);
        assert_eq!(
            s.level.materials[0].preset.as_deref(),
            Some(re_render::ascii_shader::ids::CATALOG_NAMES[0])
        );
        // Backward from index 1 returns to None.
        cycle_enum(&mut s, -1);
        assert!(s.level.materials[0].preset.is_none());
    }

    /// `+`/`-` on an enum field hints; `c` on a numeric field hints. No mutation.
    #[test]
    fn wrong_key_for_field_is_a_hinted_noop() {
        let mut s = level_with_two_materials();
        s.material_sel = 0;
        // Enum field + numeric key.
        s.material_field = MaterialField::Surface;
        s.dirty = false;
        adjust_field(&mut s, 1);
        assert!(!s.dirty, "+/- on enum field is a no-op");
        // Numeric field + enum key.
        s.material_field = MaterialField::Rank;
        s.dirty = false;
        cycle_enum(&mut s, 1);
        assert!(!s.dirty, "c on numeric field is a no-op");
    }

    /// The recipe cycle reaches Image, and the Image-specific fields (texture /
    /// subcell / fit / span) edit the material's top-level image keys.
    #[test]
    fn image_recipe_and_fields_are_authorable() {
        let mut s = level_with_two_materials();
        s.material_sel = 0;
        // Give the level a texture so the binding cycle has a candidate.
        s.level.textures.push(re_level::LevelTexture {
            name: "wall".into(),
            file: "assets/texture/wall.webp".into(),
            max_size: None,
            height: None,
            ..Default::default()
        });

        // Recipe cycles all the way to Image (last entry in RECIPES).
        s.material_field = MaterialField::Recipe;
        adjust_field(&mut s, -1); // None → (wrap) last recipe = Image
        assert!(
            matches!(
                s.level.materials[0].recipe,
                Some(MaterialRecipe::Image { .. })
            ),
            "recipe wraps backward to Image"
        );

        // Texture binding: `c` cycles (none) → wall.
        s.material_field = MaterialField::Texture;
        assert!(s.level.materials[0].texture.is_none());
        cycle_enum(&mut s, 1);
        assert_eq!(s.level.materials[0].texture.as_deref(), Some("wall"));
        // Wrap back to (none).
        cycle_enum(&mut s, 1);
        assert!(s.level.materials[0].texture.is_none());

        // Subcell: one step walks the resolution LADDER, off → half.
        s.material_field = MaterialField::Subcell;
        assert_eq!(s.level.materials[0].image_subcell, SubcellMode::Off);
        cycle_enum(&mut s, 1);
        assert_eq!(s.level.materials[0].image_subcell, SubcellMode::Half);
        cycle_enum(&mut s, 1);
        assert_eq!(s.level.materials[0].image_subcell, SubcellMode::Quad);

        // Fit: contain (default) → cover (one step forward from contain).
        s.material_field = MaterialField::Fit;
        assert_eq!(s.level.materials[0].image_fit, Fit::Contain);
        cycle_enum(&mut s, 1);
        assert_eq!(s.level.materials[0].image_fit, Fit::Cover);

        // Span: editing a component seeds Some(default) then steps it.
        s.material_field = MaterialField::SpanAlongSize;
        assert!(s.level.materials[0].image_span.is_none());
        adjust_field(&mut s, 2); // 1.0 + 2*0.5 = 2.0
        let span = s.level.materials[0].image_span.expect("span seeded");
        assert!((span.along_size - 2.0).abs() < 1e-4, "along_size stepped");
        // Size axes never go non-positive.
        for _ in 0..20 {
            adjust_field(&mut s, -1);
        }
        assert!(s.level.materials[0].image_span.unwrap().along_size >= 0.5);
    }

    #[test]
    fn material_tint_resolves_id_to_glyph_fg() {
        let mut s = level_with_two_materials();
        s.level.materials[0].fg = 0x00_20FF80;
        s.level.materials[0].glyph = Some('+' as u32);
        let (g, fg) = material_tint(&s, 1).expect("id 1 resolves");
        assert_eq!(g, '+');
        assert_eq!(fg & 0xFF_FFFF, 0x20FF80);
        // id 0 ⇒ no tint; an out-of-range id ⇒ None.
        assert!(material_tint(&s, 0).is_none());
        assert!(material_tint(&s, 99).is_none());
    }

    // ── STEP 1: colour stepping must not destroy authored data ───────────────

    /// The shipped corpus authors 234 distinct colours and only 3 are swatches,
    /// so the OFF-palette path is the common one. It must land on the nearest
    /// swatch by RGB distance — NOT on index 0 (white), which is what the old
    /// `None => PALETTE[0]` snap did and which destroyed the authored hue on
    /// the first `+` press.
    #[test]
    fn off_palette_colour_steps_to_the_nearest_swatch_not_white() {
        // A deep red: nearest swatch is red (0xFF4040), a long way from white.
        let stepped = step_palette(0x00_C00000, 1);
        assert_ne!(stepped, PALETTE[0], "must not snap to white");
        assert_eq!(stepped, 0x00_FF4040, "lands on red, the nearest swatch");
        // Same for a bright teal → cyan, and a charcoal → near-black. (A DARK
        // teal such as 0x106070 is genuinely closest to near-black, not to
        // cyan — the metric is RGB distance, not hue angle.)
        assert_eq!(step_palette(0x00_50C0F0, 1), 0x00_40D0FF);
        assert_eq!(step_palette(0x00_2A2F35, 1), 0x00_303840);
        // Stepping DOWN from an off-palette value is equally non-destructive.
        assert_eq!(step_palette(0x00_C00000, -1), 0x00_FF4040);
    }

    /// Every result of a step is a palette swatch (the invariant the pre-existing
    /// `PALETTE.contains(...)` assertions rely on), whichever direction and
    /// whatever the starting colour.
    #[test]
    fn every_step_result_is_a_palette_swatch() {
        for &start in &[
            0x00_000000u32,
            0x00_FFFFFF,
            0x00_123456,
            0x00_C00000,
            0x00_7F7F7F,
        ] {
            for delta in [-1i32, 1] {
                let out = step_palette(start, delta);
                assert!(
                    PALETTE.contains(&(out & 0xFF_FFFF)),
                    "{start:#08X} step {delta} gave {out:#08X}, not a swatch"
                );
            }
        }
    }

    /// ON-palette behaviour is UNCHANGED: ±1 through the list, wrapping at both
    /// ends. This is the regression guard for the refactor.
    #[test]
    fn on_palette_stepping_still_wraps_by_one() {
        for (i, &p) in PALETTE.iter().enumerate() {
            let n = PALETTE.len();
            assert_eq!(step_palette(p, 1), PALETTE[(i + 1) % n], "forward from {i}");
            assert_eq!(
                step_palette(p, -1),
                PALETTE[(i + n - 1) % n],
                "back from {i}"
            );
        }
        // The wrap ends explicitly.
        assert_eq!(step_palette(PALETTE[0], -1), PALETTE[PALETTE.len() - 1]);
        assert_eq!(step_palette(PALETTE[PALETTE.len() - 1], 1), PALETTE[0]);
    }

    /// An off-palette colour steps somewhere ADJACENT IN HUE: the swatch it
    /// lands on is closer to it than white is, for every authored-looking
    /// colour that is not itself nearest to white.
    #[test]
    fn off_palette_step_stays_near_the_authored_hue() {
        for &c in &[
            0x00_C00000u32,
            0x00_106070,
            0x00_2A2F35,
            0x00_804010,
            0x00_30A050,
        ] {
            let out = step_palette(c, 1);
            assert!(
                rgb_dist2(out, c) < rgb_dist2(PALETTE[0], c),
                "{c:#08X} landed on {out:#08X}, further away than white"
            );
        }
    }

    /// The distance metric is the plain sum of squared component differences.
    #[test]
    fn rgb_dist2_is_the_sum_of_squared_components() {
        assert_eq!(rgb_dist2(0x00_000000, 0x00_000000), 0);
        assert_eq!(rgb_dist2(0x00_010203, 0x00_000000), 1 + 4 + 9);
        // Symmetric, and the alpha byte is ignored.
        assert_eq!(
            rgb_dist2(0x00_102030, 0x00_405060),
            rgb_dist2(0x00_405060, 0x00_102030)
        );
        assert_eq!(rgb_dist2(0xFF_102030, 0x00_102030), 0);
    }

    /// Equidistant swatches are resolved by the DIRECTION OF TRAVEL, so `+` and
    /// `-` never disagree about which of two tied swatches they meant.
    #[test]
    fn nearest_palette_breaks_ties_by_direction() {
        // Exactly midway between red (FF4040, idx 3) and orange (FF9020, idx 4).
        let mid = 0x00_FF6830;
        assert_eq!(
            rgb_dist2(PALETTE[3], mid),
            rgb_dist2(PALETTE[4], mid),
            "tie setup"
        );
        assert_eq!(nearest_palette(mid, 1), 4, "forward keeps the later swatch");
        assert_eq!(
            nearest_palette(mid, -1),
            3,
            "backward keeps the earlier one"
        );
    }
}
