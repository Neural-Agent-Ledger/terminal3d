//! `surface_fx` — the SEMANTIC SURFACE post-layer.
//!
//! Five lore-named surfaces the world demands a material for — **frost / wet**,
//! **redaction**, **watching**, **wear / residue**, **decay** — realised as a
//! single OPTIONAL post-layer on a [`Material`](crate::ascii_shader::Material).
//!
//! ## Why a post-layer, not a new recipe
//! A [`TextureRecipe`](crate::ascii_shader::TextureRecipe) variant must be threaded
//! through `props` / `sample` / `height_at`; a semantic surface is really a *finish*
//! applied OVER whatever base recipe the cell already shades to (a redacted CodeWall,
//! a frosted brick, a rotting circuit). So each is a [`SurfaceFx`] applied right at
//! the end of [`Material::apply_lighting`](crate::ascii_shader::Material::apply_lighting), after the base texel, lighting, pulse,
//! flow-hue and fog have produced an ordinary [`Texel`]. It mutates `{glyph, fg, bg}`
//! in place — never the geometry, never the recipe dispatch.
//!
//! ## Contract (the golden-safety guarantee)
//! - The field is `Option<SurfaceFx>` on `Material`, defaulting to `None` → a material
//!   that does not opt in is **byte-identical** to today. The catalog APPENDS the five
//!   `frost` / `wet_stone` / `redacted` / `watcher` / `worn_floor` / `decay` slots; no
//!   existing id shifts.
//! - Every value is a **pure function of `(coord, ctx.state, ctx.tick/time, params)`**
//!   — the same determinism contract the recipes hold, so goldens pin it. A
//!   FAR-FROZEN cell suppresses the time-varying parts (handled by the caller passing
//!   `frozen`), so the far field stays quiet.
//! - State binding stays **generic**: the redaction/decay masks read `ctx.state` only
//!   as a `u32` "clearing level" the game-side bridge encodes — `re_render` never
//!   learns it is a puzzle.

use crate::ascii_shader::{brighten_rgb, dim_rgb, hash2i, hash3i, mix_rgb, smoothstep, Texel};
use glam::Vec2;

// ─────────────────────────── named tunables ─────────────────────────────────

/// FROST lattice density: worley feature cells per world-unit. Higher ⇒ smaller,
/// denser ice crystals.
const FROST_SCALE: f32 = 4.0;
/// FROST accretion coverage `[0,1]`: the worley-distance threshold under which a
/// cell carries a crystal speck. Higher ⇒ more of the surface frosts over.
const FROST_COVERAGE: f32 = 0.42;
/// FROST cold-tint blend `[0,1]`: how far a frosted cell's colour pulls toward the
/// pale-blue ice palette. The crystalline specks pull harder than the rime film.
const FROST_TINT: f32 = 0.55;
/// FROST crystal minimum cold-tint blend `[0,1]`: the floor a crystal speck pulls
/// toward the ice colour, high enough that even the warmest base reads cold blue.
const FROST_CRYSTAL_TINT_MIN: f32 = 0.75;
/// FROST pale-blue ice colour the accretion tints toward (packed `0x00RRGGBB`). A
/// distinctly cold hue (red well below blue) so a crystal reads blue over any base.
const FROST_COLD_RGB: u32 = 0x008CD0FF;
/// FROST crystal glyphs (sparse, by worley-cell hash). A small kindred set of
/// asterisk/cross "ice star" marks over the base.
const FROST_GLYPHS: [u32; 4] = [
    0x2744, // ❄ snowflake
    0x2735, // ✵ six-pointed star
    0x273B, // ✻ teardrop-spoked asterisk
    0x00B7, // · faint rime speck
];

/// WET sheen specular lift `[0,1]`: how much a wet cell brightens its highlight
/// channel (we have no relief here at post-time, so this is a cheap cool-glint
/// boost on the inked fg, modulated by a worley "puddle" mask).
const WET_SHEEN: f32 = 0.5;
/// WET cool-highlight colour the sheen mixes toward (a cold water glint).
const WET_HIGHLIGHT_RGB: u32 = 0x00B0D8FF;
/// WET puddle lattice density: worley cells per world-unit for the wet patches.
const WET_SCALE: f32 = 2.4;

/// REDACTION light-bar glyph (`░`) — a partially censored cell (a word half-known).
const REDACT_LIGHT_GLYPH: u32 = 0x2591; // ░
/// REDACTION full-bar glyph (`█`) — a fully censored cell (a name not yet assembled).
const REDACT_FULL_GLYPH: u32 = 0x2588; // █
/// REDACTION bar colour — a flat surveillance grey the censor paints over the text.
const REDACT_BAR_RGB: u32 = 0x00202428;
/// REDACTION full-bar fg — marginally lighter than the bg so the `█` censor block
/// is not a fully degenerate fg==bg cell (which a readability linter flags as
/// "invisible"): the block keeps a faint surveillance-grey grain, still reading as a
/// solid redaction, never as legible text.
const REDACT_FULL_FG_RGB: u32 = 0x00343A40;
/// REDACTION light-bar colour — a slightly lighter grey for the `░` half-redacted
/// cell, so the two redaction levels read apart.
const REDACT_LIGHT_RGB: u32 = 0x00484E54;

/// WATCHER lattice period (world-units) the eye/face motif tiles on. One motif per
/// `WATCH_PERIOD × WATCH_PERIOD` block so it reads as discrete watching cells.
const WATCH_PERIOD: f32 = 3.0;
/// WATCHER proximity-brighten distance (world cells): at/below this the motif glows
/// toward its emissive peak (it "notices" you as you approach).
const WATCH_NEAR: f32 = 6.0;
/// WATCHER far distance (world cells): beyond this the motif is at its dim resting
/// brightness.
const WATCH_FAR: f32 = 16.0;
/// WATCHER maximum proximity emissive gain (multiplier on the inked fg at `WATCH_NEAR`).
const WATCH_GAIN: f32 = 1.8;
/// WATCHER iris colour the pupil/eye glows (a cold pale stare).
const WATCH_EYE_RGB: u32 = 0x00DCE8FF;
/// WATCHER eye/pupil glyphs by lattice-phase (the "turned-toward-you" motif).
const WATCH_EYE_GLYPH: u32 = 0x25C9; // ◉ ringed iris
const WATCH_PUPIL_GLYPH: u32 = 0x25CF; // ● solid pupil
const WATCH_LID_GLYPH: u32 = 0x2014; // — a closed/lidded cell (the lattice gaps)

/// WEAR groove density: scanline-like worn paths per world-unit on a floor.
const WEAR_SCALE: f32 = 1.6;
/// WEAR darkening `[0,1]`: how much a scuffed groove dims the floor (worn-darker).
const WEAR_DARKEN: f32 = 0.35;
/// WEAR dust-speck coverage `[0,1]`: sparse hashed light specks (settled dust).
const WEAR_DUST: f32 = 0.10;
/// WEAR dust glyphs — faint settled specks over the worn floor.
const WEAR_DUST_GLYPHS: [u32; 3] = [0x00B7, 0x2024, 0x002E]; // · ․ .
/// WEAR dust colour — a pale settled grey-tan.
const WEAR_DUST_RGB: u32 = 0x00A89880;

/// DECAY corruption density `[0,1]` at full intensity: the fraction of cells that
/// rot to a mojibake glyph. Scaled by the effective intensity (param + state).
const DECAY_DENSITY: f32 = 0.5;
/// DECAY sickly tint blend `[0,1]`: how far a rotted cell pulls toward the bilious
/// decay colour.
const DECAY_TINT: f32 = 0.6;
/// DECAY sickly colour (a dead green-grey) a corrupted cell tints toward.
const DECAY_RGB: u32 = 0x004A5238;
/// DECAY mojibake glyph set — the bounded "bit-rot" replacement characters (box-
/// drawing / block fragments that read as a broken/garbled cell). Deterministic
/// pick by a per-cell hash — this is the local re-implementation of re_glitch's
/// bit-rot idea, scoped to ONE surface (no re_glitch dependency).
const DECAY_GLYPHS: [u32; 8] = [
    0x2592, // ▒ medium shade
    0x2593, // ▓ dark shade
    0x25A0, // ■ black square
    0x2580, // ▀ upper half block
    0x2584, // ▄ lower half block
    0x00A4, // ¤ currency (mojibake staple)
    0x00BF, // ¿ inverted question (garble)
    0x2426, // ␦ substitute symbol
];

// ─────────────────────────── the layer ──────────────────────────────────────

/// A SEMANTIC SURFACE finish applied over the shaded base [`Texel`]. Each variant is
/// a self-contained, deterministic post-effect; [`SurfaceFx::apply`] mutates the cell
/// in place. `Copy` data so the host [`Material`](crate::ascii_shader::Material) stays
/// `Copy`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SurfaceFx {
    /// FROST — cold worley-crystal accretion: pale-blue rime film + sparse ice-star
    /// glyphs where the crystal lattice is dense. Reads as a frozen surface; the
    /// cold-axis semantic.
    Frost,
    /// WET / CONDENSATION — a cool specular sheen over the base, gated by a worley
    /// "puddle" mask so the wet patches read as discrete pools rather than a flat wash.
    Wet,
    /// REDACTION — censor bars (`█`/`░`) over the base glyph, gated by a `[0,1]`
    /// reveal level derived from `ctx.state`: `reveal = 0` ⇒ fully redacted (`█`),
    /// rising state LIFTS the redaction (the name assembling) until `reveal >= 1`
    /// shows the underlying text. `hold` cells (a per-cell hash above the reveal
    /// front) drop to `░` then clear, so the lift looks like a dissolve, not a wipe.
    Redaction,
    /// WATCHING — a world-fixed eye/face lattice that brightens on approach. The
    /// motif tiles on [`WATCH_PERIOD`]; the iris/pupil cells emissive-lift toward the
    /// camera (proximity), the most-repeated horror image. World-fixed (cheap) — no
    /// billboard sampler.
    Watching,
    /// WEAR / RESIDUE — scuffed worn grooves (dimmed scanline paths) + sparse settled
    /// dust specks on a floor. A low-effort static residue finish.
    Wear,
    /// DECAY / ROT — BOUNDED local bit-rot: a deterministic fraction of cells swap to
    /// a mojibake glyph + a sickly tint, intensity from `state`/`pulse` (deadness /
    /// attempts) or the fixed `base` floor. Localised to THIS surface only — never a
    /// global post-pass. `base` `[0,1]` is the resting corruption floor; the effective
    /// intensity is `base` lifted by a state term.
    Decay {
        /// The resting corruption floor `[0,1]`; effective intensity is this
        /// value lifted by a state term.
        base: f32,
    },
}

impl SurfaceFx {
    /// True if this finish's look varies over time / pulse and so must keep the host
    /// material re-shading (defeats the idle dirty-flag). Only [`SurfaceFx::Watching`]
    /// (proximity is camera-driven, recomputed by the cast anyway) and a state/pulse-
    /// reactive [`SurfaceFx::Decay`] are dynamic — but both are functions of inputs the
    /// cast already re-derives per frame (distance / state / pulse), NOT of `ctx.time`,
    /// so none of them need to force animation here. They are intentionally STATIC in
    /// the `is_animated` sense (a settled frame with fixed state re-shades identically).
    #[inline]
    pub fn is_animated(self) -> bool {
        false
    }

    /// Apply the finish to a shaded `texel` at world `coord`.
    ///
    /// `state` is the cell's generic state-bus word (`0` for an empty bus → the
    /// resting look: fully redacted, resting decay floor, dim watcher). `distance` is
    /// the cell's view distance (world cells) for the proximity-driven watcher.
    /// `frozen` (far-freeze) suppresses the proximity brighten so the far field is
    /// quiet. Pure, allocation-free.
    pub fn apply(self, texel: &mut Texel, coord: Vec2, state: u32, distance: f32, frozen: bool) {
        match self {
            SurfaceFx::Frost => apply_frost(texel, coord),
            SurfaceFx::Wet => apply_wet(texel, coord),
            SurfaceFx::Redaction => apply_redaction(texel, coord, state),
            SurfaceFx::Watching => apply_watching(texel, coord, distance, frozen),
            SurfaceFx::Wear => apply_wear(texel, coord),
            SurfaceFx::Decay { base } => apply_decay(texel, coord, state, base),
        }
    }
}

// ─────────────────────────── P4 frost / wet ─────────────────────────────────

/// Worley F1 distance to the nearest jittered feature point of a `scale`-density
/// integer lattice, plus the feature cell's hash (for per-crystal variation). A
/// local, single-seed re-implementation of the recipe worley (kept private to the
/// module so the finish does not depend on the recipe's `worley` signature).
#[inline]
fn fx_worley(seed: u32, p: Vec2) -> (f32, u32) {
    let ix = p.x.floor() as i32;
    let iy = p.y.floor() as i32;
    let mut best = f32::INFINITY;
    let mut best_cell = 0u32;
    for dy in -1..=1 {
        for dx in -1..=1 {
            let cx = ix + dx;
            let cy = iy + dy;
            let h = hash3i(seed as i32, cx, cy);
            let jx = (h & 0xFFFF) as f32 / 65535.0;
            let jy = ((h >> 16) & 0xFFFF) as f32 / 65535.0;
            let feat = Vec2::new(cx as f32 + jx, cy as f32 + jy);
            let d = (feat - p).length();
            if d < best {
                best = d;
                best_cell = h;
            }
        }
    }
    (best, best_cell)
}

fn apply_frost(texel: &mut Texel, coord: Vec2) {
    // A rime FILM tints the whole surface cold-pale; the worley crystal lattice adds
    // sparse ice-star glyphs where a feature point is near.
    let (f1, cell) = fx_worley(0x0F305700 ^ FROST_SEED, coord * FROST_SCALE);
    // Rime film: everywhere, a gentle cold tint (so even un-crystalled cells read cold).
    texel.fg = mix_rgb(texel.fg, FROST_COLD_RGB, FROST_TINT * 0.45);
    texel.bg = mix_rgb(texel.bg, FROST_COLD_RGB, FROST_TINT * 0.30);
    // Crystal accretion: near a feature point, swap to an ice-star glyph + pull the
    // tint harder + brighten (the crystal catches light).
    if f1 < FROST_COVERAGE {
        let g = FROST_GLYPHS[(cell as usize) & 3];
        texel.glyph = g;
        // The crystal core brightens; the rim stays the film tint. The crystal pulls
        // HARD to the cold ice colour (so a crystal is unambiguously cold blue even
        // over a warm base), then brightens by its proximity to the feature centre.
        let core = 1.0 - smoothstep(0.0, FROST_COVERAGE, f1); // 1 at centre → 0 at rim
                                                              // Floor the crystal tint high enough that even the warmest base reads cold blue.
        let crystal_tint = (FROST_CRYSTAL_TINT_MIN + 0.2 * core).clamp(0.0, 1.0);
        texel.fg = mix_rgb(texel.fg, FROST_COLD_RGB, crystal_tint);
        texel.fg = brighten_rgb(texel.fg, 1.0 + 0.5 * core);
    }
}

fn apply_wet(texel: &mut Texel, coord: Vec2) {
    // Worley "puddle" mask: wet where a feature point is near; the sheen is a cool
    // highlight lifted in over the inked fg, strongest at the puddle centre.
    let (f1, _) = fx_worley(0x57E70000 ^ FROST_SEED, coord * WET_SCALE);
    let wet = 1.0 - smoothstep(0.25, 0.7, f1); // 1 (deep puddle) → 0 (dry)
    if wet <= 0.0 {
        return;
    }
    let sheen = WET_SHEEN * wet;
    // Pull the fg toward the cool glint and brighten it (a specular boost the relief
    // pass would give a real wet surface — applied here as a post-finish so it works
    // over any base recipe). The bg darkens slightly (a wet surface reads deeper).
    texel.fg = mix_rgb(texel.fg, WET_HIGHLIGHT_RGB, 0.35 * sheen);
    texel.fg = brighten_rgb(texel.fg, 1.0 + sheen);
    texel.bg = dim_rgb(texel.bg, 1.0 - 0.25 * wet);
}

// ─────────────────────────── P3 redaction ───────────────────────────────────

/// Decode the generic state word into a redaction REVEAL level `[0,1]`. `0` ⇒ fully
/// redacted (the resting/empty-bus look); the bridge raises it as the name assembles.
/// We read the low byte as a 0..=255 level so the bridge can ramp it smoothly without
/// `re_render` knowing the puzzle meaning.
#[inline]
pub fn redaction_reveal(state: u32) -> f32 {
    ((state & 0xFF) as f32 / 255.0).clamp(0.0, 1.0)
}

fn apply_redaction(texel: &mut Texel, coord: Vec2, state: u32) {
    let reveal = redaction_reveal(state);
    // Per-cell dissolve threshold: a stable hash in [0,1]. A cell clears (shows the
    // underlying glyph) once the reveal front passes its threshold; just below, it is
    // half-redacted (`░`); well below, fully redacted (`█`). This makes the lift read
    // as a grainy dissolve (the name assembling cell by cell) rather than a hard wipe.
    let ix = coord.x.floor() as i32;
    let iy = coord.y.floor() as i32;
    let thr = (hash2i(ix ^ 0x5249, iy ^ 0x4544) >> 8) as f32 / (1u32 << 24) as f32;
    if reveal >= thr {
        // Cleared — the underlying text shows through untouched.
        return;
    }
    // Still redacted: the closer the front, the lighter the bar.
    let margin = thr - reveal; // > 0
    if margin < 0.18 {
        texel.glyph = REDACT_LIGHT_GLYPH;
        texel.fg = REDACT_LIGHT_RGB;
        texel.bg = REDACT_BAR_RGB;
    } else {
        texel.glyph = REDACT_FULL_GLYPH;
        // Faintly-lighter fg than bg so the `█` block keeps a hint of grain (and is not
        // a degenerate fg==bg cell) while still reading as solid redaction, not text.
        texel.fg = REDACT_FULL_FG_RGB;
        texel.bg = REDACT_BAR_RGB;
    }
}

// ─────────────────────────── P2 watching ────────────────────────────────────

fn apply_watching(texel: &mut Texel, coord: Vec2, distance: f32, frozen: bool) {
    // World-fixed lattice: one eye motif per WATCH_PERIOD block. The block-local
    // position decides the motif glyph (a ringed iris at the centre cell, a solid
    // pupil just off-centre, a lidded `—` in the surrounding gaps) so the block reads
    // as a face/eye turned out of the wall.
    let bx = (coord.x / WATCH_PERIOD).floor();
    let by = (coord.y / WATCH_PERIOD).floor();
    let lx = coord.x - bx * WATCH_PERIOD; // [0,WATCH_PERIOD)
    let ly = coord.y - by * WATCH_PERIOD;
    // Centre of the block is the eye; a small radius around it is the iris/pupil.
    let c = WATCH_PERIOD * 0.5;
    let r = Vec2::new(lx - c, ly - c).length();
    let is_eye = r < 0.6;
    let is_iris = r < 1.1;
    let glyph = if is_eye {
        WATCH_PUPIL_GLYPH
    } else if is_iris {
        WATCH_EYE_GLYPH
    } else {
        WATCH_LID_GLYPH
    };
    texel.glyph = glyph;
    // Proximity brighten: the eye notices you. At/below WATCH_NEAR the gain peaks; by
    // WATCH_FAR it rests. A far-frozen cell holds at rest (quiet far field).
    let prox = if frozen {
        0.0
    } else {
        1.0 - smoothstep(WATCH_NEAR, WATCH_FAR, distance) // 1 near → 0 far
    };
    // Only the iris/pupil glow toward the eye colour; the lids stay the base tone.
    if is_iris {
        texel.fg = mix_rgb(texel.fg, WATCH_EYE_RGB, 0.4 + 0.5 * prox);
        let gain = 1.0 + (WATCH_GAIN - 1.0) * prox;
        texel.fg = brighten_rgb(texel.fg, gain);
    }
}

// ─────────────────────────── P7 wear / residue ──────────────────────────────

fn apply_wear(texel: &mut Texel, coord: Vec2) {
    // Worn grooves: a warped scanline path the foot traffic wore darker. Use a value-
    // noise-warped sine so the grooves meander like a trodden path rather than ruler
    // lines. Dim the groove cells; the high ridges keep full brightness.
    let warp = (hash2i(
        (coord.x * 3.0).floor() as i32,
        (coord.y * 3.0).floor() as i32,
    ) >> 8) as f32
        / (1u32 << 24) as f32; // [0,1) stable warp
    let band = ((coord.y * WEAR_SCALE + warp * 0.8).sin() * 0.5 + 0.5).powf(2.0); // 0..1
    let groove = smoothstep(0.55, 0.95, band); // 1 in a worn groove
    if groove > 0.0 {
        texel.fg = dim_rgb(texel.fg, 1.0 - WEAR_DARKEN * groove);
        texel.bg = dim_rgb(texel.bg, 1.0 - WEAR_DARKEN * 0.6 * groove);
    }
    // Settled dust: sparse hashed light specks over the floor.
    let ix = (coord.x * 2.0).floor() as i32;
    let iy = (coord.y * 2.0).floor() as i32;
    let h = hash2i(ix ^ 0x4455, iy ^ 0x5354);
    let speck = (h >> 8) as f32 / (1u32 << 24) as f32;
    if speck < WEAR_DUST {
        texel.glyph = WEAR_DUST_GLYPHS[(h as usize) % WEAR_DUST_GLYPHS.len()];
        texel.fg = mix_rgb(texel.fg, WEAR_DUST_RGB, 0.6);
    }
}

// ─────────────────────────── P6 surface decay ───────────────────────────────

/// The effective corruption intensity `[0,1]` of a decay finish: its resting `base`
/// floor lifted by the cell's state word. We read the low byte of `state` as a
/// 0..=255 "deadness/attempts" level (the bridge encodes it generically) and let it
/// drive the corruption UP from the floor toward 1. Pure.
#[inline]
pub fn decay_intensity(base: f32, state: u32) -> f32 {
    let lift = (state & 0xFF) as f32 / 255.0;
    (base + (1.0 - base) * lift).clamp(0.0, 1.0)
}

fn apply_decay(texel: &mut Texel, coord: Vec2, state: u32, base: f32) {
    let intensity = decay_intensity(base, state);
    if intensity <= 0.0 {
        return;
    }
    // Bounded bit-rot: a per-cell hash decides if this cell rots, and which mojibake
    // glyph. Deterministic (a function of the world cell + a fixed seed) — this is the
    // local re-implementation of the bit-rot ALGORITHM (corrupt a hashed fraction),
    // scoped to ONE surface, with NO re_glitch dependency and NO global state.
    let ix = coord.x.floor() as i32;
    let iy = coord.y.floor() as i32;
    let h = hash3i(0x52_4F_54_00u32 as i32, ix, iy); // "ROT\0"
    let roll = (h >> 8) as f32 / (1u32 << 24) as f32; // [0,1)
                                                      // A sickly tint spreads across the whole surface with intensity (the rot in the
                                                      // air); the glyph corruption is the discrete bit-rot on the hashed fraction.
    texel.fg = mix_rgb(texel.fg, DECAY_RGB, DECAY_TINT * intensity * 0.5);
    texel.bg = mix_rgb(texel.bg, DECAY_RGB, DECAY_TINT * intensity * 0.7);
    if roll < DECAY_DENSITY * intensity {
        let g = DECAY_GLYPHS[(h >> 3) as usize % DECAY_GLYPHS.len()];
        texel.glyph = g;
        texel.fg = mix_rgb(texel.fg, DECAY_RGB, DECAY_TINT);
        // Corrupted cells also dim (the surface is dying, not glowing).
        texel.fg = dim_rgb(texel.fg, 0.8);
    }
}

/// Per-finish seed salt so the frost / wet worley fields do not align with each
/// other or the base recipe noise.
const FROST_SEED: u32 = 0x0000_F305;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ascii_shader::unpack_rgb;

    fn texel(glyph: u32, fg: u32, bg: u32) -> Texel {
        Texel {
            glyph,
            fg,
            bg,
            rank: 0,
        }
    }

    /// FROST picks cold cyan + a crystal glyph over a crystal cell, and tints cold
    /// even off-crystal. We scan a small grid to guarantee at least one crystal cell.
    #[test]
    fn frost_yields_cold_crystals() {
        let base_fg = 0x00FF8040; // warm orange base
        let mut crystal_found = false;
        for gy in 0..40 {
            for gx in 0..40 {
                let mut t = texel(b'#' as u32, base_fg, 0x00101010);
                let coord = Vec2::new(gx as f32 * 0.25, gy as f32 * 0.25);
                SurfaceFx::Frost.apply(&mut t, coord, 0, 4.0, false);
                // Every cell pulls cooler than the warm base (more blue than red shifted).
                let c = unpack_rgb(t.fg);
                assert!(
                    c.z >= unpack_rgb(base_fg).z - 0.01,
                    "frost must not REDUCE the blue channel below base"
                );
                if FROST_GLYPHS.contains(&t.glyph) {
                    crystal_found = true;
                    // A crystal cell is markedly blue (cold).
                    assert!(c.z > c.x, "frost crystal must be cold (blue > red)");
                }
            }
        }
        assert!(
            crystal_found,
            "frost must produce at least one crystal glyph over the grid"
        );
    }

    /// WET lifts the fg toward a cool glint on at least some (puddle) cells and never
    /// corrupts the glyph.
    #[test]
    fn wet_adds_cool_sheen() {
        let base_fg = 0x00404040;
        let mut any_sheen = false;
        for gy in 0..30 {
            for gx in 0..30 {
                let mut t = texel(b'=' as u32, base_fg, 0x00080808);
                let coord = Vec2::new(gx as f32 * 0.3, gy as f32 * 0.3);
                SurfaceFx::Wet.apply(&mut t, coord, 0, 4.0, false);
                assert_eq!(t.glyph, b'=' as u32, "wet must not change the glyph");
                if t.fg != base_fg {
                    any_sheen = true;
                    let c = unpack_rgb(t.fg);
                    assert!(c.z >= c.x, "wet sheen must be cool (blue ≥ red)");
                }
            }
        }
        assert!(any_sheen, "wet must brighten at least one puddle cell");
    }

    /// REDACTION at reveal 0 (empty bus) censors EVERY cell to a bar glyph; at reveal
    /// 1 (full) it clears every cell back to the base glyph.
    #[test]
    fn redaction_full_then_clear() {
        let base_glyph = b'A' as u32;
        let bars = [REDACT_FULL_GLYPH, REDACT_LIGHT_GLYPH];
        for gy in 0..20 {
            for gx in 0..20 {
                let coord = Vec2::new(gx as f32, gy as f32);
                // Fully redacted (state 0).
                let mut t = texel(base_glyph, 0x00FFFFFF, 0x00101010);
                SurfaceFx::Redaction.apply(&mut t, coord, 0, 4.0, false);
                assert!(
                    bars.contains(&t.glyph),
                    "reveal 0 must censor every cell (got {:#x} at {gx},{gy})",
                    t.glyph
                );
                // Fully revealed (state low byte = 255).
                let mut t2 = texel(base_glyph, 0x00FFFFFF, 0x00101010);
                SurfaceFx::Redaction.apply(&mut t2, coord, 0xFF, 4.0, false);
                assert_eq!(
                    t2.glyph, base_glyph,
                    "reveal 1 must clear every cell back to the base text"
                );
            }
        }
    }

    /// REDACTION reveal monotonicity: more cleared cells at a higher reveal level.
    #[test]
    fn redaction_lifts_with_state() {
        let base_glyph = b'X' as u32;
        let count_cleared = |state: u32| -> usize {
            let mut n = 0;
            for gy in 0..40 {
                for gx in 0..40 {
                    let mut t = texel(base_glyph, 0x00FFFFFF, 0x00101010);
                    SurfaceFx::Redaction.apply(
                        &mut t,
                        Vec2::new(gx as f32, gy as f32),
                        state,
                        4.0,
                        false,
                    );
                    if t.glyph == base_glyph {
                        n += 1;
                    }
                }
            }
            n
        };
        let low = count_cleared(0x40);
        let high = count_cleared(0xC0);
        assert!(
            low < high,
            "redaction must clear MORE cells as state rises ({low} !< {high})"
        );
    }

    /// WATCHING places an iris/pupil motif and brightens it on approach (near > far).
    #[test]
    fn watching_brightens_on_approach() {
        // Centre of a block is the eye. Pick a coord at a block centre.
        let coord = Vec2::new(WATCH_PERIOD * 0.5, WATCH_PERIOD * 0.5);
        let mut near = texel(b'.' as u32, 0x00303030, 0x00050505);
        let mut far = texel(b'.' as u32, 0x00303030, 0x00050505);
        SurfaceFx::Watching.apply(&mut near, coord, 0, 1.0, false);
        SurfaceFx::Watching.apply(&mut far, coord, 0, 30.0, false);
        assert!(
            near.glyph == WATCH_PUPIL_GLYPH || near.glyph == WATCH_EYE_GLYPH,
            "block centre must be an eye/pupil motif"
        );
        let ln = unpack_rgb(near.fg).dot(glam::Vec3::new(0.299, 0.587, 0.114));
        let lf = unpack_rgb(far.fg).dot(glam::Vec3::new(0.299, 0.587, 0.114));
        assert!(
            ln > lf,
            "watcher must be BRIGHTER near ({ln}) than far ({lf})"
        );
    }

    /// DECAY at a fixed base swaps a fraction of cells to a mojibake glyph + sickly
    /// tint, and rises with state.
    #[test]
    fn decay_corrupts_and_scales_with_state() {
        let base_glyph = b'o' as u32;
        let count_rot = |base: f32, state: u32| -> usize {
            let mut n = 0;
            for gy in 0..40 {
                for gx in 0..40 {
                    let mut t = texel(base_glyph, 0x00C0C0C0, 0x00101010);
                    SurfaceFx::Decay { base }.apply(
                        &mut t,
                        Vec2::new(gx as f32, gy as f32),
                        state,
                        4.0,
                        false,
                    );
                    if DECAY_GLYPHS.contains(&t.glyph) {
                        n += 1;
                    }
                }
            }
            n
        };
        // A meaningful resting floor corrupts some cells.
        let resting = count_rot(0.3, 0);
        assert!(resting > 0, "decay floor must corrupt at least one cell");
        // Rising state corrupts MORE.
        let driven = count_rot(0.3, 0xFF);
        assert!(
            driven > resting,
            "decay must intensify with state ({resting} !< {driven})"
        );
    }

    /// decay_intensity / redaction_reveal helper monotonicity (the bridge contract).
    #[test]
    fn state_decoders_monotone() {
        assert!(decay_intensity(0.2, 0) < decay_intensity(0.2, 0xFF));
        assert_eq!(decay_intensity(0.0, 0), 0.0);
        assert!(redaction_reveal(0) < redaction_reveal(0x80));
        assert_eq!(redaction_reveal(0), 0.0);
    }
}
