//! Flowing-TEXT pipe treatment — diegetic command-line words streaming through a
//! POWERED puzzle pipe.
//!
//! Where the base flow material scrolls an anonymous glyph pattern
//! input→output along a tagged `WallTrace` pipe cell (see
//! [`crate::ascii_shader`] `flow_shift_coord` / `flow_is_hot`), this module replaces
//! that pattern with a stream of READABLE WORDS — short command-line-ish tokens
//! ("fire", "leaf", "emit", "model", "echo", …) repeating along the pipe run. The
//! lore frame: the pipe carries the commands other stuck consciousnesses inject
//! into the shell, and the apparent flow direction depends on which side you
//! stand to read it.
//!
//! ## What it does (the four behaviours)
//!
//! 1. **WORDS, not noise.** A repeating pool of real tokens is laid end-to-end along
//!    the pipe axis, each separated by [`WORD_SEP`]. The glyph for a cell is the
//!    *actual letter* at that position in the stream — no duplicated-glyph filler.
//!    WHICH pool is a [`Words`] register: the OWNING NODE's authored word pool
//!    (threaded in as `PipeTextCtx::pool` from the level's
//!    [`PoolTable`](re_core::PoolTable)) in either its clean or its corrupted
//!    [`PoolRegister`], with the built-in [`PIPE_WORDS`] as the no-pool fallback. The
//!    shade path streams the pool ([`overlay_from_pool`]); the corrupted register
//!    engages on a level whose authored glitch density passes
//!    [`POOL_ALT_GLITCH_THRESHOLD`].
//! 2. **Per-word colour.** Each word is drawn in the PIPE'S OWN colour (the level's
//!    trace material), varied in value by [`WORD_SHADES`] (indexed by a hash of the
//!    word), so adjacent words are distinguishable while the stream stays one wire.
//!    Only the READING face (vertical, [`PipeFace::Reading`]) is full contrast; an
//!    underside or top ([`PipeFace::Quiet`]) carries the words as a sparse, dim texture.
//! 3. **Staggered scroll.** Words flow slowly along the pipe. DIFFERENT rows
//!    (`wall_v` bands) scroll at DIFFERENT speeds and with a per-row START OFFSET, so
//!    the rows move "по очереди" rather than in lockstep. Deterministic function of
//!    `(time, row)`.
//! 4. **View-dependent direction.** The base flow direction comes from the state bus
//!    ([`FlowDir`]); the apparent direction is then FLIPPED by which side the camera
//!    views the pipe from — `sign(view_dir · tangent_u)`. Walk around to the opposite
//!    face and the same stream reads as running the other way.
//!
//! ## The "glass tube" (underline) effect
//!
//! Text cells get a colored UNDERLINE ([`CellStyle::with_underline_color`]) plus a
//! darkened backing `bg`. In the present layer a colored underline is a real third
//! per-cell colour channel (SGR `58;2;r;g;b`), drawn as a thin lit rule UNDER the
//! glyph — reading as the front edge / meniscus of a glass tube the text sits behind.
//! Combined with the dim, slightly tube-tinted `bg`, the word reads as "behind glass"
//! rather than painted flat on a wall. The underline colour is the word's own hue
//! brightened, so the glass glows faintly in the word's colour.
//!
//! ## Purity / determinism
//!
//! Pure compute. Every output is a function of `(world/wall coord, time, state, view
//! vector)` — golden-reproducible, no clock, no allocation, no terminal, no `unsafe`.
//! A cell that is NOT a powered pipe is left untouched (the caller only invokes this
//! when `flow_is_hot`), so non-pipe goldens are byte-identical.

use crate::ascii_shader::Texel;
use glam::{Vec2, Vec3};
use re_core::{CellStyle, FlowDir, NodePool};

/// Built-in word pool streamed along a powered pipe. Short command-line-ish
/// tokens — the diegetic shell commands the system carries. The pool REPEATS
/// along the run.
///
/// This is now the FALLBACK register: a pipe whose owning node authored a word pool
/// can stream those words instead (see [`Words`] / [`overlay_from_pool`]), which is
/// what makes "reading the flow IS reading her memory" true. A demo level with no
/// pool data keeps this list, so those frames are unchanged.
///
/// Kept lower-case + 3–6 chars so each word reads cleanly across a few glyph-cells.
/// ASCII only (every byte is one glyph-cell).
pub const PIPE_WORDS: &[&str] = &[
    "fire", "leaf", "emit", "model", "echo", "wait", "loop", "sync", "ping", "fork", "kill",
    "load", "seed", "void", "bind",
];

/// Separator drawn between two adjacent words in the stream. A middle dot reads as a
/// deliberate token boundary (not a space the eye loses), keeping the words distinct
/// as they flow. One glyph-cell wide.
pub const WORD_SEP: char = '·';

/// Per-word SHADE table: `(brightness, whiten)` pairs applied to the pipe's OWN hue.
/// Indexed by a hash of the word so a given token always gets the same shade, and
/// adjacent DIFFERENT words differ.
///
/// Deliberately NOT a fixed palette (e.g. chartreuse/cyan/sand/violet/mint/rose
/// laid over every pipe in every level): once the vertical renderer draws pipe
/// undersides, a fixed palette fills the upper third of the frame with pastel
/// text and pastel contour blocks that belong to no level's palette. A word's
/// colour is instead the pipe's own lit colour — the level's trace material —
/// varied only in value, so a stream reads as ONE wire carrying several
/// tokens, never a rainbow.
pub const WORD_SHADES: &[(f32, f32)] = &[
    (1.00, 0.00),
    (0.80, 0.00),
    (1.00, 0.22),
    (0.90, 0.10),
    (0.72, 0.00),
    (1.00, 0.12),
];

/// Peak channel a READING-face word is normalised to before its shade applies, so a
/// dim or shadowed trace material still yields a legible word (the hue is kept, the
/// value is lifted). `0..=255`.
pub const WORD_READ_PEAK: f32 = 224.0;

/// Neutral word colour used when the pipe's own colour carries no hue to derive from
/// (black or near-black base): a pale grey, never a pastel accent.
pub const WORD_NEUTRAL: u32 = 0x00C8_C8C0;

/// Which face of a pipe cell is being shaded — decides how loud the words are.
///
/// The player READS a pipe from its side: a vertical face at or near eye level. The
/// horizontal faces (the underside seen when a pipe runs overhead, the top seen from a
/// gallery) are seen at a grazing angle, stretch every letter across many cells, and
/// used to fill the upper view with a text wall. They now carry the words as a quiet
/// texture: sparse, low-contrast, no glass underline.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PipeFace {
    /// A vertical face — full words, full contrast, glass underline.
    #[default]
    Reading,
    /// A horizontal face (underside / top) — sparse, dim texture.
    Quiet,
}

/// Foreshortening at and above which a VERTICAL face is also quiet. A pipe side seen
/// along its length (a grazing angle, `1/cos θ ≥ 2.5`, i.e. more than ~66° off
/// face-on) stretches each letter across several columns into runs like `SSSSSSUUUU`
/// that nobody reads; it is the pipe's side, but not a face the player reads from.
pub const READ_MAX_FORESHORTEN: f32 = 2.5;

/// Largest `|z|` of the unit view direction at which a vertical face still counts as
/// read at EYE LEVEL (≈27° above or below the horizon). A pipe side far overhead — the
/// run right above the player's head, magnified so one letter spans a dozen columns —
/// is outside it and goes quiet; the same run seen across the room, near the horizon,
/// stays a reading face.
pub const READ_MAX_VIEW_Z: f32 = 0.45;

impl PipeFace {
    /// Classify a pipe face from its world normal's `z`, its foreshortening (`1/cos`
    /// of the horizontal view angle, as the shade path computes it) and the `z` of the
    /// unit view direction. Horizontal faces, grazing sides and sides seen steeply
    /// above or below eye level are [`PipeFace::Quiet`]. Pure.
    #[inline]
    pub fn classify(normal_z: f32, foreshorten: f32, view_z: f32) -> PipeFace {
        if normal_z.abs() > 0.5
            || foreshorten >= READ_MAX_FORESHORTEN
            || view_z.abs() > READ_MAX_VIEW_Z
        {
            PipeFace::Quiet
        } else {
            PipeFace::Reading
        }
    }
}

/// Largest on-screen width, in columns, of ONE stream letter on a reading face. A
/// pipe side close overhead would magnify every letter across many columns
/// (`SSSSSSUUUUUU`); the shade path re-scales the stream to the face's footprint so a
/// letter is never wider than this. ONE column, not two: at 2 every magnified
/// letter still doubled (`CCRRIIMMSS`), so the footprint is the face anchor's
/// exact screen span instead.
pub const READ_MAX_GLYPH_COLS: f32 = 1.0;

/// Fraction of QUIET-face texels that keep a letter; the rest show only the dim
/// glass backing. World-anchored (see [`quiet_keeps`]), so the pattern does not churn.
pub const QUIET_DENSITY: f32 = 0.22;

/// The glyph a kept QUIET-face texel shows: a speck, never a letter.
pub const QUIET_SPECK: char = '·';

/// How far a QUIET-face letter moves from its backing toward the word colour
/// (`0` = invisible, `1` = full reading contrast).
pub const QUIET_CONTRAST: f32 = 0.30;

/// Extra dimming of the glass backing on a QUIET face, on top of [`GLASS_BG_DIM`]
/// (the backing is not darkened as far, so the underside reads as a surface in the
/// room's light rather than a black tube). Multiplier on the INPUT bg.
pub const QUIET_BG_DIM: f32 = 0.70;

/// Sub-glyph subdivision of the quiet-face density mask. Each stream glyph cell is cut
/// into `QUIET_SUBDIV × QUIET_SUBDIV` world patches so a letter stretched over many
/// screen cells by a grazing underside is speckled rather than kept or dropped whole.
const QUIET_SUBDIV: f32 = 3.0;

/// Base scroll rate (stream-positions per second) for the FIRST pipe row. Slow on
/// purpose — the player reads the words, not a strobe. Per-row speed scales this by
/// [`ROW_SPEED_STEP`].
pub const PIPE_TEXT_SCROLL_RATE: f32 = 1.6;

/// Per-row scroll-speed increment: row `r` scrolls at
/// `PIPE_TEXT_SCROLL_RATE * (1 + r * ROW_SPEED_STEP)`. Different rows therefore drift
/// at visibly different speeds (the streams shear past each other) rather than as one
/// rigid block.
pub const ROW_SPEED_STEP: f32 = 0.35;

/// Per-row START-OFFSET (stream-positions) so rows begin staggered ("по очереди"):
/// row `r` is pre-advanced by `r * ROW_PHASE_STEP`. Combined with [`ROW_SPEED_STEP`]
/// this makes the rows enter at different times AND move at different rates.
pub const ROW_PHASE_STEP: f32 = 2.7;

/// How many distinct scroll ROWS the pipe's vertical (`wall_v ∈ [0,1]`) extent is cut
/// into. Each band gets its own speed + phase. A small count keeps adjacent rows
/// distinguishable without shearing the words apart letter-by-letter.
pub const PIPE_TEXT_ROWS: u32 = 5;

/// World-along units that map to ONE stream cell (one glyph along the pipe). Sets how
/// "zoomed" the text is on the pipe surface — smaller ⇒ more letters per world-cell.
/// Tuned so a typical word spans a readable handful of glyph-cells.
pub const ALONG_PER_GLYPH: f32 = 0.32;

/// Brightness multiplier applied to the cell `bg` to make the glass backing read as a
/// dim tube behind the text (`< 1.0` darkens). The text glyph stays bright on top; the
/// darker backing + the lit underline give the "behind glass" read.
pub const GLASS_BG_DIM: f32 = 0.35;

/// Brightness multiplier for the underline (glass-edge) colour relative to the word
/// hue (`> 1.0` brightens) — the lit front edge of the tube glints in the word's hue.
pub const GLASS_UNDERLINE_LIFT: f32 = 1.25;

/// One built FNV-1a hash of an ASCII byte slice (stable, alloc-free) — used to pick a
/// word's hue so the SAME token always gets the SAME colour.
#[inline]
fn fnv1a(bytes: &[u8]) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for &b in bytes {
        h ^= u32::from(b);
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

/// The shade `(brightness, whiten)` for a word, hashing the WORD TEXT (not the index)
/// so re-ordering the pool keeps a token's shade. Pure.
#[inline]
fn shade_for_word(word: &str) -> (f32, f32) {
    let h = fnv1a(word.as_bytes());
    WORD_SHADES[(h as usize) % WORD_SHADES.len()]
}

/// The colour a word is drawn in on a READING face: the pipe's own colour `base`
/// normalised to [`WORD_READ_PEAK`] (hue kept, value lifted so a shadowed trace still
/// reads), then the word's shade. A hue-less base falls back to [`WORD_NEUTRAL`]. Pure.
#[inline]
pub fn word_colour(base: u32, word: &str) -> u32 {
    let r = ((base >> 16) & 0xFF) as f32;
    let g = ((base >> 8) & 0xFF) as f32;
    let b = (base & 0xFF) as f32;
    let peak = r.max(g).max(b);
    let norm = if peak < 8.0 {
        WORD_NEUTRAL
    } else {
        scale_rgb(base, WORD_READ_PEAK / peak)
    };
    let (bright, whiten) = shade_for_word(word);
    mix_rgb(scale_rgb(norm, bright), 0x00FF_FFFF, whiten)
}

/// Is the QUIET-face texel at `(along, wall_v)` one of the sparse [`QUIET_DENSITY`]
/// fraction that keeps its letter? A hash of the WORLD patch (not the screen cell, not
/// the scrolled stream position), so the mask is fixed to the surface: the words flow
/// beneath it and it never flickers as the camera moves. Pure.
#[inline]
fn quiet_keeps(along: f32, wall_v: f32) -> bool {
    let a = (along / ALONG_PER_GLYPH * QUIET_SUBDIV).floor() as i32;
    let v = (wall_v.clamp(0.0, 1.0) * PIPE_TEXT_ROWS as f32 * QUIET_SUBDIV).floor() as i32;
    let mut h = (a as u32).wrapping_mul(0x9E37_79B1) ^ (v as u32).wrapping_mul(0x85EB_CA77);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    ((h & 0xFFFF) as f32 / 65536.0) < QUIET_DENSITY
}

/// Glitch density (the level's authored `[effects] glitch`, `[0,1]`) STRICTLY above
/// which a powered pipe streams the owning node's CORRUPTED `pool_alt` register
/// instead of its clean `pool`.
///
/// Why a whole-level scalar and not a per-pipe flag: there is no per-pipe
/// corrupt flag in the `.lvl` format — the narrative decay is a property of
/// the LEVEL, so the words rot when the level does. `0.22` sits inside the
/// ACTUAL authored range (campaign glitch spans 0.10–0.30): the early
/// movements (L5 = 0.20, L6 = 0.22) keep clean memory-words, and the register
/// flips from the grief movement onward (L8 = 0.25, L7 = 0.30) — corruption
/// of her words begins exactly where forgetting becomes the theme (the river
/// LETHE). Strictly-greater, so a level authored exactly at the bar stays
/// clean (a deliberate "no surprise at the boundary" choice — this is why
/// L6's 0.22 does NOT flip).
pub const POOL_ALT_GLITCH_THRESHOLD: f32 = 0.22;

/// WHICH of a node's two authored word registers a pipe streams. Decided per LEVEL
/// from the glitch density ([`POOL_ALT_GLITCH_THRESHOLD`]), not per pipe.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PoolRegister {
    /// The node's clean `pool` — her memory as she remembers it.
    #[default]
    Clean,
    /// The node's corrupted `pool_alt` — the same memory after the decay.
    Alt,
}

impl PoolRegister {
    /// The register a level with this glitch density streams. Non-finite input is
    /// treated as no glitch (a defined answer for every input shape).
    #[inline]
    pub fn for_glitch(glitch: f32) -> PoolRegister {
        if glitch.is_finite() && glitch > POOL_ALT_GLITCH_THRESHOLD {
            PoolRegister::Alt
        } else {
            PoolRegister::Clean
        }
    }
}

/// WHICH word list a pipe streams: the engine's built-in fallback, or one of the two
/// registers of the pool of the node that owns this pipe (resolved through the
/// level's [`PoolTable`](re_core::PoolTable)).
///
/// A tiny borrow-only enum, not a trait object and not a copy: the pool words stay
/// where they are and the stream walks them by index. All three registers go through
/// the SAME stream walker, so there is one streaming implementation, not three.
#[derive(Clone, Copy, Debug)]
pub enum Words<'a> {
    /// The engine's built-in [`PIPE_WORDS`] — used when the pipe has no pool entry.
    Builtin,
    /// The owning node's authored CLEAN pool, read in ANCHOR-FIRST order
    /// ([`NodePool::ordered_word`]): the node's own contributed word leads the stream,
    /// the rest of its pool follows.
    Pool(&'a NodePool),
    /// The owning node's authored CORRUPTED pool (`pool_alt`), streamed on a level
    /// whose glitch density passes [`POOL_ALT_GLITCH_THRESHOLD`]. Anchor-first when
    /// the alt list is index-aligned with the clean one (see [`alt_ordered_word`]).
    PoolAlt(&'a NodePool),
}

impl<'a> Words<'a> {
    /// Number of words in this register.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Words::Builtin => PIPE_WORDS.len(),
            Words::Pool(p) => p.len(),
            Words::PoolAlt(p) => p.pool_alt.len(),
        }
    }

    /// Whether the register carries no words (a `Pool`/`PoolAlt` with nothing
    /// authored — the caller should then fall back rather than stream nothing).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The `i`-th word, or `None` past the end.
    ///
    /// Matches on `*self` (the enum is `Copy`) so the bound pool reference carries the
    /// POOL's lifetime `'a`, not a reborrow of this `&self`.
    #[inline]
    pub fn word(&self, i: usize) -> Option<&'a str> {
        match *self {
            Words::Builtin => PIPE_WORDS.get(i).copied(),
            Words::Pool(p) => p.ordered_word(i),
            Words::PoolAlt(p) => alt_ordered_word(p, i),
        }
    }
}

/// The `i`-th word of a node's CORRUPTED register, in the closest thing to
/// anchor-first order the data supports.
///
/// `NodePool::anchor` indexes the CLEAN pool, so it cannot be applied blindly to
/// `pool_alt`. Two cases, both honest:
///
/// * `pool_alt.len() == pool.len()` — the alt list is the clean list re-spelled, so
///   `pool_alt[k]` IS the corrupted counterpart of `pool[k]`; the anchor's
///   counterpart therefore leads the stream, exactly as in the clean register.
/// * otherwise — the author wrote an independent list of a different length, so
///   there is no counterpart to lead with; it streams in AUTHORED order.
///
/// Pure, allocation-free.
#[inline]
pub fn alt_ordered_word(pool: &NodePool, i: usize) -> Option<&str> {
    let alt = &pool.pool_alt;
    if alt.is_empty() {
        return None;
    }
    if alt.len() != pool.pool.len() {
        return alt.get(i).map(String::as_str);
    }
    let a = pool.anchor_index();
    if i == 0 {
        return Some(alt[a].as_str());
    }
    let mut seen = 0usize;
    for (k, w) in alt.iter().enumerate() {
        if k == a {
            continue;
        }
        seen += 1;
        if seen == i {
            return Some(w.as_str());
        }
    }
    None
}

/// Total length of the repeating stream in glyph-cells: every word plus one separator
/// after it. Pure + allocation-free — just a sum over the small register.
#[inline]
fn stream_len_of(words: Words<'_>) -> u32 {
    (0..words.len())
        .filter_map(|i| words.word(i))
        .map(|w| w.chars().count() as u32 + 1) // +1 for the trailing separator
        .sum()
}

/// Total length of the BUILT-IN stream (the historical helper, kept for the tests and
/// for callers that do not carry a pool).
#[inline]
fn stream_len() -> u32 {
    stream_len_of(Words::Builtin)
}

/// Resolve stream position `pos` (already reduced mod the stream length) to the
/// `(glyph, word)` it lands on: the actual letter at that spot, or the separator
/// between words. `word` is the token the glyph belongs to (the separator is attributed
/// to the word it follows, so the trailing `·` shares that word's hue). Pure,
/// allocation-free — walks the small register.
#[inline]
fn glyph_at_in<'a>(words: Words<'a>, pos: u32) -> (char, &'a str) {
    let mut acc = 0u32;
    for i in 0..words.len() {
        let Some(word) = words.word(i) else { break };
        let n = word.chars().count() as u32;
        if pos < acc + n {
            // Inside this word: the real letter at (pos - acc).
            let idx = (pos - acc) as usize;
            let ch = word.chars().nth(idx).unwrap_or(WORD_SEP);
            return (ch, word);
        }
        if pos == acc + n {
            // The separator slot following this word.
            return (WORD_SEP, word);
        }
        acc += n + 1;
    }
    // pos was already reduced mod the stream length, so we always hit above; this is an
    // unreachable defensive default that still returns a defined glyph. An EMPTY
    // register (which the callers exclude) also lands here.
    (WORD_SEP, words.word(0).unwrap_or(PIPE_WORDS[0]))
}

/// [`glyph_at_in`] on the built-in register. Test-only since the streaming entry
/// points pass their register explicitly; the built-in-register tests read it directly.
#[cfg(test)]
#[inline]
fn glyph_at(pos: u32) -> (char, &'static str) {
    glyph_at_in(Words::Builtin, pos)
}

/// The flow direction AFTER the view-dependent flip. The base direction comes from the
/// state bus (`state`); the sign of `view_dir · tangent_u` (which side of the pipe the
/// camera reads it from) then flips it, so looking at the same pipe from the opposite
/// face reverses the apparent flow.
///
/// `tangent_u` is the pipe's surface-horizontal (cross) axis; `view_dir` points from
/// the surface back toward the camera. A `None` light context (focused unit tests with
/// no lighting) keeps the un-flipped base direction. Returns `+1.0` / `-1.0` for a
/// powered pipe, `0.0` for an unpowered one. Pure.
#[inline]
pub fn view_flow_sign(state: u32, view_dir: Option<Vec3>, tangent_u: Vec3) -> f32 {
    let base = match re_core::decode_flow_state(state) {
        FlowDir::None => return 0.0,
        FlowDir::Forward => 1.0,
        FlowDir::Backward => -1.0,
    };
    match view_dir {
        // No lighting context → no view side known → un-flipped base direction.
        None => base,
        Some(v) => {
            let d = v.dot(tangent_u);
            // Reading the pipe from the −u side reverses the apparent flow. A
            // near-zero dot (edge-on) keeps the base sign rather than flickering.
            if d < 0.0 {
                -base
            } else {
                base
            }
        }
    }
}

/// The integer ROW band `[0, PIPE_TEXT_ROWS)` a `wall_v ∈ [0,1]` falls into. Each band
/// scrolls at its own speed/phase. Pure.
#[inline]
fn row_of(wall_v: f32) -> u32 {
    let v = wall_v.clamp(0.0, 1.0);
    ((v * PIPE_TEXT_ROWS as f32) as u32).min(PIPE_TEXT_ROWS - 1)
}

/// The stream position (mod stream length) for a cell, given its world-along
/// coordinate, its row band, the flow sign, and the time. Different rows get a
/// different speed ([`ROW_SPEED_STEP`]) AND a different start offset ([`ROW_PHASE_STEP`])
/// so they move staggered. Pure function of `(along, row, sign, time)`.
#[inline]
fn stream_pos(along: f32, row: u32, sign: f32, time: f32) -> u32 {
    stream_pos_len(along, row, sign, time, stream_len())
}

/// [`stream_pos`] with the stream length supplied — so a pool register of a different
/// total length scrolls at the same rate and wraps at its own period.
#[inline]
fn stream_pos_len(along: f32, row: u32, sign: f32, time: f32, len: u32) -> u32 {
    let len = len.max(1);
    let speed = PIPE_TEXT_SCROLL_RATE * (1.0 + row as f32 * ROW_SPEED_STEP);
    let phase = row as f32 * ROW_PHASE_STEP;
    // Base spatial position along the pipe (which glyph-cell of the stream this world
    // point is), plus the scroll (sign chooses direction), plus the per-row phase.
    let spatial = along / ALONG_PER_GLYPH;
    let scrolled = spatial - sign * speed * time + phase;
    // Reduce to [0, len) with a floor-based modulo that is correct for negatives.
    let lenf = len as f32;
    let m = scrolled - (scrolled / lenf).floor() * lenf;
    (m as u32).min(len - 1)
}

/// Inputs the caller threads in from the `SampleCtx` for a powered pipe cell. Grouped
/// so the call site in `shade_cell`/`Material::sample` stays a single line.
///
/// The `'a` lifetime carries the WORD POOL borrow: the shade path looks the
/// pipe's tag up in the level's [`PoolTable`](re_core::PoolTable) and hands the owning
/// node's pool in here by reference — no clone, no allocation, and the pool stays owned
/// by the level's text-surface registry.
#[derive(Clone, Copy, Debug)]
pub struct PipeTextCtx<'a> {
    /// State-bus word for this pipe (decodes to the base [`FlowDir`]).
    pub state: u32,
    /// World-along coordinate on the pipe face (the flow axis): `wall_along`.
    pub along: f32,
    /// Vertical wall coordinate `[0,1]` (selects the scroll row): `wall_v`.
    pub wall_v: f32,
    /// Elapsed seconds (the scroll clock): `time`.
    pub time: f32,
    /// View direction (surface→camera, unit) for the side-flip; `None` if no lighting.
    pub view_dir: Option<Vec3>,
    /// Pipe surface cross-axis (unit) for the side-flip: `tangent_u`.
    pub tangent_u: Vec3,
    /// The WORD POOL of the node that owns this pipe, looked up by the cell's pipe tag
    /// in the level's [`PoolTable`](re_core::PoolTable). `None` (a demo level, an
    /// unauthored node, or no level loaded) ⇒ the built-in [`PIPE_WORDS`] register.
    ///
    /// [`overlay_from_pool`] — what the shade path calls — streams these words;
    /// [`overlay`] is the built-in-only entry kept as the reference/fallback.
    pub pool: Option<&'a NodePool>,
    /// WHICH register of that pool to stream: clean `pool`, or the corrupted
    /// `pool_alt` on a level whose glitch density passes
    /// [`POOL_ALT_GLITCH_THRESHOLD`]. A whole-level fact, handed down with the pool by
    /// `scene_text::with_pipe_pool_register`.
    pub register: PoolRegister,
    /// The pipe's OWN lit colour (the level's trace material as shaded for this cell),
    /// which every word's colour is derived from ([`word_colour`]). `None` — a focused
    /// unit test with no material — uses [`WORD_NEUTRAL`].
    pub base: Option<u32>,
    /// Which face is shaded: the READING side at full contrast, or a QUIET
    /// underside/top.
    pub face: PipeFace,
}

impl<'a> PipeTextCtx<'a> {
    /// The register this cell streams from, resolving the full FALLBACK CHAIN:
    ///
    /// 1. the requested register of the owning node's pool, when it has words;
    /// 2. the node's OTHER register, when the requested one is unauthored (most nodes
    ///    write only a clean `pool`, so a glitched level must not go silent — it
    ///    shows the clean memory-word rather than nothing);
    /// 3. the built-in [`PIPE_WORDS`] list, when the pipe has no pool entry at all
    ///    (a demo level).
    #[inline]
    pub fn words(&self) -> Words<'a> {
        let Some(p) = self.pool else {
            return Words::Builtin;
        };
        let (first, second) = match self.register {
            PoolRegister::Clean => (Words::Pool(p), Words::PoolAlt(p)),
            PoolRegister::Alt => (Words::PoolAlt(p), Words::Pool(p)),
        };
        if !first.is_empty() {
            first
        } else if !second.is_empty() {
            second
        } else {
            Words::Builtin
        }
    }
}

/// The result of overlaying flowing text on a powered-pipe cell: the new glyph + fg,
/// and the glass STYLE (colored underline + the dim backing applied to `bg`). The
/// caller writes `texel.glyph/fg/bg` and merges the style into the cell.
#[derive(Clone, Copy, Debug)]
pub struct PipeTextResult {
    /// The replacement glyph.
    pub glyph: u32,
    /// The replacement foreground colour (packed `0x00RRGGBB`).
    pub fg: u32,
    /// The glass-dimmed backing colour (the caller's existing `bg` darkened).
    pub bg: u32,
    /// The glass underline colour (packed `0x00RRGGBB`) to set via
    /// [`CellStyle::with_underline_color`] — turns the underline attribute on too.
    /// `None` on a QUIET face: a lit rule under every texel of an underside is exactly
    /// the text-wall the quiet face exists to remove.
    pub underline: Option<u32>,
}

/// Multiply a packed `0x00RRGGBB` colour by `f`, clamped per-channel. Local helper so
/// this module needs no `ascii_shader` colour internals. Pure.
#[inline]
fn scale_rgb(rgb: u32, f: f32) -> u32 {
    let r = ((rgb >> 16) & 0xFF) as f32 * f;
    let g = ((rgb >> 8) & 0xFF) as f32 * f;
    let b = (rgb & 0xFF) as f32 * f;
    let c = |x: f32| x.clamp(0.0, 255.0) as u32;
    (c(r) << 16) | (c(g) << 8) | c(b)
}

/// Linear blend of two packed colours, `a·(1−t) + b·t`. Pure.
#[inline]
fn mix_rgb(a: u32, b: u32, t: f32) -> u32 {
    let t = t.clamp(0.0, 1.0);
    let ch = |sh: u32| {
        let x = ((a >> sh) & 0xFF) as f32;
        let y = ((b >> sh) & 0xFF) as f32;
        ((x + (y - x) * t).round().clamp(0.0, 255.0) as u32) << sh
    };
    ch(16) | ch(8) | ch(0)
}

/// Overlay flowing words on a POWERED-pipe cell, streaming the BUILT-IN word list
/// only. Returns the new glyph/fg/bg + the glass underline colour. The caller invokes
/// this only when the pipe is hot (powered), so an unpowered pipe and every non-pipe
/// cell are untouched. Pure, allocation-free.
///
/// This is the anonymous-stream reference entry: it ignores `ctx.pool` entirely. The
/// SHADE PATH calls [`overlay_from_pool`]; `overlay` remains for callers (and tests)
/// that want the register-independent built-in stream.
///
/// `bg` is the cell's current background (darkened here for the glass backing). The fg
/// is the per-word hue, kept bright so the letter reads on top of the dim tube.
pub fn overlay(ctx: &PipeTextCtx, bg: u32) -> PipeTextResult {
    overlay_words(ctx, bg, Words::Builtin)
}

/// [`overlay`] streaming the OWNING NODE'S word pool — clean `pool` or corrupted
/// `pool_alt` per [`PipeTextCtx::register`] — and falling back to the built-in list
/// when the pipe has no pool entry. "Reading the flow IS reading her memory."
///
/// THIS is what the shade path calls. A level with authored pools therefore shows its
/// own memory-words on every powered pipe; a demo level with no pool data streams the
/// built-in list exactly as before.
pub fn overlay_from_pool(ctx: &PipeTextCtx, bg: u32) -> PipeTextResult {
    overlay_words(ctx, bg, ctx.words())
}

/// The shared implementation: stream `words` along the pipe. Both entry points differ
/// only in which register they pass, so there is exactly one streaming code path.
fn overlay_words(ctx: &PipeTextCtx, bg: u32, words: Words<'_>) -> PipeTextResult {
    let sign = view_flow_sign(ctx.state, ctx.view_dir, ctx.tangent_u);
    // `sign == 0.0` means unpowered; the caller gates on `flow_is_hot` so this should
    // not occur, but stay defined: a zero sign just freezes the stream spatially.
    let row = row_of(ctx.wall_v);
    // An EMPTY register would have no stream to walk; fall back rather than divide by
    // its zero length (defensive — `Words::words()` already excludes an empty pool).
    let words = if words.is_empty() {
        Words::Builtin
    } else {
        words
    };
    let pos = stream_pos_len(ctx.along, row, sign, ctx.time, stream_len_of(words));
    let (ch, word) = glyph_at_in(words, pos);

    let hue = word_colour(ctx.base.unwrap_or(WORD_NEUTRAL), word);
    match ctx.face {
        PipeFace::Reading => {
            // The separator reads slightly dimmer than its word's letters so the eye
            // groups letters into tokens (the gap between words is visually quieter).
            let fg = if ch == WORD_SEP {
                scale_rgb(hue, 0.6)
            } else {
                hue
            };
            PipeTextResult {
                glyph: u32::from(ch),
                fg,
                bg: scale_rgb(bg, GLASS_BG_DIM),
                underline: Some(scale_rgb(hue, GLASS_UNDERLINE_LIFT)),
            }
        }
        PipeFace::Quiet => {
            // A quiet face carries NO words: a side seen along its length or far
            // above the eye is not a face anyone reads, and even a sparse
            // speckle of letters there adds up to a lettered ceiling. It keeps
            // the dim glass backing with a faint, world-anchored speck
            // texture (`·`) so the tube still reads as a surface.
            let back = scale_rgb(bg, QUIET_BG_DIM);
            let keep = ch != WORD_SEP && quiet_keeps(ctx.along, ctx.wall_v);
            PipeTextResult {
                glyph: if keep {
                    u32::from(QUIET_SPECK)
                } else {
                    u32::from(' ')
                },
                fg: mix_rgb(back, hue, QUIET_CONTRAST),
                bg: back,
                underline: None,
            }
        }
    }
}

/// Apply a [`PipeTextResult`] to a [`Texel`] in place and return the glass
/// [`CellStyle`] to merge into the cell. The underline colour drives the "glass tube"
/// edge; the dimmed `bg` is the tube backing. Convenience for the single call site.
#[inline]
pub fn apply_to_texel(texel: &mut Texel, res: PipeTextResult) -> CellStyle {
    texel.glyph = res.glyph;
    texel.fg = res.fg;
    texel.bg = res.bg;
    match res.underline {
        Some(u) => CellStyle::NONE.with_underline_color(u),
        None => CellStyle::NONE,
    }
}

/// 2D-coord convenience used by the existing `flow_shift_coord` site if a caller wants
/// the stream position from a sample `coord` instead of the world-along (the matrix
/// flow scrolls `coord.y`). Kept for parity; the shipped path uses `wall_along`. Pure.
#[inline]
pub fn stream_pos_from_coord(coord: Vec2, row: u32, sign: f32, time: f32) -> u32 {
    stream_pos(coord.y, row, sign, time)
}

#[cfg(test)]
mod tests {
    use super::*;
    use re_core::encode_flow_state;

    /// Every glyph the stream produces over a full period is either a real letter of
    /// one of the pool words or the separator — NEVER a duplicated-glyph filler, and
    /// never a char absent from the word it is attributed to.
    #[test]
    fn words_are_real_no_filler() {
        let len = stream_len();
        for pos in 0..len {
            let (ch, word) = glyph_at(pos);
            if ch == WORD_SEP {
                continue;
            }
            assert!(
                word.contains(ch),
                "glyph {ch:?} at pos {pos} is not a real letter of its word {word:?} \
                 (filler leaked into the stream)"
            );
        }
    }

    /// Walking the stream linearly reconstructs the pool words in order, separated by
    /// the separator — i.e. readable words, not noise.
    #[test]
    fn stream_reconstructs_words_in_order() {
        let len = stream_len();
        let mut built = String::new();
        for pos in 0..len {
            built.push(glyph_at(pos).0);
        }
        let expected: String = PIPE_WORDS
            .iter()
            .map(|w| format!("{w}{WORD_SEP}"))
            .collect();
        assert_eq!(built, expected, "stream must spell the pool words in order");
    }

    /// Different words get different hues (per-word colour distinguishes adjacent
    /// tokens). At least two of the pool words must differ in hue.
    #[test]
    fn per_word_color_differs() {
        let base = 0x00FF_D46A;
        let h0 = word_colour(base, PIPE_WORDS[0]);
        let any_diff = PIPE_WORDS.iter().any(|w| word_colour(base, w) != h0);
        assert!(
            any_diff,
            "all pool words hashed to the SAME colour — per-word shade is not working"
        );
        // The same word is always the same colour (stable).
        assert_eq!(word_colour(base, "echo"), word_colour(base, "echo"));
    }

    /// A word's colour is the PIPE'S hue, not a fixed pastel: for a saturated base
    /// every word keeps the base's dominant channel dominant, and two different bases
    /// give two different colours for the same word.
    #[test]
    fn word_colour_is_derived_from_the_pipe_colour() {
        let red = 0x00FF_4020;
        let green = 0x0030_E060;
        for w in PIPE_WORDS {
            let c = word_colour(red, w);
            let (r, g, b) = ((c >> 16) & 0xFF, (c >> 8) & 0xFF, c & 0xFF);
            assert!(r > g && r > b, "word {w:?} on a red pipe drew {c:#08x}");
            assert_ne!(
                word_colour(green, w),
                c,
                "the word colour must follow the pipe"
            );
        }
        // A dim base is lifted to a legible value, hue kept.
        let dim = word_colour(0x0020_1008, "fire");
        assert!(
            (dim >> 16) & 0xFF >= 150,
            "a dim trace must still yield a legible word"
        );
        // No hue at all -> neutral grey family, not a pastel accent.
        let n = word_colour(0, "fire");
        let (r, g, b) = ((n >> 16) & 0xFF, (n >> 8) & 0xFF, n & 0xFF);
        assert!(
            r.abs_diff(g) < 24 && g.abs_diff(b) < 24,
            "hue-less base -> neutral, got {n:#08x}"
        );
    }

    /// A QUIET face (underside/top) is a sparse, low-contrast texture: most texels
    /// carry no letter, no texel carries the glass underline, and a kept letter is far
    /// closer to its backing than a reading-face letter is.
    #[test]
    fn quiet_face_is_sparse_and_low_contrast() {
        let base = 0x00FF_D46A;
        let bg = 0x0030_2010;
        let lum = |c: u32| ((c >> 16) & 0xFF) + ((c >> 8) & 0xFF) + (c & 0xFF);
        let mut inked = 0;
        let mut total = 0;
        for i in 0..400 {
            let along = i as f32 * 0.037;
            for j in 0..10 {
                let mut ctx = ctx_with(None, along, 0.0);
                ctx.wall_v = j as f32 / 10.0;
                ctx.base = Some(base);
                ctx.face = PipeFace::Quiet;
                let q = overlay(&ctx, bg);
                ctx.face = PipeFace::Reading;
                let r = overlay(&ctx, bg);
                assert!(
                    q.underline.is_none(),
                    "a quiet face must not draw the glass rule"
                );
                let dq = lum(q.fg).abs_diff(lum(q.bg));
                let dr = lum(r.fg).abs_diff(lum(r.bg));
                assert!(
                    dq * 2 < dr,
                    "quiet contrast {dq} must be well under reading {dr}"
                );
                total += 1;
                if q.glyph != u32::from(' ') {
                    inked += 1;
                }
            }
        }
        let frac = inked as f32 / total as f32;
        assert!(
            frac > 0.05 && frac < 0.40,
            "quiet face ink fraction {frac} must be a sparse texture, not empty nor a wall"
        );
    }

    /// Undersides, tops and grazing sides are quiet; a side seen near face-on reads.
    #[test]
    fn face_classification() {
        assert_eq!(
            PipeFace::classify(-1.0, 1.0, 0.0),
            PipeFace::Quiet,
            "underside"
        );
        assert_eq!(PipeFace::classify(1.0, 1.0, 0.0), PipeFace::Quiet, "top");
        assert_eq!(
            PipeFace::classify(0.0, 1.0, 0.1),
            PipeFace::Reading,
            "face-on side"
        );
        assert_eq!(
            PipeFace::classify(0.0, 1.8, -0.2),
            PipeFace::Reading,
            "angled side"
        );
        assert_eq!(
            PipeFace::classify(0.0, 4.0, 0.0),
            PipeFace::Quiet,
            "grazing side"
        );
        assert_eq!(
            PipeFace::classify(0.0, 1.0, -0.7),
            PipeFace::Quiet,
            "side overhead"
        );
    }

    /// The quiet mask is fixed to the WORLD surface: the same point keeps or drops its
    /// letter regardless of time, so the texture never flickers while words scroll.
    #[test]
    fn quiet_mask_is_world_anchored() {
        let base = 0x00FF_D46A;
        for i in 0..200 {
            let along = i as f32 * 0.05;
            let kept = |t: f32| {
                let mut ctx = ctx_with(None, along, t);
                ctx.wall_v = 0.4;
                ctx.base = Some(base);
                ctx.face = PipeFace::Quiet;
                overlay(&ctx, 0x0030_2010).glyph != u32::from(' ')
            };
            // A letter may be a separator at one time (never inked), so only assert
            // the mask side: a point the mask drops is blank at EVERY time.
            if !quiet_keeps(along, 0.4) {
                for k in 0..8 {
                    assert!(!kept(k as f32 * 0.3), "masked point inked at t={k}");
                }
            }
        }
    }

    /// Two different rows scroll at different SPEEDS and start at different PHASES, so
    /// at a fixed (along, time) they land on different stream positions and they
    /// diverge over time.
    #[test]
    fn rows_scroll_staggered() {
        let along = 1.0;
        let sign = 1.0;
        // At t=0 the per-row phase offset alone must separate the rows.
        let p0 = stream_pos(along, 0, sign, 0.0);
        let p1 = stream_pos(along, 1, sign, 0.0);
        assert_ne!(p0, p1, "rows must start at different phases (по очереди)");

        // Over a short interval the faster row advances further than the slower one:
        // compare displacement of row 0 vs row 2 against the same start.
        let dt = 0.5;
        let adv0 = displacement(0, dt);
        let adv2 = displacement(2, dt);
        assert!(
            adv2 > adv0,
            "a higher row must scroll FASTER (row2 adv {adv2} <= row0 adv {adv0})"
        );
    }

    /// Helper: signed continuous scroll displacement for a row over `dt` (pre-modulo),
    /// to assert relative speed without aliasing through the wrap.
    fn displacement(row: u32, dt: f32) -> f32 {
        let speed = PIPE_TEXT_SCROLL_RATE * (1.0 + row as f32 * ROW_SPEED_STEP);
        speed * dt
    }

    /// Flipping the view side reverses the flow direction: the same powered pipe read
    /// from the +u side vs the −u side yields opposite signs.
    #[test]
    fn view_side_flips_direction() {
        let state = encode_flow_state(FlowDir::Forward);
        let u = Vec3::new(1.0, 0.0, 0.0);
        let from_plus = view_flow_sign(state, Some(Vec3::new(1.0, 0.0, 0.0)), u);
        let from_minus = view_flow_sign(state, Some(Vec3::new(-1.0, 0.0, 0.0)), u);
        assert_eq!(from_plus, 1.0);
        assert_eq!(from_minus, -1.0);
        assert_eq!(
            from_plus, -from_minus,
            "viewing the pipe from the opposite side must reverse the apparent flow"
        );
    }

    /// An unpowered pipe (state 0) produces no flow sign regardless of view side.
    #[test]
    fn unpowered_pipe_has_no_flow() {
        let u = Vec3::new(1.0, 0.0, 0.0);
        assert_eq!(view_flow_sign(0, Some(u), u), 0.0);
        assert_eq!(view_flow_sign(0, None, u), 0.0);
    }

    /// The glass treatment sets a colored underline (the "behind glass" cue) and dims
    /// the backing bg below the input bg.
    #[test]
    fn glass_sets_underline_and_dims_bg() {
        let ctx = PipeTextCtx {
            state: encode_flow_state(FlowDir::Forward),
            along: 0.0,
            wall_v: 0.5,
            time: 0.0,
            view_dir: Some(Vec3::new(1.0, 0.0, 0.0)),
            tangent_u: Vec3::new(1.0, 0.0, 0.0),
            pool: None,
            register: PoolRegister::Clean,
            base: None,
            face: PipeFace::Reading,
        };
        let bg_in = 0x0040_4040_u32; // mid-grey backing
        let res = overlay(&ctx, bg_in);

        // Underline colour is present (non-zero) → the glass edge will draw.
        assert!(
            res.underline.is_some_and(|u| u != 0),
            "glass underline colour must be set"
        );

        // Backing is dimmer than the input bg (a darker tube behind the text).
        let lum = |c: u32| ((c >> 16) & 0xFF) + ((c >> 8) & 0xFF) + (c & 0xFF);
        assert!(
            lum(res.bg) < lum(bg_in),
            "glass backing bg ({:#08x}) must be dimmer than input bg ({:#08x})",
            res.bg,
            bg_in
        );

        let style = CellStyle::NONE.with_underline_color(res.underline.unwrap_or(0));
        assert!(
            style.is_underline(),
            "the glass style must enable the underline attribute"
        );
    }

    /// A powered pipe's stream moves over time (the words flow): two distinct times at
    /// a fixed cell give different glyphs at least once across a sweep.
    #[test]
    fn powered_stream_flows_over_time() {
        let mk = |t: f32| {
            let ctx = PipeTextCtx {
                state: encode_flow_state(FlowDir::Forward),
                along: 0.0,
                wall_v: 0.2,
                time: t,
                view_dir: Some(Vec3::new(1.0, 0.0, 0.0)),
                tangent_u: Vec3::new(1.0, 0.0, 0.0),
                pool: None,
                register: PoolRegister::Clean,
                base: None,
                face: PipeFace::Reading,
            };
            overlay(&ctx, 0x00_303030).glyph
        };
        let mut seen = std::collections::BTreeSet::new();
        for i in 0..40 {
            seen.insert(mk(i as f32 * 0.1));
        }
        assert!(
            seen.len() > 1,
            "a powered pipe's text stream must change over time (saw {} glyphs)",
            seen.len()
        );
    }

    // ── the WORD-POOL plumbing ──

    fn pool_of(words: &[&str], anchor: usize) -> NodePool {
        NodePool {
            pool: words.iter().map(|s| s.to_string()).collect(),
            pool_alt: Vec::new(),
            anchor,
        }
    }

    fn ctx_with<'a>(pool: Option<&'a NodePool>, along: f32, time: f32) -> PipeTextCtx<'a> {
        PipeTextCtx {
            state: encode_flow_state(FlowDir::Forward),
            along,
            wall_v: 0.1,
            time,
            view_dir: Some(Vec3::new(1.0, 0.0, 0.0)),
            tangent_u: Vec3::new(1.0, 0.0, 0.0),
            pool,
            register: PoolRegister::Clean,
            base: None,
            face: PipeFace::Reading,
        }
    }

    /// [`ctx_with`] in an explicit word register (the glitched-level path).
    fn ctx_reg<'a>(
        pool: Option<&'a NodePool>,
        register: PoolRegister,
        along: f32,
        time: f32,
    ) -> PipeTextCtx<'a> {
        PipeTextCtx {
            register,
            ..ctx_with(pool, along, time)
        }
    }

    /// A node authoring BOTH registers.
    fn pool_both(clean: &[&str], alt: &[&str], anchor: usize) -> NodePool {
        NodePool {
            pool: clean.iter().map(|s| s.to_string()).collect(),
            pool_alt: alt.iter().map(|s| s.to_string()).collect(),
            anchor,
        }
    }

    /// A table entry REACHES `PipeTextCtx`: the ctx reports the node's pool as its
    /// register, in anchor-first order.
    #[test]
    fn pool_entry_reaches_the_ctx_register() {
        let p = pool_of(&["learn", "edge", "seen"], 1);
        let ctx = ctx_with(Some(&p), 0.0, 0.0);
        let words = ctx.words();
        assert!(
            matches!(words, Words::Pool(_)),
            "the ctx must carry the pool"
        );
        assert_eq!(words.len(), 3);
        assert_eq!(words.word(0), Some("edge"), "anchor word leads the stream");
        assert_eq!(words.word(1), Some("learn"));
        assert_eq!(words.word(2), Some("seen"));
    }

    /// No table entry (or an empty pool) ⇒ the built-in register, unchanged.
    #[test]
    fn no_pool_entry_falls_back_to_the_builtin_register() {
        let none = ctx_with(None, 0.0, 0.0);
        assert!(matches!(none.words(), Words::Builtin));
        assert_eq!(none.words().len(), PIPE_WORDS.len());

        let empty = NodePool::EMPTY;
        let ctx = ctx_with(Some(&empty), 0.0, 0.0);
        assert!(
            matches!(ctx.words(), Words::Builtin),
            "an authored-but-empty pool must fall back, not stream nothing"
        );
    }

    /// The pool register STREAMS: walking one period of a pool spells that pool's
    /// words (anchor first), separated by the separator — real words, no filler.
    #[test]
    fn pool_stream_spells_the_pool_words() {
        let p = pool_of(&["river", "cold"], 0);
        let words = Words::Pool(&p);
        let len = stream_len_of(words);
        let built: String = (0..len).map(|pos| glyph_at_in(words, pos).0).collect();
        assert_eq!(built, format!("river{WORD_SEP}cold{WORD_SEP}"));
        // Every emitted letter belongs to the word it is attributed to.
        for pos in 0..len {
            let (ch, w) = glyph_at_in(words, pos);
            if ch != WORD_SEP {
                assert!(w.contains(ch), "glyph {ch:?} is not a letter of {w:?}");
            }
        }
    }

    /// `overlay` (the built-in-only reference entry) IGNORES the pool, in EITHER
    /// register — so a caller that wants the anonymous stream still gets it.
    #[test]
    fn overlay_is_byte_identical_with_and_without_a_pool() {
        let p = pool_of(&["zzz", "qqq"], 0);
        for i in 0..24 {
            let (along, t) = (i as f32 * 0.21, i as f32 * 0.13);
            let a = overlay(&ctx_with(None, along, t), 0x0030_3030);
            let b = overlay(&ctx_with(Some(&p), along, t), 0x0030_3030);
            assert_eq!(
                (a.glyph, a.fg, a.bg, a.underline),
                (b.glyph, b.fg, b.bg, b.underline),
                "`overlay` must not consume the pool yet (golden discipline)"
            );
        }
    }

    /// `overlay_from_pool` DOES consume it — the entry point the pool-words item flips
    /// the shade path to. Proven by finding a pool-only letter that the built-in
    /// register cannot produce.
    #[test]
    fn overlay_from_pool_streams_the_pool_words() {
        // 'z'/'q' appear in no built-in word, so seeing one proves the pool was used.
        let p = pool_of(&["zqzq"], 0);
        let mut saw_pool_letter = false;
        let mut builtin_ever_matched = false;
        for i in 0..64 {
            let (along, t) = (i as f32 * 0.17, i as f32 * 0.07);
            let from_pool = overlay_from_pool(&ctx_with(Some(&p), along, t), 0x0030_3030);
            let ch = char::from_u32(from_pool.glyph).unwrap_or('?');
            if ch == 'z' || ch == 'q' {
                saw_pool_letter = true;
            }
            let builtin = overlay(&ctx_with(Some(&p), along, t), 0x0030_3030);
            if builtin.glyph == from_pool.glyph {
                builtin_ever_matched = true;
            }
        }
        assert!(
            saw_pool_letter,
            "overlay_from_pool must stream the node's own words"
        );
        let _ = builtin_ever_matched; // both registers may coincide on the separator
    }

    /// `overlay_from_pool` with NO pool is exactly `overlay` — the fallback is honest.
    #[test]
    fn overlay_from_pool_without_a_pool_matches_overlay() {
        for i in 0..16 {
            let (along, t) = (i as f32 * 0.3, i as f32 * 0.2);
            let a = overlay(&ctx_with(None, along, t), 0x0020_2020);
            let b = overlay_from_pool(&ctx_with(None, along, t), 0x0020_2020);
            assert_eq!(
                (a.glyph, a.fg, a.bg, a.underline),
                (b.glyph, b.fg, b.bg, b.underline)
            );
        }
    }

    // ── the CLEAN vs ALT register flip ──

    /// The THRESHOLD RULE: the alt register engages only STRICTLY above the bar, and
    /// a non-finite density is treated as no glitch.
    #[test]
    fn glitch_threshold_selects_the_register() {
        assert_eq!(PoolRegister::for_glitch(0.0), PoolRegister::Clean);
        assert_eq!(
            PoolRegister::for_glitch(POOL_ALT_GLITCH_THRESHOLD),
            PoolRegister::Clean,
            "exactly at the bar stays clean (strictly-greater rule)"
        );
        assert_eq!(
            PoolRegister::for_glitch(POOL_ALT_GLITCH_THRESHOLD + 0.001),
            PoolRegister::Alt
        );
        assert_eq!(PoolRegister::for_glitch(1.0), PoolRegister::Alt);
        assert_eq!(PoolRegister::for_glitch(f32::NAN), PoolRegister::Clean);
        assert_eq!(PoolRegister::default(), PoolRegister::Clean);
    }

    /// CLEAN vs ALT at the ctx level: the same node, the same pipe, two registers —
    /// and the alt stream spells the CORRUPTED words.
    #[test]
    fn alt_register_streams_the_corrupted_pool() {
        let p = pool_both(&["seen", "named"], &["s33n", "n4m3d"], 0);

        let clean = ctx_reg(Some(&p), PoolRegister::Clean, 0.0, 0.0);
        assert!(matches!(clean.words(), Words::Pool(_)));
        assert_eq!(clean.words().word(0), Some("seen"));

        let alt = ctx_reg(Some(&p), PoolRegister::Alt, 0.0, 0.0);
        assert!(matches!(alt.words(), Words::PoolAlt(_)));
        assert_eq!(alt.words().len(), 2);
        // Index-aligned lists ⇒ the anchor's CORRUPTED counterpart leads.
        assert_eq!(alt.words().word(0), Some("s33n"));
        assert_eq!(alt.words().word(1), Some("n4m3d"));

        // And the streamed glyphs actually differ between the registers.
        let mut differed = false;
        for i in 0..64 {
            let (along, t) = (i as f32 * 0.17, i as f32 * 0.07);
            let a = overlay_from_pool(
                &ctx_reg(Some(&p), PoolRegister::Clean, along, t),
                0x0020_2020,
            );
            let b = overlay_from_pool(&ctx_reg(Some(&p), PoolRegister::Alt, along, t), 0x0020_2020);
            differed |= a.glyph != b.glyph;
        }
        assert!(differed, "the alt register must render different glyphs");
    }

    /// The alt anchor rule, both branches: index-aligned lists lead with the anchor's
    /// counterpart; a DIFFERENT-length alt list streams in authored order (there is no
    /// counterpart to lead with, and inventing one would misattribute the anchor).
    #[test]
    fn alt_order_is_anchor_first_only_when_index_aligned() {
        let aligned = pool_both(&["a", "b", "c"], &["A", "B", "C"], 2);
        assert_eq!(
            alt_ordered_word(&aligned, 0),
            Some("C"),
            "anchor counterpart"
        );
        assert_eq!(alt_ordered_word(&aligned, 1), Some("A"));
        assert_eq!(alt_ordered_word(&aligned, 2), Some("B"));
        assert_eq!(alt_ordered_word(&aligned, 3), None);

        let ragged = pool_both(&["a", "b", "c"], &["X", "Y"], 2);
        assert_eq!(alt_ordered_word(&ragged, 0), Some("X"), "authored order");
        assert_eq!(alt_ordered_word(&ragged, 1), Some("Y"));
        assert_eq!(alt_ordered_word(&ragged, 2), None);

        // No alt words at all ⇒ nothing to read.
        assert_eq!(alt_ordered_word(&pool_of(&["a"], 0), 0), None);
    }

    /// FALLBACK CHAIN: a glitched level whose node authored NO `pool_alt` shows the
    /// clean memory-word (never silence); a pipe with no pool entry at all shows the
    /// built-in list in either register.
    #[test]
    fn alt_register_falls_back_to_clean_then_builtin() {
        let clean_only = pool_of(&["river"], 0);
        let ctx = ctx_reg(Some(&clean_only), PoolRegister::Alt, 0.0, 0.0);
        assert!(
            matches!(ctx.words(), Words::Pool(_)),
            "an unauthored pool_alt must fall back to the CLEAN pool, not go blank"
        );
        assert_eq!(ctx.words().word(0), Some("river"));

        let empty = NodePool::EMPTY;
        for reg in [PoolRegister::Clean, PoolRegister::Alt] {
            assert!(matches!(
                ctx_reg(Some(&empty), reg, 0.0, 0.0).words(),
                Words::Builtin
            ));
            assert!(matches!(
                ctx_reg(None, reg, 0.0, 0.0).words(),
                Words::Builtin
            ));
        }
    }

    /// An ALT-only node (no clean words) still streams: the alt register carries it in
    /// BOTH register requests, so the words never vanish.
    #[test]
    fn alt_only_node_streams_its_alt_words() {
        let alt_only = pool_both(&[], &["ghost"], 0);
        for reg in [PoolRegister::Clean, PoolRegister::Alt] {
            let ctx = ctx_reg(Some(&alt_only), reg, 0.0, 0.0);
            assert!(matches!(ctx.words(), Words::PoolAlt(_)), "register {reg:?}");
            assert_eq!(ctx.words().word(0), Some("ghost"));
        }
    }

    /// The alt register goes through the SAME stream walker: one period spells its
    /// words, separated by the separator, with no filler.
    #[test]
    fn alt_stream_spells_the_alt_words() {
        let p = pool_both(&["one", "two"], &["0n3", "tw0"], 0);
        let words = Words::PoolAlt(&p);
        let len = stream_len_of(words);
        let built: String = (0..len).map(|pos| glyph_at_in(words, pos).0).collect();
        assert_eq!(built, format!("0n3{WORD_SEP}tw0{WORD_SEP}"));
    }
}
