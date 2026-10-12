//! The substrate's own cell alphabet — the buffer's wall texture.
//!
//! Four glyphs, one per state a cell of the substrate can be in:
//!
//! | glyph | name | ink | meaning |
//! |---|---|---|---|
//! | `░` | [`UNCLAIMED`] | 25% | nobody has taken this cell |
//! | `█` | [`HELD`] | 100% | claimed and still held |
//! | `▒` | [`SCAR`] | 50% | released — half eroded, the hold still legible |
//! | `✕` | [`BURNED`] | ~18% | released past the point where anything is legible |
//!
//! Three of the four are the Unicode shade ramp, and that is the point: the tiers are
//! read by INK COVERAGE, not by colour. `█ → ▒` is the erosion the `held · released`
//! threshold encodes, made visible; `░` is the never-touched ground beneath both; and
//! `✕` is deliberately OFF the ramp, because burning is the one state that is a MARK
//! rather than a density. See [`ink_coverage`] and the finding recorded there.
//!
//! Two scalars drive the field. `held` is the fraction of the whole field that has
//! ever been claimed; `released` is the fraction **of that held set** that has since
//! been let go. The states are four NESTED tiers — `✕ ⊆ ▒ ⊆ █` — and every threshold
//! is a fraction of the one above it:
//!
//! ```text
//! rank < held · released · burn_fraction(released)  ->  ✕
//! rank < held · released                            ->  ▒
//! rank < held                                       ->  █
//! otherwise                                         ->  ░
//! ```
//!
//! That nesting is the whole design, and it is what two earlier drafts got wrong —
//! each at a different tier, each in the same way:
//!
//! 1. **The scar threshold is `held · released`, not `released`.** Releasing takes a
//!    FRACTION OF THE HELD SET, so at `held = 0.5, released = 0.5` half the held
//!    cells scar and half stay `█` — an EROSION. Thresholding on `released` alone
//!    makes the two thresholds coincide at parity: the entire held set converts in
//!    one step (a flip, not an erosion) and `█` is unreachable whenever
//!    `released >= held`.
//! 2. **Burning is a fraction of the SCAR set, not a re-lettering of it.** The first
//!    implementation flipped every scar to `✕` the instant `released` crossed
//!    [`BURN_RELEASE`] — the same mistake one tier down, and it made a wall showing
//!    scarred AND burned cells together unreachable at every setting.
//!
//! `releasing_erodes_the_held_set_it_does_not_flip_it` and
//! `burning_erodes_the_scar_set_it_does_not_relabel_it` measure the two, and
//! `the_four_tiers_nest_per_cell` pins the containment they share.
//!
//! **World-anchored, not screen-anchored.** A cell's state is a hash of its FLOORED
//! world coordinate, so the same wall cell reads the same glyph from any camera pose.
//! If it were keyed on anything view-relative the pattern would swim as the player
//! walked — the same discipline `ascii_shader::Material::texture_sample` already
//! follows for every other recipe.
//!
//! Pure compute: no terminal, no allocation, no `unsafe`, no clock. Two integer
//! hashes and three comparisons per cell.

use glam::{Vec2, Vec3};

use crate::ascii_shader::{hash01, hash2i, hash3i};

// ───────────────────────────── the alphabet ─────────────────────────────────

/// Unclaimed substrate — the resting field. A light shade block, so an untouched
/// wall reads as texture rather than as ink.
pub const UNCLAIMED: char = '░';
/// A held cell — claimed, and still claimed. The solid block: the densest glyph in
/// the alphabet, so a held region reads as mass.
pub const HELD: char = '█';
/// A scar — the cell was held and has been released.
///
/// Half of it is gone; the half that remains is what makes the hold still legible.
///
/// The MEDIUM shade, chosen by rendering three candidates and looking at the frames.
/// The first draft used `·` (a middle dot), which measured correctly on every colour
/// axis and still read as a HOLE PUNCHED IN THE WALL — see [`ink_coverage`] for the
/// arithmetic that explains why, and why no amount of brightness could have fixed it.
pub const SCAR: char = '▒';
/// A burned cell — released so completely that the hold is no longer legible.
///
/// The `✕` is a cancellation, not a mark. Deliberately OFF the shade ramp the other
/// three sit on: burning is the one state that is not a density.
pub const BURNED: char = '✕';

/// Every glyph [`cell_glyph`] can return, in state order. The output contract: no
/// input shape of any kind produces a glyph outside this set.
pub const ALPHABET: [char; 4] = [UNCLAIMED, HELD, SCAR, BURNED];

/// Roughly what fraction of its cell each alphabet glyph inks.
///
/// # Why this exists
///
/// A terminal cell's apparent brightness is `coverage · fg + (1 − coverage) · bg`, so
/// the GLYPH is the dominant brightness channel and the colour is a modifier on it.
/// The first version of this alphabet got the colour tiers strictly ordered — `█`
/// brightest, `✕` darkest, measured and asserted — and the wall still read wrong,
/// because `·` inks about 4% of its cell against `░`'s 25%. The scar came out as the
/// DARKEST thing on the wall, below even the burn, and a field of scars read as a gap
/// in the geometry rather than as a marked surface. Raising the scar's colour could
/// not have fixed it: closing a 6× coverage deficit on brightness alone would have put
/// the scar past `█` and inverted the tier order.
///
/// So coverage is a first-class property of the alphabet and is stated here, and
/// `ascii_shader`'s `substrate_tiers_are_monotone_in_brightness` asserts the
/// ordering of the quantity a viewer actually sees rather than of `fg` alone.
///
/// The three shade blocks are EXACT — Unicode defines U+2591/2592/2593 as light,
/// medium and dark shade at 25/50/75%. `✕` is a stroke glyph and therefore
/// font-dependent; `0.18` is an estimate for a typical monospace face, and every
/// assertion built on it has margin far wider than the plausible range.
///
/// # Keyed on the CHARACTER, deliberately
///
/// The arms below match glyph LITERALS, not [`HELD`] / [`SCAR`] / … — because coverage
/// is a fact about the character, not about the role it has been given. Keying on the
/// semantic constants is self-referential and was tried first: with `SCAR => 0.50` the
/// table reported 50% coverage for whatever `SCAR` happened to be, so swapping the
/// scar back to `·` changed nothing it could see and the guard built on it passed the
/// very mutation it exists to catch. The two REJECTED scar candidates (`·` and `▓`)
/// are listed too, so that guard can judge any future substitution rather than only
/// the ones shipping today.
#[must_use]
#[inline]
pub const fn ink_coverage(glyph: char) -> f32 {
    match glyph {
        '█' => 1.00, // U+2588 FULL BLOCK
        '▓' => 0.75, // U+2593 DARK SHADE   — rejected scar candidate C
        '▒' => 0.50, // U+2592 MEDIUM SHADE — the shipped SCAR
        '░' => 0.25, // U+2591 LIGHT SHADE
        '✕' => 0.18, // U+2715 MULTIPLICATION X — a stroke glyph, estimated
        '·' => 0.04, // U+00B7 MIDDLE DOT  — rejected scar candidate B (the hole)
        ' ' => 0.00, // a blank cell inks nothing at all
        // Anything else: assume a mid stroke glyph rather than 0, so a caller that
        // reaches here gets a usable number instead of a phantom hole.
        _ => 0.20,
    }
}

// ─────────────────────────────── tunables ───────────────────────────────────

/// Where the burn RAMP starts: below this `released` nothing burns at all.
///
/// Burning is the last stretch of the release ramp, not a switch at the end of it.
/// Between here and `released = 1` the burn tier grows continuously out of the scar
/// tier, so a wall can show `▒` and `✕` together — see [`burn_fraction`].
pub const BURN_RELEASE: f32 = 0.95;

/// Width of the burn ramp — the interval of `released` over which the scar set is
/// consumed by the burn set. Positive by construction (asserted below), so the
/// division in [`burn_fraction`] cannot blow up.
const BURN_RAMP: f32 = 1.0 - BURN_RELEASE;
const _: () = assert!(BURN_RAMP > 0.0, "the burn ramp must have positive width");

/// How many alphabet cells span one world unit on a wall surface.
///
/// The alphabet is a GRID the eye can count, so one cell must be a small fraction of
/// a wall face, not the whole face — at 1.0 an entire unit wall would carry a single
/// glyph. Read by `ascii_shader`'s `Substrate` recipe, not by [`cell_glyph`] (which
/// floors exactly what it is handed, so its contract stays literal and the density
/// stays an authoring knob).
pub const CELLS_PER_UNIT: f32 = 3.0;

/// Salt folded into the seed hash. Keeps a substrate field from correlating with any
/// other world-lattice hash in the crate that happens to share a material seed.
const SEED_SALT: i32 = 0x5EED_0A17_u32 as i32;

// ──────────────────────────────── the rule ──────────────────────────────────

/// Floor a world axis to its integer cell index.
///
/// `floor` FIRST, then cast: `-0.5 as i32` truncates toward zero and would fold cells
/// `-1` and `0` together, mirroring the whole field about the origin. `as` on a
/// non-finite float is defined in Rust (NaN → 0, ±∞ → the i32 bounds), so this is
/// total — no input can panic here.
#[inline]
const fn cell_index(v: f32) -> i32 {
    v.floor() as i32
}

/// Clamp a `[0,1]` channel, reading a non-finite value as `0` — "at rest".
///
/// `f32::clamp` PROPAGATES NaN, and a NaN threshold makes every comparison in
/// [`glyph_for_rank`] false at once, so it must be filtered here rather than relied on
/// downstream. At rest is the safest of the four readings: a broken channel shows an
/// untouched wall, never a burned one.
#[inline]
const fn norm01(v: f32) -> f32 {
    if v.is_nan() {
        0.0
    } else {
        v.clamp(0.0, 1.0)
    }
}

/// This world cell's stable rank in `[0,1)` — its fixed place in the conversion order.
///
/// A cell with a LOW rank is claimed early and released early; a cell with a high rank
/// is claimed last. That single number is what makes the field NESTED: a rising `held`
/// can only ever add cells, never take one back.
///
/// The seed is folded through a second mix round rather than being handed to
/// [`hash3i`] as a fourth coordinate (it has only three), so a change of seed
/// re-deals the entire field instead of translating it — a seed that merely shifted
/// the lattice would be decorative.
#[must_use]
#[inline]
pub fn cell_rank(world: Vec3, seed: u32) -> f32 {
    let cell = hash3i(
        cell_index(world.x),
        cell_index(world.y),
        cell_index(world.z),
    );
    hash01(hash2i(cell as i32, (seed as i32) ^ SEED_SALT))
}

/// How far along the burn ramp `released` sits, in `[0,1]`.
///
/// `0` at and below [`BURN_RELEASE`] (nothing burns), `1` at `released = 1` (every
/// scar burns), and a continuous ramp between — which is the whole point: burning is
/// a FRACTION OF THE SCAR SET, exactly as scarring is a fraction of the held set.
///
/// The first draft of this module made burning a re-lettering instead: every scar
/// flipped to `✕` the moment `released` crossed [`BURN_RELEASE`]. That abandoned the
/// nesting discipline one tier down, and it made a wall showing `▒` AND `✕` together
/// unreachable at every setting — the interval this function ramps across was empty.
#[must_use]
#[inline]
pub fn burn_fraction(released: f32) -> f32 {
    ((norm01(released) - BURN_RELEASE) / BURN_RAMP).clamp(0.0, 1.0)
}

/// The glyph for a cell of the given `rank`, under `held` / `released`.
///
/// Four NESTED tiers, stated most-converted first — each threshold is a fraction of
/// the one above it, so `✕ ⊆ ▒ ⊆ █` holds all the way down:
///
/// | condition | glyph | tier |
/// |---|---|---|
/// | `rank < held · released · burn_fraction(released)` | [`BURNED`] | burned |
/// | `rank < held · released` | [`SCAR`] | released |
/// | `rank < held` | [`HELD`] | claimed |
/// | otherwise | [`UNCLAIMED`] | at rest |
///
/// Each threshold is `≤` the one below it (every factor lies in `[0,1]`), so the
/// `if` / `else if` chain below is the table and nothing else — and the nesting is
/// arithmetic, not something the tests merely hope for.
///
/// Split out from [`cell_glyph`] so a caller that already holds the rank (the wall
/// recipe wants it for tone) pays for one hash, not two.
#[must_use]
#[inline]
pub fn glyph_for_rank(rank: f32, held: f32, released: f32) -> char {
    let held = norm01(held);
    let released = norm01(released);
    let scarred = held * released;
    if rank < scarred * burn_fraction(released) {
        BURNED
    } else if rank < scarred {
        SCAR
    } else if rank < held {
        HELD
    } else {
        UNCLAIMED
    }
}

/// The cell alphabet glyph for the substrate cell containing `world`.
///
/// Total: every input shape — zero, negative, NaN or infinite coordinates, channels
/// outside `[0,1]`, any seed — returns one of [`ALPHABET`]. Never panics, never
/// allocates.
#[must_use]
#[inline]
pub fn cell_glyph(world: Vec3, held: f32, released: f32, seed: u32) -> char {
    glyph_for_rank(cell_rank(world, seed), held, released)
}

// ───────────────────── the wall-surface adapter (ascii_shader) ──────────────

/// The floor under an authored `scale`.
///
/// A non-positive or non-finite density would collapse every surface coordinate onto
/// cell 0 (one glyph for the whole level); this keeps the grid a grid whatever a
/// `.lvl` asks for.
pub const MIN_CELLS_PER_UNIT: f32 = 0.05;

/// Ink tone for [`UNCLAIMED`], in `[0,1]`.
///
/// The four tones drive the wall recipe's MONOTONE ink ramp: the alphabet must read as
/// four TIERS even at a distance where the glyph itself has gone to mush, so they are
/// spread wide and strictly ordered — a held cell is the brightest thing on the wall,
/// a burn the darkest. The order is a contract, pinned by
/// `ascii_shader`'s `substrate_tiers_are_monotone_in_brightness`; reordering these
/// constants without reordering that test's expectation is a defect.
pub const TONE_UNCLAIMED: f32 = 0.30;
/// See [`TONE_UNCLAIMED`].
pub const TONE_HELD: f32 = 0.92;
/// See [`TONE_UNCLAIMED`].
pub const TONE_SCAR: f32 = 0.52;
/// See [`TONE_UNCLAIMED`].
pub const TONE_BURNED: f32 = 0.10;
/// Peak-to-peak per-cell tone jitter.
///
/// Small: enough to keep a large single-state region from reading as a flat paint
/// chip, far too small to blur the four tiers into each other (the tightest tier gap
/// is `TONE_SCAR - TONE_UNCLAIMED = 0.22`).
pub const TONE_JITTER: f32 = 0.06;

/// The rank of the alphabet cell under a WALL SURFACE coordinate.
///
/// `coord` is the shader's world-anchored surface coordinate — `coord.x` is the wall
/// `u` (the caster's `tex_u` plus the face's CONSTANT world axis as an integer
/// offset, see `crate::face_span`) and `coord.y` is the world height. All three world
/// axes are therefore already represented in the pair, which is why the third axis is
/// handed to [`cell_rank`] as a constant: a wall's cells are indexed by where they
/// sit ON the wall, not by the wall's thickness.
///
/// `scale` converts world units to alphabet cells — the density knob, applied HERE
/// rather than inside [`cell_rank`] so that function's contract stays literal ("floor
/// the coordinate you were handed").
#[must_use]
#[inline]
pub fn surface_rank(coord: Vec2, scale: f32, seed: u32) -> f32 {
    let s = if scale.is_nan() {
        MIN_CELLS_PER_UNIT
    } else {
        scale.max(MIN_CELLS_PER_UNIT)
    };
    cell_rank(Vec3::new(coord.x * s, coord.y * s, 0.0), seed)
}

/// The ink tone for a state, with a small per-cell jitter so a large single-state
/// region does not read as flat paint.
///
/// The jitter is decorrelated from `rank` by a fractional multiply rather than used
/// raw: within one state the rank is confined to that state's sub-interval, so a raw
/// `rank` jitter would be a systematic RAMP across the state (and its span would grow
/// and shrink with `held`) instead of per-cell noise. One multiply, no extra hash.
#[must_use]
#[inline]
pub fn tone(glyph: char, rank: f32) -> f32 {
    /// Multiplier that scatters the within-state rank interval across `[0,1)`.
    const DECORRELATE: f32 = 31.7;
    let base = match glyph {
        HELD => TONE_HELD,
        SCAR => TONE_SCAR,
        BURNED => TONE_BURNED,
        // `UNCLAIMED` and — defensively — anything else: the resting tone.
        _ => TONE_UNCLAIMED,
    };
    let j = (rank * DECORRELATE).fract();
    (base + TONE_JITTER * (j - 0.5)).clamp(0.0, 1.0)
}

// ─────────────────── the VOICE materials' state-bus mark ────────────────────
//
// Two catalog materials render this alphabet from PUZZLE STATE rather than from two
// authored scalars. Both read ONE scalar — the level's monotone solve-progress MARK
// — and spend it on a different channel:
//
//   mnemos_hold    held = mark, released = 0     ░ converts to █ as progress rises
//   lethe_release  held = 1,    released = mark  █ scars to ▒ across the whole field
//
// Everything about the IRREVERSIBILITY lives in the mark, not here: these are pure
// functions of it, exactly like every other reading of `cell_glyph`. The high-water
// `max()` and its clear-on-reset are level-session state and live in `re_game`.
//
// ## Why the mark rides in a `u32`
//
// The state bus (`re_core::CellStateMap`) carries exactly two values per tag:
// `state: u32` and `pulse: f32`. `pulse` is documented as a transient KICK — a tag
// that just fired, a switch that just toggled — which the bridge zeroes on the next
// rebuild. A level that must be HELD cannot ride a channel whose contract is that it
// decays, so the mark goes in the `u32`, quantised as parts per thousand.
//
// ## The alias, stated plainly
//
// `state` is used everywhere else to INDEX a `State` glyph family (powered / dead /
// decoy / won) or to carry a pipe's packed flow word. Here the same word is a SCALAR.
// Two readings of one channel are safe only because they never meet:
//
//   * on a MATERIAL — `ascii_shader`'s `the_voice_materials_carry_no_other_state_reader`
//     asserts that no material carrying the `Voice` recipe also carries a `State` glyph
//     family, an `OnState` animation track, or a state-reading `SurfaceFx`. Asserted,
//     not assumed: an invariant that depends on nobody later attaching a glyph family
//     is exactly the kind that decays silently.
//   * on a TAG — the mark is published under [`MNEMOS_TAG`] / [`LETHE_TAG`], reserved
//     at the very top of the non-pipe tag namespace, far above the small contiguous
//     node/colour ids and below `re_core::PIPE_TAG_FLAG`.
//
// And where they DO meet — a voice cell that was mis-tagged and reads a family index
// or a flow word instead — the failure is benign by construction: those words are all
// tiny (0..=7), so [`decode_mark`] returns ≈0 and the wall reads AT REST. A mis-wired
// voice wall shows an untouched surface, never a completed one.
// (`an_aliased_family_or_flow_word_reads_as_at_rest` measures it.)

/// Quantisation of the mark carried in the state-bus `u32`: PARTS PER THOUSAND.
///
/// A thousand steps is finer than the alphabet can show — a wall is a few hundred
/// cells at most, so one part in a thousand is well below one cell of conversion —
/// while leaving the word's remaining range (a `u32` holds 4 million times this)
/// unused and obviously not a family index.
pub const MARK_PARTS: u32 = 1000;

/// The reserved state-bus tag a `mnemos_hold` cell reads its mark from.
///
/// Placed at the top of the NON-PIPE half of the `u16` tag space: puzzle node tags
/// and 1-based colour-material ids both grow from `0` upward and a v0.1 level has a
/// handful of each, so nothing reaches here; `re_core::PIPE_TAG_FLAG` (bit 15) keeps
/// the pipe namespace above. The two asserts below are the machine-checked half of
/// that argument.
///
/// It costs nothing in the dense [`re_core::CellStateMap`]: any level with a pipe
/// already writes tag `0x8000`, which sizes the backing vector past this.
pub const MNEMOS_TAG: u16 = 0x7FFD;
/// The reserved state-bus tag a `lethe_release` cell reads its mark from. See
/// [`MNEMOS_TAG`] for the namespace argument.
pub const LETHE_TAG: u16 = 0x7FFE;

const _: () = assert!(
    MNEMOS_TAG & re_core::PIPE_TAG_FLAG == 0 && LETHE_TAG & re_core::PIPE_TAG_FLAG == 0,
    "a voice tag must sit OUTSIDE the pipe namespace"
);
const _: () = assert!(MNEMOS_TAG != LETHE_TAG, "the two voices need distinct tags");

/// Is `tag` one of the two reserved voice tags?
///
/// Read by the shade path's tag resolution: a voice cell normally takes its tag from
/// its MATERIAL (the mark is level-global, so there is nothing per-cell about it), but
/// a level that stamps a voice tag into a cell's `side_data` explicitly keeps it — that
/// is how an author cross-wires one wall to the OTHER voice's mark. Anything else on a
/// voice cell (a colour id, a stray `0`) is not a mark and is ignored in favour of the
/// material's own tag.
#[must_use]
#[inline]
pub const fn is_voice_tag(tag: u16) -> bool {
    tag == MNEMOS_TAG || tag == LETHE_TAG
}

/// Quantise a `[0,1]` progress into the state-bus word, in [`MARK_PARTS`].
///
/// Total: NaN and anything below `0` encode to `0` (at rest — the safe reading, see
/// [`norm01`]); anything at or above `1` encodes to [`MARK_PARTS`]. `re_game`'s
/// bridge is the only production caller.
#[must_use]
#[inline]
pub fn encode_mark(progress: f32) -> u32 {
    // `round`, not truncate: a progress of exactly 1 must reach MARK_PARTS, and a
    // half-step should land on the nearer part rather than always down.
    (norm01(progress) * MARK_PARTS as f32).round() as u32
}

/// Read a `[0,1]` mark back out of a state-bus word — the inverse of
/// [`encode_mark`], SATURATING rather than wrapping.
///
/// Saturation is the load-bearing part: any word this function was not meant to see
/// (a family index, a pipe flow word, an out-of-range write) still yields a mark in
/// `[0,1]`, so no input can drive the alphabet outside its four tiers.
#[must_use]
#[inline]
pub fn decode_mark(word: u32) -> f32 {
    if word >= MARK_PARTS {
        1.0
    } else {
        word as f32 / MARK_PARTS as f32
    }
}

/// Which of the two channels a voice material spends its mark on.
///
/// The whole difference between the two voices. `mnemos_hold` spends the mark on
/// `held` over an empty field — the wall FILLS, and a cell that has filled stays
/// filled. `lethe_release` starts from a whole field and spends the mark on
/// `released` — the wall ERODES, and a scar never heals. Same alphabet, same
/// function, opposite direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoiceMode {
    /// `held = mark`, `released = 0` — an empty field converting to [`HELD`].
    Hold,
    /// `held = 1`, `released = mark` — a whole field eroding to [`SCAR`] (and, over
    /// the last stretch of the ramp past [`BURN_RELEASE`], to [`BURNED`]).
    Release,
}

impl VoiceMode {
    /// The reserved state-bus tag this voice's cells read their mark from.
    #[must_use]
    #[inline]
    pub const fn tag(self) -> u16 {
        match self {
            Self::Hold => MNEMOS_TAG,
            Self::Release => LETHE_TAG,
        }
    }

    /// The `(held, released)` pair this voice hands [`glyph_for_rank`] at `mark`.
    #[must_use]
    #[inline]
    pub const fn channels(self, mark: f32) -> (f32, f32) {
        let m = norm01(mark);
        match self {
            Self::Hold => (m, 0.0),
            Self::Release => (1.0, m),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ───────────────────────────── probe tunables ────────────────────────────

    /// Side of the square probe patch. 32 ⇒ 1024 cells, so a proportion measured on
    /// it has a binomial standard deviation of ~1.6% at `p = 0.5` and the bands below
    /// can be tight enough to MEAN something. (The plan drafted 20×20 = 400, whose
    /// 2.5% sd forces bands wide enough to admit a wrong rule.)
    const PATCH: i32 = 32;
    /// Cells in one probe patch.
    const PATCH_CELLS: usize = (PATCH * PATCH) as usize;
    /// The probe seed. Deliberately small and unremarkable — the seed sweep below
    /// checks that nothing about the field depends on it being special.
    const PROBE_SEED: u32 = 3;
    /// How far a measured proportion may sit from its predicted value, as a fraction
    /// of the patch. ~2.5 binomial sd at `p = 0.5` over [`PATCH_CELLS`] cells: wide
    /// enough that the hash's own lumpiness cannot fail it, narrow enough that every
    /// wrong threshold in the table lands outside it by tens of percent.
    const PROPORTION_TOL: f32 = 0.04;

    /// The probe patch's world cells: `PATCH × PATCH` of them on a fixed z plane, at
    /// cell CENTRES (`+0.5`) so no sample sits on a lattice boundary where a float
    /// wobble could flip which cell it belongs to.
    fn patch_cells() -> impl Iterator<Item = Vec3> {
        (0..PATCH_CELLS).map(|i| {
            Vec3::new(
                (i as i32 % PATCH) as f32 + 0.5,
                (i as i32 / PATCH) as f32 + 0.5,
                1.5,
            )
        })
    }

    /// The whole probe patch's glyphs at `(held, released)`.
    fn field(held: f32, released: f32, seed: u32) -> Vec<char> {
        patch_cells()
            .map(|w| cell_glyph(w, held, released, seed))
            .collect()
    }

    /// Counts of `[░, █, ▒, ✕]` over a field. Asserts on the way that no fifth glyph
    /// exists, so the census can never silently drop an out-of-alphabet return.
    fn census(f: &[char]) -> [usize; 4] {
        let mut c = [0usize; 4];
        for g in f {
            let i = ALPHABET
                .iter()
                .position(|a| a == g)
                .unwrap_or_else(|| panic!("{g:?} is not in the cell alphabet"));
            c[i] += 1;
        }
        c
    }

    /// The set of cells that are CLAIMED (anything but unclaimed), as a bit per cell.
    fn claimed_set(f: &[char]) -> Vec<bool> {
        f.iter().map(|g| *g != UNCLAIMED).collect()
    }

    /// The set of cells that have been RELEASED (scarred or burned).
    fn released_set(f: &[char]) -> Vec<bool> {
        f.iter().map(|g| *g == SCAR || *g == BURNED).collect()
    }

    /// The set of cells that have BURNED — the innermost tier.
    fn burned_set(f: &[char]) -> Vec<bool> {
        f.iter().map(|g| *g == BURNED).collect()
    }

    /// Is `a` a subset of `b`? (Every set bit in `a` is set in `b`.)
    fn is_subset(a: &[bool], b: &[bool]) -> bool {
        a.iter().zip(b).all(|(x, y)| !*x || *y)
    }

    /// A measured count as a fraction of the patch.
    fn frac(n: usize) -> f32 {
        n as f32 / PATCH_CELLS as f32
    }

    /// Assert a measured count sits within [`PROPORTION_TOL`] of `want`.
    #[track_caller]
    fn assert_frac(got: usize, want: f32, what: &str) {
        let f = frac(got);
        assert!(
            (f - want).abs() <= PROPORTION_TOL,
            "{what}: measured {f:.4} ({got}/{PATCH_CELLS}), expected {want:.4} ± {PROPORTION_TOL}"
        );
    }

    // ─────────────────────────── world anchoring ─────────────────────────────

    /// World-anchored, not screen-anchored: the same wall cell shows the same glyph
    /// from any camera pose, or the pattern would swim as the player walks.
    ///
    /// Four separate claims, because "anchored" is four properties and an
    /// implementation can satisfy any three of them while being useless:
    /// 1. it is a pure function (the same input twice gives the same glyph);
    /// 2. it VARIES across the field — a `cell_glyph` that ignored `world` entirely
    ///    would pass claim 1 and every distribution claim below that only counts;
    /// 3. it is constant WITHIN a world cell (sub-cell camera creep cannot flip it)
    ///    and different BETWEEN neighbouring cells;
    /// 4. it FLOORS rather than truncates, so the field is not mirrored at the origin.
    #[test]
    fn the_cell_alphabet_is_anchored_to_the_world() {
        // 1. Pure.
        let once = cell_glyph(Vec3::new(4.25, 7.5, 2.0), 0.0, 0.0, 7);
        let again = cell_glyph(Vec3::new(4.25, 7.5, 2.0), 0.0, 0.0, 7);
        assert_eq!(once, again, "the same world cell must always read the same");

        // 2. It VARIES. At half held the field must be a MIX, not one glyph — this is
        //    the claim that kills a `cell_glyph` which never reads `world`.
        let mixed = field(0.5, 0.0, PROBE_SEED);
        let [unclaimed, held, scar, burn] = census(&mixed);
        assert!(
            unclaimed > 0 && held > 0,
            "at held = 0.5 the field must be a MIX of held and unclaimed, got \
             ░{unclaimed} █{held} ▒{scar} ✕{burn} — the glyph does not depend on the \
             world coordinate"
        );

        // 3. Constant within a cell, different between cells. The whole point of
        //    flooring: sub-cell camera creep must not re-deal the wall.
        for i in 0..64 {
            let (bx, by) = ((i % 8) as f32, (i / 8) as f32);
            let inside = [
                Vec3::new(bx, by, 1.0),
                Vec3::new(bx + 0.01, by + 0.5, 1.5),
                Vec3::new(bx + 0.999, by + 0.999, 1.999),
            ];
            let want = cell_glyph(inside[0], 0.5, 0.25, PROBE_SEED);
            for probe in inside {
                assert_eq!(
                    cell_glyph(probe, 0.5, 0.25, PROBE_SEED),
                    want,
                    "{probe:?} sits in the same world cell as {:?} and must read the same",
                    inside[0]
                );
            }
        }
        // …and neighbouring cells are independent: over the patch, a cell and its
        // +x neighbour differ on a substantial fraction. (A field that were constant
        // per ROW, say, would pass claim 3 and fail here.)
        let neighbour_differs = (0..PATCH_CELLS)
            .filter(|i| {
                let (col, row) = ((*i as i32 % PATCH) as f32, (*i as i32 / PATCH) as f32);
                cell_glyph(Vec3::new(col + 0.5, row + 0.5, 1.5), 0.5, 0.0, PROBE_SEED)
                    != cell_glyph(Vec3::new(col + 1.5, row + 0.5, 1.5), 0.5, 0.0, PROBE_SEED)
            })
            .count();
        assert!(
            frac(neighbour_differs) > 0.30,
            "only {neighbour_differs}/{PATCH_CELLS} cells differ from their +x neighbour \
             — the field is banded, not per-cell (expected ~50% at held = 0.5)"
        );

        // 4. FLOOR, not truncate. `-0.5 as i32` is 0, so a truncating implementation
        //    folds cells -1 and 0 together and mirrors the whole field about the
        //    origin. Probe both halves of cell -1 and require them to agree…
        let mut minus_one_differs_from_zero = 0;
        for k in 0..64 {
            let row = k as f32 + 0.5;
            let lo = cell_glyph(Vec3::new(-0.9, row, 1.5), 0.5, 0.0, PROBE_SEED);
            let hi = cell_glyph(Vec3::new(-0.1, row, 1.5), 0.5, 0.0, PROBE_SEED);
            assert_eq!(
                lo, hi,
                "x = -0.9 and x = -0.1 are both in cell -1 (row {k})"
            );
            if lo != cell_glyph(Vec3::new(0.5, row, 1.5), 0.5, 0.0, PROBE_SEED) {
                minus_one_differs_from_zero += 1;
            }
        }
        // …and require cells -1 and 0 to be genuinely DIFFERENT cells, or the check
        // above would be satisfied by an implementation that truncated.
        assert!(
            minus_one_differs_from_zero > 8,
            "cells -1 and 0 gave the same glyph on {} of 64 rows — the coordinate is \
             being TRUNCATED, not floored, so the field is mirrored at the origin",
            64 - minus_one_differs_from_zero
        );
    }

    /// A different SEED must RE-DEAL the field. Otherwise every level carrying this
    /// material shows the identical pattern and the seed is decorative.
    ///
    /// Measured, not merely asserted `!=`: two independent deals of the same field at
    /// `held = 0.5` differ on `2·p·(1−p) = 50%` of cells, so the band is centred
    /// there. A seed that only TRANSLATED the lattice would land far below it; an
    /// ignored seed lands at exactly 0.
    #[test]
    fn a_different_seed_re_deals_the_whole_field() {
        for (s0, s1) in [(11u32, 12u32), (0, 1), (7, 0xDEAD_BEEF), (u32::MAX, 0)] {
            let a = field(0.5, 0.0, s0);
            let b = field(0.5, 0.0, s1);
            let differ = a.iter().zip(&b).filter(|(x, y)| x != y).count();
            assert!(
                (0.30..=0.70).contains(&frac(differ)),
                "seeds {s0} and {s1} differ on {differ}/{PATCH_CELLS} cells \
                 ({:.3}) — two independent deals differ on ~50%; near 0 means the \
                 seed is not reaching the hash",
                frac(differ)
            );
        }
    }

    /// The rank field must be UNIFORM over `[0,1)`. Every proportion claimed in this
    /// module reads a threshold against this rank, so if the rank piled up the
    /// distribution tests would be measuring the hash's bias rather than the rule.
    #[test]
    fn the_cell_rank_is_uniform_over_the_unit_interval() {
        const BUCKETS: usize = 16;
        let mut hist = [0usize; BUCKETS];
        let mut n = 0usize;
        for z in 0..4 {
            for w in patch_cells() {
                let r = cell_rank(w + Vec3::new(0.0, 0.0, z as f32), PROBE_SEED);
                assert!((0.0..1.0).contains(&r), "rank {r} left [0,1)");
                hist[(r * BUCKETS as f32) as usize % BUCKETS] += 1;
                n += 1;
            }
        }
        let want = n as f32 / BUCKETS as f32;
        for (i, count) in hist.iter().enumerate() {
            let ratio = *count as f32 / want;
            assert!(
                (0.7..=1.3).contains(&ratio),
                "rank bucket {i}/{BUCKETS} holds {count} of {n} (×{ratio:.3} of even) \
                 — the rank field is not uniform, so every proportion below is \
                 measuring the hash, not the rule"
            );
        }
    }

    // ─────────────────────────── the endpoint table ──────────────────────────

    /// The four CORNERS of the `(held, released)` square, pinned EXACTLY.
    ///
    /// The plan drafted `assert!(u0 > 380)` out of 400 at the origin — which a
    /// `cell_glyph` returning `'░'` unconditionally satisfies. That hole is closed
    /// here from both sides: the origin must be ENTIRELY unclaimed (not merely
    /// mostly), and `held = 1` must leave ZERO unclaimed cells, which no constant
    /// return can do.
    #[test]
    fn the_corners_of_the_parameter_square_are_exact() {
        let n = PATCH_CELLS;
        // At rest: the whole field is unclaimed.
        assert_eq!(
            census(&field(0.0, 0.0, PROBE_SEED)),
            [n, 0, 0, 0],
            "at rest every cell must be unclaimed"
        );
        // Releasing with NOTHING held converts nothing: the scar threshold is
        // `held · released`, which is 0 when `held` is 0. A rule thresholded on
        // `released` alone scars the whole field here.
        assert_eq!(
            census(&field(0.0, 1.0, PROBE_SEED)),
            [n, 0, 0, 0],
            "releasing an empty hold must convert nothing — the scar threshold is \
             held · released, so it is 0 whenever nothing is held"
        );
        // Fully held, nothing released: the whole field is held. THIS is the
        // assertion a constant-'░' implementation cannot survive.
        assert_eq!(
            census(&field(1.0, 0.0, PROBE_SEED)),
            [0, n, 0, 0],
            "with everything held no cell may remain unclaimed"
        );
        // Fully held, fully released: the whole field is burned.
        assert_eq!(
            census(&field(1.0, 1.0, PROBE_SEED)),
            [0, 0, 0, n],
            "a fully released full hold burns entirely"
        );
    }

    // ────────────────────────────── monotonicity ─────────────────────────────

    /// Raising `held` claims a GROWING, NESTED subset of the field, in proportion to
    /// `held` itself.
    ///
    /// Nesting is the strong form of "monotone": it is a per-CELL claim, not an
    /// aggregate one. An implementation that re-dealt the field at each `held` could
    /// still grow its COUNT monotonically while flickering individual cells in and
    /// out — which is exactly the artefact that would make a wall boil as the puzzle
    /// advanced. The count is checked against `held` too, so a rule that converted at
    /// the wrong rate (say `held²`) fails even though it is perfectly nested.
    #[test]
    fn held_claims_a_growing_nested_subset_of_the_field() {
        const STEPS: usize = 32;
        let mut prev: Option<Vec<bool>> = None;
        let mut prev_count = 0usize;
        let mut strict_increases = 0;
        for k in 0..=STEPS {
            let held = k as f32 / STEPS as f32;
            let f = field(held, 0.0, PROBE_SEED);
            let set = claimed_set(&f);
            let count = set.iter().filter(|b| **b).count();

            // The claimed fraction tracks `held` — the rank is uniform, so the
            // fraction of cells with `rank < held` IS `held`.
            assert_frac(
                count,
                held,
                &format!("claimed fraction at held = {held:.4}"),
            );

            if let Some(p) = &prev {
                assert!(
                    is_subset(p, &set),
                    "at held = {held:.4} a previously claimed cell became unclaimed — \
                     the claim must NEST, not re-deal"
                );
                assert!(
                    count >= prev_count,
                    "the claimed count fell at held = {held:.4}"
                );
                if count > prev_count {
                    strict_increases += 1;
                }
            }
            // With nothing released, a claimed cell is HELD and nothing else.
            let [_, h, s, b] = census(&f);
            assert_eq!(
                (h, s, b),
                (count, 0, 0),
                "at released = 0 nothing may scar or burn"
            );

            prev = Some(set);
            prev_count = count;
        }
        assert!(
            strict_increases >= STEPS - 4,
            "the claimed count grew on only {strict_increases} of {STEPS} steps — the \
             field is converting in a few big jumps, not smoothly with `held`"
        );
    }

    /// **The amendment, measured.** Releasing ERODES the held set: it takes a
    /// fraction OF THE HELD SET, so at `held = 0.5, released = 0.5` half the held
    /// cells scar and half stay `█`.
    ///
    /// Under the drafted rule (`rank < held` AND `rank < released`) the two
    /// thresholds coincide at parity: the whole held set converts in one step and
    /// `█` is unreachable for every `released >= held`. Each assertion below is
    /// chosen so that rule fails it, and fails it by a wide margin rather than by a
    /// rounding error.
    #[test]
    fn releasing_erodes_the_held_set_it_does_not_flip_it() {
        const HELD: f32 = 0.5;
        const STEPS: usize = 20;

        // The CLAIMED set is invariant in `released`: releasing moves cells within
        // the held set, it never claims a new cell nor un-claims one.
        let claimed_at_rest = claimed_set(&field(HELD, 0.0, PROBE_SEED));
        let claimed_count = claimed_at_rest.iter().filter(|b| **b).count();

        let mut prev_released: Option<Vec<bool>> = None;
        let mut distinct_held_counts = std::collections::BTreeSet::new();
        for k in 0..=STEPS {
            let released = k as f32 / STEPS as f32;
            let f = field(HELD, released, PROBE_SEED);
            let [_, held_n, scar_n, burn_n] = census(&f);

            assert_eq!(
                claimed_set(&f),
                claimed_at_rest,
                "at released = {released:.3} the CLAIMED set changed — releasing must \
                 redistribute the held set, never grow or shrink it"
            );

            // The released set NESTS as `released` rises (per-cell, as above).
            let rel = released_set(&f);
            if let Some(p) = &prev_released {
                assert!(
                    is_subset(p, &rel),
                    "at released = {released:.3} a scarred cell went back to held — a \
                     release must not be undone by releasing MORE"
                );
            }
            prev_released = Some(rel);

            // The predicted split: the scar threshold is `held · released`, and the
            // rank is uniform, so `held · released` of the FIELD scars and the rest of
            // the claimed set stays held.
            assert_frac(
                scar_n + burn_n,
                HELD * released,
                &format!("released fraction at released = {released:.3}"),
            );
            assert_frac(
                held_n,
                HELD * (1.0 - released),
                &format!("still-held fraction at released = {released:.3}"),
            );
            distinct_held_counts.insert(held_n);
        }

        // THE PARITY CASE, stated on its own because it is the whole amendment.
        let [_, held_n, scar_n, burn_n] = census(&field(HELD, 0.5, PROBE_SEED));
        assert_eq!(burn_n, 0, "0.5 is far below the burn threshold");
        assert!(
            held_n > 0,
            "at held = 0.5, released = 0.5 the ENTIRE held set converted (█ = 0). That \
             is the drafted rule (`rank < released`), which makes the two thresholds \
             coincide at parity — a flip, not an erosion"
        );
        let held_share = held_n as f32 / claimed_count as f32;
        assert!(
            (0.42..=0.58).contains(&held_share),
            "at parity {held_n} of {claimed_count} claimed cells are still held \
             ({held_share:.3}) — half the held set must scar and half must remain"
        );
        assert!(
            (scar_n as f32 / claimed_count as f32 - (1.0 - held_share)).abs() < 1e-3,
            "the claimed set must partition into held and scarred with nothing else"
        );

        // EROSION, not a step: the held count passes through many intermediate values
        // across the sweep. The drafted rule produces exactly two (full, then zero).
        assert!(
            distinct_held_counts.len() >= STEPS - 2,
            "the still-held count took only {} distinct values across {} release steps \
             — the held set is flipping, not eroding",
            distinct_held_counts.len(),
            STEPS + 1
        );
    }

    /// **The same amendment, one tier down.** Burning ERODES the scar set: it takes a
    /// fraction OF THE SCAR SET across the ramp from [`BURN_RELEASE`] to 1, so partway
    /// along it a wall shows `▒` AND `✕` together.
    ///
    /// The first implementation RE-LETTERED instead — every scar flipped to `✕` the
    /// instant `released` crossed [`BURN_RELEASE`]. That is the identical defect the
    /// scar tier already had (a flip, not an erosion), and its signature is that the
    /// mixed state below is unreachable at every setting: the interval this test
    /// sweeps was empty. The endpoints are unchanged by the fix, which is why they are
    /// asserted here too — they are the regression check on the tier ABOVE.
    #[test]
    fn burning_erodes_the_scar_set_it_does_not_relabel_it() {
        /// Steps across the burn ramp. Fine enough that "many distinct burn counts"
        /// below is a statement about a continuum, not about a handful of samples.
        const RAMP_STEPS: usize = 25;

        for held in [0.3_f32, 0.6, 1.0] {
            // ENDPOINT 1 — at the foot of the ramp nothing burns, and the scar set is
            // whole. (Unchanged from the re-lettering implementation.)
            let [u_foot, _, s_foot, b_foot] = census(&field(held, BURN_RELEASE, PROBE_SEED));
            assert_eq!(
                b_foot, 0,
                "held {held}: at BURN_RELEASE nothing may burn yet"
            );
            assert!(s_foot > 0, "held {held}: there must BE a scar set to erode");

            // ENDPOINT 2 — at full release every scar has burned and none remain.
            // (Also unchanged: `burn_fraction(1.0) == 1`.)
            let [u_top, h_top, s_top, b_top] = census(&field(held, 1.0, PROBE_SEED));
            assert_eq!(s_top, 0, "held {held}: at released = 1 no scar may remain");
            assert_eq!(
                h_top, 0,
                "held {held}: at released = 1 nothing may stay held"
            );
            assert!(
                b_top > 0,
                "held {held}: at released = 1 the held set must have burned"
            );
            // The unclaimed tier never moves along the ramp: `rank < held` does not
            // mention `released` at all, so this is exact, not approximate.
            assert_eq!(
                u_foot, u_top,
                "held {held}: the burn ramp changed the UNCLAIMED count"
            );

            // THE RAMP — the interval the re-lettering rule left empty.
            let mut prev_burn: Option<Vec<bool>> = None;
            let mut distinct_burn_counts = std::collections::BTreeSet::new();
            let mut mixed_settings = 0;
            for k in 0..=RAMP_STEPS {
                let released = BURN_RELEASE + BURN_RAMP * k as f32 / RAMP_STEPS as f32;
                let f = field(held, released, PROBE_SEED);
                let [_, _, scar_n, burn_n] = census(&f);
                let burn = burned_set(&f);

                // The burn set NESTS inside itself as the ramp advances — per cell, so
                // a re-deal that merely grew the COUNT would fail here.
                if let Some(p) = &prev_burn {
                    assert!(
                        is_subset(p, &burn),
                        "held {held}, released {released:.4}: a burned cell went back to \
                         a scar — burning must not be undone by releasing MORE"
                    );
                }
                // …and inside the SCAR set, which is the containment being fixed.
                assert!(
                    is_subset(&burn, &released_set(&f)),
                    "held {held}, released {released:.4}: a burned cell is not released"
                );

                // The predicted burn fraction: `held · released · burn_fraction`.
                assert_frac(
                    burn_n,
                    held * released * burn_fraction(released),
                    &format!("burned fraction at held {held}, released {released:.4}"),
                );
                if scar_n > 0 && burn_n > 0 {
                    mixed_settings += 1;
                }
                distinct_burn_counts.insert(burn_n);
                prev_burn = Some(burn);
            }

            // THE DEFECT, stated directly: partway along the ramp the wall must show
            // scarred AND burned cells at once. Under the re-lettering rule this is 0
            // at every held, because the burn tier goes 0 → everything in one step.
            assert!(
                mixed_settings >= RAMP_STEPS - 4,
                "held {held}: only {mixed_settings} of {} ramp settings show `▒` and \
                 `✕` together — burning is re-lettering the scar set, not eroding it",
                RAMP_STEPS + 1
            );
            // EROSION, not a step: many intermediate burn counts. The re-lettering
            // rule produces exactly two (none, then all).
            assert!(
                distinct_burn_counts.len() >= RAMP_STEPS - 4,
                "held {held}: the burn count took only {} distinct values across {} ramp \
                 steps — the scar set is flipping, not eroding",
                distinct_burn_counts.len(),
                RAMP_STEPS + 1
            );
        }
    }

    /// The containment the whole design rests on, asserted PER CELL over a grid of
    /// settings: `✕ ⊆ ▒ ⊆ █`, i.e. every burned cell is released and every released
    /// cell is claimed.
    ///
    /// This is the one instrument that catches a tier which stops nesting, at whatever
    /// depth. Both drafted rules broke a containment; neither broke it in a way a
    /// count-only assertion would notice, because both kept their tier's TOTAL
    /// plausible while re-dealing which cells were in it.
    #[test]
    fn the_four_tiers_nest_per_cell() {
        let settings = [0.0_f32, 0.13, 0.37, 0.5, 0.75, 0.94, 0.95, 0.97, 0.99, 1.0];
        let mut seen_all_four = 0;
        for held in settings {
            for released in settings {
                let f = field(held, released, PROBE_SEED);
                let (claimed, rel, burn) = (claimed_set(&f), released_set(&f), burned_set(&f));
                assert!(
                    is_subset(&burn, &rel),
                    "({held}, {released}): a burned cell is not in the released set"
                );
                assert!(
                    is_subset(&rel, &claimed),
                    "({held}, {released}): a released cell is not in the claimed set"
                );
                if census(&f).iter().all(|n| *n > 0) {
                    seen_all_four += 1;
                }
            }
        }
        // Anti-vacuity: the nesting claim is trivially true for a field that only ever
        // shows one or two tiers. Some setting must put ALL FOUR glyphs on the wall at
        // once — which is precisely what the re-lettering rule could never do.
        assert!(
            seen_all_four > 0,
            "no setting in the sweep showed all four glyphs at once — the tiers cannot \
             coexist, so the nesting assertions above are vacuous"
        );
    }

    // ──────────────────────────── total on all input ─────────────────────────

    /// Every input shape produces a defined glyph from [`ALPHABET`] — the output
    /// contract. Degenerate channels, degenerate coordinates, degenerate seeds; no
    /// panic, no fifth glyph, no NaN escaping into the renderer.
    #[test]
    fn every_input_shape_produces_one_of_the_four_glyphs() {
        let coords = [
            Vec3::ZERO,
            Vec3::new(-0.0, -0.0, -0.0),
            Vec3::new(1e9, -1e9, 1e9),
            Vec3::new(f32::MAX, f32::MIN, 0.0),
            Vec3::new(f32::INFINITY, f32::NEG_INFINITY, 0.0),
            Vec3::new(f32::NAN, f32::NAN, f32::NAN),
            Vec3::new(-0.5, -1.5, -2.5),
        ];
        let channels = [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -1.0,
            -0.0,
            0.0,
            0.5,
            1.0,
            2.0,
            1e9,
        ];
        for w in coords {
            for held in channels {
                for released in channels {
                    for seed in [0u32, 1, PROBE_SEED, u32::MAX] {
                        let g = cell_glyph(w, held, released, seed);
                        assert!(
                            ALPHABET.contains(&g),
                            "cell_glyph({w:?}, {held}, {released}, {seed}) = {g:?}, \
                             which is not in the cell alphabet"
                        );
                    }
                }
            }
        }
    }

    /// A NaN channel reads as AT REST, the documented (and safest) of the four
    /// readings: a broken state channel shows an untouched wall, never a burned one.
    /// Pinned separately from the totality sweep above, which only checks membership.
    #[test]
    fn a_non_finite_channel_reads_as_at_rest() {
        let n = PATCH_CELLS;
        assert_eq!(
            census(&field(f32::NAN, 0.5, PROBE_SEED)),
            [n, 0, 0, 0],
            "a NaN `held` must read as nothing held"
        );
        // A NaN `released` leaves the held set intact (released reads as 0), and an
        // out-of-range one saturates rather than wrapping.
        assert_eq!(
            census(&field(1.0, f32::NAN, PROBE_SEED)),
            [0, n, 0, 0],
            "a NaN `released` must read as nothing released"
        );
        assert_eq!(
            census(&field(1.0, -5.0, PROBE_SEED)),
            [0, n, 0, 0],
            "a negative `released` clamps to 0, it does not wrap"
        );
        assert_eq!(
            census(&field(5.0, 5.0, PROBE_SEED)),
            [0, 0, 0, n],
            "an over-range hold and release clamp to 1, they do not wrap"
        );
    }

    /// `glyph_for_rank` and `cell_glyph` must agree — the two-hash and one-hash paths
    /// are the same rule, so a caller that reuses a rank it already has (the wall
    /// recipe, for its tone) cannot drift from the alphabet.
    #[test]
    fn the_split_rank_path_agrees_with_the_direct_one() {
        for w in patch_cells() {
            for (held, released) in [
                (0.0, 0.0),
                (0.5, 0.0),
                (0.5, 0.5),
                (0.5, 0.97),
                (0.5, 1.0),
                (1.0, 1.0),
            ] {
                assert_eq!(
                    glyph_for_rank(cell_rank(w, PROBE_SEED), held, released),
                    cell_glyph(w, held, released, PROBE_SEED),
                    "{w:?} at ({held}, {released})"
                );
            }
        }
    }

    /// The measured erosion curve, as a table a reviewer can check by eye against the
    /// spec: the `█` column empties gradually into `▒`, and then — across the burn
    /// ramp — the `▒` column empties gradually into `✕`.
    ///
    /// Both drafted rules would fail a row here. The `(0.5, 0.5)` row kills the
    /// "scar threshold is `released`" rule (it predicts `█ = 0`); the `(0.5, 0.97)`
    /// row kills the "burning re-letters the scars" rule (it predicts `▒ = 0`,
    /// `✕ = 0.485`). The two endpoints `(0.5, 1.0)` and `(1.0, 1.0)` are the
    /// regression check: the burn-ramp fix must leave them exactly where they were.
    #[test]
    fn the_erosion_curve_reads_as_an_erosion() {
        let points = [
            // (held, released, ░, █, ▒, ✕) as fractions of the field.
            (0.0_f32, 0.0_f32, 1.00_f32, 0.00_f32, 0.00_f32, 0.00_f32),
            (0.5, 0.0, 0.50, 0.50, 0.00, 0.00),
            (0.5, 0.5, 0.50, 0.25, 0.25, 0.00),
            // The foot of the burn ramp: fully scarred, nothing burned yet.
            (0.5, 0.95, 0.50, 0.025, 0.475, 0.00),
            // MIDWAY UP the ramp — the state the re-lettering rule could not reach:
            // `burn_fraction(0.97) = 0.4`, so 40% of the 0.485 scar set has burned.
            (0.5, 0.97, 0.50, 0.015, 0.291, 0.194),
            (0.5, 1.0, 0.50, 0.00, 0.00, 0.50),
            (1.0, 1.0, 0.00, 0.00, 0.00, 1.00),
        ];
        for (held, released, wu, wh, ws, wb) in points {
            let [u, h, s, b] = census(&field(held, released, PROBE_SEED));
            let what = format!("({held}, {released})");
            assert_frac(u, wu, &format!("{what} ░"));
            assert_frac(h, wh, &format!("{what} █"));
            assert_frac(s, ws, &format!("{what} ▒"));
            assert_frac(b, wb, &format!("{what} ✕"));
        }
    }

    // ────────────────────── the voice materials' mark ────────────────────────

    /// The mark's quantisation, pinned as a LITERAL table on both sides.
    ///
    /// Deliberately not a round-trip: `encode`/`decode` agreeing proves only that
    /// they share a bug. `re_game` writes the word and this module reads it, so the
    /// wire value itself is the contract and it is spelled out here in numbers a
    /// reviewer can check against "parts per thousand" without running anything.
    #[test]
    fn the_mark_quantises_to_parts_per_thousand() {
        // ENCODE — the exact words `re_game` puts on the bus.
        for (progress, want) in [
            (0.0_f32, 0_u32),
            (0.001, 1),
            (0.25, 250),
            (0.5, 500),
            (0.999, 999),
            (1.0, 1000),
        ] {
            assert_eq!(encode_mark(progress), want, "encode_mark({progress})");
        }
        // DECODE — the exact marks this module reads back.
        for (word, want) in [
            (0_u32, 0.0_f32),
            (1, 0.001),
            (250, 0.25),
            (500, 0.5),
            (999, 0.999),
            (1000, 1.0),
        ] {
            assert!(
                (decode_mark(word) - want).abs() < 1e-6,
                "decode_mark({word}) = {}, expected {want}",
                decode_mark(word)
            );
        }
        // TOTAL on every degenerate input, both directions — no panic, no wrap, and
        // a broken channel reads as AT REST rather than as a finished wall.
        for bad in [f32::NAN, f32::NEG_INFINITY, -1.0, -0.0] {
            assert_eq!(
                encode_mark(bad),
                0,
                "a broken progress ({bad}) must read at rest"
            );
        }
        for big in [f32::INFINITY, 1.5, 1e9] {
            assert_eq!(
                encode_mark(big),
                MARK_PARTS,
                "an over-range progress ({big}) saturates"
            );
        }
        for over in [MARK_PARTS + 1, 0xFFFF, u32::MAX] {
            assert_eq!(
                decode_mark(over),
                1.0,
                "word {over} must saturate, not wrap"
            );
        }
    }

    /// The quantisation must be fine enough that the wall reads as a RAMP.
    ///
    /// A coarser encoding (a percentage, say, or a small enum) would still satisfy
    /// every claim above and still convert the field monotonically — it would just
    /// do it in visible steps. Measured on the FIELD, not on the number: across 200
    /// evenly spaced progress values the held count must take nearly that many
    /// distinct values.
    #[test]
    fn the_mark_resolves_finely_enough_to_read_as_a_ramp() {
        const STEPS: usize = 200;
        let mut counts = std::collections::BTreeSet::new();
        for k in 0..=STEPS {
            let mark = decode_mark(encode_mark(k as f32 / STEPS as f32));
            let (held, released) = VoiceMode::Hold.channels(mark);
            let [_, h, _, _] = census(&field(held, released, PROBE_SEED));
            counts.insert(h);
        }
        assert!(
            counts.len() >= STEPS - 8,
            "the held count took only {} distinct values across {} progress steps — \
             the mark's quantisation is coarse enough to read as stair-steps",
            counts.len(),
            STEPS + 1
        );
    }

    /// The two voices spend the mark on OPPOSITE channels, measured on the field.
    ///
    /// The prediction is arithmetic, not a re-statement of `channels()`: the rank is
    /// uniform, so at mark `m` the HOLD voice holds `m` of the field and the RELEASE
    /// voice releases `m` of it — and every cell the release voice has not released
    /// is still `█`, because it started from a whole field.
    #[test]
    fn the_two_voices_spend_the_mark_on_opposite_channels() {
        for k in 0..=20 {
            let mark = k as f32 / 20.0;
            let (hh, hr) = VoiceMode::Hold.channels(mark);
            let [hu, hh_n, hs, hb] = census(&field(hh, hr, PROBE_SEED));
            assert_frac(hh_n, mark, &format!("mnemos █ at mark {mark:.2}"));
            assert_frac(hu, 1.0 - mark, &format!("mnemos ░ at mark {mark:.2}"));
            assert_eq!(
                (hs, hb),
                (0, 0),
                "mnemos never releases: it only ever holds"
            );

            let (rh, rr) = VoiceMode::Release.channels(mark);
            let [ru, r_held, rs, rb] = census(&field(rh, rr, PROBE_SEED));
            assert_eq!(
                ru, 0,
                "lethe starts from a WHOLE field — nothing is unclaimed"
            );
            assert_frac(rs + rb, mark, &format!("lethe ▒+✕ at mark {mark:.2}"));
            assert_frac(r_held, 1.0 - mark, &format!("lethe █ at mark {mark:.2}"));
        }
        // The endpoints, exactly: an empty wall that fills, and a whole wall that goes.
        let n = PATCH_CELLS;
        let hold_full = VoiceMode::Hold.channels(1.0);
        assert_eq!(
            census(&field(hold_full.0, hold_full.1, PROBE_SEED)),
            [0, n, 0, 0]
        );
        let rel_rest = VoiceMode::Release.channels(0.0);
        assert_eq!(
            census(&field(rel_rest.0, rel_rest.1, PROBE_SEED)),
            [0, n, 0, 0]
        );
        // …and at FULL release the field is burned, not scarred. `released = 1` is
        // the top of the burn ramp, so the last stretch of the mark takes the wall
        // past legibility. Pinned here because it is the one place the shipped
        // reading departs from the plan's one-word summary ("scars to ▒").
        let rel_full = VoiceMode::Release.channels(1.0);
        assert_eq!(
            census(&field(rel_full.0, rel_full.1, PROBE_SEED)),
            [0, 0, 0, n],
            "a fully released wall BURNS — mark 1.0 sits at the top of the burn ramp"
        );
    }

    /// The alias, and why it is survivable: a state word that was never a mark —
    /// a `State` family index, a packed pipe flow word, an empty bus — decodes to
    /// ≈0, and BOTH voices then render their resting look.
    ///
    /// This is the failure mode of a mis-tagged voice wall, and it is the benign
    /// one: an untouched surface, never a completed or a burned one. (The two
    /// readings not meeting in the first place is asserted in `ascii_shader` on the
    /// material and in `re_game` on the tag; this measures what happens if both
    /// guards were somehow wrong.)
    #[test]
    fn an_aliased_family_or_flow_word_reads_as_at_rest() {
        /// How many of the patch's cells an aliased word may move. `7/1000` of the
        /// field is 7 cells here; the bar is the next whole cell up. Stated as a
        /// COUNT rather than "none" because the aliased words are not literally 0
        /// — the point is that the wall is visually untouched, not that the
        /// arithmetic is degenerate.
        const MAX_MOVED: usize = 8;
        let rest_hold = {
            let (h, r) = VoiceMode::Hold.channels(0.0);
            field(h, r, PROBE_SEED)
        };
        let rest_release = {
            let (h, r) = VoiceMode::Release.channels(0.0);
            field(h, r, PROBE_SEED)
        };
        // The two resting looks are the ones a mis-tagged wall must fall back to.
        assert_eq!(
            census(&rest_hold),
            [PATCH_CELLS, 0, 0, 0],
            "mnemos rests unclaimed"
        );
        assert_eq!(
            census(&rest_release),
            [0, PATCH_CELLS, 0, 0],
            "lethe rests whole"
        );

        // 0..=7 covers every node-state lane (powered/dead/decoy/won) and every
        // packed pipe flow word the bus otherwise carries.
        for word in 0u32..=7 {
            let mark = decode_mark(word);
            assert!(
                mark < 0.01,
                "state word {word} decodes to a mark of {mark} — a family index must \
                 not read as meaningful progress"
            );
            for (mode, rest) in [
                (VoiceMode::Hold, &rest_hold),
                (VoiceMode::Release, &rest_release),
            ] {
                let (h, r) = mode.channels(mark);
                let moved = field(h, r, PROBE_SEED)
                    .iter()
                    .zip(rest)
                    .filter(|(a, b)| a != b)
                    .count();
                assert!(
                    moved <= MAX_MOVED,
                    "an aliased word {word} moved {moved} of {PATCH_CELLS} cells on \
                     {mode:?} — a mis-tagged voice wall must read as untouched"
                );
            }
        }
    }

    /// The two reserved tags are distinct and sit outside the pipe namespace.
    ///
    /// The `const _: () = assert!(…)` pair beside the constants makes this a build
    /// error; restated here so the reason is visible where the field tests are, and
    /// so `VoiceMode::tag` — the accessor the bridge and the recipe both go through
    /// — is exercised rather than only the raw constants.
    #[test]
    fn the_two_voices_read_distinct_reserved_tags() {
        assert_ne!(VoiceMode::Hold.tag(), VoiceMode::Release.tag());
        for m in [VoiceMode::Hold, VoiceMode::Release] {
            assert_eq!(
                m.tag() & re_core::PIPE_TAG_FLAG,
                0,
                "{m:?} must not be a pipe tag"
            );
            // Above any node index or colour id a v0.1 level can produce: the dense
            // bus is keyed by the SAME `u16` and both of those spaces grow from 0.
            assert!(
                m.tag() > 0x7F00,
                "{m:?} must sit at the top of the tag space"
            );
        }
    }
}
