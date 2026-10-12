//! `glyph_family` — KINDRED-SYMBOL maps + the family-driven super-glyph selector.
//!
//! This module realises the player's "make maps of KINDRED symbols" idea against
//! the material spec's super-glyph (§7), animation (§12) and particle-channel
//! (§13) layers. A [`GlyphFamily`] is a curated, ORDERED sequence of *close*
//! glyphs — symbols that read as variants of one shape (`o 0 O`, `· ∘ ○ ●`,
//! `▁▂▃▄▅▆▇█`). Substituting one member for another shifts the look WITHOUT
//! changing the meaning, which is exactly what makes a "FLOATING" glyph: cycle a
//! neutral family over time and a surface shimmers / drifts like haze or mist.
//!
//! Every family is tagged with a [`FamilyKind`] that says how it is *meant* to be
//! driven:
//!
//! * [`Neutral`](FamilyKind::Neutral) — similar shapes, cycle over `time` for a
//!   haze / flow flicker (the floating effect). Order is decorative, not ranked.
//! * [`Growth`](FamilyKind::Growth) — small→large, indexed by `distance` (LOD) or
//!   a particle's birth→death intensity. Order IS the magnitude ramp.
//! * [`WithinAnim`](FamilyKind::WithinAnim) — a rotation / fill-sweep cycle of one
//!   glyph (`| / — \`, `▏▎▍▌▋▊▉█`), driven by `time` or `state`.
//! * [`Positional`](FamilyKind::Positional) — a left/centre/right triplet (`▏▎▍▌`
//!   left-fill vs `▐▕` right) selectable by a `normal` / `incidence` lean, for a
//!   depth / normal cue.
//!
//! ## Reliability
//!
//! Every glyph here is from the "reliable alphabet": ASCII (U+0020–007E),
//! Box Drawing (U+2500–257F), Block Elements (U+2580–259F) and basic geometry
//! (U+25A0–25FF), all width-1 in any monospace font. A handful of long-standing
//! Latin-1 typographic glyphs (`°`, `·`, `≈`, `∙`, `•`, `◦`) are also width-1 and
//! ship in every monospace font; they are the only out-of-block members.
//! [`is_reliable`] is the single gate the unit tests assert against, so the
//! policy is testable.
//!
//! ## Determinism + anti-flicker
//!
//! Selection is a PURE function of `(family, driver, phase, seed)` — no clock, no
//! `rand`. The per-cell `phase`/`seed` come from a WORLD hash ([`cell_phase`]),
//! so neighbouring cells DESYNC (the floating field doesn't pulse in lockstep) yet
//! a given world spot is stable frame-to-frame. The driver is expected to arrive
//! ALREADY quantized (the same [`quantize_lod`](crate::ascii_shader::quantize_lod)
//! deadband the rest of the shader uses), so the floating glyph drifts smoothly
//! and does not jitter — the spec's "discrete output changes only on a real,
//! hysteretic threshold crossing" discipline (§0/§G) reused verbatim.
//!
//! All families are `&'static [char]` slices — no per-cell heap allocation in the
//! hot path.

use crate::ascii_shader::hash2i;

/// What a [`GlyphFamily`] is FOR — how its ordered members are meant to be driven.
/// Pure data; lets a tool/test group families and a selector pick a sensible
/// default driver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FamilyKind {
    /// Similar shapes, no magnitude order — cycle over `time` for a haze / flow
    /// shimmer (the FLOATING effect). Substituting members is meaning-preserving.
    Neutral,
    /// Small→large ramp — index by `distance` (LOD) or particle intensity. Order
    /// is the magnitude (member 0 = smallest / faintest, last = largest / fullest).
    Growth,
    /// A within-glyph animation cycle (rotation / fill sweep) — index by `time`
    /// or `state`. Order is the animation phase.
    WithinAnim,
    /// A left/centre/right (or low/mid/high fill) lean — index by a `normal` /
    /// `incidence` signal for a depth / normal cue. Order runs left→right.
    Positional,
}

/// A KINDRED-SYMBOL MAP: an ordered sequence of close glyphs that read as variants
/// of one shape, tagged with how it is meant to be driven ([`FamilyKind`]).
///
/// Pure data (`Copy`, holds a `&'static [char]`) so a [`crate::Material`] can carry
/// `Some(GlyphFamily)` and stay `Copy` / `static`. The `glyphs` slice is the map;
/// [`pick`](Self::pick) turns a driver value into one member.
#[derive(Clone, Copy, Debug)]
pub struct GlyphFamily {
    /// Stable name an author/tool references (e.g. `"haze_o"`, `"growth_dot"`).
    pub name: &'static str,
    /// What this family is for + its default driver semantics.
    pub kind: FamilyKind,
    /// The ordered kindred glyphs — the "map". Never empty (the library asserts).
    pub glyphs: &'static [char],
}

impl GlyphFamily {
    /// Number of members (≥ 1 for every library family).
    #[inline]
    pub fn len(&self) -> usize {
        self.glyphs.len()
    }

    /// True if the family carries no glyphs (never true for a library family;
    /// guarded so a hand-built empty family can't panic [`pick`](Self::pick)).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.glyphs.is_empty()
    }

    /// The member at `index`, clamped into range (an empty family yields a space).
    #[inline]
    pub fn at(&self, index: usize) -> u32 {
        if self.glyphs.is_empty() {
            return u32::from(' ');
        }
        self.glyphs[index.min(self.glyphs.len() - 1)] as u32
    }

    /// Pick a member for a normalized driver `t ∈ [0,1]`, mapped across the whole
    /// sequence (member 0 at `t=0`, last member at `t=1`). Clamped + rounded to the
    /// nearest member. Pure. Used by the GROWTH (distance) and POSITIONAL
    /// (normal/incidence) drivers, where the position in the sequence IS the cue.
    #[inline]
    pub fn pick(&self, t: f32) -> u32 {
        let n = self.glyphs.len();
        if n == 0 {
            return u32::from(' ');
        }
        let t = t.clamp(0.0, 1.0);
        let idx = (t * (n as f32 - 1.0) + 0.5) as usize;
        self.glyphs[idx.min(n - 1)] as u32
    }

    /// Pick a member by CYCLING the sequence at `phase` cycles (the FLOATING /
    /// within-glyph path). `phase` is in whole-cycle units (e.g. `speed * time`);
    /// `seed` desyncs neighbouring cells so the field doesn't pulse in lockstep.
    /// Wraps modulo the length — a loop, not a clamp — so a neutral family shimmers
    /// forever. Pure + allocation-free.
    #[inline]
    pub fn cycle(&self, phase: f32, seed: u32) -> u32 {
        let n = self.glyphs.len();
        if n == 0 {
            return u32::from(' ');
        }
        // Per-cell phase offset in [0,1) so neighbours are out of step.
        let off = (seed & 0xFFFF) as f32 / 65_536.0;
        let p = (phase + off).rem_euclid(n as f32);
        let idx = (p as usize) % n;
        self.glyphs[idx] as u32
    }
}

/// Which per-cell DRIVER a family-glyph selection reads — the spec §7 super-glyph
/// selectors, narrowed to the four the player asked for. Pure data on a material.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FamilyDriver {
    /// `time-driven` cycle — the FLOATING / haze / flow shimmer. Loops the family
    /// at `speed` cycles/sec, desynced per cell. Best paired with a `Neutral` or
    /// `WithinAnim` family. ANIMATED (forces re-shade / cache bypass).
    Time,
    /// `distance-driven` GROWTH — picks a member by a normalized, QUANTIZED distance
    /// (near = small/early member, far = large/late, or vice-versa per family
    /// order). LOD glyph. Static (cacheable): no `time` term.
    Distance,
    /// `normal/incidence-driven` POSITIONAL lean — picks a member by a signed
    /// `[-1,1]` lean signal (left/centre/right). Static.
    Normal,
    /// `state-driven` frame — picks a member by an integer state index (a small
    /// finite-state look: sleeping / active / cursed). Static.
    State,
}

impl FamilyDriver {
    /// True if this driver depends on `time` — so a material carrying it must be
    /// treated as ANIMATED (bypass the [`SampleMemo`](crate::ascii_shader::SampleMemo)
    /// cache, re-shade every tick). Only [`Time`](Self::Time) floats; the rest are
    /// pure functions of static per-cell geometry and stay cacheable.
    #[inline]
    pub fn is_animated(self) -> bool {
        matches!(self, FamilyDriver::Time)
    }
}

/// An optional super-glyph LAYER a [`crate::Material`] can carry: a [`GlyphFamily`]
/// plus the [`FamilyDriver`] that selects within it plus the float speed. When
/// present it REPLACES the recipe's base glyph with a kindred-symbol substitution
/// (the floating effect); when absent the material is byte-for-byte unchanged.
///
/// Pure data (`Copy`) so a `Material` stays `Copy` / `static`.
#[derive(Clone, Copy, Debug)]
pub struct FamilyLayer {
    /// The kindred-symbol set to substitute the base glyph within.
    pub family: GlyphFamily,
    /// What selects the active symbol within `family`.
    pub driver: FamilyDriver,
    /// Cycles/second for the [`Time`](FamilyDriver::Time) driver (ignored by the
    /// static drivers). The float rate of the haze.
    pub speed: f32,
}

impl FamilyLayer {
    /// True if this layer's look changes over `time` (drives the material's
    /// `is_animated`, so the renderer keeps re-shading + bypasses the memo cache).
    #[inline]
    pub fn is_animated(&self) -> bool {
        self.driver.is_animated()
    }

    /// The DRIVER INPUTS the selector reads, gathered per cell. A small POD the
    /// shader fills from the signals it already has (`time`, the QUANTIZED distance,
    /// a normal/incidence lean, a state index) and hands to [`select`](Self::select).
    /// Keeping it explicit makes the selector a pure, unit-testable function.
    #[inline]
    pub fn select(&self, d: &FamilyDrive, phase_seed: u32) -> u32 {
        match self.driver {
            FamilyDriver::Time => {
                // Floating cycle: phase = speed * time, desynced per cell.
                self.family.cycle(self.speed * d.time, phase_seed)
            }
            FamilyDriver::Distance => {
                // GROWTH / LOD: a normalized, already-quantized distance picks the
                // member (the caller quantizes so the glyph is deadbanded).
                self.family.pick(d.distance01)
            }
            FamilyDriver::Normal => {
                // POSITIONAL lean: map a signed [-1,1] lean onto [0,1] across the
                // sequence (left → centre → right).
                self.family.pick(0.5 * (d.lean + 1.0))
            }
            FamilyDriver::State => {
                // Finite-state frame: index straight into the sequence.
                self.family.at(d.state as usize)
            }
        }
    }
}

/// The per-cell DRIVER INPUTS a [`FamilyLayer`] selector reads. Filled by the
/// shader from signals it already computes; a focused test can build it directly.
/// Pure POD.
#[derive(Clone, Copy, Debug)]
pub struct FamilyDrive {
    /// Elapsed seconds (the floating clock). Only read by the `Time` driver.
    pub time: f32,
    /// A normalized, ALREADY-QUANTIZED distance `[0,1]` (0 = nearest, 1 = far). The
    /// caller quantizes (the LOD deadband) so the growth glyph is anti-flicker
    /// stable. Read by the `Distance` driver.
    pub distance01: f32,
    /// A signed lean `[-1,1]` (e.g. `tangent_u · view` or a normal component) for
    /// the POSITIONAL driver: −1 leans left, 0 centre, +1 right.
    pub lean: f32,
    /// A small integer STATE index for the `State` driver (sleeping/active/cursed…).
    pub state: u32,
}

impl FamilyDrive {
    /// A zeroed drive (centre lean, near, state 0, t=0) — a neutral base for tests.
    pub const ZERO: FamilyDrive = FamilyDrive {
        time: 0.0,
        distance01: 0.0,
        lean: 0.0,
        state: 0,
    };
}

/// Derive a deterministic per-cell PHASE/SEED from a world coordinate so that
/// neighbouring cells DESYNC (the floating field doesn't pulse in lockstep) while
/// a given world spot stays stable frame-to-frame (anti-flicker: world-anchored).
/// Pure — reuses the shader's [`hash2i`]. Quantizes the coord to a coarse lattice
/// so a tiny camera move maps to the SAME seed (the same deadband discipline the
/// LOD path uses) rather than re-rolling the phase every sub-cell frame.
#[inline]
pub fn cell_phase(coord_x: f32, coord_y: f32) -> u32 {
    // Coarse world lattice (≈ the wall-texel granularity). Floor, not round, so a
    // cell owns a stable bucket.
    let ix = (coord_x * 3.0).floor() as i32;
    let iy = (coord_y * 3.0).floor() as i32;
    hash2i(ix, iy)
}

// ───────────────────────── reliability gate ─────────────────────────────────

/// True if `g` is from the spec §6 "reliable alphabet" (width-1, present in any
/// monospace font): ASCII printable, Box Drawing, Block Elements, basic geometry —
/// PLUS the small set of long-standing width-1 typographic glyphs the spec's own
/// ASCII examples use for the haze/flow families (`° · ∙ • ◦ ≈ ∶ ⁞ ¦`). CJK,
/// emoji, and "extended ASCII" bytes are rejected. The library tests assert every
/// family member passes this.
#[inline]
pub fn is_reliable(g: char) -> bool {
    let c = g as u32;
    // ASCII printable (U+0020–007E).
    (0x20..=0x7E).contains(&c)
        // Box Drawing (U+2500–257F).
        || (0x2500..=0x257F).contains(&c)
        // Block Elements (U+2580–259F).
        || (0x2580..=0x259F).contains(&c)
        // Geometric Shapes basics (U+25A0–25FF) — ○ ● ◦ ◜◝◞◟ etc.
        || (0x25A0..=0x25FF).contains(&c)
        // The vetted width-1 typographic extras the spec's examples use.
        || matches!(
            c,
            0x00B0 // °  degree
            | 0x00B7 // ·  middle dot
            | 0x00A6 // ¦  broken bar
            | 0x2218 // ∘  ring operator (small hollow circle)
            | 0x2219 // ∙  bullet operator
            | 0x2022 // •  bullet
            | 0x2248 // ≈  almost equal
            | 0x2014 // —  em dash (spinner tick)
            | 0x203E // ‾  overline
            | 0x2236 // ∶  ratio
            | 0x205E // ⁞  vertical four dots
        )
}

// ───────────────────────── the curated LIBRARY ──────────────────────────────
//
// Maps of kindred symbols: each is an ordered slice of close glyphs from the
// reliable alphabet, tagged with its KIND. All `&'static` — zero per-cell
// allocation.

// NEUTRAL / haze-flow: similar shapes, substitute for shimmer without changing
// meaning. The FLOATING effect cycles these over time.

/// `o 0 O ° ◦` — rounded "o" kindred: the canonical floating glyph the player named.
pub const HAZE_O: GlyphFamily = GlyphFamily {
    name: "haze_o",
    kind: FamilyKind::Neutral,
    glyphs: &['o', '0', 'O', '°', '◦'],
};

/// `. · ∙ • ●` — dot kindred, faint→solid points. Mist / dust motes.
pub const HAZE_DOT: GlyphFamily = GlyphFamily {
    name: "haze_dot",
    kind: FamilyKind::Neutral,
    glyphs: &['.', '·', '∙', '•', '●'],
};

/// `~ - ≈ ‾` — horizontal wave/flow kindred. Surface ripple / heat shimmer.
pub const HAZE_WAVE: GlyphFamily = GlyphFamily {
    name: "haze_wave",
    kind: FamilyKind::Neutral,
    glyphs: &['~', '-', '≈', '‾'],
};

/// `: ∶ ⁞` — vertical-dot kindred. A faint vertical stipple flow.
pub const HAZE_COLON: GlyphFamily = GlyphFamily {
    name: "haze_colon",
    kind: FamilyKind::Neutral,
    glyphs: &[':', '∶', '⁞'],
};

/// `| ¦ ╎ ┊` — vertical bar kindred (solid→broken). Falling-rain / curtain flow.
pub const HAZE_BAR: GlyphFamily = GlyphFamily {
    name: "haze_bar",
    kind: FamilyKind::Neutral,
    glyphs: &['|', '¦', '╎', '┊'],
};

// GROWTH small→large: LOD / distance / particle birth. Order IS the magnitude.

/// `· ∘ o O ●` — point→ring→disc growth. Particle birth / distance LOD.
pub const GROWTH_DOT: GlyphFamily = GlyphFamily {
    name: "growth_dot",
    kind: FamilyKind::Growth,
    glyphs: &['·', '∘', 'o', 'O', '●'],
};

/// `. : ░ ▒ ▓ █` — sparse→solid density growth (the canonical density ramp).
pub const GROWTH_DENSITY: GlyphFamily = GlyphFamily {
    name: "growth_density",
    kind: FamilyKind::Growth,
    glyphs: &['.', ':', '░', '▒', '▓', '█'],
};

/// `▁▂▃▄▅▆▇█` — vertical fill growth (a level / bar). Rising magnitude.
pub const GROWTH_BARS: GlyphFamily = GlyphFamily {
    name: "growth_bars",
    kind: FamilyKind::Growth,
    glyphs: &['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'],
};

// WITHIN-GLYPH ANIMATION: a rotation / fill-sweep cycle of one glyph.

/// `| / — \` — a rotating tick (4-phase spinner). Classic within-glyph animation.
pub const ANIM_SPIN: GlyphFamily = GlyphFamily {
    name: "anim_spin",
    kind: FamilyKind::WithinAnim,
    glyphs: &['|', '/', '—', '\\'],
};

/// `◜◝◞◟` — a quarter-arc orbiting the cell. Smooth rotation cycle.
pub const ANIM_ARC: GlyphFamily = GlyphFamily {
    name: "anim_arc",
    kind: FamilyKind::WithinAnim,
    glyphs: &['◜', '◝', '◞', '◟'],
};

/// `▏▎▍▌▋▊▉█` — a left→full fill SWEEP cycle. A loading / charge sweep.
pub const ANIM_SWEEP: GlyphFamily = GlyphFamily {
    name: "anim_sweep",
    kind: FamilyKind::WithinAnim,
    glyphs: &['▏', '▎', '▍', '▌', '▋', '▊', '▉', '█'],
};

// POSITIONAL left/centre/right: bias by a normal / incidence lean.

/// `▏▎▍▌ ▐▕` — a left-fill→right-fill lean. Maps a signed lean to which side of the
/// cell is inked, a cheap depth / normal cue (member 0 = leans LEFT, last = RIGHT).
pub const LEAN_FILL: GlyphFamily = GlyphFamily {
    name: "lean_fill",
    kind: FamilyKind::Positional,
    // left-heavy … centre … right-heavy
    glyphs: &['▏', '▎', '▍', '▌', '▐', '▕'],
};

/// `╴─╶` (then `╸━╺`-style) — a left/centre/right horizontal stub triplet. A
/// minimal left/centre/right positional pick by normal lean.
pub const LEAN_STUB: GlyphFamily = GlyphFamily {
    name: "lean_stub",
    kind: FamilyKind::Positional,
    glyphs: &['╴', '─', '╶'],
};

// STATE families: a small FINITE-STATE look, indexed by the cell's live state word
// (the reactive-materials state bus, spec §2). Order IS the state index — member 0 is
// the resting/default look (an EMPTY state map reads state 0 → member 0), so a
// material wearing one of these is unchanged until the bridge drives a non-zero state.

// STATE ENCODING CONTRACT (shared with the `re_game` bridge — kept generic here):
// member 0 is the RESTING / default look, which an EMPTY state bus selects for every
// cell, so a material is byte-identical to its shipped fixed glyph until the bridge
// drives a non-zero state. The bridge's node-state lanes
// ([`re_core::CellStateMap`]) are:  0 = powered (the shipped, active look),
// 1 = dead/cold, 2 = decoy, 3 = won. Pipe/trace lanes: 0 = on (shipped), 1 = off.

/// PUZZLE NODE states — `◉ ◌ ○ ◍`: `0`=powered (the filled ringed neuron — the
/// shipped, RESTING look an empty bus shows), `1`=dead/cold (a hollow, unlit ring),
/// `2`=decoy (a plain hollow circle — present but inert), `3`=won (a dotted/charged
/// ring). The glyph alone separates a powered node from a dead one; the bridge also
/// lifts the colour. Member 0 == the shipped `◉`, so an unwired node is byte-identical.
pub const STATE_NODE: GlyphFamily = GlyphFamily {
    name: "state_node",
    kind: FamilyKind::WithinAnim,
    //        0 powered 1 dead   2 decoy  3 won
    glyphs: &['◉', '◌', '○', '◍'],
};

/// PUZZLE DOOR states — `▒ ▒ ▒ ▓`: the exit stays its quiet panel (`▒`) while the
/// puzzle is unsolved (states 0–2) and brightens to a solid `▓` charged gate when
/// WON (state 3). Member 0 == the shipped `▒`, so an unwired door is byte-identical.
pub const STATE_DOOR: GlyphFamily = GlyphFamily {
    name: "state_door",
    kind: FamilyKind::WithinAnim,
    //        0 idle   1 idle   2 idle   3 won
    glyphs: &['▒', '▒', '▒', '▓'],
};

/// PUZZLE PIPE / TRACE states — `─ · ─ ━`: a signal trace that reads as a solid rule
/// when its pipe is ON (states 0/2 → the shipped resting look) and a faint stipple
/// (`·`) when OFF (state 1), brightening to a heavy rule (`━`) when WON (state 3), so
/// a live connection is legible from a dead one. Member 0 (on) is the resting look.
pub const STATE_TRACE: GlyphFamily = GlyphFamily {
    name: "state_trace",
    kind: FamilyKind::WithinAnim,
    //        0 on     1 off    2 on     3 on(won)
    glyphs: &['─', '·', '─', '━'],
};

/// The whole curated library, in documentation order. A tool/test enumerates this
/// (`for f in glyph_family::LIBRARY`) — the single source of truth the unit tests
/// assert over (non-empty + reliable-alphabet).
pub const LIBRARY: &[GlyphFamily] = &[
    HAZE_O,
    HAZE_DOT,
    HAZE_WAVE,
    HAZE_COLON,
    HAZE_BAR,
    GROWTH_DOT,
    GROWTH_DENSITY,
    GROWTH_BARS,
    ANIM_SPIN,
    ANIM_ARC,
    ANIM_SWEEP,
    LEAN_FILL,
    LEAN_STUB,
    STATE_NODE,
    STATE_DOOR,
    STATE_TRACE,
];

/// Resolve a family by name from the [`LIBRARY`] (e.g. for a `family = NAME`
/// authoring path). `None` if unknown. Allocation-free.
pub fn family_by_name(name: &str) -> Option<GlyphFamily> {
    LIBRARY.iter().copied().find(|f| f.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_library_family_is_nonempty_and_reliable() {
        assert!(!LIBRARY.is_empty(), "library must have families");
        for f in LIBRARY {
            assert!(!f.is_empty(), "family {} is empty", f.name);
            for &g in f.glyphs {
                assert!(
                    is_reliable(g),
                    "family {} member U+{:04X} ('{}') is not reliable",
                    f.name,
                    g as u32,
                    g
                );
            }
        }
    }

    #[test]
    fn time_driven_neutral_family_cycles_floating_over_time() {
        // The FLOATING effect: a neutral family substitutes different members as
        // time advances, so a fixed cell shimmers. Over one period we must see
        // MORE THAN ONE distinct glyph (it is not pinned).
        let layer = FamilyLayer {
            family: HAZE_O,
            driver: FamilyDriver::Time,
            speed: 1.0,
        };
        let seed = cell_phase(2.3, 4.1);
        let mut seen = std::collections::BTreeSet::new();
        for k in 0..HAZE_O.len() {
            let d = FamilyDrive {
                time: k as f32, // speed 1.0 → one member per second
                ..FamilyDrive::ZERO
            };
            seen.insert(layer.select(&d, seed));
        }
        assert!(
            seen.len() > 1,
            "time-driven neutral family must cycle (saw {} glyphs)",
            seen.len()
        );
    }

    #[test]
    fn neighbouring_cells_desync() {
        // Two different world cells must (generally) pick different phases, so the
        // floating field doesn't pulse in lockstep.
        let layer = FamilyLayer {
            family: HAZE_DOT,
            driver: FamilyDriver::Time,
            speed: 1.0,
        };
        let s0 = cell_phase(0.1, 0.1);
        let s1 = cell_phase(9.7, 3.2);
        let d = FamilyDrive {
            time: 0.37,
            ..FamilyDrive::ZERO
        };
        // Not a hard guarantee for ALL pairs, but these two are chosen to differ.
        assert_ne!(s0, s1, "distinct cells should hash to distinct seeds");
        // The picked glyph may or may not differ; the SEED differing is the desync.
        let _ = (layer.select(&d, s0), layer.select(&d, s1));
    }

    #[test]
    fn distance_driven_grows_with_distance() {
        // GROWTH: nearer (distance01 ~0) → an early/small member; far (~1) → a
        // late/large member. The chosen index must be MONOTONE non-decreasing.
        let layer = FamilyLayer {
            family: GROWTH_DENSITY,
            driver: FamilyDriver::Distance,
            speed: 0.0,
        };
        let glyph_at = |t: f32| {
            let d = FamilyDrive {
                distance01: t,
                ..FamilyDrive::ZERO
            };
            layer.select(&d, 0)
        };
        let near = glyph_at(0.0);
        let mid = glyph_at(0.5);
        let far = glyph_at(1.0);
        assert_eq!(near, GROWTH_DENSITY.glyphs[0] as u32, "near = smallest");
        assert_eq!(
            far,
            *GROWTH_DENSITY.glyphs.last().unwrap() as u32,
            "far = largest"
        );
        // index(near) <= index(mid) <= index(far)
        let idx = |g: u32| {
            GROWTH_DENSITY
                .glyphs
                .iter()
                .position(|&c| c as u32 == g)
                .unwrap()
        };
        assert!(
            idx(near) <= idx(mid) && idx(mid) <= idx(far),
            "growth is monotone"
        );
    }

    #[test]
    fn normal_driven_positional_picks_by_lean() {
        // POSITIONAL: lean −1 → first (leftmost) member, +1 → last (rightmost),
        // 0 → a middle member. Picks differ across the lean range.
        let layer = FamilyLayer {
            family: LEAN_FILL,
            driver: FamilyDriver::Normal,
            speed: 0.0,
        };
        let pick = |lean: f32| {
            let d = FamilyDrive {
                lean,
                ..FamilyDrive::ZERO
            };
            layer.select(&d, 0)
        };
        let left = pick(-1.0);
        let centre = pick(0.0);
        let right = pick(1.0);
        assert_eq!(left, LEAN_FILL.glyphs[0] as u32, "lean -1 → leftmost");
        assert_eq!(
            right,
            *LEAN_FILL.glyphs.last().unwrap() as u32,
            "lean +1 → rightmost"
        );
        assert!(
            left != right && centre != left,
            "lean must select distinct members"
        );
    }

    #[test]
    fn state_driven_family_picks_distinct_glyphs_per_state() {
        // The STATE BUS look (spec §2): a `State`-driven family indexes its glyph by
        // the cell's live state word, so a powered node (state 1) and a dead node
        // (state 0) MUST read as different glyphs.
        let layer = FamilyLayer {
            family: STATE_NODE,
            driver: FamilyDriver::State,
            speed: 0.0,
        };
        let pick = |state: u32| {
            let d = FamilyDrive {
                state,
                ..FamilyDrive::ZERO
            };
            layer.select(&d, 0)
        };
        let powered = pick(0);
        let dead = pick(1);
        let won = pick(3);
        assert_ne!(dead, powered, "powered node must differ from a dead node");
        assert_eq!(
            powered, STATE_NODE.glyphs[0] as u32,
            "state 0 = powered/resting member"
        );
        assert_eq!(dead, STATE_NODE.glyphs[1] as u32, "state 1 = dead member");
        assert_eq!(won, STATE_NODE.glyphs[3] as u32, "state 3 = won member");
        // Out-of-range state clamps to the last member (never panics).
        assert_eq!(pick(99), *STATE_NODE.glyphs.last().unwrap() as u32);
    }

    #[test]
    fn time_driver_is_animated_others_are_not() {
        assert!(FamilyDriver::Time.is_animated());
        assert!(!FamilyDriver::Distance.is_animated());
        assert!(!FamilyDriver::Normal.is_animated());
        assert!(!FamilyDriver::State.is_animated());
    }

    #[test]
    fn pick_clamps_out_of_range() {
        assert_eq!(GROWTH_DOT.pick(-5.0), GROWTH_DOT.glyphs[0] as u32);
        assert_eq!(
            GROWTH_DOT.pick(5.0),
            *GROWTH_DOT.glyphs.last().unwrap() as u32
        );
    }
}
