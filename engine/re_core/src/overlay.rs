//! `overlay` — the engine-pure **Scalable Material Overlay** data types.
//!
//! This is the DATA shape of the semantic superstructure that rides *over* the
//! voxel model (`re_world::Map`). It carries (a) material **zones**, (b) crisp
//! scalable **lines**, (c) a per-zone/line **scale rule** (the slot where
//! "fractal" lives as one value), and (d) authored **insets** that override the
//! rule at a chosen octave/region.
//!
//! This module is **types + carrier fields + parse only**. There is NO resolve
//! logic, NO rendering, NO behavioural change here. An empty [`Overlay`] (the
//! [`Default`]) is the bottom rung of the degradation ladder: it must reproduce
//! today's frame byte-for-byte. Every primitive therefore derives [`Default`]
//! to an empty / inert value.
//!
//! Crate purity: this module is pure data — no terminal, no `re_render`, no
//! `bevy_render`. It lives in `re_core` exactly like [`crate::Framebuffer`], so
//! `re_world` (storage), `re_level` (parse), and later `re_render` (resolve) all
//! reference ONE source of truth.
//!
//! ## Decided design picks (spec §3 ▶ REC, recorded so the shape is not re-litigated)
//!
//! * **DP1 = D (hybrid), id-first.** A [`Region`] is an enum whose MVP variant is a
//!   voxel-cell **box** ([`Region::Box`]) — the discrete octave-0 shape that aligns
//!   with the existing `Map.side_data` channel and collision. The enum is left OPEN
//!   for a future analytic variant (see the `// Analytic(...) — P1+` placeholder);
//!   the analytic path is NOT implemented in P0.
//! * **DP4.** [`ScaleRule`] = `None | Crisp | SelfSimilar(SelfSimilarParams)`,
//!   `Default = None` (today's flat look). [`SelfSimilarParams`] is a plain POD that
//!   MIRRORS the tunable fields of `re_render::fractal::FractalParams` so a later
//!   phase can map POD → `FractalParams`. **`re_core` MUST NOT depend on
//!   `re_render`**, so the fields are duplicated here by value, not imported.
//! * **DP6-B.** An [`Inset`]'s content is a **referenced child scene**
//!   ([`InsetContent::ChildScene`]) — "an inset is just another level". No loading
//!   logic in P0; only the reference id/path is stored.
//! * **DP2-B (authoring shape).** A [`Line`]'s path is a simple [`LinePath`]
//!   polyline ([`Vec<Vec3>`]) for now. SDF lowering of strokes is **P2** — not done
//!   here.

use crate::Vec3;

/// Newtype: which authored material a [`Zone`] binds. A thin index/key over the
/// per-level material set (`re_world::Map::materials` / the `.lvl` `[material]`
/// sections). Kept as a `String` name for P0 so the overlay parse can carry the
/// authored material name *as written* and resolution-against-the-registry is
/// deferred to a later phase (mirrors how `SurfaceMaterials` carries `Option<String>`
/// names resolved in `finalize`). An empty name = "no binding / inherit".
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct MaterialId(pub String);

/// Newtype: which line-material a [`Line`] binds. Separate type from [`MaterialId`]
/// so the type system keeps zone-materials and line-materials from being confused,
/// even though both are name-keys for P0. An empty name = "no binding".
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct LineMaterialId(pub String);

/// Newtype: a reference from a [`Zone`] to one of an overlay's [`Inset`]s, by the
/// inset's authored `name`. Resolved against [`Overlay::insets`] in a later phase;
/// P0 only carries the reference. An empty name is never produced by the parser.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct InsetId(pub String);

/// How a zone / line behaves across zoom octaves — the **scale rule** (spec §2.1,
/// DP4). The slot where "fractal" lives as ONE value among `None`/`Crisp`.
///
/// `Default = None` is the bottom rung of the degradation ladder (§5): no scale
/// behaviour, the zone looks the same at voxel resolution — i.e. today's frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum ScaleRule {
    /// No scale behaviour — voxel-resolution, looks the same at any zoom. Today's
    /// look; the default so an authored-but-unscaled zone is byte-identical.
    #[default]
    None,
    /// Resolution-independent: the zone/edge never pixelates, but no NEW detail is
    /// generated. The minimal meaning of "scalable" (spec DP4 `Crisp`). No render
    /// behaviour is attached in P0 — this only records the authored intent.
    Crisp,
    /// The **fractal** case: self-similar detail repeats across octaves, driven by
    /// the carried [`SelfSimilarParams`] (a POD mirror of
    /// `re_render::fractal::FractalParams`). Procedural, infinite. No evaluation in
    /// P0 — the params are stored for a later phase to map into `FractalParams`.
    SelfSimilar(SelfSimilarParams),
}

/// A plain-old-data mirror of the tunable fields of `re_render::fractal::FractalParams`.
///
/// `re_core` MUST NOT depend on `re_render`, so the fractal shape parameters are
/// duplicated here BY VALUE. A later phase maps this POD → `FractalParams` at the
/// render boundary (`re_render` owns the conversion, never `re_core`). Keep the
/// field set in sync with `FractalParams`; the doc-comments below paraphrase it.
///
/// [`Default`] is a quiet field (no arms, no warp, no twist) — an inert spiral that
/// adds nothing if a `SelfSimilar` rule is authored without explicit params.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SelfSimilarParams {
    /// Spiral arms; `0` = pure concentric rings. Mirrors `FractalParams::arms`.
    pub arms: u8,
    /// fBm enrichment strength `[0,1]` (self-similar across octaves). Mirrors
    /// `FractalParams::octave_warp`.
    pub octave_warp: f32,
    /// Arm curl: shears angle by log-radius (log spiral); `0` = straight rays.
    /// Mirrors `FractalParams::twist`.
    pub twist: f32,
}

/// How a [`Zone`]'s (or [`Inset`]'s) area is defined (spec DP1).
///
/// P0 ships the **id-first hybrid (DP1 ▶ REC = D)**: the MVP variant is a discrete
/// voxel-cell box ([`Box`](Self::Box)) — the octave-0, collision-aligned shape that
/// matches the existing `Map.side_data` channel and the `.lvl` `region =
/// (x0,y0,z0)-(x1,y1,z1)` authoring syntax already used by `[zone_input]`.
///
/// The enum is intentionally OPEN for the future analytic variant (resolution-
/// independent SDF / half-plane / CSG) the spec calls for at recursion frontiers —
/// see the `Analytic` placeholder. That variant is **P1+** and is NOT implemented
/// here; `#[non_exhaustive]` keeps adding it a non-breaking change for downstream
/// `match`es.
///
/// [`Default`] is an empty box (`min == max == (0,0,0)`) — an inert region that
/// covers nothing, so a defaulted [`Zone`] resolves to nothing.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Region {
    /// An inclusive axis-aligned voxel-cell box `[min, max]` in octave-0 grid
    /// coordinates (`(x,y,z)`). The discrete MVP shape; aligns with collision and
    /// the `(x0,y0,z0)-(x1,y1,z1)` `.lvl` region syntax.
    Box {
        /// Inclusive minimum corner `(x0, y0, z0)`.
        min: (i32, i32, i32),
        /// Inclusive maximum corner `(x1, y1, z1)`.
        max: (i32, i32, i32),
    },
    // Analytic(AnalyticRegion) — P1+: a resolution-independent SDF / half-plane /
    // CSG region for crisp edges and zoom recursion (spec DP1-B / DP1-D). NOT
    // implemented in P0. Left as a comment so the intent is recorded without
    // shipping dead types; `#[non_exhaustive]` makes adding it non-breaking.
}

impl Default for Region {
    /// An empty (zero-extent) box at the origin — covers nothing.
    fn default() -> Self {
        Region::Box {
            min: (0, 0, 0),
            max: (0, 0, 0),
        }
    }
}

impl Region {
    /// True if `(x,y,z)` lies inside this region (octave-0 cell test). For a
    /// [`Box`](Self::Box) it is the inclusive AABB containment test. Pure data
    /// helper — NO resolve-in-cast-path logic; the renderer's resolve is a later
    /// phase. Provided so P0 unit tests (and future callers) have one containment
    /// definition.
    pub fn contains(&self, x: i32, y: i32, z: i32) -> bool {
        match self {
            Region::Box { min, max } => {
                let (x0, x1) = (min.0.min(max.0), min.0.max(max.0));
                let (y0, y1) = (min.1.min(max.1), min.1.max(max.1));
                let (z0, z1) = (min.2.min(max.2), min.2.max(max.2));
                x >= x0 && x <= x1 && y >= y0 && y <= y1 && z >= z0 && z <= z1
            }
        }
    }
}

/// The authoring shape of a [`Line`]: a simple world-space polyline (DP2-B).
///
/// P0 stores the raw polyline as authored. SDF lowering of strokes (so lines and
/// analytic zones share one sampler) is **P2** — not done here. [`Default`] is an
/// empty path (no points).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LinePath {
    /// Ordered world-space vertices of the polyline. Two or more points form a
    /// stroke; fewer is an inert/degenerate path (carried, never rendered in P0).
    pub points: Vec<Vec3>,
}

/// What an [`Inset`] ("врезка") substitutes at its (octave, region) — spec DP6-B
/// ▶ REC: a **referenced child scene** ("an inset is just another level").
///
/// P0 stores only the reference; there is NO loading logic. [`Default`] is an empty
/// child-scene reference (empty path), an inert placeholder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InsetContent {
    /// A reference to a child scene by id/path (e.g. a `.lvl` file). Loaded by a
    /// later phase (P4); P0 only carries the string. [`Default`] is this variant
    /// with an empty reference.
    ChildScene(String),
}

impl Default for InsetContent {
    /// An empty child-scene reference — an inert placeholder (no scene to load).
    fn default() -> Self {
        InsetContent::ChildScene(String::new())
    }
}

/// A bounded **material zone** (spec §2.1 primitive 1): a region carrying a material
/// binding, an optional [`ScaleRule`], and an optional list of [`Inset`] references.
///
/// [`Default`] is an unnamed, empty-region, unbound, [`ScaleRule::None`], no-inset
/// zone — inert.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Zone {
    /// Authoring name (the `[zone NAME]` section name). For readability / inset
    /// cross-reference; not required to be unique by P0.
    pub name: String,
    /// The zone's area (DP1; P0 = a voxel box).
    pub region: Region,
    /// The material this zone binds (name-key; empty ⇒ no binding).
    pub material: MaterialId,
    /// How the zone behaves across octaves (DP4; default `None` = today).
    pub scale: ScaleRule,
    /// References to [`Inset`]s (by name) that override the rule inside this zone.
    /// Empty ⇒ none.
    pub insets: Vec<InsetId>,
}

/// A first-class **line** (spec §2.1 primitive 2): a crisp authored stroke carrying
/// a line-material and a [`ScaleRule`].
///
/// [`Default`] is an unnamed, empty-path, unbound, [`ScaleRule::None`] line — inert.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Line {
    /// Authoring name (the `[line NAME]` section name).
    pub name: String,
    /// The stroke geometry (DP2-B polyline; SDF lowering is P2).
    pub path: LinePath,
    /// The line-material this stroke binds (name-key; empty ⇒ no binding).
    pub material: LineMaterialId,
    /// How the line behaves across octaves (DP4; default `None`).
    pub scale: ScaleRule,
}

/// An authored **inset ("врезка")** (spec §2.1 primitive 4): content that overrides
/// the procedural scale rule at a specific octave range + region.
///
/// `at_octave` is the inclusive octave range `(min, max)` at which the inset wins
/// over the rule (the "rule + insets" hybrid along the scale axis). [`Default`] is
/// an unnamed inset at octave `(0, 0)`, empty region, empty child-scene reference —
/// inert.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Inset {
    /// Authoring name (the `[inset NAME]` section name); referenced by [`InsetId`].
    pub name: String,
    /// Inclusive octave range `(min, max)` at which this inset overrides the rule.
    pub at_octave: (i32, i32),
    /// The region (at that octave) the inset covers.
    pub region: Region,
    /// What the inset substitutes (DP6-B referenced child scene). No loading in P0.
    pub content: InsetContent,
}

/// A **resolved inset** — the engine-pure, render-ready realization of one authored
/// [`Inset`] after `re_level::build_map` has loaded its child scene (P4).
///
/// This is the DATA CONTRACT that keeps `re_render` free of `.lvl` loading: `re_level`
/// owns the (cycle-guarded, one-level-deep) child-scene load and distils each inset to
/// just the three numbers the SelfSimilar shade needs — a region, a phase window, and a
/// representative colour. `re_render` consumes a `&[ResolvedInset]` and never touches a
/// level file.
///
/// ## Why a PHASE-BAND, not an octave NUMBER (the approved P4 honesty)
///
/// The infinite-zoom field (`re_render::fractal`) is PERIODIC over one octave
/// (`LN2` in log-radius) — that periodicity IS the seamless loop, so octave N is
/// byte-identical to octave N+1 and there is no "current octave" counter to key off.
/// An inset therefore cannot reveal at a distinguishable octave; instead it reveals
/// during a **phase-band** `[lo, hi)` of the reduced zoom phase in `[0, LN2)`. When the
/// cell is in [`region`](Self::region) AND the live `phase.zoom` falls in
/// [`phase_band`](Self::phase_band), the inset's colour overrides the fractal field —
/// then the loop carries the phase out of the band and the inset fades, reappearing once
/// per octave loop. (True accumulating zoom-depth + full child-scene geometry
/// compositing are DEFERRED.)
///
/// DETERMINISM (spec DP7): `phase_band` and `rgb` are fixed at build time from the
/// authored inset + child scene, so the same parent `.lvl` always resolves to the
/// identical `ResolvedInset`, and the override at a given `(coord, reduced phase)` is
/// identical across every loop.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedInset {
    /// The octave-0 cell region the inset covers (the same authored [`Inset::region`]).
    /// Membership is the inclusive AABB test [`Region::contains`].
    pub region: Region,
    /// The reveal window `[lo, hi)` in reduced zoom phase, each in `[0, LN2)`. The
    /// membership test handles wraparound (`lo > hi` ⇒ the band wraps past `LN2`), so a
    /// band may straddle the loop seam. Derived deterministically from
    /// [`Inset::at_octave`] (see `re_level::build_map`).
    pub phase_band: (f32, f32),
    /// Representative colour the inset paints, packed `0x00RR_GGBB`. Taken from the
    /// child scene's first material base colour (see `re_level::build_map` for the
    /// fallback chain). The SelfSimilar shade replaces the fractal field colour with
    /// this inside `region` during `phase_band`.
    pub rgb: u32,
}

impl ResolvedInset {
    /// True iff `reduced_phase` (a zoom phase already reduced into `[0, LN2)`) lies in
    /// this inset's [`phase_band`](Self::phase_band) `[lo, hi)`, handling the wraparound
    /// case where the band straddles the `LN2` loop seam (`lo > hi`).
    ///
    /// `ln2` is passed in (rather than imported) so `re_core` stays free of the render
    /// crate's `fractal::LN2` constant — the caller (`re_render`) supplies its own
    /// `LN2`. The two are the same `std::f32::consts::LN_2` value.
    #[inline]
    pub fn phase_in_band(&self, reduced_phase: f32, _ln2: f32) -> bool {
        let (lo, hi) = self.phase_band;
        if lo <= hi {
            reduced_phase >= lo && reduced_phase < hi
        } else {
            // Wrapped band: [lo, LN2) ∪ [0, hi).
            reduced_phase >= lo || reduced_phase < hi
        }
    }
}

/// The **Scalable Material Overlay** (spec §2): the optional semantic superstructure
/// over a voxel [`crate::layer`]-free `re_world::Map`. Three parallel collections of
/// the four primitives ([`ScaleRule`] is carried *inside* zones/lines, not as a
/// separate collection).
///
/// [`Default`] is **all-empty** — the bottom rung of the degradation ladder (§5).
/// An empty overlay carries no zones, no lines, no insets, so the resolve/render of
/// a later phase has nothing to do and the frame is byte-identical to today. Every
/// existing level (no `[zone]`/`[line]`/`[inset]` sections) parses to exactly this.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Overlay {
    /// Material zones (spec primitive 1). Empty ⇒ no zones.
    pub zones: Vec<Zone>,
    /// Crisp scalable lines (spec primitive 2). Empty ⇒ no lines.
    pub lines: Vec<Line>,
    /// Authored insets (spec primitive 4). Empty ⇒ no insets.
    pub insets: Vec<Inset>,
}

impl Overlay {
    /// True if the overlay carries no zones, no lines, and no insets — the inert
    /// state in which a later phase's resolve/render is a no-op and the frame is
    /// byte-identical to today. Equivalent to `*self == Overlay::default()` for the
    /// empty case, but O(1) and intent-revealing.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.zones.is_empty() && self.lines.is_empty() && self.insets.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlay_default_is_all_empty() {
        let o = Overlay::default();
        assert!(o.zones.is_empty());
        assert!(o.lines.is_empty());
        assert!(o.insets.is_empty());
        assert!(o.is_empty());
        // The inert default is the byte-identical bottom rung of the ladder.
        assert_eq!(o, Overlay::default());
    }

    #[test]
    fn scale_rule_default_is_none() {
        assert_eq!(ScaleRule::default(), ScaleRule::None);
        // SelfSimilar carries the POD mirror of FractalParams.
        let p = SelfSimilarParams {
            arms: 5,
            octave_warp: 0.3,
            twist: 0.9,
        };
        assert_eq!(ScaleRule::SelfSimilar(p), ScaleRule::SelfSimilar(p));
        assert_ne!(ScaleRule::SelfSimilar(p), ScaleRule::None);
    }

    #[test]
    fn region_default_is_empty_box_covering_nothing() {
        let r = Region::default();
        assert_eq!(
            r,
            Region::Box {
                min: (0, 0, 0),
                max: (0, 0, 0)
            }
        );
        // A zero-extent box still contains its single origin cell, nothing else.
        assert!(r.contains(0, 0, 0));
        assert!(!r.contains(1, 0, 0));
    }

    #[test]
    fn region_box_contains_is_inclusive_and_corner_order_agnostic() {
        // Authored corners may be given in either order; contains normalises them.
        let r = Region::Box {
            min: (3, 1, 0),
            max: (1, 4, 2),
        };
        assert!(r.contains(1, 1, 0));
        assert!(r.contains(3, 4, 2));
        assert!(r.contains(2, 2, 1));
        assert!(!r.contains(0, 2, 1));
        assert!(!r.contains(2, 5, 1));
    }

    #[test]
    fn resolved_inset_phase_band_membership_handles_wraparound() {
        let ln2 = std::f32::consts::LN_2;
        let region = Region::Box {
            min: (1, 0, 1),
            max: (3, 0, 3),
        };
        // A NON-wrapped band [0.1, 0.4): in iff lo <= phase < hi.
        let a = ResolvedInset {
            region: region.clone(),
            phase_band: (0.1, 0.4),
            rgb: 0x010203,
        };
        assert!(!a.phase_in_band(0.05, ln2));
        assert!(a.phase_in_band(0.1, ln2)); // inclusive lo
        assert!(a.phase_in_band(0.39, ln2));
        assert!(!a.phase_in_band(0.4, ln2)); // exclusive hi
                                             // A WRAPPED band [0.6, 0.1): in iff phase >= 0.6 OR phase < 0.1.
        let b = ResolvedInset {
            region,
            phase_band: (0.6, 0.1),
            rgb: 0x010203,
        };
        assert!(b.phase_in_band(0.65, ln2));
        assert!(b.phase_in_band(0.05, ln2)); // wrapped low end
        assert!(!b.phase_in_band(0.3, ln2)); // the gap
        assert!(!b.phase_in_band(0.1, ln2)); // exclusive wrapped hi
    }

    #[test]
    fn primitives_default_inert() {
        assert!(Zone::default().name.is_empty());
        assert_eq!(Zone::default().scale, ScaleRule::None);
        assert!(Zone::default().insets.is_empty());
        assert!(Line::default().path.points.is_empty());
        assert_eq!(Inset::default().at_octave, (0, 0));
        assert_eq!(
            Inset::default().content,
            InsetContent::ChildScene(String::new())
        );
    }
}
