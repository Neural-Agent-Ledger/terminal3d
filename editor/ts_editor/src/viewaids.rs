//! THE VIEW AID — **onion skin** (2D).
//!
//! Borrowed from Aseprite: ghost the layer(s)
//! BELOW *and* ABOVE the one being edited into the 2D pane, so a stairwell /
//! shaft / lift shaft lines up without flipping z back and forth.
//!
//! It is **pure view state**. Nothing here mutates [`re_world::Map`], nothing
//! here touches `crate::state::EditorState::level`, and therefore nothing here
//! can reach `io::save` / `re_lvlcheck` — [`onion_ghost`] only ever READS the
//! map, and the painter turns what it returns into colour.
//!
//! ## Why colour and not glyphs
//!
//! A GPU editor ghosts a layer with alpha. A glyph grid has three channels —
//! glyph, fg, bg — and the glyph channel is already fully spent saying *what a
//! cell is* (`#` wall, `_` floor, `%` matrix, an entity marker, …). Substituting
//! a ghost glyph would therefore destroy the one piece of information the ghost
//! exists to carry. So a ghost keeps its kind's glyph and is separated purely in
//! COLOUR: dimmed by distance and pulled halfway toward a direction tint — warm
//! amber for BELOW, cool blue for ABOVE. That also keeps a ghost away from the
//! cursor's white-on-indigo, which is the other thing it must never be confused
//! with.
//!
//! ## What a ghost is NOT
//!
//! A ghost is paint, never a target. The hit-test (`crate::view::screen_to_cell`
//! → `state.z`) never learns that onion skin exists, so clicking a cell that is
//! occupied only on an adjacent layer edits the LIVE layer, exactly as it does
//! with the aid off. `chrome::draw_grid` also only ever consults a ghost when the
//! live cell is `Empty`, so a ghost can never paint over real geometry.
use re_world::{CellKind, Map};

// ─── ONION SKIN ──────────────────────────────────────────────────────────────

/// How many layers either side of the edit layer the onion skin ghosts.
///
/// Configurable depth rather than a fixed below+above pair: a two-storey shaft is
/// exactly the case where one layer of context is not enough, and the extra state
/// costs one `match` arm. Cycled through [`ONION_MODES`] by `G`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum OnionSkin {
    /// No ghosting — `chrome::resolve_cell` runs its original path untouched.
    #[default]
    Off,
    /// Ghost `z-1` and `z+1`.
    One,
    /// Ghost `z-2 ..= z+2`.
    Two,
}

/// The cycle order `G` steps through — the const-table idiom this crate uses for
/// every other cycled option (`state::SHAPE_CYCLE`, `staircase::StairSub`, …).
/// `Off` is first, so a fresh editor starts with the aid OFF.
pub const ONION_MODES: [OnionSkin; 3] = [OnionSkin::Off, OnionSkin::One, OnionSkin::Two];

impl OnionSkin {
    /// Layers ghosted either side of the edit layer (`0` when off).
    pub const fn depth(self) -> i32 {
        match self {
            Self::Off => 0,
            Self::One => 1,
            Self::Two => 2,
        }
    }

    /// Short display label for the 2D viewport header.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::One => "\u{00B1}1",
            Self::Two => "\u{00B1}2",
        }
    }

    /// `true` while the aid is doing anything at all.
    pub const fn is_on(self) -> bool {
        !matches!(self, Self::Off)
    }
}

/// Step the onion-skin depth by `delta` (±1) through [`ONION_MODES`], wrapping.
/// Mirrors `state::cycle_shape`.
pub fn cycle_onion(current: OnionSkin, delta: i32) -> OnionSkin {
    let n = ONION_MODES.len() as i32;
    let idx = ONION_MODES.iter().position(|&m| m == current).unwrap_or(0) as i32;
    let next = (idx + delta).rem_euclid(n) as usize;
    ONION_MODES[next]
}

/// One ghosted cell found on a neighbouring layer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Ghost {
    /// What sits there (never [`CellKind::Empty`]).
    pub kind: CellKind,
    /// Signed layer offset from the edit layer: negative = BELOW, positive =
    /// ABOVE. Never `0` — the live layer is not a ghost.
    pub dz: i32,
}

/// The nearest occupied neighbouring layer within the onion-skin depth, or
/// `None`.
///
/// Searched by increasing distance, BELOW before ABOVE at equal distance — the
/// same downward bias the pre-existing height overlay has, so turning the aid on
/// never *flips* a ghost the author was already reading. Returns `None`
/// immediately when the aid is off, which is what makes the off path in
/// `chrome::resolve_cell` byte-identical to the code before this module existed.
///
/// The caller must only ask about a cell whose LIVE layer is empty; this function
/// does not look at `z` itself, and a ghost must never cover real geometry.
pub fn onion_ghost(map: &Map, x: i32, y: i32, z: i32, skin: OnionSkin) -> Option<Ghost> {
    let depth = skin.depth();
    for d in 1..=depth {
        for dz in [-d, d] {
            let kind = map.cell_at(x, y, z + dz);
            if kind != CellKind::Empty {
                return Some(Ghost { kind, dz });
            }
        }
    }
    None
}

/// Ghost tint for a layer BELOW the edit layer — a warm amber, the "buried"
/// direction.
pub const ONION_BELOW_TINT: u32 = 0x00_8A5A20;
/// Ghost tint for a layer ABOVE the edit layer — a cool blue, the "overhead"
/// direction. Warm/cool is the fastest hue pair to tell apart at low luminance.
pub const ONION_ABOVE_TINT: u32 = 0x00_2E62A8;

/// Per-channel scale of an `0x00RRGGBB` colour.
const fn scale(color: u32, num: u32, den: u32) -> u32 {
    let red = ((color >> 16) & 0xFF) * num / den;
    let green = ((color >> 8) & 0xFF) * num / den;
    let blue = (color & 0xFF) * num / den;
    (red << 16) | (green << 8) | blue
}

/// Per-channel midpoint of two `0x00RRGGBB` colours.
const fn mix_half(lhs: u32, rhs: u32) -> u32 {
    let red = u32::midpoint((lhs >> 16) & 0xFF, (rhs >> 16) & 0xFF);
    let green = u32::midpoint((lhs >> 8) & 0xFF, (rhs >> 8) & 0xFF);
    let blue = u32::midpoint(lhs & 0xFF, rhs & 0xFF);
    (red << 16) | (green << 8) | blue
}

/// The foreground a ghost is painted in: the cell kind's own colour dimmed by
/// `1/(|dz|+1)` (the same distance ladder the height overlay already used) and
/// then pulled HALFWAY toward the direction tint.
///
/// Two guarantees this buys, both asserted below:
/// * a ghost is strictly darker than the live colour of the same kind, so it can
///   never be read as a cell on the edit layer;
/// * a below-ghost and an above-ghost of the same kind at the same distance are
///   different colours, so "there is a floor under me" and "there is a floor over
///   me" do not look alike.
pub const fn ghost_fg(kind_fg: u32, dz: i32) -> u32 {
    let d = dz.unsigned_abs();
    let dimmed = scale(kind_fg, 1, d + 1);
    let tint = if dz < 0 {
        ONION_BELOW_TINT
    } else {
        ONION_ABOVE_TINT
    };
    mix_half(dimmed, tint)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An 8×8×4 test world: a solid slab on every layer, so a ghost from any
    /// neighbouring layer is unambiguous.
    fn slab_map() -> Map {
        let mut m = Map::new(8, 8, 4);
        for z in 0..4 {
            for y in 0..8 {
                for x in 0..8 {
                    m.set_cell(x, y, z, CellKind::WallSolid, 0);
                }
            }
        }
        m
    }

    #[test]
    fn onion_is_off_by_default_and_cycles_off_pm1_pm2() {
        assert_eq!(OnionSkin::default(), OnionSkin::Off);
        assert_eq!(ONION_MODES[0], OnionSkin::Off);
        let mut m = OnionSkin::default();
        m = cycle_onion(m, 1);
        assert_eq!(m, OnionSkin::One);
        m = cycle_onion(m, 1);
        assert_eq!(m, OnionSkin::Two);
        m = cycle_onion(m, 1);
        assert_eq!(m, OnionSkin::Off, "the cycle wraps back to off");
        assert_eq!(
            cycle_onion(OnionSkin::Off, -1),
            OnionSkin::Two,
            "backwards too"
        );
        assert_eq!(OnionSkin::Off.depth(), 0);
        assert_eq!(OnionSkin::Two.depth(), 2);
        assert!(!OnionSkin::Off.is_on());
        assert!(OnionSkin::One.is_on());
    }

    #[test]
    fn onion_off_never_reports_a_ghost() {
        let m = slab_map();
        for z in 0..4 {
            assert_eq!(onion_ghost(&m, 3, 3, z, OnionSkin::Off), None);
        }
    }

    #[test]
    fn onion_finds_below_and_above_within_depth() {
        let mut m = Map::new(4, 4, 5);
        m.set_cell(1, 1, 0, CellKind::Floor, 0); // 2 below z=2
        m.set_cell(1, 1, 3, CellKind::WallSolid, 0); // 1 above z=2
                                                     // Depth 1 sees only the layer above (the floor is 2 down).
        let g = onion_ghost(&m, 1, 1, 2, OnionSkin::One).expect("z+1 is occupied");
        assert_eq!(g.dz, 1);
        assert_eq!(g.kind, CellKind::WallSolid);
        // Depth 2 still prefers the NEARER neighbour.
        let g = onion_ghost(&m, 1, 1, 2, OnionSkin::Two).expect("still the nearer one");
        assert_eq!(g.dz, 1);
        // From z=1 the floor is 1 below and the wall 2 above: below wins.
        let g = onion_ghost(&m, 1, 1, 1, OnionSkin::Two).expect("z-1 is occupied");
        assert_eq!(g.dz, -1);
        assert_eq!(g.kind, CellKind::Floor);
    }

    #[test]
    fn onion_ties_prefer_below() {
        let mut m = Map::new(4, 4, 5);
        m.set_cell(1, 1, 1, CellKind::Floor, 0);
        m.set_cell(1, 1, 3, CellKind::WallSolid, 0);
        let g = onion_ghost(&m, 1, 1, 2, OnionSkin::One).expect("both sides occupied");
        assert_eq!(
            g.dz, -1,
            "equal distance keeps the pre-existing downward bias"
        );
    }

    #[test]
    fn onion_never_looks_past_its_depth() {
        let mut m = Map::new(4, 4, 8);
        m.set_cell(1, 1, 7, CellKind::WallSolid, 0);
        assert_eq!(
            onion_ghost(&m, 1, 1, 3, OnionSkin::Two),
            None,
            "4 away, depth 2"
        );
    }

    #[test]
    fn onion_ghost_is_out_of_bounds_safe() {
        let m = slab_map();
        assert_eq!(onion_ghost(&m, -5, -5, 1, OnionSkin::Two), None);
        assert_eq!(
            onion_ghost(&m, 3, 3, 0, OnionSkin::Two).map(|g| g.dz),
            Some(1),
            "below z=0 is void, so the ghost comes from above"
        );
    }

    #[test]
    fn a_ghost_is_always_darker_than_the_live_colour() {
        let lum = |c: u32| ((c >> 16) & 0xFF) + ((c >> 8) & 0xFF) + (c & 0xFF);
        for base in [0x00_C8C8C8u32, 0x00_6688AA, 0x00_33CC55, 0x00_FFCC44] {
            for dz in [-2i32, -1, 1, 2] {
                let g = ghost_fg(base, dz);
                assert!(
                    lum(g) < lum(base),
                    "ghost {g:06X} must be dimmer than live {base:06X} (dz={dz})"
                );
            }
        }
    }

    #[test]
    fn below_and_above_ghosts_are_different_colours() {
        for base in [0x00_C8C8C8u32, 0x00_6688AA] {
            assert_ne!(
                ghost_fg(base, -1),
                ghost_fg(base, 1),
                "direction must read off the colour alone"
            );
        }
        // Warm below / cool blue above: the channels separate them.
        let below = ghost_fg(0x00_C8C8C8, -1);
        let above = ghost_fg(0x00_C8C8C8, 1);
        assert!(below & 0xFF < above & 0xFF, "above is the bluer of the two");
        assert!(
            (below >> 16) & 0xFF > (above >> 16) & 0xFF,
            "below is the redder"
        );
    }

    #[test]
    fn a_farther_ghost_is_dimmer_than_a_nearer_one() {
        let lum = |c: u32| ((c >> 16) & 0xFF) + ((c >> 8) & 0xFF) + (c & 0xFF);
        assert!(lum(ghost_fg(0x00_C8C8C8, -2)) < lum(ghost_fg(0x00_C8C8C8, -1)));
        assert!(lum(ghost_fg(0x00_C8C8C8, 2)) < lum(ghost_fg(0x00_C8C8C8, 1)));
    }
}

/// PAINTED-OUTPUT tests: what the aid actually puts on screen, plus the
/// byte-identity golden that proves an editor with the aid OFF draws exactly
/// what the editor drew before this module existed.
#[cfg(test)]
mod paint_tests {
    use super::*;
    use crate::chrome::{RULER_H, RULER_W};
    use crate::state::EditorState;
    use re_core::Framebuffer;

    /// FNV-1a over every cell's `(glyph, fg, bg)` — a terminal-independent digest
    /// of a painted framebuffer, the same idea as `tests/golden/`.
    fn digest(fb: &Framebuffer) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for c in fb.cells() {
            for v in [c.glyph, c.fg, c.bg] {
                for b in v.to_le_bytes() {
                    h ^= u64::from(b);
                    h = h.wrapping_mul(0x0000_0100_0000_01B3);
                }
            }
        }
        h
    }

    /// THE golden fixture: an 8×8×4 world with a walled shell on z0..z2, a floor
    /// on z0, a pillar that exists ONLY on z2 (above the edit layer) and one that
    /// exists only on z0 (below it). Edit layer z=1, zoom 1, cursor (2,2).
    fn fixture() -> EditorState {
        let mut s = EditorState::new_empty(8, 8, 4);
        for z in 0..4usize {
            for x in 0..8usize {
                for y in 0..8usize {
                    let edge = x == 0 || y == 0 || x == 7 || y == 7;
                    if edge && z < 3 {
                        s.map.set_cell(x, y, z, CellKind::WallSolid, 0);
                    } else if z == 0 {
                        s.map.set_cell(x, y, z, CellKind::Floor, 0);
                    }
                }
            }
        }
        s.map.set_cell(3, 3, 2, CellKind::WallSolid, 0); // only ABOVE the edit layer
        s.map.set_cell(5, 5, 0, CellKind::WallSolid, 0); // only BELOW it
        s.z = 1;
        s.cursor = (2, 2);
        s.zoom = 1;
        s
    }

    /// Framebuffer coordinate of world cell `(wx, wy)` in a `draw_grid` at origin,
    /// zoom 1, pan 0 — the painter's own placement (`cells_x0 + vx * zoom`).
    fn at(wx: usize, wy: usize) -> (usize, usize) {
        (RULER_W + wx, RULER_H + wy)
    }

    fn grid_of(state: &EditorState) -> Framebuffer {
        let mut fb = Framebuffer::new(40, 20);
        crate::chrome::draw_grid(state, &mut fb, 0, 0, 40, 20);
        fb
    }

    /// ACCEPTANCE — OFF IS BYTE-IDENTICAL TO THE EDITOR BEFORE THIS FEATURE.
    ///
    /// These three digests were captured by running the SAME fixture through the
    /// SAME painters at commit `f0d3d28`, i.e. before `viewaids` existed. A
    /// default editor must still produce them exactly. Any drift — a stray badge
    /// in a caption, a ghost that leaks in with the aid off, a colour nudged "by
    /// one" — fails here.
    ///
    /// One digest per PAINTER, never a combined one: the grid pane, the 3D
    /// preview and the two viewport captions are pinned separately, so a change
    /// to one of them cannot hide inside another's number.
    ///
    /// They are NOT to be refreshed to make a change pass: the aid is off by
    /// default, so nothing legitimate can move these numbers.
    const GRID_DIGEST_BEFORE: u64 = 0xBB87_B6BA_4761_EDDE;
    // PREVIEW_DIGEST_BEFORE re-pinned 2026-09-23: the 3D
    // preview goes through the REAL `re_render` raycast, and a default shading
    // path change (same root cause
    // that moved every one of the 33 `golden_framebuffer` shade hashes — see
    // `tests/golden/README.md`). GRID_DIGEST_BEFORE and HEADER_DIGEST_BEFORE
    // are untouched, which confirms this is a shading-only drift in the
    // shared renderer, not a regression in the onion-skin feature this test
    // actually guards. Old value: 0xD069_3E77_9695_70D7.
    //
    // Re-pinned again 2026-09-24 (set-dressing round) from 0x3B1E_E334_E537_8A35,
    // same shape of cause: the shared raycast gained the vertical all-height
    // pass (the band above the eye slice is now drawn geometry, not backdrop)
    // and the legibility floor (`re_render/src/legibility.rs`) lifts the
    // near-black side wall (bg 0x010205 -> ~0x171F3B) to a readable dim blue
    // with its panel texture. Verified by rendering the pinned and the new
    // 40x20 preview side by side. GRID_DIGEST_BEFORE and HEADER_DIGEST_BEFORE
    // still match, so the onion-skin-off guarantee this test owns is intact.
    //
    // Re-pinned 2026-09-24 (distance depth effect default-on) from
    // 0x358C_27D1_1BB5_F38D: `DepthContrastConfig::from_env()` now seeds
    // `enabled: true`. Verified the old digest still passes under `TS_DEPTH=0`,
    // so the drift is that shading change alone; grid/header digests unchanged.
    //
    // Re-pinned 2026-09-24 (b) (dark-strip / node-dimming / short-band-mottling /
    // TX5-genplate fix round) from 0x5CA7_F2C5_7DC1_565D: the shared raycast's floor/ceiling loop
    // (`cast_column_depth`, `crate::clamp_to_wall_depth`) now clamps a floor/
    // ceiling row's depth to a nearer wall's own perpendicular depth in the same
    // column — the dark-strip fix — which is a GEOMETRY (depth) change, present in
    // essentially any scene with a wall standing in front of floor, so the shared
    // preview raycast necessarily re-hashes. GRID_DIGEST_BEFORE and
    // HEADER_DIGEST_BEFORE (neither of which re-casts the scene) still match,
    // confirming the onion-skin-off guarantee this test owns is intact and the
    // drift is confined to the raycast this fix targeted.
    //
    // Re-pinned (c) (found while reviewing the dark-strip fix's own contact
    // sheet) from 0x9F74_FCF9_0E34_B3E3: two
    // further GEOMETRY/shading changes to the same shared raycast, both confined
    // to `crates/re_render`. First, `contour.rs`'s `classify_edge` now draws a
    // wall→floor contact line unconditionally off the SURFACE-KIND change instead
    // of the depth jump the clamp above collapsed (restores a contact edge the
    // clamp had silently deleted). Second, `clamp_to_wall_depth` itself is now
    // WINDOWED (`FLOOR_CLAMP_MAX_DENOM_PAST_WALL`) rather than unconditional — it
    // only bounds a row within a small denom-distance of the wall's own crossover,
    // instead of flattening a near wall's entire remaining floor band to one
    // constant depth (a regression the unconditional version introduced,
    // reported against `L1-first-pulse.lvl` / `08-material-demo.lvl`). Both are
    // present in essentially any scene with a wall standing in front of floor, so
    // the shared preview raycast re-hashes again. GRID_DIGEST_BEFORE and
    // HEADER_DIGEST_BEFORE still match, same as the prior re-pin.
    const PREVIEW_DIGEST_BEFORE: u64 = 0xC75B_1665_8532_5082;
    const HEADER_DIGEST_BEFORE: u64 = 0x228E_F1A3_D8F1_BCF5;

    #[test]
    fn with_the_aid_off_every_painter_is_byte_identical_to_before_the_feature() {
        let s = fixture();
        assert_eq!(s.onion, OnionSkin::Off, "onion skin is off by default");

        assert_eq!(
            digest(&grid_of(&s)),
            GRID_DIGEST_BEFORE,
            "the 2D pane drifted from the pre-feature painting"
        );

        let mut fb = Framebuffer::new(40, 20);
        crate::preview::render_preview(&s, &mut fb, 0, 0, 40, 20);
        assert_eq!(
            digest(&fb),
            PREVIEW_DIGEST_BEFORE,
            "the 3D preview drifted from the pre-feature raycast"
        );

        let mut fb = Framebuffer::new(60, 12);
        let row = |y: usize| crate::layout::Rect {
            x: 0,
            y,
            w: 60,
            h: 1,
        };
        crate::chrome::draw_grid_header(&s, &mut fb, row(0), false, false);
        crate::chrome::draw_preview_header(&s, &mut fb, row(2), false, false);
        assert_eq!(
            digest(&fb),
            HEADER_DIGEST_BEFORE,
            "a viewport caption gained a badge it must not show while the aid is off"
        );
    }

    /// The other half of "off changes nothing": geometry that exists ONLY above
    /// the edit layer contributes nothing to the pane with the aid off.
    #[test]
    fn with_onion_off_the_layer_above_is_invisible() {
        let base = fixture();
        let mut more = fixture();
        for x in 1..7 {
            for y in 1..7 {
                more.map.set_cell(x, y, 2, CellKind::WallSolid, 0);
            }
        }
        assert_eq!(
            digest(&grid_of(&base)),
            digest(&grid_of(&more)),
            "with the aid off, a whole extra storey above must not change one cell"
        );
    }

    // ─── ONION SKIN, ON ──────────────────────────────────────────────────────

    #[test]
    fn onion_on_paints_the_layer_above_as_a_dim_blue_ghost() {
        let mut s = fixture();
        // (3,3) is empty on the edit layer z=1 and solid on z=2. Clear the z=0
        // floor under it too, so the ONLY neighbour is the one ABOVE — otherwise
        // the equal-distance downward bias (rightly) claims the cell.
        s.map.set_cell(3, 3, 0, CellKind::Empty, 0);
        assert_eq!(s.map.cell_at(3, 3, 1), CellKind::Empty);
        let (px, py) = at(3, 3);

        let off = grid_of(&s).get(px, py).copied().expect("in pane");
        s.onion = OnionSkin::One;
        let on = grid_of(&s).get(px, py).copied().expect("in pane");

        assert_ne!(off, on, "the aid must actually change the cell");
        assert_eq!(on.glyph, '#' as u32, "a ghost keeps its KIND's glyph");
        let (lx, ly) = at(0, 3);
        let live = grid_of(&s)
            .get(lx, ly)
            .copied()
            .expect("shell wall in pane");
        assert_eq!(live.glyph, '#' as u32, "the shell wall is the same kind");
        let lum = |c: u32| ((c >> 16) & 0xFF) + ((c >> 8) & 0xFF) + (c & 0xFF);
        assert!(
            lum(on.fg) < lum(live.fg),
            "the ghost ({:06X}) must be dimmer than the live wall ({:06X})",
            on.fg,
            live.fg
        );
        assert_eq!(on.fg, ghost_fg(0x00_C8C8C8, 1), "the above tint");
        assert!(
            on.fg & 0xFF > (on.fg >> 16) & 0xFF,
            "above ghosts read blue"
        );
        assert_eq!(
            on.bg, 0,
            "a ghost paints no background — the cursor owns that"
        );
    }

    #[test]
    fn onion_on_paints_the_layer_below_in_the_other_hue() {
        let mut s = fixture();
        s.onion = OnionSkin::One;
        // Clear the z=0 floor under (3,3) so that probe has a neighbour only ABOVE.
        s.map.set_cell(3, 3, 0, CellKind::Empty, 0);
        // (5,5): empty on z=1 and z=2, solid on z=0 (one below).
        let (px, py) = at(5, 5);
        let below = grid_of(&s).get(px, py).copied().expect("in pane");
        // (3,3): empty on z=1, solid on z=2 (one above).
        let (ax, ay) = at(3, 3);
        let above = grid_of(&s).get(ax, ay).copied().expect("in pane");
        assert_eq!(below.glyph, '#' as u32);
        assert_eq!(above.glyph, '#' as u32);
        assert_ne!(
            below.fg, above.fg,
            "below and above must not be the same colour — the direction is the point"
        );
        assert_eq!(below.fg, ghost_fg(0x00_C8C8C8, -1));
        assert!(
            (below.fg >> 16) & 0xFF > below.fg & 0xFF,
            "below ghosts read warm"
        );
    }

    #[test]
    fn a_ghost_never_paints_over_a_live_cell() {
        let mut s = fixture();
        // (2,4) solid on the EDIT layer and also on the layer above.
        s.map.set_cell(2, 4, 1, CellKind::WallSolid, 0);
        s.map.set_cell(2, 4, 2, CellKind::Floor, 0);
        let (px, py) = at(2, 4);
        let off = grid_of(&s).get(px, py).copied().expect("in pane");
        s.onion = OnionSkin::Two;
        let on = grid_of(&s).get(px, py).copied().expect("in pane");
        assert_eq!(off, on, "a live cell paints identically with the aid on");
        assert_eq!(
            on.glyph, '#' as u32,
            "and it is still the LIVE cell's glyph"
        );
    }

    #[test]
    fn a_ghost_is_never_painted_as_the_cursor() {
        let mut s = fixture();
        s.cursor = (3, 3); // a cell that is a GHOST at z=1
        s.onion = OnionSkin::One;
        let (px, py) = at(3, 3);
        let c = grid_of(&s).get(px, py).copied().expect("in pane");
        assert_eq!(
            c.fg, 0x00_FFFFFF,
            "the cursor keeps its high-contrast white"
        );
        assert_ne!(c.bg, 0, "and its highlight background");
    }

    #[test]
    fn depth_two_reaches_a_layer_depth_one_cannot() {
        let mut s = fixture();
        s.z = 0;
        s.cursor = (0, 0);
        // (3,3) is solid on z=2 only — two above the edit layer, and empty on z=0.
        assert_eq!(s.map.cell_at(3, 3, 0), CellKind::Floor);
        s.map.set_cell(3, 3, 0, CellKind::Empty, 0);
        let (px, py) = at(3, 3);
        s.onion = OnionSkin::One;
        let near = grid_of(&s).get(px, py).copied().expect("in pane");
        s.onion = OnionSkin::Two;
        let far = grid_of(&s).get(px, py).copied().expect("in pane");
        assert_ne!(near, far, "+/-2 must reach a storey +/-1 does not");
        assert_eq!(far.fg, ghost_fg(0x00_C8C8C8, 2));
    }

    fn header_text(fb: &Framebuffer, y: usize, w: usize) -> String {
        (0..w)
            .filter_map(|x| fb.get(x, y).and_then(|c| char::from_u32(c.glyph)))
            .collect::<String>()
            .trim_end()
            .to_string()
    }

    #[test]
    fn the_2d_header_shows_the_onion_depth() {
        let read = |s: &EditorState| -> String {
            let mut fb = Framebuffer::new(60, 1);
            let r = crate::layout::Rect {
                x: 0,
                y: 0,
                w: 60,
                h: 1,
            };
            crate::chrome::draw_grid_header(s, &mut fb, r, false, false);
            header_text(&fb, 0, 60)
        };
        let mut s = fixture();
        assert!(
            !read(&s).contains("onion"),
            "silent while off: {}",
            read(&s)
        );
        s.onion = OnionSkin::One;
        assert!(
            read(&s).contains("onion \u{00B1}1"),
            "header must say so: {}",
            read(&s)
        );
        s.onion = OnionSkin::Two;
        assert!(
            read(&s).contains("onion \u{00B1}2"),
            "and the depth: {}",
            read(&s)
        );
    }

    // ─── ONION SKIN IS PAINT, NOT A TARGET ───────────────────────────────────

    /// ACCEPTANCE — A GHOST IS NOT CLICKABLE.
    ///
    /// (3,3) is empty on the edit layer z=1 and solid on z=2, so with the aid on
    /// it is painted as a ghost. Clicking it must paint the LIVE layer, exactly
    /// as it does with the aid off — the hit-test never learns ghosts exist.
    #[test]
    fn clicking_a_ghost_cell_edits_the_live_layer() {
        use re_present::{EditorMouse, EditorMouseButton, EditorMouseKind};

        let mut state = fixture();
        state.onion = OnionSkin::Two;
        state.view_mode = crate::state::ViewMode::Full2D;
        state.brush = CellKind::Floor;
        let (fbw, fbh) = (120usize, 40usize);
        let panes = crate::layout::compute(state.view_mode, fbw, fbh, false);
        // The SAME scene rect the painter draws into and the hit-test inverts.
        let scene = crate::mouse::grid_scene_rect(panes.grid);
        let px = (scene.x + RULER_W + 3) as u16;
        let py = (scene.y + RULER_H + 3) as u16;

        let above_before = state.map.cell_at(3, 3, 2);
        crate::mouse::apply_mouse(
            &mut state,
            EditorMouse {
                col: px,
                row: py,
                kind: EditorMouseKind::Press,
                button: Some(EditorMouseButton::Left),
            },
            fbw,
            fbh,
        );
        assert_eq!(
            state.cursor,
            (3, 3),
            "the click resolved to the ghosted cell"
        );
        assert_eq!(state.z, 1, "the click did not move the edit layer");
        assert_eq!(
            state.map.cell_at(3, 3, 1),
            CellKind::Floor,
            "the brush landed on the LIVE layer"
        );
        assert_eq!(
            state.map.cell_at(3, 3, 2),
            above_before,
            "the ghosted layer above was not touched"
        );
    }

    // ─── THE KEYS ────────────────────────────────────────────────────────────

    fn key(code: re_present::EditorKeyCode) -> re_present::EditorKey {
        re_present::EditorKey { code, shift: false }
    }

    #[test]
    fn g_cycles_onion() {
        use re_present::EditorKeyCode as E;
        let mut s = EditorState::new_empty(8, 8, 4);
        s.z = 2;
        assert_eq!(s.onion, OnionSkin::Off);
        crate::edit::apply_key(&mut s, key(E::Char('G')));
        assert_eq!(s.onion, OnionSkin::One);
        crate::edit::apply_key(&mut s, key(E::Char('G')));
        assert_eq!(s.onion, OnionSkin::Two);
        crate::edit::apply_key(&mut s, key(E::Char('G')));
        assert_eq!(s.onion, OnionSkin::Off, "G wraps back to off");
        assert!(!s.dirty, "a view aid never edits the document");
    }

    /// The aid is reachable through the ONE keymap table, so it exists in the
    /// hint bar, the `?` sheet and the `:` palette. (The two drift tests in
    /// `keymap` prove the table→surface direction; this proves the chord the
    /// table advertises is the chord `edit::apply_key` actually answers to.)
    #[test]
    fn every_view_aid_chord_in_the_keymap_is_a_chord_the_editor_answers() {
        use re_present::EditorKeyCode as E;
        let b = crate::keymap::KEYMAP
            .iter()
            .find(|b| b.chord == "G" && b.group == "view / files")
            .expect("chord G must be in the keymap table");
        assert!(b.palette, "G must be palette-reachable");
        assert_eq!(b.key, Some(E::Char('G')), "G must be executable");
        let mut s = EditorState::new_empty(8, 8, 4);
        // Executing the palette's own synthesized key must do the thing.
        crate::edit::apply_key(&mut s, key(E::Char('G')));
        assert!(s.onion.is_on());
    }
}
