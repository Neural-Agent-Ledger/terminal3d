//! `re_render::node_sprite` — the NODE SPRITE post-pass.
//!
//! Pipeline stage: runs right after `lines::apply` in every cast entry point
//! (`raycast_fov`, `composite_dump_once`, `composite_render_sys`) — the last
//! thing drawn before present.
//! Inputs: the [`Map`]'s puzzle-node [`Object`](re_world::CellKind::Object)
//! cells + the live [`CameraPose`]/[`Fov`] + the framebuffer already shaded by
//! the geometry pass.
//! Output: each node's screen cells overwritten with a small glyph cluster (a
//! bounded, tiered, fixed-template sprite), plus (for an embedded node) its
//! own wall footprint repainted as a socket plate.
//! Knobs: [`NODE_RADIUS`], [`DIAMETER_CAP_FRAC`], [`Tier`]/[`TIER_DIMS`],
//! [`crate::config::node_sprite_enabled`] (the `TS_NODE_SPRITE=0` kill switch).
//!
//! A puzzle node stops being wall-voxel geometry you look at and becomes an OBJECT
//! drawn at its projected centre, with a bounded, tiered, fixed-template glyph
//! cluster. The voxel stays solid for collision; the RENDER path treats a
//! FREE-STANDING node as air (see `crate::is_sprite_node` and its hooks in the DDA /
//! `offslice::role_of` / `shadow` / `holo`) and an EMBEDDED node (below) as a
//! recessed SOCKET plate.
//!
//! A screen-space size cap keeps a node from reading as a wall-sized block up
//! close, and the off-slice vertical pass treats a free-standing sprite node
//! as air on every plane rather than leaving a "dome" hole. The door sentinel
//! and inert set-dressing tags are excluded from node detection separately:
//! `is_sprite_node`/`is_puzzle_node_tag` exclude `side_data >=
//! RESERVED_TAG_BASE`, which both live at/above.
//!
//! Several look-and-feel choices worth calling out explicitly: rim glyphs are
//! picked by the rim's TANGENT direction, not its radial position — a radial
//! pick reads as a "plus sign", not a round orb, and the M tier's ellipse
//! formula additionally pinches to a single point top/bottom at that size (so
//! M is hand-authored instead, see below). Pipe stubs draw FIRST, so the
//! node's own core/body always overwrites a stub rather than the reverse. A
//! stub's colour comes from that pipe's own flow state
//! ([`re_core::decode_flow_state`]), never a generic default. A node whose
//! centre projects just behind the camera plane still draws its still-visible
//! sliver instead of vanishing outright. Two small legibility additions round
//! it out: a contact shadow under a free-standing node (the invisible voxel
//! would otherwise cast none), and a bolder rim variant + bg tint when the
//! cell behind the node is already a box-drawing pattern (dead-node
//! findability against a busy background).
//!
//! ## Mechanism
//!
//! 1. **Scan** the map for `Object` cells that are puzzle nodes
//!    ([`is_puzzle_node_tag`]), split into FREE-STANDING ([`scan_nodes`]) and
//!    EMBEDDED ([`scan_embedded_nodes`], [`is_embedded`]).
//! 2. **Socket pass** — for every embedded node, repaint its own already-drawn
//!    wall-voxel footprint (found by its `(surface, material)` stamp) from the old
//!    full-cell fill into a recessed plate ([`repaint_socket`]).
//! 3. **Contact shadow** — for every free-standing node with a floor beneath it, a
//!    small darkened disc on that floor.
//! 4. **Project** each node's world centre `(x+.5, y+.5, z+.5)` through the SAME
//!    basis the cast/lines use ([`crate::lines::LineViewBasis::project_sphere`]) —
//!    one projection, so an on-slice and an off-slice node share the exact same
//!    code path.
//! 5. **Size** — `diameter_rows = 2·R·proj_plane_dist/depth`, R = [`NODE_RADIUS`],
//!    capped at [`DIAMETER_CAP_FRAC`] of the framebuffer's rows.
//! 6. **Tier** — the capped diameter selects one of six FIXED tiers
//!    ([`Tier`]/[`TIER_DIMS`]), with hysteresis on the live path ([`select_tier`]).
//! 7. **Resolve** a small fixed grid of [`Role`]-tagged cells per node (rim, body,
//!    highlight, core, halo) INCLUDING an occlusion test against the TRUE
//!    multi-plane geometry ([`crate::lines::LineViewBasis::visible_except`], which
//!    excludes the node's own cell so an embedded node's own now-solid socket face
//!    cannot occlude the orb drawn on top of it).
//! 8. **Draw pipe stubs FIRST**, for every axis-adjacent PIPE cell, coloured by
//!    that pipe's own flow state — skipped entirely for a fully-occluded node (no
//!    floating stub) and never targeting a cell inside the node's own template
//!    footprint (no stub-over-core).
//! 9. **Draw the sprites** (far → near, so a near node overdraws a far one).
//!
//! ## Embedded nodes / sockets
//!
//! [`is_embedded`]: a node voxel counts as embedded when any of its 4 LATERAL (x/y)
//! neighbours is solid geometry (`WallSolid`) or a wall-trace surface (`WallTrace` —
//! painted OR a live pipe). Deliberately NOT a floor/ceiling neighbour: level geometry
//! auto-stamps the storey slab at `z = 0`/`z = sz - 1`, so a 6-axis rule would flag
//! almost every node one clear layer above the walking floor and defeat the
//! free-standing sprite path entirely — every real cavity a node sits in is a WALL
//! or PIPE neighbour, never a bare slab one. Making such a node see-through would
//! punch a hole into what should read as continuous solid — a node recessed into a
//! wall alcove, or spliced directly against a pipe run. `crate::is_sprite_node` excludes these
//! (the DDA/off-slice/shadow/holo passes keep them SOLID, exactly like an ordinary
//! `Object` cell, so they paint the OLD full-cell `PUZZLE_NODE` fill via the existing
//! geometry pipeline — unchanged, no new polygon math). [`repaint_socket`] then finds
//! that already-resolved footprint by its `(SurfaceKind::WallOther, tag)` stamp and
//! recolours it: blank interior, a one-cell box-drawing frame at the silhouette
//! edge, both in a fixed low-contrast tone independent of node state (the small orb
//! drawn on top carries the state colour instead — no fill on the socket face
//! itself). A free-standing node (no such contact) keeps the original
//! invisible-voxel + sprite path untouched.
//!
//! ## Rollback
//!
//! `TS_NODE_SPRITE=0` ([`crate::config::node_sprite_enabled`]) makes
//! `is_sprite_node` always `false` (today's wall-voxel node, embedded sockets
//! included) AND makes [`apply`] a hard no-op — byte-identical to the pre-sprite
//! renderer either way.
//!
//! ## Honest scope note
//!
//! The tier templates (S/L5/L7/L9) are PROCEDURALLY generated from a fixed
//! radius/aspect rule per tier (an ellipse classify: rim band / body / one
//! highlight / core); the M tier IS hand-authored (a literal small rounded box) —
//! the generic ellipse formula pinches to a single point at that size and reads as
//! a plus-sign / star, not an orb. Switch plates are NOT implemented here.

use std::cell::RefCell;

use re_core::{Cell, CellStyle, Framebuffer, SurfaceKind, Vec3};
use re_world::{CameraPose, CellKind, Map};

use crate::ascii_shader::{dim_rgb, luma_of, mix_rgb, pack_rgb, unpack_rgb};
use crate::glyph_family::STATE_NODE;
use crate::lines::{self, LineViewBasis, ScreenPoint};
use crate::{config, state_bus, Fov};

/// World radius of a puzzle-node sphere (world cells).
pub const NODE_RADIUS: f32 = 0.25;

/// A node's on-screen diameter never exceeds this fraction of the framebuffer's
/// row count, however close the camera stands — the "screen-space bound" that
/// keeps a close-up node from reading as a wall-sized block.
pub const DIAMETER_CAP_FRAC: f32 = 0.29;

/// Hysteresis half-band (rows of diameter) around a tier boundary: a node already
/// showing tier `T` only leaves `T` once the (capped) diameter clears the relevant
/// boundary by more than this — so walking near a boundary distance does not
/// flicker the template every frame.
pub const TIER_HYSTERESIS: f32 = 0.4;

/// Amendment: a DEAD node must stay at least [`Tier::S`] while within this many
/// world cells of perpendicular depth, so a dead node never shrinks to a
/// near-invisible dot exactly where the player most needs to read "this one is
/// dead".
pub const DEAD_MIN_TIER_DEPTH: f32 = 12.0;

/// A rim/tangent offset is classified "on this axis" once it dominates the other by
/// this ratio; the remaining wedge (neither axis dominant) is a diagonal corner.
const RIM_AXIS_RATIO: f32 = 1.4;

/// Six fixed template tiers, smallest to largest (matches [`TIER_ORDER`]/[`TIER_MIN_D`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Tier {
    Xs,
    S,
    M,
    L5,
    L7,
    L9,
}

/// [`Tier`] variants in ascending size order — the index into this array is the
/// tier's rank, used by the hysteresis boundary math.
const TIER_ORDER: [Tier; 6] = [Tier::Xs, Tier::S, Tier::M, Tier::L5, Tier::L7, Tier::L9];

/// The MINIMUM capped-diameter (rows) each tier in [`TIER_ORDER`] requires — the
/// measured thresholds: S ≥ 1.2, M ≥ 2.5, L5 ≥ 4.5, L7 ≥ 6.5, L9 ≥ 8.5.
/// `Xs`'s floor is 0 (anything smaller than S).
const TIER_MIN_D: [f32; 6] = [0.0, 1.2, 2.5, 4.5, 6.5, 8.5];

/// Fixed `(rows, cols)` for each [`TIER_ORDER`] entry. `cols ≈ rows × 2.1` (the
/// measured on-screen aspect at the shipping `cell_aspect`), rounded to an odd
/// width so the template has a true centre column.
const TIER_DIMS: [(usize, usize); 6] = [(1, 1), (1, 3), (3, 7), (5, 11), (7, 15), (9, 19)];

#[inline]
fn tier_index(t: Tier) -> usize {
    TIER_ORDER
        .iter()
        .position(|&x| x == t)
        .expect("Tier is always in TIER_ORDER")
}

#[inline]
fn tier_dims(t: Tier) -> (usize, usize) {
    TIER_DIMS[tier_index(t)]
}

/// The tier a raw (already diameter-capped) size strictly falls into, with no
/// hysteresis — the geometric answer.
fn strict_tier(d: f32) -> Tier {
    if d >= TIER_MIN_D[5] {
        Tier::L9
    } else if d >= TIER_MIN_D[4] {
        Tier::L7
    } else if d >= TIER_MIN_D[3] {
        Tier::L5
    } else if d >= TIER_MIN_D[2] {
        Tier::M
    } else if d >= TIER_MIN_D[1] {
        Tier::S
    } else {
        Tier::Xs
    }
}

/// Apply [`TIER_HYSTERESIS`] around the strict boundary nearest `prev`: `prev` is
/// kept unless `d` has crossed ITS OWN boundary (leaving toward a smaller tier) or
/// the NEXT tier's boundary (entering a larger one) by more than the hysteresis
/// band. A `None` previous tier (first frame / a non-persistent single-shot cast)
/// returns the strict answer with no banding. Pure, no I/O.
fn select_tier(d: f32, prev: Option<Tier>) -> Tier {
    let strict = strict_tier(d);
    let Some(p) = prev else {
        return strict;
    };
    if p == strict {
        return p;
    }
    let p_idx = tier_index(p);
    let s_idx = tier_index(strict);
    if s_idx < p_idx {
        // Shrinking: only actually leave `p` once `d` is more than the band below
        // `p`'s own floor.
        if d < TIER_MIN_D[p_idx] - TIER_HYSTERESIS {
            strict
        } else {
            p
        }
    } else {
        // Growing: only leave `p` once `d` clears the NEXT tier's floor by the band.
        let next_idx = (p_idx + 1).min(TIER_MIN_D.len() - 1);
        if d >= TIER_MIN_D[next_idx] + TIER_HYSTERESIS {
            strict
        } else {
            p
        }
    }
}

thread_local! {
    /// Per-node-tag tier memory for the CURRENT (persistent) cast stream — the live
    /// path only (`apply_persistent`). A single-shot cast (`apply`, used by
    /// `raycast_fov`/`composite_dump_once`) never reads or writes this, so a dump /
    /// golden frame is always the pure `strict_tier` answer: deterministic, no
    /// cross-test leakage between unrelated single-shot casts on the same thread.
    static TIER_MEMORY: RefCell<Vec<(u16, Tier)>> = const { RefCell::new(Vec::new()) };
}

/// Look up + update the persistent tier for `tag` at diameter `d` (the live path).
fn tier_with_memory(tag: u16, d: f32) -> Tier {
    TIER_MEMORY.with(|m| {
        let mut m = m.borrow_mut();
        let prev = m.iter().find(|(t, _)| *t == tag).map(|(_, tier)| *tier);
        let tier = select_tier(d, prev);
        match m.iter_mut().find(|(t, _)| *t == tag) {
            Some(entry) => entry.1 = tier,
            None => m.push((tag, tier)),
        }
        tier
    })
}

/// Test-only: forget all persistent tier memory (so a hysteresis test starts clean
/// regardless of what ran earlier on this thread).
#[cfg(test)]
pub(crate) fn clear_tier_memory_for_test() {
    TIER_MEMORY.with(|m| m.borrow_mut().clear());
}

// ─────────────────────────── node identity / embedding ───────────────────────

/// True for an `Object` cell whose `side_data` is a PUZZLE NODE index
/// (`0..RESERVED_TAG_BASE`) — the identity check, independent of whether it renders
/// as a free-standing sprite or an embedded socket. The exit door
/// ([`crate::DOOR_TAG_SENTINEL`]) and inert set-dressing
/// (`re_core::INERT_OBJECT_TAG`) both live at/above `RESERVED_TAG_BASE`, so they are
/// excluded and always render as ordinary solid wall-voxel geometry.
#[inline]
pub(crate) fn is_puzzle_node_tag(k: CellKind, side_data: u16) -> bool {
    k == CellKind::Object && side_data < re_core::RESERVED_TAG_BASE
}

/// Does the voxel at `(x, y, z)` touch solid wall mass or a pipe run on any of
/// its 4 LATERAL (x/y) faces, such that rendering it as air would expose a cavity —
/// a node recessed into a wall alcove (a node tucked against the perimeter wall,
/// or against an interior partition) or spliced directly
/// against a live pipe run (against a routed `/` trace)? `WallTrace`
/// counts whether or not it carries a live pipe tag (a painted trace is still solid
/// wall mass).
///
/// Deliberately LATERAL-ONLY: a floor/ceiling (`Floor`/`Ceiling`) neighbour does
/// NOT count. `stamp_geometry` auto-stamps `z = 0` as `Floor` and `z = sz - 1` as
/// `Ceiling` for every interior column, so almost every node in the shipped
/// campaign (one clear layer above the walking floor — a level's node plane
/// typically sits at `z = 2` in a 5-layer room, `z = 1`/`z = 3` are the open walking/pipe
/// layers either side) would count as "touching a slab" under a 6-neighbour rule,
/// converting the common free-standing case into a socket and defeating the sprite
/// feature entirely — checked against real node
/// placements across the campaign, the actual cavity-exposing contact in each is a WALL or PIPE
/// neighbour, never a bare floor/ceiling one. Pure grid read, no allocation.
pub(crate) fn is_embedded(map: &Map, x: i32, y: i32, z: i32) -> bool {
    const LATERAL: [(i32, i32, i32); 4] = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)];
    LATERAL.iter().any(|&(dx, dy, dz)| {
        let (nx, ny, nz) = (x + dx, y + dy, z + dz);
        map.in_bounds(nx, ny, nz)
            && matches!(
                map.cell_at(nx, ny, nz),
                CellKind::WallSolid | CellKind::WallTrace
            )
    })
}

/// Low-contrast, state-INDEPENDENT SOCKET tones: a recessed plate, never the
/// bright `◉` fill — the small orb drawn on top of it carries the state colour.
/// Chosen in the same dim blue-slate family the game's ambient/wall palette already
/// uses, so a socket reads as "part of the room's fabric", not a foreign UI chip.
const SOCKET_FILL: u32 = 0x0013_1A20;
const SOCKET_FRAME: u32 = 0x0038_4A58;

/// Recolour every already-drawn cell of an EMBEDDED node's own wall-voxel footprint
/// — found by its `(SurfaceKind::WallOther, tag)` stamp, which the ordinary
/// cast/contour/lines passes already resolved against the TRUE multi-plane geometry
/// (a doorway corner clipping the cube clips the socket exactly the same way, for
/// free) — from the old full-cell `PUZZLE_NODE` fill into a recessed socket: blank
/// interior, a one-cell box-drawing frame at the silhouette edge. `O(cols*rows)`,
/// once per embedded node (a handful); called BEFORE any sprite/stub drawing so the
/// orb layers cleanly on top of the plate.
fn repaint_socket(fb: &mut Framebuffer, cols: usize, rows: usize, tag: u16) {
    let n = cols * rows;
    let mut in_patch = vec![false; n];
    let mut any = false;
    for ri in 0..rows {
        for ci in 0..cols {
            if let Some(c) = fb.get(ci, ri) {
                if c.surface == SurfaceKind::WallOther && c.material == tag {
                    in_patch[ri * cols + ci] = true;
                    any = true;
                }
            }
        }
    }
    if !any {
        return; // occluded/off-screen this frame — nothing to recolour.
    }
    let at = |ci: usize, ri: usize| in_patch[ri * cols + ci];
    for ri in 0..rows {
        for ci in 0..cols {
            if !at(ci, ri) {
                continue;
            }
            let up = ri > 0 && at(ci, ri - 1);
            let down = ri + 1 < rows && at(ci, ri + 1);
            let left = ci > 0 && at(ci - 1, ri);
            let right = ci + 1 < cols && at(ci + 1, ri);
            let Some(existing) = fb.get(ci, ri).copied() else {
                continue;
            };
            let (glyph, fg) = if !(up && down && left && right) {
                // A silhouette edge: missing a vertical neighbour ⇒ the boundary
                // runs horizontally here (a top/bottom edge); missing a horizontal
                // neighbour ⇒ a side edge; missing both ⇒ a corner nub.
                let g = if !up || !down {
                    '─'
                } else if !left || !right {
                    '│'
                } else {
                    '·'
                };
                (u32::from(g), SOCKET_FRAME)
            } else {
                (u32::from(' '), 0)
            };
            let mut out = existing;
            out.glyph = glyph;
            out.fg = fg;
            out.bg = SOCKET_FILL;
            out.style = CellStyle::NONE;
            fb.set(ci, ri, out);
        }
    }
}

// ───────────────────────────── contact shadow (minor fix) ─────────────────────────

/// Horizontal-cell-unit radius of a free-standing node's floor contact shadow (the
/// invisible voxel stopped casting one — see `shadow.rs`'s `is_sprite_node` skip).
/// Aspect-corrected exactly like `posteffect::screen_radius`, so the disc reads
/// circular rather than a tall ellipse.
const CONTACT_SHADOW_RADIUS: f32 = 2.2;
/// Darken factor at the shadow's centre (`0` = no change, `1` = black), fading
/// linearly to `0` at [`CONTACT_SHADOW_RADIUS`].
const CONTACT_SHADOW_STRENGTH: f32 = 0.55;

/// Draw a small darkened disc on the floor beneath a FREE-STANDING node, if any
/// (a node hovering over a pit, or one with solid geometry between it and the
/// nearest lower floor, casts none). Only ever touches cells the cast already
/// resolved as `Floor` this frame — occlusion is therefore free: a floor cell
/// occluded by nearer geometry was never drawn as `Floor` in the first place, so it
/// is never a candidate here.
fn draw_contact_shadow(
    fb: &mut Framebuffer,
    map: &Map,
    basis: &LineViewBasis,
    cols: usize,
    rows: usize,
    node: &SpriteNode,
) {
    let mut floor_z = None;
    let mut z = node.z;
    while z >= 0 {
        let k = map.cell_at(node.x, node.y, z);
        if k == CellKind::Floor {
            floor_z = Some(z);
            break;
        }
        if z != node.z && k.is_solid() {
            break; // solid geometry blocks line of sight to any lower floor.
        }
        z -= 1;
    }
    let Some(fz) = floor_z else { return };
    let contact = Vec3::new(node.x as f32 + 0.5, node.y as f32 + 0.5, fz as f32 + 1.0);
    let Some(p) = basis.project(contact) else {
        return;
    };
    let r = CONTACT_SHADOW_RADIUS;
    let (col_lo, col_hi) = (
        (p.col - r).floor().max(0.0) as usize,
        (p.col + r).ceil().min(cols as f32 - 1.0) as usize,
    );
    let (row_lo, row_hi) = (
        (p.row - r).floor().max(0.0) as usize,
        (p.row + r).ceil().min(rows as f32 - 1.0) as usize,
    );
    if col_lo > col_hi || row_lo > row_hi {
        return;
    }
    for ri in row_lo..=row_hi {
        for ci in col_lo..=col_hi {
            let Some(existing) = fb.get(ci, ri).copied() else {
                continue;
            };
            if existing.surface != SurfaceKind::Floor {
                continue;
            }
            let dx = ci as f32 - p.col;
            let dy = (ri as f32 - p.row) * crate::posteffect::CELL_ASPECT;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist >= r {
                continue;
            }
            let k = 1.0 - (1.0 - dist / r) * CONTACT_SHADOW_STRENGTH;
            let mut out = existing;
            out.fg = dim_rgb(existing.fg, k);
            out.bg = dim_rgb(existing.bg, k);
            fb.set(ci, ri, out);
        }
    }
}

/// Puzzle-node states (mirrors `re_render::glyph_family::STATE_NODE` / the bridge's
/// encoding): `0` powered, `1` dead, `2` decoy, `3` won. Any other state-bus value
/// clamps to the LAST member (powered's ◉ is member 0, so an unwired/garbage state
/// intentionally does NOT default here — [`NodeState::from_bus`] documents the clamp).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NodeState {
    Powered,
    Dead,
    Decoy,
    Won,
}

impl NodeState {
    fn from_bus(word: u32) -> Self {
        match word {
            0 => NodeState::Powered,
            1 => NodeState::Dead,
            2 => NodeState::Decoy,
            _ => NodeState::Won, // 3, or any out-of-range word, clamps to the last state.
        }
    }
}

/// One state's full palette. Packed `0x00RRGGBB`.
struct StatePalette {
    rim: u32,
    core: u32,
    body_lit: u32,
    body_dark: u32,
    /// `None` = no glow halo for this state (DEAD/DECOY stay quiet).
    halo: Option<u32>,
}

fn palette_for(state: NodeState) -> StatePalette {
    match state {
        NodeState::Powered => StatePalette {
            rim: 0x00FF_C738,
            core: 0x00FF_F4D0,
            body_lit: 0x004A_3510,
            body_dark: 0x0016_0F04,
            halo: Some(0x003A_2808),
        },
        NodeState::Dead => StatePalette {
            rim: 0x005E_6E80,
            core: 0x009F_B0C4,
            body_lit: 0x001E_2A36,
            body_dark: 0x000C_1116,
            halo: None,
        },
        NodeState::Decoy => StatePalette {
            rim: 0x0055_5555,
            core: 0x0080_8080,
            body_lit: 0x001C_1C1C,
            body_dark: 0x000E_0E0E,
            halo: None,
        },
        NodeState::Won => StatePalette {
            rim: 0x00FF_F8E0,
            core: 0x00FF_FFFF,
            body_lit: 0x006A_5020,
            body_dark: 0x0024_1A08,
            halo: Some(0x005A_4410),
        },
    }
}

/// Findability: DEAD's slate (rim `#5E6E80` / core `#9FB0C4`) is low-contrast
/// against a dark blue/teal floor (the common case at 8-15 cells) — the tone
/// and the floor sit at similar dim luminance and similar cool hue, so the
/// node all but disappears. Below this background luminance
/// threshold, [`dead_palette_for_bg`] swaps in a brighter (but still unambiguously
/// COOL, never warm/bright like POWERED) pop tone instead; at/above it, the
/// original dim slate already reads fine (a bright/light bg makes even a mid-tone
/// read as a dark, legible silhouette), so it is kept unchanged (back-compat: a
/// scene whose background always exceeds the threshold renders byte-identical to
/// before this fix).
const DEAD_DARK_BG_LUMA: f32 = 0.16;
const DEAD_RIM_POP: u32 = 0x00BF_E3FF;
const DEAD_CORE_POP: u32 = 0x00E8_F6FF;

fn dead_palette_for_bg(bg_luma: f32) -> StatePalette {
    if bg_luma < DEAD_DARK_BG_LUMA {
        StatePalette {
            rim: DEAD_RIM_POP,
            core: DEAD_CORE_POP,
            body_lit: 0x0022_2E38,
            body_dark: 0x000C_1116,
            halo: None,
        }
    } else {
        palette_for(NodeState::Dead)
    }
}

/// What one template cell draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    /// Not part of the sprite's footprint — leave the cell untouched.
    Skip,
    /// bg-only glow, OUTSIDE the rim, keeping whatever glyph/fg is already there.
    /// Powered/won only.
    Halo,
    /// Rounded box-drawing rim.
    Rim,
    /// bg-only lit-sphere body fill (blank glyph).
    Body,
    /// A fixed specular fleck over the body.
    Highlight,
    /// The one state-core glyph (◉ / ◌ / ○ / ◍), plus the literal parens on the `S`
    /// tier.
    Core,
    /// A literal `(`/`)` bracket around `Core` on the `S` tier (rim-coloured).
    CoreBracket(char),
}

/// Classify every `(row, col)` of a `(h, w)` template into a [`Role`]. Deterministic
/// pure geometry — the whole point of a FIXED template (`select_tier` already
/// picked which shape; this never varies within a tier).
fn classify(h: usize, w: usize) -> Vec<Role> {
    if h == 1 && w == 1 {
        return vec![Role::Core]; // Xs: the bare state glyph.
    }
    if h == 1 {
        // S: literal "(core)" bracket, `w` is always odd and ≥ 3.
        let mut row = vec![Role::Body; w];
        row[0] = Role::CoreBracket('(');
        row[w - 1] = Role::CoreBracket(')');
        row[w / 2] = Role::Core;
        return row;
    }
    if h == 3 && w == 7 {
        // M tier: a HAND-AUTHORED small rounded box, not the generic ellipse
        // formula below — at 3 rows the ellipse's rim band pinches to a single
        // point top/bottom, which combined with a radial rim_glyph
        // reads as a plus-sign/star rather than a rounded orb. A literal frame always
        // reads as a small orb.
        // No highlight: at this size the generic highlight offset lands adjacent
        // to the core, which reads as touching it rather than a separate sheen.
        let mut out = vec![Role::Skip; h * w];
        for r in [0usize, 2] {
            out[r * w] = Role::Halo;
            for c in 1..=5 {
                out[r * w + c] = Role::Rim;
            }
            out[r * w + 6] = Role::Halo;
        }
        out[w] = Role::Halo; // row 1, col 0
        out[w + 1] = Role::Rim;
        out[w + 2] = Role::Body;
        out[w + 3] = Role::Core;
        out[w + 4] = Role::Body;
        out[w + 5] = Role::Rim;
        out[w + 6] = Role::Halo;
        return out;
    }
    let cy = (h as f32 - 1.0) / 2.0;
    let cx = (w as f32 - 1.0) / 2.0;
    let half_h = cy.max(0.5);
    let half_w = cx.max(0.5);
    let mut out = vec![Role::Skip; h * w];
    for row in 0..h {
        for col in 0..w {
            let ny = (row as f32 - cy) / half_h;
            let nx = (col as f32 - cx) / half_w;
            let r = (nx * nx + ny * ny).sqrt();
            out[row * w + col] = if r <= 0.72 {
                Role::Body
            } else if r <= 1.02 {
                Role::Rim
            } else if r <= 1.32 {
                Role::Halo
            } else {
                Role::Skip
            };
        }
    }
    // CORE: the exact centre cell.
    let (cry, crx) = (cy.round() as usize, cx.round() as usize);
    out[cry * w + crx] = Role::Core;
    // ONE fixed specular highlight, upper-left of centre, only where it lands on
    // BODY (small tiers may not have room).
    let hy = (cy - half_h * 0.42).round().max(0.0) as usize;
    let hx = (cx - half_w * 0.42).round().max(0.0) as usize;
    if (hy, hx) != (cry, crx) && hy < h && hx < w && out[hy * w + hx] == Role::Body {
        out[hy * w + hx] = Role::Highlight;
    }
    out
}

/// Box-drawing glyphs a Rim cell might land on TOP of (a circuit-trace wall, an
/// auto-contour silhouette, an authored `[line]`) — the "rim camouflage" minor fix:
/// a thin light rule blends into an existing box-drawing pattern behind it.
const BOX_DRAWING_BG: &[char] = &[
    '─', '│', '┌', '┐', '└', '┘', '┏', '┓', '┗', '┛', '━', '┃', '┼', '├', '┤', '┬', '┴', '═', '║',
    '╱', '╲', '╭', '╮', '╰', '╯',
];

#[inline]
fn is_box_drawing_bg(glyph: u32) -> bool {
    char::from_u32(glyph).is_some_and(|c| BOX_DRAWING_BG.contains(&c))
}

/// Choose a rounded box-drawing rim glyph for offset `(dy, dx)` from the template
/// centre (screen rows/cols; `dy` down, `dx` right) — classified by the rim's
/// TANGENT direction, not by the radial `(dy, dx)` position itself (a purely radial
/// classification mis-picks the glyph at the rim's four cardinal points): `dy` is
/// first converted to horizontal-cell-equivalent units via `posteffect::CELL_ASPECT`
/// (a terminal cell is ~2:1 tall, exactly the correction `posteffect::screen_radius`
/// already applies), THEN compared against `dx`. A position near the TOP/BOTTOM of
/// the orb (the aspect-corrected vertical offset dominates) has a HORIZONTAL
/// tangent → `─`; a position near a SIDE (the horizontal offset dominates) has a
/// VERTICAL tangent → `│`; the remaining wedge is a rounded corner. `heavy` selects
/// the bold box-drawing variant (`━┃┏┓┗┛`) instead of the light one, for the rim
/// camouflage fix (a thin light rule disappears against a box-drawing background).
fn rim_glyph(dy: f32, dx: f32, heavy: bool) -> char {
    if dy.abs() < 1e-3 && dx.abs() < 1e-3 {
        return '·'; // degenerate (Xs/S never reach here); never actually indexed.
    }
    let dy_a = dy * crate::posteffect::CELL_ASPECT;
    let horiz = dy_a.abs() >= dx.abs() * RIM_AXIS_RATIO;
    let vert = dx.abs() >= dy_a.abs() * RIM_AXIS_RATIO;
    if horiz {
        if heavy {
            '━'
        } else {
            '─'
        }
    } else if vert {
        if heavy {
            '┃'
        } else {
            '│'
        }
    } else if dx < 0.0 && dy < 0.0 {
        if heavy {
            '┏'
        } else {
            '╭'
        }
    } else if dx >= 0.0 && dy < 0.0 {
        if heavy {
            '┓'
        } else {
            '╮'
        }
    } else if dx < 0.0 && dy >= 0.0 {
        if heavy {
            '┗'
        } else {
            '╰'
        }
    } else if heavy {
        '┛'
    } else {
        '╯'
    }
}

/// A breathing brightness multiplier for rim/halo/core — mirrors `puzzle_node()`'s
/// `Animation::Pulse` (speed 0.6 Hz, amplitude 0.12): a calm "it lives" pulse, not
/// a strobe. `time = 0` (the deterministic single-shot casts) gives multiplier 1.0.
fn breathe(time: f32) -> f32 {
    1.0 + 0.12 * (time * 0.6 * std::f32::consts::TAU).sin()
}

#[inline]
fn scale_rgb(c: u32, k: f32) -> u32 {
    pack_rgb(unpack_rgb(c) * k.clamp(0.0, 2.0))
}

/// One puzzle node found on the map: its state-bus tag and world cell.
struct SpriteNode {
    tag: u16,
    x: i32,
    y: i32,
    z: i32,
}

/// Scan for FREE-STANDING sprite nodes (`crate::is_sprite_node`, i.e. NOT embedded —
/// these render as air, drawn as a floating orb).
fn scan_nodes(map: &Map) -> Vec<SpriteNode> {
    let (sx, sy, sz) = map.size();
    let mut out = Vec::new();
    for z in 0..sz as i32 {
        for y in 0..sy as i32 {
            for x in 0..sx as i32 {
                if map.cell_at(x, y, z) == CellKind::Object {
                    let sd = map.side_data_at(x, y, z);
                    if crate::is_sprite_node(map, x, y, z, CellKind::Object, sd) {
                        out.push(SpriteNode { tag: sd, x, y, z });
                    }
                }
            }
        }
    }
    out
}

/// Scan for EMBEDDED puzzle nodes (a puzzle-node identity that `is_sprite_node`
/// excludes because [`is_embedded`] is true) — these stay solid/opaque geometry
/// (painted by the ordinary caster) and get a socket face + an orb on top, rather
/// than being air.
fn scan_embedded_nodes(map: &Map) -> Vec<SpriteNode> {
    if !config::node_sprite_enabled() {
        return Vec::new();
    }
    let (sx, sy, sz) = map.size();
    let mut out = Vec::new();
    for z in 0..sz as i32 {
        for y in 0..sy as i32 {
            for x in 0..sx as i32 {
                if map.cell_at(x, y, z) == CellKind::Object {
                    let sd = map.side_data_at(x, y, z);
                    if is_puzzle_node_tag(CellKind::Object, sd) && is_embedded(map, x, y, z) {
                        out.push(SpriteNode { tag: sd, x, y, z });
                    }
                }
            }
        }
    }
    out
}

/// The single-shot, DETERMINISTIC entry point: `raycast_fov` and
/// `composite_dump_once` call this. No cross-call tier memory — every node's tier
/// is the pure [`strict_tier`] answer, so a `--dump`/golden frame is reproducible
/// with no dependence on what rendered before it. Hard no-op when the map has no
/// sprite nodes or [`config::node_sprite_enabled`] is off.
pub fn apply(fb: &mut Framebuffer, map: &Map, cam: &CameraPose, fov: &Fov, time: f32) {
    apply_inner(fb, map, cam, fov, time, false);
}

/// The LIVE entry point: `composite_render_sys` calls this every frame. Tier
/// selection is stateful (persistent per-node-tag hysteresis via [`TIER_MEMORY`])
/// so a player walking near a tier boundary does not see the template flicker.
pub fn apply_persistent(fb: &mut Framebuffer, map: &Map, cam: &CameraPose, fov: &Fov, time: f32) {
    apply_inner(fb, map, cam, fov, time, true);
}

/// One node's fully resolved draw plan, computed BEFORE any pixel is written (so
/// dead-contrast sampling reads the true pre-paint background, and stubs can be
/// drawn strictly before ANY node's orb, using a footprint known up front).
struct Prepared {
    node: SpriteNode,
    /// `(ci, ri, role, dy, dx)` for every non-`Skip` template cell, REGARDLESS of
    /// occlusion — the geometric footprint pipe stubs must never draw into.
    footprint: Vec<(usize, usize, Role, f32, f32)>,
    /// Indices into `footprint` that passed the occlusion test this frame.
    visible: Vec<usize>,
    rim_color: u32,
    core_color: u32,
    halo_color: Option<u32>,
    core_glyph: u32,
    pal_lit: u32,
    pal_dark: u32,
    cy: f32,
    cx: f32,
}

fn apply_inner(
    fb: &mut Framebuffer,
    map: &Map,
    cam: &CameraPose,
    fov: &Fov,
    time: f32,
    persist: bool,
) {
    if !config::node_sprite_enabled() {
        return;
    }
    let (cols, rows) = (fb.cols(), fb.rows());
    if cols == 0 || rows == 0 {
        return;
    }
    let free = scan_nodes(map);
    let embedded = scan_embedded_nodes(map);
    if free.is_empty() && embedded.is_empty() {
        return; // hard no-op, byte-identical fb.
    }

    let basis = LineViewBasis::new(cam, fov, cols, rows);

    // Recolour every embedded node's already-painted wall-voxel footprint into
    // a socket BEFORE any sprite/stub drawing, so the orb layers on top of it.
    for n in &embedded {
        repaint_socket(fb, cols, rows, n.tag);
    }
    // Minor fix: restore a small contact shadow under every free-standing node.
    for n in &free {
        draw_contact_shadow(fb, map, &basis, cols, rows, n);
    }

    let mut all: Vec<SpriteNode> = Vec::with_capacity(free.len() + embedded.len());
    all.extend(free);
    all.extend(embedded);

    // Resolve projection + depth once per node (`project_sphere` still draws a
    // node whose exact centre sits just behind the camera plane), then order
    // FAR → NEAR so a near node overdraws a far one.
    struct Resolved {
        node: SpriteNode,
        centre: ScreenPoint,
    }
    let mut resolved: Vec<Resolved> = all
        .drain(..)
        .filter_map(|node| {
            let world = Vec3::new(
                node.x as f32 + 0.5,
                node.y as f32 + 0.5,
                node.z as f32 + 0.5,
            );
            basis
                .project_sphere(world, NODE_RADIUS)
                .map(|centre| Resolved { node, centre })
        })
        .collect();
    resolved.sort_by(|a, b| b.centre.depth.total_cmp(&a.centre.depth));

    let prepared: Vec<Prepared> = resolved
        .into_iter()
        .map(|Resolved { node, centre }| {
            resolve_node(fb, map, &basis, cols, rows, node, centre, time, persist)
        })
        .collect();

    // Pass 1: pipe stubs, for every node that has at least one visible cell,
    // never targeting a cell in ANY node's own footprint.
    for p in &prepared {
        if p.visible.is_empty() {
            continue; // fully occluded/off-screen this frame — no floating stub.
        }
        draw_pipe_stubs(fb, map, &basis, cols, rows, &p.node, &p.footprint);
    }
    // Pass 2: the sprites themselves, far → near (the order `prepared` is already in).
    for p in &prepared {
        paint_node(fb, p);
    }
}

#[allow(clippy::too_many_arguments)]
fn resolve_node(
    fb: &Framebuffer,
    map: &Map,
    basis: &LineViewBasis,
    cols: usize,
    rows: usize,
    node: SpriteNode,
    centre: ScreenPoint,
    time: f32,
    persist: bool,
) -> Prepared {
    let depth = centre.depth.max(0.05);
    let raw_d = 2.0 * NODE_RADIUS * basis.proj_plane_dist() / depth;
    let d = raw_d.min(DIAMETER_CAP_FRAC * rows as f32);

    let state = NodeState::from_bus(state_bus::state_of(node.tag));
    let mut tier = if persist {
        tier_with_memory(node.tag, d)
    } else {
        select_tier(d, None)
    };
    // Findability amendment: a DEAD node stays legible up close — never below `S`
    // while within `DEAD_MIN_TIER_DEPTH` world cells.
    if state == NodeState::Dead && tier == Tier::Xs && depth <= DEAD_MIN_TIER_DEPTH {
        tier = Tier::S;
    }

    // Dead-node findability contrast: sample the TRUE pre-paint background at the
    // node's own projected centre cell (post-socket-repaint for an embedded node,
    // so the pick is made against what the orb will actually sit on) and pick the
    // pop tone only when that background is dark.
    let pal = if state == NodeState::Dead {
        let ci = centre.col.round().clamp(0.0, cols as f32 - 1.0) as usize;
        let ri = centre.row.round().clamp(0.0, rows as f32 - 1.0) as usize;
        let bg_luma = fb.get(ci, ri).map(|c| luma_of(c.bg)).unwrap_or(0.0);
        dead_palette_for_bg(bg_luma)
    } else {
        palette_for(state)
    };
    let pulse = state_bus::pulse_of(node.tag).clamp(0.0, 1.0);
    let bright = breathe(time);
    let rim_color = mix_rgb(scale_rgb(pal.rim, bright), 0x00FF_FFFF, pulse * 0.6);
    let core_color = mix_rgb(scale_rgb(pal.core, bright), 0x00FF_FFFF, pulse * 0.5);
    let halo_color = pal
        .halo
        .map(|h| mix_rgb(scale_rgb(h, bright), 0x00FF_FFFF, pulse * 0.4));
    let core_glyph = STATE_NODE.at(match state {
        NodeState::Powered => 0,
        NodeState::Dead => 1,
        NodeState::Decoy => 2,
        NodeState::Won => 3,
    });

    let (h, w) = tier_dims(tier);
    let roles = classify(h, w);
    let cy = (h as f32 - 1.0) / 2.0;
    let cx = (w as f32 - 1.0) / 2.0;
    let occlusion_depth = (depth - NODE_RADIUS).max(0.02);
    let except = Some((node.x, node.y, node.z));

    // Gather the FULL geometric footprint first (every non-`Skip`,
    // on-screen template cell — this is what a pipe stub must never draw into),
    // then separately test occlusion for each — `visible` records which indices
    // into `footprint` actually pass, for the paint pass.
    let mut footprint = Vec::new();
    let mut visible = Vec::new();
    for row in 0..h {
        for col in 0..w {
            let role = roles[row * w + col];
            if role == Role::Skip {
                continue;
            }
            if role == Role::Halo && halo_color.is_none() {
                continue; // no glow for this state — a quiet dead/decoy node.
            }
            let dy = row as f32 - cy;
            let dx = col as f32 - cx;
            let screen_col = centre.col + dx;
            let screen_row = centre.row + dy;
            let ci = screen_col.round();
            let ri = screen_row.round();
            if ci < -0.5
                || ri < -0.5
                || ci as isize >= cols as isize
                || ri as isize >= rows as isize
            {
                continue; // per-cell screen clip: only cells IN the frame count.
            }
            let (ci, ri) = (ci as usize, ri as usize);
            let idx = footprint.len();
            footprint.push((ci, ri, role, dy, dx));
            // `except` excludes the node's own (possibly now-solid, embedded)
            // cell from the occlusion test, so a socket's own face never hides the
            // orb drawn on top of it.
            if basis.visible_except(map, screen_col, screen_row, occlusion_depth, except) {
                visible.push(idx);
            }
        }
    }
    Prepared {
        node,
        footprint,
        visible,
        rim_color,
        core_color,
        halo_color,
        core_glyph,
        pal_lit: pal.body_lit,
        pal_dark: pal.body_dark,
        cy,
        cx,
    }
}

/// Lit-sphere bg colour at offset `(dy, dx)` from a tier's centre `(cy, cx)`: a
/// Lambert term against a fixed key light from the upper-left, `mix(dark, lit, N·L)`.
/// Deterministic pure geometry — no time term, so body shading never flickers.
fn body_bg(lit: u32, dark: u32, dy: f32, dx: f32, cy: f32, cx: f32) -> u32 {
    let half_h = cy.max(0.5);
    let half_w = cx.max(0.5);
    let nx = (dx / half_w / 0.72).clamp(-1.0, 1.0);
    let ny = (dy / half_h / 0.72).clamp(-1.0, 1.0);
    let nz = (1.0 - (nx * nx + ny * ny)).max(0.0).sqrt();
    // Key light from the upper-left: negative dx (left), negative dy (up), toward
    // the viewer.
    let (lx, ly, lz) = (-0.6f32, -0.55f32, 0.65f32);
    let inv_len = 1.0 / (lx * lx + ly * ly + lz * lz).sqrt();
    let lambert = (nx * lx * inv_len + ny * ly * inv_len + nz * lz * inv_len).clamp(0.0, 1.0);
    mix_rgb(dark, lit, lambert)
}

#[inline]
fn write_cell(fb: &mut Framebuffer, ci: usize, ri: usize, glyph: u32, fg: u32, bg: u32) {
    let mut cell = Cell {
        glyph,
        fg,
        bg,
        style: CellStyle::NONE,
        ..fb.get(ci, ri).copied().unwrap_or_default()
    };
    cell.surface = SurfaceKind::WallOther; // keeps a node reading as "object" downstream.
    fb.set(ci, ri, cell);
}

/// Draw every VISIBLE cell of one resolved node: rim (with the rim-camouflage
/// heavy-variant + bg-tint fix), lit-sphere body, one specular highlight, the state
/// core glyph, and (powered/won) the bg-only glow halo just outside the rim.
fn paint_node(fb: &mut Framebuffer, p: &Prepared) {
    for &idx in &p.visible {
        let (ci, ri, role, dy, dx) = p.footprint[idx];
        let Some(existing) = fb.get(ci, ri).copied() else {
            continue;
        };
        match role {
            Role::Halo => {
                // bg-only glow: keep the existing glyph/fg, just lift the bg
                // toward the glow colour.
                let glow = p.halo_color.expect("checked in resolve_node");
                let mut out = existing;
                out.bg = mix_rgb(existing.bg, glow, 0.45);
                fb.set(ci, ri, out);
            }
            Role::Rim => {
                // Rim camouflage fix: a thin light rule disappears against an
                // already box-drawing background (a circuit-trace wall, another
                // silhouette/line) — go bold + tint the bg toward the rim colour.
                let heavy = is_box_drawing_bg(existing.glyph);
                let glyph = rim_glyph(dy, dx, heavy);
                let bg = body_bg(p.pal_lit, p.pal_dark, dy, dx, p.cy, p.cx);
                let bg = if heavy {
                    mix_rgb(bg, p.rim_color, 0.35)
                } else {
                    bg
                };
                write_cell(fb, ci, ri, glyph as u32, p.rim_color, bg);
            }
            Role::Body => {
                write_cell(
                    fb,
                    ci,
                    ri,
                    u32::from(' '),
                    0,
                    body_bg(p.pal_lit, p.pal_dark, dy, dx, p.cy, p.cx),
                );
            }
            Role::Highlight => {
                write_cell(
                    fb,
                    ci,
                    ri,
                    u32::from('∙'),
                    p.rim_color,
                    body_bg(p.pal_lit, p.pal_dark, dy, dx, p.cy, p.cx),
                );
            }
            Role::Core => {
                write_cell(
                    fb,
                    ci,
                    ri,
                    p.core_glyph,
                    p.core_color,
                    body_bg(p.pal_lit, p.pal_dark, dy, dx, p.cy, p.cx),
                );
            }
            Role::CoreBracket(ch) => {
                write_cell(
                    fb,
                    ci,
                    ri,
                    ch as u32,
                    p.rim_color,
                    body_bg(p.pal_lit, p.pal_dark, dy, dx, p.cy, p.cx),
                );
            }
            Role::Skip => unreachable!("filtered above"),
        }
    }
}

/// Draw a short occlusion-aware stroke from the sprite's rim toward each
/// axis-adjacent PIPE cell (`WallTrace` carrying a flagged pipe tag), so a node
/// does not float disconnected from the network it belongs to.
///
/// `footprint` is the node's OWN full geometric template (every non-`Skip`
/// cell, regardless of occlusion) — a stub target landing on one of those
/// cells is skipped outright, so a stub can never paint over the node's own
/// core/body. The stub's on/off colour reads the pipe's OWN flow state via
/// `re_core::decode_flow_state` (bit-packed powered + direction), not a bare
/// `state != 1` comparison, which would misread the empty-bus default (`0`,
/// meaning cold/unpowered) as "on".
fn draw_pipe_stubs(
    fb: &mut Framebuffer,
    map: &Map,
    basis: &LineViewBasis,
    cols: usize,
    rows: usize,
    node: &SpriteNode,
    footprint: &[(usize, usize, Role, f32, f32)],
) {
    const DIRS: [(i32, i32, i32); 4] = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)];
    let world_centre = Vec3::new(
        node.x as f32 + 0.5,
        node.y as f32 + 0.5,
        node.z as f32 + 0.5,
    );
    let except = Some((node.x, node.y, node.z));
    for (dx, dy, dz) in DIRS {
        let (nx, ny, nz) = (node.x + dx, node.y + dy, node.z + dz);
        if !map.in_bounds(nx, ny, nz) {
            continue;
        }
        if map.cell_at(nx, ny, nz) != CellKind::WallTrace {
            continue;
        }
        let sd = map.side_data_at(nx, ny, nz);
        let Some(pipe_tag) = re_core::decode_pipe_tag(sd) else {
            continue; // an ordinary painted trace, not a live pipe.
        };
        // Rim point: from the node centre toward the shared face (world radius
        // NODE_RADIUS), to the projected centre of that face.
        let dir = Vec3::new(dx as f32, dy as f32, dz as f32);
        let rim_point = world_centre + dir * NODE_RADIUS;
        let face_point = world_centre + dir * 0.5;
        let (Some(a), Some(b)) = (basis.project(rim_point), basis.project(face_point)) else {
            continue;
        };
        let flow = re_core::decode_flow_state(state_bus::state_of(pipe_tag));
        let on = !matches!(flow, re_core::FlowDir::None);
        let color = if on { 0x0060_C0FF } else { 0x0040_5060 };
        lines::draw_segment(fb, cols, rows, a, b, color, |c, r, d| {
            let (ci, ri) = (c.round(), r.round());
            if ci >= 0.0 && ri >= 0.0 {
                let (ci, ri) = (ci as usize, ri as usize);
                if footprint
                    .iter()
                    .any(|&(fci, fri, _, _, _)| fci == ci && fri == ri)
                {
                    return false; // never draw a stub over the node's own footprint.
                }
            }
            basis.visible_except(map, c, r, d, except)
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use re_core::{encode_flow_state, encode_pipe_side_data, CellStateMap, FlowDir};
    use re_world::{stamp_geometry, GeometrySpec, ObjectCell};

    /// A room tall enough that a node one plane above the eye (`z = 2`, the shipped
    /// campaign convention) has a clear air gap from the auto-stamped floor (`z = 0`)
    /// and ceiling (`z = sz - 1`) on every side, so it is FREE-STANDING (not
    /// [`is_embedded`]) unless a test deliberately adds touching geometry.
    /// `stamp_geometry` auto-fills `z == 0` as `Floor` and `z == sz - 1` as `Ceiling`
    /// for every non-perimeter `(x, y)` — a shorter room (node flush on the floor)
    /// correctly counts as embedded (see [`is_embedded`]), so these fixtures widen the room
    /// instead of changing the node's role.
    fn room_with_node(sx: usize, sy: usize, sz: usize, node: (i32, i32, i32)) -> Map {
        let objects = [ObjectCell {
            x: node.0,
            y: node.1,
            z: node.2,
            tag: 0,
        }];
        stamp_geometry(&GeometrySpec {
            size: (sx, sy, sz),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &objects,
        })
    }

    fn room_with_node_and_pipe(
        node: (i32, i32, i32),
        pipe_dir: (i32, i32, i32),
        pipe_tag: u16,
    ) -> Map {
        let mut map = room_with_node(10, 10, 5, node);
        let p = (
            node.0 + pipe_dir.0,
            node.1 + pipe_dir.1,
            node.2 + pipe_dir.2,
        );
        map.set_cell(
            p.0 as usize,
            p.1 as usize,
            p.2 as usize,
            CellKind::WallTrace,
            encode_pipe_side_data(pipe_tag),
        );
        map
    }

    fn cam_at(x: f32, y: f32, z: f32, yaw: f32) -> CameraPose {
        CameraPose {
            pos: Vec3::new(x, y, z),
            yaw,
            pitch: 0.0,
        }
    }

    /// The RAW geometry+shade pass only (no contour/lines/node_sprite overlays) —
    /// exactly what `node_sprite::apply` runs against in every real cast entry
    /// point, and the one framebuffer state `repaint_socket` can actually find a
    /// `(SurfaceKind::WallOther, tag)` footprint to recolour in.
    fn cast_raw(map: &Map, cam: &CameraPose, fov: &Fov, cols: usize, rows: usize) -> Framebuffer {
        let registry = crate::registry_for_map(map);
        let mut fb = Framebuffer::new(cols, rows);
        crate::raycast_with_time_fov(map, cam, &registry, fov, 0.0, &mut fb);
        fb
    }

    // ── pure tier / hysteresis math (unchanged behaviour, still pinned) ────────

    #[test]
    fn strict_tier_thresholds_match_the_design_table() {
        assert_eq!(strict_tier(0.0), Tier::Xs);
        assert_eq!(strict_tier(1.19), Tier::Xs);
        assert_eq!(strict_tier(1.2), Tier::S);
        assert_eq!(strict_tier(2.5), Tier::M);
        assert_eq!(strict_tier(4.5), Tier::L5);
        assert_eq!(strict_tier(6.5), Tier::L7);
        assert_eq!(strict_tier(8.5), Tier::L9);
        assert_eq!(strict_tier(99.0), Tier::L9);
    }

    #[test]
    fn hysteresis_holds_the_previous_tier_inside_the_band() {
        assert_eq!(select_tier(2.3, Some(Tier::M)), Tier::M);
        assert_eq!(select_tier(2.0, Some(Tier::M)), Tier::S);
        assert_eq!(select_tier(4.7, Some(Tier::M)), Tier::M);
        assert_eq!(select_tier(5.0, Some(Tier::M)), Tier::L5);
        assert_eq!(select_tier(2.3, None), Tier::S);
    }

    #[test]
    fn tier_with_memory_is_stateful_and_test_isolated() {
        clear_tier_memory_for_test();
        assert_eq!(tier_with_memory(7, 2.6), Tier::M);
        assert_eq!(tier_with_memory(7, 2.3), Tier::M);
        clear_tier_memory_for_test();
        assert_eq!(tier_with_memory(7, 2.3), Tier::S);
        clear_tier_memory_for_test();
    }

    #[test]
    fn bounded_footprint_is_non_increasing_and_capped_as_distance_grows() {
        let fov = Fov::default();
        let dists = [0.75f32, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0];
        let mut last_d = f32::INFINITY;
        let mut last_rank = usize::MAX;
        for dist in dists {
            let cam = cam_at(7.5 - dist, 7.5, 2.5, std::f32::consts::FRAC_PI_2);
            let basis = LineViewBasis::new(&cam, &fov, 100, 32);
            let centre = basis
                .project(Vec3::new(7.5, 7.5, 2.5))
                .expect("in front of the camera");
            let raw_d = 2.0 * NODE_RADIUS * basis.proj_plane_dist() / centre.depth;
            let d = raw_d.min(DIAMETER_CAP_FRAC * 32.0);
            assert!(
                d <= DIAMETER_CAP_FRAC * 32.0 + 1e-4,
                "never exceeds the screen-space cap"
            );
            assert!(
                d <= last_d + 1e-3,
                "footprint must be non-increasing as distance grows"
            );
            let rank = tier_index(strict_tier(d));
            assert!(
                rank <= last_rank,
                "tier rank must be monotone non-increasing as distance grows"
            );
            last_d = d;
            last_rank = rank;
        }
    }

    #[test]
    fn on_slice_and_off_slice_share_the_same_projection_formula_at_equal_horizontal_distance() {
        let fov = Fov::default();
        let cam = cam_at(3.5, 7.5, 2.5, std::f32::consts::FRAC_PI_2);
        let basis = LineViewBasis::new(&cam, &fov, 100, 32);
        let c_on = basis
            .project(Vec3::new(7.5, 7.5, 2.5))
            .expect("on-slice, same z as the eye");
        let c_off = basis
            .project(Vec3::new(7.5, 7.5, 3.5))
            .expect("off-slice, one plane above");
        assert!(
            (c_on.depth - c_off.depth).abs() < 1e-4,
            "same horizontal distance ⇒ same perpendicular depth"
        );
        let d_on = (2.0 * NODE_RADIUS * basis.proj_plane_dist() / c_on.depth)
            .min(DIAMETER_CAP_FRAC * 32.0);
        let d_off = (2.0 * NODE_RADIUS * basis.proj_plane_dist() / c_off.depth)
            .min(DIAMETER_CAP_FRAC * 32.0);
        assert_eq!(
            strict_tier(d_on),
            strict_tier(d_off),
            "parity: on-slice/off-slice give the same tier"
        );
        assert!(
            (d_on - d_off).abs() < 1e-4,
            "and the same on-screen diameter"
        );
    }

    // ── rim tangent (not radial) mapping ────────────────────────────────────────

    #[test]
    fn rim_glyph_is_tangent_not_radial() {
        // Top / bottom of the orb (the aspect-corrected vertical offset dominates)
        // has a HORIZONTAL tangent; sides (the horizontal offset dominates) have a
        // VERTICAL tangent — a naive radial mapping would swap these.
        assert_eq!(
            rim_glyph(-4.0, 0.0, false),
            '─',
            "top of the orb ⇒ horizontal rule"
        );
        assert_eq!(
            rim_glyph(4.0, 0.0, false),
            '─',
            "bottom of the orb ⇒ horizontal rule"
        );
        assert_eq!(
            rim_glyph(0.0, -8.0, false),
            '│',
            "left side ⇒ vertical rule"
        );
        assert_eq!(
            rim_glyph(0.0, 8.0, false),
            '│',
            "right side ⇒ vertical rule"
        );
        // True diagonal corners: dx magnitude matched to the ASPECT-CORRECTED dy
        // (dx ≈ 2·dy, since a cell is ~2:1 tall), so neither axis dominates.
        assert_eq!(rim_glyph(-1.0, -2.0, false), '╭', "upper-left corner");
        assert_eq!(rim_glyph(-1.0, 2.0, false), '╮', "upper-right corner");
        assert_eq!(rim_glyph(1.0, -2.0, false), '╰', "lower-left corner");
        assert_eq!(rim_glyph(1.0, 2.0, false), '╯', "lower-right corner");
        // Heavy variant swaps in the bold box-drawing set (rim camouflage fix).
        assert_eq!(rim_glyph(-4.0, 0.0, true), '━');
        assert_eq!(rim_glyph(0.0, 8.0, true), '┃');
        assert_eq!(rim_glyph(-1.0, -2.0, true), '┏');
    }

    #[test]
    fn m_tier_is_a_framed_box_not_a_plus_sign() {
        // The M tier (3x7) must NOT use the generic ellipse formula (which
        // pinches to a single-point top/bottom rim at this size and, combined with
        // a radial rim_glyph, would read as a plus-sign). It is a literal small
        // rounded box: a closed top/bottom edge (not a single point), vertical
        // sides, no highlight touching the core.
        let roles = classify(3, 7);
        for c in 1..=5 {
            assert_eq!(roles[c], Role::Rim, "top edge col {c} must be Rim");
            assert_eq!(
                roles[2 * 7 + c],
                Role::Rim,
                "bottom edge col {c} must be Rim"
            );
        }
        assert_eq!(roles[7 + 1], Role::Rim, "left side");
        assert_eq!(roles[7 + 5], Role::Rim, "right side");
        assert_eq!(roles[7 + 3], Role::Core, "dead-centre core");
        assert!(
            !roles.contains(&Role::Highlight),
            "M must not carry a highlight (it would touch the core)"
        );
        // With the correct rim-tangent mapping, the top edge reads as a horizontal
        // rule and the sides as vertical rules — never a swapped "plus sign" shape.
        assert_eq!(
            rim_glyph(0.0 - 1.0, 3.0 - 3.0, false),
            '─',
            "top-centre must be a horizontal rule"
        );
        assert_eq!(
            rim_glyph(1.0 - 1.0, 1.0 - 3.0, false),
            '│',
            "left side must be a vertical rule"
        );
    }

    #[test]
    fn is_box_drawing_bg_flags_the_camouflage_glyph_set_only() {
        for g in ['─', '│', '┏', '┛', '═', '║', '╭', '╯'] {
            assert!(
                is_box_drawing_bg(u32::from(g)),
                "{g:?} must count as a camouflage background"
            );
        }
        for g in [' ', 'a', '◉', '∙', '('] {
            assert!(
                !is_box_drawing_bg(u32::from(g)),
                "{g:?} must not be flagged"
            );
        }
    }

    // ── Dead-node findability contrast ──────────────────────────────────────────

    #[test]
    fn dead_palette_pops_on_a_dark_background_and_keeps_the_dim_tone_on_a_light_one() {
        let dim = palette_for(NodeState::Dead);
        let dark_bg = dead_palette_for_bg(0.05);
        let light_bg = dead_palette_for_bg(0.6);
        assert_eq!(
            light_bg.core, dim.core,
            "a light/bright background keeps the original dim slate"
        );
        assert_ne!(
            dark_bg.core, dim.core,
            "a dark background swaps in the brighter pop tone"
        );
        let luma = |c: u32| unpack_rgb(c).dot(Vec3::new(0.2126, 0.7152, 0.0722));
        assert!(
            luma(dark_bg.core) > luma(dim.core),
            "the pop tone must be brighter than the dim slate"
        );
        // Both variants stay clearly COOL (blue/cyan-leaning) — never confusable
        // with a warm/bright POWERED node.
        let cool = |c: u32| {
            let v = unpack_rgb(c);
            v.z >= v.x
        };
        assert!(
            cool(dark_bg.core) && cool(dim.core),
            "dead must always read cool, whatever the background"
        );
    }

    // ── embedded nodes / sockets ─────────────────────────────────────────────────

    #[test]
    fn is_puzzle_node_tag_excludes_reserved_sentinels() {
        assert!(is_puzzle_node_tag(CellKind::Object, 0));
        assert!(is_puzzle_node_tag(
            CellKind::Object,
            re_core::RESERVED_TAG_BASE - 1
        ));
        assert!(!is_puzzle_node_tag(
            CellKind::Object,
            re_core::RESERVED_TAG_BASE
        ));
        assert!(!is_puzzle_node_tag(
            CellKind::Object,
            re_core::DOOR_OBJECT_TAG
        ));
        assert!(
            !is_puzzle_node_tag(CellKind::WallSolid, 5),
            "identity requires an Object cell"
        );
    }

    #[test]
    fn is_embedded_detects_lateral_wall_contact_but_not_open_air() {
        let mut map = room_with_node(10, 10, 5, (5, 5, 2));
        assert!(
            !is_embedded(&map, 5, 5, 2),
            "an isolated node in open air is free-standing"
        );
        map.set_cell(6, 5, 2, CellKind::WallSolid, 0);
        assert!(
            is_embedded(&map, 5, 5, 2),
            "a lateral WallSolid neighbour makes it embedded"
        );
    }

    #[test]
    fn is_embedded_detects_lateral_pipe_trace_contact() {
        let mut pipe_map = room_with_node(10, 10, 5, (5, 5, 2));
        pipe_map.set_cell(5, 6, 2, CellKind::WallTrace, encode_pipe_side_data(3));
        assert!(
            is_embedded(&pipe_map, 5, 5, 2),
            "a lateral pipe/trace neighbour makes it embedded"
        );
    }

    #[test]
    fn is_embedded_ignores_a_bare_floor_or_ceiling_neighbour() {
        // A node one layer above the auto-stamped floor (the shipped campaign's
        // own convention — node planes typically sit at z=2 in a 5-layer
        // room) must stay FREE-STANDING: a 6-axis rule here would flag almost
        // every shipped node and silently defeat the sprite feature (regression
        // guard for `offslice::tests::a_sprite_node_is_air_on_every_plane`, which
        // exercises exactly this z=1-above-floor placement).
        let flush_map = room_with_node(10, 10, 4, (5, 5, 1));
        assert!(
            !is_embedded(&flush_map, 5, 5, 1),
            "a bare floor/ceiling neighbour alone must not embed"
        );
    }

    #[test]
    fn embedded_node_stays_solid_and_gets_a_socket_plus_orb() {
        config::set_node_sprite_enabled(true);
        clear_tier_memory_for_test();
        let mut map = room_with_node(10, 10, 5, (5, 5, 2));
        map.set_cell(6, 5, 2, CellKind::WallSolid, 0); // lateral contact ⇒ embedded.
        assert!(is_embedded(&map, 5, 5, 2));
        let sd = map.side_data_at(5, 5, 2);
        assert!(
            !crate::is_sprite_node(&map, 5, 5, 2, CellKind::Object, sd),
            "embedded ⇒ not a sprite-air cell"
        );
        assert!(
            map.cell_at(5, 5, 2).is_solid(),
            "collision must be untouched"
        );

        let cam = cam_at(1.5, 5.5, 2.5, std::f32::consts::FRAC_PI_2);
        let fov = Fov::default();
        let (cols, rows) = (100, 32);
        let mut fb = cast_raw(&map, &cam, &fov, cols, rows);
        apply(&mut fb, &map, &cam, &fov, 0.0);

        let core = STATE_NODE.at(0);
        assert!(
            fb.cells().iter().any(|c| c.glyph == core),
            "an embedded node still shows its orb core"
        );
        let has_socket_tone = fb
            .cells()
            .iter()
            .any(|c| c.bg == SOCKET_FILL || c.fg == SOCKET_FRAME);
        assert!(
            has_socket_tone,
            "the embedded node's own footprint must carry the socket plate tones"
        );
    }

    #[test]
    fn free_standing_node_is_unaffected_by_the_socket_pass() {
        config::set_node_sprite_enabled(true);
        clear_tier_memory_for_test();
        let map = room_with_node(10, 10, 5, (5, 5, 2));
        let sd = map.side_data_at(5, 5, 2);
        assert!(
            crate::is_sprite_node(&map, 5, 5, 2, CellKind::Object, sd),
            "free-standing ⇒ a sprite-air cell"
        );
        let cam = cam_at(1.5, 5.5, 2.5, std::f32::consts::FRAC_PI_2);
        let fov = Fov::default();
        let (cols, rows) = (100, 32);
        let mut fb = cast_raw(&map, &cam, &fov, cols, rows);
        apply(&mut fb, &map, &cam, &fov, 0.0);
        assert!(
            !fb.cells()
                .iter()
                .any(|c| c.bg == SOCKET_FILL || c.fg == SOCKET_FRAME),
            "a free-standing node must never take the socket path"
        );
    }

    // ── pipe stubs ───────────────────────────────────────────────────────────────

    #[test]
    fn draw_pipe_stubs_reads_the_pipes_own_flow_state_not_the_node_tag() {
        let map = room_with_node_and_pipe((5, 5, 2), (1, 0, 0), 9);
        let cam = cam_at(1.5, 5.5, 2.5, std::f32::consts::FRAC_PI_2);
        let fov = Fov::default();
        let (cols, rows) = (100, 32);
        let basis = LineViewBasis::new(&cam, &fov, cols, rows);
        let node = SpriteNode {
            tag: 0,
            x: 5,
            y: 5,
            z: 2,
        };

        // ON: the PIPE's own tag (9) is powered/forward — reads hot, regardless of
        // the NODE's own tag (0, never written here).
        let mut state = CellStateMap::new();
        state.set_state(9, encode_flow_state(FlowDir::Forward));
        {
            let _scope = state_bus::CellStateScope::new(&state);
            let mut fb = Framebuffer::new(cols, rows);
            draw_pipe_stubs(&mut fb, &map, &basis, cols, rows, &node, &[]);
            assert!(
                fb.cells().iter().any(|c| c.fg == 0x0060_C0FF),
                "a Forward-flow pipe must draw its hot colour"
            );
        }
        // OFF: the empty bus (state 0 for tag 9) is unpowered — reads cold. A naive
        // comparison against the bare word `!= 1` would misread this exact
        // empty default as "on".
        state_bus::clear_active();
        let mut fb = Framebuffer::new(cols, rows);
        draw_pipe_stubs(&mut fb, &map, &basis, cols, rows, &node, &[]);
        assert!(
            !fb.cells().iter().any(|c| c.fg == 0x0060_C0FF),
            "an unpowered/empty-bus pipe must not read as hot"
        );
        assert!(
            fb.cells().iter().any(|c| c.fg == 0x0040_5060),
            "it must read as the cold stub colour instead"
        );
    }

    #[test]
    fn draw_pipe_stubs_never_writes_into_a_footprint_cell() {
        let map = room_with_node_and_pipe((5, 5, 2), (1, 0, 0), 9);
        let cam = cam_at(1.5, 5.5, 2.5, std::f32::consts::FRAC_PI_2);
        let fov = Fov::default();
        let (cols, rows) = (100, 32);
        let basis = LineViewBasis::new(&cam, &fov, cols, rows);
        let node = SpriteNode {
            tag: 0,
            x: 5,
            y: 5,
            z: 2,
        };
        let mut state = CellStateMap::new();
        state.set_state(9, encode_flow_state(FlowDir::Forward));
        let _scope = state_bus::CellStateScope::new(&state);

        // Baseline: with an EMPTY footprint (nothing excluded) the stub actually draws.
        let mut drawn_fb = Framebuffer::new(cols, rows);
        draw_pipe_stubs(&mut drawn_fb, &map, &basis, cols, rows, &node, &[]);
        assert!(
            drawn_fb.cells().iter().any(|c| c.fg == 0x0060_C0FF),
            "premise: the stub draws with no exclusion"
        );

        // A footprint claiming EVERY on-screen cell forces every candidate point
        // the stub could touch to be excluded — the frame must stay untouched.
        let footprint: Vec<_> = (0..rows)
            .flat_map(|r| (0..cols).map(move |c| (c, r, Role::Body, 0.0, 0.0)))
            .collect();
        let mut blocked_fb = Framebuffer::new(cols, rows);
        let before = blocked_fb.clone();
        draw_pipe_stubs(&mut blocked_fb, &map, &basis, cols, rows, &node, &footprint);
        assert_eq!(
            blocked_fb.cells(),
            before.cells(),
            "every candidate cell was excluded by the footprint guard"
        );
    }

    // ── edge / camera-plane clipping ─────────────────────────────────────────

    #[test]
    fn project_sphere_still_draws_a_node_whose_exact_centre_is_behind_the_plane() {
        let cam = cam_at(5.0, 5.0, 1.5, 0.0); // facing +y
        let fov = Fov::default();
        let basis = LineViewBasis::new(&cam, &fov, 100, 32);
        let on_plane = Vec3::new(5.0, 5.0, 1.5);
        assert!(
            basis.project(on_plane).is_none(),
            "premise: the bare point projection is dropped at perp ≈ 0"
        );
        assert!(
            basis.project_sphere(on_plane, NODE_RADIUS).is_some(),
            "the sphere's visible cap must still project"
        );
        let well_behind = Vec3::new(5.0, 5.0 - 2.0, 1.5);
        assert!(
            basis.project_sphere(well_behind, NODE_RADIUS).is_none(),
            "a sphere fully behind the plane still culls"
        );
    }

    #[test]
    fn a_node_just_behind_the_camera_plane_still_draws_cells() {
        config::set_node_sprite_enabled(true);
        clear_tier_memory_for_test();
        let map = room_with_node(10, 10, 5, (5, 5, 2)); // world centre (5.5, 5.5, 2.5)
                                                        // Facing +y (yaw 0); the camera has walked just past the node's plane
                                                        // (ry = -0.1, well under NODE_RADIUS but past PERP_MIN) while staying
                                                        // laterally centred on it — the plain `project` drops it, but its
                                                        // still-visible front cap must still draw, at/near screen centre.
        let cam = cam_at(5.5, 5.6, 2.5, 0.0);
        let fov = Fov::default();
        let mut fb = Framebuffer::new(100, 32);
        apply(&mut fb, &map, &cam, &fov, 0.0);
        let core = STATE_NODE.at(0);
        assert!(
            fb.cells().iter().any(|c| c.glyph == core),
            "the node's visible front cap must still draw"
        );
    }

    #[test]
    fn edge_clip_draws_the_in_frame_part_of_a_node_whose_centre_is_off_the_left_edge() {
        // The other half of camera-plane clipping: a centre that projects fine (perp comfortably above
        // PERP_MIN) but lands outside the pixel grid entirely — `resolve_node`
        // must clip per template-cell (node_sprite.rs's own screen-bounds check),
        // not cull the whole node just because `centre` itself is off-canvas.
        state_bus::clear_active();
        let map = room_with_node(10, 10, 5, (5, 5, 2));
        let cam = cam_at(5.5, 5.5, 2.5, 0.0);
        let fov = Fov::default();
        let (cols, rows) = (80, 60); // generous cap headroom so the diameter cap never binds
        let basis = LineViewBasis::new(&cam, &fov, cols, rows);
        let node = SpriteNode {
            tag: 0,
            x: 5,
            y: 5,
            z: 2,
        };

        // Force a large template (close depth) via the SAME formula `resolve_node`
        // uses, so the test's expectations track the real implementation instead
        // of a duplicated guess.
        let depth = 0.4_f32;
        let raw_d = 2.0 * NODE_RADIUS * basis.proj_plane_dist() / depth;
        let d = raw_d.min(DIAMETER_CAP_FRAC * rows as f32);
        let tier = select_tier(d, None);
        let (_, w) = tier_dims(tier);
        let cx = (w as f32 - 1.0) / 2.0;
        assert!(
            cx >= 3.0,
            "premise: pick a depth that yields a template wide enough to test clipping"
        );

        // Centre far enough left that the template's LEFT half is deep off-canvas
        // while its RIGHT edge (offset +cx) lands 3 columns inside the frame.
        let centre = ScreenPoint {
            col: 3.0 - cx,
            row: (rows as f32 - 1.0) / 2.0,
            depth,
        };
        assert!(
            centre.col + cx >= 0.0 && centre.col + cx < cols as f32,
            "premise: right edge is in-frame"
        );
        assert!(centre.col - cx < -0.5, "premise: left edge is off-canvas");

        let fb = Framebuffer::new(cols, rows);
        let prepared = resolve_node(&fb, &map, &basis, cols, rows, node, centre, 0.0, false);

        assert!(
            !prepared.footprint.is_empty(),
            "template cells still inside the frame must be recorded, not culled wholesale"
        );
        assert!(
            prepared.footprint.iter().all(|&(ci, ri, ..)| ci < cols && ri < rows),
            "every recorded footprint cell must be in-bounds — that's the whole point of the per-cell clip"
        );
        assert!(
            prepared.footprint.iter().any(|&(ci, ..)| ci <= 4),
            "the clipped-in part must sit near the left screen edge, where the template's right edge actually lands"
        );
        assert!(
            !prepared.visible.is_empty(),
            "the clipped-in cells must also pass occlusion in this open, unobstructed room"
        );
    }

    // ── whole-pass behaviour: occlusion, defaults, no-op, stability, rollback ────

    #[test]
    fn occluded_node_draws_nothing() {
        config::set_node_sprite_enabled(true);
        clear_tier_memory_for_test();
        let mut map = room_with_node(10, 10, 5, (7, 5, 2));
        for y in 0..10 {
            map.set_cell(4, y, 2, CellKind::WallSolid, 0);
        }
        let cam = cam_at(1.5, 5.5, 2.5, std::f32::consts::FRAC_PI_2);
        let fov = Fov::default();
        let mut fb = Framebuffer::new(100, 32);
        apply(&mut fb, &map, &cam, &fov, 0.0);
        let core = STATE_NODE.at(0);
        assert!(
            !fb.cells().iter().any(|c| c.glyph == core),
            "a fully occluded node must draw no core glyph"
        );
    }

    #[test]
    fn empty_state_bus_reads_powered_matching_todays_default() {
        config::set_node_sprite_enabled(true);
        clear_tier_memory_for_test();
        state_bus::clear_active();
        let map = room_with_node(10, 10, 5, (5, 5, 2));
        let cam = cam_at(1.5, 5.5, 2.5, std::f32::consts::FRAC_PI_2);
        let fov = Fov::default();
        let mut fb = Framebuffer::new(100, 32);
        apply(&mut fb, &map, &cam, &fov, 0.0);
        let core = STATE_NODE.at(0);
        assert!(
            fb.cells().iter().any(|c| c.glyph == core),
            "empty bus ⇒ powered, the shipped default"
        );
    }

    #[test]
    fn dead_node_never_drops_below_tier_s_within_the_findability_range() {
        config::set_node_sprite_enabled(true);
        clear_tier_memory_for_test();
        const ROWS: usize = 16;
        const COLS: usize = 60;
        let map = room_with_node(30, 10, 5, (25, 5, 2));
        let mut state = CellStateMap::new();
        state.set_state(0, 1); // dead
        let _scope = state_bus::CellStateScope::new(&state);
        // 11 world cells away — inside DEAD_MIN_TIER_DEPTH (12) — chosen far enough
        // out that the node's NATURAL (unforced) size sits at or below the `S`
        // floor, so the forced-minimum branch is actually exercised.
        let cam = cam_at(14.5, 5.5, 2.5, std::f32::consts::FRAC_PI_2);
        let fov = Fov::default();
        let (cols, rows) = (COLS, ROWS);
        let basis = LineViewBasis::new(&cam, &fov, cols, rows);
        let centre = basis.project(Vec3::new(25.5, 5.5, 2.5)).unwrap();
        assert!(
            centre.depth <= DEAD_MIN_TIER_DEPTH,
            "premise: within the findability range"
        );
        let mut fb = Framebuffer::new(cols, rows);
        apply(&mut fb, &map, &cam, &fov, 0.0);
        let core = STATE_NODE.at(1); // dead
        assert!(
            fb.cells().iter().any(|c| c.glyph == core),
            "dead core must still be drawn (never below S)"
        );
        assert!(
            fb.cells().iter().any(|c| c.glyph == u32::from('(')),
            "S tier's bracket must be present"
        );
    }

    #[test]
    fn dead_node_reads_the_pop_tone_against_a_dark_background_behind_it() {
        config::set_node_sprite_enabled(true);
        clear_tier_memory_for_test();
        let map = room_with_node(10, 10, 5, (5, 5, 2));
        let mut state = CellStateMap::new();
        state.set_state(0, 1); // dead
        let _scope = state_bus::CellStateScope::new(&state);
        let cam = cam_at(3.0, 5.5, 2.5, std::f32::consts::FRAC_PI_2);
        let fov = Fov::default();
        let (cols, rows) = (100, 32);
        let basis = LineViewBasis::new(&cam, &fov, cols, rows);
        let centre = basis.project(Vec3::new(5.5, 5.5, 2.5)).unwrap();
        let ci = centre.col.round().clamp(0.0, cols as f32 - 1.0) as usize;
        let ri = centre.row.round().clamp(0.0, rows as f32 - 1.0) as usize;
        let mut fb = Framebuffer::new(cols, rows);
        // A near-black background, as a dark floor would leave behind the node.
        fb.set(
            ci,
            ri,
            Cell {
                bg: 0x0004_0608,
                ..Cell::default()
            },
        );
        apply(&mut fb, &map, &cam, &fov, 0.0);
        let core = STATE_NODE.at(1); // dead
        let core_cell = fb
            .cells()
            .iter()
            .find(|c| c.glyph == core)
            .expect("dead core must draw");
        assert_eq!(
            core_cell.fg, DEAD_CORE_POP,
            "against a near-black background, the pop tone must be chosen"
        );
    }

    #[test]
    fn no_op_without_nodes() {
        config::set_node_sprite_enabled(true);
        clear_tier_memory_for_test();
        let map = stamp_geometry(&GeometrySpec {
            size: (8, 8, 4),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[],
        });
        let cam = cam_at(2.0, 2.0, 1.5, 0.0);
        let fov = Fov::default();
        let mut fb = Framebuffer::new(40, 20);
        let before = fb.clone();
        apply(&mut fb, &map, &cam, &fov, 0.0);
        assert_eq!(fb.cells(), before.cells(), "no sprite nodes ⇒ hard no-op");
    }

    #[test]
    fn stability_static_pose_renders_identical_glyphs_across_time() {
        config::set_node_sprite_enabled(true);
        clear_tier_memory_for_test();
        let map = room_with_node(10, 10, 5, (5, 5, 2));
        let cam = cam_at(1.5, 5.5, 2.5, std::f32::consts::FRAC_PI_2);
        let fov = Fov::default();
        let glyphs_at = |t: f32| {
            let mut fb = Framebuffer::new(80, 24);
            apply(&mut fb, &map, &cam, &fov, t);
            fb.cells().iter().map(|c| c.glyph).collect::<Vec<_>>()
        };
        assert_eq!(
            glyphs_at(0.0),
            glyphs_at(0.4),
            "breathe changes colour only, never the glyph grid"
        );
        assert_eq!(glyphs_at(0.0), glyphs_at(1.3));
    }

    #[test]
    fn ts_node_sprite_0_disables_the_pass_and_is_reproducible() {
        clear_tier_memory_for_test();
        let map = room_with_node(10, 10, 5, (5, 5, 2));
        let cam = cam_at(1.5, 5.5, 2.5, std::f32::consts::FRAC_PI_2);
        let fov = Fov::default();

        config::set_node_sprite_enabled(true);
        let mut on = Framebuffer::new(100, 32);
        apply(&mut on, &map, &cam, &fov, 0.0);

        config::set_node_sprite_enabled(false);
        let mut off1 = Framebuffer::new(100, 32);
        apply(&mut off1, &map, &cam, &fov, 0.0);
        let mut off2 = Framebuffer::new(100, 32);
        apply(&mut off2, &map, &cam, &fov, 0.0);

        assert_eq!(
            off1.cells(),
            off2.cells(),
            "TS_NODE_SPRITE=0 is reproducible"
        );
        assert_ne!(
            on.cells(),
            off1.cells(),
            "toggling the pass must change the frame"
        );
        let core = STATE_NODE.at(0);
        assert!(
            !off1.cells().iter().any(|c| c.glyph == core),
            "disabled ⇒ apply() is a hard no-op, no orb glyph"
        );

        config::set_node_sprite_enabled(true); // restore the shipping default for sibling tests.
    }

    // ── Minor: floor contact shadow restore ──────────────────────────────────────

    #[test]
    fn free_standing_node_over_open_floor_casts_a_small_contact_shadow() {
        config::set_node_sprite_enabled(true);
        clear_tier_memory_for_test();
        let map = room_with_node(10, 10, 5, (5, 5, 2));
        let cam = cam_at(1.5, 5.5, 2.5, std::f32::consts::FRAC_PI_2);
        let fov = Fov::default();
        let (cols, rows) = (100, 32);
        let before = cast_raw(&map, &cam, &fov, cols, rows);
        let mut fb = before.clone();
        apply(&mut fb, &map, &cam, &fov, 0.0);
        let darkened = (0..fb.cells().len())
            .filter(|&i| before.cells()[i].surface == SurfaceKind::Floor)
            .any(|i| fb.cells()[i].bg != before.cells()[i].bg);
        assert!(
            darkened,
            "a free-standing node over open floor must darken some floor cells"
        );
    }
}
