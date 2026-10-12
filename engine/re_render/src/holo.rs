//! `re_render::holo` — hologram objects (transparent wireframe overlay).
//!
//! A HOLOGRAM object is a connected set of transparent `CellKind::Hologram` voxels
//! (a box ⇒ a cube; an arbitrary voxel set ⇒ a mesh). The raycaster passes THROUGH
//! them — the wall/floor/ceiling cast already wrote the scene BEHIND. This pass
//! then OVERLAYS, per screen cell, a glowing depth-coloured CONTOUR wireframe with
//! horizontal scanlines, leaving the interior see-through.
//!
//! It runs as a SEPARATE post-pass over the framebuffer (after the column loop) so
//! it can compare NEIGHBOURING columns to find the object's left/right silhouette
//! edges (vertical contour lines) — a single column can't see its neighbour. Two
//! cheap stack-only grid walks per column collect the front/back hologram-face
//! crossing depths (no per-cell heap alloc); one reusable per-cast `Vec<HoloCol>`
//! (allocated only when the map actually HAS holograms) holds them. Then a draw
//! loop writes the contour glyphs + scanlines. Deterministic: pure float maths +
//! a fixed time drift, no `rand`, no clock in the inner loop.
//!
//! Pure mechanical extraction from `lib.rs`; no terminal, no behaviour change.

use crate::ascii_shader;
use crate::{floori, offslice, DIST_INFINITY, PERP_MIN};
use re_core::Framebuffer;
use re_world::{CellKind, Map};

/// Bright FRONT contour + glyph colour (the near silhouette / scanlines): a strong
/// hologram BLUE. The player asked for "front contour AND glyph are ONE colour".
pub(crate) const HOLO_FRONT: u32 = 0x0030_60FF;
/// The (front, back) contour colours for a hologram `style`. Style 0 (and any
/// reserved value) is the original blue pair, byte-identical to before styles
/// existed; a preset or authored colour ([`re_core::decode_holo_style`]) is the
/// front, and its back is the same hue lifted toward white, as the default cyan is
/// to its blue.
#[inline]
pub(crate) fn holo_colours(style: u16) -> (u32, u32) {
    match re_core::decode_holo_style(style) {
        None => (HOLO_FRONT, HOLO_BACK),
        Some(front) => (front, ascii_shader::mix_rgb(front, 0x00FF_FFFF, 0.35)),
    }
}

/// Every map layer that holds at least one hologram cell, ascending. The pass
/// draws each of them (a hologram at z = 5..7 is as visible from the ground as one
/// at eye height). Only called when the map has holograms.
pub(crate) fn holo_layers(map: &Map) -> Vec<i32> {
    let (sx, sy, sz) = map.size();
    (0..sz as i32)
        .filter(|&z| {
            (0..sy as i32).any(|y| (0..sx as i32).any(|x| map.cell_at(x, y, z).is_hologram()))
        })
        .collect()
}

/// Does any hologram (on any layer) draw in this column? The eye layer is capped
/// at `eye_max_dist` (its primary wall), other layers by their own geometry —
/// exactly as [`holo_pass`] decides. Used to mark special columns for re-cast.
#[allow(clippy::too_many_arguments)]
pub(crate) fn holo_column_hit(
    map: &Map,
    layers: &[i32],
    eye_z: i32,
    ox: f32,
    oy: f32,
    ray_dir_x: f32,
    ray_dir_y: f32,
    eye_max_dist: f32,
) -> bool {
    layers.iter().any(|&z| {
        let (cap, stop) = if z == eye_z {
            (eye_max_dist, false)
        } else {
            (DIST_INFINITY, true)
        };
        holo_scan(map, z, ox, oy, ray_dir_x, ray_dir_y, cap, stop).hit
    })
}

/// Soft BACK/SIDE contour colour (far edges seen THROUGH the transparent body): a
/// dimmer light-blue / CYAN, so depth reads near→far as bright-blue→faint-cyan.
pub(crate) const HOLO_BACK: u32 = 0x0060_C0FF;
/// Brightness the BACK/SIDE contour is dimmed to (it must be "less noticeable").
pub(crate) const HOLO_BACK_DIM: f32 = 0.45;
/// A faint inner bg tint multiplied into interior cells so the body reads as a
/// volume of cool light without going opaque (kept very low ⇒ still see-through).
const HOLO_BODY_TINT: u32 = 0x0008_1830;
/// Scanline period in SCREEN ROWS: a scan glyph lands every Nth row over the
/// object (the classic hologram look the player asked for — "every 4th row").
pub(crate) const HOLO_SCAN_PERIOD: i32 = 4;
/// Scanline downward drift speed (rows per second) over `ctx.time`.
const HOLO_SCAN_DRIFT: f32 = 2.0;
/// Max hologram grid-walk steps (bounded like the primary DDA).
const HOLO_MAX_STEPS: i32 = 256;

/// Per-column hologram crossing summary, collected in the first sub-pass and
/// consumed (with its neighbours) by the draw sub-pass. Pure data, `Copy`, no
/// allocation.
#[derive(Clone, Copy)]
pub(crate) struct HoloCol {
    /// Did this column's ray cross any hologram voxel (in front of the wall)?
    pub(crate) hit: bool,
    /// Perp distance to the NEAREST (front) hologram face — drives the slice.
    front_perp: f32,
    /// Perp distance to the FARTHEST hologram face (back of the body) — the back
    /// contour rides here when it differs from the front.
    back_perp: f32,
    /// `0` = the front face was a y-const (NS) face, `1` = x-const (EW) face.
    front_axis: u8,
    /// Style/colour tag from the front hologram cell's `side_data`
    /// ([`re_core::decode_holo_style`]; 0 = the default blue).
    style: u16,
    /// Grid x/y of the front (entry) hologram cell, so the draw can ask whether the
    /// hologram continues into the layer above/below (no seam line between layers).
    front_cell: (i32, i32),
}

impl HoloCol {
    const NONE: HoloCol = HoloCol {
        hit: false,
        front_perp: DIST_INFINITY,
        back_perp: DIST_INFINITY,
        front_axis: 0,
        style: 0,
        front_cell: (0, 0),
    };
}

/// Walk the voxel grid along one ray, recording the nearest + farthest HOLOGRAM
/// face crossings up to `max_dist` (the wall hit distance, so a wall occludes the
/// hologram). The walk steps voxels with the same `side_dist`/`delta_dist` DDA the
/// primary cast uses; it does NOT stop on a hologram (transparent) — it stops at
/// `max_dist`, the map edge, or the step cap. Pure grid reads, no allocation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn holo_scan(
    map: &Map,
    cell_z: i32,
    ox: f32,
    oy: f32,
    ray_dir_x: f32,
    ray_dir_y: f32,
    max_dist: f32,
    stop_on_solid: bool,
) -> HoloCol {
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

    let mut out = HoloCol::NONE;
    // Was the PREVIOUS stepped cell a hologram? A FALSE→TRUE transition is a front
    // (entry) face; TRUE→FALSE is a back (exit) face. We record the perp at each
    // transition; the first entry sets `front_perp`/`front_axis`, every transition
    // updates `back_perp` (the farthest crossing).
    let mut prev_holo = map.cell_at(map_x, map_y, cell_z).is_hologram();
    if prev_holo {
        // The camera is already inside a hologram voxel: the front face is at 0.
        out.hit = true;
        out.front_perp = PERP_MIN;
        out.back_perp = PERP_MIN;
        out.style = map.side_data_at(map_x, map_y, cell_z);
        out.front_cell = (map_x, map_y);
    }

    for _ in 0..HOLO_MAX_STEPS {
        // Perp distance to the boundary we are about to cross (this face's depth).
        let (perp, axis) = if side_dist_x < side_dist_y {
            let p = side_dist_x;
            side_dist_x += delta_dist_x;
            map_x += step_x;
            (p, 1u8)
        } else {
            let p = side_dist_y;
            side_dist_y += delta_dist_y;
            map_y += step_y;
            (p, 0u8)
        };
        if perp > max_dist {
            break; // a wall (or the far cap) occludes everything beyond here.
        }
        if !map.in_bounds(map_x, map_y, cell_z) {
            break; // escaped the map.
        }
        let kind = map.cell_at(map_x, map_y, cell_z);
        // NODE SPRITE: a FREE-STANDING puzzle node is collidable but render-
        // transparent — the holo silhouette walk must pass through it, not stop on
        // it (see `crate::is_sprite_node`). An EMBEDDED node (D4) is solid again
        // (a socket, not a hole), so it stops the walk like the geometry it sits in.
        let sprite_node = crate::is_sprite_node(
            map,
            map_x,
            map_y,
            cell_z,
            kind,
            map.side_data_at(map_x, map_y, cell_z),
        );
        if !sprite_node
            && stop_on_solid
            && (kind.is_solid() || matches!(kind, CellKind::Floor | CellKind::Ceiling))
        {
            // Off the eye layer there is no primary wall depth to cap the walk, so
            // the layer's own geometry occludes (a wall between us and the holo).
            break;
        }
        let now_holo = kind.is_hologram();
        if now_holo != prev_holo {
            // A silhouette face crossing (entry or exit of the hologram body).
            let perp = perp.max(PERP_MIN);
            if !out.hit {
                out.hit = true;
                out.front_perp = perp;
                out.front_axis = axis;
                out.style = map.side_data_at(map_x, map_y, cell_z);
                out.front_cell = (map_x, map_y);
            }
            out.back_perp = perp; // farthest crossing so far.
            prev_holo = now_holo;
        }
    }
    out
}

/// Pick the contour glyph for a wireframe edge by its orientation:
/// * a TOP/BOTTOM (horizontal) edge → `─`,
/// * a LEFT/RIGHT (vertical) edge → `│`,
/// * a corner where both meet → a box corner `┌┐└┘`.
#[inline]
fn holo_edge_glyph(top: bool, bottom: bool, left: bool, right: bool) -> u32 {
    match (top, bottom, left, right) {
        (true, _, true, _) => 0x250C,                // ┌
        (true, _, _, true) => 0x2510,                // ┐
        (_, true, true, _) => 0x2514,                // └
        (_, true, _, true) => 0x2518,                // ┘
        (true, _, _, _) | (_, true, _, _) => 0x2500, // ─
        (_, _, true, _) | (_, _, _, true) => 0x2502, // │
        _ => 0x2500,
    }
}

/// The hologram overlay pass. See the module docs for the model.
#[allow(clippy::too_many_arguments)]
pub(crate) fn holo_pass(
    map: &Map,
    fb: &mut Framebuffer,
    cell_z: i32,
    pos_x: f32,
    pos_y: f32,
    dir_x: f32,
    dir_y: f32,
    plane_x: f32,
    plane_y: f32,
    horizon: i32,
    proj_plane_dist: f32,
    vshift_per_perp: f32,
    time: f32,
) {
    let cols = fb.cols();
    let rows = fb.rows() as i32;
    if cols == 0 || rows <= 0 {
        return;
    }

    let layers = holo_layers(map);
    // Scanline phase: drift DOWN over time (mod the period), so the scan rows crawl.
    let scan_phase = (time * HOLO_SCAN_DRIFT).rem_euclid(HOLO_SCAN_PERIOD as f32);
    let vp = offslice::VProj::new(cell_z, horizon, proj_plane_dist, vshift_per_perp);
    let mut hcols: Vec<HoloCol> = vec![HoloCol::NONE; cols];
    // Far layers first, so where two layers overlap on screen the layer nearest the
    // eye paints last.
    let mut order = layers;
    order.sort_by_key(|&z| std::cmp::Reverse((z - cell_z).abs()));
    for z in order {
        let eye_layer = z == cell_z;
        // SUB-PASS 1 — collect each column's hologram crossing on this layer. One
        // Vec per cast (only reached when the map HAS holograms); no per-cell
        // allocation.
        for (x, hc) in hcols.iter_mut().enumerate() {
            let camera_x = 2.0 * (x as f32 + 0.5) / cols as f32 - 1.0;
            let ray_dir_x = dir_x + plane_x * camera_x;
            let ray_dir_y = dir_y + plane_y * camera_x;
            *hc = if eye_layer {
                // Up to where? The wall this column hit (so a wall in front of the
                // hologram occludes it). The framebuffer's `depth` at the horizon
                // row carries that (wall slice or floor/ceiling depth). Use a
                // generous cap if it reads as sky.
                let wall_depth = fb
                    .get(x, horizon.clamp(0, rows - 1) as usize)
                    .map(|c| c.depth)
                    .unwrap_or(DIST_INFINITY);
                let max_dist = if wall_depth.is_finite() && wall_depth < DIST_INFINITY {
                    wall_depth
                } else {
                    DIST_INFINITY
                };
                holo_scan(map, z, pos_x, pos_y, ray_dir_x, ray_dir_y, max_dist, false)
            } else {
                // Another layer: its own walls stop the walk, and each drawn cell
                // is depth-tested against what the cast already wrote there.
                holo_scan(
                    map,
                    z,
                    pos_x,
                    pos_y,
                    ray_dir_x,
                    ray_dir_y,
                    DIST_INFINITY,
                    true,
                )
            };
        }
        draw_layer(map, fb, &hcols, z, eye_layer, &vp, scan_phase);
    }
}

/// SUB-PASS 2 for one layer — draw the wireframe. For each hologram column, the
/// front face maps to a screen slice; its top/bottom rows are horizontal contour
/// (only where the hologram does not continue into the layer above/below), its
/// left/right edges (where the neighbour column is NOT a hologram, or is at a very
/// different depth) are vertical contour, the back face contributes a softer
/// contour seen through the body, and every Nth row is a scanline.
fn draw_layer(
    map: &Map,
    fb: &mut Framebuffer,
    hcols: &[HoloCol],
    z: i32,
    eye_layer: bool,
    vp: &offslice::VProj,
    scan_phase: f32,
) {
    let rows = fb.rows() as i32;
    for (x, &hc) in hcols.iter().enumerate() {
        if !hc.hit {
            continue;
        }
        let (front_rgb, back_rgb) = holo_colours(hc.style);
        let (fx, fy) = hc.front_cell;
        let open_above = !map.cell_at(fx, fy, z + 1).is_hologram();
        let open_below = !map.cell_at(fx, fy, z - 1).is_hologram();
        // The hologram cell spans world [z, z+1], projected exactly like every
        // wall (`offslice::VProj`, eye shear included) and rasterised with the same
        // row rule, so a hologram's edges agree with the walls around it. Each face
        // at its own perp.
        let (f_top, f_bot) = offslice::span_rows(
            vp.row(z as f32 + 1.0, hc.front_perp),
            vp.row(z as f32, hc.front_perp),
        );
        // Back slice extent (farther ⇒ smaller). Drawn for the back contour only.
        let back = hc.back_perp.max(hc.front_perp);
        let (b_top, b_bot) =
            offslice::span_rows(vp.row(z as f32 + 1.0, back), vp.row(z as f32, back));

        // Left/right SILHOUETTE detection: this column is a vertical edge if the
        // neighbour column has no hologram, or its front face is a big depth jump
        // away (the object's side turned away from us). A ~0.6-cell depth step
        // counts as an edge so a flat front face is solid line-free in its middle.
        let edge_vs = |n: Option<&HoloCol>| -> bool {
            match n {
                None => true,
                Some(nc) => !nc.hit || (nc.front_perp - hc.front_perp).abs() > 0.6,
            }
        };
        let left_edge = edge_vs(if x == 0 { None } else { hcols.get(x - 1) });
        let right_edge = edge_vs(hcols.get(x + 1));

        let draw_top = f_top.max(0);
        let draw_bot = f_bot.min(rows);
        for row in draw_top..draw_bot {
            let is_top = row == f_top && open_above;
            let is_bot = row == f_bot - 1 && open_below;
            // A scanline row: every Nth screen row, drifting with time.
            let is_scan = ((row as f32 - scan_phase).rem_euclid(HOLO_SCAN_PERIOD as f32)) < 1.0;
            // Back contour: a row that lands on the BACK slice's top/bottom edge and
            // is INSIDE the front body (so it reads as seen-through-the-volume).
            let is_back_edge = hc.back_perp > hc.front_perp + 0.05
                && ((row == b_top && open_above) || (row == b_bot - 1 && open_below))
                && row > f_top
                && row < f_bot - 1;

            let mut cell = match fb.get(x, row as usize) {
                Some(c) => *c,
                None => continue,
            };
            // Off the eye layer, geometry nearer than the hologram (a floor slab, a
            // balcony edge, a ramp) already owns this cell.
            if !eye_layer && cell.depth < hc.front_perp - 0.05 {
                continue;
            }

            if is_top || is_bot || left_edge || right_edge {
                // FRONT contour: bright line + matching glyph colour.
                cell.glyph = holo_edge_glyph(is_top, is_bot, left_edge, right_edge);
                cell.fg = front_rgb;
                cell.depth = hc.front_perp;
                // SMART-MATERIAL SCAN-GLOW: a CURLY colored underline in the
                // hologram hue under the contour glyph → the projected light reads
                // as glowing just beneath the wireframe edge.
                cell.style = holo_underline_glow(cell.style, cell.fg);
            } else if is_back_edge {
                // BACK / SIDE contour: soft, dimmer, seen through the body.
                cell.glyph = 0x2500; // ─
                cell.fg = dim_rgb_local(back_rgb, HOLO_BACK_DIM);
                // Keep the existing (scene) bg so it still reads as see-through.
                cell.style = holo_underline_glow(cell.style, cell.fg);
            } else if is_scan {
                // SCANLINE: a horizontal rule, same family as the front contour.
                cell.glyph = 0x2500; // ─
                cell.fg = dim_rgb_local(front_rgb, 0.85);
                cell.depth = hc.front_perp;
                // The scan ROWS carry the curly scan-glow — the signature effect.
                cell.style = holo_underline_glow(cell.style, cell.fg);
            } else {
                // INTERIOR: see-through. Leave the scene glyph/fg; only wash the bg a
                // hair toward the hologram's hue so the body reads as a faint volume
                // of light.
                cell.bg = add_rgb_local(cell.bg, body_tint(hc.style, front_rgb));
                fb.set(x, row as usize, cell);
                continue;
            }
            fb.set(x, row as usize, cell);
        }
    }
}

/// The faint interior bg wash: the original cool tint for the default style, the
/// style's own hue at the same (very low) strength otherwise.
#[inline]
fn body_tint(style: u16, front: u32) -> u32 {
    if re_core::decode_holo_style(style).is_none() {
        HOLO_BODY_TINT
    } else {
        dim_rgb_local(front, 0.19)
    }
}

/// The HOLOGRAM SCAN-GLOW underline accent: stamp a CURLY (undercurl) colored
/// underline, in a brightened take on the cell's own hologram-hue `fg`, onto a
/// wireframe / scanline cell's [`re_core::CellStyle`].
///
/// This is the hologram-OBJECT analogue of the material underline-accent layer
/// ([`crate::ascii_shader::cell_style`]'s `UnderlineAccent`): the holo wireframe
/// pass writes its contour/scan cells directly (it does not go through a
/// `Material`), so the accent is applied here. The curl sits just under the line
/// glyph → the projected light reads as glowing beneath each contour edge and scan
/// row. Pure, allocation-free. The dump emits it as `4:3` + `58;2;…`; live ratatui
/// falls back to a single colored underline (see `re_present`'s substyle note).
#[inline]
fn holo_underline_glow(style: re_core::CellStyle, fg: u32) -> re_core::CellStyle {
    // Lift the fg toward white a touch so the under-glow reads as a distinct,
    // brighter accent in the same hue family (not a flat copy of the glyph colour).
    let glow = ascii_shader::mix_rgb(fg, 0x00FF_FFFF, 0.22);
    style
        .with_underline_color(glow)
        .with_underline_style(re_core::UnderlineStyle::Curly)
}

/// Local packed-colour dim (avoids importing the shader helper into the hot path
/// signature churn). `factor` in `[0,1]`.
#[inline]
pub(crate) fn dim_rgb_local(c: u32, factor: f32) -> u32 {
    ascii_shader::dim_rgb(c, factor)
}

/// Local additive colour blend (saturating per channel) for the faint body tint.
#[inline]
fn add_rgb_local(a: u32, b: u32) -> u32 {
    let ar = (a >> 16) & 0xFF;
    let ag = (a >> 8) & 0xFF;
    let ab = a & 0xFF;
    let br = (b >> 16) & 0xFF;
    let bg = (b >> 8) & 0xFF;
    let bb = b & 0xFF;
    let r = (ar + br).min(0xFF);
    let g = (ag + bg).min(0xFF);
    let bl = (ab + bb).min(0xFF);
    (r << 16) | (g << 8) | bl
}
