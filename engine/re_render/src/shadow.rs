//! `re_render::shadow` — worked-out cast shadows.
//!
//! A DDA shadow ray on the SAME voxel grid the main cast walks: from a shaded
//! cell's world hit point we march toward the static KEY light and ask the grid
//! `is the path blocked?`. A blocked path → the key light is occluded → the cell
//! is in (cast) shadow → its key diffuse + key specular are attenuated. This is
//! an honest, engine-pure occlusion model — only `map.cell_at` /
//! `is_solid` grid reads, no terminal, no allocation.
//!
//! `Soft` adds a few jittered taps over a small penumbra disc + distance-to-blocker
//! disc widening (PCSS-lite) for a soft, fractional shadow that reads in the ASCII
//! fg/bg + glyph ramp. `Hard` is one ray (binary); `Off` skips the pass entirely
//! (shadow_key stays 1.0 → pre-shadow output).
//!
//! This module is a pure mechanical extraction from `lib.rs` (the raycast core's
//! shadow sub-system); the constants it consumes (`SHADOW_*`) are owned by
//! [`crate::ascii_shader`] and imported here, not redefined.

use crate::ascii_shader::ShadowQuality;
use crate::ascii_shader::{
    SHADOW_BIAS, SHADOW_MAX_STEPS, SHADOW_SOFT_BASE, SHADOW_SOFT_MAX, SHADOW_TAPS,
};
use crate::floori;
use re_world::Map;

/// March a single shadow ray on the voxel grid from `(ox,oy,oz)` toward unit
/// direction `dir`, up to [`SHADOW_MAX_STEPS`] voxels. Returns `Some(dist)` (the
/// world distance to the first solid blocker) if occluded, else `None` (the ray
/// escaped to open space / the light without hitting anything → unoccluded).
///
/// A 3D DDA using the same `step / side_dist / delta_dist` scheme as the main
/// cast, extended to the z axis so a wall ABOVE/below the hit point (a ceiling
/// lip, a raised block) also casts. Pure: only `map.cell_at(...).is_solid()`.
#[inline]
pub(crate) fn shadow_ray_hit(
    map: &Map,
    ox: f32,
    oy: f32,
    oz: f32,
    dir: glam::Vec3,
    max_steps: u32,
) -> Option<f32> {
    // Guard a degenerate direction (normalize_or_zero may have produced 0).
    if dir.length_squared() < 1e-12 {
        return None;
    }
    let mut map_x = floori(ox);
    let mut map_y = floori(oy);
    let mut map_z = floori(oz);

    let inv = |d: f32| if d == 0.0 { 1e30 } else { (1.0 / d).abs() };
    let delta_x = inv(dir.x);
    let delta_y = inv(dir.y);
    let delta_z = inv(dir.z);

    let bound = |o: f32, m: i32, step: i32, delta: f32| -> (i32, f32) {
        if step < 0 {
            (-1, (o - m as f32) * delta)
        } else {
            (1, (m as f32 + 1.0 - o) * delta)
        }
    };
    let (step_x, mut sdx) = bound(ox, map_x, if dir.x < 0.0 { -1 } else { 1 }, delta_x);
    let (step_y, mut sdy) = bound(oy, map_y, if dir.y < 0.0 { -1 } else { 1 }, delta_y);
    let (step_z, mut sdz) = bound(oz, map_z, if dir.z < 0.0 { -1 } else { 1 }, delta_z);

    let mut travelled;
    for _ in 0..max_steps {
        // Advance along whichever axis reaches its next voxel boundary first.
        if sdx < sdy && sdx < sdz {
            travelled = sdx;
            sdx += delta_x;
            map_x += step_x;
        } else if sdy < sdz {
            travelled = sdy;
            sdy += delta_y;
            map_y += step_y;
        } else {
            travelled = sdz;
            sdz += delta_z;
            map_z += step_z;
        }
        if !map.in_bounds(map_x, map_y, map_z) {
            return None; // escaped the world → reached open space / the light.
        }
        let kind = map.cell_at(map_x, map_y, map_z);
        if kind.is_solid() {
            // NODE SPRITE: a FREE-STANDING puzzle node stays collidable but is
            // render-transparent — a shadow ray must pass through it too, or every
            // node would cast a wall-sized shadow blob (see `crate::is_sprite_node`).
            // An EMBEDDED node is deliberately left OUT of this skip: it is
            // solid geometry again (a recessed socket, not a hole), so it casts a
            // shadow like the wall/pipe/slab it's set into.
            if crate::is_sprite_node(
                map,
                map_x,
                map_y,
                map_z,
                kind,
                map.side_data_at(map_x, map_y, map_z),
            ) {
                continue;
            }
            return Some(travelled.max(0.0));
        }
    }
    // Ran out of march budget without a hit — treat as unoccluded.
    None
}

/// The geometric inputs to [`shadow_key_for_lod`] for one shaded cell: its perpendicular
/// `depth` (world cells, drives the distance-LOD), the world hit point `(hx, hy)` on
/// plane `cell_z`, and the geometric face normal `face_n`. Bundled so the entry point
/// stays under the argument-count lint.
#[derive(Clone, Copy)]
pub(crate) struct ShadowSample {
    pub(crate) depth: f32,
    pub(crate) hx: f32,
    pub(crate) hy: f32,
    pub(crate) cell_z: i32,
    pub(crate) face_n: glam::Vec3,
}

/// The KEY-light shadow factor `[0,1]` for a shaded cell at world hit point
/// `(hx,hy)` on plane `cell_z`, with geometric face normal `face_n`. `1.0` = fully
/// key-lit, `0.0` = fully in the key's cast shadow, fractional = penumbra.
///
/// * [`ShadowQuality::Off`] → always `1.0` (pre-shadow output).
/// * [`ShadowQuality::Hard`] → one ray: `0.0` if blocked, else `1.0`.
/// * [`ShadowQuality::Soft`] → [`SHADOW_TAPS`] jittered rays over a penumbra disc;
///   `shadow_key = unoccluded_taps / taps`. The disc widens with the distance to
///   the nearest blocker (PCSS-lite) so a far caster throws a softer edge.
///
/// DISTANCE-LOD (perf): `depth` is the shaded cell's perpendicular distance (world
/// cells). The effective quality is degraded with depth via the UNIFIED
/// [`crate::lod::LodConfig`] tier (shared with the shade path), so the hot
/// floor/ceiling loop only pays full soft-shadow cost in the NEAR tier the player
/// reads:
///   * FAR tier → skip entirely (`1.0`), regardless of `quality`.
///   * MID tier → at most a single HARD ray.
///   * NEAR tier → the requested `quality` (full Soft).
///
/// The tier thresholds are world-distance and configurable; the default reproduces
/// the historical 6/13 bands so small-grid golden cameras (everything well inside
/// NEAR) are byte-identical. The shadow ray's own march budget is also capped
/// by `depth` — a far cell's occluder is at most a handful of cells away on screen,
/// so the full [`SHADOW_MAX_STEPS`] is wasted there.
///
/// The origin is the hit point lifted a touch off the surface along `face_n`
/// ([`SHADOW_BIAS`]) so a lit face never self-shadows on its own voxel (shadow
/// acne). Pure + allocation-free: a handful of integer grid reads per cell.
///
/// The distance-LOD band is resolved from the unified [`crate::lod::LodConfig`]
/// (+ an optional per-material [`crate::lod::LodOverride`] `lod_ov`), NOT the bare
/// `SHADOW_SOFT_DIST`/`SHADOW_HARD_DIST` literals. The tier maps to the shadow-quality
/// ceiling: NEAR→Soft (full requested), MID→Hard (one ray), FAR→Off (skip). With the
/// default config this reproduces the historical 6/13 bands exactly; the hysteresis
/// deadband (when a previous tier is known) removes the hard-cutoff pop a naive
/// distance threshold would otherwise show. A hero material's `lod_ov` keeps its cast shadow soft farther out
/// than the global curve — threaded for BOTH walls and floor/ceiling cells. Pass
/// `None` for the global curve (the back-compatible, byte-identical default).
#[inline]
pub(crate) fn shadow_key_for_lod(
    map: &Map,
    quality: ShadowQuality,
    sample: ShadowSample,
    key_dir: glam::Vec3,
    lod_ov: Option<crate::lod::LodOverride>,
) -> f32 {
    use crate::lod::ShadowTierQuality;
    let ShadowSample {
        depth,
        hx,
        hy,
        cell_z,
        face_n,
    } = sample;
    if quality == ShadowQuality::Off {
        return 1.0;
    }
    if !depth.is_finite() {
        return 1.0;
    }
    // DISTANCE-LOD via the UNIFIED tier system: one near/mid/far decision shared with
    // the shade path, so shadow + shade degrade together. The tier's shadow CEILING
    // clamps the requested quality DOWN — FAR skips, MID drops to a single hard ray,
    // NEAR keeps the requested quality. (A non-finite depth handled above.)
    let lod_cfg = crate::lod::active();
    let tier_shadow = lod_cfg.tier_for(depth, lod_ov).detail(&lod_cfg).shadow;
    let (lod_quality, mid_band) = match tier_shadow {
        ShadowTierQuality::Off => return 1.0,
        ShadowTierQuality::Hard => (ShadowQuality::Hard, true),
        ShadowTierQuality::Soft => (quality, false),
    };
    // Back-facing surfaces (the key is behind the face) are self-shadowed by N·L
    // anyway; skip the ray. A small epsilon keeps grazing faces out of the test.
    if face_n.dot(key_dir) <= 0.02 {
        return 1.0;
    }
    // March budget. NEAR cells keep the FULL [`SHADOW_MAX_STEPS`] so their occlusion
    // is byte-identical to the pre-LOD output. MID cells (single hard ray) cap the
    // budget by depth — an occluder that still matters at depth `d` is within ~`d`
    // cells along the (mostly vertical) key ray, so the far tail of the march is
    // wasted there. FAR cells already returned above.
    let max_steps = if mid_band {
        (depth.ceil() as u32 + 2).clamp(4, SHADOW_MAX_STEPS)
    } else {
        SHADOW_MAX_STEPS
    };
    // Sample the surface at the cell's centre in z (the hit is on the wall/floor of
    // plane cell_z); bias off the face so the origin sits just outside the solid.
    let origin = glam::Vec3::new(hx, hy, cell_z as f32 + 0.5) + face_n * SHADOW_BIAS;

    if lod_quality == ShadowQuality::Hard {
        return match shadow_ray_hit(map, origin.x, origin.y, origin.z, key_dir, max_steps) {
            Some(_) => 0.0,
            None => 1.0,
        };
    }

    // Soft (PCSS-lite). First a centre ray: if it misses AND its neighbourhood is
    // open the cell is fully lit; if it hits, the blocker distance sets the disc
    // radius (near blocker → crisp, far blocker → wide penumbra).
    let centre = shadow_ray_hit(map, origin.x, origin.y, origin.z, key_dir, max_steps);
    let radius = match centre {
        // Distance-to-blocker softening: lerp the disc half-angle from BASE (near)
        // toward MAX (far) over the march range.
        Some(d) => {
            let t = (d / SHADOW_MAX_STEPS as f32).clamp(0.0, 1.0);
            SHADOW_SOFT_BASE + (SHADOW_SOFT_MAX - SHADOW_SOFT_BASE) * t
        }
        None => SHADOW_SOFT_BASE,
    };

    // Build an orthonormal basis spanning the disc perpendicular to the key dir.
    let up = if key_dir.z.abs() < 0.9 {
        glam::Vec3::Z
    } else {
        glam::Vec3::X
    };
    let tu = key_dir.cross(up).normalize_or_zero();
    let tv = key_dir.cross(tu).normalize_or_zero();

    // Jittered taps over the disc. The offsets are a fixed low-discrepancy set
    // (deterministic → golden-safe), scaled by the penumbra radius. The first tap
    // reuses the centre result (no extra ray).
    let mut unoccluded: u32 = if centre.is_none() { 1 } else { 0 };
    for i in 1..SHADOW_TAPS {
        // A cheap deterministic spiral offset in the disc.
        let a = i as f32 * 2.399_963_2; // golden angle
        let r = (i as f32 / SHADOW_TAPS as f32).sqrt() * radius;
        let off = tu * (a.cos() * r) + tv * (a.sin() * r);
        let dir = (key_dir + off).normalize_or_zero();
        if shadow_ray_hit(map, origin.x, origin.y, origin.z, dir, max_steps).is_none() {
            unoccluded += 1;
        }
    }
    unoccluded as f32 / SHADOW_TAPS as f32
}
