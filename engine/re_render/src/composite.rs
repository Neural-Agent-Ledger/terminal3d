//! `re_render::composite` — near raycast → opaque/transparent layer split.
//!
//! Builds the two NEAR composite layers (rows 1+2 of the tiered composite
//! stack) from one DDA raycast. The cast fills a scratch [`Framebuffer`]; each
//! shaded cell is then routed by its material's
//! [`Material::is_transparent`](crate::ascii_shader::Material::is_transparent)
//! flag into either the `opaque` layer (slot 1, `Opaque`) or the `transparent`
//! layer (slot 2, `OverKeepBg`). The `present` masks come for free from
//! [`Layer::set`].
//!
//! This module does NOT touch the `far` layer (owned by [`crate::far_cache`]) or
//! the `emitters` layer (owned by `emitter.rs`).

use bevy::prelude::*;
use re_core::{Cell, CompositeStack, FrameTimingMetrics, Framebuffer, Layer};
use re_world::{CameraPose, Map};

use crate::ascii_shader::MaterialRegistry;
use crate::emitter::{splat_particles_kinds, Emitter, EmitterPlugin, EmitterTickSet, ParticlePool};
use crate::far_cache::FarCache;
use crate::gbuffer::NearGBuffer;
use crate::near_cache::NearRing;
use crate::quadrant::{blit_quadrant, needs_subcell, SubSample};
use crate::tier::{far_needs_rebake, TierClock};
use crate::{
    raycast_gameplay_capture, raycast_with_time, registry_for_map, reshade_animated_into,
    shade_subray, surface_material, Fov, SubViewBasis,
};

/// True iff every component of the camera pose is finite (no `NaN`/`inf`). A
/// non-finite pose would poison the quantized near key, the ring bake and every
/// ray, so the render entry points early-return on it (robustness guard).
#[inline]
fn pose_is_finite(cam: &CameraPose) -> bool {
    cam.pos.x.is_finite()
        && cam.pos.y.is_finite()
        && cam.pos.z.is_finite()
        && cam.yaw.is_finite()
        && cam.pitch.is_finite()
}

/// While a near-ring bake is IN PROGRESS, restart it only on a genuine position
/// MOVE (≥0.5 cell on x or y). A pure rotation is covered by the bake's angular
/// margin, so it must NOT restart the in-flight ring (else continuous mouse-look
/// would restart it forever and it would never complete). Shared by the pan and
/// structural near-ring prime sites so the threshold is defined once.
#[inline]
fn near_ring_in_progress_restart_on_move(near_ring: &NearRing, cam: &CameraPose) -> bool {
    let (_sy, sp) = near_ring.bake_start_pose();
    (cam.pos.x - sp.x).abs() >= 0.5 || (cam.pos.y - sp.y).abs() >= 0.5
}

/// Run ONE incremental near-ring bake step and, when it COMPLETES the back ring,
/// record the completion frame in `dirty.near_ring_bake_frame` (the slow-safety-tick
/// clock). The triplicated bake-prime tail (`bake_step` + `if completed { record }`)
/// of the pan and structural branches, hoisted into one place.
#[allow(clippy::too_many_arguments)]
#[inline]
fn near_ring_prime_step(
    near_ring: &mut NearRing,
    map: &Map,
    cam: &CameraPose,
    registry: &MaterialRegistry,
    fov: &Fov,
    time: f32,
    should_restart: bool,
    clock: &TierClock,
    dirty: &mut CompositeDirty,
) {
    let completed = near_ring.bake_step(map, cam, registry, fov, time, should_restart);
    if completed {
        dirty.near_ring_bake_frame = clock.frame;
    }
}

/// True iff the map has zero volume (any dimension is 0) — there is nothing to
/// cast against, so the render entry points produce nothing rather than walk an
/// empty grid (robustness guard).
#[inline]
fn map_is_empty(map: &Map) -> bool {
    let (sx, sy, sz) = map.size();
    sx == 0 || sy == 0 || sz == 0
}

/// Run the near raycast for `map` + `cam` and SPLIT the shaded cells into
/// `stack.opaque` vs `stack.transparent` by material transparency.
///
/// A cell whose surface material reports
/// [`Material::is_transparent`](crate::ascii_shader::Material::is_transparent) goes to the
/// transparent layer (hologram/glass — glyph+fg over the bg behind); every other
/// cell goes to the opaque layer. Both layers are cleared first; only written
/// cells are marked present, so the composite fold stays O(populated cells).
///
/// The bare static-cast entry point uses `time = 0`; the Bevy render system
/// wired below threads live time / memo through here instead.
pub fn build_world_layers(
    map: &Map,
    cam: &CameraPose,
    registry: &MaterialRegistry,
    stack: &mut CompositeStack,
) {
    // Back-compat convenience: allocate a one-shot scratch and forward. The hot
    // (per-frame) path uses [`build_world_layers_into`] with a PERSISTENT scratch
    // so no Framebuffer is allocated on a moving frame.
    let (cols, rows) = stack.dims();
    let mut scratch = Framebuffer::new(cols, rows);
    build_world_layers_into(map, cam, registry, &mut scratch, stack);
}

/// As [`build_world_layers`] but the caller hands in a PERSISTENT `scratch`
/// [`Framebuffer`] (allocated once, reused every frame) — no per-frame heap churn
/// on the near path (critic M3). `scratch` is resized to the stack dims if needed,
/// then the near scene is cast into it and split into opaque/transparent layers.
pub fn build_world_layers_into(
    map: &Map,
    cam: &CameraPose,
    registry: &MaterialRegistry,
    scratch: &mut Framebuffer,
    stack: &mut CompositeStack,
) {
    let (cols, rows) = stack.dims();
    stack.opaque.clear();
    stack.transparent.clear();
    if cols == 0 || rows == 0 {
        return;
    }
    if scratch.cols() != cols || scratch.rows() != rows {
        scratch.resize(cols, rows);
    }
    raycast_with_time(map, cam, registry, 0.0, scratch);
    split_scratch_into_layers(registry, scratch, stack);
}

/// As [`build_world_layers`] but with an explicit [`Fov`] driving the cast — the
/// `--dump` / configured-FOV path. Convenience wrapper that allocates a one-shot
/// scratch; the per-frame system uses [`build_world_layers_fov_into`].
pub fn build_world_layers_fov(
    map: &Map,
    cam: &CameraPose,
    registry: &MaterialRegistry,
    fov: &Fov,
    stack: &mut CompositeStack,
) {
    let (cols, rows) = stack.dims();
    let mut scratch = Framebuffer::new(cols, rows);
    build_world_layers_fov_into(map, cam, registry, fov, &mut scratch, stack);
}

/// As [`build_world_layers_fov`] but with a caller-owned PERSISTENT `scratch`
/// [`Framebuffer`] — the hot path's zero-per-frame-alloc variant (critic M3).
pub fn build_world_layers_fov_into(
    map: &Map,
    cam: &CameraPose,
    registry: &MaterialRegistry,
    fov: &Fov,
    scratch: &mut Framebuffer,
    stack: &mut CompositeStack,
) {
    let (cols, rows) = stack.dims();
    stack.opaque.clear();
    stack.transparent.clear();
    if cols == 0 || rows == 0 {
        return;
    }
    if scratch.cols() != cols || scratch.rows() != rows {
        scratch.resize(cols, rows);
    }
    // GAMEPLAY path: cast with the standing vertical-shear reference so a jump/fall
    // shears the world (rising reads as rising) while a grounded player renders
    // centred. The generic public `raycast*` entries use the camera's own z (shift 0)
    // so previews/QA never skew — see `re_render::raycast_with_time_fov_gameplay`.
    crate::raycast_with_time_fov_gameplay(map, cam, registry, fov, 0.0, scratch);
    split_scratch_into_layers(registry, scratch, stack);
}

/// As [`build_world_layers_fov_into`] but at an explicit animation `time` AND
/// populating the SELECTIVE-RE-SHADE G-buffer (`capture`): every drawn near cell's
/// exact [`crate::shade_cell`] inputs are cached and the animated, non-mirror cells
/// indexed, so a later animation-only frame re-shades only those from cached
/// geometry ([`reshade_animated_into`](crate::reshade_animated_into)). The layer output is identical to a
/// non-capturing cast at the same `time` — the capture is a pure side-channel.
///
/// This is the FULL (structural) near cast on the live composite path; the capture
/// it builds is what the cheap animation-only path replays.
#[allow(clippy::too_many_arguments)] // caster hand-off (scene + scratch + capture).
pub fn build_world_layers_fov_capture_into(
    map: &Map,
    cam: &CameraPose,
    registry: &MaterialRegistry,
    fov: &Fov,
    time: f32,
    scratch: &mut Framebuffer,
    capture: &mut NearGBuffer,
    stack: &mut CompositeStack,
) {
    let (cols, rows) = stack.dims();
    stack.opaque.clear();
    stack.transparent.clear();
    if cols == 0 || rows == 0 {
        return;
    }
    if scratch.cols() != cols || scratch.rows() != rows {
        scratch.resize(cols, rows);
    }
    raycast_gameplay_capture(map, cam, registry, fov, time, capture, scratch);
    split_scratch_into_layers(registry, scratch, stack);
}

/// Route each shaded scratch cell into the opaque vs transparent layer by its
/// surface material's [`Material::is_transparent`] flag. Shared by both build
/// paths; reads `scratch`, writes the two near layers (already cleared by caller).
fn split_scratch_into_layers(
    registry: &MaterialRegistry,
    scratch: &Framebuffer,
    stack: &mut CompositeStack,
) {
    let (cols, rows) = stack.dims();
    for row in 0..rows {
        for col in 0..cols {
            let cell = match scratch.get(col, row) {
                Some(c) => *c,
                None => continue,
            };
            // Route by the cell's surface material. The same dispatch the shader
            // uses (`surface` + `material` tag → MaterialId → registry).
            let mat_id = surface_material(cell.surface, cell.material);
            let mat = registry.get(mat_id);
            if mat.is_underline_only_transparent() {
                // GLASS-MARKER (underline-only transparent): paint NOTHING into either
                // near layer, so the scene behind it resolves untouched (fully
                // see-through). Its colored-underline "glass marker" is stamped over the
                // resolved cell by [`apply_glass_markers`] AFTER the fold — see there.
                // Skipping both layers is exactly "no fg/bg/glyph", as the design asks.
                continue;
            }
            if mat.is_transparent() {
                stack.transparent.set(col, row, cell);
            } else {
                stack.opaque.set(col, row, cell);
            }
        }
    }
}

/// Stamp the "glass marker" colored underline of every UNDERLINE-ONLY TRANSPARENT
/// cell onto the resolved frame, AFTER the layer fold. A glass-marker cell was
/// skipped from both near layers by [`split_scratch_into_layers`], so the resolved
/// cell shows whatever is BEHIND it (the scene, untouched). This pass re-reads the
/// scratch, finds those cells by material, and copies ONLY their underline (the
/// colored-underline channel + its substyle shape) onto the resolved cell — leaving
/// the resolved glyph / fg / bg from behind exactly as they are. The result: a
/// see-through cell whose only mark is a coloured underline rule, the "glass" cue.
///
/// Pure compute, no allocation; reads `scratch` (the per-frame near cast) + the
/// registry, writes the resolved `out`. A scene with no glass-marker material does
/// nothing (the per-cell material check short-circuits).
fn apply_glass_markers(registry: &MaterialRegistry, scratch: &Framebuffer, out: &mut Framebuffer) {
    let (cols, rows) = (
        scratch.cols().min(out.cols()),
        scratch.rows().min(out.rows()),
    );
    for row in 0..rows {
        for col in 0..cols {
            let Some(cell) = scratch.get(col, row) else {
                continue;
            };
            let mat_id = surface_material(cell.surface, cell.material);
            if !registry.get(mat_id).is_underline_only_transparent() {
                continue;
            }
            // The scratch cell carries the glass-marker underline (stamped by
            // `cell_style_full`'s `Volume`/`Inked` underline accent). Copy ONLY that
            // colored-underline channel + substyle onto the resolved cell behind it,
            // keeping its glyph/fg/bg/depth — so the cell stays see-through with just
            // the glass-marker rule.
            if let Some(below) = out.get(col, row).copied() {
                let mut merged = below;
                merged.style = re_core::CellStyle {
                    flags: below.style.flags | (cell.style.flags & re_core::CellStyle::UNDERLINE),
                    underline: if cell.style.has_underline_color() {
                        cell.style.underline
                    } else {
                        below.style.underline
                    },
                    underline_style: if cell.style.underline_style().is_some() {
                        cell.style.underline_style()
                    } else {
                        below.style.underline_style()
                    },
                };
                out.set(col, row, merged);
            }
        }
    }
}

// ─────────────────────── local 2×2 sub-cell refinement ──────────────────────
//
// The NEAR opaque layer is cast at one ray per cell (`build_world_layers_fov`).
// This pass then runs the cheap `quadrant::needs_subcell` classifier over that
// layer and, for the small fraction of cells it marks (silhouette edges), casts
// FOUR sub-rays at the cell's 2×2 sub-centres and replaces the full glyph with a
// `quadrant` blit (one of 16 block glyphs + a near/far 2-colour split). Flat / far
// cells keep their single glyph — so the overhead is the marked fraction × 4 rays,
// not a full-frame supersample (spec §7 economy).

/// Cells at/beyond this perpendicular depth are NEVER refined: distant background
/// must not pay the sub-cell cost (and must not noise — it is also where the
/// far-freeze lives). Tuned to the near tier's working range; a cell farther than a
/// handful of rooms reads fine as a single glyph.
///
/// DISTANCE-GATING (perf): set just past the shadow FAR band
/// ([`SHADOW_HARD_DIST`](crate::ascii_shader::SHADOW_HARD_DIST))
/// so silhouette refinement and cast-shadow detail fade out together. A 2×2 quadrant
/// blit on a cell this far away occupies ~1 glyph of screen — its anti-alias is
/// invisible — so refining it is wasted sub-rays. Lowering this from the historical
/// 16.0 only drops genuinely far cells (everything a player reads is well inside it).
pub const SUBCELL_MAX_DEPTH: f32 = 12.0;

/// The four sub-centres of a screen cell at `(col, row)`, in fractional cell units:
/// `(col + qx, row + qy)` for `qx, qy ∈ {0.25, 0.75}`, ordered `[TL, TR, BL, BR]` to
/// match the `quadrant` mask bit order.
#[inline]
fn sub_centres(col: usize, row: usize) -> [(f32, f32); 4] {
    let (c, r) = (col as f32, row as f32);
    [
        (c + 0.25, r + 0.25), // TL
        (c + 0.75, r + 0.25), // TR
        (c + 0.25, r + 0.75), // BL
        (c + 0.75, r + 0.75), // BR
    ]
}

/// Gather the up-to-four orthogonal neighbour depths of `(col, row)` from `opaque`
/// into `buf`, returning the populated slice. A neighbour that is absent (no opaque
/// cell — open sky / far gap) contributes `f32::INFINITY` (a silhouette against
/// space). Edge cells simply have fewer neighbours.
#[inline]
fn neighbour_depths(opaque: &Layer, col: usize, row: usize, buf: &mut [f32; 4]) -> usize {
    let (cols, rows) = opaque.dims();
    let mut n = 0;
    let push = |c: isize, r: isize, buf: &mut [f32; 4], n: &mut usize| {
        if c < 0 || r < 0 || c as usize >= cols || r as usize >= rows {
            return;
        }
        let d = opaque
            .get(c as usize, r as usize)
            .map(|cell| cell.depth)
            .unwrap_or(f32::INFINITY);
        buf[*n] = d;
        *n += 1;
    };
    let (ci, ri) = (col as isize, row as isize);
    push(ci - 1, ri, buf, &mut n);
    push(ci + 1, ri, buf, &mut n);
    push(ci, ri - 1, buf, &mut n);
    push(ci, ri + 1, buf, &mut n);
    n
}

/// Refine the NEAR opaque layer with the local 2×2 quadrant blitter. Convenience
/// wrapper that allocates a one-shot marks buffer and forwards to
/// [`refine_subcells_buf`]. The per-frame system uses [`refine_subcells_buf`] with
/// a PERSISTENT buffer (no per-frame heap churn). Returns the number of cells
/// refined (the marked fraction — used by the economy test).
pub fn refine_subcells(
    map: &Map,
    cam: &CameraPose,
    registry: &MaterialRegistry,
    fov: &Fov,
    opaque: &mut Layer,
) -> usize {
    let mut marks: Vec<(usize, usize)> = Vec::new();
    refine_subcells_buf(map, cam, registry, fov, opaque, &mut marks)
}

/// As [`refine_subcells`] but the caller hands in a PERSISTENT `marks` buffer
/// (allocated once, reused every frame) — no per-frame heap allocation on the near
/// path (perf). The buffer is `.clear()`ed on entry, so any prior contents are
/// discarded; only its CAPACITY is reused. For each present opaque cell nearer than
/// [`SUBCELL_MAX_DEPTH`], run the silhouette classifier against its neighbours'
/// depths; a marked cell casts four sub-rays (via [`shade_subray`]) and is
/// overwritten with the quadrant blit.
///
/// Pure compute apart from the caller-owned `marks` scratch (a fixed 4-element
/// neighbour buffer + a 4-sample array stay on the stack). Reads the opaque layer's
/// pre-cast depths to classify, then writes the refined cell back into the SAME layer.
pub fn refine_subcells_buf(
    map: &Map,
    cam: &CameraPose,
    registry: &MaterialRegistry,
    fov: &Fov,
    opaque: &mut Layer,
    marks: &mut Vec<(usize, usize)>,
) -> usize {
    let (cols, rows) = opaque.dims();
    if cols == 0 || rows == 0 {
        marks.clear();
        return 0;
    }
    let basis = SubViewBasis::new(cam, fov, cols, rows);

    // Collect the (col,row) to refine FIRST (immutable classify pass), then apply the
    // overwrites — so a refined cell's new depth never feeds a later cell's classify
    // (the silhouette decision must use the original one-ray G-buffer). The marks
    // buffer is caller-owned + reused; clear it (keep capacity) before refilling.
    marks.clear();
    let mut nbuf = [0.0f32; 4];
    for row in 0..rows {
        for col in 0..cols {
            let here = match opaque.get(col, row) {
                Some(c) => *c,
                None => continue, // absent (sky gap) — nothing to refine.
            };
            if !(here.depth.is_finite() && here.depth < SUBCELL_MAX_DEPTH) {
                continue; // far / frozen / miss — never refined.
            }
            let n = neighbour_depths(opaque, col, row, &mut nbuf);
            if needs_subcell(here.depth, &nbuf[..n], false) && !is_fitted_text_cell(registry, &here)
            {
                marks.push((col, row));
            }
        }
    }

    for &(col, row) in marks.iter() {
        let centres = sub_centres(col, row);
        let mut samples = [SubSample::from_cell(&Cell::default()); 4];
        for (k, &(cf, rf)) in centres.iter().enumerate() {
            let cell = shade_subray(map, registry, &basis, cf, rf);
            samples[k] = SubSample::from_cell(&cell);
        }
        let blit = blit_quadrant(samples);
        // Preserve the original cell's geometry fields (surface/material/side/hit) so
        // the present + postfx layers still see a coherent G-buffer; only the glyph,
        // the two colours and the depth (nearest sub-sample) are replaced.
        if let Some(orig) = opaque.get(col, row).copied() {
            let refined = Cell {
                glyph: blit.glyph,
                fg: blit.fg,
                bg: blit.bg,
                depth: blit.depth,
                ..orig
            };
            opaque.set(col, row, refined);
        }
    }

    marks.len()
}

/// Is this cell a FITTED TEXT surface (a `[note]` placard or a code wall)?
///
/// Sub-cell refinement replaces a cell with a 2×2 quadrant blit built from four
/// sub-rays at ±¼ cell. On ordinary geometry that is an anti-aliased silhouette — a
/// clear win. On a TEXT panel it is destructive: the ±¼ sub-rays land in the
/// NEIGHBOURING source characters, so a refined cell shows a character from one column
/// over. That is precisely the `thhe latttice` doubling (and its dropped letters) that
/// the fitted-span clamp exists to remove: the clamp lays the panel out 1:1, and the
/// refine pass then re-deals ~2 cells per column out of that alignment. A text cell owns
/// its glyph, so it is never refined.
///
/// One registry lookup, evaluated only for a cell that already passed the depth +
/// silhouette classifier (a small minority of near cells), so the classify pass keeps
/// its cost.
#[inline]
fn is_fitted_text_cell(registry: &MaterialRegistry, cell: &Cell) -> bool {
    matches!(
        registry
            .get(crate::surface_material(cell.surface, cell.material))
            .recipe,
        crate::ascii_shader::TextureRecipe::NoteWall { .. }
            | crate::ascii_shader::TextureRecipe::CodeWall { .. }
    )
}

// ──────────────────── tiered composite render plugin ────────────────────────
//
// `CompositeRenderPlugin` is the LIVE renderer that produces the `Framebuffer`
// (it superseded an earlier flat single-pass raycast plugin, now removed). Instead
// of one flat raycast it drives the full tiered composite:
//
//   far    (rare)         — radial-sample the cached panorama; re-bake only on a
//                           bucket-crossing turn / cell move / slow safety tick.
//   opaque + transparent  — near DDA cast, gated by a dirty-flag (idle frame ⇒ 0
//                           near rays).
//   emitters (every frame)— integrated by `EmitterPlugin`, then splatted here.
//   resolve               — fold the four layers into the `Framebuffer`.
//
// It owns the `CompositeStack`, `FarCache`, `TierClock` resources and sizes them
// to the `Framebuffer`, resizing when it resizes. A scene without a camera early-
// returns; an idle frame (pose/size unchanged, map unchanged, no animated near
// material, far still valid within its bucket) skips the near cast AND the far
// re-bake — only the (cheap) emitter splat + resolve run, so motion stays smooth.

/// Per-frame bookkeeping for the composite path: the pose the FAR ring was last
/// baked at (so the tier policy can decide a re-bake) and the inputs of the last
/// NEAR cast (so an idle frame skips it). Bare components sidestep the two-glam
/// clash between `bevy::prelude` and `re_world`'s `Vec3`.
#[derive(Resource)]
struct CompositeDirty {
    /// Frame index the far ring was last baked at.
    far_bake_frame: u64,
    /// Yaw the far ring was last baked at.
    far_bake_yaw: f32,
    /// Camera xy the far ring was last baked at.
    far_bake_x: f32,
    far_bake_y: f32,
    /// The last NEAR cast key — `None` forces the first cast.
    near_last: Option<NearKey>,
    /// Wall-clock time (seconds, from `Res<Time>`) of the last NEAR re-cast.
    /// Used to THROTTLE animation-only re-casts to [`ANIM_RECAST_DT`] so a slow
    /// surface pulse refreshes at ~30 Hz instead of every render frame while the
    /// camera/geometry is unchanged. STRUCTURAL re-casts (camera move / map change /
    /// resize) ignore this gate and re-cast immediately. `f32::NEG_INFINITY` forces
    /// the first cast (any elapsed time is past the threshold).
    last_near_recast_secs: f32,
    /// The yaw/pitch/size the far layer was last FILLED at. `fill_far_layer` is a
    /// pure function of (cached panorama, yaw, pitch, view size); when none of
    /// those changed since the last fill the far layer is already correct, so the
    /// fill is skipped on an idle frame. `None` (and a rebake/resize) forces a fill.
    far_fill_last: Option<FarFillKey>,
    /// Frame index the NEAR ring was last (re)baked at — drives the ring's slow
    /// safety re-bake so a baked animated near material eventually refreshes even while
    /// the pan path serves it (mirrors the far ring's lazy safety tick).
    near_ring_bake_frame: u64,
}

impl Default for CompositeDirty {
    fn default() -> Self {
        CompositeDirty {
            far_bake_frame: 0,
            far_bake_yaw: 0.0,
            far_bake_x: 0.0,
            far_bake_y: 0.0,
            near_last: None,
            // Forces the first frame's animation-only branch (if reached) to pass
            // the throttle gate; the very first frame is structural anyway
            // (`near_last == None`).
            last_near_recast_secs: f32::NEG_INFINITY,
            far_fill_last: None,
            near_ring_bake_frame: 0,
        }
    }
}

/// Slow safety re-bake of the NEAR ring (frames). Even while the rotation-pan path
/// serves the ring, re-bake it at least this often so a baked animated near material
/// (a wall pulse/shift) refreshes. Smaller than the far safety tick — the near scene
/// is the readable foreground — but still lazy enough that a static scene almost never
/// pays for it. ~1 s at 60 Hz.
const NEAR_RING_SAFETY_FRAMES: u64 = 60;

/// The inputs that, if unchanged, make a far-layer fill redundant. The far fill
/// radial-samples the cached panorama by yaw (with a pitch row-shift) into a
/// view-sized layer. A rebake or resize resets this (the panorama / size changed),
/// forcing a re-fill.
///
/// QUANTIZED (stability spec §1.2, finding S1). The key used to carry the RAW
/// `cam.yaw` / `cam.pitch`, so any sub-degree mouse nudge (`MOUSE_YAW_SENS` steps are
/// a small fraction of a column) re-ran `fill_far_layer` — re-lerping every far cell's
/// fg/bg by a few units and re-snapping its glyph at the `frac = 0.5` boundary. That
/// is whole-backdrop churn for a change nobody can see, and it defeated the present
/// layer's dirty-diff on frames where NOTHING else moved.
///
/// Now:
/// * `yaw_bucket` uses the SAME [`NEAR_YAW_BUCKET`] the near key uses, so the far
///   layer refills exactly on the turns where the near tier would also re-key —
///   the two tiers step together instead of one sliding under the other.
/// * `pitch_shift` is the [`crate::pitch_row_shift`] the fill actually applies, i.e.
///   ONE FAR-LAYER ROW of granularity. The fill's output depends on pitch ONLY
///   through that shift, so this is exact as well as coarse: two pitches with the
///   same row shift produce a byte-identical far layer.
///
/// The fill itself still runs at the LIVE pose — the key decides *whether* to fill,
/// never *what* to fill with — so the pan / `pitch_row_shift` path is unchanged at
/// every bucket boundary.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct FarFillKey {
    yaw_bucket: i32,
    pitch_shift: i32,
    cols: usize,
    rows: usize,
}

impl FarFillKey {
    #[inline]
    fn of(cam: &CameraPose, cols: usize, rows: usize) -> Self {
        FarFillKey {
            yaw_bucket: (cam.yaw / NEAR_YAW_BUCKET).round() as i32,
            pitch_shift: crate::pitch_row_shift(cam.pitch, rows),
            cols,
            rows,
        }
    }
}

/// Near-yaw bucket width (radians) for the cell-quantized near key. A turn smaller
/// than this reuses the persisted near layers (no re-cast); the far bucket +
/// emitter tier carry the visible smoothness. Finer than the FAR
/// [`ARC_BUCKET`](crate::tier::ARC_BUCKET)
/// (the near scene is the readable foreground), but still wide enough that a
/// sub-degree mouse jitter does NOT re-cast. ~2.8° = a 128-bucket ring.
pub const NEAR_YAW_BUCKET: f32 = std::f32::consts::TAU / 128.0;

/// Minimum wall-clock interval (seconds) between ANIMATION-ONLY near re-casts.
/// When the geometry/camera/map are all unchanged but a renderable near material
/// animates (e.g. level 01's circuit/matrix wall pulse), the near scene is re-cast
/// at most this often — ~30 Hz — instead of every render frame. The pulse is slow,
/// so 30 Hz is visually indistinguishable from 60 Hz while halving the dominant
/// idle cost (re-shading ~15k cells). FRAMERATE-INDEPENDENT: the gate compares
/// elapsed `Res<Time>` seconds, so a 144 Hz display throttles to the same ~30 Hz.
///
/// This gate ONLY applies to animation-only dirtiness. A STRUCTURAL change (camera
/// moved past its cell/bucket, map changed, resize) re-casts IMMEDIATELY — full
/// responsiveness, no stale geometry. Tunable in the 24–30 Hz band.
pub const ANIM_RECAST_DT: f32 = 1.0 / 30.0;

/// SELECTIVE-RE-SHADE worst-case guard. When an animation-only frame's animated
/// cells exceed this fraction of the whole grid, the selective per-cell re-shade is
/// NO cheaper than a full re-cast (it re-shades nearly every cell anyway, with extra
/// per-cell bookkeeping), so we fall back to the existing ~30 Hz THROTTLED full
/// re-cast — guaranteeing the all-animated, wall-in-your-face view is never worse
/// than today.
///
/// POLICY (animation-only frame; nothing structural changed):
/// * animated fraction ≤ this AND a valid G-buffer exists ⇒ SELECTIVE re-shade,
///   run EVERY frame (full 60 Hz smoothness) — it is cheap (only the animated cells,
///   no DDA, no shadow march, no subcell).
/// * animated fraction > this (or no valid G-buffer yet) ⇒ THROTTLED full re-cast
///   ([`near_should_recast`] at [`ANIM_RECAST_DT`], ~30 Hz) — the prior behaviour.
///
/// 0.6 is deliberately generous: a typical near view (a few animated wall runs amid
/// static floor/ceiling/brick) sits well under it, so the cheap path engages every
/// frame; only a view almost entirely filled by an animated surface trips the guard.
pub const ANIM_SELECTIVE_MAX_FRAC: f32 = 0.6;

/// Decide whether the NEAR layers must be re-cast this frame, splitting the reason
/// into STRUCTURAL (immediate) vs ANIMATION-ONLY (throttled). Pure so the policy is
/// unit-testable without a Bevy `App`.
///
/// - `structural` = camera moved past its cell/bucket key, the map changed, or the
///   view resized. Re-cast IMMEDIATELY (full responsiveness, no throttle).
/// - `anim_renderable` = a near material this map can render is animated.
/// - `now_secs` / `last_recast_secs` = wall-clock seconds (from `Res<Time>`) now and
///   at the last re-cast.
///
/// Returns `true` iff structural, OR (animation-only AND at least [`ANIM_RECAST_DT`]
/// seconds elapsed since the last re-cast).
#[inline]
fn near_should_recast(
    structural: bool,
    anim_renderable: bool,
    now_secs: f32,
    last_recast_secs: f32,
) -> bool {
    structural || (anim_renderable && (now_secs - last_recast_secs >= ANIM_RECAST_DT))
}

/// Near-pitch bucket width (radians). Same philosophy as [`NEAR_YAW_BUCKET`] for
/// the vertical look axis: a sub-bucket pitch nudge reuses the persisted layers.
pub const NEAR_PITCH_BUCKET: f32 = std::f32::consts::TAU / 128.0;

/// The inputs that, if unchanged, make a NEAR re-cast redundant. QUANTIZED to the
/// cell lattice (position floored to whole world cells) + a coarse yaw/pitch bucket
/// — the same cell-quantized-motion economy the far tier uses. So a sub-cell nudge
/// or a sub-bucket turn REUSES the persisted near layers (cache HIT), while
/// crossing a cell boundary or a yaw/pitch bucket re-casts (cache MISS). The
/// framebuffer size is exact (a resize must always re-cast).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct NearKey {
    /// Camera cell (floored world position) — sub-cell motion maps to the same key.
    cell_x: i32,
    cell_y: i32,
    /// Vertical bucket at ~0.1-cell granularity. Unlike x/y (floored to whole cells),
    /// the NEAR tier must RE-CAST across a jump/fall arc — the vertical eye shear
    /// moves the geometry continuously within a single cell, so a whole-cell `cell_z`
    /// would freeze the image mid-jump. ~0.1-cell buckets track the arc smoothly while
    /// still collapsing micro-jitter to a cache HIT. Standing height is a single
    /// stable bucket, so a grounded player still gets the dirty-flag economy.
    z_bucket: i32,
    /// Quantized yaw / pitch bucket indices (round to nearest bucket).
    yaw_bucket: i32,
    pitch_bucket: i32,
    /// Quantized FOV bucket (0.1° granularity over the effective preset + aspect +
    /// stretch). The cast projection is a function of the [`Fov`], so a FOV change
    /// MUST invalidate the near cache or a static-camera zoom would keep serving the
    /// old-FOV geometry (the FOV-zoom reveal would have nothing new to wipe to). The
    /// 0.1° granularity collapses float jitter to a cache HIT while catching any real
    /// preset change (`[`/`]`, the zoom toggle). The `--dump`/golden path never builds
    /// a `NearKey` (it calls `composite_dump_once`), so this is golden-neutral.
    fov_bucket: i32,
    cols: usize,
    rows: usize,
}

impl NearKey {
    fn of(cam: &CameraPose, fov: &Fov, cols: usize, rows: usize) -> Self {
        // Fold the three projection-affecting FOV fields into one quantized bucket so
        // any change to the effective field of view re-casts. 0.1° / 0.001-unit
        // granularity: real changes flip the bucket, float noise does not.
        let fov_q = (fov.h_preset_deg * 10.0).round() as i32
            ^ ((fov.cell_aspect * 1000.0).round() as i32).rotate_left(11)
            ^ ((fov.stretch * 1000.0).round() as i32).rotate_left(21);
        NearKey {
            cell_x: cam.pos.x.floor() as i32,
            cell_y: cam.pos.y.floor() as i32,
            z_bucket: (cam.pos.z * 10.0).round() as i32,
            yaw_bucket: (cam.yaw / NEAR_YAW_BUCKET).round() as i32,
            pitch_bucket: (cam.pitch / NEAR_PITCH_BUCKET).round() as i32,
            fov_bucket: fov_q,
            cols,
            rows,
        }
    }
}

/// The tiered-composite render plugin: the LIVE renderer that produces the
/// framebuffer. Adds [`EmitterPlugin`] (so the particle pool ticks), the composite
/// resources, and the [`composite_render_sys`] that bakes/samples the far layer,
/// casts the near layers on-dirty, splats the emitters, and resolves into the
/// [`Framebuffer`].
///
/// It installs the [`MaterialRegistry`] + [`Fov`] and keeps the registry synced to
/// the map (the `sync_registry` system below), so this plugin is self-contained.
pub struct CompositeRenderPlugin;

impl Plugin for CompositeRenderPlugin {
    fn build(&self, app: &mut App) {
        // Seed the render knobs ONCE from the `TS_*` env vars (dev convenience), then
        // insert as a resource. INSERT-IF-ABSENT so the game / a settings row can
        // override it before adding the plugin. This is the ONLY env read; the hot
        // path reads the resource, never the env.
        if !app.world().contains_resource::<crate::RenderConfig>() {
            app.insert_resource(crate::RenderConfig::from_env());
        }
        // Unified distance LOD config (near/mid/far tiers + hysteresis). INSERT-IF-ABSENT
        // so the game / a settings row / manual tuning can override the thresholds
        // before adding the plugin. The ONLY env read for LOD; the hot path reads the
        // resource (applied to the thread-local once per cast), never the env.
        if !app.world().contains_resource::<crate::LodConfig>() {
            app.insert_resource(crate::LodConfig::from_env());
        }
        // Runtime DRAW-DISTANCE (far-freeze cutoff + DDA step cap). INSERT-IF-ABSENT so
        // the game / a graphics-quality settings row can shorten / lengthen the reach
        // before adding the plugin. The default reproduces the historical consts
        // (`FAR_FREEZE_DIST` = 32, `MAX_DDA_STEPS` = 256), so a default build is
        // byte-identical; the hot path reads the thread-local applied once per cast.
        if !app.world().contains_resource::<crate::FarFreeze>() {
            app.insert_resource(crate::FarFreeze::default());
        }
        // Universal DEPTH-CONTRAST policy (far surfaces lose contrast / fade to backdrop).
        // INSERT-IF-ABSENT so the game / a settings row / manual tuning can override
        // it before adding the plugin. DEFAULT OFF (`TS_DEPTH` unset) ⇒ byte-identical;
        // the hot path reads the thread-local applied once per cast, never the env.
        if !app
            .world()
            .contains_resource::<crate::DepthContrastConfig>()
        {
            app.insert_resource(crate::DepthContrastConfig::from_env());
        }
        // STRUCTURE-as-LINES contour + colour-volume config. INSERT-IF-ABSENT; seeded
        // ONCE from `TS_CONTOUR*` / `TS_SLOPE*` here. Until 2026-09-02 the render sys
        // re-read the env every frame (stability spec §1.2 S4); now the sync sys
        // installs this resource into the thread-local `contour::active()` reads.
        if !app.world().contains_resource::<crate::ContourConfig>() {
            app.insert_resource(crate::ContourConfig::from_env());
        }
        app.init_resource::<MaterialRegistry>()
            .init_resource::<Fov>()
            .init_resource::<CompositeStackRes>()
            .init_resource::<FarCacheRes>()
            .init_resource::<NearRingRes>()
            .init_resource::<NearScratch>()
            .init_resource::<TierClock>()
            .init_resource::<CompositeDirty>()
            // STATE BUS (reactive-materials spec §2): the per-tag live-state map the
            // shade path reads. INSERT-IF-ABSENT and defaults to EMPTY, so with no
            // bridge (a level / a dump with no puzzle) every tag reads state 0 →
            // today's frame, byte-for-byte. The `re_game` bridge writes it from the
            // live `PuzzleState`; `composite_render_sys` publishes it to the cast.
            .init_resource::<re_core::CellStateMap>()
            // Local 2×2 sub-cell refinement is ON by default (the silhouette
            // anti-alias the design calls for). INSERT-IF-ABSENT so the game / a
            // settings row can disable it (perf tier) before adding the plugin.
            .init_resource::<SubCellConfig>()
            // FORCE-REDRAW override (defeat the idle dirty-flag for a transition window,
            // e.g. the FOV-zoom reveal). Default `force = false` ⇒ idle economy intact.
            .init_resource::<RenderForceDirty>()
            // Per-stage render timing sink (microseconds). INSERT-IF-ABSENT so the
            // resource always exists even if re_game forgot to register it; the
            // default lives in re_core. The overlay/present layer reads it.
            .init_resource::<FrameTimingMetrics>()
            .add_plugins(EmitterPlugin)
            // sync the registry first (map changes), then run the composite AFTER the
            // emitter tick so the splat sees this frame's integrated particles.
            .add_systems(
                PostUpdate,
                (composite_sync_registry_sys, composite_render_sys)
                    .chain()
                    .after(EmitterTickSet),
            );
    }
}

/// FORCE-REDRAW override for the idle dirty-flag (the "defeat the cache this frame"
/// hook).
///
/// The composite render path normally SKIPS the near cast and the rotation-pan on an
/// idle frame (pose/map/size/FOV unchanged) — the idle optimization. A few effects,
/// though, need a FRESH near cast EVERY frame even while the camera is static: the
/// FOV-zoom REVEAL (`re_game::zoom`) snaps the `Fov` and then wipes the old frame to
/// the new-FOV frame over a short transition; with a static camera the dirty-flag
/// (and a stale rotation-pan ring whose `ring_cols` didn't change) would otherwise
/// keep serving the OLD-FOV geometry, so the reveal would have nothing new to wipe to
/// and nothing visibly animates.
///
/// While [`force`](Self::force) is `true`, [`composite_render_sys`] treats the frame
/// as STRUCTURALLY dirty (re-casts the near layers at the live `Fov`) and DISABLES the
/// pan (a stale ring can't steal the frame). The setter owns the lifetime: drive it
/// from the effect's "is this transition active?" predicate each frame and clear it
/// when the transition ends. Default `false` ⇒ the idle economy is untouched, so a
/// static idle frame still caches exactly as before (golden-neutral, perf-neutral).
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RenderForceDirty {
    /// `true` ⇒ force a fresh near cast + present this frame (defeat the idle cache).
    pub force: bool,
}

/// Toggle for the local 2×2 sub-cell refinement pass ([`refine_subcells`]). Default
/// `enabled = true` (the silhouette anti-alias is on). A perf tier / settings row can
/// switch it off; the near opaque layer then keeps its one-ray-per-cell glyphs.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubCellConfig {
    /// Whether marked silhouette cells cast the four sub-rays + quadrant-blit.
    pub enabled: bool,
}

impl Default for SubCellConfig {
    fn default() -> Self {
        SubCellConfig { enabled: true }
    }
}

/// Owns the persistent [`CompositeStack`] (the four layers) as a resource so it is
/// allocated once and resized in place. A newtype because `CompositeStack` lives in
/// `re_core` and cannot derive `Resource` there without a terminal-free Bevy dep.
#[derive(Resource)]
struct CompositeStackRes(CompositeStack);

impl Default for CompositeStackRes {
    fn default() -> Self {
        // Allocated empty; `composite_render_sys` resizes to the framebuffer.
        CompositeStackRes(CompositeStack::new(0, 0))
    }
}

/// Owns the [`FarCache`] as a resource (it already derives `Resource`, but a
/// newtype keeps the height-resize bookkeeping here and away from `far_cache.rs`).
#[derive(Resource)]
struct FarCacheRes(FarCache);

impl Default for FarCacheRes {
    fn default() -> Self {
        FarCacheRes(FarCache::new(0))
    }
}

/// Owns the [`NearRing`] (rotation-pan near cache) as a resource. Allocated empty;
/// `composite_render_sys` sizes it to the framebuffer + FOV on the first frame.
#[derive(Resource)]
struct NearRingRes(NearRing);

impl Default for NearRingRes {
    fn default() -> Self {
        NearRingRes(NearRing::new(0, 0, 1.0))
    }
}

/// PERSISTENT scratch [`Framebuffer`] for the near cast (critic M3): allocated once
/// and reused across frames, so a moving frame does NOT allocate a ~75 KB
/// framebuffer on the hot path. Resized in place only on a real dimension change.
/// Also carries the PERSISTENT sub-cell marks buffer so the silhouette refinement
/// pass ([`refine_subcells_buf`]) reuses one allocation across frames instead of
/// heap-allocating a fresh `Vec` every re-cast (perf).
#[derive(Resource)]
struct NearScratch {
    fb: Framebuffer,
    /// Reused (`.clear()`ed each frame) silhouette marks buffer.
    marks: Vec<(usize, usize)>,
    /// PERSISTENT selective-re-shade G-buffer: the per-cell cached [`crate::shade_cell`]
    /// inputs + the animated-cell index, populated by a FULL near cast and replayed
    /// (animated cells only) on an animation-only frame. Generation-cleared in place
    /// per full cast — no per-frame allocation.
    gbuf: NearGBuffer,
    /// Reused column-index buffer for the special-cell (mirror/holo) recast on a
    /// PANNED frame: [`crate::recast_special_columns_into`] fills it with the columns
    /// it freshly cast so the composite can overwrite just those in the panned layers.
    /// Cleared + refilled each panned frame; capacity persists (no per-frame alloc).
    special_cols: Vec<usize>,
}

impl Default for NearScratch {
    fn default() -> Self {
        // Allocated empty; `build_world_layers_*_into` resizes to the stack dims.
        NearScratch {
            fb: Framebuffer::new(0, 0),
            marks: Vec::new(),
            gbuf: NearGBuffer::default(),
            special_cols: Vec::new(),
        }
    }
}

/// Keep the [`MaterialRegistry`] in sync with the map (rebuild only when the map
/// changes), so the composite plugin is self-contained.
fn composite_sync_registry_sys(
    map: Res<Map>,
    mut registry: ResMut<MaterialRegistry>,
    lod_cfg: Res<crate::LodConfig>,
    far_freeze: Res<crate::FarFreeze>,
    depth_cfg: Res<crate::DepthContrastConfig>,
    contour_cfg: Res<crate::ContourConfig>,
) {
    if map.is_changed() {
        *registry = registry_for_map(&map);
    }
    // Install the active CONTOUR config for this thread (read by the contour post-pass
    // in the chained render sys via `contour::active()`). A `Copy` store — no env read.
    contour_cfg.apply();
    // Install the runtime DRAW-DISTANCE bounds (far-freeze cutoff + DDA step cap) for
    // this thread BEFORE the chained `composite_render_sys` casts. Same pattern as the
    // LOD config below: applied here (not the render sys) to keep the render sys under
    // Bevy's system-param limit, and chained on the same thread so the thread-local is
    // live for the cast. Default = the historical consts ⇒ byte-identical.
    far_freeze.apply();
    // Install the active LOD config (near/mid/far tiers + hysteresis) for this thread
    // BEFORE the chained `composite_render_sys` casts this frame, so every shade +
    // shadow decision reads the SAME tier thresholds. This sys is `.chain()`-ed
    // directly before the render on the same thread, so the thread-local set here is
    // live for the cast. Explicit — no env read in the hot path; the resource was
    // seeded once at plugin build. (Applied here, not in the render sys, to keep the
    // render sys under Bevy's 16-param system limit.)
    lod_cfg.apply();
    // Install the active UNIVERSAL DEPTH-CONTRAST policy (far surfaces lose contrast /
    // dissolve into the backdrop, near surfaces keep full ink) for this thread BEFORE
    // the chained casts. DEFAULT OFF ⇒ the shade pass is a strict no-op (byte-identical
    // to today); `TS_DEPTH=1` (seeded at plugin build) turns it on. Same thread-local
    // pattern as LOD / far-freeze — explicit, no env read in the hot path.
    depth_cfg.apply();
}

/// The composite render system (spec §2/§3). Order of operations:
///
/// 1. tick the [`TierClock`];
/// 2. resize the stack + far cache to the framebuffer if it changed;
/// 3. FAR: if [`far_needs_rebake`] (yaw bucket crossed / ≥1 cell moved / slow timer)
///    re-bake the ring and record the bake pose; ALWAYS radial-sample it into the
///    far layer (cheap);
/// 4. NEAR: gated by a pose/size dirty-flag + the registry's `any_animated` —
///    re-cast the opaque + transparent layers, else keep them;
/// 5. EMITTERS: splat the (already-ticked) pool into the emitters layer, occluded by
///    the opaque layer;
/// 6. resolve the four layers into the [`Framebuffer`].
#[allow(clippy::too_many_arguments)]
fn composite_render_sys(
    map: Res<Map>,
    registry: Res<MaterialRegistry>,
    fov: Res<Fov>,
    subcell: Res<SubCellConfig>,
    // Read-only config resources tupled into one system param: Bevy caps a system's
    // param list at 16, and the state bus + force-redraw flag push us past the edge —
    // grouping the render knobs + the state map + the force-dirty flag keeps us under
    // the cap with no behaviour change.
    cfg: (
        Res<crate::RenderConfig>,
        Res<re_core::CellStateMap>,
        Res<RenderForceDirty>,
    ),
    pool: Res<ParticlePool>,
    time: Res<Time>,
    cam_q: Query<&CameraPose>,
    mut stack_res: ResMut<CompositeStackRes>,
    mut far_res: ResMut<FarCacheRes>,
    mut near_ring_res: ResMut<NearRingRes>,
    mut near_scratch: ResMut<NearScratch>,
    mut clock: ResMut<TierClock>,
    mut dirty: ResMut<CompositeDirty>,
    mut fb: ResMut<Framebuffer>,
    mut timing: ResMut<FrameTimingMetrics>,
) {
    let Ok(cam) = cam_q.single() else {
        return; // no camera: produce nothing.
    };
    // ROBUSTNESS: a non-finite (NaN/inf) pose would poison the quantized key, the
    // ring bake and every ray. Skip the frame cleanly (keep the last good frame).
    if !pose_is_finite(cam) {
        return;
    }

    // Install the active render knobs for this thread before ANY cast this frame
    // (far + near + sub-cell all run here on this thread). Explicit — no env read in
    // the hot path; the resource was seeded once at plugin build.
    let (render_cfg, state_map, force_dirty) = &cfg;
    render_cfg.apply();
    // Seed the NODE SPRITE on/off knob from `TS_NODE_SPRITE` for this frame's cast
    // (same "explicit thread-local, env read once" discipline as `render_cfg`).
    crate::config::apply_node_sprite_env();

    // STATE BUS (reactive-materials spec §2): publish the live per-tag state map to
    // this thread for the whole cast (far + near + sub-cell all run here). The shade
    // path reads it back by cell tag in `shade_cell`. EMPTY (no bridge) ⇒ every tag
    // reads state 0 → byte-identical to today. The RAII guard clears the bus on every
    // exit path (including the early returns below + a panic), so a later cast on this
    // thread without a map is never affected by a stale one.
    let _state_scope = crate::state_bus::CellStateScope::new(state_map);

    clock.tick();

    let (cols, rows) = (fb.cols(), fb.rows());
    if cols == 0 || rows == 0 {
        return;
    }
    // ROBUSTNESS: an empty / zero-volume map has no cells to cast; produce nothing.
    if map_is_empty(&map) {
        return;
    }

    // SKY: publish this frame's animation clock, camera ground position, player
    // yaw, and (if the level authors one) the COMPILED sky program for
    // `apply_sky` (`lib.rs`'s sky region) to read back via a thread-local —
    // the same "cast-scoped global" pattern `state_bus`/`config`/`draw_dist`
    // already use, since `apply_sky`'s callers sit outside this file. Compiled
    // ONCE per frame here (never per cell). A map with no `[sky]` program
    // installs `None`, so `apply_sky` falls through to the plain
    // `sky::map_sky_texel` path.
    {
        let sky_now = time.elapsed_secs();
        crate::sky::set_active_sky_time(sky_now);
        crate::sky::set_active_cam_xy((cam.pos.x, cam.pos.y));
        crate::sky::set_active_player_yaw(cam.yaw);
        crate::sky::set_active_program(map.sky_spec().map(crate::sky::SkyProgram::compile));
    }

    // Per-stage timing accumulators (microseconds). Each stage is CONDITIONAL; a
    // local left at 0.0 means "this stage did not run this frame", which is exactly
    // what the overlay should show. All five are written into the resource ONCE at
    // the end so a skipped stage truthfully reads 0.0 (no stale carry-over).
    // The NEAR cast and sub-cell stages are CONDITIONAL (gated by `near_dirty` /
    // `subcell.enabled`); they default to 0.0 = "did not run this frame". The far,
    // emitter and resolve stages run on every rendered frame, so they are bound at
    // their measurement point below (no skipped-default needed).
    let mut near_cast_us = 0.0f32;
    let mut subcell_us = 0.0f32;

    let stack = &mut stack_res.0;
    // Resize ONLY on a real dimension change. `Layer::resize` clears presence even
    // on equal dims, which would wipe the persisted NEAR layers on an idle frame; a
    // genuine resize forces a re-cast anyway (the near key's cols/rows changed), so a
    // resize-driven clear is always followed by a re-cast.
    let resized = stack.dims() != (cols, rows);
    if resized {
        stack.resize(cols, rows);
    }
    far_res.0.resize_rows(rows);
    // Size the near rotation-pan ring to the live view + FOV. The ring width derives
    // from the runtime hfov (+ margin), so a resize that changes the aspect re-derives
    // it (and invalidates the ring, forcing a fresh structural cast + rebake).
    let view = crate::view_params(cols, rows, &fov);
    near_ring_res.0.resize(cols, rows, view.hfov);

    // ── FAR tier (rare) ──
    // `dirty.far_bake_*` records the pose of the last COMPLETED (served) ring; the
    // re-bake gate measures drift from it. The bake itself is INCREMENTAL: it fills a
    // separate back ring ~one chunk/frame while the served ring keeps being sampled,
    // then swaps atomically on completion — so a re-bake never stalls the backdrop and
    // never serves a half-baked ring (no tear). See `FarCache::bake_step`.
    let dyaw = cam.yaw - dirty.far_bake_yaw;
    let dx = cam.pos.x - dirty.far_bake_x;
    let dy = cam.pos.y - dirty.far_bake_y;
    let moved_cells = (dx * dx + dy * dy).sqrt();
    let frames_since = clock.frame.wrapping_sub(dirty.far_bake_frame);
    let far_t = std::time::Instant::now();
    let mut rebaked = false;
    if !far_res.0.is_valid() {
        // FIRST-EVER bake (level load): cast the whole ring in ONE shot. Baking it
        // incrementally would show an empty/torn backdrop for the ~8 ramp-up frames
        // on load (ugly); the one-time spike is hidden by the load transition. After
        // this, every re-bake is incremental.
        far_res.0.rebake(&map, cam, &registry);
        dirty.far_bake_frame = clock.frame;
        dirty.far_bake_yaw = cam.yaw;
        dirty.far_bake_x = cam.pos.x;
        dirty.far_bake_y = cam.pos.y;
        rebaked = true;
    } else {
        // Steady state. Drive the incremental bake when the tier policy says the ring
        // is stale, OR keep an already-started bake running to completion.
        let in_progress = far_res.0.bake_in_progress();
        if far_needs_rebake(frames_since, dyaw, moved_cells) || in_progress {
            // Mid-bake-move policy: if a bake is already running and the camera has
            // moved SIGNIFICANTLY (out of yaw bucket or ≥1 cell) since the bake
            // STARTED, restart it from the new pose — never swap in a ring baked from
            // a stale viewpoint. A tiny move lets the current bake finish.
            let should_restart = if in_progress {
                let (start_yaw, start_pos) = far_res.0.bake_start_pose();
                let sdyaw = cam.yaw - start_yaw;
                let sdx = cam.pos.x - start_pos.x;
                let sdy = cam.pos.y - start_pos.y;
                let smoved = (sdx * sdx + sdy * sdy).sqrt();
                far_needs_rebake(0, sdyaw, smoved)
            } else {
                false
            };
            let completed = far_res.0.bake_step(&map, cam, &registry, should_restart);
            if completed {
                // The swapped-in ring records the pose its columns were cast FROM
                // (the bake's start pose), so the re-bake gate measures drift from
                // there — not from where the camera happens to be at swap time.
                let (start_yaw, start_pos) = (far_res.0.base_yaw, far_res.0.base_pos);
                dirty.far_bake_frame = clock.frame;
                dirty.far_bake_yaw = start_yaw;
                dirty.far_bake_x = start_pos.x;
                dirty.far_bake_y = start_pos.y;
                rebaked = true;
            }
        }
    }
    // Radial-sample the cached ring into the far layer — but ONLY when something
    // that affects the sample changed. `fill_far_layer` is a pure function of
    // (panorama, yaw, pitch, view size); if none of those moved since the last
    // fill the far layer is already correct, so an idle frame skips the fill
    // entirely (~0 µs instead of ~300 µs). A rebake or resize always re-fills.
    let fill_key = FarFillKey::of(cam, cols, rows);
    let far_fill_dirty =
        resized || rebaked || !far_res.0.is_valid() || dirty.far_fill_last != Some(fill_key);
    if far_fill_dirty {
        far_res.0.fill_far_layer(cam, &mut stack.far);
        dirty.far_fill_last = Some(fill_key);
    }
    let far_bake_us = far_t.elapsed().as_secs_f32() * 1.0e6;

    // ── NEAR tier (on-dirty) ──
    // Re-cast the opaque + transparent layers only when the pose/size changed, the
    // map changed, an animated near material is live, or a resize cleared them.
    // Otherwise the persisted layers from the previous frame are reused — 0 near
    // rays on a truly idle frame (spec §7). `build_world_layers_fov` clears + refills
    // both near layers, so no separate clear is needed when re-casting.
    let key = NearKey::of(cam, &fov, cols, rows);
    let now_secs = time.elapsed_secs();
    // STRUCTURAL dirty: the camera moved past its cell/bucket key, the map changed,
    // the view was resized, OR a FORCE-REDRAW is in effect (e.g. the FOV-zoom reveal
    // wants a fresh new-FOV frame every transition frame on a STATIC camera — see
    // [`RenderForceDirty`]). These change (or must refresh) the GEOMETRY on screen, so
    // they re-cast IMMEDIATELY — full responsiveness, never throttled, never stale.
    let structural_dirty =
        resized || dirty.near_last != Some(key) || map.is_changed() || force_dirty.force;
    // ASCII FX R&D — SKY: an animated sky program (spec §2.0 step 1's
    // "map-local sky animated flag") joins the existing material-driven
    // animation check, so a level with ONLY an animated sky (no animated
    // material) still keeps re-casting instead of freezing.
    let sky_animated = crate::sky::sky_is_animated(&map);
    let anim_renderable = crate::map_local_any_animated(&map, &registry) || sky_animated;
    // ANIMATION-ONLY dirty: nothing structural changed, but a renderable near
    // material animates (e.g. level 01's circuit/matrix wall pulse), so the shading
    // — not the geometry — drifts with time. MAP-LOCAL check (not registry-global
    // `any_animated`): the registry always carries the full animated CATALOG by name
    // for `preset` resolution, so the global flag is permanently true and would
    // re-cast every idle frame. `map_local_any_animated` counts only materials this
    // map can actually RENDER (surface slots always; node/door iff Object cells; holo
    // iff the map has holograms). Catalog name-entries never force the flag.
    //
    // THROTTLE: an animation-only re-cast is allowed only once every
    // [`ANIM_RECAST_DT`] seconds (~30 Hz, framerate-independent via `Res<Time>`);
    // between those it REUSES the persisted near layers like an idle cache hit. The
    // pulse is slow, so 30 Hz reads identically to 60 Hz while halving the dominant
    // idle cost. `now_secs - last >= dt` is robust to the first-frame
    // `NEG_INFINITY` and to a `Time` reset (a backwards jump just re-casts once).
    // Split the persistent near scratch into independent field borrows so the cast
    // (fb + gbuf) and the subcell-skip step (marks + gbuf) can borrow disjoint fields
    // at once (the struct holds them together purely to be one Bevy resource).
    let NearScratch {
        fb: near_fb,
        marks: near_marks,
        gbuf: near_gbuf,
        special_cols: near_special_cols,
    } = &mut *near_scratch;

    // SELECTIVE RE-SHADE eligibility (animation-only frame): nothing structural
    // changed, a renderable near material animates, AND a valid captured G-buffer
    // from the last full cast exists. The animated-cell fraction (vs the whole grid)
    // decides cheap-selective vs throttled-full (the worst-case guard).
    let grid_cells = (cols * rows).max(1);
    let anim_frac = near_gbuf.animated_len() as f32 / grid_cells as f32;
    // ASCII FX R&D — SKY: `reshade_animated_into` deliberately LEAVES every
    // sky-applied cell as persisted (it depends only on ray direction, not
    // time, in the pre-sky-program world) — see its own doc comment. With an
    // animated sky program that would freeze the sky on every selective-reshade
    // frame, so an animated sky routes through the (still throttled, ~30 Hz)
    // FULL re-cast below instead, which re-runs `apply_sky` fresh for every
    // sky cell.
    let selective_ok = !structural_dirty
        && anim_renderable
        && near_gbuf.is_valid()
        && anim_frac <= ANIM_SELECTIVE_MAX_FRAC
        && !sky_animated;

    // ── NEAR ROTATION-PAN (the rotation = pan win) ──
    // parallax ∝ 1/distance: a pure rotation has zero parallax, so the near scene can
    // be PANNED from a baked ring with NO re-cast. Eligibility (the correctness gate):
    //  * the position has not moved past the near parallax threshold AND the whole
    //    viewport stays inside the ring's baked arc at the panned yaw (`can_pan`);
    //  * the map did not change and the view did not resize;
    // SPECIAL CELLS (mirror/holo) are VIEW-DIRECTION dependent GEOMETRY and can NEVER
    // be panned (a mirror reflects a different scene as you turn; a holo silhouette
    // re-projects). NOTE: this is view-dependent *geometry*, not lighting — so the
    // `REVISION_VIEWINDEP_LIGHT` toggle (which drops view-dependent SHADING from the
    // static pan, making it byte-exact) does NOT shrink this set: mirrors and
    // holograms still recast here regardless. The set was already minimal.
    // They no longer disable the pan for the WHOLE frame: the (majority) static
    // geometry still pans, and only the (minority) special COLUMNS are re-cast fresh
    // at the live view + overwritten into the panned layers — see the pan branch's
    // `recast_special_columns_into` step. The result is EXACT (static panned, special
    // re-cast = byte-identical to a full cast for every cell).
    // When eligible we (a) keep the served ring fresh by re-baking it incrementally
    // when the camera has drifted toward the margin / a slow safety tick fires (the
    // back ring fills over several frames while the served ring keeps panning — no
    // tear/stall), and (b) FILL the near layers from the served ring (zero rays).
    let plane_scale = view.plane_scale;
    let has_special = crate::map_local_has_special_cells(&map, &registry);
    let near_ring = &mut near_ring_res.0;
    // A FORCE-REDRAW also disables the pan: a stale rotation-pan ring (baked at the
    // OLD FOV, and whose `ring_cols` may not have changed for a small FOV delta) must
    // not steal the frame and serve old-FOV geometry — fall through to the fresh
    // structural cast below so the new-FOV frame is actually produced.
    let pan_eligible =
        !resized && !map.is_changed() && !force_dirty.force && near_ring.can_pan(cam, plane_scale);

    let mut panned = false;
    if pan_eligible {
        let near_t = std::time::Instant::now();
        // Keep the ring fresh: re-bake (incrementally) when the camera has turned/moved
        // far enough that a future frame would run OFF the served ring, or the slow
        // safety tick fires (so a baked animated material refreshes). The served ring
        // keeps panning until the back ring completes + swaps.
        let in_progress = near_ring.bake_in_progress();
        let ring_safety =
            clock.frame.wrapping_sub(dirty.near_ring_bake_frame) >= NEAR_RING_SAFETY_FRAMES;
        // "Drifting toward the edge": within half the margin of the served arc edge, or
        // a measurable position move — start re-centring the ring on the current pose so
        // a continued rotation never runs OFF the served arc.
        let dyaw_edge = crate::near_cache_ang_delta(cam.yaw, near_ring.base_yaw).abs();
        let drift = dyaw_edge >= near_ring.margin() * 0.5
            || (cam.pos.x - near_ring.base_pos.x).abs() >= 0.25
            || (cam.pos.y - near_ring.base_pos.y).abs() >= 0.25;
        if in_progress || drift || ring_safety {
            // A bake started at yaw Y0 covers the WHOLE arc around Y0, so a pure rotation
            // within the margin must NOT restart it. Restart ONLY on a position move —
            // that genuinely invalidates the in-flight ring's parallax (shared helper).
            let should_restart =
                in_progress && near_ring_in_progress_restart_on_move(near_ring, cam);
            near_ring_prime_step(
                near_ring,
                &map,
                cam,
                &registry,
                &fov,
                now_secs,
                should_restart,
                &clock,
                &mut dirty,
            );
        }
        near_ring.fill_near_layers(cam, plane_scale, &mut stack.opaque, &mut stack.transparent);

        // ── SPECIAL-CELL RECAST (mirror / hologram) over the panned layers ──
        // The ring panned the STATIC geometry exactly, but mirrors reflect a
        // view-dependent scene and holograms re-project — neither is correct at the
        // live yaw when baked at `base_yaw`. So, only when the scene actually has
        // special cells, fresh-cast ONLY the columns the live view shows them in and
        // overwrite those columns in the panned layers. Result = EXACT (static panned,
        // special re-cast = byte-identical to a full cast). Special cells are a
        // minority, so this costs a fraction of a full cast — the rotation win.
        if has_special {
            if near_fb.cols() != cols || near_fb.rows() != rows {
                near_fb.resize(cols, rows);
            }
            near_fb.clear();
            crate::recast_special_columns_into(
                &map,
                cam,
                &registry,
                &fov,
                now_secs,
                near_fb,
                near_special_cols,
            );
            // Route each freshly-cast cell of every special column into the correct
            // near layer (opaque vs transparent by material transparency — the same
            // split the full cast uses), OVERWRITING the panned value. A special
            // column's PRESENT-cell set + per-cell routing is identical to the pan
            // (geometry is exact under pan; only the view-dependent VALUE differs), so
            // overwriting present cells is sufficient — no panned cell needs clearing.
            for &col in near_special_cols.iter() {
                for row in 0..rows {
                    let Some(cell) = near_fb.get(col, row).copied() else {
                        continue;
                    };
                    let mat_id = crate::surface_material(cell.surface, cell.material);
                    if registry.get(mat_id).is_transparent() {
                        stack.transparent.set(col, row, cell);
                    } else {
                        stack.opaque.set(col, row, cell);
                    }
                }
            }
        }
        near_cast_us = near_t.elapsed().as_secs_f32() * 1.0e6;
        // The pan re-built both near layers; the persisted G-buffer no longer matches
        // the on-screen geometry (different yaw), so invalidate the selective path: the
        // next genuine structural cast re-captures it. Keep `near_last`/the recast clock
        // unchanged (the pan is not a cast). Animation refreshes at the ring's rebake
        // cadence (the safety tick) — the lazy near-ring policy, mirroring far.
        near_gbuf.invalidate();
        panned = true;
    }

    if panned {
        // handled above
    } else if structural_dirty {
        // ── FULL (structural) near cast + capture the G-buffer ──
        // Geometry changed (camera/map/resize), so re-cast every column and re-build
        // the per-cell cache + animated index the selective path replays. Cast at the
        // live `now_secs` so the captured frame and later selective re-shades share
        // one clock.
        let near_t = std::time::Instant::now();
        build_world_layers_fov_capture_into(
            &map, cam, &registry, &fov, now_secs, near_fb, near_gbuf, stack,
        );
        near_cast_us = near_t.elapsed().as_secs_f32() * 1.0e6;
        // ── LOCAL 2×2 SUB-CELL (silhouette anti-alias) ──
        // Only after a real re-cast. Marked cells cast 4 sub-rays + a quadrant blit;
        // each marked cell is FLAGGED in the G-buffer so the selective re-shade never
        // overwrites its persisted blit (which a full re-cast would re-blit).
        if subcell.enabled {
            let sub_t = std::time::Instant::now();
            refine_subcells_buf(&map, cam, &registry, &fov, &mut stack.opaque, near_marks);
            for &(mc, mr) in near_marks.iter() {
                near_gbuf.mark_skip(mc, mr);
            }
            subcell_us = sub_t.elapsed().as_secs_f32() * 1.0e6;
        }
        // Keep the key in sync so a LATER camera move (different key) is still a
        // STRUCTURAL re-cast (immediate). Record the re-cast time so a later
        // THROTTLED full re-cast (the guard fallback) waits a full `ANIM_RECAST_DT`.
        dirty.near_last = Some(key);
        dirty.last_near_recast_secs = now_secs;
        // PRIME THE ROTATION-PAN RING at the new pose. One incremental chunk per
        // structural frame: the ring fills its back buffer over several frames while we
        // keep casting, then becomes pannable — so a stand-and-turn after a walk pans
        // for free. A move RESTARTS the in-progress bake at the new pose (the served
        // ring would be stale).
        //
        // The ring is primed EVEN when the scene has special cells: it bakes the static
        // (pannable) geometry, and a later rotation pans that ring + re-casts only the
        // special columns (see the pan branch). A special cell baked into the ring at a
        // stale yaw is harmlessly overwritten by the per-frame special-column recast.
        {
            let in_progress = near_ring.bake_in_progress();
            let should_restart = if in_progress {
                // Restart the in-flight ring only on a position move (a pure rotation is
                // covered by the bake's margin — see the pan branch). Shared helper.
                near_ring_in_progress_restart_on_move(near_ring, cam)
            } else {
                // No bake running: start one whenever the served ring is missing or no
                // longer covers this pose (so the next rotation can pan).
                !near_ring.can_pan(cam, plane_scale)
            };
            if in_progress || should_restart {
                near_ring_prime_step(
                    near_ring,
                    &map,
                    cam,
                    &registry,
                    &fov,
                    now_secs,
                    should_restart,
                    &clock,
                    &mut dirty,
                );
            }
        }
    } else if selective_ok {
        // ── SELECTIVE RE-SHADE (animation-only, low animated fraction) ──
        // Re-shade ONLY the animated, non-mirror, non-subcell cells from cached
        // geometry at the live time. EVERY frame (full 60 Hz smoothness): it is cheap
        // — no DDA, no shadow march, no subcell, only the animated cells. Static cells
        // (and subcell blits) stay exactly as the last full cast persisted them.
        let near_t = std::time::Instant::now();
        reshade_animated_into(
            &map,
            &registry,
            // The runtime draw-distance far-freeze, applied for this thread by
            // `composite_sync_registry_sys` from the live `FarFreeze` resource (default
            // = `FAR_FREEZE_DIST`, byte-identical to today).
            crate::draw_dist::active_far_freeze(),
            near_gbuf,
            now_secs,
            stack,
        );
        near_cast_us = near_t.elapsed().as_secs_f32() * 1.0e6;
        // The geometry/key is unchanged; `dirty.near_last` already equals `key`.
        // Do NOT advance `last_near_recast_secs` — the throttle gate governs only the
        // FULL-re-cast fallback below, not the (cheap, every-frame) selective path.
    } else if near_should_recast(
        // ── THROTTLED FULL re-cast (worst-case guard, or no valid G-buffer yet) ──
        // The animated fraction is too high for the selective path to pay off (or the
        // first animation-only frame before any capture). Fall back to the existing
        // ~30 Hz throttled full re-cast — never worse than today.
        structural_dirty,
        anim_renderable,
        now_secs,
        dirty.last_near_recast_secs,
    ) {
        let near_t = std::time::Instant::now();
        build_world_layers_fov_capture_into(
            &map, cam, &registry, &fov, now_secs, near_fb, near_gbuf, stack,
        );
        near_cast_us = near_t.elapsed().as_secs_f32() * 1.0e6;
        if subcell.enabled {
            let sub_t = std::time::Instant::now();
            refine_subcells_buf(&map, cam, &registry, &fov, &mut stack.opaque, near_marks);
            for &(mc, mr) in near_marks.iter() {
                near_gbuf.mark_skip(mc, mr);
            }
            subcell_us = sub_t.elapsed().as_secs_f32() * 1.0e6;
        }
        dirty.near_last = Some(key);
        dirty.last_near_recast_secs = now_secs;
    }

    // ── EMITTERS (every frame) ──
    // Splat the (already-ticked) pool; the opaque layer occludes STANDARD glyph
    // splats. Split borrow so all refer to the same stack. `splat_particles_kinds`
    // clears BOTH the emitters layer (standard glyphs) and the godrays layer
    // (colour-only shafts) and routes each particle by kind.
    let re_core::CompositeStack {
        opaque,
        emitters,
        godrays,
        ..
    } = &mut *stack;
    let emit_t = std::time::Instant::now();
    splat_particles_kinds(&pool, cam, &fov, opaque, emitters, godrays);
    let emitter_us = emit_t.elapsed().as_secs_f32() * 1.0e6;

    // ── resolve ──
    // Clear `fb` so cells uncovered by any layer (none here — far covers the frame
    // once baked) are reset; then fold the four layers in.
    let resolve_t = std::time::Instant::now();
    fb.clear();
    stack.resolve(&mut fb);
    // GLASS-MARKER (underline-only transparent) pass: a see-through cell that paints
    // nothing but a coloured-underline rule over the scene behind it. Runs after the
    // fold (the cell already shows what is behind) and before the contour pass. Reads
    // the near scratch (populated on the full-cast path; glass-marker materials are in
    // the no-pan special set so a frame that shows them is freshly cast). A scene with
    // no glass-marker material does nothing.
    apply_glass_markers(&registry, near_fb, &mut fb);
    // STRUCTURE-as-LINES + COLOR-as-VOLUME post-pass over the resolved frame: draw
    // silhouette/depth-edge box-drawing contours and the subtle per-face/depth
    // colour-volume shift. The live render uses the env-seeded shipping config (the
    // same knobs a manual tuning pass drives via the `--dump` path), so the interactive
    // game and a dump agree. No-op where no edge exists; off when disabled. The config
    // is the resource installed by `composite_sync_registry_sys` this frame (falls back
    // to the env read on a thread that never applied one — dumps, tests).
    crate::contour::apply(&mut fb, &crate::contour::active());
    // AUTHORED `[line]` overlay strokes: draw the
    // map's authored polylines as crisp box-drawing vector strokes RIGHT AFTER the
    // auto-contour (explicit lines layer over the auto silhouettes). Depends only on
    // (pose, overlay, projection) — no animation — so it rides the same cast dirty-flag
    // as contour: a static idle frame reuses `fb` and never re-enters this pass. A map
    // with no `overlay.lines` is a hard no-op ⇒ byte-identical to today.
    crate::lines::apply(&mut fb, &map, cam, &fov, &registry);
    // NODE SPRITE post-pass (RC1/RC2 fix, right after lines): draw every puzzle
    // node as a screen-space-bounded object at its projected centre. No-op when
    // the map has no sprite nodes or `TS_NODE_SPRITE=0`.
    crate::node_sprite::apply_persistent(&mut fb, &map, cam, &fov, now_secs);
    let resolve_us = resolve_t.elapsed().as_secs_f32() * 1.0e6;

    // Publish per-stage timing ONCE. Skipped stages keep their 0.0 init so the
    // overlay reads "didn't run this frame". `present_us`, `frame_dt_us`, `fps_ema`
    // and `frame_count` are owned by re_present and left untouched here.
    timing.far_bake_us = far_bake_us;
    timing.near_cast_us = near_cast_us;
    timing.subcell_us = subcell_us;
    timing.emitter_us = emitter_us;
    timing.resolve_us = resolve_us;
    timing.render_total_us = far_bake_us + near_cast_us + subcell_us + emitter_us + resolve_us;
}

/// Fixed timestep the dump's emitter warm-up ticks at — the same 30 Hz the live
/// `ANIM_RECAST_DT` throttle assumes elsewhere in this file.
const DUMP_EMITTER_TICK_DT: f32 = 1.0 / 30.0;

/// How many deterministic ticks [`composite_dump_once`] runs a fresh particle pool
/// through before splatting it into the still frame.
///
/// Was `24` (0.8s of simulated time) — enough for a dense, fast-lived effect (god
/// rays, bursts) to spread visibly, but for a SPARSE, long-lived ambient field
/// (`[effects] snow`, `SNOW_LIFETIME` 2.2s ± 0.7 spread) 0.8s is well short of the
/// lifetime, so at capture time the pool has only accumulated `rate · 0.8s` flakes
/// — none have died yet, so the live count is still climbing toward its steady
/// state of `rate · lifetime`, not there yet. This showed up as the
/// occupancy-aware lattice fix (`re_level::snow_emitters`) alone lifting
/// `L8-cold-room.lvl` from 11 to only 13 visible flake cells against a
/// 20-cell floor: the lattice fix stopped placing emitters inside solid
/// shelving, but the population every emitter had built up by tick 24 was
/// still capped well under equilibrium.
///
/// `90` ticks = 3.0s covers `SNOW_LIFETIME + SNOW_LIFETIME_SPREAD` (2.9s) with a
/// small margin, so a snow field is at its steady-state population — not still
/// ramping up — by the time the dump frame is captured. Denser/shorter-lived
/// effects (god rays, bursts) were already well-spread by tick 24 and are
/// unaffected in kind, only given a few more (harmless) seconds of the same drift.
/// This changes every emitter-bearing dump/golden frame's exact byte content
/// (more simulated time ⇒ different particle positions) — expected, and the
/// golden hashes are re-blessed separately from this fix.
const DUMP_EMITTER_WARMUP_TICKS: u32 = 90;

/// A single-pass composite for the `--dump` path: far re-bake + near cast + a few
/// emitter steps + resolve, into `fb`. No Bevy App / resources required — the dump
/// binary (re_game) calls this directly with the LEVEL's resolved emitters.
///
/// `emitters` is the level's emitter set (empty slice ⇒ no particles). The emitter
/// step runs a fixed number of deterministic ticks from a fresh pool (fixed seed)
/// so a dump shows particles in flight and is reproducible. Far is always re-baked
/// (a dump is a cold start). Mirrors the system's order so a dump matches a live
/// frame's structure.
///
/// Robustness: a 0-size framebuffer, an empty/zero-volume map, or a non-finite
/// camera pose early-return cleanly (no panic).
pub fn composite_dump_once(
    map: &Map,
    cam: &CameraPose,
    registry: &MaterialRegistry,
    fov: &Fov,
    emitters: &[Emitter],
    fb: &mut Framebuffer,
) {
    let (cols, rows) = (fb.cols(), fb.rows());
    if cols == 0 || rows == 0 || map_is_empty(map) || !pose_is_finite(cam) {
        // Nothing castable: leave the framebuffer cleared and return.
        fb.clear();
        return;
    }
    // SKY: install the frame-0 sky clock/camera/program, exactly mirroring the
    // live path's install so that at `t = 0` (goldens, `--dump`) the same
    // frame is produced every run. A map with no `[sky]` program installs
    // `None`, matching the plain `sky::map_sky_texel` path.
    crate::sky::set_active_sky_time(0.0);
    crate::sky::set_active_cam_xy((cam.pos.x, cam.pos.y));
    crate::sky::set_active_player_yaw(cam.yaw);
    crate::sky::set_active_program(map.sky_spec().map(crate::sky::SkyProgram::compile));
    // Apply the SAME render knobs the live `CompositeRenderPlugin` seeds (same env
    // seed), so a `--dump` golden frame is byte-identical to the live default render.
    crate::RenderConfig::from_env().apply();
    // Seed the NODE SPRITE on/off knob from `TS_NODE_SPRITE` too — BEFORE the cast,
    // since `is_sprite_node` (DDA/off-slice/shadow/holo) reads it back this frame.
    crate::config::apply_node_sprite_env();
    // Seed the LOD tiers from `TS_LOD_*` for the `--dump` / manual-tuning path too.
    crate::LodConfig::from_env().apply();
    // Seed the depth-contrast policy from `TS_DEPTH*` so a `--dump` reflects it
    // (DEFAULT OFF ⇒ byte-identical when `TS_DEPTH` is unset).
    crate::DepthContrastConfig::from_env().apply();
    let mut stack = CompositeStack::new(cols, rows);

    // FAR: cold bake + sample.
    let mut far = FarCache::new(rows);
    far.rebake(map, cam, registry);
    far.fill_far_layer(cam, &mut stack.far);

    // NEAR.
    build_world_layers_fov(map, cam, registry, fov, &mut stack);
    // LOCAL 2×2 SUB-CELL: refine silhouette edges in the opaque layer (the dump
    // shows the same anti-aliased near geometry a live frame does).
    refine_subcells(map, cam, registry, fov, &mut stack.opaque);

    // EMITTERS: a few deterministic ticks so particles have spread out and are
    // visible in the still frame (a single tick at t=0 leaves them at the origin).
    // An empty `emitters` slice spawns nothing — the dump shows clean geometry.
    if !emitters.is_empty() {
        let mut pool = ParticlePool::default();
        let mut rng = crate::emitter::ParticleRng::seed(0xD0E5_1A77);
        for _ in 0..DUMP_EMITTER_WARMUP_TICKS {
            crate::emitter::tick_emitters(emitters, &mut pool, DUMP_EMITTER_TICK_DT, &mut rng);
        }
        let re_core::CompositeStack {
            opaque,
            emitters,
            godrays,
            ..
        } = &mut stack;
        splat_particles_kinds(&pool, cam, fov, opaque, emitters, godrays);
    }

    // RESOLVE.
    fb.clear();
    stack.resolve(fb);
    // GLASS-MARKER pass: stamp the underline-only-transparent "glass" cells over the
    // resolved scene behind them (the dump shows the same see-through + underline a
    // live frame does). We re-build a one-shot scratch by re-casting, since the dump's
    // `build_world_layers_fov` used its own internal scratch we don't keep here.
    {
        let mut marker_scratch = Framebuffer::new(cols, rows);
        crate::raycast_with_time_fov_gameplay(map, cam, registry, fov, 0.0, &mut marker_scratch);
        apply_glass_markers(registry, &marker_scratch, fb);
    }
    // STRUCTURE-as-LINES + COLOR-as-VOLUME post-pass (same as the live render path),
    // so a `--dump` shows the shipping contour + colour-volume look.
    crate::contour::apply(fb, &crate::ContourConfig::from_env());
    // AUTHORED `[line]` overlay strokes (Scalable Material Overlay §6 P2), same hook as
    // the live path: crisp box-drawing vector strokes over the auto-contour. No-op when
    // `map.overlay().lines` is empty ⇒ a `--dump` of any existing level is byte-identical.
    crate::lines::apply(fb, map, cam, fov, registry);
    // NODE SPRITE post-pass, same hook as the live path. `time = 0.0`: the dump is
    // a cold, deterministic single frame (mirrors `raycast_fov`'s `0.0`).
    crate::node_sprite::apply(fb, map, cam, fov, 0.0);
}

/// Convenience for the NO-LEVEL test / preview path: dump with the built-in
/// [`demo_emitters`](crate::emitter::demo_emitters) instead of a level's resolved
/// emitter set. Real level dumps call [`composite_dump_once`] with the level's
/// emitters.
pub fn composite_dump_once_demo(
    map: &Map,
    cam: &CameraPose,
    registry: &MaterialRegistry,
    fov: &Fov,
    fb: &mut Framebuffer,
) {
    let emitters = crate::emitter::demo_emitters();
    composite_dump_once(map, cam, registry, fov, &emitters, fb);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry_for_map;

    #[test]
    fn far_fill_key_ignores_sub_bucket_yaw_and_pitch_jitter() {
        // Stability spec §1.2 / finding S1: the far fill key used to carry the RAW
        // yaw, so a mouse-look nudge far below one near-yaw bucket re-ran
        // `fill_far_layer` over the whole backdrop. It must now be a cache HIT.
        let base = CameraPose {
            pos: glam::Vec3::new(2.5, 2.5, 0.5),
            yaw: 0.7,
            pitch: 0.02,
        };
        let k0 = FarFillKey::of(&base, 100, 32);
        // A nudge of a tenth of a bucket (well under one mouse-look step's visible
        // effect) and a pitch nudge inside the same far-layer ROW.
        let jitter = CameraPose {
            yaw: base.yaw + NEAR_YAW_BUCKET * 0.1,
            pitch: base.pitch + 0.001,
            ..base
        };
        assert_eq!(
            k0,
            FarFillKey::of(&jitter, 100, 32),
            "sub-bucket jitter must HIT"
        );
        // A full bucket turn re-keys — the far layer follows a real turn.
        let turned = CameraPose {
            yaw: base.yaw + NEAR_YAW_BUCKET,
            ..base
        };
        assert_ne!(
            k0,
            FarFillKey::of(&turned, 100, 32),
            "a full bucket must re-fill"
        );
        // A resize always re-fills.
        assert_ne!(k0, FarFillKey::of(&base, 100, 33), "a resize must re-fill");
    }

    #[test]
    fn far_fill_key_re_keys_on_every_pitch_row() {
        // Pitch is bucketed to the row shift the fill actually applies, so the far
        // horizon still tracks pitch one row at a time (no vertical staleness).
        let rows = 32usize;
        let cam = |pitch: f32| CameraPose {
            pos: glam::Vec3::ZERO,
            yaw: 0.0,
            pitch,
        };
        let mut seen_shifts = std::collections::BTreeSet::new();
        for i in 0..=40 {
            let pitch = -0.6 + i as f32 * 0.03;
            let k = FarFillKey::of(&cam(pitch), 100, rows);
            assert_eq!(
                k.pitch_shift,
                crate::pitch_row_shift(pitch, rows),
                "the key's pitch bucket IS the fill's row shift"
            );
            seen_shifts.insert(k.pitch_shift);
        }
        assert!(
            seen_shifts.len() > 4,
            "a pitch sweep must step through several row buckets, saw {seen_shifts:?}"
        );
    }

    #[test]
    fn near_key_changes_with_fov_on_a_static_camera() {
        // THE CACHE-INVALIDATION INVARIANT behind the FOV-zoom reveal fix: with a
        // STATIC camera, a FOV preset change MUST produce a different near key — else
        // the composite would cache the OLD-FOV geometry and the reveal would have
        // nothing new to wipe to. (Before the fix `NearKey` ignored the FOV entirely.)
        let cam = CameraPose::spawn_in_test_room();
        let base = Fov::default(); // 65°
        let zoomed = Fov {
            h_preset_deg: 30.0,
            ..Fov::default()
        };
        let k_base = NearKey::of(&cam, &base, 80, 40);
        let k_zoom = NearKey::of(&cam, &zoomed, 80, 40);
        assert_ne!(
            k_base, k_zoom,
            "a FOV change on a static camera must change the near key (re-cast)"
        );
        // And the SAME FOV at the SAME pose must still HIT (idle economy preserved):
        // identical inputs ⇒ identical key.
        let k_base2 = NearKey::of(&cam, &base, 80, 40);
        assert_eq!(
            k_base, k_base2,
            "unchanged FOV + pose must keep the cache key (idle hit)"
        );
    }

    #[test]
    fn build_splits_into_opaque_and_clears_far_emitters() {
        // A real test room: build the world layers and assert the opaque layer is
        // populated, the far + emitters layers are untouched (absent), and every
        // populated opaque cell is genuinely non-transparent.
        let map = re_world::test_room();
        let cam = CameraPose::spawn_in_test_room();
        let registry = registry_for_map(&map);

        let (cols, rows) = (40usize, 24usize);
        let mut stack = CompositeStack::new(cols, rows);
        // Pre-seed far + emitters so we can prove build_world_layers leaves them.
        stack.far.set(0, 0, re_core::Cell::default());
        stack.emitters.set(0, 0, re_core::Cell::default());

        build_world_layers(&map, &cam, &registry, &mut stack);

        // The near cast must have populated SOME opaque cells (walls/floor/ceil).
        let mut opaque_count = 0usize;
        for row in 0..rows {
            for col in 0..cols {
                if stack.opaque.is_present(col, row) {
                    opaque_count += 1;
                }
            }
        }
        assert!(opaque_count > 0, "near cast produced no opaque cells");

        // far + emitters are NOT this module's job — left as we set them.
        assert!(stack.far.is_present(0, 0));
        assert!(stack.emitters.is_present(0, 0));

        // No cell should be present in BOTH opaque and transparent at once.
        for row in 0..rows {
            for col in 0..cols {
                let in_op = stack.opaque.is_present(col, row);
                let in_tr = stack.transparent.is_present(col, row);
                assert!(!(in_op && in_tr), "a cell landed in both layers");
            }
        }
    }

    #[test]
    fn composite_dump_once_produces_a_nonempty_frame() {
        // The full single-pass composite on the test room must resolve a non-empty
        // framebuffer (far backdrop + near geometry + emitter splats).
        let map = re_world::test_room();
        let cam = CameraPose::spawn_in_test_room();
        let registry = registry_for_map(&map);
        let fov = Fov::default();

        let (cols, rows) = (60usize, 30usize);
        let mut fb = Framebuffer::new(cols, rows);
        composite_dump_once_demo(&map, &cam, &registry, &fov, &mut fb);

        // A resolved frame must carry non-default cells (real geometry shaded in).
        let mut non_default = 0usize;
        for row in 0..rows {
            for col in 0..cols {
                if let Some(c) = fb.get(col, row) {
                    if *c != re_core::Cell::default() {
                        non_default += 1;
                    }
                }
            }
        }
        assert!(
            non_default > 0,
            "composite_dump_once produced an empty framebuffer"
        );
    }

    #[test]
    fn godrays_emitter_tints_a_dump_frame_keeping_glyphs() {
        // A GodRays emitter in the test room must, through the FULL dump producer,
        // TINT at least one cell warm while KEEPING its (non-space) glyph — the
        // colour-only-shaft contract, end-to-end (split into the godrays layer +
        // ColorMix fold). We compare the same frame with vs. without the emitter so
        // the difference is provably the shaft, not the scene.
        let map = re_world::test_room();
        let cam = CameraPose::spawn_in_test_room();
        let registry = registry_for_map(&map);
        let fov = Fov::default();
        let (cols, rows) = (60usize, 30usize);

        // Baseline: no emitters.
        let mut base = Framebuffer::new(cols, rows);
        composite_dump_once(&map, &cam, &registry, &fov, &[], &mut base);

        // A wide, near, slow god-rays cloud filling the view so it surely overlaps
        // some glyph cells regardless of the test room's exact geometry.
        let (dir_x, dir_y) = cam.dir();
        let ahead = cam.pos + glam::Vec3::new(dir_x * 3.0, dir_y * 3.0, 0.0);
        let em = Emitter {
            spread_rad: 80.0_f32.to_radians(),
            speed: 0.6,
            rate: 200.0,
            lifetime: 4.0,
            ..crate::emitter::Emitter::god_rays(ahead, glam::Vec3::new(0.0, 0.0, -1.0))
        };
        let mut lit = Framebuffer::new(cols, rows);
        composite_dump_once(&map, &cam, &registry, &fov, &[em], &mut lit);

        // Find a cell that DIFFERS and whose glyph is unchanged + non-space, with the
        // fg pushed warmer (more red relative to blue) than the baseline.
        let warm_kept = (0..rows)
            .flat_map(|r| (0..cols).map(move |c| (c, r)))
            .any(|(c, r)| {
                let (Some(b), Some(l)) = (base.get(c, r), lit.get(c, r)) else {
                    return false;
                };
                if b.glyph != l.glyph || l.glyph == u32::from(' ') || l.glyph == 0 {
                    return false;
                }
                let warmer = |x: u32| {
                    let r = ((x >> 16) & 0xFF) as i32;
                    let bl = (x & 0xFF) as i32;
                    r - bl
                };
                l != b && warmer(l.fg) > warmer(b.fg)
            });
        assert!(
            warm_kept,
            "a god-rays shaft must tint a cell warmer while keeping its glyph"
        );
    }

    #[test]
    fn subcell_overhead_is_small_not_every_cell() {
        // The local 2×2 refinement must mark only the SILHOUETTE fraction of cells,
        // never the whole frame — the economy invariant (overhead ≪ 4×). On the test
        // room the marked cells are the pillar/door/wall-floor edges, a small
        // minority of the populated opaque cells.
        let map = re_world::test_room();
        let cam = CameraPose::spawn_in_test_room();
        let registry = registry_for_map(&map);
        let fov = Fov::default();

        let (cols, rows) = (80usize, 40usize);
        let mut stack = CompositeStack::new(cols, rows);
        build_world_layers_fov(&map, &cam, &registry, &fov, &mut stack);

        // Count populated opaque cells BEFORE refinement.
        let mut populated = 0usize;
        for row in 0..rows {
            for col in 0..cols {
                if stack.opaque.is_present(col, row) {
                    populated += 1;
                }
            }
        }
        assert!(populated > 0, "near cast produced no opaque cells");

        let marked = refine_subcells(&map, &cam, &registry, &fov, &mut stack.opaque);
        // The marked fraction must be a clear MINORITY — far from "all cells". A
        // generous ceiling (45%) still proves it is not a blanket supersample; in
        // practice it is much smaller. It must also be > 0 (real edges exist).
        assert!(marked > 0, "expected some silhouette cells to be marked");
        assert!(
            (marked as f32) < 0.45 * populated as f32,
            "marked {marked} of {populated} opaque cells — overhead too high (should be a minority)"
        );
    }

    #[test]
    fn subcell_refinement_writes_quadrant_glyphs() {
        // After refinement, the marked cells carry one of the 16 quadrant glyphs.
        use crate::quadrant::QUADRANT_GLYPHS;
        let map = re_world::test_room();
        let cam = CameraPose::spawn_in_test_room();
        let registry = registry_for_map(&map);
        let fov = Fov::default();

        let (cols, rows) = (80usize, 40usize);
        let mut stack = CompositeStack::new(cols, rows);
        build_world_layers_fov(&map, &cam, &registry, &fov, &mut stack);
        let marked = refine_subcells(&map, &cam, &registry, &fov, &mut stack.opaque);
        assert!(marked > 0);

        let quad: std::collections::HashSet<u32> =
            QUADRANT_GLYPHS.iter().map(|c| u32::from(*c)).collect();
        let mut saw_quadrant = false;
        for row in 0..rows {
            for col in 0..cols {
                if let Some(c) = stack.opaque.get(col, row) {
                    if quad.contains(&c.glyph) && c.glyph != u32::from(' ') {
                        saw_quadrant = true;
                    }
                }
            }
        }
        assert!(
            saw_quadrant,
            "refined cells must carry a quadrant block glyph"
        );
    }

    #[test]
    fn subcell_refinement_is_deterministic() {
        let map = re_world::test_room();
        let cam = CameraPose::spawn_in_test_room();
        let registry = registry_for_map(&map);
        let fov = Fov::default();
        let render = || {
            let mut stack = CompositeStack::new(64, 32);
            build_world_layers_fov(&map, &cam, &registry, &fov, &mut stack);
            refine_subcells(&map, &cam, &registry, &fov, &mut stack.opaque);
            (0..32)
                .flat_map(|r| (0..64).map(move |c| (c, r)))
                .map(|(c, r)| stack.opaque.get(c, r).copied())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            render(),
            render(),
            "subcell refinement must be deterministic"
        );
    }

    #[test]
    fn composite_dump_once_is_deterministic() {
        // Same scene + camera → byte-identical resolved framebuffer (golden-safe).
        let map = re_world::test_room();
        let cam = CameraPose::spawn_in_test_room();
        let registry = registry_for_map(&map);
        let fov = Fov::default();

        let render = || {
            let mut fb = Framebuffer::new(48, 24);
            composite_dump_once_demo(&map, &cam, &registry, &fov, &mut fb);
            (0..24)
                .flat_map(|r| (0..48).map(move |c| (c, r)))
                .map(|(c, r)| fb.get(c, r).copied().unwrap_or_default())
                .collect::<Vec<_>>()
        };
        assert_eq!(render(), render(), "composite dump must be deterministic");
    }

    // ── animation-only near re-cast throttle (~30 Hz) ──

    #[test]
    fn structural_dirty_always_recasts_immediately() {
        // A structural change (camera/map/resize) re-casts NOW regardless of time
        // since the last re-cast — even 0 seconds, even with no animation.
        assert!(near_should_recast(true, false, 0.0, 0.0));
        assert!(near_should_recast(true, true, 100.0, 100.0));
        // Even if the throttle interval has NOT elapsed, structural wins.
        assert!(near_should_recast(true, true, 100.001, 100.0));
    }

    #[test]
    fn animation_only_is_throttled_to_anim_recast_dt() {
        // Not structural, animation renderable: re-cast only once the interval passed.
        let last = 10.0;
        // Just before the interval: REUSE (no re-cast).
        assert!(!near_should_recast(
            false,
            true,
            last + ANIM_RECAST_DT * 0.5,
            last
        ));
        assert!(!near_should_recast(
            false,
            true,
            last + ANIM_RECAST_DT - 0.001,
            last
        ));
        // At / past the interval: re-cast.
        assert!(near_should_recast(false, true, last + ANIM_RECAST_DT, last));
        assert!(near_should_recast(
            false,
            true,
            last + ANIM_RECAST_DT * 2.0,
            last
        ));
    }

    #[test]
    fn no_animation_no_structural_never_recasts() {
        // A truly idle frame (static surfaces, camera fixed): never re-cast no matter
        // how much time passes — the pure idle-cache hit.
        assert!(!near_should_recast(false, false, 0.0, f32::NEG_INFINITY));
        assert!(!near_should_recast(false, false, 1000.0, 0.0));
    }

    #[test]
    fn first_frame_neg_infinity_passes_throttle() {
        // The seeded `last_near_recast_secs = NEG_INFINITY` makes any elapsed time
        // exceed the threshold, so an animation-only first frame casts once.
        assert!(near_should_recast(false, true, 0.0, f32::NEG_INFINITY));
    }

    // ── H1: cell-quantized near key (idle-economy) ──

    fn pose(x: f32, y: f32, yaw: f32, pitch: f32) -> CameraPose {
        CameraPose {
            pos: glam::Vec3::new(x, y, 1.5),
            yaw,
            pitch,
        }
    }

    #[test]
    fn near_key_reuses_on_subcell_nudge_recasts_on_cell_move() {
        let fov = Fov::default();
        let base = pose(3.5, 3.5, 0.0, 0.0);
        let k0 = NearKey::of(&base, &fov, 80, 40);

        // A sub-cell nudge (same cell, same yaw/pitch bucket) → SAME key (cache HIT:
        // the near layers are reused, 0 rays).
        let nudged = pose(3.5 + 0.001, 3.5 - 0.002, 0.0, 0.0);
        assert_eq!(
            k0,
            NearKey::of(&nudged, &fov, 80, 40),
            "a sub-cell nudge must NOT invalidate the near cache"
        );

        // Crossing a cell boundary → DIFFERENT key (cache MISS: re-cast).
        let moved = pose(4.5, 3.5, 0.0, 0.0);
        assert_ne!(
            k0,
            NearKey::of(&moved, &fov, 80, 40),
            "crossing a cell boundary must re-cast the near scene"
        );
    }

    #[test]
    fn near_key_reuses_on_subbucket_turn_recasts_past_bucket() {
        let fov = Fov::default();
        let base = pose(3.5, 3.5, 0.0, 0.0);
        let k0 = NearKey::of(&base, &fov, 80, 40);

        // A turn smaller than the near yaw bucket → SAME key (reuse).
        let jitter = pose(3.5, 3.5, NEAR_YAW_BUCKET * 0.25, 0.0);
        assert_eq!(
            k0,
            NearKey::of(&jitter, &fov, 80, 40),
            "a sub-bucket turn must NOT invalidate the near cache"
        );

        // A turn well past a bucket → DIFFERENT key (re-cast).
        let turned = pose(3.5, 3.5, NEAR_YAW_BUCKET * 2.0, 0.0);
        assert_ne!(
            k0,
            NearKey::of(&turned, &fov, 80, 40),
            "a turn past the yaw bucket must re-cast"
        );

        // A pitch nudge within a bucket reuses; past a bucket re-casts.
        let pitch_jit = pose(3.5, 3.5, 0.0, NEAR_PITCH_BUCKET * 0.25);
        assert_eq!(k0, NearKey::of(&pitch_jit, &fov, 80, 40));
        let pitched = pose(3.5, 3.5, 0.0, NEAR_PITCH_BUCKET * 2.0);
        assert_ne!(k0, NearKey::of(&pitched, &fov, 80, 40));
    }

    #[test]
    fn near_key_recasts_on_resize() {
        let fov = Fov::default();
        let p = pose(3.5, 3.5, 0.0, 0.0);
        assert_ne!(
            NearKey::of(&p, &fov, 80, 40),
            NearKey::of(&p, &fov, 80, 41),
            "a resize must change the key (the layers must be re-cast)"
        );
    }

    // ── M3: persistent scratch reuse (no per-frame alloc on the near path) ──

    #[test]
    fn build_into_reuses_scratch_and_matches_oneshot() {
        // The persistent-scratch path must produce the SAME layers as the one-shot
        // allocating path — proving the scratch reuse is purely an allocation win,
        // not a behaviour change.
        let map = re_world::test_room();
        let cam = CameraPose::spawn_in_test_room();
        let registry = registry_for_map(&map);
        let fov = Fov::default();
        let (cols, rows) = (64usize, 32usize);

        let mut oneshot = CompositeStack::new(cols, rows);
        build_world_layers_fov(&map, &cam, &registry, &fov, &mut oneshot);

        // Reuse ONE scratch across two casts; the result must equal the one-shot.
        let mut scratch = Framebuffer::new(cols, rows);
        let mut reused = CompositeStack::new(cols, rows);
        build_world_layers_fov_into(&map, &cam, &registry, &fov, &mut scratch, &mut reused);
        // Second cast into the SAME scratch (the hot-path scenario): still correct.
        build_world_layers_fov_into(&map, &cam, &registry, &fov, &mut scratch, &mut reused);

        for row in 0..rows {
            for col in 0..cols {
                assert_eq!(
                    oneshot.opaque.get(col, row).copied(),
                    reused.opaque.get(col, row).copied(),
                    "scratch-reuse opaque differs at ({col},{row})"
                );
                assert_eq!(
                    oneshot.transparent.get(col, row).copied(),
                    reused.transparent.get(col, row).copied(),
                    "scratch-reuse transparent differs at ({col},{row})"
                );
            }
        }
    }

    // ── ROBUSTNESS: degenerate inputs early-return cleanly (no panic) ──

    #[test]
    fn dump_handles_zero_size_framebuffer() {
        let map = re_world::test_room();
        let cam = CameraPose::spawn_in_test_room();
        let registry = registry_for_map(&map);
        let fov = Fov::default();
        let mut fb = Framebuffer::new(0, 0);
        // Must not panic.
        composite_dump_once(&map, &cam, &registry, &fov, &[], &mut fb);
        assert_eq!((fb.cols(), fb.rows()), (0, 0));
    }

    #[test]
    fn dump_handles_empty_map() {
        let map = re_world::Map::new(0, 0, 0);
        let cam = CameraPose::spawn_in_test_room();
        let registry = registry_for_map(&map);
        let fov = Fov::default();
        let mut fb = Framebuffer::new(40, 20);
        composite_dump_once(&map, &cam, &registry, &fov, &[], &mut fb);
        // Empty map → no geometry; every cell stays default (cleared).
        for row in 0..fb.rows() {
            for col in 0..fb.cols() {
                assert_eq!(fb.get(col, row).copied(), Some(re_core::Cell::default()));
            }
        }
    }

    #[test]
    fn dump_handles_non_finite_pose() {
        let map = re_world::test_room();
        let registry = registry_for_map(&map);
        let fov = Fov::default();
        let cam = CameraPose {
            pos: glam::Vec3::new(f32::NAN, 3.5, 1.5),
            yaw: 0.0,
            pitch: 0.0,
        };
        let mut fb = Framebuffer::new(40, 20);
        // Must not panic; produces a cleared frame.
        composite_dump_once(&map, &cam, &registry, &fov, &[], &mut fb);
        for row in 0..fb.rows() {
            for col in 0..fb.cols() {
                assert_eq!(fb.get(col, row).copied(), Some(re_core::Cell::default()));
            }
        }
    }

    #[test]
    fn dump_with_empty_emitters_still_renders_geometry() {
        // Passing an empty emitter slice (an unauthored level) must still resolve the
        // far + near geometry — only the particles are absent.
        let map = re_world::test_room();
        let cam = CameraPose::spawn_in_test_room();
        let registry = registry_for_map(&map);
        let fov = Fov::default();
        let mut fb = Framebuffer::new(60, 30);
        composite_dump_once(&map, &cam, &registry, &fov, &[], &mut fb);
        let non_default = (0..fb.rows())
            .flat_map(|r| (0..fb.cols()).map(move |c| (c, r)))
            .filter(|&(c, r)| fb.get(c, r).copied() != Some(re_core::Cell::default()))
            .count();
        assert!(
            non_default > 0,
            "geometry must render even with no emitters"
        );
    }

    #[test]
    fn glass_marker_keeps_scene_behind_and_stamps_underline() {
        use crate::ascii_shader::{mat_pane_for_test, MaterialRegistry, MaterialSpec};
        use re_core::{Cell, SurfaceKind};

        // A registry whose slot 0 (surface dispatch for WallSolid) is the glass-marker
        // pane. The scratch carries the pane cell (a coloured underline in its style);
        // the resolved frame already shows the scene BEHIND (a red wall). After the
        // glass-marker pass the resolved cell must keep the red wall's glyph/fg/bg and
        // gain ONLY the pane's underline.
        let pane = mat_pane_for_test();
        let reg = MaterialRegistry::from_specs(&[MaterialSpec {
            name: "pane",
            material: pane,
        }]);

        // Scratch: a single pane cell, surface WallSolid (→ slot 0), carrying the
        // glass-marker underline in its style (as cell_style_full would stamp).
        let mut scratch = Framebuffer::new(1, 1);
        let mut pane_cell = Cell {
            glyph: u32::from('|'),
            fg: 0x0080_C8FF,
            bg: 0x0000_0000,
            depth: 2.0,
            surface: SurfaceKind::WallSolid,
            material: 0,
            ..Cell::default()
        };
        pane_cell.style = re_core::CellStyle::NONE.with_underline_color(0x0080_C8FF);
        scratch.set(0, 0, pane_cell);

        // Resolved frame: the scene BEHIND the pane — a red wall.
        let mut out = Framebuffer::new(1, 1);
        let behind = Cell {
            glyph: u32::from('#'),
            fg: 0x00FF_0000,
            bg: 0x0040_0000,
            depth: 9.0,
            surface: SurfaceKind::WallSolid,
            ..Cell::default()
        };
        out.set(0, 0, behind);

        apply_glass_markers(&reg, &scratch, &mut out);

        let c = out.get(0, 0).unwrap();
        // Glyph / fg / bg from behind are untouched — the pane is see-through.
        assert_eq!(
            c.glyph,
            u32::from('#'),
            "glass keeps the scene glyph behind it"
        );
        assert_eq!(c.fg, 0x00FF_0000, "glass keeps the scene fg behind it");
        assert_eq!(c.bg, 0x0040_0000, "glass keeps the scene bg behind it");
        // ONLY the underline (glass marker) is added.
        assert_eq!(
            c.style.underline_color(),
            Some(0x0080_C8FF),
            "glass stamps its underline marker"
        );
    }

    // ───────────────────── pitch-pan blank regression (cache bug) ─────────────
    //
    // Drives the LIVE composite system (`CompositeRenderPlugin`) across frames so the
    // near rotation-pan ring becomes valid, then sweeps `cam.pitch` and asserts the
    // resolved framebuffer is NON-EMPTY at every pitch. Reproduces the playtest report
    // "pitch up/down → pure black, only emitters visible, wiggle to recover": the pan
    // path served a near ring whose pitch row-shift pushed all geometry off-screen.

    /// Count non-default cells in the App's `Framebuffer` resource.
    fn fb_non_default(app: &bevy::app::App) -> usize {
        let fb = app.world().resource::<Framebuffer>();
        let (cols, rows) = (fb.cols(), fb.rows());
        let mut n = 0usize;
        for row in 0..rows {
            for col in 0..cols {
                if fb.get(col, row).copied() != Some(re_core::Cell::default()) {
                    n += 1;
                }
            }
        }
        n
    }

    /// Build a headless App running the real composite renderer on the test room.
    fn pitch_repro_app(cols: usize, rows: usize) -> bevy::app::App {
        use bevy::MinimalPlugins;
        let mut app = bevy::app::App::new();
        app.add_plugins(MinimalPlugins)
            .add_plugins(CompositeRenderPlugin);
        app.insert_resource(re_world::test_room());
        app.insert_resource(Framebuffer::new(cols, rows));
        let cam = CameraPose::spawn_in_test_room();
        app.world_mut().spawn(cam);
        app
    }

    /// REGRESSION: pitching the camera up or down must NEVER blank the frame. We warm
    /// the ring with several level-pitch frames (so the pan path is taken), then sweep
    /// pitch from the down-clamp to the up-clamp — including the bucket boundaries near
    /// ±k·TAU/128 and the ±π/3 clamp — re-running the renderer at each. Every frame must
    /// resolve real geometry.
    ///
    /// The sweep range is the WIDENED ±π/3 (60°) clamp (= `re_render::PITCH_LIMIT_RAD`
    /// = `re_present::PITCH_LIMIT`). At the extremes the near cast's horizon shift hits
    /// ±rows/2, so the whole column is sky/ceiling (up) or floor (down) — the near
    /// opaque layer still fills every row, so even though the far panorama is vacated
    /// beyond its baked band, the composite is never black. This asserts that across the
    /// full new band.
    #[test]
    fn pitch_sweep_never_blanks_through_the_pan_path() {
        let (cols, rows) = (80usize, 40usize);
        let mut app = pitch_repro_app(cols, rows);

        // Warm-up: run several frames at level pitch so the far ring bakes and the near
        // rotation-pan ring completes a bake and becomes `valid` (so a later pitch frame
        // takes the PAN branch — the suspect path).
        for _ in 0..16 {
            app.update();
        }
        assert!(
            fb_non_default(&app) > 0,
            "warm-up frame must render geometry"
        );

        // Sweep pitch across the full WIDENED range (±π/3). Step is a fraction of the
        // bucket so we land on, just below, and just above every bucket boundary up to
        // π/3, and we include the exact clamp.
        let limit = crate::PITCH_LIMIT_RAD;
        assert!(
            (limit - std::f32::consts::FRAC_PI_3).abs() < 1e-6,
            "sweep must cover the widened ±60° band"
        );
        let step = NEAR_PITCH_BUCKET / 3.0;
        let mut pitch = -limit;
        let mut blanks: Vec<f32> = Vec::new();
        while pitch <= limit + 1e-4 {
            // Set the live pitch and run one render frame through the cache/pan path.
            {
                let mut q = app.world_mut().query::<&mut CameraPose>();
                let mut cam = q.single_mut(app.world_mut()).unwrap();
                cam.pitch = pitch.clamp(-limit, limit);
            }
            app.update();
            if fb_non_default(&app) == 0 {
                blanks.push(pitch);
            }
            pitch += step;
        }

        assert!(
            blanks.is_empty(),
            "framebuffer went BLANK at pitch values {blanks:?} — \
             pitching must never blank the frame"
        );
    }

    /// Count rows that contain at least one DEFAULT (unpainted) cell — a candidate
    /// "black band". A fully-filled frame has zero such rows (every row is covered by
    /// the near opaque sky/ceiling/floor fill, even where the far panorama vacates).
    fn rows_with_holes(app: &bevy::app::App) -> Vec<usize> {
        let fb = app.world().resource::<Framebuffer>();
        let (cols, rows) = (fb.cols(), fb.rows());
        let mut holes = Vec::new();
        for row in 0..rows {
            let mut any_default = false;
            for col in 0..cols {
                if fb.get(col, row).copied() == Some(re_core::Cell::default()) {
                    any_default = true;
                    break;
                }
            }
            if any_default {
                holes.push(row);
            }
        }
        holes
    }

    /// REGRESSION (widened-pitch fill): at the WIDENED ±π/3 clamp extremes — looking
    /// straight-ish UP and DOWN — the frame must be FULLY covered, with NO black band
    /// where the far panorama is vacated by the (now larger) horizon shift. The near
    /// opaque layer paints every row (wall + sky/ceiling above the horizon, floor
    /// below), so the composite has zero unpainted rows at ±60°. This is the core
    /// guarantee for the wider band: the bigger vacated far region is backed by near
    /// fill, not black.
    #[test]
    fn widened_pitch_extremes_fill_with_no_black_band() {
        let (cols, rows) = (80usize, 40usize);
        let mut app = pitch_repro_app(cols, rows);
        for _ in 0..16 {
            app.update();
        }

        let limit = crate::PITCH_LIMIT_RAD;
        assert!(
            (limit - std::f32::consts::FRAC_PI_3).abs() < 1e-6,
            "test asserts the widened ±60° clamp"
        );

        // Sweep the upper / lower extremes (the clamp itself plus a few bucket steps
        // just inside it, where the horizon shift is near ±rows/2 and the far panorama
        // is most vacated). Every probed frame must be hole-free.
        let step = NEAR_PITCH_BUCKET;
        for k in 0..4 {
            for &sign in &[1.0f32, -1.0f32] {
                let pitch = (sign * (limit - k as f32 * step)).clamp(-limit, limit);
                {
                    let mut q = app.world_mut().query::<&mut CameraPose>();
                    let mut cam = q.single_mut(app.world_mut()).unwrap();
                    cam.pitch = pitch;
                }
                app.update();
                let holes = rows_with_holes(&app);
                assert!(
                    holes.is_empty(),
                    "BLACK BAND at pitch {pitch:.4} rad ({:.1}°): rows {holes:?} have \
                     unpainted cells — the widened band must be fully filled by the near \
                     sky/ceiling/floor fill",
                    pitch.to_degrees()
                );
            }
        }
    }
}
