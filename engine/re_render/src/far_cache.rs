//! `re_render::far_cache` — the cylindrical far panorama.
//!
//! The far backdrop is the rarely-recomputed tier of the composite stack: we
//! bake a panorama that spans a full revolution (TAU radians) around the
//! camera into a wide [`Layer`] ONCE, then on
//! every subsequent frame fill the view-width far layer by **radial-sampling**
//! that panorama at the current yaw — a column offset + wrap, **zero raycasting**.
//! That radial-sample path is the economy claim, so it is real and tested here.
//!
//! Re-baking (casting the wide ring) is gated by [`crate::tier::far_needs_rebake`]
//! upstream; this module exposes [`FarCache::rebake`] which performs the actual
//! cast. The cast reuses the existing per-column raycast over yaw-stepped poses.

use bevy::prelude::Resource;
use glam::Vec3;
use re_core::{Cell, Layer, LayerBlend};
use re_world::{CameraPose, Map};

use crate::ascii_shader::MaterialRegistry;
use crate::raycast_with_time;

/// Number of panorama columns covering a full TAU revolution. A turn slides a
/// view-width window over this ring. 480 matches the level format's panoramic
/// yaw encoding (`PANORAMIC_COLS`) so a yaw maps cleanly to a ring column.
pub const PANORAMA_COLS: usize = 480;

/// How many ring columns the INCREMENTAL re-bake casts per frame. The 480-column
/// ring used to be cast in ONE `rebake` call (~39 ms worst-frame spike); splitting
/// it into chunks spreads that across `PANORAMA_COLS / BAKE_CHUNK_COLS ≈ 8` frames
/// so the per-frame far cost is ~one chunk (~5 ms) instead of the whole ring.
///
/// 64 columns matches the bake's internal slab width (one FOV cast per chunk), so a
/// chunk is exactly one slab cast — the cheapest unit that still copies cleanly into
/// the ring. 480 / 64 = 7.5 → the bake completes in 8 frames.
pub const BAKE_CHUNK_COLS: usize = 64;

/// The cached cylindrical far backdrop plus the pose it was baked at.
///
/// `panorama` is `PANORAMA_COLS` wide and as tall as the view. `base_yaw` /
/// `base_pos` record the bake pose so the radial sample can offset by the current
/// yaw relative to the bake, and the tier policy can decide when to re-bake.
#[derive(Resource, Clone, Debug)]
pub struct FarCache {
    /// The CURRENTLY SERVED ring. [`fill_far_layer`](Self::fill_far_layer) always
    /// radial-samples THIS buffer, so it is never mid-write — an in-progress bake
    /// writes the separate [`back`](Self::back) buffer and only swaps when complete.
    panorama: Layer,
    /// The SCRATCH / back ring an incremental bake fills column-by-column over
    /// several frames. Never sampled; atomically swapped into `panorama` on
    /// completion (so the served ring is never half-baked → no tear).
    back: Layer,
    /// Yaw the SERVED ring (`panorama`) was baked at (radians).
    pub base_yaw: f32,
    /// Camera position the SERVED ring (`panorama`) was baked at.
    pub base_pos: Vec3,
    /// `false` until the first successful bake; the fill path produces an empty
    /// (absent) far layer while invalid.
    pub valid: bool,
    /// `true` while an incremental bake is filling `back`. The served `panorama`
    /// stays valid and is sampled normally throughout.
    bake_in_progress: bool,
    /// Next ring column the incremental bake will write into `back` (0 = not started
    /// / just restarted; `PANORAMA_COLS` = finished, pending swap).
    bake_next_col: usize,
    /// Yaw the IN-PROGRESS bake started at — becomes `base_yaw` on swap so the
    /// served ring records the pose its columns were actually cast from.
    bake_start_yaw: f32,
    /// Camera position the IN-PROGRESS bake started at — becomes `base_pos` on swap.
    bake_start_pos: Vec3,
    /// GLYPH MEMORY for the far deadband, WORLD/DIRECTION-anchored. Keyed by
    /// the `(ring column, panorama row)` pair the glyph came from — **not** by
    /// the view/screen cell that happened to display it. Row-major over
    /// `PANORAMA_COLS × panorama.rows()`, i.e. it is sized to the PANORAMA,
    /// never the view width, and a fixed ring cell keeps its own hysteresis
    /// state no matter which screen column is currently showing it.
    ///
    /// This replaces a PER-VIEW-CELL memory that was a genuine screen anchor:
    /// under a continuous turn, the ring pair a given screen column shows
    /// changes every frame (a pure function of yaw), so a memory keyed on the
    /// SCREEN cell was comparing
    /// each frame's decisive-crossing test against a glyph left by a DIFFERENT,
    /// unrelated ring pair from one/more frames back — occasionally producing a
    /// spurious hold purely by coincidental glyph re-use (`painted_cache` cycles
    /// glyphs every 26 ring columns, so a collision is common), never because
    /// that ring pair itself was ever ambiguous. Keying on the ring pair's own
    /// identity means a ring cell's hysteresis is seeded ONLY by its own prior
    /// history — see `far_glyph_memory_is_world_anchored_not_screen_anchored`.
    ///
    /// [`FAR_GLYPH_NO_MEMORY`] means "no glyph on record for this ring cell yet".
    /// Cleared whenever the panorama content changes wholesale (a resize, or a
    /// bake swap — old memory from a superseded bake must not seed the new one's
    /// hysteresis, `rebake`/`bake_step`).
    prev_glyph: Vec<u32>,
}

/// Sentinel in [`FarCache::prev_glyph`] for "no glyph on record at this cell" (first
/// fill after a resize, or a cell the previous fill left absent). `u32::MAX` is not a
/// valid Unicode scalar, so it can never collide with a real glyph.
const FAR_GLYPH_NO_MEMORY: u32 = u32::MAX;

/// Half-width of the far-layer glyph HYSTERESIS band around the sub-column blend
/// midpoint (`frac = 0.5`), where `frac` is how far the view column sits between the
/// two adjacent panorama columns it blends.
///
/// Without it, a yaw that hovers near a half-column boundary flips EVERY far cell's
/// glyph between its two neighbouring ring columns from fill to fill — a whole-
/// backdrop alternation for a sub-column change nobody can see. With it, a fill whose
/// `frac` lands inside `0.5 ± FAR_GLYPH_DEADBAND` KEEPS the glyph the previous fill
/// chose (when that glyph is still one of the two candidates); only a decisive
/// crossing switches. Colours still blend continuously, so the backdrop keeps sliding
/// smoothly — only the discrete channel is damped.
///
/// 0.1 = a 20%-of-a-column band. Wide enough to swallow mouse-look jitter
/// (`MOUSE_YAW_SENS` steps are far smaller than a column), narrow enough that a real
/// turn never drags a stale glyph more than a fifth of a column past its boundary.
pub const FAR_GLYPH_DEADBAND: f32 = 0.1;

impl FarCache {
    /// Allocate an unbaked far cache sized for a `rows`-tall view. The panorama is
    /// always [`PANORAMA_COLS`] wide; `rows` matches the framebuffer height.
    pub fn new(rows: usize) -> Self {
        FarCache {
            // The panorama is an Opaque backdrop (composite slot 0).
            panorama: Layer::new(PANORAMA_COLS, rows, LayerBlend::Opaque),
            back: Layer::new(PANORAMA_COLS, rows, LayerBlend::Opaque),
            base_yaw: 0.0,
            base_pos: Vec3::ZERO,
            valid: false,
            bake_in_progress: false,
            bake_next_col: 0,
            bake_start_yaw: 0.0,
            bake_start_pos: Vec3::ZERO,
            // Sized to the PANORAMA (world-anchored), not the view — see the field doc.
            prev_glyph: vec![FAR_GLYPH_NO_MEMORY; PANORAMA_COLS * rows],
        }
    }

    /// Whether the served ring (`panorama`) has been baked at least once and is safe
    /// to sample. Mirrors [`crate::gbuffer::NearGBuffer::is_valid`] so the composite
    /// path queries near + far cache validity through one consistent method pair.
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.valid
    }

    /// Whether an incremental bake is currently in progress (filling the back ring).
    /// The served ring (`panorama`) is still sampled normally; this is exposed for
    /// the tier policy + tests.
    #[inline]
    pub fn bake_in_progress(&self) -> bool {
        self.bake_in_progress
    }

    /// Signed yaw + position the IN-PROGRESS bake started at (the reference the
    /// mid-bake-move restart policy measures against). Meaningless when no bake is in
    /// progress. Exposed so the composite tier can compute the "moved significantly
    /// since the bake started" predicate with the SAME thresholds it uses elsewhere.
    #[inline]
    pub fn bake_start_pose(&self) -> (f32, Vec3) {
        (self.bake_start_yaw, self.bake_start_pos)
    }

    /// `(panorama_cols, rows)`.
    #[inline]
    pub fn dims(&self) -> (usize, usize) {
        self.panorama.dims()
    }

    /// Resize the panorama height to match a new view height (clearing + marking
    /// invalid so the next frame re-bakes). Width stays [`PANORAMA_COLS`].
    pub fn resize_rows(&mut self, rows: usize) {
        if self.panorama.rows() != rows {
            self.panorama.resize(PANORAMA_COLS, rows);
            self.back.resize(PANORAMA_COLS, rows);
            self.valid = false;
            // A resize changes the bake geometry; discard any partial back ring so
            // the next bake re-casts at the new height (never swap a stale-size ring).
            self.bake_in_progress = false;
            self.bake_next_col = 0;
            // The glyph memory is sized to the panorama; a height change invalidates it.
            self.prev_glyph.clear();
            self.prev_glyph
                .resize(PANORAMA_COLS * rows, FAR_GLYPH_NO_MEMORY);
        }
    }

    /// Discard the ring-keyed glyph memory. Called whenever the SERVED panorama's
    /// content changes wholesale (a bake swap): the old memory was seeded by the
    /// PREVIOUS bake's content at each ring column, and a fresh bake can put
    /// entirely different world content at the same ring index (the player moved,
    /// or time advanced), so carrying it forward would seed the new hysteresis
    /// with a stale, unrelated glyph — the same screen-vs-world anchoring
    /// mistake the ring-keyed memory itself avoids, just on the bake axis
    /// instead of the screen axis.
    fn forget_glyph_memory(&mut self) {
        self.prev_glyph.fill(FAR_GLYPH_NO_MEMORY);
    }

    /// Map a world yaw (radians) to a fractional panorama column. The ring spans
    /// `[0, TAU) → [0, PANORAMA_COLS)`. The yaw convention matches
    /// [`CameraPose::dir`] (yaw=0 faces +y).
    #[inline]
    fn yaw_to_ring_col(yaw: f32) -> f32 {
        let tau = std::f32::consts::TAU;
        let frac = yaw.rem_euclid(tau) / tau;
        frac * PANORAMA_COLS as f32
    }

    /// Fill the view-width far `out` layer by RADIAL-SAMPLING the cached panorama
    /// at `cam`'s current yaw + pitch — no raycasting, just a per-column offset into
    /// the ring with wrap. The view's centre column maps to the camera's facing yaw;
    /// columns spread ±`out.cols()/2` around it.
    ///
    /// Turn-stability details:
    /// * **Sub-column interpolation** — a view column maps to a FRACTIONAL ring
    ///   column; the cell's COLOURS (fg/bg) are linearly blended between the two
    ///   adjacent ring columns by the fraction, so a slow turn slides the backdrop
    ///   smoothly instead of snapping column-by-column (kills the turn shimmer).
    ///   The discrete glyph/geometry fields come from the nearer ring column —
    ///   damped by the [`FAR_GLYPH_DEADBAND`] hysteresis band so a `frac`
    ///   hovering at the 0.5 boundary does NOT alternate the whole backdrop's
    ///   glyphs between the two candidate columns.
    /// * **Pitch horizon-shift** — the panorama is baked at pitch 0, so when the
    ///   camera pitches we read panorama row `row − pitch_row_shift(pitch)`: the far
    ///   horizon tracks the SAME shift the near cast applies, so far and near share
    ///   one horizon (no pitch seam). Rows that fall outside the baked panorama stay
    ///   absent (open sky above / below the panorama).
    /// * **World-anchored glyph memory** — the deadband's "previous glyph"
    ///   lookup is keyed on the RING pair `(ring0, src_row)`, not on `(col, row)`.
    ///   A given ring cell's hysteresis therefore depends only on ITS OWN prior
    ///   fills, never on whatever unrelated ring cell a screen column showed a
    ///   frame ago — see the [`prev_glyph`](Self::prev_glyph) field doc and
    ///   `far_glyph_memory_is_world_anchored_not_screen_anchored`.
    ///
    /// If the cache is not [`valid`](Self::valid) the `out` layer is left fully
    /// absent (the far backdrop simply doesn't contribute until first bake).
    pub fn fill_far_layer(&mut self, cam: &CameraPose, out: &mut Layer) {
        out.clear();
        if !self.valid {
            return;
        }
        let (cols, rows) = out.dims();
        if cols == 0 || rows == 0 {
            return;
        }
        let (pan_w, pan_rows) = self.panorama.dims();
        // The memory is sized to the PANORAMA, never the view width — a view-size
        // change does not touch it (only `resize_rows`/a bake does; see the field
        // doc). Steady state: the length already matches and this is a no-op.
        if self.prev_glyph.len() != pan_w * pan_rows {
            self.prev_glyph.clear();
            self.prev_glyph
                .resize(pan_w * pan_rows, FAR_GLYPH_NO_MEMORY);
        }
        let pan_cols = PANORAMA_COLS as f32;
        // Centre view column samples the camera-facing ring column; one ring col per
        // view col, but sampled at a FRACTIONAL position for sub-column blending.
        let centre_ring = Self::yaw_to_ring_col(cam.yaw);
        let half = cols as f32 * 0.5;
        // Pitch slides the backdrop vertically by the SAME row shift the near cast
        // uses, so the two tiers share one horizon.
        let pitch_shift = crate::pitch_row_shift(cam.pitch, rows);
        for col in 0..cols {
            // Fractional ring column, wrapped into [0, PANORAMA_COLS).
            let ring_f = (centre_ring + (col as f32 - half)).rem_euclid(pan_cols);
            let c0 = ring_f.floor();
            let frac = ring_f - c0;
            let ring0 = c0 as usize % PANORAMA_COLS;
            let ring1 = (ring0 + 1) % PANORAMA_COLS;
            for row in 0..rows {
                // Read the panorama row offset by pitch (baked at pitch 0).
                let src_row = row as i32 - pitch_shift;
                if src_row < 0 || src_row as usize >= rows || src_row as usize >= pan_rows {
                    continue; // outside the baked panorama → open sky, leave absent.
                }
                let src_row = src_row as usize;
                let a = self.panorama.cell_raw(ring0, src_row);
                let b = self.panorama.cell_raw(ring1, src_row);
                // WORLD-anchored memory key: the ring pair's own identity, not
                // the screen cell displaying it. `ring0` is the lower candidate index,
                // which is stable across frames for THIS ring cell however many
                // different screen columns come to show it while the camera turns.
                let mem = src_row * pan_w + ring0;
                match (a, b) {
                    (Some(a), Some(b)) => {
                        // Discrete fields from the nearer column (with hysteresis);
                        // colours blended.
                        let near = if far_take_b(frac, self.prev_glyph[mem], a.glyph, b.glyph) {
                            b
                        } else {
                            a
                        };
                        let mut cell = *near;
                        cell.fg = lerp_rgb(a.fg, b.fg, frac);
                        cell.bg = lerp_rgb(a.bg, b.bg, frac);
                        self.prev_glyph[mem] = cell.glyph;
                        out.set(col, row, cell);
                    }
                    // One side absent: no honest blend — take whichever exists.
                    (Some(a), None) => {
                        self.prev_glyph[mem] = a.glyph;
                        out.set(col, row, *a);
                    }
                    (None, Some(b)) => {
                        self.prev_glyph[mem] = b.glyph;
                        out.set(col, row, *b);
                    }
                    // Nothing written here: forget the glyph so the deadband cannot
                    // resurrect it after the cell reappears.
                    (None, None) => self.prev_glyph[mem] = FAR_GLYPH_NO_MEMORY,
                }
            }
        }
    }

    /// Bake the ring columns `[ring_col0, ring_col0 + count)` of `target` by casting
    /// one FOV slab centred on that arc at `pos` and copying its columns in. Shared
    /// by the one-shot [`rebake`](Self::rebake) and the incremental
    /// [`bake_step`](Self::bake_step). `scratch` is a reused `slab_cols`-wide
    /// framebuffer (no per-call alloc on the hot path). `time = 0` always — the far
    /// backdrop is FROZEN so distant materials never pulse and never go stale from
    /// animation; the NEAR tier still animates via its own time input.
    fn bake_slab(
        target: &mut Layer,
        scratch: &mut re_core::Framebuffer,
        map: &Map,
        registry: &MaterialRegistry,
        pos: Vec3,
        ring_col0: usize,
        count: usize,
    ) {
        let (pan_cols, rows) = target.dims();
        let tau = std::f32::consts::TAU;
        // Yaw at the CENTRE of this ring slab (inverse of yaw_to_ring_col).
        let centre_ring = ring_col0 as f32 + count as f32 * 0.5;
        let slab_yaw = (centre_ring / pan_cols as f32) * tau;
        let pose = CameraPose {
            pos,
            yaw: slab_yaw,
            pitch: 0.0,
        };
        scratch.clear();
        // FROZEN animation: time = 0 so the distant backdrop is static.
        raycast_with_time(map, &pose, registry, 0.0, scratch);
        // Copy the slab's centre columns into the ring. The scratch cast is
        // slab_cols wide centred on slab_yaw; its column `c` maps to ring column
        // `ring_col0 + c` for the first `count` columns.
        for c in 0..count {
            for row in 0..rows {
                if let Some(cell) = scratch.get(c, row) {
                    target.set(ring_col0 + c, row, *cell);
                } else {
                    target.set(ring_col0 + c, row, Cell::default());
                }
            }
        }
    }

    /// Re-bake the WHOLE cylindrical panorama in ONE call by casting a full TAU of
    /// view around `cam`'s position. Records the bake pose and marks the cache valid.
    ///
    /// This is the EXPENSIVE all-at-once path (~480-column cast). It is used for the
    /// FIRST-EVER bake on level load — see [`bake_step`](Self::bake_step)'s docs for
    /// why the first bake is one-shot — and remains available for callers / tests
    /// that want a synchronous ring. Steady-state re-bakes use the INCREMENTAL
    /// [`bake_step`](Self::bake_step) to avoid the worst-frame spike.
    ///
    /// Frozen animation (`time = 0`) — the far backdrop never pulses.
    pub fn rebake(&mut self, map: &Map, cam: &CameraPose, registry: &MaterialRegistry) {
        let (pan_cols, rows) = self.panorama.dims();
        if pan_cols == 0 || rows == 0 {
            return;
        }
        let slab_cols = BAKE_CHUNK_COLS.min(pan_cols);
        let mut scratch = re_core::Framebuffer::new(slab_cols, rows);
        let mut ring_col0 = 0usize;
        while ring_col0 < pan_cols {
            let this_slab = slab_cols.min(pan_cols - ring_col0);
            Self::bake_slab(
                &mut self.panorama,
                &mut scratch,
                map,
                registry,
                cam.pos,
                ring_col0,
                this_slab,
            );
            ring_col0 += this_slab;
        }
        self.base_yaw = cam.yaw;
        self.base_pos = cam.pos;
        self.valid = true;
        // A one-shot bake supersedes any partial incremental bake.
        self.bake_in_progress = false;
        self.bake_next_col = 0;
        // Fresh content at every ring column — the old glyph memory no longer
        // describes what is there (see `forget_glyph_memory`).
        self.forget_glyph_memory();
    }

    /// Advance an INCREMENTAL re-bake by ONE chunk ([`BAKE_CHUNK_COLS`] columns),
    /// writing the SCRATCH (`back`) ring while the currently-served `panorama` is
    /// left untouched. Returns `true` on the frame the bake COMPLETES (the back ring
    /// is atomically swapped into `panorama` and `base_yaw`/`base_pos` are set to the
    /// pose the bake STARTED at — the pose its columns were cast from).
    ///
    /// Drive this every frame `far_needs_rebake` is true. It starts a fresh bake at
    /// `cam` when none is in progress, then bakes one chunk per call. The served ring
    /// is NEVER half-baked (no tear): the back ring is filled in full before the swap.
    ///
    /// MID-BAKE CAMERA MOVEMENT POLICY (documented):
    /// * `should_restart` lets the caller signal that the camera moved SIGNIFICANTLY
    ///   since this bake STARTED (a yaw bucket crossing or a ≥1-cell move from
    ///   `bake_start_*`). When a bake is already in progress and `should_restart` is
    ///   true, the partial back ring is DISCARDED and the bake restarts from the new
    ///   `cam` pose — the panorama being baked must reflect where the camera is now.
    /// * A TINY move (`should_restart` false) lets the in-progress bake FINISH at its
    ///   original pose; the tier policy will simply re-trigger another bake afterwards
    ///   if the camera has drifted out of bucket by then. This bounds churn so a slow
    ///   creep does not perpetually restart and never complete.
    pub fn bake_step(
        &mut self,
        map: &Map,
        cam: &CameraPose,
        registry: &MaterialRegistry,
        should_restart: bool,
    ) -> bool {
        let (pan_cols, rows) = self.panorama.dims();
        if pan_cols == 0 || rows == 0 {
            return false;
        }
        // (Re)start the bake: fresh, or a significant mid-bake move discards the partial.
        if !self.bake_in_progress || should_restart {
            self.bake_in_progress = true;
            self.bake_next_col = 0;
            self.bake_start_yaw = cam.yaw;
            self.bake_start_pos = cam.pos;
        }

        let slab_cols = BAKE_CHUNK_COLS.min(pan_cols);
        let mut scratch = re_core::Framebuffer::new(slab_cols, rows);
        // Bake exactly one chunk into the BACK ring at the bake's START pose (so all
        // columns of one ring share a single consistent viewpoint).
        let ring_col0 = self.bake_next_col;
        let this_slab = slab_cols.min(pan_cols - ring_col0);
        Self::bake_slab(
            &mut self.back,
            &mut scratch,
            map,
            registry,
            self.bake_start_pos,
            ring_col0,
            this_slab,
        );
        self.bake_next_col = ring_col0 + this_slab;

        // Complete? Atomically swap the freshly-baked back ring into service.
        if self.bake_next_col >= pan_cols {
            std::mem::swap(&mut self.panorama, &mut self.back);
            self.base_yaw = self.bake_start_yaw;
            self.base_pos = self.bake_start_pos;
            self.valid = true;
            self.bake_in_progress = false;
            self.bake_next_col = 0;
            // Fresh content at every ring column (see `forget_glyph_memory`).
            self.forget_glyph_memory();
            return true;
        }
        false
    }
}

/// The far-layer glyph HYSTERESIS decision: given the sub-column blend fraction
/// `frac ∈ [0,1)`, the glyph this view cell carried on the PREVIOUS fill
/// (`prev`, [`FAR_GLYPH_NO_MEMORY`] for none), and the two candidate ring columns'
/// glyphs, return `true` to take column `b`'s discrete fields and `false` to take
/// `a`'s.
///
/// * Decisive: `frac ≥ 0.5 + `[`FAR_GLYPH_DEADBAND`] ⇒ `b`; `frac ≤ 0.5 − band` ⇒ `a`.
/// * Inside the band: KEEP whichever candidate matches the previous glyph. If the
///   memory is empty, or both candidates carry that same glyph (nothing to keep), or
///   it matches neither (the panorama moved under us — a rebake, or the yaw jumped a
///   whole column), fall back to the plain `frac < 0.5` rule.
///
/// Pure and total: every input shape yields a defined choice, and with
/// [`FAR_GLYPH_DEADBAND`] `= 0` it is exactly the old `frac < 0.5` snap.
#[inline]
fn far_take_b(frac: f32, prev: u32, glyph_a: u32, glyph_b: u32) -> bool {
    if !frac.is_finite() {
        return false;
    }
    if frac >= 0.5 + FAR_GLYPH_DEADBAND {
        return true;
    }
    if frac <= 0.5 - FAR_GLYPH_DEADBAND {
        return false;
    }
    // Inside the hysteresis band: hold the previous choice when it is unambiguous.
    if prev != FAR_GLYPH_NO_MEMORY && glyph_a != glyph_b {
        if prev == glyph_b {
            return true;
        }
        if prev == glyph_a {
            return false;
        }
    }
    frac >= 0.5
}

/// Linearly interpolate two packed `0x00RRGGBB` colours per channel by `t ∈ [0,1]`.
/// Used to blend the far backdrop between adjacent ring columns so a slow turn
/// slides the colour smoothly instead of snapping (turn-shimmer fix).
#[inline]
fn lerp_rgb(a: u32, b: u32, t: f32) -> u32 {
    let t = t.clamp(0.0, 1.0);
    let lerp = |ca: u32, cb: u32| -> u32 {
        let fa = ca as f32;
        let fb = cb as f32;
        (fa + (fb - fa) * t).round().clamp(0.0, 255.0) as u32
    };
    let r = lerp((a >> 16) & 0xFF, (b >> 16) & 0xFF);
    let g = lerp((a >> 8) & 0xFF, (b >> 8) & 0xFF);
    let bl = lerp(a & 0xFF, b & 0xFF);
    (r << 16) | (g << 8) | bl
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell(glyph: char) -> Cell {
        Cell {
            glyph: u32::from(glyph),
            depth: 100.0,
            ..Cell::default()
        }
    }

    /// Build a FarCache and paint its panorama directly so we can test the
    /// radial-sample path WITHOUT a map/registry (the sample is the economy claim).
    fn painted_cache(rows: usize) -> FarCache {
        let mut fc = FarCache::new(rows);
        // Paint each ring column with a glyph encoding its column index mod 26 so
        // we can read which ring column a view column sampled.
        for col in 0..PANORAMA_COLS {
            let g = (b'a' + (col % 26) as u8) as char;
            for row in 0..rows {
                fc.panorama.set(col, row, cell(g));
            }
        }
        fc.valid = true;
        fc.base_yaw = 0.0;
        fc
    }

    fn glyph_of(col: usize) -> u32 {
        u32::from((b'a' + (col % 26) as u8) as char)
    }

    #[test]
    fn invalid_cache_fills_nothing() {
        let mut fc = FarCache::new(4);
        let mut out = Layer::new(8, 4, LayerBlend::Opaque);
        out.set(0, 0, cell('Z')); // sentinel that must be cleared
        let cam = CameraPose {
            pos: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
        };
        fc.fill_far_layer(&cam, &mut out);
        assert!(
            out.get(0, 0).is_none(),
            "invalid cache must clear the layer"
        );
    }

    #[test]
    fn radial_sample_centre_matches_facing_yaw() {
        let mut fc = painted_cache(2);
        let cols = 10usize;
        let mut out = Layer::new(cols, 2, LayerBlend::Opaque);
        // yaw = 0 → centre ring col 0. Centre view col = cols/2 = 5.
        let cam = CameraPose {
            pos: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
        };
        fc.fill_far_layer(&cam, &mut out);
        // view col 5 maps to ring col 0.
        assert_eq!(out.get(5, 0).unwrap().glyph, glyph_of(0));
        // view col 6 maps to ring col 1.
        assert_eq!(out.get(6, 0).unwrap().glyph, glyph_of(1));
        // view col 4 maps to ring col PANORAMA_COLS-1 (wrap).
        assert_eq!(out.get(4, 0).unwrap().glyph, glyph_of(PANORAMA_COLS - 1));
    }

    #[test]
    fn radial_sample_offsets_with_yaw() {
        let mut fc = painted_cache(1);
        let cols = 10usize;
        let mut out = Layer::new(cols, 1, LayerBlend::Opaque);
        // Quarter turn: yaw = TAU/4 → centre ring col = PANORAMA_COLS/4 = 120.
        let cam = CameraPose {
            pos: Vec3::ZERO,
            yaw: std::f32::consts::TAU / 4.0,
            pitch: 0.0,
        };
        fc.fill_far_layer(&cam, &mut out);
        // centre view col 5 → ring col 120.
        assert_eq!(out.get(5, 0).unwrap().glyph, glyph_of(120));
        assert_eq!(out.get(7, 0).unwrap().glyph, glyph_of(122));
    }

    #[test]
    fn radial_sample_wraps_full_circle() {
        // yaw just below TAU should land near ring col 0 (wrap), not out of range.
        let mut fc = painted_cache(1);
        let cols = 4usize;
        let mut out = Layer::new(cols, 1, LayerBlend::Opaque);
        let cam = CameraPose {
            pos: Vec3::ZERO,
            yaw: std::f32::consts::TAU - 1e-4,
            pitch: 0.0,
        };
        fc.fill_far_layer(&cam, &mut out);
        // Every view col must be filled (no panic / no gap) and in-range glyphs.
        for col in 0..cols {
            let g = out.get(col, 0).unwrap().glyph;
            assert!((u32::from('a')..=u32::from('z')).contains(&g));
        }
    }

    #[test]
    fn yaw_to_ring_col_wraps() {
        let tau = std::f32::consts::TAU;
        assert!((FarCache::yaw_to_ring_col(0.0) - 0.0).abs() < 1e-3);
        assert!(
            (FarCache::yaw_to_ring_col(tau) - 0.0).abs() < 1e-3,
            "TAU wraps to 0"
        );
        assert!((FarCache::yaw_to_ring_col(tau / 2.0) - 240.0).abs() < 1e-3);
        // negative yaw wraps positive.
        assert!((FarCache::yaw_to_ring_col(-tau / 4.0) - 360.0).abs() < 1e-3);
    }

    // ── H2: sub-column colour interpolation kills turn shimmer ──

    /// Paint a strong red CONTRAST between the two ring columns the centre view
    /// column brackets at yaw≈0 (col 0 = red 0, col 1 = red 240, others mid-grey),
    /// so the interpolated colour the radial sample produces is unambiguous.
    fn colour_cache(rows: usize) -> FarCache {
        let mut fc = FarCache::new(rows);
        for col in 0..PANORAMA_COLS {
            let r = match col {
                0 => 0u32,
                1 => 240u32,
                _ => 120u32,
            };
            let fg = r << 16;
            let c = Cell {
                glyph: u32::from('#'),
                fg,
                bg: fg,
                depth: 100.0,
                ..Cell::default()
            };
            for row in 0..rows {
                fc.panorama.set(col, row, c);
            }
        }
        fc.valid = true;
        fc.base_yaw = 0.0;
        fc
    }

    fn red(c: u32) -> i32 {
        ((c >> 16) & 0xFF) as i32
    }

    #[test]
    fn far_sample_interpolates_colour_on_a_slow_turn() {
        // A small yaw delta (a fraction of one ring column) must move the sampled
        // colour SMOOTHLY between adjacent ring columns (0 → 240) — not snap by a
        // whole column. The centre view column brackets ring cols 0 and 1 at yaw≈0.
        let mut fc = colour_cache(1);
        let cols = 10usize;
        let one_ring_col_yaw = std::f32::consts::TAU / PANORAMA_COLS as f32;

        let mut sample_red_at = |yaw: f32| -> i32 {
            let mut out = Layer::new(cols, 1, LayerBlend::Opaque);
            let cam = CameraPose {
                pos: Vec3::ZERO,
                yaw,
                pitch: 0.0,
            };
            fc.fill_far_layer(&cam, &mut out);
            red(out.get(cols / 2, 0).unwrap().fg)
        };

        let r0 = sample_red_at(0.0);
        let r_quarter = sample_red_at(one_ring_col_yaw * 0.25);
        let r_three_q = sample_red_at(one_ring_col_yaw * 0.75);
        let r1 = sample_red_at(one_ring_col_yaw);
        // Bracketed, monotonic slide from 0 toward 240 across the sub-column range.
        assert!(
            r0 <= r_quarter && r_quarter <= r_three_q && r_three_q <= r1,
            "far colour must slide smoothly with yaw: {r0},{r_quarter},{r_three_q},{r1}"
        );
        // Endpoints sit on the two columns; the sub-column samples interpolate
        // strictly BETWEEN them (genuine blend, not a snap).
        assert_eq!(r0, 0, "centre col reads ring col 0 at yaw 0");
        assert!(
            r1 >= 239,
            "one ring-column of yaw reaches ring col 1 (~240)"
        );
        assert!(
            r_quarter > r0 && r_three_q < r1,
            "sub-column turns must interpolate strictly between the columns \
             ({r_quarter} between {r0} and {r1}; {r_three_q} between {r0} and {r1})"
        );
    }

    // ── Incremental bake ──

    /// A tiny map + registry so the bake actually casts (the incremental-bake tests
    /// need a real `rebake`/`bake_step`, not the painted-panorama shortcut).
    fn bake_inputs() -> (Map, MaterialRegistry, CameraPose) {
        let map = re_world::test_room();
        let registry = crate::registry_for_map(&map);
        let cam = CameraPose {
            pos: Vec3::new(3.0, 3.0, 0.5),
            yaw: 0.3,
            pitch: 0.0,
        };
        (map, registry, cam)
    }

    /// Snapshot every ring cell of a cache's SERVED panorama as raw cells so two
    /// bakes can be compared byte-for-byte.
    fn snapshot(fc: &FarCache) -> Vec<Option<Cell>> {
        let (cols, rows) = fc.panorama.dims();
        let mut out = Vec::with_capacity(cols * rows);
        for col in 0..cols {
            for row in 0..rows {
                out.push(fc.panorama.cell_raw(col, row).copied());
            }
        }
        out
    }

    #[test]
    fn incremental_bake_keeps_old_ring_until_swap() {
        let (map, registry, cam) = bake_inputs();
        let rows = 24usize;

        // Establish a FIRST served ring at a clearly different pose (one-shot).
        let mut fc = FarCache::new(rows);
        let first_pose = CameraPose {
            pos: Vec3::new(10.0, 10.0, 0.5),
            yaw: 1.7,
            pitch: 0.0,
        };
        fc.rebake(&map, &first_pose, &registry);
        assert!(fc.valid);
        let old_ring = snapshot(&fc);
        let old_base_yaw = fc.base_yaw;

        // Now drive an INCREMENTAL bake at the new pose. Until it completes, the
        // served ring + base pose must be UNCHANGED (no tear, no stall).
        let mut completed = false;
        let mut steps = 0;
        while !completed {
            let restart = false;
            completed = fc.bake_step(&map, &cam, &registry, restart);
            steps += 1;
            if !completed {
                assert!(fc.valid, "served ring stays valid mid-bake");
                assert!(fc.bake_in_progress(), "bake is in progress until swap");
                assert_eq!(snapshot(&fc), old_ring, "served ring unchanged mid-bake");
                assert_eq!(fc.base_yaw, old_base_yaw, "base pose unchanged mid-bake");
            }
            assert!(steps < 100, "bake must terminate");
        }
        // After completion the ring swapped and base pose tracks the NEW bake pose.
        assert!(!fc.bake_in_progress());
        assert_eq!(
            fc.base_yaw, cam.yaw,
            "completed bake records the start pose yaw"
        );
        assert!(
            snapshot(&fc) != old_ring,
            "the swapped ring is the freshly baked one"
        );
        // Expected number of chunks (480 / 64 = 8).
        let expected_steps = PANORAMA_COLS.div_ceil(BAKE_CHUNK_COLS);
        assert_eq!(steps, expected_steps, "one chunk per step");
    }

    #[test]
    fn incremental_bake_matches_one_shot_byte_for_byte() {
        let (map, registry, cam) = bake_inputs();
        let rows = 24usize;

        // One-shot reference.
        let mut one_shot = FarCache::new(rows);
        one_shot.rebake(&map, &cam, &registry);

        // Incremental: first establish a (different) valid ring, then incrementally
        // re-bake at the SAME `cam` pose. The result must be identical to one-shot.
        let mut incr = FarCache::new(rows);
        let other = CameraPose {
            pos: Vec3::new(7.0, 12.0, 0.5),
            yaw: 2.9,
            pitch: 0.0,
        };
        incr.rebake(&map, &other, &registry);
        let mut done = false;
        let mut guard = 0;
        while !done {
            done = incr.bake_step(&map, &cam, &registry, false);
            guard += 1;
            assert!(guard < 100);
        }

        assert_eq!(incr.base_yaw, one_shot.base_yaw);
        assert_eq!(incr.base_pos, one_shot.base_pos);
        assert_eq!(
            snapshot(&incr),
            snapshot(&one_shot),
            "completed incremental ring must equal a one-shot bake of the same pose"
        );
    }

    #[test]
    fn mid_bake_significant_move_restarts_from_new_pose() {
        let (map, registry, cam) = bake_inputs();
        let rows = 16usize;
        let mut fc = FarCache::new(rows);
        fc.rebake(&map, &cam, &registry);

        // Start an incremental bake at `cam`, do one chunk.
        fc.bake_step(&map, &cam, &registry, false);
        assert!(fc.bake_in_progress());
        let (start_yaw, _) = fc.bake_start_pose();
        assert_eq!(start_yaw, cam.yaw);

        // A significant mid-bake move with should_restart = true must reset the bake
        // to the NEW pose (start_col back to a fresh ring at the new yaw).
        let moved = CameraPose {
            pos: cam.pos,
            yaw: cam.yaw + 1.0,
            pitch: 0.0,
        };
        fc.bake_step(&map, &moved, &registry, true);
        let (new_start_yaw, _) = fc.bake_start_pose();
        assert_eq!(new_start_yaw, moved.yaw, "restart adopts the new pose");
    }

    #[test]
    fn far_horizon_tracks_pitch() {
        // The panorama is baked at pitch 0; pitching must slide the sampled far row
        // by the SAME shift the near cast uses (shared `pitch_row_shift`), so far +
        // near keep one horizon. Paint each ROW with a distinct glyph; a pitch shift
        // must change which panorama row a given view row reads.
        let rows = 20usize;
        let mut fc = FarCache::new(rows);
        for col in 0..PANORAMA_COLS {
            for row in 0..rows {
                let g = (b'A' + (row % 26) as u8) as char;
                fc.panorama.set(col, row, cell(g));
            }
        }
        fc.valid = true;

        let cols = 8usize;
        let probe_row = rows / 2;
        let mut sample_glyph = |pitch: f32| -> Option<u32> {
            let mut out = Layer::new(cols, rows, LayerBlend::Opaque);
            let cam = CameraPose {
                pos: Vec3::ZERO,
                yaw: 0.0,
                pitch,
            };
            fc.fill_far_layer(&cam, &mut out);
            out.get(cols / 2, probe_row).map(|c| c.glyph)
        };

        let level = sample_glyph(0.0).expect("level frame samples a row");
        let shift = crate::pitch_row_shift(0.3, rows);
        assert!(shift != 0, "test pitch must produce a non-zero row shift");
        let pitched = sample_glyph(0.3).expect("pitched frame samples a row");
        // At probe_row the level frame reads panorama row `probe_row`; the pitched
        // frame reads `probe_row - shift` → a different painted glyph.
        let expected = (b'A' + ((probe_row as i32 - shift) as usize % 26) as u8) as char;
        assert_eq!(pitched, u32::from(expected), "far horizon must track pitch");
        assert_ne!(
            pitched, level,
            "pitch must move the far backdrop vertically"
        );
    }
    // ── far glyph deadband ───────────────────────────────────────────────────

    /// Yaw whose centre ring column lands exactly on `ring + frac`, so every view
    /// column of an even-width view blends the same pair with the same `frac`.
    fn yaw_for_frac(ring: f32) -> f32 {
        (ring / PANORAMA_COLS as f32) * std::f32::consts::TAU
    }

    #[test]
    fn far_glyph_deadband_holds_the_glyph_across_a_micro_crossing() {
        // The churn case: `frac` hovering at the 0.5 sub-column boundary used to
        // flip EVERY far cell between the two candidate ring columns. Two fills at
        // 0.48 → 0.52 must now keep the FIRST fill's glyph (both inside the band).
        let mut fc = painted_cache(1);
        let cols = 10usize;
        let mut out = Layer::new(cols, 1, LayerBlend::Opaque);
        let probe = cols / 2; // view col 5 → ring pair (0, 1) at yaw≈0.

        let cam_a = CameraPose {
            pos: Vec3::ZERO,
            yaw: yaw_for_frac(0.48),
            pitch: 0.0,
        };
        fc.fill_far_layer(&cam_a, &mut out);
        let first = out.get(probe, 0).unwrap().glyph;
        assert_eq!(
            first,
            glyph_of(0),
            "frac 0.48 (no memory) takes the nearer column"
        );

        let cam_b = CameraPose {
            pos: Vec3::ZERO,
            yaw: yaw_for_frac(0.52),
            pitch: 0.0,
        };
        fc.fill_far_layer(&cam_b, &mut out);
        assert_eq!(
            out.get(probe, 0).unwrap().glyph,
            first,
            "0.48 → 0.52 crosses 0.5 by less than FAR_GLYPH_DEADBAND: hold the glyph"
        );
    }

    #[test]
    fn far_glyph_deadband_yields_to_a_decisive_crossing() {
        // The other half of the contract: a REAL turn must still switch. 0.48 → 0.65
        // is outside the band, so the glyph moves to the far column.
        let mut fc = painted_cache(1);
        let cols = 10usize;
        let mut out = Layer::new(cols, 1, LayerBlend::Opaque);
        let probe = cols / 2;

        let cam_a = CameraPose {
            pos: Vec3::ZERO,
            yaw: yaw_for_frac(0.48),
            pitch: 0.0,
        };
        fc.fill_far_layer(&cam_a, &mut out);
        assert_eq!(out.get(probe, 0).unwrap().glyph, glyph_of(0));

        let cam_b = CameraPose {
            pos: Vec3::ZERO,
            yaw: yaw_for_frac(0.65),
            pitch: 0.0,
        };
        fc.fill_far_layer(&cam_b, &mut out);
        assert_eq!(
            out.get(probe, 0).unwrap().glyph,
            glyph_of(1),
            "0.48 → 0.65 is a decisive crossing: the glyph must switch"
        );
    }

    #[test]
    fn far_glyph_deadband_still_blends_colour_continuously() {
        // Only the DISCRETE channel is damped — the sub-column colour blend must
        // keep sliding, otherwise a slow turn re-introduces the backdrop snap.
        let mut fc = colour_cache(1);
        let cols = 10usize;
        let mut out = Layer::new(cols, 1, LayerBlend::Opaque);
        let probe = cols / 2;
        let mut reds = Vec::new();
        for frac in [0.45_f32, 0.5, 0.55] {
            let cam = CameraPose {
                pos: Vec3::ZERO,
                yaw: yaw_for_frac(frac),
                pitch: 0.0,
            };
            fc.fill_far_layer(&cam, &mut out);
            reds.push(((out.get(probe, 0).unwrap().fg >> 16) & 0xFF) as i32);
        }
        assert!(
            reds[0] != reds[1] || reds[1] != reds[2],
            "colour must still interpolate inside the glyph deadband: {reds:?}"
        );
    }

    // ── world-anchored glyph memory ──────────────────────────────────────────

    /// Regression for a screen-anchored memory bug, exercised through the LIVE
    /// incremental path (`fill_far_layer` called once per frame, in a loop — never
    /// `composite_dump_once`): turning the camera through MANY SMALL steps must
    /// never leave a screen column's glyph biased by unrelated content it showed
    /// earlier in the turn.
    ///
    /// Setup: a fixed screen column (`probe`) sweeps through ring columns 0..=4 via
    /// ~17 separate per-frame fills (small steps), settling on the glyph for ring 5
    /// ('f'). Then the camera JUMPS straight to a ring pair it has NEVER shown
    /// before — (265, 266), glyphs ('f','g') — at the exact ambiguous midpoint of
    /// the deadband (`frac == 0.5`). Ring 265 happens to carry the SAME letter
    /// ('f') the sweep left behind (`painted_cache` cycles every 26 ring columns,
    /// so this kind of repeat is routine, not contrived).
    ///
    /// A memory keyed on the SCREEN cell would compare that leftover 'f' against
    /// ring 265's candidates, find a coincidental match, and hold 'f' — biasing a
    /// ring pair that has never been ambiguous before with history that belongs
    /// to a completely different, unrelated part of the world. That is precisely
    /// "one texture stretched over the view, not scrolling with the turn": the
    /// discrete channel would occasionally hang on content that has nothing to do
    /// with what should be there. The WORLD-anchored memory has no record at ring
    /// 265 yet, so it must fall back to the unbiased `frac >= 0.5 ⇒ take b` rule
    /// and show 'g'.
    #[test]
    fn far_glyph_memory_is_world_anchored_not_screen_anchored() {
        let mut fc = painted_cache(1);
        let cols = 10usize;
        let mut out = Layer::new(cols, 1, LayerBlend::Opaque);
        let probe = cols / 2; // ring_f == centre_ring exactly at this column.

        // Turn in small steps from ring 0 toward ring 5 — many real per-frame fills.
        let mut ring = 0.0f32;
        while ring < 5.2 {
            let cam = CameraPose {
                pos: Vec3::ZERO,
                yaw: yaw_for_frac(ring),
                pitch: 0.0,
            };
            fc.fill_far_layer(&cam, &mut out);
            ring += 0.31;
        }
        let after_sweep = out.get(probe, 0).unwrap().glyph;
        assert_eq!(
            after_sweep,
            glyph_of(5),
            "the sweep settles on ring 5's glyph ('f') at the probe column"
        );

        // The coincidental repeat this test relies on: 265 and 5 share a letter.
        assert_eq!(
            glyph_of(265),
            glyph_of(5),
            "265 % 26 == 5 % 26 (26-glyph cycle)"
        );
        assert_ne!(
            glyph_of(266),
            glyph_of(5),
            "the NEXT ring column must differ"
        );

        // Jump straight to the never-before-shown ring pair (265, 266) at the exact
        // ambiguous midpoint of the deadband.
        let jump = CameraPose {
            pos: Vec3::ZERO,
            yaw: yaw_for_frac(265.5),
            pitch: 0.0,
        };
        fc.fill_far_layer(&jump, &mut out);
        assert_eq!(
            out.get(probe, 0).unwrap().glyph,
            glyph_of(266),
            "a ring pair shown for the FIRST time must resolve by the unbiased \
             frac>=0.5 rule ('g'), not by a stale glyph the sweep left at this \
             SCREEN column for unrelated ring content — that stale-glyph bias is \
             exactly what world-anchoring the memory removes"
        );

        // And a cache with NO turn history at all resolves the identical ambiguous
        // jump the same way — proving the sweep's memory had no bearing here: the
        // ring pair's OWN history is what should decide this, and it has none.
        let mut fresh = painted_cache(1);
        fresh.fill_far_layer(&jump, &mut out);
        assert_eq!(
            out.get(probe, 0).unwrap().glyph,
            glyph_of(266),
            "a fresh cache (no unrelated sweep history) must resolve identically"
        );
    }

    /// A bake swap must forget the ring-keyed memory: the new panorama can put
    /// different content at any ring column, so carrying old glyphs forward would
    /// seed the new hysteresis with a stale, unrelated value (the bake-axis twin of
    /// the screen-anchor fix above).
    #[test]
    fn bake_swap_forgets_the_glyph_memory() {
        let (map, registry, cam) = bake_inputs();
        let rows = 8usize;
        let mut fc = FarCache::new(rows);
        fc.rebake(&map, &cam, &registry);
        // Force a specific memory entry, as a real fill would.
        fc.prev_glyph[0] = u32::from('Z');
        assert_eq!(fc.prev_glyph[0], u32::from('Z'));

        // A second one-shot bake (fresh content) must wipe it.
        let other = CameraPose {
            pos: Vec3::new(1.0, 1.0, 0.5),
            yaw: 1.1,
            pitch: 0.0,
        };
        fc.rebake(&map, &other, &registry);
        assert!(
            fc.prev_glyph.iter().all(|&g| g == FAR_GLYPH_NO_MEMORY),
            "a rebake must forget every ring cell's glyph memory"
        );
    }

    #[test]
    fn far_take_b_is_total_and_reduces_to_the_old_snap() {
        let none = FAR_GLYPH_NO_MEMORY;
        // Decisive both ways, with and without memory.
        assert!(far_take_b(0.9, none, 1, 2));
        assert!(
            far_take_b(0.9, 1, 1, 2),
            "a decisive crossing overrides memory"
        );
        assert!(!far_take_b(0.1, none, 1, 2));
        assert!(
            !far_take_b(0.1, 2, 1, 2),
            "a decisive crossing overrides memory"
        );
        // Inside the band with NO memory ⇒ the plain `frac < 0.5` rule (old behaviour).
        assert!(!far_take_b(0.49, none, 1, 2));
        assert!(far_take_b(0.5, none, 1, 2));
        // Inside the band, memory matches NEITHER candidate (panorama moved under us)
        // ⇒ fall back to the plain rule rather than freeze on a stale glyph.
        assert!(far_take_b(0.55, 99, 1, 2));
        assert!(!far_take_b(0.45, 99, 1, 2));
        // Identical candidates: nothing to hold, plain rule.
        assert!(far_take_b(0.55, 7, 7, 7));
        // Degenerate input is defined, never a panic.
        assert!(!far_take_b(f32::NAN, none, 1, 2));
    }
}
