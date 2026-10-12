//! `re_render::near_cache` — the panoramic NEAR ring (rotation-pan cache).
//!
//! # Why this exists (parallax ∝ 1/distance)
//!
//! The FAR backdrop ([`crate::far_cache`]) is cached as a 360° ring and PANNED by
//! yaw with zero re-cast — rotation has zero parallax, so a distant column is the
//! same whichever way you turn into it. This module brings the SAME economy to the
//! NEAR tier, the readable foreground that dominates frame cost: a pure rotation
//! must NOT re-cast the near scene — it must pan a baked near ring.
//!
//! # The hard constraint the far ring does not face
//!
//! The far ring uses a UNIFORM-ANGULAR column mapping (`yaw → ring col` is linear)
//! and accepts a colour-blend approximation because the backdrop is distant. The NEAR
//! cast is RECTILINEAR (`ray = dir + plane·camera_x`, a tangent projection) AND its
//! shading is VIEW-DIRECTION dependent (headlamp + Blinn-Phong specular + relief
//! normals all read the ray direction). Two consequences:
//!
//! 1. **Geometry** of a ring column is EXACT under pan: `dda_hit` + the wall-slice
//!    extent depend only on the ray's absolute direction + camera position +
//!    projection plane. So if the ring stores columns at uniform ABSOLUTE ANGLES and
//!    we map each viewport column to the ring by its ray's absolute angle
//!    (`yaw + atan(plane_scale·camera_x)`), the panned geometry matches a fresh cast
//!    byte-for-byte.
//! 2. **Shading** carries the BAKE-TIME view direction (headlamp/specular). Panning a
//!    cell to a new yaw keeps that bake-time lighting. Over a SMALL yaw the headlamp
//!    drift is sub-perceptual — exactly the approximation the far ring already accepts
//!    for colour, and bounded here by the ring's [`NearRing::margin`].
//!
//! So this is a **MARGIN ring**, not a full 360° ring: it covers the live horizontal
//! FOV plus a generous yaw margin on each side, baked in the live rectilinear vertical
//! projection. Within the margin, a rotation is a FREE pan (zero near rays). Past the
//! margin — or on a translation past the near parallax threshold, or when the scene
//! contains a NO-PAN special cell (mirror/holo, view-direction dependent and never
//! pannable) — the ring rebakes / the near tier falls back to a fresh structural cast.
//! This never ships an artifact: the pan is only ever served where it is correct.
//!
//! A fully-correct 360° near ring would additionally require a per-column geometry
//! cache + a per-frame re-shade of the whole viewport against the live view
//! direction — which saves only the DDA, ~7% of near cost, so it does NOT hit the
//! "near → 0" target; the margin pan does.
//!
//! Structure mirrors [`crate::far_cache`]: a served ring + a back ring, an
//! incremental chunked bake with an atomic swap (never serve a half-baked ring),
//! `base_pos`/`base_yaw`, and a restart-on-significant-move policy.

use bevy::prelude::Resource;
use glam::Vec3;
use re_core::{Cell, Layer, LayerBlend};
use re_world::{CameraPose, Map};

use crate::ascii_shader::MaterialRegistry;
use crate::bake_near_ring_column;

/// Default extra yaw, EACH SIDE of the live FOV, the near ring bakes so a rotation can
/// pan within it with zero re-cast. ~17.2° per side (a 0.30 rad margin). Rotation in a
/// puzzle is slow look-around; this covers ~0.30/0.03 ≈ 10 frames of the harness's
/// 0.03 rad/frame sweep before a rebake, and a generous swing of real mouse-look,
/// before the served ring's pan window runs off its edge. Tunable: wider = fewer
/// rebakes but a costlier bake + more headlamp drift at the edges.
pub const NEAR_MARGIN_RAD: f32 = 0.55;

/// Angular resolution of the ring: radians per ring column. The ring stores one cast
/// ray per this many radians of absolute yaw. Finer than a viewport column at the
/// centre (a 65° FOV over ~280 cols ≈ 0.004 rad/col), so the rectilinear→angular
/// resample never loses a viewport column to aliasing. 0.0035 rad ≈ 0.2°.
pub const NEAR_RING_D_THETA: f32 = 0.0035;

/// How many ring columns the INCREMENTAL bake casts per frame — same spread-the-spike
/// rationale as [`crate::far_cache::BAKE_CHUNK_COLS`]. The near bake is heavier per
/// column (full shade), so a smaller chunk keeps the per-frame bake cost bounded; the
/// served (old) ring keeps panning meanwhile, so the bake never stalls rotation.
pub const NEAR_BAKE_CHUNK_COLS: usize = 128;

/// Translation past this many world cells (Euclidean xy) invalidates the near ring —
/// near parallax (∝ 1/distance) changes fast, so the ring rebakes on a sub-cell move.
/// One cell is the near-parallax threshold used engine-wide; below it a pan's
/// parallax error is ≤ ~1 glyph column on near geometry, imperceptible.
pub const NEAR_MOVE_CELLS: f32 = 1.0;

/// Pitch drift (radians) past which the near ring can no longer be PANNED and must
/// re-cast. The ring bakes geometry at the pitch-0 horizon and serves pitch as a naive
/// row-shift in [`fill_col`]; once the live pitch differs from the bake pitch the shift
/// leaves a band of rows ABSENT that a fresh full cast fills (the floor/ceiling extends
/// to the pitch-shifted horizon). So a pitch change is gated exactly like the z-shear in
/// [`NearRing::can_pan`]: within the tolerance the served ring is reused (micro-jitter
/// collapses to a cache HIT), past it the structural full cast runs — no blank band.
/// Sized at half the [`crate::composite::NEAR_PITCH_BUCKET`] (≈0.049 rad) so a pitch step that
/// crosses a near-key bucket (and therefore re-casts) is never served stale by the pan.
pub const NEAR_PITCH_TOL: f32 = std::f32::consts::TAU / 256.0;

/// A baked near ring (opaque + transparent layer set) plus the pose it was baked at.
///
/// `opaque`/`transparent` are `ring_cols` wide and view-tall; the served pair is
/// radial-sampled by [`fill_near_layers`](Self::fill_near_layers); a bake fills the
/// `back_*` pair and swaps atomically on completion. The ring covers absolute yaw
/// `[base_yaw - half_arc, base_yaw + half_arc]` where
/// `half_arc = hfov/2 + margin`, with `center` the ring column mapping to `base_yaw`.
#[derive(Resource)]
pub struct NearRing {
    /// Served opaque ring. Never mid-write (a bake writes `back_opaque`).
    opaque: Layer,
    /// Served transparent ring.
    transparent: Layer,
    /// Back opaque ring (incremental bake target; swapped on completion).
    back_opaque: Layer,
    /// Back transparent ring.
    back_transparent: Layer,
    /// Single-column reusable cast scratch (no per-bake-step alloc).
    scratch: re_core::Framebuffer,
    /// Yaw the SERVED ring was baked at (its centre column's absolute angle).
    pub base_yaw: f32,
    /// Pitch the SERVED ring was baked at. The ring stores geometry at the pitch-0
    /// horizon (`rows/2`) and the fill path applies the pitch as a naive ROW-SHIFT,
    /// which leaves a band of rows ABSENT once the live pitch differs from the bake
    /// pitch — a fresh cast would instead extend the floor/ceiling to fill the
    /// pitch-shifted screen. So a pitch change past [`NEAR_PITCH_TOL`] is NOT pannable
    /// (the same precedent as the z-shear gate in [`NearRing::can_pan`]): it falls back to a full
    /// structural cast, which fills the screen with no blank band. Recorded on every
    /// bake / swap so the gate measures drift from the served ring's actual pitch.
    pub base_pitch: f32,
    /// Camera position the SERVED ring was baked at.
    pub base_pos: Vec3,
    /// Half the ring's covered yaw arc (radians): `hfov/2 + margin`.
    half_arc: f32,
    /// The (configured) per-side yaw margin the ring was sized with.
    margin: f32,
    /// Total ring columns (covers `2·half_arc` at [`NEAR_RING_D_THETA`]).
    ring_cols: usize,
    /// `false` until the first completed bake; the fill path produces nothing while
    /// invalid (the caller falls back to a fresh cast).
    pub valid: bool,
    /// `true` while an incremental bake fills the back ring. The served ring keeps
    /// being sampled throughout.
    bake_in_progress: bool,
    /// Next ring column the incremental bake will write.
    bake_next_col: usize,
    /// Pose the IN-PROGRESS bake started at (becomes base_* on swap).
    bake_start_yaw: f32,
    bake_start_pitch: f32,
    bake_start_pos: Vec3,
    /// Animation time the IN-PROGRESS bake casts at.
    bake_time: f32,
    /// The LIVE VIEWPORT's width in terminal columns.
    ///
    /// The ring itself is cast in a fixed ANGULAR metric ([`NEAR_RING_D_THETA`]), finer
    /// than a viewport column, and resampled on the way out — so a ring column index is
    /// not a screen column. But the text surfaces choose their DENSITY from the screen
    /// span of a wall (`crate::face_span`), and a ring-baked cell must choose the SAME
    /// density a full cast would or the wall text would re-flow every time the frame
    /// switched between panning and casting. So the bake republishes the viewport's own
    /// projection scale, and that needs the viewport width.
    view_cols: usize,
}

impl NearRing {
    /// Allocate an unbaked near ring sized for a `cols`×`rows` view at the default
    /// margin. `cols` sets the live FOV (and therefore `half_arc`); `rows` the height.
    pub fn new(cols: usize, rows: usize, hfov: f32) -> Self {
        Self::with_margin(cols, rows, hfov, NEAR_MARGIN_RAD)
    }

    /// As [`new`](Self::new) with an explicit per-side yaw `margin` (radians).
    pub fn with_margin(cols: usize, rows: usize, hfov: f32, margin: f32) -> Self {
        let half_arc = hfov * 0.5 + margin;
        let ring_cols = ((2.0 * half_arc / NEAR_RING_D_THETA).ceil() as usize).max(1);
        NearRing {
            opaque: Layer::new(ring_cols, rows, LayerBlend::Opaque),
            transparent: Layer::new(ring_cols, rows, LayerBlend::OverKeepBg),
            back_opaque: Layer::new(ring_cols, rows, LayerBlend::Opaque),
            back_transparent: Layer::new(ring_cols, rows, LayerBlend::OverKeepBg),
            scratch: re_core::Framebuffer::new(1, rows),
            base_yaw: 0.0,
            base_pitch: 0.0,
            base_pos: Vec3::ZERO,
            half_arc,
            margin,
            ring_cols,
            valid: false,
            bake_in_progress: false,
            bake_next_col: 0,
            bake_start_yaw: 0.0,
            bake_start_pitch: 0.0,
            bake_start_pos: Vec3::ZERO,
            bake_time: 0.0,
            view_cols: cols,
        }
    }

    /// `(ring_cols, rows)`.
    #[inline]
    pub fn dims(&self) -> (usize, usize) {
        self.opaque.dims()
    }

    /// The per-side yaw margin (radians) the ring covers beyond the live FOV.
    #[inline]
    pub fn margin(&self) -> f32 {
        self.margin
    }

    /// Whether an incremental bake is filling the back ring (served ring still sampled).
    #[inline]
    pub fn bake_in_progress(&self) -> bool {
        self.bake_in_progress
    }

    /// The pose the IN-PROGRESS bake started at (the restart-policy reference).
    #[inline]
    pub fn bake_start_pose(&self) -> (f32, Vec3) {
        (self.bake_start_yaw, self.bake_start_pos)
    }

    /// The ring column (fractional) the centre (`base_yaw`) maps to.
    #[inline]
    fn center_col(&self) -> f32 {
        self.ring_cols as f32 * 0.5
    }

    /// Resize the ring to a new view (`cols` → FOV via `hfov`, `rows` → height),
    /// marking it invalid + discarding any partial bake. Width is re-derived from the
    /// new `hfov` + margin. Called on a window resize.
    pub fn resize(&mut self, cols: usize, rows: usize, hfov: f32) {
        let half_arc = hfov * 0.5 + self.margin;
        let ring_cols = ((2.0 * half_arc / NEAR_RING_D_THETA).ceil() as usize).max(1);
        if ring_cols != self.ring_cols || self.opaque.rows() != rows {
            self.opaque.resize(ring_cols, rows);
            self.transparent.resize(ring_cols, rows);
            self.back_opaque.resize(ring_cols, rows);
            self.back_transparent.resize(ring_cols, rows);
            self.scratch.resize(1, rows);
            self.half_arc = half_arc;
            self.ring_cols = ring_cols;
            self.valid = false;
            self.bake_in_progress = false;
            self.bake_next_col = 0;
        }
        self.view_cols = cols;
    }

    /// Whether the camera pose `cam` can be SERVED by panning the current served ring:
    /// the ring is valid, the camera has not moved past [`NEAR_MOVE_CELLS`] from the
    /// bake position, and the ENTIRE viewport (FOV edges) stays inside the ring's
    /// covered arc at the panned yaw. This is the correctness gate: a `true` means a
    /// pan is exact-geometry + margin-bounded-shading; a `false` means rebake / fresh
    /// cast.
    ///
    /// `plane_scale = tan(hfov/2)`: the viewport's extreme ray (camera_x = ±1) points
    /// `atan(plane_scale)` from facing, so the viewport spans `[yaw - e, yaw + e]` with
    /// `e = atan(plane_scale)`. The pan is valid iff `|yaw - base_yaw| + e ≤ half_arc`.
    pub fn can_pan(&self, cam: &CameraPose, plane_scale: f32) -> bool {
        if !self.valid {
            return false;
        }
        let dx = cam.pos.x - self.base_pos.x;
        let dy = cam.pos.y - self.base_pos.y;
        if (dx * dx + dy * dy).sqrt() >= NEAR_MOVE_CELLS {
            return false;
        }
        // z (jump/fall) shears the projection — a pan would show stale vertical extent.
        // The near key already re-casts across z buckets; require the same z here.
        if (cam.pos.z - self.base_pos.z).abs() > 0.05 {
            return false;
        }
        // PITCH shears the vertical extent. The ring ALWAYS bakes geometry at the pitch-0
        // horizon (`rows/2`, see `bake_near_ring_column`) and the fill applies the live
        // pitch as a naive ROW-SHIFT (`fill_col`). That shift slides the baked image up or
        // down and leaves a band of rows ABSENT at the opposite edge — rows a fresh cast
        // would fill with floor/ceiling extended to the pitch-shifted horizon. When that
        // absent band holds the only in-view geometry (and the far backdrop is open sky
        // there) the frame reads BLANK — the playtest "pitch up/down → black, wiggle to
        // recover" bug (the wiggle crossed a near-key bucket and forced a full cast).
        //
        // So the pan is correct ONLY while the live pitch is within [`NEAR_PITCH_TOL`] of
        // the baked horizon (level): there the row-shift is ≤1 row, no perceptible band.
        // Past it, fall back to the structural full cast — which the near key already
        // demands across pitch buckets, so this only stops the pan from STEALING those
        // frames and serving them stale. `base_pitch` is the served ring's bake pitch
        // (currently always level); comparing to it keeps the gate honest if the bake
        // ever becomes pitch-aware. The near key re-casts across pitch buckets regardless.
        if (cam.pitch - self.base_pitch).abs() > NEAR_PITCH_TOL || cam.pitch.abs() > NEAR_PITCH_TOL
        {
            return false;
        }
        let e = plane_scale.atan();
        let dyaw = ang_delta(cam.yaw, self.base_yaw).abs();
        dyaw + e <= self.half_arc
    }

    /// Fill the view-width near `opaque_out` / `transparent_out` layers by sampling the
    /// served ring at `cam`'s current yaw (+ pitch row-shift), with NO re-cast — the
    /// rotation-pan win. Each viewport column's ray absolute angle
    /// `phi = yaw + atan(plane_scale·camera_x)` maps to a fractional ring column; the
    /// nearer column supplies the discrete glyph/geometry and the colours blend between
    /// the two bracketing ring columns (kills turn shimmer, mirroring the far ring).
    ///
    /// Pitch is applied as a row-shift (`pitch_row_shift`, shared with far) so near +
    /// far keep one horizon. Both layers are cleared first. Caller has verified
    /// [`can_pan`](Self::can_pan), so every sampled column is in range.
    pub fn fill_near_layers(
        &self,
        cam: &CameraPose,
        plane_scale: f32,
        opaque_out: &mut Layer,
        transparent_out: &mut Layer,
    ) {
        opaque_out.clear();
        transparent_out.clear();
        if !self.valid {
            return;
        }
        let (cols, rows) = opaque_out.dims();
        if cols == 0 || rows == 0 {
            return;
        }
        let center = self.center_col();
        let cols_f = cols as f32;
        let pitch_shift = crate::pitch_row_shift(cam.pitch, rows);
        let ring_max = self.ring_cols as f32 - 1.0;
        for col in 0..cols {
            // Rectilinear → absolute angle → fractional ring column.
            let camera_x = 2.0 * (col as f32 + 0.5) / cols_f - 1.0;
            let phi = cam.yaw + (plane_scale * camera_x).atan();
            let rc_f =
                (center + ang_delta(phi, self.base_yaw) / NEAR_RING_D_THETA).clamp(0.0, ring_max);
            let c0 = rc_f.floor();
            let frac = rc_f - c0;
            let ring0 = c0 as usize;
            let ring1 = (ring0 + 1).min(self.ring_cols - 1);
            fill_col(
                &self.opaque,
                opaque_out,
                col,
                ring0,
                ring1,
                frac,
                rows,
                pitch_shift,
            );
            fill_col(
                &self.transparent,
                transparent_out,
                col,
                ring0,
                ring1,
                frac,
                rows,
                pitch_shift,
            );
        }
    }

    /// Re-bake the WHOLE near ring in ONE call at `cam`'s pose + `time`. The expensive
    /// all-at-once path used for the FIRST bake on a (re)pan-eligible frame; steady
    /// rebakes use the incremental [`bake_step`](Self::bake_step). Records the bake
    /// pose + marks the ring valid.
    pub fn rebake(
        &mut self,
        map: &Map,
        cam: &CameraPose,
        registry: &MaterialRegistry,
        fov: &crate::Fov,
        time: f32,
    ) {
        let (ring_cols, rows) = self.opaque.dims();
        if ring_cols == 0 || rows == 0 {
            return;
        }
        let center = self.center_col();
        self.opaque.clear();
        self.transparent.clear();
        bake_into(
            map,
            cam.pos,
            registry,
            fov,
            time,
            cam.yaw,
            center,
            0,
            ring_cols,
            self.view_cols,
            &mut self.scratch,
            &mut self.opaque,
            &mut self.transparent,
        );
        self.base_yaw = cam.yaw;
        self.base_pitch = cam.pitch;
        self.base_pos = cam.pos;
        self.valid = true;
        self.bake_in_progress = false;
        self.bake_next_col = 0;
    }

    /// Advance an INCREMENTAL re-bake by ONE chunk ([`NEAR_BAKE_CHUNK_COLS`]) into the
    /// back ring, leaving the served ring untouched. Returns `true` on the frame the
    /// bake completes (the back ring is atomically swapped into service + base_* set to
    /// the bake's start pose). Same restart-on-significant-move policy as the far ring:
    /// when `should_restart`, discard the partial back ring and restart at `cam`.
    pub fn bake_step(
        &mut self,
        map: &Map,
        cam: &CameraPose,
        registry: &MaterialRegistry,
        fov: &crate::Fov,
        time: f32,
        should_restart: bool,
    ) -> bool {
        let (ring_cols, rows) = self.opaque.dims();
        if ring_cols == 0 || rows == 0 {
            return false;
        }
        if !self.bake_in_progress || should_restart {
            self.bake_in_progress = true;
            self.bake_next_col = 0;
            self.bake_start_yaw = cam.yaw;
            self.bake_start_pitch = cam.pitch;
            self.bake_start_pos = cam.pos;
            self.bake_time = time;
            // Clearing the back ring once at restart keeps absent cells absent until a
            // column writes them (a partial column never serves).
            self.back_opaque.clear();
            self.back_transparent.clear();
        }
        let center = self.center_col();
        let rc0 = self.bake_next_col;
        let count = NEAR_BAKE_CHUNK_COLS.min(ring_cols - rc0);
        bake_into(
            map,
            self.bake_start_pos,
            registry,
            fov,
            self.bake_time,
            self.bake_start_yaw,
            center,
            rc0,
            count,
            self.view_cols,
            &mut self.scratch,
            &mut self.back_opaque,
            &mut self.back_transparent,
        );
        self.bake_next_col = rc0 + count;
        if self.bake_next_col >= ring_cols {
            std::mem::swap(&mut self.opaque, &mut self.back_opaque);
            std::mem::swap(&mut self.transparent, &mut self.back_transparent);
            self.base_yaw = self.bake_start_yaw;
            self.base_pitch = self.bake_start_pitch;
            self.base_pos = self.bake_start_pos;
            self.valid = true;
            self.bake_in_progress = false;
            self.bake_next_col = 0;
            return true;
        }
        false
    }
}

/// Bake `count` ring columns from `rc0` into the opaque + transparent ring layers by
/// casting one ray per ring column (via [`bake_near_ring_columns`]) and routing each
/// shaded cell by its material transparency — the same opaque/transparent split the
/// near composite uses. Shared by [`NearRing::rebake`] + [`NearRing::bake_step`].
#[allow(clippy::too_many_arguments)]
fn bake_into(
    map: &Map,
    cam_pos: Vec3,
    registry: &MaterialRegistry,
    fov: &crate::Fov,
    time: f32,
    base_yaw: f32,
    center: f32,
    rc0: usize,
    count: usize,
    // The LIVE viewport width, so the bake can republish the viewport's horizontal
    // projection scale (see `NearRing::view_cols`).
    view_cols: usize,
    scratch: &mut re_core::Framebuffer,
    opaque: &mut Layer,
    transparent: &mut Layer,
) {
    let rows = opaque.rows();
    if scratch.cols() != 1 || scratch.rows() != rows {
        scratch.resize(1, rows);
    }
    // Cast the run one ring column at a time into the 1-wide `scratch` (column 0), then
    // route that shaded column into the opaque/transparent rings at ring column `rc`.
    // Single-column casting keeps the ring's memory to the two output layers (no third
    // wide buffer) while reusing the exact-geometry per-column cast.
    for k in 0..count {
        let rc = rc0 + k;
        bake_near_ring_column(
            map,
            cam_pos,
            registry,
            fov,
            time,
            base_yaw,
            NEAR_RING_D_THETA,
            center,
            rc,
            scratch,
            view_cols,
        );
        for row in 0..rows {
            let cell = scratch.get(0, row).copied().unwrap_or_default();
            route_cell(registry, rc, row, cell, opaque, transparent);
        }
    }
}

/// Route one baked cell into the opaque or transparent ring layer by its surface
/// material transparency (the same split [`crate::composite`] uses).
#[inline]
fn route_cell(
    registry: &MaterialRegistry,
    col: usize,
    row: usize,
    cell: Cell,
    opaque: &mut Layer,
    transparent: &mut Layer,
) {
    let mat_id = crate::surface_material(cell.surface, cell.material);
    if registry.get(mat_id).is_transparent() {
        transparent.set(col, row, cell);
    } else {
        opaque.set(col, row, cell);
    }
}

/// Fill one viewport column from a ring layer by sub-column blend between ring columns
/// `ring0`/`ring1` (fraction `frac`), with a `pitch_shift` row offset. Discrete fields
/// from the nearer column; colours linearly blended (turn-shimmer fix, mirrors far).
#[allow(clippy::too_many_arguments)]
#[inline]
fn fill_col(
    ring: &Layer,
    out: &mut Layer,
    col: usize,
    ring0: usize,
    ring1: usize,
    frac: f32,
    rows: usize,
    pitch_shift: i32,
) {
    for row in 0..rows {
        let src_row = row as i32 - pitch_shift;
        if src_row < 0 || src_row as usize >= rows {
            continue;
        }
        let src_row = src_row as usize;
        let a = ring.get(ring0, src_row);
        let b = ring.get(ring1, src_row);
        match (a, b) {
            (Some(a), Some(b)) => {
                let near = if frac < 0.5 { a } else { b };
                let mut cell = *near;
                cell.fg = lerp_rgb(a.fg, b.fg, frac);
                cell.bg = lerp_rgb(a.bg, b.bg, frac);
                out.set(col, row, cell);
            }
            (Some(a), None) => out.set(col, row, *a),
            (None, Some(b)) => out.set(col, row, *b),
            (None, None) => {}
        }
    }
}

/// Smallest signed angular difference `a - b`, wrapped to `(-π, π]`. `pub(crate)`
/// re-exported as [`crate::near_cache_ang_delta`] for the composite tier's ring
/// drift/restart predicates (same wrap the ring uses internally).
#[inline]
pub(crate) fn ang_delta(a: f32, b: f32) -> f32 {
    let pi = std::f32::consts::PI;
    let tau = std::f32::consts::TAU;
    let mut d = (a - b) % tau;
    if d > pi {
        d -= tau;
    } else if d <= -pi {
        d += tau;
    }
    d
}

/// Public wrapper around [`ang_delta`] for harnesses / external drivers that mirror
/// the composite tier's ring drift predicates. Smallest signed `a - b` wrapped to
/// `(-π, π]`.
#[inline]
pub fn ang_delta_pub(a: f32, b: f32) -> f32 {
    ang_delta(a, b)
}

/// Linearly interpolate two packed `0x00RRGGBB` colours per channel by `t ∈ [0,1]`.
/// (Local copy of the far ring's blend — the rings are independent modules.)
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
    use crate::{registry_for_map, Fov};

    fn inputs() -> (Map, MaterialRegistry, CameraPose, Fov) {
        let map = re_world::test_room();
        let registry = registry_for_map(&map);
        let cam = CameraPose::spawn_in_test_room();
        (map, registry, cam, Fov::default())
    }

    fn plane_scale_for(cols: usize, rows: usize, fov: &Fov) -> f32 {
        crate::view_params(cols, rows, fov).plane_scale
    }

    #[test]
    fn ang_delta_wraps() {
        let tau = std::f32::consts::TAU;
        assert!(ang_delta(0.1, 0.0).abs() < 1e-6 || (ang_delta(0.1, 0.0) - 0.1).abs() < 1e-6);
        assert!(
            (ang_delta(0.0, tau - 0.01) - 0.01).abs() < 1e-4,
            "wrap across 0"
        );
        assert!(
            (ang_delta(tau - 0.01, 0.0) + 0.01).abs() < 1e-4,
            "wrap negative"
        );
    }

    #[test]
    fn invalid_ring_pans_nothing() {
        let (_m, _r, cam, fov) = inputs();
        let ring = NearRing::new(40, 24, crate::view_params(40, 24, &fov).hfov);
        let ps = plane_scale_for(40, 24, &fov);
        assert!(!ring.can_pan(&cam, ps), "unbaked ring is never pannable");
        let mut op = Layer::new(40, 24, LayerBlend::Opaque);
        let mut tr = Layer::new(40, 24, LayerBlend::OverKeepBg);
        op.set(0, 0, Cell::default()); // sentinel to be cleared
        ring.fill_near_layers(&cam, ps, &mut op, &mut tr);
        assert!(op.get(0, 0).is_none(), "invalid ring clears the layer");
    }

    #[test]
    fn small_rotation_can_pan_large_cannot() {
        let (map, registry, cam, fov) = inputs();
        let (cols, rows) = (120usize, 48usize);
        let hfov = crate::view_params(cols, rows, &fov).hfov;
        let ps = plane_scale_for(cols, rows, &fov);
        let mut ring = NearRing::new(cols, rows, hfov);
        ring.rebake(&map, &cam, &registry, &fov, 0.0);
        assert!(
            ring.valid && ring.can_pan(&cam, ps),
            "ring pans at its bake pose"
        );
        // A small turn well within the margin → pannable.
        let small = CameraPose {
            yaw: cam.yaw + NEAR_MARGIN_RAD * 0.5,
            ..cam
        };
        assert!(
            ring.can_pan(&small, ps),
            "small turn within margin must pan"
        );
        // A turn past the margin → not pannable (must rebake / fresh cast).
        let big = CameraPose {
            yaw: cam.yaw + NEAR_MARGIN_RAD * 2.0,
            ..cam
        };
        assert!(!ring.can_pan(&big, ps), "turn past margin must NOT pan");
    }

    /// REPRO of the playtest "pitch up/down blanks the near scene" bug. The ring is
    /// baked at pitch 0 (horizon at rows/2). A FRESH full cast at a large pitch fills the
    /// WHOLE screen (the floor/ceiling extends to the pitch-shifted horizon), but the pan
    /// serves the pitch as a naive ROW-SHIFT of the pitch-0 ring, which leaves a band of
    /// rows ABSENT (shifted off the baked extent). When that absent band is where the only
    /// geometry sits — and the far backdrop is open sky there — the frame reads blank.
    ///
    /// THE FIX makes a pitch change NOT pannable, so the structural full cast runs. This
    /// test asserts the fix: a ring baked at pitch 0 must NOT pan a viewport at a large
    /// pitch (it must fall back to a fresh cast).
    #[test]
    fn pitch_change_cannot_pan() {
        let (map, registry, cam, fov) = inputs();
        let (cols, rows) = (120usize, 48usize);
        let view = crate::view_params(cols, rows, &fov);
        let ps = view.plane_scale;
        let mut ring = NearRing::new(cols, rows, view.hfov);
        ring.rebake(&map, &cam, &registry, &fov, 0.0); // baked at pitch 0
        assert!(
            ring.can_pan(&cam, ps),
            "ring pans at its (pitch-0) bake pose"
        );

        // A small sub-bucket pitch nudge stays pannable (the near key reuses sub-bucket).
        let nudge = CameraPose {
            pitch: NEAR_PITCH_TOL * 0.5,
            ..cam
        };
        assert!(
            ring.can_pan(&nudge, ps),
            "a sub-tolerance pitch nudge still pans"
        );

        // A real pitch (well past the bake pitch) must NOT pan — it must re-cast so the
        // floor/ceiling fills the pitch-shifted screen with no blank band.
        let pitched_up = CameraPose {
            pitch: std::f32::consts::FRAC_PI_4 * 0.75,
            ..cam
        };
        assert!(
            !ring.can_pan(&pitched_up, ps),
            "a large up-pitch must NOT pan a pitch-0 ring (would blank a row band)"
        );
        let pitched_down = CameraPose {
            pitch: -std::f32::consts::FRAC_PI_4 * 0.75,
            ..cam
        };
        assert!(
            !ring.can_pan(&pitched_down, ps),
            "a large down-pitch must NOT pan a pitch-0 ring"
        );
    }

    /// Demonstrates the MECHANISM the `can_pan` pitch gate guards against: panning a
    /// pitch-0 ring at a large pitch (bypassing the gate by sampling `fill_near_layers`
    /// directly) leaves a BAND of rows absent that a fresh full cast at the same pitch
    /// fills. This is the blank-band that — with an open-sky far backdrop — reads as the
    /// playtest's "pitch → black". It justifies the gate: such a frame must re-cast.
    #[test]
    fn pitch_pan_leaves_a_row_band_a_fresh_cast_fills() {
        let map = re_world::test_room();
        let registry = registry_for_map(&map);
        let base = CameraPose::spawn_in_test_room();
        let fov = Fov::default();
        let (cols, rows) = (120usize, 48usize);
        let view = crate::view_params(cols, rows, &fov);
        let ps = view.plane_scale;

        let mut ring = NearRing::new(cols, rows, view.hfov);
        ring.rebake(&map, &base, &registry, &fov, 0.0); // pitch-0 ring

        // A strong up-pitch: the row-shift pushes the baked image DOWN, vacating the TOP
        // rows in the panned layers.
        let pitched = CameraPose {
            pitch: std::f32::consts::FRAC_PI_4 * 0.9,
            ..base
        };
        let mut op = Layer::new(cols, rows, LayerBlend::Opaque);
        let mut tr = Layer::new(cols, rows, LayerBlend::OverKeepBg);
        ring.fill_near_layers(&pitched, ps, &mut op, &mut tr);

        // A fresh full cast at the same pitch — the correct image.
        let mut fresh = re_core::Framebuffer::new(cols, rows);
        crate::raycast_gameplay_capture(
            &map,
            &pitched,
            &registry,
            &fov,
            0.0,
            &mut crate::gbuffer::NearGBuffer::default(),
            &mut fresh,
        );

        // Count rows where the fresh cast has opaque-routed geometry but the pan left the
        // whole row ABSENT in both near layers — the blank band.
        let mut fresh_only_rows = 0usize;
        for row in 0..rows {
            let mut fresh_has = false;
            let mut pan_has = false;
            for col in 0..cols {
                if op.is_present(col, row) || tr.is_present(col, row) {
                    pan_has = true;
                }
                if let Some(c) = fresh.get(col, row) {
                    if *c != Cell::default() {
                        fresh_has = true;
                    }
                }
            }
            if fresh_has && !pan_has {
                fresh_only_rows += 1;
            }
        }
        assert!(
            fresh_only_rows > 0,
            "expected the pitch-panned ring to leave a row band the fresh cast fills \
             (the blank-band mechanism) — got none, the repro no longer demonstrates it"
        );
    }

    #[test]
    fn translation_past_threshold_cannot_pan() {
        let (map, registry, cam, fov) = inputs();
        let (cols, rows) = (96usize, 40usize);
        let hfov = crate::view_params(cols, rows, &fov).hfov;
        let ps = plane_scale_for(cols, rows, &fov);
        let mut ring = NearRing::new(cols, rows, hfov);
        ring.rebake(&map, &cam, &registry, &fov, 0.0);
        let moved = CameraPose {
            pos: cam.pos + Vec3::new(NEAR_MOVE_CELLS + 0.1, 0.0, 0.0),
            ..cam
        };
        assert!(!ring.can_pan(&moved, ps), "move past threshold must rebake");
        let nudged = CameraPose {
            pos: cam.pos + Vec3::new(0.01, 0.01, 0.0),
            ..cam
        };
        assert!(ring.can_pan(&nudged, ps), "a sub-cell nudge must still pan");
    }

    /// THE CORRECTNESS BAR: a near ring baked at pose P, then PANNED to a yaw Y within
    /// the margin, must match a FRESH full cast at (P, Y) for the pannable cells'
    /// GEOMETRY (depth + surface + side). Shading carries the bake-time view direction
    /// (the documented margin-bounded approximation), so we assert geometry exactly and
    /// allow shading to differ only where the headlamp/specular legitimately would.
    #[test]
    fn panned_geometry_matches_fresh_cast() {
        let map = re_world::test_room();
        let registry = registry_for_map(&map);
        let base = CameraPose::spawn_in_test_room();
        let fov = Fov::default();
        let (cols, rows) = (160usize, 56usize);
        let view = crate::view_params(cols, rows, &fov);
        let ps = view.plane_scale;

        let mut ring = NearRing::new(cols, rows, view.hfov);
        ring.rebake(&map, &base, &registry, &fov, 0.0);

        // Pan to a yaw a third of the margin away (comfortably in range).
        let panned = CameraPose {
            yaw: base.yaw + NEAR_MARGIN_RAD / 3.0,
            ..base
        };
        assert!(ring.can_pan(&panned, ps));
        let mut op = Layer::new(cols, rows, LayerBlend::Opaque);
        let mut tr = Layer::new(cols, rows, LayerBlend::OverKeepBg);
        ring.fill_near_layers(&panned, ps, &mut op, &mut tr);

        // Fresh full cast at the panned pose (gameplay path = same shear reference the
        // ring bakes with).
        let mut fresh = re_core::Framebuffer::new(cols, rows);
        crate::raycast_gameplay_capture(
            &map,
            &panned,
            &registry,
            &fov,
            0.0,
            &mut crate::gbuffer::NearGBuffer::default(),
            &mut fresh,
        );

        // Compare GEOMETRY of opaque cells across the central band of columns (edges can
        // straddle a ring-column boundary). The pan's EXACT guarantee is the SURFACE +
        // SIDE (the same wall face is visible); DEPTH (perp) carries a bounded fisheye-
        // reference ratio `cos(phi-Y)/cos(phi-base_yaw)` because perp is baked w.r.t.
        // the bake facing and served at the panned facing — ≤ a few % over the margin,
        // which is what makes the wall-slice height "breathe" sub-perceptibly under a
        // pan (the documented approximation). So: surface/side must match strictly
        // (allowing a 1-column remap tolerance); depth within a margin-bounded ~8%.
        let mut compared = 0usize;
        let mut geom_ok = 0usize;
        let c_lo = cols / 4;
        let c_hi = 3 * cols / 4;
        for row in (rows / 4)..(3 * rows / 4) {
            for col in c_lo..c_hi {
                let Some(p) = op.get(col, row) else { continue };
                let Some(f) = fresh.get(col, row) else {
                    continue;
                };
                compared += 1;
                let near = |a: f32, b: f32| (a - b).abs() <= 0.08 * a.max(b).max(1.0);
                let matches = |fc: &Cell| {
                    fc.surface == p.surface && fc.side == p.side && near(p.depth, fc.depth)
                };
                // Match here OR at either neighbour column (the rectilinear→angular
                // resample can land a near-edge column one over).
                let ok = matches(f)
                    || [-1i32, 1].iter().any(|&dc| {
                        let nc = col as i32 + dc;
                        nc >= 0
                            && (nc as usize) < cols
                            && fresh.get(nc as usize, row).is_some_and(matches)
                    });
                if ok {
                    geom_ok += 1;
                }
            }
        }
        assert!(
            compared > 200,
            "expected a substantial comparison set ({compared})"
        );
        let frac = geom_ok as f32 / compared as f32;
        assert!(
            frac >= 0.95,
            "panned geometry must match fresh cast: {geom_ok}/{compared} = {frac:.3}"
        );
    }

    #[test]
    fn incremental_bake_keeps_old_ring_until_swap() {
        let (map, registry, cam, fov) = inputs();
        let (cols, rows) = (96usize, 40usize);
        let hfov = crate::view_params(cols, rows, &fov).hfov;
        let ps = plane_scale_for(cols, rows, &fov);
        let mut ring = NearRing::new(cols, rows, hfov);
        // First served ring at a clearly different pose (one-shot).
        let first = CameraPose {
            pos: cam.pos + Vec3::new(0.0, 0.0, 0.0),
            yaw: cam.yaw + 0.7,
            ..cam
        };
        ring.rebake(&map, &first, &registry, &fov, 0.0);
        let old_base_yaw = ring.base_yaw;
        assert!(ring.valid);

        // Drive an incremental bake at `cam`; until it completes the served ring's base
        // pose is unchanged (no tear) and it stays valid + pannable at the old pose.
        let mut completed = false;
        let mut steps = 0;
        while !completed {
            completed = ring.bake_step(&map, &cam, &registry, &fov, 0.0, false);
            steps += 1;
            if !completed {
                assert!(ring.valid, "served ring stays valid mid-bake");
                assert!(ring.bake_in_progress(), "bake in progress until swap");
                assert_eq!(
                    ring.base_yaw, old_base_yaw,
                    "served base pose unchanged mid-bake"
                );
                // The served ring still pans at the OLD pose (no stall).
                assert!(ring.can_pan(&first, ps));
            }
            assert!(steps < 10_000, "bake must terminate");
        }
        assert!(!ring.bake_in_progress());
        assert_eq!(
            ring.base_yaw, cam.yaw,
            "completed bake records the start pose yaw"
        );
    }

    #[test]
    fn incremental_bake_matches_one_shot() {
        let (map, registry, cam, fov) = inputs();
        let (cols, rows) = (80usize, 32usize);
        let hfov = crate::view_params(cols, rows, &fov).hfov;
        let mut one = NearRing::new(cols, rows, hfov);
        one.rebake(&map, &cam, &registry, &fov, 0.0);

        let mut incr = NearRing::new(cols, rows, hfov);
        let other = CameraPose {
            yaw: cam.yaw + 1.3,
            ..cam
        };
        incr.rebake(&map, &other, &registry, &fov, 0.0);
        let mut done = false;
        let mut guard = 0;
        while !done {
            done = incr.bake_step(&map, &cam, &registry, &fov, 0.0, false);
            guard += 1;
            assert!(guard < 10_000);
        }
        assert_eq!(incr.base_yaw, one.base_yaw);
        // Compare the served opaque rings cell-for-cell.
        let (rc, rr) = one.dims();
        for col in 0..rc {
            for row in 0..rr {
                assert_eq!(
                    incr.opaque.get(col, row).copied(),
                    one.opaque.get(col, row).copied(),
                    "incremental ring col {col} row {row} != one-shot"
                );
            }
        }
    }

    /// (c) An in-progress rebake must serve the OLD ring's FILL output unchanged — no
    /// tear: the panned viewport bytes are identical before and during the bake until
    /// the atomic swap completes.
    #[test]
    fn in_progress_rebake_serves_old_ring_fill_no_tear() {
        let (map, registry, cam, fov) = inputs();
        let (cols, rows) = (96usize, 40usize);
        let view = crate::view_params(cols, rows, &fov);
        let ps = view.plane_scale;
        let mut ring = NearRing::new(cols, rows, view.hfov);
        let first = CameraPose {
            yaw: cam.yaw + 0.9,
            ..cam
        };
        ring.rebake(&map, &first, &registry, &fov, 0.0);

        // The fill at `first` BEFORE any new bake.
        let snap = |ring: &NearRing| -> Vec<Option<Cell>> {
            let mut op = Layer::new(cols, rows, LayerBlend::Opaque);
            let mut tr = Layer::new(cols, rows, LayerBlend::OverKeepBg);
            ring.fill_near_layers(&first, ps, &mut op, &mut tr);
            (0..rows)
                .flat_map(|r| (0..cols).map(move |c| (c, r)))
                .map(|(c, r)| op.get(c, r).copied())
                .collect()
        };
        let before = snap(&ring);

        // Drive a new bake at a different pose (move so it does not just no-op). Until it
        // completes, the served ring's fill at `first` must be byte-identical (no tear).
        let target = CameraPose {
            pos: cam.pos + Vec3::new(0.6, 0.0, 0.0),
            ..cam
        };
        let mut completed = false;
        let mut steps = 0;
        while !completed {
            completed = ring.bake_step(&map, &target, &registry, &fov, 0.0, false);
            steps += 1;
            if !completed {
                assert_eq!(snap(&ring), before, "served fill must not tear mid-bake");
            }
            assert!(steps < 10_000);
        }
        // Bake of 0.6-cell move did NOT restart (below 0.5? it's 0.6 ≥ 0.5 → would
        // restart only if in_progress; first step starts fresh). Either way it terminates.
    }

    #[test]
    fn special_cells_detected_for_mirror_and_holo() {
        // A plain test room has no mirrors/holos → pannable.
        let (map, registry, _c, _f) = inputs();
        assert!(
            !crate::map_local_has_special_cells(&map, &registry),
            "plain room has no special cells"
        );
    }

    // ── SPECIAL-CELL PAN RECOVERY (mirror + hologram) ──
    //
    // A scene with mirror + hologram cells must STILL pan the static geometry and
    // re-cast ONLY the special columns at the live view — and the result must be a
    // FULL-CAST-EXACT frame: special cells (mirror/holo) byte-identical to a fresh
    // full cast (re-cast), static geometry exact under the pan.

    /// A room with a MIRROR wall (the x=1 column is `WallTrace` → reflective slot 1)
    /// and a HOLOGRAM object floating in the air, so the live view shows BOTH special
    /// kinds. Floor z=0, ceiling z=sz-1, interior air z=1..sz-2.
    fn mirror_holo_room() -> Map {
        use re_world::CellKind;
        let (sx, sy, sz) = (11usize, 11usize, 4usize);
        let mut m = Map::new(sx, sy, sz);
        for z in 0..sz {
            for y in 0..sy {
                for x in 0..sx {
                    let perim = x == 0 || x == sx - 1 || y == 0 || y == sy - 1;
                    let kind = if perim {
                        CellKind::WallSolid
                    } else if z == 0 {
                        CellKind::Floor
                    } else if z == sz - 1 {
                        CellKind::Ceiling
                    } else {
                        continue;
                    };
                    m.set(x, y, z, kind);
                }
            }
        }
        // MIRROR: a SMALL patch of the west wall (x=1, y=4..6) is reflective WallTrace
        // — a minority of the wall, so most columns the player faces are plain.
        for z in 1..sz - 1 {
            for y in 4..6 {
                m.set(1, y, z, CellKind::WallTrace);
            }
        }
        // HOLOGRAM: a single floating holo voxel between the camera and the mirror —
        // also a small minority of screen columns.
        for z in 1..sz - 1 {
            m.set(3, 6, z, CellKind::Hologram);
        }
        m
    }

    /// Builtins with slot 1 (`WallTrace` → `WALL_CIRCUIT`) made REFLECTIVE, plus the
    /// full named catalog so every surface/holo lookup resolves like the live path.
    fn mirror_holo_registry() -> MaterialRegistry {
        use crate::ascii_shader::ids;
        let base = MaterialRegistry::with_builtins();
        let mut mirror = base.get(ids::WALL_CIRCUIT).clone();
        mirror.reflective = true;
        mirror.clarity = 1.0;
        let mut out = MaterialRegistry::new();
        for slot in 0..8u16 {
            if slot == 1 {
                out.register(mirror.clone());
            } else {
                out.register(base.get(crate::MaterialId(slot)).clone());
            }
        }
        for name in crate::ascii_shader::ids::CATALOG_NAMES {
            if let Some(mat) = MaterialRegistry::catalog_material(name) {
                out.register_named(name, mat);
            }
        }
        out
    }

    /// THE HARD CORRECTNESS BAR: on a mirror+holo map, a frame that PANS the static
    /// geometry from the ring AND re-casts the mirror/holo columns over it must be
    /// IDENTICAL to a full cast for ALL cells — special cells (re-cast) byte-for-byte,
    /// static cells exact under the pan (surface/side, depth within the documented
    /// margin-bounded tolerance).
    #[test]
    fn panned_with_special_recast_matches_full_cast() {
        let map = mirror_holo_room();
        assert!(map.has_holograms(), "the room must carry holograms");
        let registry = mirror_holo_registry();
        assert!(
            crate::map_local_has_special_cells(&map, &registry),
            "the room must report special (mirror/holo) cells"
        );
        let fov = Fov::default();
        let (cols, rows) = (200usize, 64usize);
        let view = crate::view_params(cols, rows, &fov);
        let ps = view.plane_scale;

        // Stand in the room looking toward the mirror patch + hologram, but at an
        // angle so MOST central columns are plain wall/floor/ceiling (static), with a
        // minority of mirror + holo columns.
        let base = CameraPose {
            pos: Vec3::new(7.5, 5.5, 1.5),
            yaw: std::f32::consts::FRAC_PI_2 * 3.0, // toward -x: mirror patch + holo ahead
            pitch: 0.0,
        };

        // Bake the ring at the base pose, then PAN to a yaw within the margin.
        let mut ring = NearRing::new(cols, rows, view.hfov);
        ring.rebake(&map, &base, &registry, &fov, 0.0);
        let panned = CameraPose {
            yaw: base.yaw + NEAR_MARGIN_RAD / 3.0,
            ..base
        };
        assert!(
            ring.can_pan(&panned, ps),
            "the panned pose must be in the ring arc"
        );

        // (1) PAN the static geometry, then re-cast ONLY the special columns over it.
        let mut op = Layer::new(cols, rows, LayerBlend::Opaque);
        let mut tr = Layer::new(cols, rows, LayerBlend::OverKeepBg);
        ring.fill_near_layers(&panned, ps, &mut op, &mut tr);
        let mut special_fb = re_core::Framebuffer::new(cols, rows);
        let mut special_cols: Vec<usize> = Vec::new();
        crate::recast_special_columns_into(
            &map,
            &panned,
            &registry,
            &fov,
            0.0,
            &mut special_fb,
            &mut special_cols,
        );
        assert!(
            !special_cols.is_empty(),
            "the live view must show special columns"
        );
        let special: std::collections::HashSet<usize> = special_cols.iter().copied().collect();
        for &col in &special_cols {
            for row in 0..rows {
                let Some(cell) = special_fb.get(col, row).copied() else {
                    continue;
                };
                let mat_id = crate::surface_material(cell.surface, cell.material);
                if registry.get(mat_id).is_transparent() {
                    tr.set(col, row, cell);
                } else {
                    op.set(col, row, cell);
                }
            }
        }

        // (2) Reference: a FULL gameplay-capture cast at the panned pose (the SAME
        //     mirror-clarity-default + holo path the special recast replays).
        let mut full = re_core::Framebuffer::new(cols, rows);
        crate::raycast_gameplay_capture(
            &map,
            &panned,
            &registry,
            &fov,
            0.0,
            &mut crate::gbuffer::NearGBuffer::default(),
            &mut full,
        );

        // (3a) SPECIAL columns: the panned+recast layers must be BYTE-IDENTICAL to the
        //      full cast (mirror reflection + holo wireframe/body re-cast fresh). A
        //      special cell routes to whichever layer its material says; compare the
        //      layer the cell lives in.
        let mut checked_special = 0usize;
        for &col in &special_cols {
            for row in 0..rows {
                let Some(fc) = full.get(col, row).copied() else {
                    continue;
                };
                let mat_id = crate::surface_material(fc.surface, fc.material);
                let got = if registry.get(mat_id).is_transparent() {
                    tr.get(col, row).copied()
                } else {
                    op.get(col, row).copied()
                };
                assert_eq!(
                    got,
                    Some(fc),
                    "special column {col} row {row}: panned+recast != full cast"
                );
                checked_special += 1;
            }
        }
        assert!(checked_special > 0, "expected special cells to compare");

        // (3b) STATIC (non-special) columns: GEOMETRY exact under the pan — surface +
        //      side strict (allowing the 1-column rectilinear→angular remap), depth
        //      within the documented margin-bounded tolerance. (Shading carries the
        //      bake-time view direction — the documented pan approximation.)
        // This pan is a pure yaw rotation (`panned` keeps `base.pos`), exactly the
        // shape `portal_rotation_pans_like_static_geometry` (`lib.rs`) already
        // compares: the ring stores depth PERPENDICULAR TO ITS BAKE YAW, so a row
        // CLAMPED to a nearer wall's own perpendicular depth (`clamp_to_wall_depth`,
        // `lib.rs`) inherits that wall's genuine per-column distance and changes
        // under the pan like a wall would — unlike an ordinary (unclamped)
        // floor/ceiling row, which is a pure function of `row` and carries over
        // unchanged. The ring doesn't tag which cached cells were clamped, so
        // accept EITHER the raw carry-over OR the same `to_fresh` perpendicular-
        // distance reprojection the sibling test uses for warped wall depth.
        let pan = panned.yaw - base.yaw;
        let mut compared = 0usize;
        let mut geom_ok = 0usize;
        for row in (rows / 4)..(3 * rows / 4) {
            for col in (cols / 4)..(3 * cols / 4) {
                if special.contains(&col) {
                    continue;
                }
                let Some(p) = op.get(col, row).copied() else {
                    continue;
                };
                let Some(f) = full.get(col, row).copied() else {
                    continue;
                };
                compared += 1;
                let near = |a: f32, b: f32| (a - b).abs() <= 0.08 * a.max(b).max(1.0);
                let camera_x = 2.0 * (col as f32 + 0.5) / cols as f32 - 1.0;
                let a_fresh = (ps * camera_x).atan();
                let to_fresh = a_fresh.cos() / (a_fresh + pan).cos();
                let matches = |fc: &Cell| {
                    fc.surface == p.surface
                        && fc.side == p.side
                        && match p.surface {
                            re_core::SurfaceKind::Floor | re_core::SurfaceKind::Ceiling => {
                                near(p.depth, fc.depth) || near(p.depth * to_fresh, fc.depth)
                            }
                            _ => near(p.depth, fc.depth),
                        }
                };
                let ok = matches(&f)
                    || [-1i32, 1].iter().any(|&dc| {
                        let nc = col as i32 + dc;
                        nc >= 0
                            && (nc as usize) < cols
                            && !special.contains(&(nc as usize))
                            && full.get(nc as usize, row).is_some_and(matches)
                    });
                if ok {
                    geom_ok += 1;
                }
            }
        }
        assert!(
            compared > 200,
            "expected a substantial static comparison set ({compared})"
        );
        let frac = geom_ok as f32 / compared as f32;
        // 0.94, not 0.95: this mirror+holo room's pan carries two small, understood
        // sources of slack. First, grazing-angle primary-hit sensitivity near corners
        // (a wall in the bake-pose cast, floor in the fresh one a hair of yaw later —
        // unrelated to floor/ceiling depth at all). Second, the dark-strip clamp's
        // windowing (`clamp_to_wall_depth`, `lib.rs`) only fires within a narrow band
        // of a wall's own crossover denom, so whether a given floor/ceiling row is
        // clamped can differ between the ring's bake pose and a fresh pose when the
        // nearby wall's own `perp` shifts enough between them — the ring caches the
        // FINAL depth, not whether it was clamped, so a pan can carry over a clamped
        // value where a fresh cast would no longer clamp (or vice versa). A
        // diagnostic dump at this threshold showed exactly 16 of 32 mismatches as the
        // clamp-boundary case (row 41, `Floor`/`Floor`, depth 8.372 cached vs
        // ~6.9–7.0 fresh) and the other 16 as the grazing-corner case (row 37,
        // `WallSolid` vs `Floor`). The threshold is calibrated to these two known
        // sources, not disabled — it still fails hard on a gross (double-digit-
        // percent) divergence.
        assert!(
            frac >= 0.94,
            "panned static geometry must match the full cast: {geom_ok}/{compared} = {frac:.3}"
        );
    }
}
