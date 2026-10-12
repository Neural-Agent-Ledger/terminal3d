//! `re_render::gbuffer` — the SELECTIVE RE-SHADE near G-buffer (animation-only
//! frame optimisation).
//!
//! # Why this exists
//!
//! The near tier's cost is ~93% per-cell SHADING (the DDA ray + cast-shadow are
//! geometry-only). On an ANIMATION-ONLY frame — camera, map and size all unchanged,
//! but a renderable near material animates (level 01's circuit/matrix wall pulse,
//! a `Pulse`/`Shift` surface) — the geometry on screen is byte-for-byte the same as
//! the last full cast; only the SHADE of the ANIMATED cells changes with `time`.
//!
//! Today such a frame does a FULL near re-cast throttled to ~30 Hz
//! ([`crate::composite::ANIM_RECAST_DT`]). This module replaces that with an EXACT,
//! cheaper path: during a full cast we cache, per drawn cell, the exact inputs
//! [`crate::shade_cell`] consumes (a [`CellGeom`] + `screen_y`). On an
//! animation-only frame we re-run `shade_cell` for ONLY the animated cells from
//! their cached geometry + the current `time`, leaving every static cell exactly
//! as it was persisted in the near layers.
//!
//! # Why it is EXACT (not an approximation)
//!
//! `shade_cell` is a pure function of `(registry, perlin, fog, far_freeze, memo,
//! CellGeom, screen_y, time)`. Across an idle period the registry / perlin / fog /
//! far-freeze are constant, and a `memo` of `None` produces output identical to a
//! populated memo (the memo is a within-frame cache, not a behaviour change). So
//! replaying `shade_cell` with the SAME cached `CellGeom` + `screen_y` and the new
//! `time` reproduces a genuine full shade — byte-identical to what a full re-cast
//! would have written for that cell. This is the crucial distinction from a
//! rejected "base ⊗ time" multiplicative cache: each animated cell gets a real,
//! complete shade, just fed cached hit data instead of a fresh DDA walk.
//!
//! # Scope / exclusions (kept exact, never an approximation)
//!
//! Only PLAIN wall / floor / ceiling cells are eligible. Two cell kinds are
//! EXCLUDED from the animated list so the path never produces a wrong pixel:
//!
//! * **Mirror cells** — a reflective surface runs an extra `mirror_composite`
//!   post-step keyed on `(refl_hit, clarity, col, row)` that the cached `CellGeom`
//!   alone cannot reproduce. Mirror clarity changes are STRUCTURAL (they re-cast),
//!   so excluding mirrors from the animation-only refresh only means a mirror that
//!   reflects an animated surface holds its last-cast frame between camera moves —
//!   exactly the status quo under the throttle.
//! * **Hologram (transparent) cells written by the holo pass** — the hologram pass
//!   is a separate wireframe walk, not a `shade_cell` call, so it is not captured
//!   here. A map with holograms is handled by the worst-case fallback (its animated
//!   fraction is high) or simply not selectively refreshed for the holo cells.
//!
//! Routing: a captured cell carries enough to re-derive whether it belongs in the
//! OPAQUE or TRANSPARENT layer (the same `Material::is_transparent` test
//! [`crate::composite`] splits on), so the re-shade writes each animated cell back
//! into the correct layer.
//!
//! No `unsafe`, no per-frame allocation (the buffers are persistent and
//! generation-cleared in place). Pure compute, no terminal.

use crate::CellGeom;

/// One captured cell's full re-shade inputs: the [`CellGeom`] the cast built plus
/// the `screen_y` it was shaded at. Replaying [`crate::shade_cell`] with these +
/// the current `time` reproduces the cell's shade exactly.
///
/// `Copy` (all fields are `Copy`) so the grid stores it inline with no allocation.
/// The fields are `pub(crate)` ([`CellGeom`] is a crate-internal type); the struct
/// is `pub` so the buffer's accessors can return it.
#[derive(Clone, Copy)]
pub struct GCell {
    /// The exact geometry the cast handed [`crate::shade_cell`].
    pub(crate) geom: CellGeom,
    /// The screen row the cell was shaded at (a `shade_cell` input — drives the
    /// recipe's screen-row dither / LOD selection).
    pub(crate) screen_y: i32,
    /// Whether [`crate::apply_sky`] overwrote this cell's colour/glyph with the
    /// direction-sampled panorama (ray-direction-only, time-independent — see
    /// `crate::column_sky_open`). A `Ceiling`/`Empty` surface no longer implies
    /// this: an ENCLOSED room's ceiling keeps its ordinary shaded material
    /// (which CAN be animated), so the skip below must key on what actually
    /// happened to this specific cell, not on its surface kind.
    pub(crate) sky: bool,
}

/// The persistent per-cell near G-buffer + the compact animated-cell index.
///
/// Allocated once (sized to the view) and reused every frame: a full cast
/// generation-clears it in O(1) (bump the gen counter; the parallel `gen` array
/// marks which cells are valid this generation) and refills it; an animation-only
/// frame reads the animated list and re-shades just those cells.
///
/// The animated list (`anim`) holds the `(col, row)` of every captured cell whose
/// resolved material is animated AND that is NOT a mirror cell — both near layers
/// in one list (the re-shade routes each to opaque/transparent at write time).
///
/// `Default` is an empty 0×0 buffer (no allocation); [`resize`](Self::resize) sizes
/// it to the view on the first cast.
#[derive(Default)]
pub struct NearGBuffer {
    cols: usize,
    rows: usize,
    /// Dense per-cell store. Index `row * cols + col`.
    cells: Vec<GCell>,
    /// Generation each cell was last written at — a cell is VALID iff `gen[i] ==
    /// generation`. Lets a full cast invalidate the whole buffer in O(1).
    gen: Vec<u32>,
    /// Current capture generation. Bumped at the start of each full cast capture.
    generation: u32,
    /// Compact list of `(col, row)` of animated, non-mirror captured cells (both
    /// layers). Rebuilt each full cast; reused (`.clear()`, keep capacity) so the
    /// re-shade never allocates.
    anim: Vec<(usize, usize)>,
    /// Whether the LAST full capture is currently valid (matches the persisted near
    /// layers). A resize / structural re-cast sets this true on completion; cleared
    /// when the buffer is resized empty.
    valid: bool,
    /// Per-cell SKIP flag for the selective re-shade: a cell the silhouette subcell
    /// pass replaced with a 2×2 quadrant blit. Re-shading such a cell from
    /// `CellGeom` would overwrite its persisted blit with a plain shade (which would
    /// NOT match a full re-cast's re-blit), so the re-shade SKIPS it — keeping the
    /// last-cast blit. Set per full cast by [`mark_skip`](Self::mark_skip) after the
    /// subcell pass; reset to all-false in [`begin_capture`](Self::begin_capture).
    skip: Vec<bool>,
}

impl NearGBuffer {
    /// `(cols, rows)` the buffer is sized for.
    #[inline]
    pub fn dims(&self) -> (usize, usize) {
        (self.cols, self.rows)
    }

    /// Resize to `cols`×`rows` if changed, clearing capture validity. A resize is
    /// always followed by a full cast (the near key's cols/rows changed), which
    /// re-captures, so this only drops a now-wrong buffer.
    pub fn resize(&mut self, cols: usize, rows: usize) {
        if cols != self.cols || rows != self.rows {
            self.cols = cols;
            self.rows = rows;
            self.cells.clear();
            self.cells.resize(
                cols * rows,
                GCell {
                    geom: CellGeom::default(),
                    screen_y: 0,
                    sky: false,
                },
            );
            self.gen.clear();
            self.gen.resize(cols * rows, 0);
            self.generation = 0;
            self.anim.clear();
            self.valid = false;
            self.skip.clear();
            self.skip.resize(cols * rows, false);
        }
    }

    /// Begin a fresh capture pass for a full cast: bump the generation (O(1)
    /// invalidate of every prior cell) and clear the animated list (keep capacity).
    /// `valid` is set true by [`finish_capture`](Self::finish_capture).
    pub fn begin_capture(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        // `wrapping_add` could in theory collide with a stale `gen[i]` after 2^32
        // captures; on the wrap, zero the gen array so no stale cell reads valid.
        if self.generation == 0 {
            self.gen.fill(0);
            self.generation = 1;
        }
        self.anim.clear();
        self.valid = false;
        self.skip.fill(false);
    }

    /// Mark `(col, row)` SKIPPED by the selective re-shade (it was subcell-blitted).
    /// Out-of-bounds ignored. Called per full cast after the silhouette pass.
    #[inline]
    pub fn mark_skip(&mut self, col: usize, row: usize) {
        if col < self.cols && row < self.rows {
            self.skip[row * self.cols + col] = true;
        }
    }

    /// Record one captured cell. Called from the cast for every drawn wall /
    /// floor / ceiling cell. `animated` ⇒ add it to the animated list (UNLESS
    /// `is_mirror`, which is excluded — see the module docs). `sky` records
    /// whether [`crate::apply_sky`] actually overwrote this cell (only the
    /// floor/ceiling pass ever passes `true`, and only when the camera's room
    /// genuinely opens overhead — see `crate::column_sky_open`); the selective
    /// re-shade skips exactly those cells, not every `Ceiling`/`Empty` surface.
    /// Out-of-bounds is ignored. `pub(crate)` — only the cast calls it (it takes
    /// the crate-internal [`CellGeom`]).
    #[inline]
    pub(crate) fn capture(
        &mut self,
        col: usize,
        row: usize,
        geom: &CellGeom,
        screen_y: i32,
        animated: bool,
        is_mirror: bool,
        sky: bool,
    ) {
        if col >= self.cols || row >= self.rows {
            return;
        }
        let i = row * self.cols + col;
        self.cells[i] = GCell {
            geom: *geom,
            screen_y,
            sky,
        };
        self.gen[i] = self.generation;
        if animated && !is_mirror {
            self.anim.push((col, row));
        }
    }

    /// Mark the capture complete + valid (the buffer now matches the persisted
    /// near layers). Called once after a full cast + split.
    pub fn finish_capture(&mut self) {
        self.valid = true;
    }

    /// Is the captured G-buffer valid for selective re-shade this frame?
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.valid
    }

    /// Invalidate the captured G-buffer WITHOUT a re-capture: mark it as no longer
    /// matching the persisted near layers, so the selective re-shade path can't replay
    /// stale geometry. Used when the near layers are rebuilt by a path that does NOT
    /// re-capture (e.g. the rotation-pan ring rebuild, which pans to a new yaw). The
    /// next genuine structural cast re-captures and re-validates.
    #[inline]
    pub fn invalidate(&mut self) {
        self.valid = false;
    }

    /// The number of animated (non-mirror) captured cells.
    #[inline]
    pub fn animated_len(&self) -> usize {
        self.anim.len()
    }

    /// The animated-cell `(col, row)` list (both layers).
    #[inline]
    pub fn animated_cells(&self) -> &[(usize, usize)] {
        &self.anim
    }

    /// Whether `(col, row)` is marked skipped (subcell-blitted) for the selective
    /// re-shade. `false` if out of bounds.
    #[inline]
    pub fn is_skipped(&self, col: usize, row: usize) -> bool {
        col < self.cols && row < self.rows && self.skip[row * self.cols + col]
    }

    /// The valid captured [`GCell`] at `(col, row)`, or `None` if absent / stale /
    /// out of bounds.
    #[inline]
    pub fn get(&self, col: usize, row: usize) -> Option<&GCell> {
        if col >= self.cols || row >= self.rows {
            return None;
        }
        let i = row * self.cols + col;
        if self.gen[i] == self.generation {
            Some(&self.cells[i])
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::build_world_layers_fov_capture_into;
    use crate::composite::ANIM_SELECTIVE_MAX_FRAC;
    use crate::{reshade_animated_into, surface_material, FarFreeze, Fov};
    use re_core::{Cell, CompositeStack};
    use re_world::{CameraPose, CellKind, Map};

    use super::NearGBuffer;

    /// A small room whose perimeter walls are the SHIPPED Shift-animated `wall_matrix`
    /// surface (built-in `WALL_MATRIX`), so the near cast captures animated cells.
    fn matrix_room() -> Map {
        const N: usize = 7;
        let mut m = Map::new(N, N, 4);
        for z in 0..4 {
            for y in 0..N {
                for x in 0..N {
                    let perim = x == 0 || y == 0 || x == N - 1 || y == N - 1;
                    let kind = if perim {
                        CellKind::WallMatrix
                    } else if z == 0 {
                        CellKind::Floor
                    } else if z == 3 {
                        CellKind::Ceiling
                    } else {
                        CellKind::Empty
                    };
                    m.set(x, y, z, kind);
                }
            }
        }
        m
    }

    fn cam_in(map: &Map) -> CameraPose {
        let (sx, sy, _) = map.size();
        CameraPose {
            pos: glam::Vec3::new(sx as f32 / 2.0, sy as f32 / 2.0, 1.5),
            yaw: 0.3,
            pitch: 0.0,
        }
    }

    /// Snapshot one layer's `(col,row) → Cell` so two stacks can be compared.
    fn snapshot(layer: &re_core::Layer, cols: usize, rows: usize) -> Vec<Option<Cell>> {
        (0..rows)
            .flat_map(|r| (0..cols).map(move |c| (c, r)))
            .map(|(c, r)| layer.get(c, r).copied())
            .collect()
    }

    /// THE CORRECTNESS BAR: a selective re-shade of the animated cells from a
    /// T0-captured G-buffer to time T1 must be BYTE-IDENTICAL (glyph+fg+bg+style,
    /// and every other Cell field) to a fresh FULL cast at T1 — for the animated
    /// cells — while leaving static cells exactly as the T0 cast persisted them.
    #[test]
    fn selective_reshade_is_byte_identical_to_full_recast() {
        let map = matrix_room();
        let cam = cam_in(&map);
        let registry = crate::registry_for_map(&map);
        let fov = Fov::default();
        let (cols, rows) = (120usize, 48usize);
        let far_freeze = FarFreeze::default().distance;

        let t0 = 1.0f32;
        let t1 = 1.5f32; // a different animation time

        // (1) FULL capturing cast at T0 → the persisted near layers + the G-buffer.
        // (NO subcell: the test compares the plain cast so the selective re-shade,
        //  which never touches subcell blits, is compared on equal footing.)
        let mut scratch = re_core::Framebuffer::new(cols, rows);
        let mut gbuf = NearGBuffer::default();
        let mut persisted = CompositeStack::new(cols, rows);
        build_world_layers_fov_capture_into(
            &map,
            &cam,
            &registry,
            &fov,
            t0,
            &mut scratch,
            &mut gbuf,
            &mut persisted,
        );
        assert!(gbuf.is_valid(), "capture must be valid after a full cast");
        assert!(
            gbuf.animated_len() > 0,
            "matrix walls must capture animated cells"
        );

        // Snapshot the T0 state so we can prove static cells are untouched.
        let op_t0 = snapshot(&persisted.opaque, cols, rows);
        let tr_t0 = snapshot(&persisted.transparent, cols, rows);

        // (2) Reference: a FRESH full cast at T1 (no capture, no subcell) — the
        //     ground truth the selective path must reproduce for animated cells.
        let mut reference = CompositeStack::new(cols, rows);
        let mut scratch2 = re_core::Framebuffer::new(cols, rows);
        let mut gbuf2 = NearGBuffer::default();
        build_world_layers_fov_capture_into(
            &map,
            &cam,
            &registry,
            &fov,
            t1,
            &mut scratch2,
            &mut gbuf2,
            &mut reference,
        );

        // (3) Selective re-shade the T0-persisted stack to T1.
        reshade_animated_into(&map, &registry, far_freeze, &gbuf, t1, &mut persisted);

        let op_t1 = snapshot(&persisted.opaque, cols, rows);
        let tr_t1 = snapshot(&persisted.transparent, cols, rows);
        let op_ref = snapshot(&reference.opaque, cols, rows);
        let tr_ref = snapshot(&reference.transparent, cols, rows);

        // Build the animated-cell set (the cells the selective path re-shaded).
        let anim: std::collections::HashSet<(usize, usize)> =
            gbuf.animated_cells().iter().copied().collect();

        let mut checked_animated = 0usize;
        for r in 0..rows {
            for c in 0..cols {
                let i = r * cols + c;
                let is_anim = anim.contains(&(c, r)) && !gbuf.is_skipped(c, r);
                if is_anim {
                    // BYTE-IDENTICAL to the full re-cast at T1.
                    assert_eq!(
                        op_t1[i], op_ref[i],
                        "animated opaque cell ({c},{r}) selective != full re-cast at T1"
                    );
                    assert_eq!(
                        tr_t1[i], tr_ref[i],
                        "animated transparent cell ({c},{r}) selective != full re-cast at T1"
                    );
                    checked_animated += 1;
                } else {
                    // STATIC (or skipped) cell: untouched since the T0 cast.
                    assert_eq!(
                        op_t1[i], op_t0[i],
                        "static opaque cell ({c},{r}) must NOT change on selective re-shade"
                    );
                    assert_eq!(
                        tr_t1[i], tr_t0[i],
                        "static transparent cell ({c},{r}) must NOT change on selective re-shade"
                    );
                }
            }
        }
        assert!(
            checked_animated > 0,
            "expected to verify some animated cells"
        );

        // Sanity: the animation actually MOVED (T1 differs from T0 somewhere in the
        // animated set), so the test isn't vacuously passing on a frozen material.
        let moved = anim.iter().any(|&(c, r)| {
            let i = r * cols + c;
            !gbuf.is_skipped(c, r) && op_t1[i] != op_t0[i]
        });
        assert!(
            moved,
            "the Shift-animated wall must change between T0 and T1"
        );
    }

    /// A `Pulse` material (catalog `monitor` is Pulse-animated) selective re-shade is
    /// likewise byte-identical to a full re-cast — proves the path is not specific to
    /// the Shift recipe.
    #[test]
    fn selective_reshade_exact_for_pulse_material() {
        use crate::ascii_shader::MaterialRegistry;
        // Build a registry whose surface slot 0 (WALL_BRICK dispatch) is a Pulse
        // material, then the rest of the usual catalog so lookups resolve.
        let pulse =
            MaterialRegistry::catalog_material("monitor").expect("monitor (Pulse) in catalog");
        assert!(pulse.is_animated(), "monitor must be animated (Pulse)");
        let mut registry = MaterialRegistry::new();
        for _ in 0..8 {
            registry.register_named("monitor", pulse.clone());
        }
        for name in crate::ascii_shader::ids::CATALOG_NAMES {
            if let Some(m) = MaterialRegistry::catalog_material(name) {
                registry.register_named(name, m);
            }
        }

        // A plain solid-wall room: WALL_BRICK dispatch (slot 0) → the Pulse material.
        let mut map = Map::new(7, 7, 4);
        for z in 0..4 {
            for y in 0..7 {
                for x in 0..7 {
                    let perim = x == 0 || y == 0 || x == 6 || y == 6;
                    let kind = if perim {
                        CellKind::WallSolid
                    } else if z == 0 {
                        CellKind::Floor
                    } else if z == 3 {
                        CellKind::Ceiling
                    } else {
                        CellKind::Empty
                    };
                    map.set(x, y, z, kind);
                }
            }
        }
        let cam = cam_in(&map);
        let fov = Fov::default();
        let (cols, rows) = (96usize, 40usize);
        let far_freeze = FarFreeze::default().distance;
        let (t0, t1) = (0.5f32, 2.0f32);

        let mut scratch = re_core::Framebuffer::new(cols, rows);
        let mut gbuf = NearGBuffer::default();
        let mut persisted = CompositeStack::new(cols, rows);
        build_world_layers_fov_capture_into(
            &map,
            &cam,
            &registry,
            &fov,
            t0,
            &mut scratch,
            &mut gbuf,
            &mut persisted,
        );
        assert!(gbuf.animated_len() > 0);

        let mut reference = CompositeStack::new(cols, rows);
        let mut scratch2 = re_core::Framebuffer::new(cols, rows);
        let mut gbuf2 = NearGBuffer::default();
        build_world_layers_fov_capture_into(
            &map,
            &cam,
            &registry,
            &fov,
            t1,
            &mut scratch2,
            &mut gbuf2,
            &mut reference,
        );

        reshade_animated_into(&map, &registry, far_freeze, &gbuf, t1, &mut persisted);

        for &(c, r) in gbuf.animated_cells() {
            if gbuf.is_skipped(c, r) {
                continue;
            }
            let mat_id = surface_material(
                gbuf.get(c, r).unwrap().geom.surface,
                gbuf.get(c, r).unwrap().geom.material,
            );
            let (sel, refc) = if registry.get(mat_id).is_transparent() {
                (
                    persisted.transparent.get(c, r).copied(),
                    reference.transparent.get(c, r).copied(),
                )
            } else {
                (
                    persisted.opaque.get(c, r).copied(),
                    reference.opaque.get(c, r).copied(),
                )
            };
            assert_eq!(sel, refc, "Pulse cell ({c},{r}) selective != full re-cast");
        }
    }

    /// The worst-case guard threshold is a sane fraction in `(0, 1]`.
    #[test]
    fn selective_max_frac_is_a_sane_fraction() {
        const { assert!(ANIM_SELECTIVE_MAX_FRAC > 0.0 && ANIM_SELECTIVE_MAX_FRAC <= 1.0) };
    }

    /// `invalidate()` clears capture validity WITHOUT touching captured cells, so a
    /// pan rebuild (which rebuilds the near layers at a new yaw but does NOT re-capture)
    /// can't be followed by a selective re-shade replaying pre-pan geometry.
    #[test]
    fn invalidate_clears_validity_so_pan_blocks_selective_replay() {
        let mut g = NearGBuffer::default();
        g.resize(4, 4);
        g.begin_capture();
        let geom = crate::CellGeom::default();
        g.capture(1, 1, &geom, 1, true, false, false);
        g.finish_capture();
        assert!(g.is_valid(), "a finished capture is valid");
        // Simulate the pan branch: the near layers were rebuilt from the ring, so the
        // captured geometry no longer matches the on-screen yaw.
        g.invalidate();
        assert!(
            !g.is_valid(),
            "after a pan rebuild the G-buffer must be invalid so selective re-shade is gated off"
        );
        // The captured cell is still PRESENT (invalidate is not a re-capture) — only the
        // validity gate flipped.
        assert!(
            g.get(1, 1).is_some(),
            "invalidate must not drop captured cells"
        );
    }

    /// `begin_capture`'s generation bump invalidates every prior cell (O(1)).
    #[test]
    fn generation_bump_invalidates_prior_cells() {
        let mut g = NearGBuffer::default();
        g.resize(4, 4);
        g.begin_capture();
        let geom = crate::CellGeom::default();
        g.capture(1, 1, &geom, 1, false, false, false);
        assert!(g.get(1, 1).is_some());
        // A new capture generation drops the prior cell without touching its memory.
        g.begin_capture();
        assert!(
            g.get(1, 1).is_none(),
            "prior-generation cell must read stale"
        );
    }
}
