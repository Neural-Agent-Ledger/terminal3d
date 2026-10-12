//! `[part]` voxel-volume authoring helpers (the pedestal OUTPUT model).
//!
//! Pure functions over a [`re_level::Part`] — no I/O, no terminal. A `Part` is a
//! named `w*h*d` voxel volume carrying a solid MASK (`1` = solid), indexed
//! `lx + w*(ly + h*lz)` (the SAME convention `re_voximport`'s serializer +
//! `re_level`'s parser use, so a populated `Part` round-trips losslessly). An
//! `[emit NODE=1] parts = …` (a [`re_level::EmitClause`]) lights named parts when
//! a node fires; that wiring is authored on the Node inspector (see `entity.rs`),
//! not here.
//!
//! THE single source of truth for mask geometry: the editor's mask-edit modal
//! (`edit.rs`) routes every solid/empty toggle through [`set_mask`] /
//! [`toggle_mask`] and every dimension change through [`resize_mask`], so the
//! mask length stays `w*h*d` and the index formula is never duplicated.
use re_level::Part;

/// Linear index of part-local voxel `(lx, ly, lz)` in `part.mask`, using the
/// canonical `lx + w*(ly + h*lz)` layout. Returns `None` when the coordinate is
/// outside the part's `w*h*d` extent (so callers never index out of bounds).
fn mask_index(part: &Part, lx: i32, ly: i32, lz: i32) -> Option<usize> {
    let (w, h, d) = (part.w, part.h, part.d);
    if lx < 0 || ly < 0 || lz < 0 || lx >= w || ly >= h || lz >= d {
        return None;
    }
    Some((lx + w * (ly + h * lz)) as usize)
}

/// Read whether part-local voxel `(lx, ly, lz)` is SOLID. Out-of-extent cells
/// read as empty (`false`).
pub fn mask_solid(part: &Part, lx: i32, ly: i32, lz: i32) -> bool {
    match mask_index(part, lx, ly, lz) {
        Some(i) => part.mask.get(i).copied().unwrap_or(0) != 0,
        None => false,
    }
}

/// Set part-local voxel `(lx, ly, lz)` solid (`true`) or empty (`false`). A
/// defined no-op when the coordinate is outside the part's extent. The mask is
/// grown to `w*h*d` (with new cells empty) if it was short, so a hand-built
/// `Part` is always safe to paint into.
pub fn set_mask(part: &mut Part, lx: i32, ly: i32, lz: i32, solid: bool) {
    let Some(i) = mask_index(part, lx, ly, lz) else {
        return;
    };
    let need = (part.w * part.h * part.d).max(0) as usize;
    if part.mask.len() < need {
        part.mask.resize(need, 0);
    }
    part.mask[i] = u8::from(solid);
}

/// Toggle part-local voxel `(lx, ly, lz)` solid↔empty, returning the NEW state
/// (`true` = now solid). A no-op returning `false` when the coordinate is outside
/// the part's extent.
pub fn toggle_mask(part: &mut Part, lx: i32, ly: i32, lz: i32) -> bool {
    let now = !mask_solid(part, lx, ly, lz);
    set_mask(part, lx, ly, lz, now);
    now
}

/// Resize a part to `w*h*d`, preserving the solid state of every voxel that lies
/// within BOTH the old and the new extent; voxels newly inside the volume are
/// born SOLID (matching the all-solid default a freshly-placed part gets), so
/// growing a part extends its solid body rather than punching holes. Each
/// dimension is clamped to `>= 1`. The mask is rebuilt to exactly `w*h*d` bytes.
pub fn resize_mask(part: &mut Part, w: i32, h: i32, d: i32) {
    let (nw, nh, nd) = (w.max(1), h.max(1), d.max(1));
    let (ow, oh, od) = (part.w, part.h, part.d);
    let mut next = vec![0u8; (nw * nh * nd) as usize];
    for lz in 0..nd {
        for ly in 0..nh {
            for lx in 0..nw {
                // Inside the OLD extent: carry the old solid state. Newly grown
                // cells default to SOLID (1) so the part stays a filled body.
                let solid = if lx < ow && ly < oh && lz < od {
                    part.mask
                        .get((lx + ow * (ly + oh * lz)) as usize)
                        .copied()
                        .unwrap_or(0)
                        != 0
                } else {
                    true
                };
                next[(lx + nw * (ly + nh * lz)) as usize] = u8::from(solid);
            }
        }
    }
    part.w = nw;
    part.h = nh;
    part.d = nd;
    part.mask = next;
}

/// Build a fresh all-solid [`re_level::Part`] of size `w*h*d` anchored at world
/// `(off_x, off_y, off_z)` with the given `name` + `color_rgb`. The mask is all
/// `1`s (length `w*h*d`), matching the on-disk "create = all solid" convention.
pub fn new_part(
    name: String,
    w: i32,
    h: i32,
    d: i32,
    off: (i32, i32, i32),
    color_rgb: u32,
) -> Part {
    let mut part = Part {
        name,
        w: w.max(1),
        h: h.max(1),
        d: d.max(1),
        mask: Vec::new(),
        off_x: off.0,
        off_y: off.1,
        off_z: off.2,
        color_rgb,
    };
    // resize_mask from a zero-extent (empty mask) births every cell solid.
    let (pw, ph, pd) = (part.w, part.h, part.d);
    part.w = 0;
    part.h = 0;
    part.d = 0;
    resize_mask(&mut part, pw, ph, pd);
    part
}

#[cfg(test)]
mod tests {
    use super::*;

    fn part_2x2x2() -> Part {
        new_part("p".into(), 2, 2, 2, (0, 0, 0), 0xFFFFFF)
    }

    #[test]
    fn new_part_is_all_solid() {
        let p = part_2x2x2();
        assert_eq!(p.mask.len(), 8, "mask length is w*h*d");
        assert!(
            p.mask.iter().all(|&b| b == 1),
            "freshly created part is all solid"
        );
    }

    #[test]
    fn index_formula_matches_serializer() {
        // index = lx + w*(ly + h*lz); set one cell and confirm the byte position.
        let mut p = new_part("p".into(), 3, 2, 2, (0, 0, 0), 0);
        // clear all then set (1, 1, 1): idx = 1 + 3*(1 + 2*1) = 1 + 3*3 = 10.
        p.mask.fill(0);
        set_mask(&mut p, 1, 1, 1, true);
        assert_eq!(p.mask[10], 1);
        assert!(mask_solid(&p, 1, 1, 1));
        assert!(!mask_solid(&p, 0, 0, 0));
    }

    #[test]
    fn toggle_flips_and_reports_state() {
        let mut p = part_2x2x2();
        assert!(!toggle_mask(&mut p, 0, 0, 0), "all-solid -> empty");
        assert!(!mask_solid(&p, 0, 0, 0));
        assert!(toggle_mask(&mut p, 0, 0, 0), "empty -> solid");
        assert!(mask_solid(&p, 0, 0, 0));
    }

    #[test]
    fn out_of_extent_edits_are_noops() {
        let mut p = part_2x2x2();
        let before = p.mask.clone();
        set_mask(&mut p, 2, 0, 0, false); // x out of range
        set_mask(&mut p, 0, 0, -1, true); // z negative
        assert_eq!(p.mask, before, "out-of-extent edits change nothing");
    }

    #[test]
    fn resize_preserves_overlap_and_births_new_cells_solid() {
        let mut p = new_part("p".into(), 2, 2, 1, (0, 0, 0), 0);
        // Carve a hole at (0,0,0), keep (1,1,0) solid.
        set_mask(&mut p, 0, 0, 0, false);
        resize_mask(&mut p, 3, 2, 1);
        assert_eq!(p.mask.len(), 6, "mask length follows the new extent");
        assert!(!mask_solid(&p, 0, 0, 0), "old hole preserved");
        assert!(mask_solid(&p, 1, 1, 0), "old solid preserved");
        assert!(mask_solid(&p, 2, 0, 0), "newly grown column is solid");
    }

    #[test]
    fn resize_clamps_each_dim_to_at_least_one() {
        let mut p = part_2x2x2();
        resize_mask(&mut p, 0, -3, 1);
        assert_eq!((p.w, p.h, p.d), (1, 1, 1), "each dim clamps to >= 1");
        assert_eq!(p.mask.len(), 1);
    }
}
