//! Per-frame metric suite + the per-orbit shimmer metric.
//!
//! [`analyze_frame`] turns a [`Framebuffer`] into a [`FrameMetrics`]: contrast /
//! separation, legibility, depth cue, dynamic range, entropy, degenerate guards,
//! and colour-blind-safety scores. [`OrbitMetrics`] then measures temporal
//! shimmer across an orbit of poses at one position.

use std::collections::HashMap;

use re_core::Framebuffer;

use crate::color::{self, Cvd, Oklab};
use crate::contrast::{self, ContrastAccum};
use crate::ink;

/// Below this Oklab ΔE, a glyph's fg is "invisible" against its own bg (or two
/// adjacent objects collapse — same threshold drives palette collisions + CVD).
pub const INVISIBILITY_EPS: f32 = 0.06;

/// Every metric computed for a single analyzed frame.
#[derive(Clone, Debug, PartialEq)]
pub struct FrameMetrics {
    // ── headline contrast / separation ──
    /// Mean Oklab ΔE across boundary (real edge) neighbour pairs.
    pub boundary_color_contrast: f32,
    /// Mean ink (glyph coverage) contrast across boundary neighbour pairs.
    pub boundary_shape_contrast: f32,
    /// Mean Oklab ΔE across interior (same-surface) neighbour pairs.
    pub interior_color_contrast: f32,
    /// Mean ink (glyph coverage) contrast across interior neighbour pairs.
    pub interior_shape_contrast: f32,
    /// boundary_color_contrast − interior_color_contrast (the headline score).
    pub separation: f32,

    // ── legibility ──
    /// % of non-space cells whose fg↔bg Oklab ΔE is below [`INVISIBILITY_EPS`].
    ///
    /// Kept for back-compat. Conflates near-field readability defects with
    /// far-field fog convergence — prefer the [`invisible_pct_near`] /
    /// [`invisible_pct_far`] split below, which separates the actionable signal
    /// (bad material colours up close) from expected fog blending at distance.
    ///
    /// [`invisible_pct_near`]: FrameMetrics::invisible_pct_near
    /// [`invisible_pct_far`]: FrameMetrics::invisible_pct_far
    pub invisible_glyph_pct: f32,
    /// % invisibility among non-space cells with finite `depth ≤` the near/far
    /// split (the median of finite cell depths). THE actionable defect signal:
    /// a glyph that vanishes against its own bg up close is a bad material
    /// choice, not fog.
    pub invisible_pct_near: f32,
    /// % invisibility among non-space cells with finite `depth >` the split.
    /// Informational: at distance, atmospheric fog blends fg and bg toward the
    /// fog colour, so high values here are expected, not a defect.
    pub invisible_pct_far: f32,
    /// The depth threshold (world units) that split near from far for this
    /// frame — the median of finite cell depths. `0.0` if no finite-depth
    /// non-space cells exist.
    pub invis_near_depth: f32,

    // ── depth cue ──
    /// Spearman rank correlation between per-cell depth and effective luma.
    pub depth_luma_spearman: f32,

    // ── dynamic range / palette ──
    /// 5th percentile of effective luma across non-space cells.
    pub luma_p5: f32,
    /// 95th percentile of effective luma across non-space cells.
    pub luma_p95: f32,
    /// `luma_p95 − luma_p5` — the frame's usable dynamic range.
    pub luma_spread: f32,
    /// Count of distinct glyphs used across the frame.
    pub distinct_glyphs: u32,
    /// Count of distinct (fg, bg) colour pairs used across the frame.
    pub distinct_colors: u32,
    /// Spread of Oklab hue angle across non-space cells (palette variety).
    pub hue_spread: f32,

    // ── information / structure ──
    /// Shannon entropy of the glyph distribution across the frame.
    pub glyph_entropy: f32,
    /// Shannon entropy of the (fg, bg) colour-pair distribution.
    pub color_entropy: f32,
    /// Vertical / horizontal boundary-edge ratio (see
    /// [`crate::contrast::ContrastAccum::edge_orientation_ratio`]).
    pub edge_orientation_ratio: f32,

    // ── degenerate guards ──
    /// True if every cell is a space (nothing rendered).
    pub is_blank: bool,
    /// True if every non-space cell uses the same glyph.
    pub single_glyph: bool,
    /// True if every non-space cell uses the same (fg, bg) colour pair.
    pub single_color: bool,

    // ── colour-blind safety ──
    /// Worst (smallest) boundary colour contrast across protan/deutan/tritan.
    pub cvd_min_boundary_contrast: f32,

    // ── diagnostics ──
    /// Count of cells whose glyph fell back to the unmapped-coverage value.
    pub unmapped_glyphs: u32,
}

/// Analyze a single framebuffer into its [`FrameMetrics`]. Pure + deterministic.
pub fn analyze_frame(fb: &Framebuffer) -> FrameMetrics {
    let cols = fb.cols();
    let rows = fb.rows();
    let n = cols * rows;

    // ── single pass: gather per-cell effective Oklab, luma, depth, histograms ──
    let mut depths: Vec<f32> = Vec::with_capacity(n);
    let mut lumas: Vec<f32> = Vec::with_capacity(n);
    let mut glyph_hist: HashMap<u32, u64> = HashMap::new();
    let mut color_hist: HashMap<u32, u64> = HashMap::new();
    let mut hues: Vec<f32> = Vec::new();
    let mut distinct_glyphs_set: std::collections::HashSet<u32> = std::collections::HashSet::new();
    let mut distinct_colors_set: std::collections::HashSet<u32> = std::collections::HashSet::new();

    let mut nonspace = 0u64;
    let mut invisible = 0u64;
    let mut unmapped = 0u32;
    let space = u32::from(' ');

    // Per non-space cell: (depth, is_invisible). Only finite-depth cells are
    // recorded — floor/sky carry INFINITY and are excluded from the split, same
    // as the existing combined metric only counts non-space cells.
    let mut invis_by_depth: Vec<(f32, bool)> = Vec::new();

    for cell in fb.cells() {
        let glyph = cell.glyph;
        let ink_l = ink::lookup(glyph);
        if !ink_l.mapped {
            unmapped += 1;
        }
        let bg_lin = color::unpack_linear(cell.bg);
        let fg_lin = color::unpack_linear(cell.fg);
        let eff = color::lerp_linear(bg_lin, fg_lin, ink_l.coverage);
        let lab = color::linear_to_oklab(eff);

        lumas.push(lab.l);
        depths.push(cell.depth);

        // Effective-colour histogram key: quantize the packed effective colour.
        let eff_packed = color::pack_linear(eff);
        *color_hist.entry(eff_packed).or_insert(0) += 1;
        distinct_colors_set.insert(eff_packed);

        *glyph_hist.entry(glyph).or_insert(0) += 1;
        distinct_glyphs_set.insert(glyph);

        if glyph != space {
            nonspace += 1;
            // Invisible glyph: fg vs bg collapse in Oklab.
            let d = color::delta_e_packed(cell.fg, cell.bg);
            let is_invis = d < INVISIBILITY_EPS;
            if is_invis {
                invisible += 1;
            }
            // Record finite-depth cells for the near/far split.
            if cell.depth.is_finite() {
                invis_by_depth.push((cell.depth, is_invis));
            }
        }

        // Hue (only meaningful with real chroma).
        if color::oklab_chroma(lab) > 0.02 {
            hues.push(color::oklab_hue(lab));
        }
    }

    // ── contrast / separation (boundary vs interior) ──
    let acc: ContrastAccum = contrast::accumulate(fb);
    let separation = acc.boundary_color() - acc.interior_color();

    // ── degenerate guards ──
    let is_blank = nonspace == 0;
    let single_glyph = distinct_glyphs_set.len() <= 1;
    let single_color = distinct_colors_set.len() <= 1;

    // ── dynamic range ──
    let (luma_p5, luma_p95) = percentiles(&mut lumas.clone(), 0.05, 0.95);
    let luma_spread = (luma_p95 - luma_p5).max(0.0);

    // ── depth↔luma Spearman ──
    let depth_luma_spearman = spearman(&depths, &lumas);

    // ── entropy ──
    let glyph_entropy = shannon_entropy(&glyph_hist, n as u64);
    let color_entropy = shannon_entropy(&color_hist, n as u64);

    // ── hue spread ──
    let hue_spread = circular_spread(&hues);

    // ── invisible % ──
    let invisible_glyph_pct = if nonspace == 0 {
        0.0
    } else {
        100.0 * invisible as f32 / nonspace as f32
    };

    // ── near/far invisibility split ──
    //
    // Split rule: the MEDIAN of finite cell depths in this frame. Chosen over a
    // fixed world-units constant because the metric crate only sees the
    // `Framebuffer`, never the fog params — an adaptive, magic-constant-free
    // threshold scales to any level/grid and keeps roughly half the rendered
    // surface in each band. Deterministic: pure function of the cell depths.
    //
    // `invisible_pct_near` (depth ≤ median) is the ACTIONABLE defect signal —
    // fg≈bg up close means bad material colours. `invisible_pct_far`
    // (depth > median) is informational — at distance, atmospheric fog blends
    // fg and bg toward the fog colour, so invisibility there is expected.
    let (invisible_pct_near, invisible_pct_far, invis_near_depth) =
        near_far_invisible(&mut invis_by_depth);

    // ── CVD: recompute boundary colour contrast under each dichromacy ──
    let cvd_min_boundary_contrast = cvd_min_boundary(fb);

    FrameMetrics {
        boundary_color_contrast: acc.boundary_color(),
        boundary_shape_contrast: acc.boundary_shape(),
        interior_color_contrast: acc.interior_color(),
        interior_shape_contrast: acc.interior_shape(),
        separation,
        invisible_glyph_pct,
        invisible_pct_near,
        invisible_pct_far,
        invis_near_depth,
        depth_luma_spearman,
        luma_p5,
        luma_p95,
        luma_spread,
        distinct_glyphs: distinct_glyphs_set.len() as u32,
        distinct_colors: distinct_colors_set.len() as u32,
        hue_spread,
        glyph_entropy,
        color_entropy,
        edge_orientation_ratio: acc.edge_orientation_ratio(),
        is_blank,
        single_glyph,
        single_color,
        cvd_min_boundary_contrast,
        unmapped_glyphs: unmapped,
    }
}

/// Split per-cell invisibility into near and far bands at the median finite
/// depth. Input: `(depth, is_invisible)` for every finite-depth non-space cell
/// (`depth` is guaranteed finite by the caller). Returns
/// `(near_pct, far_pct, split_depth)` where `near` covers `depth ≤ split` and
/// `far` covers `depth > split`. With no finite-depth cells, returns
/// `(0.0, 0.0, 0.0)`.
///
/// The split is the median of the finite depths, computed deterministically by
/// sorting; ties at the median land in the near band (`≤`), which is the
/// conservative choice for the actionable signal.
fn near_far_invisible(cells: &mut [(f32, bool)]) -> (f32, f32, f32) {
    if cells.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    // Median of the finite depths.
    let mut depths: Vec<f32> = cells.iter().map(|&(d, _)| d).collect();
    depths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let split = depths[depths.len() / 2];

    let (mut near_total, mut near_invis) = (0u64, 0u64);
    let (mut far_total, mut far_invis) = (0u64, 0u64);
    for &(depth, invis) in cells.iter() {
        if depth <= split {
            near_total += 1;
            near_invis += u64::from(invis);
        } else {
            far_total += 1;
            far_invis += u64::from(invis);
        }
    }
    let pct = |inv: u64, tot: u64| -> f32 {
        if tot == 0 {
            0.0
        } else {
            100.0 * inv as f32 / tot as f32
        }
    };
    (
        pct(near_invis, near_total),
        pct(far_invis, far_total),
        split,
    )
}

/// Recompute the mean boundary colour contrast after mapping every cell's
/// effective colour through each CVD simulation; return the worst (smallest) of
/// the three. A frame whose object edges survive all three reads safely.
fn cvd_min_boundary(fb: &Framebuffer) -> f32 {
    let cols = fb.cols();
    let rows = fb.rows();
    if cols == 0 || rows == 0 {
        return 0.0;
    }
    // Precompute per-cell effective linear colour once.
    let mut eff_lin = Vec::with_capacity(cols * rows);
    for cell in fb.cells() {
        let bg = color::unpack_linear(cell.bg);
        let fg = color::unpack_linear(cell.fg);
        let t = ink::coverage(cell.glyph);
        eff_lin.push(color::lerp_linear(bg, fg, t));
    }
    let idx = |col: usize, row: usize| row * cols + col;

    let mut worst = f32::INFINITY;
    for kind in Cvd::ALL {
        // Map each cell's effective colour through this CVD, to Oklab.
        let lab: Vec<Oklab> = eff_lin
            .iter()
            .map(|&c| color::linear_to_oklab(color::simulate_cvd(c, kind)))
            .collect();

        let mut sum = 0.0f64;
        let mut count = 0u64;
        for row in 0..rows {
            for col in 0..cols {
                let a = fb.get(col, row).expect("in-bounds");
                let ia = idx(col, row);
                if col + 1 < cols {
                    let b = fb.get(col + 1, row).expect("in-bounds");
                    if contrast::classify(a, b) == contrast::PairClass::Boundary {
                        sum += color::delta_e(lab[ia], lab[idx(col + 1, row)]) as f64;
                        count += 1;
                    }
                }
                if row + 1 < rows {
                    let b = fb.get(col, row + 1).expect("in-bounds");
                    if contrast::classify(a, b) == contrast::PairClass::Boundary {
                        sum += color::delta_e(lab[ia], lab[idx(col, row + 1)]) as f64;
                        count += 1;
                    }
                }
            }
        }
        let mean = if count == 0 {
            0.0
        } else {
            (sum / count as f64) as f32
        };
        worst = worst.min(mean);
    }
    if worst.is_finite() {
        worst
    } else {
        0.0
    }
}

// ───────────────────────────── statistics ───────────────────────────────────

/// Shannon entropy (bits) of a histogram over `total` samples.
fn shannon_entropy(hist: &HashMap<u32, u64>, total: u64) -> f32 {
    if total == 0 {
        return 0.0;
    }
    let total = total as f64;
    let mut h = 0.0f64;
    for &count in hist.values() {
        if count == 0 {
            continue;
        }
        let p = count as f64 / total;
        h -= p * p.log2();
    }
    h as f32
}

/// The `lo`/`hi` percentiles of `vals` (mutated: sorted in place). Returns
/// `(lo_val, hi_val)`; `(0,0)` for an empty slice.
fn percentiles(vals: &mut [f32], lo: f32, hi: f32) -> (f32, f32) {
    if vals.is_empty() {
        return (0.0, 0.0);
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pick = |q: f32| -> f32 {
        let n = vals.len();
        let idx = ((n as f32 - 1.0) * q).round() as usize;
        vals[idx.min(n - 1)]
    };
    (pick(lo), pick(hi))
}

/// Spearman rank correlation between two equal-length series. Returns 0 for a
/// degenerate input (empty, length mismatch, or a constant series with no rank
/// variation). Non-finite depths (open sky) are ranked as the largest values,
/// which is correct: sky is "infinitely far".
fn spearman(xs: &[f32], ys: &[f32]) -> f32 {
    let n = xs.len();
    if n < 2 || ys.len() != n {
        return 0.0;
    }
    let rx = ranks(xs);
    let ry = ranks(ys);
    pearson(&rx, &ry)
}

/// Fractional ranks (ties share the average rank). NaN/∞ sort to the high end.
fn ranks(vals: &[f32]) -> Vec<f32> {
    let n = vals.len();
    let mut idx: Vec<usize> = (0..n).collect();
    let key = |v: f32| -> f32 {
        if v.is_nan() {
            f32::INFINITY
        } else {
            v
        }
    };
    idx.sort_by(|&a, &b| {
        key(vals[a])
            .partial_cmp(&key(vals[b]))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut rank = vec![0.0f32; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        // Tie-detection: exact equality is correct here because both sides were
        // produced by the same `key()` transformation on the same sorted input.
        // We want to detect values that are byte-identical (same rank group), so
        // `==` is the right operator — partial_cmp would not distinguish ties from
        // near-misses. This is a documented false-positive-by-design.
        #[allow(clippy::float_cmp)]
        while j < n && key(vals[idx[j]]) == key(vals[idx[i]]) {
            j += 1;
        }
        // Average rank for the tie group [i, j).
        let avg = ((i + j - 1) as f32) / 2.0 + 1.0; // 1-based average
        for &k in &idx[i..j] {
            rank[k] = avg;
        }
        i = j;
    }
    rank
}

/// Pearson correlation between two equal-length series. 0 if either is constant.
fn pearson(xs: &[f32], ys: &[f32]) -> f32 {
    let n = xs.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mx = xs.iter().map(|&v| v as f64).sum::<f64>() / n;
    let my = ys.iter().map(|&v| v as f64).sum::<f64>() / n;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    for i in 0..xs.len() {
        let dx = xs[i] as f64 - mx;
        let dy = ys[i] as f64 - my;
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    if sxx <= 0.0 || syy <= 0.0 {
        return 0.0;
    }
    (sxy / (sxx.sqrt() * syy.sqrt())) as f32
}

/// A spread measure for circular (hue) data: the circular range, in radians,
/// `[0, π]`. 0 for fewer than two samples. Uses the mean resultant length:
/// `spread = 1 - R` scaled to `[0, π]` is avoided in favour of the simpler
/// "largest gap" complement so a bimodal palette reads as wide.
fn circular_spread(hues: &[f32]) -> f32 {
    if hues.len() < 2 {
        return 0.0;
    }
    // 1 - mean-resultant-length: 0 = all same hue, →1 = spread over the circle.
    let n = hues.len() as f64;
    let (mut cs, mut sn) = (0.0f64, 0.0f64);
    for &h in hues {
        cs += (h as f64).cos();
        sn += (h as f64).sin();
    }
    let r = ((cs / n).powi(2) + (sn / n).powi(2)).sqrt();
    (1.0 - r) as f32
}

// ───────────────────────────── orbit shimmer ────────────────────────────────

/// Temporal-stability metric over an orbit of poses at one position. Measures
/// the mean per-cell effective-colour change between consecutive yaw-neighbour
/// frames, normalized by the angular delta (so a big change for a tiny rotation
/// — aliasing / flicker — scores high).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrbitMetrics {
    /// Mean per-cell Oklab ΔE per radian of yaw, averaged over consecutive
    /// same-position yaw neighbours.
    pub shimmer: f32,
    /// Number of consecutive frame pairs that contributed.
    pub pairs: u32,
}

/// Compute shimmer over a sequence of frames captured at the SAME position and
/// pitch, in ascending yaw order. `yaw_step` is the angular delta (radians)
/// between consecutive frames. Frames of differing size are skipped (no pairing).
pub fn orbit_shimmer(frames: &[&Framebuffer], yaw_step: f32) -> OrbitMetrics {
    let step = yaw_step.abs().max(1e-4);
    let mut sum = 0.0f64;
    let mut pairs = 0u32;
    for w in frames.windows(2) {
        let (a, b) = (w[0], w[1]);
        if a.cols() != b.cols() || a.rows() != b.rows() {
            continue;
        }
        let n = a.cells().len();
        if n == 0 {
            continue;
        }
        let mut cell_sum = 0.0f64;
        for (ca, cb) in a.cells().iter().zip(b.cells().iter()) {
            let la = contrast::effective_oklab(ca);
            let lb = contrast::effective_oklab(cb);
            cell_sum += color::delta_e(la, lb) as f64;
        }
        sum += (cell_sum / n as f64) / step as f64;
        pairs += 1;
    }
    let shimmer = if pairs == 0 {
        0.0
    } else {
        (sum / pairs as f64) as f32
    };
    OrbitMetrics { shimmer, pairs }
}

#[cfg(test)]
mod tests {
    use super::*;
    use re_core::{Cell, SurfaceKind};

    fn cell(glyph: char, fg: u32, bg: u32, depth: f32) -> Cell {
        Cell {
            glyph: u32::from(glyph),
            fg,
            bg,
            depth,
            surface: SurfaceKind::WallSolid,
            ..Cell::default()
        }
    }

    #[test]
    fn blank_frame_is_flagged_blank() {
        let fb = Framebuffer::new(8, 8); // all default = space
        let m = analyze_frame(&fb);
        assert!(m.is_blank, "all-space frame must be is_blank");
        assert!(m.single_glyph, "all-space frame is single-glyph");
    }

    #[test]
    fn single_glyph_frame_is_flagged() {
        let mut fb = Framebuffer::new(6, 6);
        for row in 0..6 {
            for col in 0..6 {
                fb.set(col, row, cell('#', 0x00FFFFFF, 0x00101010, 3.0));
            }
        }
        let m = analyze_frame(&fb);
        assert!(m.single_glyph, "uniform-glyph frame must set single_glyph");
        assert!(!m.is_blank, "a '#'-filled frame is not blank");
    }

    #[test]
    fn depth_gradient_with_matching_luma_has_strong_spearman() {
        // A column-wise depth gradient with a luma gradient that tracks it: the
        // Spearman correlation must be strongly negative (near depth → bright,
        // far depth → dark). We build fg brightness inversely with depth.
        let cols = 16;
        let rows = 4;
        let mut fb = Framebuffer::new(cols, rows);
        for row in 0..rows {
            for col in 0..cols {
                let depth = 1.0 + col as f32; // grows left→right
                                              // brightness falls with depth
                let v = (255.0 * (1.0 - col as f32 / cols as f32)).round() as u32;
                let fg = (v << 16) | (v << 8) | v;
                fb.set(col, row, cell('#', fg, 0x00000000, depth));
            }
        }
        let m = analyze_frame(&fb);
        assert!(
            m.depth_luma_spearman < -0.8,
            "depth↔luma should be strongly negative, got {}",
            m.depth_luma_spearman
        );
    }

    #[test]
    fn spearman_perfect_monotonic() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = [2.0, 4.0, 6.0, 8.0, 10.0];
        assert!((spearman(&xs, &ys) - 1.0).abs() < 1e-5);
        let zs = [10.0, 8.0, 6.0, 4.0, 2.0];
        assert!((spearman(&xs, &zs) + 1.0).abs() < 1e-5);
    }

    /// A clearly visible cell: bright fg over dark bg.
    fn visible_cell(depth: f32) -> Cell {
        cell('#', 0x00FFFFFF, 0x00101010, depth)
    }

    // The near/far split is the median of finite depths. To get a clean,
    // unambiguous partition in synthetic frames we use a continuous depth ramp
    // (each column a distinct depth) so exactly one well-defined median exists,
    // then assign invisibility strictly by `depth <= split` vs `depth > split`.
    // `near_far_invisible` is exercised directly for the boundary logic; the
    // full `analyze_frame` path is covered by the fog/infinite-depth cases.

    #[test]
    fn invisible_near_band_high_far_zero() {
        // Near band (depth ≤ median) all invisible; far band all visible.
        let mut cells: Vec<(f32, bool)> = Vec::new();
        for d in 1..=21 {
            let depth = d as f32;
            // median of 1..=21 is 11; near = depth <= 11.
            let invis = depth <= 11.0;
            cells.push((depth, invis));
        }
        let (near, far, split) = near_far_invisible(&mut cells);
        assert!(
            (split - 11.0).abs() < 1e-3,
            "median split should be 11, got {split}"
        );
        assert!(near > 99.0, "near band should be all invisible, got {near}");
        assert!(far < 1.0, "far band should be visible, got {far}");
    }

    #[test]
    fn invisible_far_band_high_near_zero() {
        // The reverse: near visible, far invisible.
        let mut cells: Vec<(f32, bool)> = Vec::new();
        for d in 1..=21 {
            let depth = d as f32;
            let invis = depth > 11.0; // only the far band collapses
            cells.push((depth, invis));
        }
        let (near, far, split) = near_far_invisible(&mut cells);
        assert!(
            (split - 11.0).abs() < 1e-3,
            "median split should be 11, got {split}"
        );
        assert!(near < 1.0, "near band should be visible, got {near}");
        assert!(far > 99.0, "far band should be all invisible, got {far}");
    }

    #[test]
    fn fog_case_all_far_invisible_keeps_near_low() {
        // A fog-like frame through the full analyze_frame path: near cells read
        // fine, but everything beyond the median has collapsed toward a fog
        // colour (fg≈bg). The combined metric looks alarming (~half the frame
        // invisible) yet `near` stays ~0 — the whole point of the split.
        //
        // Continuous depth ramp (distinct per column) so the median is the
        // mid-column depth and the fog boundary aligns with it exactly.
        let cols = 21; // odd → unambiguous median at the centre column
        let rows = 4;
        let mut fb = Framebuffer::new(cols, rows);
        for row in 0..rows {
            for col in 0..cols {
                let depth = 1.0 + col as f32; // 1.0 ..= 21.0; median = 11.0 at col 10
                let far = depth > 11.0; // strictly beyond the median
                let c = if far {
                    cell('#', 0x00606060, 0x00606060, depth) // fog convergence fg==bg
                } else {
                    visible_cell(depth)
                };
                fb.set(col, row, c);
            }
        }
        let m = analyze_frame(&fb);
        assert!(
            (m.invis_near_depth - 11.0).abs() < 1e-3,
            "split should be the median depth 11.0, got {}",
            m.invis_near_depth
        );
        assert!(
            m.invisible_glyph_pct > 40.0,
            "combined metric should look alarming, got {}",
            m.invisible_glyph_pct
        );
        assert!(
            m.invisible_pct_near < 1.0,
            "near band must stay clean despite fog, got {}",
            m.invisible_pct_near
        );
        assert!(
            m.invisible_pct_far > 99.0,
            "far band carries the fog invisibility, got {}",
            m.invisible_pct_far
        );
    }

    #[test]
    fn invis_split_ignores_infinite_depth_cells() {
        // Sky/floor cells (INFINITY depth) are excluded from the split entirely.
        let mut fb = Framebuffer::new(4, 2);
        // Row 0: finite-depth visible near cells.
        for col in 0..4 {
            fb.set(col, 0, visible_cell(3.0));
        }
        // Row 1: INFINITY-depth invisible cells — must NOT pollute near/far.
        for col in 0..4 {
            fb.set(col, 1, cell('#', 0x00505050, 0x00505050, f32::INFINITY));
        }
        let m = analyze_frame(&fb);
        assert!(
            (m.invis_near_depth - 3.0).abs() < 1e-3,
            "split should be the median of FINITE depths (3.0), got {}",
            m.invis_near_depth
        );
        assert!(
            m.invisible_pct_near < 1.0 && m.invisible_pct_far < 1.0,
            "infinite-depth invisible cells must not enter near/far: near={} far={}",
            m.invisible_pct_near,
            m.invisible_pct_far
        );
    }

    #[test]
    fn shimmer_zero_for_identical_frames() {
        let mut fb = Framebuffer::new(4, 4);
        for row in 0..4 {
            for col in 0..4 {
                fb.set(col, row, cell('#', 0x00FF8800, 0x00000000, 2.0));
            }
        }
        let a = fb.clone();
        let b = fb.clone();
        let o = orbit_shimmer(&[&a, &b], 0.3);
        assert_eq!(o.pairs, 1);
        assert!(
            o.shimmer < 1e-5,
            "identical frames must not shimmer, got {}",
            o.shimmer
        );
    }
}
