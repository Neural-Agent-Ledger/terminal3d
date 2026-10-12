//! EYE SWARM — many procedural [`Eye`]s of VARIED SIZE
//! scattered over a rect, each with its OWN blink/saccade/breathe/shimmer
//! frequencies (drawn from its own sub-seed, so no two eyes in the swarm
//! pulse together), some looking at a shared target (the cursor, if the
//! caller supplies one via [`EyeSwarm::with_target`]), some locked onto a
//! specific neighbour, the rest wandering on their own — plus an occasional
//! synchronized beat where every eye in the swarm snaps to look at the SAME
//! place at once ("all look at you").
//!
//! ## Size classes and depth
//!
//! Every member rolls one of three `SizeClass`es — a couple of LARGE eyes,
//! more MEDIUM ones, and the most numerous SMALL ones — each a different
//! fraction of the grid cell's own extent (see `SIZE_CLASS_*`). Members are
//! placed on a `cols_n × rows_n` grid (jittered off-centre) so the field
//! still reads as scattered rather than random clumping, but a LARGE member's
//! bounding rect is deliberately allowed to spill into its neighbours' cells
//! — that overlap is the point. [`AsciiEffect::paint`] paints members in
//! DEPTH order, largest (furthest back) first, smallest (nearest) last: since
//! [`Eye`]'s own shader only ever writes a cell inside its lid opening
//! (everywhere else `shade` returns `None` and the pixel is left untouched —
//! see `eye::shade`'s early-out), painting back-to-front makes a smaller
//! foreground eye correctly occlude the part of a larger background eye it
//! overlaps, and leaves the rest of that larger eye visible — real
//! depth-sorted compositing, not a Z-buffer, because nothing here needs one.

use crate::eye::{Eye, EyeVariant};
use crate::hash::{chance, hash_unit, range};
use crate::{secs, AsciiEffect, Rect};
use re_core::Framebuffer;
use std::time::Duration;

/// Default eye count for [`EyeSwarm::random`]. Exposed so a caller/gallery
/// can size the swarm to its rect via [`EyeSwarm::random_with_count`]. Bumped
/// from the original 12 so a swarm still reads as "many small eyes" even
/// after a few of them roll LARGE and dominate the frame.
pub const DEFAULT_COUNT: usize = 16;

/// Fraction of the swarm that locks its gaze onto the shared target (when
/// one is set) vs. wanders independently vs. stares at a specific neighbour.
const P_LOOK_AT_TARGET: f32 = 0.35;
const P_LOOK_AT_NEIGHBOR: f32 = 0.20;

/// How much a sub-eye's grid cell it may be nudged off-centre by, as a
/// fraction of the cell size. LARGE eyes deliberately spill past their own
/// cell (see `SizeClass::cell_fill_frac`), so this is what actually produces
/// the overlap with neighbours the module doc describes.
const JITTER_FRAC: f32 = 0.55;

/// A member's `SizeClass` sets how large a fraction of its OWN grid cell its
/// bounding rect fills. `> 1.0` (LARGE) is deliberate: it is what lets a big
/// eye's rect spill into neighbouring cells and overlap them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SizeClass {
    Large,
    Medium,
    Small,
}

impl SizeClass {
    /// Probability table, checked in order: `roll < P_LARGE` -> Large, next
    /// `P_MEDIUM` -> Medium, else Small. A COUPLE large, MORE medium, MOST
    /// small, so the swarm reads as a few dominant eyes among many smaller ones.
    const P_LARGE: f32 = 0.14;
    const P_MEDIUM: f32 = 0.42;

    fn roll(u: f32) -> Self {
        if u < Self::P_LARGE {
            SizeClass::Large
        } else if u < Self::P_LARGE + Self::P_MEDIUM {
            SizeClass::Medium
        } else {
            SizeClass::Small
        }
    }

    /// Fraction of the member's grid cell its bounding rect fills.
    fn cell_fill_frac(self) -> f32 {
        match self {
            SizeClass::Large => 1.65,
            SizeClass::Medium => 0.85,
            SizeClass::Small => 0.42,
        }
    }
}

/// Synchronized "all look at you" beat: period and duration, each its own
/// randomized frequency per swarm seed.
const SYNC_PERIOD_RANGE: (f32, f32) = (5.0, 13.0);
const SYNC_DURATION_RANGE: (f32, f32) = (0.5, 1.3);

const S_JITTER_X: u64 = 2;
const S_JITTER_Y: u64 = 3;
const S_BEHAVIOR: u64 = 4;
const S_NEIGHBOR: u64 = 5;
const S_SYNC_PERIOD: u64 = 6;
const S_SYNC_DURATION: u64 = 7;
const S_SYNC_PHASE: u64 = 8;
const S_VARIANT_PICK: u64 = 9;
const S_SIZE_CLASS: u64 = 10;

#[derive(Clone, Copy, Debug)]
enum Behavior {
    Autonomous,
    LookAtTarget,
    LookAtNeighbor(usize),
}

struct Member {
    eye: Eye,
    rect: Rect,
    center: (f32, f32),
    behavior: Behavior,
    /// The member's own bounding rect's `min_extent` — the depth-sort key
    /// (larger = painted earlier = further back). Cached rather than
    /// re-derived from `rect` at sort time purely for readability.
    size: usize,
}

/// A scattered field of independent procedural eyes. Not `Copy` (owns a
/// `Vec`), but every field is otherwise the same "resolved once, read many"
/// shape as [`Eye`] — `paint` is still a pure function of `(self, t)`.
pub struct EyeSwarm {
    members: Vec<Member>,
    sync_period_s: f32,
    sync_duration_s: f32,
    sync_phase: f32,
    target: Option<(f32, f32)>,
}

impl EyeSwarm {
    /// [`Self::random_with_count`] at [`DEFAULT_COUNT`].
    #[must_use]
    pub fn random(seed: u64, rect: Rect) -> Self {
        Self::random_with_count(seed, rect, DEFAULT_COUNT)
    }

    /// Build a deterministic swarm of `count` eyes tiled across `rect`, each
    /// with its own jittered position, size class, variant roll and behavior
    /// (autonomous, tracking a shared target, or tracking a neighbour).
    #[must_use]
    pub fn random_with_count(seed: u64, rect: Rect, count: usize) -> Self {
        let count = count.max(1);
        let aspect = rect.cols as f32 / rect.rows.max(1) as f32;
        let cols_n = ((count as f32 * aspect).sqrt().ceil() as usize).max(1);
        let rows_n = count.div_ceil(cols_n).max(1);
        let cell_w = (rect.cols as f32 / cols_n as f32).max(1.0);
        let cell_h = (rect.rows as f32 / rows_n as f32).max(1.0);

        let mut members = Vec::with_capacity(count);
        let mut centers = Vec::with_capacity(count);
        for i in 0..count {
            let gx = (i % cols_n) as f32;
            let gy = (i / cols_n) as f32;
            let jx = range(seed, i as u64, S_JITTER_X, -JITTER_FRAC, JITTER_FRAC) * cell_w;
            let jy = range(seed, i as u64, S_JITTER_Y, -JITTER_FRAC, JITTER_FRAC) * cell_h;
            let ccol = rect.col0 as f32 + (gx + 0.5) * cell_w + jx;
            let crow = rect.row0 as f32 + (gy + 0.5) * cell_h + jy;
            centers.push((ccol, crow));
        }

        for (i, &(ccol, crow)) in centers.iter().enumerate() {
            let size_class = SizeClass::roll(hash_unit(seed, i as u64, S_SIZE_CLASS) as f32);
            let fill = size_class.cell_fill_frac();
            let w = (cell_w * fill).max(3.0) as usize;
            let h = (cell_h * fill).max(2.0) as usize;
            let sub_col0 = (ccol - w as f32 * 0.5).max(0.0) as usize;
            let sub_row0 = (crow - h as f32 * 0.5).max(0.0) as usize;
            let sub_rect = Rect::new(sub_col0, sub_row0, w.max(1), h.max(1));
            let sub_seed =
                hash_unit(seed, i as u64, 0).to_bits() ^ seed.rotate_left((i as u32) % 61);
            let mut eye = Eye::random(sub_seed, sub_rect);
            if chance(seed, i as u64, S_VARIANT_PICK, 0.5) {
                let v = match (hash_unit(seed, i as u64, S_VARIANT_PICK + 1) * 5.0) as u8 {
                    0 => EyeVariant::Human,
                    1 => EyeVariant::CatSlit,
                    2 => EyeVariant::Reptile,
                    3 => EyeVariant::Cyber,
                    _ => EyeVariant::Insect,
                };
                eye = eye.with_variant(v);
            }

            let roll = hash_unit(seed, i as u64, S_BEHAVIOR) as f32;
            let behavior = if roll < P_LOOK_AT_TARGET {
                Behavior::LookAtTarget
            } else if roll < P_LOOK_AT_TARGET + P_LOOK_AT_NEIGHBOR && count > 1 {
                let mut idx =
                    (hash_unit(seed, i as u64, S_NEIGHBOR) * count as f64) as usize % count;
                if idx == i {
                    idx = (idx + 1) % count;
                }
                Behavior::LookAtNeighbor(idx)
            } else {
                Behavior::Autonomous
            };

            members.push(Member {
                eye,
                rect: sub_rect,
                center: (ccol, crow),
                behavior,
                size: sub_rect.min_extent(),
            });
        }

        EyeSwarm {
            members,
            sync_period_s: range(
                seed,
                0,
                S_SYNC_PERIOD,
                SYNC_PERIOD_RANGE.0,
                SYNC_PERIOD_RANGE.1,
            ),
            sync_duration_s: range(
                seed,
                0,
                S_SYNC_DURATION,
                SYNC_DURATION_RANGE.0,
                SYNC_DURATION_RANGE.1,
            ),
            sync_phase: range(seed, 0, S_SYNC_PHASE, 0.0, 1.0),
            target: None,
        }
    }

    /// A shared look-at target (e.g. the mouse cell) some members track and
    /// every member snaps to during a synchronized beat.
    #[must_use]
    pub fn with_target(mut self, target: Option<(f32, f32)>) -> Self {
        self.target = target;
        self
    }

    /// Number of eyes in the swarm.
    #[must_use]
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// `true` iff the swarm has no members.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// `true` iff `t` falls inside a synchronized "all eyes look at you"
    /// beat — exposed for tests/the gallery to pick an interesting frame.
    #[must_use]
    pub fn in_sync_beat(&self, t: Duration) -> bool {
        let period = self.sync_period_s.max(0.1);
        let phase = (secs(t) / period + self.sync_phase).rem_euclid(1.0);
        phase < (self.sync_duration_s / period).clamp(0.0, 1.0)
    }

    fn resolved_target(&self, i: usize, t: Duration) -> Option<(f32, f32)> {
        if self.in_sync_beat(t) {
            return Some(self.target.unwrap_or(self.members[i].center));
        }
        match self.members[i].behavior {
            Behavior::Autonomous => None,
            Behavior::LookAtTarget => self.target,
            Behavior::LookAtNeighbor(j) => Some(self.members[j].center),
        }
    }
}

impl AsciiEffect for EyeSwarm {
    fn name(&self) -> &'static str {
        "eye_swarm"
    }

    fn paint(&self, fb: &mut Framebuffer, rect: Rect, t: Duration) {
        // Depth order: largest (furthest back) first, smallest (nearest)
        // last, so a small foreground eye correctly occludes the part of a
        // large background eye it overlaps (see the module doc). Ties break
        // on index for determinism.
        let mut order: Vec<usize> = (0..self.members.len()).collect();
        order.sort_by(|&a, &b| {
            self.members[b]
                .size
                .cmp(&self.members[a].size)
                .then(a.cmp(&b))
        });
        for i in order {
            let m = &self.members[i];
            let eye = m.eye.with_target(self.resolved_target(i, t));
            // Clip the sub-eye's own rect to the caller's rect so a swarm
            // painted into a sub-region of a larger canvas never bleeds out
            // (the sub-rects were built from `rect` at construction, so this
            // is normally a no-op; it only matters if the caller re-paints
            // the same swarm into a smaller rect than it was built for).
            let clipped = clip_rect(m.rect, rect);
            if clipped.cols == 0 || clipped.rows == 0 {
                continue;
            }
            eye.paint(fb, clipped, t);
        }
    }
}

fn clip_rect(inner: Rect, outer: Rect) -> Rect {
    let col0 = inner.col0.max(outer.col0);
    let row0 = inner.row0.max(outer.row0);
    let col1 = (inner.col0 + inner.cols).min(outer.col0 + outer.cols);
    let row1 = (inner.row0 + inner.rows).min(outer.row0 + outer.rows);
    Rect::new(
        col0,
        row0,
        col1.saturating_sub(col0),
        row1.saturating_sub(row0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const RECT: Rect = Rect {
        col0: 0,
        row0: 0,
        cols: 120,
        rows: 40,
    };

    #[test]
    fn random_builds_the_requested_count() {
        let swarm = EyeSwarm::random_with_count(1, RECT, 10);
        assert_eq!(swarm.len(), 10);
    }

    #[test]
    fn determinism_same_seed_same_members() {
        let a = EyeSwarm::random(4242, RECT);
        let b = EyeSwarm::random(4242, RECT);
        assert_eq!(a.members.len(), b.members.len());
        for (ma, mb) in a.members.iter().zip(b.members.iter()) {
            assert_eq!(ma.center, mb.center);
            assert_eq!(ma.rect, mb.rect);
        }
    }

    #[test]
    fn different_seeds_scatter_differently() {
        let a = EyeSwarm::random(1, RECT);
        let b = EyeSwarm::random(2, RECT);
        let centers_a: Vec<_> = a.members.iter().map(|m| m.center).collect();
        let centers_b: Vec<_> = b.members.iter().map(|m| m.center).collect();
        assert_ne!(centers_a, centers_b);
    }

    #[test]
    fn sync_beat_is_periodic_and_bounded() {
        let swarm = EyeSwarm::random(9, RECT);
        let mut any_in = false;
        let mut any_out = false;
        for ms in (0..20000u64).step_by(137) {
            if swarm.in_sync_beat(Duration::from_millis(ms)) {
                any_in = true;
            } else {
                any_out = true;
            }
        }
        assert!(
            any_in,
            "swarm never syncs across 20s — period/duration mis-tuned"
        );
        assert!(any_out, "swarm never releases the sync beat");
    }

    #[test]
    fn during_a_sync_beat_every_member_targets_the_same_point() {
        let swarm = EyeSwarm::random(9, RECT).with_target(Some((60.0, 20.0)));
        let t = (0..20000u64)
            .step_by(50)
            .map(Duration::from_millis)
            .find(|&t| swarm.in_sync_beat(t))
            .expect("no sync beat found in 20s");
        for i in 0..swarm.len() {
            assert_eq!(swarm.resolved_target(i, t), Some((60.0, 20.0)));
        }
    }

    #[test]
    fn paint_does_not_panic_and_stays_inside_its_rect() {
        let mut fb = Framebuffer::new(140, 44);
        let rect = Rect::new(4, 2, 130, 40);
        let swarm = EyeSwarm::random(3, rect).with_target(Some((70.0, 22.0)));
        for ms in [0u64, 400, 4000, 12000] {
            swarm.paint(&mut fb, rect, Duration::from_millis(ms));
        }
        for row in 0..44 {
            for col in 0..140 {
                if !rect.contains(col as i64, row as i64) {
                    assert_eq!(*fb.get(col, row).unwrap(), re_core::Cell::default());
                }
            }
        }
    }
}
