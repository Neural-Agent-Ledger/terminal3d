//! Per-cell [`Shape`] storage for the voxel [`Map`](crate::Map).
//!
//! # Design
//!
//! Shapes default to [`Shape::Full`] — the solid cube that is byte-identical
//! to every cell before this module existed.  A world where no shape is ever
//! written via [`ShapeStore::set`] has zero heap overhead beyond the `Vec` base
//! pointer; cells are `Full` on read without materializing any storage.
//!
//! We use a **parallel sparse `HashMap`** keyed by flat grid index.  Every cell
//! whose shape has never been set (or was set back to `Full`) is absent from the
//! map and returns `Full` on read.  This keeps the O(cells) overhead zero for
//! all existing levels.
//!
//! Access is O(1) amortized (HashMap lookup).  If a future profiling pass shows
//! the HashMap is a bottleneck at high cell counts, the store can be upgraded to
//! a parallel dense `Vec<Shape>` with a run-length compressor — but the public
//! API is stable either way.

use re_core::Shape;
use std::collections::HashMap;

/// Sparse per-cell shape store.  The cell index key is `x + sx*(y + sy*z)` —
/// the same flat layout the [`DenseStore`](crate::store::DenseStore) uses, so
/// the indices are always consistent.
#[derive(Clone, Debug, Default)]
pub struct ShapeStore {
    /// Non-`Full` cells only.  Absent ⇒ `Shape::Full`.
    inner: HashMap<usize, Shape>,
}

impl ShapeStore {
    /// Read the shape for flat index `idx`.  Absent ⇒ [`Shape::Full`].
    #[inline]
    pub fn get(&self, idx: usize) -> Shape {
        self.inner.get(&idx).copied().unwrap_or(Shape::Full)
    }

    /// Write a shape for flat index `idx`.  Writing [`Shape::Full`] removes the
    /// entry (restoring the default) so the map never accumulates redundant
    /// `Full` entries that waste memory.
    #[inline]
    pub fn set(&mut self, idx: usize, shape: Shape) {
        if shape == Shape::Full {
            self.inner.remove(&idx);
        } else {
            self.inner.insert(idx, shape);
        }
    }

    /// True if no non-`Full` shape has been set — the common case for all
    /// existing maps (levels without any wedge / ramp geometry).
    #[inline]
    pub fn is_all_full(&self) -> bool {
        self.inner.is_empty()
    }

    /// Number of cells whose shape has been explicitly set (non-`Full`).
    #[inline]
    pub fn non_full_count(&self) -> usize {
        self.inner.len()
    }
}
