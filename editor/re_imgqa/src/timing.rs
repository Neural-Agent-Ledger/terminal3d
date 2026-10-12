//! Stage timing harness — §7.1.
//!
//! Per-worker nanosecond accumulators for the three pipeline stages
//! (`raycast` / `analyze` / `html`), merged into a [`Timing`] summary embedded in
//! `report.json` and printed to the console. No external crate — `std::time`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Thread-safe stage accumulators shared across the worker pool. Each worker
/// adds its own elapsed nanos; the atomics make the merge lock-free.
#[derive(Default)]
pub struct StageTimers {
    /// Accumulated raycast-stage nanoseconds across all workers.
    pub raycast_ns: AtomicU64,
    /// Accumulated analyze-stage nanoseconds across all workers.
    pub analyze_ns: AtomicU64,
    /// Accumulated HTML-writing-stage nanoseconds across all workers.
    pub html_ns: AtomicU64,
}

impl StageTimers {
    /// A fresh, zeroed set of stage accumulators.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add `ns` to the raycast-stage total (lock-free).
    #[inline]
    pub fn add_raycast(&self, ns: u64) {
        self.raycast_ns.fetch_add(ns, Ordering::Relaxed);
    }
    /// Add `ns` to the analyze-stage total (lock-free).
    #[inline]
    pub fn add_analyze(&self, ns: u64) {
        self.analyze_ns.fetch_add(ns, Ordering::Relaxed);
    }
    /// Add `ns` to the HTML-writing-stage total (lock-free).
    #[inline]
    pub fn add_html(&self, ns: u64) {
        self.html_ns.fetch_add(ns, Ordering::Relaxed);
    }

    /// Snapshot the accumulated stage totals.
    pub fn snapshot(&self) -> StageTotals {
        StageTotals {
            raycast_ns: self.raycast_ns.load(Ordering::Relaxed),
            analyze_ns: self.analyze_ns.load(Ordering::Relaxed),
            html_ns: self.html_ns.load(Ordering::Relaxed),
        }
    }
}

/// A plain snapshot of stage nanos (summed across threads).
#[derive(Clone, Copy, Debug, Default)]
pub struct StageTotals {
    /// Total raycast-stage nanoseconds, summed across workers.
    pub raycast_ns: u64,
    /// Total analyze-stage nanoseconds, summed across workers.
    pub analyze_ns: u64,
    /// Total HTML-writing-stage nanoseconds, summed across workers.
    pub html_ns: u64,
}

/// The full timing summary embedded in the report.
#[derive(Clone, Debug, Default)]
pub struct Timing {
    /// Wall-clock total of the whole run (ns).
    pub wall_ns: u64,
    /// Per-level wall time (ns), in level order.
    pub per_level_ns: Vec<(String, u64)>,
    /// Summed stage nanos across all workers/levels.
    pub stages: StageTotals,
    /// Total frames analyzed.
    pub frames_analyzed: u64,
    /// Total photos written.
    pub photos_written: u64,
}

impl Timing {
    /// Frames analyzed per wall-clock second.
    pub fn frames_per_sec(&self) -> f64 {
        if self.wall_ns == 0 {
            0.0
        } else {
            self.frames_analyzed as f64 / (self.wall_ns as f64 / 1e9)
        }
    }
}

/// A scoped timer: returns elapsed nanoseconds when `.stop()` is called.
pub struct ScopedTimer {
    start: Instant,
}

impl ScopedTimer {
    /// Begin timing now.
    #[inline]
    pub fn start() -> Self {
        ScopedTimer {
            start: Instant::now(),
        }
    }
    /// End timing and return the elapsed nanoseconds.
    #[inline]
    pub fn stop(self) -> u64 {
        self.start.elapsed().as_nanos() as u64
    }
}

/// Format nanoseconds as a human ms string.
pub fn fmt_ms(ns: u64) -> String {
    format!("{:.2}ms", ns as f64 / 1e6)
}
