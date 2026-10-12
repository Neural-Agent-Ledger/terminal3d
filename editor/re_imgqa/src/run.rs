//! Per-level scheduler — §7.1.
//!
//! Frames are fully independent: the [`Map`] is built once per level and shared
//! immutably across workers; each worker owns its own [`Framebuffer`] and writes
//! only its own slot. A `std::thread::scope` + `AtomicUsize` work-stealing index
//! over the pose list saturates all cores with **no external crate**.
//!
//! Determinism is preserved: results are collected by pose index, so thread
//! scheduling never changes the output.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use re_core::Framebuffer;
use re_render::raycast;
use re_world::Map;

use crate::metrics::{analyze_frame, FrameMetrics};
use crate::orbit::OrbitPose;
use crate::timing::{ScopedTimer, StageTimers};

/// One analyzed pose result: its identity + metrics, collected by pose index.
#[derive(Clone, Debug)]
pub struct FrameResult {
    /// The pose's stable index, matching [`OrbitPose::index`].
    pub index: usize,
    /// The pose's short id, matching [`OrbitPose::id`].
    pub id: String,
    /// True for the level's authored spawn pose.
    pub is_spawn: bool,
    /// The computed metrics for this pose's frame.
    pub metrics: FrameMetrics,
}

/// Raycast + analyze every pose of `poses` against `map` at `cols`×`rows`, using
/// `threads` workers. Returns results ordered by pose index (deterministic).
///
/// Each worker steals the next pose via the shared atomic counter, raycasts into
/// its OWN framebuffer, analyzes, and parks the result in the shared slot at its
/// pose index. Stage nanos accumulate into `timers`.
pub fn run_level(
    map: &Map,
    poses: &[OrbitPose],
    cols: usize,
    rows: usize,
    threads: usize,
    timers: &StageTimers,
) -> Vec<FrameResult> {
    let n = poses.len();
    let slots: Vec<Mutex<Option<FrameResult>>> = (0..n).map(|_| Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);
    let threads = threads.max(1);

    std::thread::scope(|scope| {
        for _ in 0..threads {
            let next = &next;
            let slots = &slots;
            scope.spawn(move || {
                // Per-worker framebuffer, reused across the poses this worker steals.
                let mut fb = Framebuffer::new(cols, rows);
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n {
                        break;
                    }
                    let op = &poses[i];

                    let t = ScopedTimer::start();
                    fb.clear();
                    raycast(map, &op.pose, &mut fb);
                    timers.add_raycast(t.stop());

                    let t = ScopedTimer::start();
                    let metrics = analyze_frame(&fb);
                    timers.add_analyze(t.stop());

                    let res = FrameResult {
                        index: op.index,
                        id: op.id.clone(),
                        is_spawn: op.is_spawn,
                        metrics,
                    };
                    *slots[i].lock().unwrap() = Some(res);
                }
            });
        }
    });

    // Collect deterministically by pose index (slots are already index-ordered).
    slots
        .into_iter()
        .map(|m| {
            m.into_inner()
                .unwrap()
                .expect("every pose produced a result")
        })
        .collect()
}

/// Serial reference implementation (used by tests + `--threads 1` parity). Same
/// output as [`run_level`] for any thread count.
pub fn run_level_serial(
    map: &Map,
    poses: &[OrbitPose],
    cols: usize,
    rows: usize,
) -> Vec<FrameResult> {
    let mut fb = Framebuffer::new(cols, rows);
    poses
        .iter()
        .map(|op| {
            fb.clear();
            raycast(map, &op.pose, &mut fb);
            FrameResult {
                index: op.index,
                id: op.id.clone(),
                is_spawn: op.is_spawn,
                metrics: analyze_frame(&fb),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orbit::{generate, OrbitParams, PitchMode};
    use re_world::{test_room, CameraPose};

    #[test]
    fn parallel_matches_serial() {
        let map = test_room();
        let spawn = CameraPose::spawn_in_test_room();
        let params = OrbitParams {
            positions: 40,
            yaws: 16,
            pitch: PitchMode::Three,
        };
        let poses = generate(&map, spawn, &params);

        let timers = StageTimers::new();
        let par = run_level(&map, &poses, 80, 24, 4, &timers);
        let ser = run_level_serial(&map, &poses, 80, 24);

        assert_eq!(par.len(), ser.len());
        for (p, s) in par.iter().zip(ser.iter()) {
            assert_eq!(p.index, s.index, "results must be index-ordered");
            assert_eq!(p.id, s.id);
            assert_eq!(
                p.metrics, s.metrics,
                "parallel metrics must match serial for pose {}",
                p.id
            );
        }
    }

    #[test]
    fn threads_one_matches_serial() {
        let map = test_room();
        let spawn = CameraPose::spawn_in_test_room();
        let params = OrbitParams {
            positions: 10,
            yaws: 8,
            pitch: PitchMode::LevelOnly,
        };
        let poses = generate(&map, spawn, &params);

        let timers = StageTimers::new();
        let one = run_level(&map, &poses, 60, 20, 1, &timers);
        let ser = run_level_serial(&map, &poses, 60, 20);
        for (a, b) in one.iter().zip(ser.iter()) {
            assert_eq!(a.metrics, b.metrics, "--threads 1 must equal serial");
        }
    }
}
