//! `transport` — the musical beat grid. **Pure logic, no audio device.**
//!
//! kira owns the *sample-accurate clock* (a `ClockSpeed::TicksPerMinute` clock
//! whose ticks are our subdivisions). This module owns the *musical meaning* of
//! those ticks: how many ticks make a beat and a bar, which ticks are "strong"
//! (bar downbeats) vs "weak" (other beat boundaries), and — the load-bearing
//! piece — [`next_tick`], which decides *which* tick a sound should be scheduled
//! on given where we are right now.
//!
//! Nothing here touches kira or an audio device; it is unit-testable with no
//! device, mirroring the raycaster/present split used elsewhere in TERMINAL STATE.

/// Whether a tick is a bar downbeat ([`BeatClass::Strong`]) or another beat
/// boundary ([`BeatClass::Weak`]). The [`crate::policy`] table maps each sound
/// to the class of tick it should land on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BeatClass {
    /// A bar downbeat (tick 0 of each bar). Used for "important" events.
    Strong,
    /// A beat boundary that is *not* the downbeat. Used for busier events.
    Weak,
}

/// The subdivision grid: how a bar is carved into ticks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grid {
    /// Beats in one bar (the time-signature numerator). Default 4.
    pub beats_per_bar: u32,
    /// Ticks per beat: 2 = eighth-note grid, 4 = sixteenth-note grid. Default 2.
    pub subdivision: u32,
}

impl Grid {
    /// Ticks in one bar = `beats_per_bar * subdivision`.
    #[inline]
    pub fn ticks_per_bar(self) -> u64 {
        (self.beats_per_bar.max(1) as u64) * (self.subdivision.max(1) as u64)
    }

    /// Ticks in one beat = `subdivision`.
    #[inline]
    pub fn ticks_per_beat(self) -> u64 {
        self.subdivision.max(1) as u64
    }

    /// Classify an absolute integer tick. A tick that is *not* a beat boundary
    /// (an off-beat subdivision) is neither strong nor weak → `None`.
    #[inline]
    pub fn classify(self, tick: u64) -> Option<BeatClass> {
        if tick.is_multiple_of(self.ticks_per_bar()) {
            Some(BeatClass::Strong)
        } else if tick.is_multiple_of(self.ticks_per_beat()) {
            Some(BeatClass::Weak)
        } else {
            None
        }
    }
}

impl Default for Grid {
    fn default() -> Self {
        Grid {
            beats_per_bar: 4,
            subdivision: 2,
        }
    }
}

/// The full transport configuration, mirroring the spec's defaults. `playback`
/// reads `bpm`/`grid` to drive the kira clock speed and `snap_window`/
/// `humanize_ms` to schedule one-shots.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transport {
    /// Beats per minute. Default 90.0. Clock speed = `bpm * subdivision`.
    pub bpm: f64,
    /// The subdivision grid.
    pub grid: Grid,
    /// Snap window as a fraction of one tick (0..=1); default 0.18. See
    /// [`next_tick`].
    pub snap_window: f64,
    /// Humanize jitter in milliseconds (± about this much); default 12.0.
    /// 0 disables. Applied in `playback`, never to `Immediate` sounds.
    pub humanize_ms: f64,
}

impl Transport {
    /// Clock speed in ticks-per-minute for kira's `ClockSpeed::TicksPerMinute`.
    /// One tick is one subdivision, so `bpm * subdivision`.
    #[inline]
    pub fn ticks_per_minute(self) -> f64 {
        self.bpm * self.grid.subdivision.max(1) as f64
    }
}

impl Default for Transport {
    fn default() -> Self {
        Transport {
            bpm: 90.0,
            grid: Grid::default(),
            snap_window: 0.18,
            humanize_ms: 12.0,
        }
    }
}

/// A position on the tick timeline: an integer `tick` plus the `fraction`
/// `[0.0, 1.0)` elapsed into the *next* tick. Mirrors kira's `ClockTime`
/// (`ticks` + `fraction`) but with no clock identity, so it is pure data.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TickPos {
    /// The most recent whole tick that has elapsed.
    pub tick: u64,
    /// How far into the gap to the next tick we are, in `[0.0, 1.0)`.
    pub fraction: f64,
}

impl TickPos {
    /// Construct a tick position. `fraction` is clamped into `[0.0, 1.0)`.
    #[inline]
    pub fn new(tick: u64, fraction: f64) -> Self {
        let fraction = if fraction.is_finite() {
            fraction.clamp(0.0, 1.0 - f64::EPSILON)
        } else {
            0.0
        };
        TickPos { tick, fraction }
    }
}

// ─────────────────────────────── next_tick ──────────────────────────────────
//
// 🎓 LEARNING-MODE CONTRIBUTION POINT.
//
// This small function is the heart of the "sounds land on the beat" behaviour.
// It is deliberately isolated and short so it can be tuned without touching kira
// or the rest of the audio stack. The unit tests below pin its contract; change
// the body freely as long as they stay green (or update them deliberately).

/// Choose which absolute tick a `class` event should be scheduled on, given the
/// current position `now`.
///
/// ## Snap-window semantics
/// `snap_window` is a fraction of one tick (e.g. `0.18` = 18 % of a tick). We
/// look at the *nearest* tick of the requested `class` — which may be just
/// behind us (we slightly overshot it) or just ahead. If that nearest tick is
/// within `snap_window` of `now`, the event "snaps" onto it and plays almost
/// immediately (the human ear forgives a sub-tick early/late nudge). Otherwise
/// we wait and schedule on the *next* tick of that class so the event still
/// lands cleanly on the grid instead of in the gap.
///
/// ## Strong vs weak
/// [`BeatClass::Strong`] ticks are bar downbeats (`tick % ticks_per_bar == 0`);
/// [`BeatClass::Weak`] ticks are the other beat boundaries. We search forward
/// for the next tick of the requested class and also consider the most recent
/// past tick of that class for the snap-back case, then pick whichever the
/// snap-window selects. The returned tick is always `>= now.tick` (we never
/// schedule into the past), and bar wraparound is automatic because we work in
/// absolute tick numbers.
pub fn next_tick(now: TickPos, class: BeatClass, grid: Grid, snap_window: f64) -> u64 {
    let period = period_for(class, grid); // tick spacing of this class
    let phase = phase_for(class, grid); // first tick of the class within a bar
                                        // The current continuous position, measured in ticks.
    let pos = now.tick as f64 + now.fraction;
    // Index of the period-grid slot just at/below us, then the candidate ticks
    // of this class straddling `now`: `prev` (<= pos) and `next` (> prev).
    let k = ((pos - phase as f64) / period as f64).floor();
    let prev = (phase as f64 + k * period as f64).max(phase as f64);
    let next = prev + period as f64;
    // Snap back to `prev` only if it is not in the past (relative to now.tick)
    // and we are within the window after it — that is the "play ~immediately"
    // case. Otherwise schedule on `next`: this covers both being within the
    // window *before* `next` (snap forward) and being in the gap (wait for the
    // grid). Both yield the same tick, so they share one arm.
    let snap = snap_window.clamp(0.0, 1.0) * period as f64;
    let chosen = if prev as u64 >= now.tick && pos - prev <= snap {
        prev as u64
    } else {
        next as u64
    };
    // The weak candidate grid is the *beat* grid, which includes bar downbeats
    // (multiples of `ticks_per_bar`). A `Weak` request must never land on a
    // downbeat (that is `Strong`), so skip downbeat candidates forward to the
    // next eligible weak tick. `Strong` is already exact, so this is a no-op
    // for it.
    if class == BeatClass::Weak {
        let mut t = chosen;
        while grid.classify(t) == Some(BeatClass::Strong) {
            t += period;
        }
        t
    } else {
        chosen
    }
}

/// Tick spacing between consecutive ticks of `class`.
#[inline]
fn period_for(class: BeatClass, grid: Grid) -> u64 {
    match class {
        // Strong ticks recur once per bar.
        BeatClass::Strong => grid.ticks_per_bar(),
        // Weak ticks fall on every beat; the snap math treats the beat grid as
        // the candidate set and `next_tick` skips any downbeat candidate forward
        // (a downbeat classifies as `Strong`, never `Weak`).
        BeatClass::Weak => grid.ticks_per_beat(),
    }
}

/// The first tick of `class` within a bar (its phase offset). Strong starts at
/// the downbeat (0); weak's first non-downbeat beat is one beat in.
#[inline]
fn phase_for(class: BeatClass, grid: Grid) -> u64 {
    match class {
        BeatClass::Strong => 0,
        BeatClass::Weak => grid.ticks_per_beat(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: Grid = Grid {
        beats_per_bar: 4,
        subdivision: 2,
    };
    // With this grid: ticks_per_beat = 2, ticks_per_bar = 8.
    // Strong ticks: 0, 8, 16, ...   Weak ticks (beats): 2, 4, 6, 10, 12, 14, ...

    #[test]
    fn grid_classification() {
        assert_eq!(G.classify(0), Some(BeatClass::Strong));
        assert_eq!(G.classify(8), Some(BeatClass::Strong));
        assert_eq!(G.classify(2), Some(BeatClass::Weak));
        assert_eq!(G.classify(6), Some(BeatClass::Weak));
        assert_eq!(G.classify(1), None); // off-beat subdivision
        assert_eq!(G.classify(3), None);
    }

    #[test]
    fn exactly_on_a_strong_tick_returns_it() {
        // Sitting exactly on tick 8 (a downbeat) → schedule on 8.
        let now = TickPos::new(8, 0.0);
        assert_eq!(next_tick(now, BeatClass::Strong, G, 0.18), 8);
    }

    #[test]
    fn exactly_on_a_weak_tick_returns_it() {
        // Sitting exactly on tick 2 (a weak beat) → schedule on 2.
        let now = TickPos::new(2, 0.0);
        assert_eq!(next_tick(now, BeatClass::Weak, G, 0.18), 2);
    }

    #[test]
    fn within_snap_window_after_tick_snaps_back_to_current() {
        // Just past tick 8 by 0.1 of a strong period (8 ticks) is well inside
        // an 0.18 window → snap back to 8 (play ~immediately), not 16.
        let now = TickPos::new(8, 0.1);
        assert_eq!(next_tick(now, BeatClass::Strong, G, 0.18), 8);
    }

    #[test]
    fn within_snap_window_before_tick_snaps_forward() {
        // Weak period is 2 ticks; window 0.4 → 0.8 ticks. At tick 3, fraction
        // 0.5 we are 0.5 ticks before weak tick 4 (< 0.8) → snap forward to 4.
        let now = TickPos::new(3, 0.5);
        assert_eq!(next_tick(now, BeatClass::Weak, G, 0.4), 4);
    }

    #[test]
    fn outside_snap_window_takes_next_tick_of_class() {
        // Strong period 8 ticks, window 0.18 → 1.44 ticks. At tick 10 (= 2 past
        // the downbeat at 8), we are 2 ticks past 8 (> 1.44) and 6 before 16
        // (> 1.44) → next strong tick is 16.
        let now = TickPos::new(10, 0.0);
        assert_eq!(next_tick(now, BeatClass::Strong, G, 0.18), 16);
    }

    #[test]
    fn weak_skips_to_next_beat_when_in_gap() {
        // At tick 5 (off-beat) with a small window, the nearest weak beats are 4
        // (behind, 1 tick back) and 6 (ahead, 1 tick). Window 0.18*2 = 0.36 ticks
        // excludes both → schedule on the next weak beat, 6.
        let now = TickPos::new(5, 0.0);
        assert_eq!(next_tick(now, BeatClass::Weak, G, 0.18), 6);
    }

    #[test]
    fn bar_boundary_wraparound() {
        // Near the end of bar 0 (downbeat 8 is the next bar): at tick 7 with a
        // window large enough (0.5*8 = 4 ticks) we are 1 tick before 8 → snap to
        // the downbeat 8, which is the next bar's start.
        let now = TickPos::new(7, 0.0);
        assert_eq!(next_tick(now, BeatClass::Strong, G, 0.5), 8);
        // And from tick 9 the next strong tick is the following bar at 16.
        let now2 = TickPos::new(9, 0.0);
        assert_eq!(next_tick(now2, BeatClass::Strong, G, 0.18), 16);
    }

    #[test]
    fn never_schedules_into_the_past() {
        // Whatever the inputs, the chosen tick is at or after the current tick.
        for tick in 0..40u64 {
            for &frac in &[0.0, 0.25, 0.5, 0.75] {
                let now = TickPos::new(tick, frac);
                for class in [BeatClass::Strong, BeatClass::Weak] {
                    let t = next_tick(now, class, G, 0.18);
                    assert!(t >= tick, "scheduled {t} < now {tick} for {class:?}");
                }
            }
        }
    }

    #[test]
    fn weak_request_never_returns_a_strong_tick() {
        // For a wide range of positions and snap windows, a Weak request must
        // never resolve onto a bar downbeat (which classifies as Strong).
        for tick in 0..64u64 {
            for &frac in &[0.0, 0.1, 0.25, 0.5, 0.75, 0.9] {
                for &snap in &[0.0, 0.18, 0.4, 0.5, 0.9] {
                    let now = TickPos::new(tick, frac);
                    let t = next_tick(now, BeatClass::Weak, G, snap);
                    assert_ne!(
                        G.classify(t),
                        Some(BeatClass::Strong),
                        "weak request at tick {tick} frac {frac} snap {snap} \
                         returned strong tick {t}"
                    );
                    // It must still be a real weak beat boundary, not the past.
                    assert_eq!(G.classify(t), Some(BeatClass::Weak));
                    assert!(t >= tick);
                }
            }
        }
    }

    #[test]
    fn weak_at_downbeat_neighbourhood_skips_to_next_weak() {
        // Sitting on/near downbeat 8 with a generous snap that would otherwise
        // pick 8 → must skip forward to the next weak beat, 10.
        let now = TickPos::new(8, 0.0);
        assert_eq!(next_tick(now, BeatClass::Weak, G, 0.9), 10);
        // Just before downbeat 16 (tick 15) with a wide window would snap to 16
        // (Strong) → skip to 18.
        let now2 = TickPos::new(15, 0.0);
        assert_eq!(next_tick(now2, BeatClass::Weak, G, 0.9), 18);
    }

    #[test]
    fn ticks_per_minute_uses_subdivision() {
        let tr = Transport::default();
        assert_eq!(tr.ticks_per_minute(), 90.0 * 2.0);
    }
}
