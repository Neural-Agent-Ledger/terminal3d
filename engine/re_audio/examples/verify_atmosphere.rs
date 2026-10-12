//! Silent, offline verification for the three atmosphere goals (reverb in
//! the right places, evolving drone music, convincing 3D spatial audio).
//! Opens no audio device; writes WAV files + prints numeric analysis.
//!
//! ```text
//! cargo run -p re_audio --example verify_atmosphere -- <out_dir>
//! ```
//!
//! Writes, into `<out_dir>`:
//!
//! * `drone_<mood>.wav` — ~66 s of the evolving drone-music layer for each
//!   [`AmbienceMood`], three chord segments back to back with a real
//!   `CHORD_CROSSFADE_S`-style crossfade between them (forced, rather than
//!   waited-for — `MusicSequencer`'s real chord holds are minutes long, too
//!   slow to demonstrate three changes in one short file).
//! * `walk_corridor_hall_outdoors_closet.wav` — one continuous drone source
//!   heard through four different rooms in turn (Goal 1: reverb changes with
//!   *position*, not just level).
//! * `circle_behind_wall.wav` — a source orbiting the listener once,
//!   occluded (a scripted wall) for one quarter of the orbit (Goal 3: pan,
//!   air absorption, ITD, occlusion).
//!
//! No spectrogram PNGs: this machine has no Python / plotting library
//! (`python3` is not on PATH here — checked before writing this). The
//! numeric analysis below (RT60, RMS/peak/clipping, zero-crossing-rate
//! brightness proxy, L/R level + lag) is the offline evidence instead; see
//! `lane_audio/README.md` for how to read it.

use std::path::PathBuf;

use re_audio::acoustics::{Room, RoomVerb, SPEED_OF_SOUND};
use re_audio::drone_music::{chord_degrees, render_music_segment_baked};
use re_audio::dsp::Stereo;
use re_audio::local_room::{estimate_room, RoomProbe};
use re_audio::offline::{positional_loops, write_wav, OfflineMix};
use re_audio::soundscape::MAX_POSITIONAL_VOICES;
use re_audio::spatial::{Listener, LoopKind, PositionalSource, SourceKey};
use re_audio::{AmbienceMood, SAMPLE_RATE};

/// Equal-power crossfade two buffers end-to-end: the last `xfade` frames of
/// `a` overlap the first `xfade` frames of `b`.
fn concat_crossfade(a: &Stereo, b: &Stereo, xfade: usize) -> Stereo {
    let xfade = xfade.min(a.len()).min(b.len());
    let n = a.len() + b.len() - xfade;
    let mut out = Stereo::silent(n);
    out.l[..a.len()].copy_from_slice(&a.l);
    out.r[..a.len()].copy_from_slice(&a.r);
    let start = a.len() - xfade;
    for i in 0..xfade {
        let t = (i as f32 + 1.0) / (xfade as f32 + 1.0);
        let (gout, gin) = ((1.0 - t).sqrt(), t.sqrt());
        out.l[start + i] = out.l[start + i] * gout + b.l[i] * gin;
        out.r[start + i] = out.r[start + i] * gout + b.r[i] * gin;
    }
    out.l[a.len()..].copy_from_slice(&b.l[xfade..]);
    out.r[a.len()..].copy_from_slice(&b.r[xfade..]);
    out
}

/// Zero-crossing rate (Hz), a cheap brightness proxy: NOT a true spectral
/// centroid (that needs an FFT this offline tool does not have access to
/// without a plotting/DSP dependency), but monotonic with "how much top end
/// is left" for the kind of broadband, filtered content this layer renders —
/// exactly what the air-absorption / occlusion cutoffs are meant to move.
fn zero_crossing_hz(mono: impl Iterator<Item = f32>) -> f32 {
    let mut prev = 0.0f32;
    let mut crossings = 0u32;
    let mut n = 0u32;
    for x in mono {
        if n > 0 && prev.signum() != x.signum() && x != 0.0 {
            crossings += 1;
        }
        prev = x;
        n += 1;
    }
    if n < 2 {
        return 0.0;
    }
    crossings as f32 * SAMPLE_RATE as f32 / (2.0 * n as f32)
}

/// Loop-seam discontinuity check for a buffer meant to be looped
/// (`buf[len-1]` immediately followed by `buf[0]` again): the value jump AT
/// the wrap boundary, and the jump in local slope across it (a value-only
/// check can miss a click that is inaudible in level but audible as a
/// derivative kink). Both should be small relative to the signal's own peak
/// — NOT a comparison of the head and tail regions' general levels (those
/// are two different, unrelated points of an evolving waveform and will
/// legitimately differ by a lot even in a perfectly seamless loop).
fn seam_jump(s: &Stereo) -> (f32, f32) {
    let n = s.len();
    let value_l = (s.l[n - 1] - s.l[0]).abs();
    let value_r = (s.r[n - 1] - s.r[0]).abs();
    let slope_l = ((s.l[n - 1] - s.l[n - 2]) - (s.l[1] - s.l[0])).abs();
    let slope_r = ((s.r[n - 1] - s.r[n - 2]) - (s.r[1] - s.r[0])).abs();
    (value_l.max(value_r), slope_l.max(slope_r))
}

/// Median absolute adjacent-sample delta over the WHOLE buffer — the
/// "background" sample-to-sample jitter a bright/noisy signal has EVERYWHERE
/// (not just at the loop point). [`seam_jump`] is only meaningful relative to
/// this: a seam jump comparable to the background jitter is inaudible (it is
/// just what any two adjacent samples of this signal look like); only a seam
/// jump that stands out ABOVE the background is a real click.
fn background_jitter(s: &Stereo) -> f32 {
    let mut deltas: Vec<f32> = s.l.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
    deltas.sort_by(f32::total_cmp);
    deltas[deltas.len() / 2]
}

/// Estimate the inter-channel lag (samples, L relative to R — positive = L
/// lags R, i.e. the source is toward the right) by cross-correlating a short
/// excerpt over a small window. Returns `(lag_samples, level_db_l_minus_r)`.
fn lr_lag_and_level(s: &Stereo, start: usize, len: usize, max_lag: i32) -> (i32, f32) {
    let end = (start + len).min(s.len());
    let l = &s.l[start..end];
    let r = &s.r[start..end];
    let mut best_lag = 0;
    let mut best_corr = f32::NEG_INFINITY;
    for lag in -max_lag..=max_lag {
        let mut sum = 0.0f32;
        let mut count = 0u32;
        for (i, &li) in l.iter().enumerate() {
            let j = i as i32 + lag;
            if j >= 0 && (j as usize) < r.len() {
                sum += li * r[j as usize];
                count += 1;
            }
        }
        if count > 0 {
            let corr = sum / count as f32;
            if corr > best_corr {
                best_corr = corr;
                best_lag = lag;
            }
        }
    }
    let rms = |x: &[f32]| (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt();
    let (rl, rr) = (rms(l).max(1e-9), rms(r).max(1e-9));
    (best_lag, 20.0 * (rl / rr).log10())
}

fn report(name: &str, s: &Stereo) {
    println!(
        "  {name:<42} {:>6.1}s  rms {:.3}  peak {:.3}  clip {}",
        s.len() as f32 / SAMPLE_RATE as f32,
        s.rms(),
        s.peak(),
        if s.peak() >= 0.999 {
            "YES <-- check"
        } else {
            "no"
        },
    );
}

fn save(dir: &std::path::Path, name: &str, s: &Stereo) {
    write_wav(&dir.join(name), s).expect("write wav");
    report(name, s);
}

fn main() {
    let dir: PathBuf = std::env::args()
        .nth(1)
        .map_or_else(|| PathBuf::from("target/atmosphere"), PathBuf::from);
    std::fs::create_dir_all(&dir).expect("create out dir");

    println!("=== Goal 2: evolving drone music, ~66s / mood, 3 forced chord changes ===");
    let hall = Room::from_cells(20, 20, 8, false, 0.12);
    for mood in AmbienceMood::ALL {
        let tag = format!("{mood:?}").to_lowercase();
        let degrees = chord_degrees(mood);
        let segs: Vec<Stereo> = (0..3)
            .map(|i| {
                let deg = degrees[i % degrees.len()];
                render_music_segment_baked(
                    mood,
                    deg,
                    &hall,
                    0x5EED ^ (mood as u32) ^ (i as u32 * 0x9E37_79B9),
                )
            })
            .collect();
        for (i, seg) in segs.iter().enumerate() {
            let (value_jump, _slope_jump) = seam_jump(seg);
            let jitter = background_jitter(seg);
            // A seam jump within a few× the buffer's own typical adjacent-
            // sample jitter is inaudible — it is what any two neighbouring
            // samples of this signal look like, seam or not. Only a jump
            // that stands out well above that background is a real click.
            let ratio = value_jump / jitter.max(1e-6);
            println!(
                "    {tag} chord[{i}] loop-seam value-jump {value_jump:.5} vs background jitter {jitter:.5} (×{ratio:.1}) — {}",
                if ratio < 8.0 { "seamless" } else { "AUDIBLE — check" }
            );
        }
        let xfade = SAMPLE_RATE as usize * 8; // matches drone_music::CHORD_CROSSFADE_S
        let mut out = segs[0].clone();
        for seg in &segs[1..] {
            out = concat_crossfade(&out, seg, xfade);
        }
        // This is a listening reference, not the live mix level
        // (`drone_music::MUSIC_LEVEL` attenuates it further before it is
        // summed with everything else on the device) — normalize (not a
        // flat multiply) so three different chords' differing natural peaks
        // never clip the file.
        out.normalize_to(0.9);
        save(&dir, &format!("drone_{tag}.wav"), &out);
    }

    println!("=== Goal 1: reverb changes with position (one drone heard through 4 rooms) ===");
    // A dry, un-roomed hum: the mono lamp loop, upmixed, long enough to hear
    // each room's tail clearly and to let RT60 register before the next cut.
    let loops = positional_loops();
    let dry_seg_secs = 6.0f32;
    let dry_len = (SAMPLE_RATE as f32 * dry_seg_secs) as usize;
    let dry_mono = &loops[LoopKind::LampHum.index()];
    let dry = Stereo {
        l: (0..dry_len).map(|i| dry_mono[i % dry_mono.len()]).collect(),
        r: (0..dry_len).map(|i| dry_mono[i % dry_mono.len()]).collect(),
    };
    let corridor = estimate_room(&RoomProbe {
        east: 1.2,
        west: 1.2,
        north: 30.0,
        south: 30.0,
        up: 1.5,
        down: 1.0,
        open_sky: false,
        absorption: 0.14,
    });
    let big_hall = estimate_room(&RoomProbe {
        east: 10.0,
        west: 10.0,
        north: 12.0,
        south: 12.0,
        up: 6.0,
        down: 1.0,
        open_sky: false,
        absorption: 0.18,
    });
    let outdoors = estimate_room(&RoomProbe {
        east: 25.0,
        west: 25.0,
        north: 25.0,
        south: 25.0,
        up: 40.0,
        down: 1.0,
        open_sky: true,
        absorption: 0.3,
    });
    let closet = estimate_room(&RoomProbe {
        east: 0.8,
        west: 0.8,
        north: 0.8,
        south: 0.8,
        up: 1.0,
        down: 1.0,
        open_sky: false,
        absorption: 0.2,
    });
    let rooms = [
        ("corridor", corridor),
        ("hall", big_hall),
        ("outdoors", outdoors),
        ("closet", closet),
    ];
    let mut walk = None::<Stereo>;
    for (name, room) in rooms {
        println!(
            "    {name:<10} size {:?} m  open_sky {}  RT60 {:.2}s",
            room.size_m.map(|v| (v * 10.0).round() / 10.0),
            room.open_sky,
            room.rt60(),
        );
        let wet = RoomVerb::new(&room).process(&dry, room.rt60().min(2.5));
        walk = Some(match walk {
            None => wet,
            Some(prev) => concat_crossfade(&prev, &wet, SAMPLE_RATE as usize / 4),
        });
    }
    let walk = walk.expect("4 rooms produced audio");
    save(&dir, "walk_corridor_hall_outdoors_closet.wav", &walk);

    println!("=== Goal 3: a source circling the listener, occluded behind a wall for one quarter-turn ===");
    let room = Room::from_cells(16, 16, 4, false, 0.15);
    let orbit_secs = 12.0f32;
    let radius = 3.0f32;
    let mut mix = OfflineMix::new(orbit_secs);
    mix.add_positional(&loops, MAX_POSITIONAL_VOICES, |t| {
        let frac = (t / orbit_secs).fract();
        let angle = frac * std::f32::consts::TAU;
        let pos = [radius * angle.sin(), radius * angle.cos(), 1.5];
        // "Behind a wall" for the quarter-turn centred on due-west (frac in [0.625, 0.875)).
        let occlusion = if (0.625..0.875).contains(&frac) {
            0.9
        } else {
            0.0
        };
        let sources = vec![PositionalSource {
            key: SourceKey {
                kind: LoopKind::EmitterHiss,
                id: 0,
            },
            pos,
            level: 1.0,
            rate: 1.0,
            occlusion,
        }];
        (
            Listener {
                pos: [0.0, 0.0, 1.5],
                yaw: 0.0,
            },
            sources,
        )
    });
    let circling = mix.finish(&room);
    save(&dir, "circle_behind_wall.wav", &circling);

    // Right-of-listener excerpt (frac ~0.25 -> angle pi/2 -> pos = (radius, 0) = due east
    // of a listener facing +y, which is to the RIGHT by this crate's pan convention).
    let right_start = (orbit_secs * 0.22 * SAMPLE_RATE as f32) as usize;
    let (lag, level_db) = lr_lag_and_level(&circling, right_start, SAMPLE_RATE as usize / 2, 40);
    println!(
        "    at ~right (frac 0.22-0.27): L-vs-R lag {lag:+} samples ({:+.3} ms), L-R level {level_db:+.2} dB",
        lag as f32 * 1000.0 / SAMPLE_RATE as f32,
    );
    let left_start = (orbit_secs * 0.72 * SAMPLE_RATE as f32) as usize;
    let (lag2, level_db2) = lr_lag_and_level(&circling, left_start, SAMPLE_RATE as usize / 2, 40);
    println!(
        "    at ~left  (frac 0.72-0.77): L-vs-R lag {lag2:+} samples ({:+.3} ms), L-R level {level_db2:+.2} dB",
        lag2 as f32 * 1000.0 / SAMPLE_RATE as f32,
    );

    let ahead_start = 0usize;
    let ahead_hz = zero_crossing_hz(
        circling.l[ahead_start..ahead_start + SAMPLE_RATE as usize / 2]
            .iter()
            .copied(),
    );
    let occluded_start = (orbit_secs * 0.75 * SAMPLE_RATE as f32) as usize;
    let occluded_hz = zero_crossing_hz(
        circling.l[occluded_start..(occluded_start + SAMPLE_RATE as usize / 2).min(circling.len())]
            .iter()
            .copied(),
    );
    println!(
        "    brightness proxy (zero-crossing Hz): clear-ish {ahead_hz:.0} Hz vs occluded {occluded_hz:.0} Hz ({})",
        if occluded_hz < ahead_hz { "darker behind the wall, as expected" } else { "UNEXPECTED — check" },
    );

    println!("\nAir absorption / distance reference (spatial::air_absorption_cutoff_hz already unit-tested in-crate):");
    println!("  speed of sound used by Room::early_reflections: {SPEED_OF_SOUND} m/s");

    println!("\nwrote {}", dir.display());
}
