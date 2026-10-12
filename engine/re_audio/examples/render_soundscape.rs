//! Offline render of the soundscape to WAV files — no audio device is opened.
//!
//! ```text
//! cargo run -p re_audio --example render_soundscape -- <out_dir>
//! ```
//!
//! Writes, into `<out_dir>` (default `target/soundscape`):
//!
//! * `before_<mood>.wav` — the old mono drone bed, looped (the flat reference);
//! * `ambience_<mood>_small.wav` / `_hall.wav` — the new layered, room-baked
//!   stereo ambience of the same mood in a 5×5×3 closet and a 20×20×8 hall;
//! * `progress_sweep.wav` — a level being solved: nodes light one by one, a
//!   refusal spikes the tension, the circuit completes; the harmony follows;
//! * `cue_gallery.wav` — every grammar cue in turn, 1.3 s apart;
//! * `positional_walk.wav` — walking past a lamp (left) and a powered pipe
//!   (right), then turning round: the sources cross the stereo field;
//! * `footsteps.wav` — eight steps on each surface.

use std::path::PathBuf;

use re_audio::acoustics::Room;
use re_audio::cue::{Cue, NoteVoice, Surface};
use re_audio::dsp::Stereo;
use re_audio::offline::{add_ambience, positional_loops, write_wav, OfflineMix};
use re_audio::soundscape::{
    pipe_hum_level, pipe_hum_rate, SceneAssets, SceneSpec, MAX_POSITIONAL_VOICES,
};
use re_audio::spatial::{Listener, LoopKind, PositionalSource, SourceKey};
use re_audio::{AmbienceMood, SynthBank, SAMPLE_RATE};

fn save(dir: &std::path::Path, name: &str, s: &Stereo) {
    let p = dir.join(name);
    write_wav(&p, s).expect("write wav");
    println!(
        "{:<34} {:>5.1}s  rms {:.3}  peak {:.3}  L/R corr {:+.2}",
        name,
        s.len() as f32 / SAMPLE_RATE as f32,
        s.rms(),
        s.peak(),
        s.lr_correlation()
    );
}

fn main() {
    let dir: PathBuf = std::env::args()
        .nth(1)
        .map_or_else(|| PathBuf::from("target/soundscape"), PathBuf::from);
    std::fs::create_dir_all(&dir).expect("create out dir");
    let bank = SynthBank::build();

    let small = Room::from_cells(5, 5, 3, false, 0.18);
    let hall = Room::from_cells(20, 20, 8, false, 0.12);
    let secs = 12.0;

    for mood in AmbienceMood::ALL {
        let tag = format!("{mood:?}").to_lowercase();
        // Before: the mono drone, upmixed exactly as the old device path did.
        let drone = &bank.get(mood.drone()).frames;
        let mut before = OfflineMix::new(secs);
        before.add_loop(&Stereo::from_mono(drone), |_| 0.8);
        save(
            &dir,
            &format!("before_{tag}.wav"),
            &before.finish(&Room {
                absorption: 0.95,
                ..small
            }),
        );

        for (room, rname) in [(small, "small"), (hall, "hall")] {
            let scene = SceneAssets::render(SceneSpec::new(mood, room, 0x5EED ^ mood as u32));
            let mut mix = OfflineMix::new(secs);
            add_ambience(&mut mix, &scene, 0.8, |_| 0.35, |_| 0.0);
            save(
                &dir,
                &format!("ambience_{tag}_{rname}.wav"),
                &mix.finish(&room),
            );
        }
    }

    // A level being solved in a mid-size server room.
    let room = Room::from_cells(12, 10, 4, false, 0.15);
    let scene = SceneAssets::render(SceneSpec::new(AmbienceMood::ServerHum, room, 0xA11));
    let total = 5u32;
    let lit_at = [3.0f32, 6.0, 9.5, 13.0, 16.0];
    let refusal_at = 11.0f32;
    let progress = move |t: f32| {
        let lit = lit_at.iter().filter(|&&a| t >= a).count() as u32;
        re_audio::soundscape::harmony_progress(lit, total, t >= 17.0)
    };
    let tension = move |t: f32| {
        let mut ten = re_audio::soundscape::Tension::default();
        if t >= refusal_at {
            ten.pulse(Cue::RefusedNoPower.tension());
            ten.decay(t - refusal_at);
        }
        ten.level()
    };
    let mut mix = OfflineMix::new(24.0);
    add_ambience(&mut mix, &scene, 0.8, progress, tension);
    let ear = Listener {
        pos: [6.0, 5.0, 1.5],
        yaw: 0.0,
    };
    for (k, &t) in lit_at.iter().enumerate() {
        let lit = k as u32 + 1;
        let pos = [3.0 + 1.5 * k as f32, 8.0, 1.5];
        mix.add_cue(
            &bank,
            t - 0.4,
            Cue::SwitchOn {
                used: lit.min(3),
                budget: 3,
            },
            Some((pos, ear)),
        );
        mix.add_cue(&bank, t, Cue::NodeLit { lit, total }, Some((pos, ear)));
    }
    mix.add_cue(
        &bank,
        refusal_at,
        Cue::RefusedNoPower,
        Some(([9.0, 6.0, 1.5], ear)),
    );
    mix.add_cue(&bank, 17.0, Cue::CircuitComplete, None);
    save(&dir, "progress_sweep.wav", &mix.finish(&room));

    // Every grammar cue, centred.
    let gallery: Vec<Cue> = vec![
        Cue::AimSwitch,
        Cue::SwitchOn { used: 1, budget: 3 },
        Cue::SwitchOn { used: 3, budget: 3 },
        Cue::SwitchOff,
        Cue::InteractEmpty,
        Cue::RefusedNoPower,
        Cue::RefusedCycle,
        Cue::NodeLit { lit: 1, total: 4 },
        Cue::NodeLit { lit: 4, total: 4 },
        Cue::NodeUnlit { lit: 3, total: 4 },
        Cue::CircuitComplete,
        Cue::CircuitBroken,
        Cue::NoteRead(NoteVoice::SystemLog),
        Cue::NoteRead(NoteVoice::Researcher),
        Cue::NoteRead(NoteVoice::ModelThought),
        Cue::Portal,
        Cue::Storey { up: true },
        Cue::Storey { up: false },
        Cue::FirstLight { shards: 1 },
        Cue::FirstLight { shards: 7 },
        Cue::MenuMove,
        Cue::MenuConfirm,
        Cue::MenuBack,
        Cue::Win,
    ];
    let mut mix = OfflineMix::new(gallery.len() as f32 * 1.3 + 2.0);
    for (i, c) in gallery.iter().enumerate() {
        mix.add_cue(&bank, 0.3 + i as f32 * 1.3, *c, None);
        println!("  gallery {:>5.1}s  {c:?}", 0.3 + i as f32 * 1.3);
    }
    save(&dir, "cue_gallery.wav", &mix.finish(&Room::default()));

    // Walk along +y past a lamp on the left and a powered pipe on the right,
    // then turn round (yaw 0 → π over 12..16 s) and hear them swap sides.
    let loops = positional_loops();
    for (k, l) in LoopKind::ALL.iter().zip(loops.iter()) {
        let rms = (l.iter().map(|v| v * v).sum::<f32>() / l.len() as f32).sqrt();
        println!("  loop {k:?}: rms {rms:.3}");
    }
    let room = Room::from_cells(8, 20, 4, false, 0.15);
    let mut mix = OfflineMix::new(16.0);
    mix.add_positional(&loops, MAX_POSITIONAL_VOICES, |t| {
        let y = 1.0 + t.min(12.0) * 1.2;
        let yaw = if t > 12.0 {
            std::f32::consts::PI * ((t - 12.0) / 4.0).min(1.0)
        } else {
            0.0
        };
        let sources = vec![
            PositionalSource {
                key: SourceKey {
                    kind: LoopKind::LampHum,
                    id: 0,
                },
                pos: [1.5, 8.0, 3.0],
                level: 0.9,
                rate: 1.0,
                occlusion: 0.0,
            },
            PositionalSource {
                key: SourceKey {
                    kind: LoopKind::PipeHum,
                    id: 0,
                },
                pos: [6.5, 12.0, 1.0],
                level: pipe_hum_level(0.7),
                rate: pipe_hum_rate(0.7),
                occlusion: 0.0,
            },
            PositionalSource {
                key: SourceKey {
                    kind: LoopKind::NodeGlow,
                    id: 0,
                },
                pos: [4.0, 18.0, 1.5],
                level: 0.8,
                rate: re_audio::soundscape::node_glow_rate(2),
                occlusion: 0.0,
            },
        ];
        (
            Listener {
                pos: [4.0, y, 1.5],
                yaw,
            },
            sources,
        )
    });
    save(&dir, "positional_walk.wav", &mix.finish(&room));

    // Footsteps on each surface.
    let surfaces = [Surface::Hard, Surface::Metal, Surface::Wet, Surface::Soft];
    let mut mix = OfflineMix::new(surfaces.len() as f32 * 4.0);
    for (s, &surface) in surfaces.iter().enumerate() {
        for k in 0..8u32 {
            let t = s as f32 * 4.0 + 0.2 + k as f32 * 0.42;
            mix.add_cue(&bank, t, Cue::Footstep { surface, stride: k }, None);
        }
    }
    save(&dir, "footsteps.wav", &mix.finish(&Room::default()));
    println!("wrote {}", dir.display());
}
