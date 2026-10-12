//! `rng` — the App's reproducible, ECS-managed RNG, reused from `bevy_rand`.
//! No seams.
//!
//! ## Why this module exists (reuse-first policy)
//!
//! TERMINAL STATE needs randomness in a few places. The ones that are pinned by
//! GOLDEN / determinism tests — the synth noise + director arrangement
//! (`re_audio`) and the glitch value-noise (`re_glitch`) — keep their bespoke
//! `Lcg` / SplitMix hashes, because their *exact byte stream* is asserted by
//! tests (`same_seed_same_stream`, the glitch goldens). Swapping the generator
//! under them would change those bytes and break the goldens for no gain.
//!
//! For any NEW randomness wired through the Bevy `App` (cosmetic jitter, future
//! generative content, etc.) we do NOT hand-roll yet another LCG — we reuse
//! [`bevy_rand`]'s `WyRand` global entropy source, seeded deterministically so
//! the whole App stays reproducible. This is the additive, ecosystem-first
//! adoption: a new capability made AVAILABLE without touching working code.
//!
//! `bevy_rand` 0.14 targets Bevy `^0.18` and depends only on
//! `bevy_app`/`ecs`/`math`/`reflect` — all in our headless feature set. No
//! render, no winit: it runs under `MinimalPlugins`.
//!
//! Access the global RNG from any system with
//! `Single<&mut bevy_prng::WyRand, With<bevy_rand::prelude::GlobalRng>>`.

use bevy::prelude::*;
use bevy_prng::WyRand;
use bevy_rand::prelude::EntropyPlugin;

/// The fixed 8-byte seed for the App's global `WyRand`. Pinning it keeps the
/// whole App reproducible run-to-run (the same property the bespoke `Lcg`
/// paths give their own streams). The 8 seed bytes are FROZEN (the old project
/// name `REVISION` happens to be 8 bytes) — changing them would shift every
/// `WyRand` stream and break determinism, so they are kept verbatim.
pub const GLOBAL_RNG_SEED: [u8; 8] = *b"REVISION";

/// Adds a single, deterministically-seeded global `WyRand` entropy source to
/// the App. Additive: nothing else changes, and systems opt in by querying the
/// global RNG only when they need it.
pub struct GameRngPlugin;

impl Plugin for GameRngPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(EntropyPlugin::<WyRand>::with_seed(GLOBAL_RNG_SEED));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::MinimalPlugins;
    use bevy_rand::prelude::GlobalRng;
    // `next_u32` comes from the rand_core `Rng` trait that `WyRand` implements
    // (in rand_core 0.10 the old `RngCore` name is deprecated in favour of `Rng`).
    use rand_core::Rng;

    /// Drain `n` u32s from the global RNG into `out` (a resource), proving the
    /// `Single<&mut WyRand, With<GlobalRng>>` access pattern compiles + runs
    /// headless under MinimalPlugins.
    #[derive(Resource, Default)]
    struct Drawn(Vec<u32>);

    fn draw_three(mut rng: Single<&mut WyRand, With<GlobalRng>>, mut out: ResMut<Drawn>) {
        for _ in 0..3 {
            // `Single` derefs to `&mut WyRand`; call the trait method on it.
            out.0.push(rng.next_u32());
        }
    }

    fn run_once_collect() -> Vec<u32> {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .add_plugins(GameRngPlugin)
            .init_resource::<Drawn>()
            .add_systems(Update, draw_three);
        app.update();
        app.world().resource::<Drawn>().0.clone()
    }

    #[test]
    fn global_rng_is_available_headless() {
        // It runs under MinimalPlugins (no render/winit) and yields values.
        let v = run_once_collect();
        assert_eq!(v.len(), 3, "global WyRand must be queryable headless");
    }

    #[test]
    fn fixed_seed_is_reproducible() {
        // Same fixed seed → identical stream across two fresh Apps. This is the
        // determinism guarantee that lets bevy_rand stand in for a hand-rolled
        // seeded LCG when we DON'T have a pinned golden to preserve.
        let a = run_once_collect();
        let b = run_once_collect();
        assert_eq!(a, b, "fixed-seed global WyRand must be reproducible");
        // Sanity: not the degenerate all-zero stream.
        assert!(a.iter().any(|&x| x != 0), "stream should not be all zero");
    }
}
