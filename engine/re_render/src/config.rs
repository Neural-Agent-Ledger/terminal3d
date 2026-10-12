//! Render configuration — the three formerly-process-global rendering knobs,
//! promoted to a proper Bevy [`Resource`] so they are TESTABLE without environment
//! variables (the old `LazyLock`/`env::var` reads were process-global and could not
//! be exercised from a unit test).
//!
//! The three knobs:
//! * [`RenderConfig::view_independent_light`] — drop the view-direction-dependent
//!   lighting terms (Blinn-Phong specular + headlamp `N·V`). DEFAULT `true`: it
//!   improves readability AND makes the near-ring rotation pan byte-exact (a baked
//!   shade is invariant to facing). This is a deliberate departure from the strict
//!   Blinn-Phong intent.
//! * [`RenderConfig::texel_snap`] — `Some(texels_per_cell)` snaps the world sample
//!   coord to a texel grid (enables the cross-frame world-tex cache to actually hit).
//!   DEFAULT `None` (OFF): snapping coarsens far-field depth, so the continuous look
//!   ships by default.
//! * [`RenderConfig::tex_cache`] — enable the cross-frame world-tex cache. DEFAULT
//!   `false`; treated as ON whenever `texel_snap.is_some()` (snapping is what makes
//!   the cache hit) — see [`RenderConfig::tex_cache_effective`].
//!
//! ## How the values reach the hot shade path
//!
//! The shade path is a deep trait dispatch (`Material::sample` → recipe → relief →
//! `light_surface`), so the two shading-time knobs cannot be threaded as a plain
//! function argument without a multi-file API churn. Instead the cast entry point
//! APPLIES the resource's values once per cast via [`apply`] (an explicit write of a
//! thread-local cell), and the shader reads them back via
//! [`view_independent_lighting`] / [`texel_snap_texels_per_cell`]. Crucially this is
//! NO LONGER an env read: the value is set explicitly, so a test sets it explicitly
//! too (no process-global env, no `LazyLock`). The env is read EXACTLY ONCE, at
//! [`CompositeRenderPlugin::build`](crate::CompositeRenderPlugin), to seed the
//! resource for dev convenience.
//!
//! The `tex_cache` knob is threaded explicitly (it is read at the cast boundary in
//! `lib.rs`, not inside the deep dispatch), so it needs no thread-local.

use std::cell::Cell as StdCell;

use bevy::prelude::Resource;

/// The three rendering knobs, as a Bevy [`Resource`]. Constructed directly in tests
/// (no env); seeded once from the `TS_*` env vars at plugin build for dev use.
#[derive(Resource, Debug, Clone, Copy, PartialEq)]
pub struct RenderConfig {
    /// Enable the cross-frame world-tex cache. The cache is exact (a hit replays the
    /// byte-identical texture half; lighting always re-runs fresh) but delivers ~0
    /// reuse unless [`texel_snap`](Self::texel_snap) is on — so the EFFECTIVE switch
    /// is [`tex_cache_effective`](Self::tex_cache_effective): `tex_cache || texel_snap.is_some()`.
    pub tex_cache: bool,
    /// `Some(texels_per_cell)` snaps the world sample coord (and wall-v) to a texel
    /// grid before the recipe/relief runs → a stable cross-frame cache key. `None`
    /// (default) ⇒ the continuous, byte-identical-to-pre-toggle look.
    pub texel_snap: Option<f32>,
    /// Drop the view-direction-dependent lighting terms (specular + headlamp facing)
    /// so a cell's shade is invariant to camera facing. Default `true` (the
    /// exact-pan + readability look).
    pub view_independent_light: bool,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            // OFF by default — treated ON whenever texel_snap is set (see
            // `tex_cache_effective`).
            tex_cache: false,
            // OFF by default: snapping coarsens far-field depth.
            texel_snap: None,
            // ON by default: better readability + byte-exact near pan.
            view_independent_light: true,
        }
    }
}

impl RenderConfig {
    /// The look that reproduces the PRE-TOGGLE Blinn-Phong renderer byte-for-byte:
    /// view-dependent lighting ON (i.e. `view_independent_light = false`), no texel
    /// snap, no tex cache. Used by tests asserting the legacy baseline and as the
    /// default the no-terminal cast entry points fall back to.
    pub const LEGACY: Self = Self {
        tex_cache: false,
        texel_snap: None,
        view_independent_light: false,
    };

    /// Whether the cross-frame world-tex cache should actually engage: explicitly
    /// requested OR auto-enabled by texel-snap (snapping is what turns the cache from
    /// a ~0.7%-hit no-op into a real win — the snapped coord gives a colliding key).
    #[inline]
    pub fn tex_cache_effective(&self) -> bool {
        self.tex_cache || self.texel_snap.is_some()
    }

    /// Seed a config from the `TS_*` env vars (the renamed REVISION→TERMINAL STATE
    /// prefix). Called ONCE at plugin build for dev convenience; unset vars keep the
    /// field's [`Default`]. This is the ONLY place the process env is consulted.
    ///
    /// * `TS_VIEWINDEP_LIGHT` — `1`/`true` ⇒ on, `0`/`false` ⇒ off (overrides the
    ///   `true` default either way); unset ⇒ default.
    /// * `TS_TEXEL_SNAP=<texels_per_cell>` — a finite `> 0` value ⇒ `Some(v)`;
    ///   unset/unparseable/`≤0` ⇒ default (`None`).
    /// * `TS_TEXCACHE` — present (any value) ⇒ `tex_cache = true`; unset ⇒ default.
    pub fn from_env() -> Self {
        let mut cfg = Self::default();
        if let Ok(raw) = std::env::var("TS_VIEWINDEP_LIGHT") {
            let t = raw.trim();
            if t == "1" || t.eq_ignore_ascii_case("true") {
                cfg.view_independent_light = true;
            } else if t == "0" || t.eq_ignore_ascii_case("false") {
                cfg.view_independent_light = false;
            }
        }
        if let Ok(raw) = std::env::var("TS_TEXEL_SNAP") {
            if let Ok(v) = raw.trim().parse::<f32>() {
                if v.is_finite() && v > 0.0 {
                    cfg.texel_snap = Some(v);
                }
            }
        }
        if std::env::var_os("TS_TEXCACHE").is_some() {
            cfg.tex_cache = true;
        }
        cfg
    }

    /// Install this config's shading-time knobs as the ACTIVE values the deep shade
    /// dispatch reads back (via [`view_independent_lighting`] /
    /// [`texel_snap_texels_per_cell`]). An explicit write — no env. The cast applies
    /// the live [`RenderConfig`] resource at its entry; tests apply a constructed one.
    #[inline]
    pub fn apply(&self) {
        apply(self.view_independent_light, self.texel_snap, self.tex_cache);
    }
}

// ───────────── active shading-time config (explicit, never env) ──────────────
//
// The two shading-time knobs are read deep inside the `Material`/recipe trait
// dispatch where a plain argument cannot reach without a large API churn. They are
// held in a thread-local cell that is ONLY ever written explicitly — by [`apply`],
// from a `RenderConfig`. This is NOT the old `LazyLock`/env global: there is no
// `env::var` in this path, so a unit test sets the values directly and asserts the
// resulting shade with zero environment dependence.

thread_local! {
    /// `view_independent_light` for the current cast. Defaults to the [`RenderConfig`]
    /// default (`true`) so an un-applied path still uses the shipping look.
    static VIEWINDEP_LIGHT: StdCell<bool> = const { StdCell::new(true) };
    /// `texel_snap` for the current cast. Defaults to `None` (off).
    static TEXEL_SNAP: StdCell<Option<f32>> = const { StdCell::new(None) };
    /// Raw `tex_cache` flag for the current cast. Defaults to `false`.
    static TEX_CACHE: StdCell<bool> = const { StdCell::new(false) };
}

/// Explicitly set the active shading-time knobs for the current thread. Called by the
/// cast from the live [`RenderConfig`], and by tests directly. No env is consulted.
#[inline]
pub fn apply(view_independent_light: bool, texel_snap: Option<f32>, tex_cache: bool) {
    VIEWINDEP_LIGHT.with(|c| c.set(view_independent_light));
    TEXEL_SNAP.with(|c| c.set(texel_snap));
    TEX_CACHE.with(|c| c.set(tex_cache));
}

/// Whether the cross-frame world-tex cache should engage for the current cast:
/// the raw flag OR texel-snap (snapping is what makes it hit). Read at the cast
/// boundary in `lib.rs` instead of the old `REVISION_TEXCACHE` env probe.
#[inline]
pub fn tex_cache_effective() -> bool {
    TEX_CACHE.with(|c| c.get()) || texel_snap_texels_per_cell().is_some()
}

/// Whether view-independent lighting is ON for the current cast (specular + headlamp
/// facing dropped). Read by [`light_surface`](crate::ascii_shader) and the family-glyph
/// lean. Set explicitly via [`apply`]; default `true`.
#[inline]
pub fn view_independent_lighting() -> bool {
    VIEWINDEP_LIGHT.with(|c| c.get())
}

/// Texels-per-cell for sample-coord snapping when ON, else `None` (OFF, the default
/// continuous look). Set explicitly via [`apply`]; default `None`.
#[inline]
pub fn texel_snap_texels_per_cell() -> Option<f32> {
    TEXEL_SNAP.with(|c| c.get())
}

// ───────────── NODE SPRITE on/off (explicit, never env in the hot path) ──────────
//
// `is_sprite_node` (`crate::lib`) is called from the DDA / off-slice / shadow / holo
// passes — hot per-cast paths — so, exactly like `VIEWINDEP_LIGHT`/`TEXEL_SNAP`
// above, the knob is a thread-local set EXPLICITLY once per cast, never an
// `env::var` read from inside the loop.

thread_local! {
    /// NODE SPRITE post-pass on/off for the current cast. Default ON (`true`) so an
    /// un-seeded path (a test that never calls [`apply_node_sprite_env`]) still uses
    /// the shipping look.
    static NODE_SPRITE: StdCell<bool> = const { StdCell::new(true) };
}

/// Explicitly set the NODE SPRITE knob for the current thread's cast. No env in the
/// hot path.
#[inline]
pub fn set_node_sprite_enabled(v: bool) {
    NODE_SPRITE.with(|c| c.set(v));
}

/// Is the NODE SPRITE post-pass ON for the current cast? Read by
/// [`crate::is_sprite_node`] (DDA/off-slice/shadow/holo) and
/// [`crate::node_sprite::apply`].
#[inline]
pub fn node_sprite_enabled() -> bool {
    NODE_SPRITE.with(|c| c.get())
}

/// Seed [`node_sprite_enabled`] from `TS_NODE_SPRITE` — `0` turns the pass OFF
/// (restores the pre-sprite wall-voxel node look byte-for-byte: `is_sprite_node`
/// always reports `false`, so a node is `is_solid` + paints again exactly like
/// today, and `node_sprite::apply` becomes a hard no-op); anything else, including
/// unset, leaves it ON (the default). Called once per cast entry point
/// (`raycast_fov`, `composite_dump_once`, `composite_render_sys`), mirroring
/// `RenderConfig::from_env()` / `DepthContrastConfig::from_env()` — the ONLY place
/// this env var is read.
pub fn apply_node_sprite_env() {
    set_node_sprite_enabled(parse_node_sprite_env(
        std::env::var("TS_NODE_SPRITE").ok().as_deref(),
    ));
}

/// Pure parse of the `TS_NODE_SPRITE` value (`None` = unset): `"0"` (after
/// trimming) ⇒ off, everything else ⇒ on. Split out from [`apply_node_sprite_env`]
/// so the parsing rule is unit-testable without mutating the real process env.
#[inline]
fn parse_node_sprite_env(raw: Option<&str>) -> bool {
    match raw {
        Some(v) => v.trim() != "0",
        None => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_the_visual_critic_choice() {
        let c = RenderConfig::default();
        assert!(
            c.view_independent_light,
            "view-indep lighting is the DEFAULT"
        );
        assert_eq!(c.texel_snap, None, "texel snap is OFF by default");
        assert!(!c.tex_cache, "raw tex_cache flag is OFF by default");
        assert!(
            !c.tex_cache_effective(),
            "with no snap + flag off, the cache stays off"
        );
    }

    #[test]
    fn legacy_reproduces_pre_toggle_look() {
        let c = RenderConfig::LEGACY;
        assert!(
            !c.view_independent_light,
            "LEGACY keeps view-DEPENDENT Blinn-Phong"
        );
        assert_eq!(c.texel_snap, None);
        assert!(!c.tex_cache_effective());
    }

    #[test]
    fn texel_snap_auto_enables_the_tex_cache() {
        // The whole point of the texel-snap knob: it makes the cross-frame tex cache
        // HIT, so `tex_cache_effective` must be true even with the raw flag off.
        let c = RenderConfig {
            tex_cache: false,
            texel_snap: Some(8.0),
            view_independent_light: true,
        };
        assert!(
            c.tex_cache_effective(),
            "texel_snap = Some(..) must auto-enable the tex cache"
        );
    }

    #[test]
    fn apply_threads_the_active_knobs_without_env() {
        // CONSTRUCT a config and apply it — no env var touched. The shader-facing
        // accessors must then reflect exactly what we set.
        RenderConfig {
            tex_cache: false,
            texel_snap: Some(4.0),
            view_independent_light: false,
        }
        .apply();
        assert!(!view_independent_lighting());
        assert_eq!(texel_snap_texels_per_cell(), Some(4.0));

        RenderConfig {
            tex_cache: false,
            texel_snap: None,
            view_independent_light: true,
        }
        .apply();
        assert!(view_independent_lighting());
        assert_eq!(texel_snap_texels_per_cell(), None);

        // Restore the default so sibling tests on this thread see the shipping look.
        RenderConfig::default().apply();
    }

    #[test]
    fn applied_snap_makes_the_hot_path_enable_the_cache() {
        // The cast reads `config::tex_cache_effective()` (was `REVISION_TEXCACHE`) to
        // decide whether to thread the cross-frame world-tex cache. Constructing a
        // `RenderConfig` with texel_snap and applying it must flip that decision ON —
        // proving the formerly-env-gated cache enable is now testable from a config.
        RenderConfig {
            tex_cache: false,
            texel_snap: None,
            view_independent_light: true,
        }
        .apply();
        assert!(
            !tex_cache_effective(),
            "no snap + flag off ⇒ the hot path leaves the cache OFF"
        );

        RenderConfig {
            tex_cache: false,
            texel_snap: Some(8.0),
            view_independent_light: true,
        }
        .apply();
        assert!(
            tex_cache_effective(),
            "texel_snap = Some(8.0) ⇒ the hot path enables the cache (it now hits)"
        );

        // The raw flag alone also enables it, independent of snap.
        RenderConfig {
            tex_cache: true,
            texel_snap: None,
            view_independent_light: true,
        }
        .apply();
        assert!(
            tex_cache_effective(),
            "tex_cache flag alone enables the cache"
        );

        RenderConfig::default().apply();
    }

    #[test]
    fn node_sprite_defaults_on_and_is_settable_without_env() {
        set_node_sprite_enabled(true);
        assert!(node_sprite_enabled());
        set_node_sprite_enabled(false);
        assert!(!node_sprite_enabled());
        // Restore so sibling tests on this thread see the shipping default.
        set_node_sprite_enabled(true);
    }

    #[test]
    fn node_sprite_env_parse_rule() {
        assert!(parse_node_sprite_env(None), "unset ⇒ on (the default)");
        assert!(!parse_node_sprite_env(Some("0")), "\"0\" ⇒ off");
        assert!(
            !parse_node_sprite_env(Some(" 0 ")),
            "whitespace-padded \"0\" still trims to off"
        );
        assert!(parse_node_sprite_env(Some("1")), "any non-\"0\" value ⇒ on");
        assert!(
            parse_node_sprite_env(Some("false")),
            "\"false\" is not \"0\" ⇒ on (by design)"
        );
    }
}
