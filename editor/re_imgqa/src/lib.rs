// Test-module assert_eq! on f32 values are intentional exact-equal checks.
// Allow float_cmp only in test compilation units; lib code remains denied.
// The one non-test exception (Spearman tie-detection in metrics.rs) carries
// its own local #[allow] with a justification comment at the call site.
#![cfg_attr(test, allow(clippy::float_cmp))]
#![warn(missing_docs)]
//! `re_imgqa` — a headless render-quality linter for the terminal3d ASCII raycaster.
//!
//! It loads each `.lvl` level, generates an orbit of camera poses, raycasts each
//! into a [`re_core::Framebuffer`], computes readability / contrast metrics,
//! writes truecolor HTML "photos" (selectively), and emits a ranked HTML report
//! plus machine-readable JSON. It is **diagnostic only** — it prints and ranks
//! numbers; a human decides. No pass/fail gate.
//!
//! ## Layer purity
//!
//! This crate never touches the terminal/present layer — it consumes the very
//! same [`re_core::Framebuffer`] contract Layer B consumes. Its dependencies are
//! `re_core`, `re_world`, `re_render`, `re_level`, `re_puzzle`, and `glam`. All
//! colour (Oklab), HTML, and JSON math is hand-rolled (zero external crates).
//!
//! ## Core model — reading a cell numerically
//!
//! A cell is an **ink mask** over a background, so
//! `effective_color = lerp(bg_linear, fg_linear, ink(glyph))` (linear sRGB),
//! converted to Oklab. All colour contrast is Oklab ΔE on the effective colour;
//! shape contrast is `|ink diff|`. See [`color`], [`ink`], and [`contrast`].
//!
//! The single-frame entry point is [`analyze_frame`].

pub mod color;
pub mod contrast;
pub mod html;
pub mod ink;
pub mod metrics;
pub mod orbit;
pub mod run;
pub mod static_map;
pub mod text_bind;
pub mod timing;

pub use metrics::{analyze_frame, FrameMetrics, OrbitMetrics};
pub use static_map::{analyze_static, StaticReport};
