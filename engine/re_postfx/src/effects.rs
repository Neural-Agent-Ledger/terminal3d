//! The seven post-process passes. Each is a pure `apply(fb, &Fx, master)` (some
//! also take a clean `src` copy for neighbour sampling). They read depth /
//! surface / fg / bg and modify fg / bg / glyph only — the cell `style`,
//! geometry, and all other fields are preserved (every cell is rebuilt from the
//! source cell with `..` so untouched fields pass through unchanged).

pub(crate) mod bloom;
pub(crate) mod chromatic;
pub(crate) mod color_temp;
pub(crate) mod dither;
pub(crate) mod dof;
pub(crate) mod edge;
pub(crate) mod film_grain;
pub(crate) mod grade;
pub(crate) mod posterize;
pub(crate) mod scanlines;
pub(crate) mod ssao;
pub(crate) mod vignette;
pub(crate) mod world_anchor;
