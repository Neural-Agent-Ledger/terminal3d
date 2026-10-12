//! `re_render::bullet_sprite` — distance-accurate ASCII point-sprite for a projectile.
//!
//! The VISUAL half of the multiplayer-replication testbed. A bullet is a
//! single world-space point; this module projects it into the viewport and
//! writes `{glyph, fg, bg}` into a [`Layer`] with size/colour scaled by
//! distance, occluded against the opaque-geometry depth buffer.
//!
//! It is **engine-pure**: no terminal, no networking, no I/O, no per-frame heap
//! allocation. `ts_netplay` calls [`splat_bullets`] (or the single-bullet
//! [`project_bullet`] + [`write_bullet`]) on the list of replicated bullet world
//! positions — networking types never reach `re_render`.
//!
//! ## Projection (mirrors `emitter::splat_particles`)
//!
//! The world→screen projection is the INVERSE of the raycaster's per-column cast,
//! identical to the one inlined in [`crate::emitter::splat_particles`]:
//!
//! * camera basis `dir = (sin yaw, cos yaw)`, plane direction `(cos yaw, −sin yaw)`,
//! * `depth   = (p − cam) · dir`      — the fisheye-free PERPENDICULAR distance, the
//!   exact value the raycaster stores in [`Cell::depth`], so the occlusion test
//!   against the opaque layer is apples-to-apples,
//! * `lateral = (p − cam) · plane_dir`,
//! * `camera_x = (lateral / depth) / plane_scale ∈ [−1, 1]` ⇒
//!   `col = (camera_x + 1)/2 · cols`,
//! * `row = horizon − proj_plane_dist · (p.z − cam.z) / depth`.
//!
//! NOTE: that projection lives INLINE inside `emitter::splat_particles`, not
//! as a standalone `pub(crate)` helper, so there is nothing to import. Rather
//! than make emitter internals public, the ~12-line projection is
//! re-implemented here against the SAME public inputs (`view_params`, `cam.dir()`,
//! `plane_scale`, `proj_plane_dist`). If `emitter.rs` is ever refactored to expose a
//! `pub(crate) fn project_world(...)`, this module should switch to calling it. The
//! two must stay in lock-step; the `projection_matches_emitter` test guards that.

use glam::{Vec3, Vec3Swizzles};
use re_core::{Cell, Layer};
use re_world::CameraPose;

use crate::ascii_shader::{dim_rgb, mix_rgb, FogParams};
use crate::{view_params, Fov};

// ─────────────────────────────── tunables ───────────────────────────────────

/// Perpendicular depth (world cells) at/under which a bullet is rendered NEAR: a
/// large bright glyph plus a background glow halo. Spec §6: "Near (e.g. < 4)".
pub const NEAR_DEPTH: f32 = 4.0;

/// Perpendicular depth (world cells) at/over which a bullet is rendered FAR: a dim,
/// fog-tinted point, no background. Spec §6: "Far (e.g. > 18)". Anything between
/// `NEAR_DEPTH` and this is the MID tier.
pub const FAR_DEPTH: f32 = 18.0;

/// NEAR-tier primary glyph: the heaviest dot — a bright, fat projectile up close.
pub const GLYPH_NEAR: char = '\u{25CF}'; // ● BLACK CIRCLE
/// MID-tier glyph: a medium bullet mass.
pub const GLYPH_MID: char = '\u{2022}'; // • BULLET
/// FAR-tier glyph: a single dim point.
pub const GLYPH_FAR: char = '\u{00B7}'; // · MIDDLE DOT

/// The bullet's bright base foreground (NEAR end of the fg interpolation). A hot
/// tracer white-yellow so a near bullet pops against the muted world.
pub const FG_BRIGHT: u32 = 0x00FF_F2C0;
/// The bullet's dim foreground (FAR end, before fog tint is mixed in). A muted amber
/// so a distant tracer is a faint ember, not a hard white speck.
pub const FG_DIM: u32 = 0x008A_6A30;

/// Background glow colour written behind a NEAR bullet (the halo). A deep warm glow
/// that reads as a muzzle/tracer bloom without washing the glyph out.
pub const BG_GLOW: u32 = 0x003A_2808;

/// Distance at which the NEAR-tier glow has fully faded out. The glow intensity
/// ramps `1.0` at `depth = 0` → `0.0` at this depth, so the halo only ever appears
/// on genuinely close bullets and dies smoothly into the MID tier.
pub const GLOW_FADE_DEPTH: f32 = NEAR_DEPTH;

/// Default fog for a bullet when the caller has no per-level atmosphere handy.
/// Far bullets fade toward this; matches the renderer's fade-to-black default.
pub const DEFAULT_FOG: FogParams = FogParams::DEFAULT;

// ───────────────────────────── depth tier ───────────────────────────────────

/// Which distance band a bullet falls into — selects glyph weight + whether a glow
/// halo is drawn (spec §6 table).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BulletTier {
    /// Close: large bright glyph + a background glow halo.
    Near,
    /// Medium range: a brighter mid-weight glyph, no background.
    Mid,
    /// Far: a dim, fog-tinted point, no background.
    Far,
}

impl BulletTier {
    /// Classify a perpendicular `depth` (world cells) into a tier.
    #[inline]
    pub fn from_depth(depth: f32) -> Self {
        if depth <= NEAR_DEPTH {
            BulletTier::Near
        } else if depth >= FAR_DEPTH {
            BulletTier::Far
        } else {
            BulletTier::Mid
        }
    }

    /// The glyph (as `u32` code point, the [`Cell::glyph`] representation) for this tier.
    #[inline]
    pub fn glyph(self) -> u32 {
        match self {
            BulletTier::Near => GLYPH_NEAR as u32,
            BulletTier::Mid => GLYPH_MID as u32,
            BulletTier::Far => GLYPH_FAR as u32,
        }
    }
}

// ──────────────────────────── projection ────────────────────────────────────

/// A bullet projected to the viewport: its screen cell + perpendicular depth.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectedBullet {
    /// Screen column (`0..cols`).
    pub col: usize,
    /// Screen row (`0..rows`).
    pub row: usize,
    /// Perpendicular forward distance (world cells) — the occlusion + tier driver,
    /// the same quantity stored in [`Cell::depth`] by the raycaster.
    pub depth: f32,
}

/// Project a bullet's world position into a `(col, row, depth)` cell, or `None` if it
/// is behind the camera, outside the horizontal FOV, or off-screen vertically.
///
/// Pure: a function of `(pos, cam, fov, cols, rows)` only — no occlusion test here
/// (that needs the opaque layer; see [`write_bullet`] / [`splat_bullets`]).
///
/// This is the projection from `emitter::splat_particles`, re-derived against the
/// same public inputs (see the module-level NOTE on why it is not imported).
pub fn project_bullet(
    pos: Vec3,
    cam: &CameraPose,
    fov: &Fov,
    cols: usize,
    rows: usize,
) -> Option<ProjectedBullet> {
    if cols == 0 || rows == 0 {
        return None;
    }
    let view = view_params(cols, rows, fov);
    let plane_scale = view.plane_scale;
    let proj = view.proj_plane_dist;

    // Camera basis (matches the raycaster's column projection).
    let (dir_x, dir_y) = cam.dir();
    let plane_dir_x = cam.yaw.cos();
    let plane_dir_y = -cam.yaw.sin();
    let horizon = (rows as f32) * 0.5;

    let rel = pos.xy() - cam.pos.xy();
    let depth = rel.x * dir_x + rel.y * dir_y; // perpendicular forward distance
    if depth <= 1e-3 {
        return None; // behind / on the camera plane
    }
    let lateral = rel.x * plane_dir_x + rel.y * plane_dir_y;
    let camera_x = (lateral / depth) / plane_scale;
    if !(-1.0..=1.0).contains(&camera_x) {
        return None; // outside horizontal FOV
    }
    let colf = (camera_x + 1.0) * 0.5 * cols as f32;
    let col = colf.floor() as i32;
    if col < 0 || col as usize >= cols {
        return None;
    }

    let dz = pos.z - cam.pos.z;
    let rowf = horizon - proj * dz / depth;
    let row = rowf.floor() as i32;
    if row < 0 || row as usize >= rows {
        return None;
    }

    Some(ProjectedBullet {
        col: col as usize,
        row: row as usize,
        depth,
    })
}

// ─────────────────────────── colour / cell ──────────────────────────────────

/// The foreground colour for a bullet at perpendicular `depth`: interpolate
/// bright→dim across the `[0, FAR_DEPTH]` span, then blend toward the fog colour
/// (mirrors `Material::sample`'s fog step — `fog` is `1.0` near → `0.0` far, so a
/// `None`-tint fog fades to black and a tinted fog fades toward `color`).
#[inline]
pub fn bullet_fg(depth: f32, fog: &FogParams) -> u32 {
    // bright→dim by raw distance, independent of the fog band edges so the tracer
    // reads as fading with range even inside a short fog band.
    let t = (depth / FAR_DEPTH).clamp(0.0, 1.0);
    let base = mix_rgb(FG_BRIGHT, FG_DIM, t);
    let fog_f = distance_fog(depth, fog);
    match fog.color {
        None => dim_rgb(base, fog_f),
        Some(c) => mix_rgb(base, c, 1.0 - fog_f),
    }
}

/// The background glow for a NEAR bullet: [`BG_GLOW`] scaled by how close the bullet
/// is (full at `depth = 0`, gone at [`GLOW_FADE_DEPTH`]). Returns `0` (transparent)
/// for any non-near depth — only the NEAR tier carries a halo (spec §6).
#[inline]
pub fn bullet_bg(depth: f32) -> u32 {
    if depth >= GLOW_FADE_DEPTH {
        return 0;
    }
    let glow = (1.0 - depth / GLOW_FADE_DEPTH).clamp(0.0, 1.0);
    dim_rgb(BG_GLOW, glow)
}

/// Build the [`Cell`] for a bullet at a projected `depth` — glyph by tier, fg
/// interpolated bright→dim + fog-blended, bg glow on the NEAR tier only. `depth` is
/// stored so a later overlay (or the present diff) can depth-test it consistently.
#[inline]
pub fn bullet_cell(depth: f32, fog: &FogParams) -> Cell {
    let tier = BulletTier::from_depth(depth);
    Cell {
        glyph: tier.glyph(),
        fg: bullet_fg(depth, fog),
        bg: bullet_bg(depth),
        depth,
        ..Cell::default()
    }
}

// ───────────────────────────── write ────────────────────────────────────────

/// Write ONE projected bullet into `out`, depth-tested against `opaque`:
///
/// * **Occlusion** — if a solid wall in `opaque` at the bullet's cell is NEARER than
///   the bullet (`wall.depth < depth`), the bullet is behind it and nothing is
///   written (spec §6).
/// * Otherwise the bullet [`Cell`] (glyph + fg + bg by tier) is set at `(col, row)`.
/// * **NEAR size** — a NEAR bullet additionally paints a faint glow into the bg of
///   its 4 orthogonal neighbours (an apparent-size bloom), depth-tested per cell and
///   only where that neighbour is currently empty, so it never overwrites a wall, a
///   nearer object, or a MID/FAR bullet drawn there. This is the spec's optional
///   "extra apparent size" near-field, done as a halo of background cells (no glyph),
///   which is the cleanest single-`Layer` analogue of a 2×2 blit.
///
/// Returns `true` if the bullet's primary cell was drawn (i.e. not occluded /
/// off-screen), `false` otherwise.
pub fn write_bullet(p: ProjectedBullet, fog: &FogParams, opaque: &Layer, out: &mut Layer) -> bool {
    let (cols, rows) = out.dims();
    if p.col >= cols || p.row >= rows {
        return false;
    }
    // Occlusion against opaque geometry.
    if let Some(wall) = opaque.get(p.col, p.row) {
        if wall.depth < p.depth {
            return false;
        }
    }

    let cell = bullet_cell(p.depth, fog);
    out.set(p.col, p.row, cell);

    // NEAR-tier apparent-size halo: bleed the glow into orthogonal neighbours' bg.
    if BulletTier::from_depth(p.depth) == BulletTier::Near {
        let glow = bullet_bg(p.depth);
        if glow != 0 {
            // Slightly dimmer than the centre so the bloom falls off outward.
            let halo = dim_rgb(glow, 0.55);
            let (col, row) = (p.col as isize, p.row as isize);
            for (dc, dr) in [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)] {
                let nc = col + dc;
                let nr = row + dr;
                if nc < 0 || nr < 0 || nc as usize >= cols || nr as usize >= rows {
                    continue;
                }
                let (nc, nr) = (nc as usize, nr as usize);
                // Occlude the halo cell too.
                if let Some(wall) = opaque.get(nc, nr) {
                    if wall.depth < p.depth {
                        continue;
                    }
                }
                // Only fill EMPTY cells — never stomp a glyph already drawn here.
                let occupied = out
                    .get(nc, nr)
                    .map(|c| c.glyph != u32::from(' ') || c.bg != 0)
                    .unwrap_or(false);
                if occupied {
                    continue;
                }
                out.set(
                    nc,
                    nr,
                    Cell {
                        glyph: u32::from(' '),
                        fg: 0,
                        bg: halo,
                        depth: p.depth,
                        ..Cell::default()
                    },
                );
            }
        }
    }

    true
}

/// Project + write a whole list of bullet world positions into `out`, depth-tested
/// against `opaque`. `out` is CLEARED first, so only live, visible bullets mark
/// cells present (the sparse-overlay claim — same contract as
/// `emitter::splat_particles`).
///
/// This is the entry point `ts_netplay` calls each frame on its replicated bullets:
/// `splat_bullets(&bullet_world_positions, &cam, &fov, &fog, &opaque, &mut layer)`.
/// No networking type crosses the boundary — just `&[Vec3]`.
///
/// Returns the number of bullets actually drawn (visible, not occluded / off-screen).
pub fn splat_bullets(
    bullets: &[Vec3],
    cam: &CameraPose,
    fov: &Fov,
    fog: &FogParams,
    opaque: &Layer,
    out: &mut Layer,
) -> usize {
    out.clear();
    let (cols, rows) = out.dims();
    if cols == 0 || rows == 0 {
        return 0;
    }
    let mut drawn = 0;
    for &pos in bullets {
        if let Some(proj) = project_bullet(pos, cam, fov, cols, rows) {
            if write_bullet(proj, fog, opaque, out) {
                drawn += 1;
            }
        }
    }
    drawn
}

// ─────────────────────────── fog (private mirror) ───────────────────────────

/// Distance fog: `1.0` near → `0.0` far over the `[fog.near, fog.far]` band. A copy
/// of the (private) `ascii_shader::distance_fog` so the bullet fog reads identically
/// to wall fog without widening that module's surface. A degenerate band
/// (`far <= near`) is a hard step at `near`.
#[inline]
fn distance_fog(distance: f32, fog: &FogParams) -> f32 {
    if distance <= fog.near {
        1.0
    } else if distance >= fog.far || fog.far <= fog.near {
        0.0
    } else {
        1.0 - (distance - fog.near) / (fog.far - fog.near)
    }
}

// ───────────────────────────────── tests ────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// A camera at the origin looking down +Y (`yaw = 0`, since `dir = (sin, cos)`).
    fn cam_facing_north() -> CameraPose {
        CameraPose {
            pos: Vec3::new(0.0, 0.0, 0.5),
            yaw: 0.0,
            pitch: 0.0,
        }
    }

    /// An empty opaque layer (no walls) — nothing occludes.
    fn empty_opaque(cols: usize, rows: usize) -> Layer {
        Layer::new(cols, rows, re_core::LayerBlend::Opaque)
    }

    /// A bullet overlay layer (glyph + its own bg glow over the world).
    fn bullet_layer(cols: usize, rows: usize) -> Layer {
        Layer::new(cols, rows, re_core::LayerBlend::GlyphOnly)
    }

    #[test]
    fn tier_classification_matches_spec_bands() {
        assert_eq!(BulletTier::from_depth(0.0), BulletTier::Near);
        assert_eq!(BulletTier::from_depth(NEAR_DEPTH), BulletTier::Near);
        assert_eq!(BulletTier::from_depth(NEAR_DEPTH + 0.1), BulletTier::Mid);
        assert_eq!(BulletTier::from_depth(FAR_DEPTH - 0.1), BulletTier::Mid);
        assert_eq!(BulletTier::from_depth(FAR_DEPTH), BulletTier::Far);
        assert_eq!(BulletTier::from_depth(100.0), BulletTier::Far);
    }

    #[test]
    fn far_bullet_is_a_dim_point_no_bg() {
        let cell = bullet_cell(40.0, &DEFAULT_FOG);
        assert_eq!(cell.glyph, GLYPH_FAR as u32, "far → dim point glyph");
        assert_eq!(cell.bg, 0, "far bullet carries no background");
        // Fully fogged out (depth >= fog.far) → fg dimmed to black with the default
        // fade-to-black fog.
        assert_eq!(cell.fg, 0, "far bullet fades out under default fog");
    }

    #[test]
    fn near_bullet_is_large_bright_glyph_with_bg_glow() {
        let cell = bullet_cell(0.5, &DEFAULT_FOG);
        assert_eq!(cell.glyph, GLYPH_NEAR as u32, "near → large glyph");
        assert!(cell.bg != 0, "near bullet has a bg glow halo");
        assert!(cell.fg != 0, "near bullet fg is bright");
        // Near fg must be brighter than a mid bullet's fg (closer = brighter).
        let mid = bullet_cell(10.0, &DEFAULT_FOG);
        let lum = |c: u32| {
            let r = (c >> 16) & 0xFF;
            let g = (c >> 8) & 0xFF;
            let b = c & 0xFF;
            r + g + b
        };
        assert!(
            lum(cell.fg) > lum(mid.fg),
            "near fg ({:06X}) brighter than mid fg ({:06X})",
            cell.fg,
            mid.fg
        );
    }

    #[test]
    fn fg_interpolates_bright_to_dim_with_distance() {
        // No fog band engaged here so we isolate the bright→dim interpolation.
        let no_fog = FogParams {
            near: 1000.0,
            far: 2000.0,
            color: None,
            height: None,
        };
        let lum = |c: u32| ((c >> 16) & 0xFF) + ((c >> 8) & 0xFF) + (c & 0xFF);
        let near = bullet_fg(0.0, &no_fog);
        let far = bullet_fg(FAR_DEPTH, &no_fog);
        assert!(lum(near) > lum(far), "closer bullet is brighter");
        assert_eq!(near, FG_BRIGHT, "at depth 0 fg is the bright base");
    }

    #[test]
    fn fog_color_tints_a_distant_bullet_toward_the_fog() {
        let blue_fog = FogParams {
            near: 2.0,
            far: 20.0,
            color: Some(0x0000_00FF),
            height: None,
        };
        let fg = bullet_fg(19.0, &blue_fog);
        // Heavily tinted toward blue: blue channel should dominate red.
        let r = (fg >> 16) & 0xFF;
        let b = fg & 0xFF;
        assert!(
            b > r,
            "distant bullet tinted toward fog blue (fg={:06X})",
            fg
        );
    }

    // ── DIRECTIONS: the four cardinal fire directions project to the right region ──

    #[test]
    fn bullet_ahead_projects_near_center_growing_toward_camera() {
        let cam = cam_facing_north();
        let (cols, rows) = (80usize, 40usize);
        // Straight ahead (down +Y), far then near.
        let far = project_bullet(Vec3::new(0.0, 16.0, 0.5), &cam, &Fov::default(), cols, rows)
            .expect("ahead-far visible");
        let near = project_bullet(Vec3::new(0.0, 2.0, 0.5), &cam, &Fov::default(), cols, rows)
            .expect("ahead-near visible");
        let center = cols / 2;
        // Both centred horizontally (within a cell or two of mid-screen).
        assert!(
            (far.col as isize - center as isize).abs() <= 2,
            "far ahead is centred"
        );
        assert!(
            (near.col as isize - center as isize).abs() <= 2,
            "near ahead is centred"
        );
        // Depth shrinks as it approaches → grows in apparent weight.
        assert!(near.depth < far.depth, "approaching bullet depth shrinks");
        assert_eq!(BulletTier::from_depth(near.depth), BulletTier::Near);
        assert_eq!(BulletTier::from_depth(far.depth), BulletTier::Mid);
    }

    #[test]
    fn bullet_to_the_right_projects_right_of_center() {
        let cam = cam_facing_north();
        let (cols, rows) = (80usize, 40usize);
        // Ahead and to the +X (camera's right when facing +Y: plane_dir = (cos,−sin) =
        // (1,0) at yaw 0, so +X is positive lateral → right of centre).
        let p = project_bullet(Vec3::new(4.0, 10.0, 0.5), &cam, &Fov::default(), cols, rows)
            .expect("right bullet visible");
        assert!(
            p.col > cols / 2,
            "bullet to the right lands right of centre (col={})",
            p.col
        );
    }

    #[test]
    fn bullet_to_the_left_projects_left_of_center() {
        let cam = cam_facing_north();
        let (cols, rows) = (80usize, 40usize);
        let p = project_bullet(
            Vec3::new(-4.0, 10.0, 0.5),
            &cam,
            &Fov::default(),
            cols,
            rows,
        )
        .expect("left bullet visible");
        assert!(
            p.col < cols / 2,
            "bullet to the left lands left of centre (col={})",
            p.col
        );
    }

    #[test]
    fn bullet_behind_camera_is_not_projected() {
        let cam = cam_facing_north();
        // Behind the camera (−Y) → depth ≤ 0 → culled.
        let p = project_bullet(Vec3::new(0.0, -5.0, 0.5), &cam, &Fov::default(), 80, 40);
        assert!(p.is_none(), "bullet behind camera (away) is not drawn");
    }

    // ── OCCLUSION ──

    #[test]
    fn bullet_behind_a_wall_writes_nothing() {
        let cam = cam_facing_north();
        let (cols, rows) = (80usize, 40usize);
        let fov = Fov::default();
        // Bullet straight ahead at depth ~10.
        let proj = project_bullet(Vec3::new(0.0, 10.0, 0.5), &cam, &fov, cols, rows)
            .expect("bullet visible before occlusion");

        // Opaque wall at the SAME cell but NEARER (depth 5 < 10) → bullet occluded.
        let mut opaque = empty_opaque(cols, rows);
        opaque.set(
            proj.col,
            proj.row,
            Cell {
                glyph: u32::from('#'),
                depth: 5.0,
                ..Cell::default()
            },
        );

        let mut out = bullet_layer(cols, rows);
        let drawn = write_bullet(proj, &DEFAULT_FOG, &opaque, &mut out);
        assert!(!drawn, "occluded bullet returns not-drawn");
        assert!(
            out.get(proj.col, proj.row).map(|c| c.glyph) == Some(u32::from(' '))
                || out.get(proj.col, proj.row).is_none(),
            "occluded bullet leaves its cell empty"
        );
    }

    #[test]
    fn bullet_in_front_of_a_wall_is_drawn() {
        let cam = cam_facing_north();
        let (cols, rows) = (80usize, 40usize);
        let fov = Fov::default();
        let proj = project_bullet(Vec3::new(0.0, 6.0, 0.5), &cam, &fov, cols, rows)
            .expect("bullet visible");
        // Wall FARTHER than the bullet (depth 20 > 6) → bullet drawn.
        let mut opaque = empty_opaque(cols, rows);
        opaque.set(
            proj.col,
            proj.row,
            Cell {
                depth: 20.0,
                ..Cell::default()
            },
        );
        let mut out = bullet_layer(cols, rows);
        assert!(write_bullet(proj, &DEFAULT_FOG, &opaque, &mut out));
        assert_ne!(
            out.get(proj.col, proj.row).map(|c| c.glyph),
            Some(u32::from(' '))
        );
    }

    #[test]
    fn splat_clears_and_counts_visible_bullets() {
        let cam = cam_facing_north();
        let (cols, rows) = (80usize, 40usize);
        let fov = Fov::default();
        let opaque = empty_opaque(cols, rows);
        let mut out = bullet_layer(cols, rows);

        let bullets = vec![
            Vec3::new(0.0, 8.0, 0.5),  // visible ahead
            Vec3::new(-1.5, 8.0, 0.5), // visible, left of centre (inside FOV)
            Vec3::new(0.0, -5.0, 0.5), // behind camera → culled
        ];
        let drawn = splat_bullets(&bullets, &cam, &fov, &DEFAULT_FOG, &opaque, &mut out);
        assert_eq!(drawn, 2, "two of three bullets are visible");

        // A second splat with no bullets clears the layer.
        let cleared = splat_bullets(&[], &cam, &fov, &DEFAULT_FOG, &opaque, &mut out);
        assert_eq!(cleared, 0);
    }

    #[test]
    fn projection_matches_emitter_screen_mapping() {
        // Sanity: our col mapping agrees with the documented emitter projection — a
        // dead-centre bullet maps to ~cols/2, a hard-right one to a higher column,
        // and depth equals perpendicular distance (here exactly Y, facing +Y).
        let cam = cam_facing_north();
        let (cols, rows) = (100usize, 50usize);
        let centre =
            project_bullet(Vec3::new(0.0, 10.0, 0.5), &cam, &Fov::default(), cols, rows).unwrap();
        assert!((centre.col as isize - 50).abs() <= 1);
        assert!((centre.depth - 10.0).abs() < 1e-3, "depth == perp distance");
    }
}
