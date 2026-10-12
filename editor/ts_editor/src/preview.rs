//! First-person ASCII parity preview, rendered with the SAME engine as the game.
use crate::state::EditorState;
use re_core::{Cell, Framebuffer, Vec3};
use re_level::fields::FieldShape;
use re_render::{
    composite::composite_dump_once, lines::world_to_screen, registry_for_map, Emitter, Fov,
};

/// Render the parity preview into the rect (x0,y0,w,h) of `fb`.
///
/// LIVE EMITTER PREVIEW: the level's authored/edited `EmitterSpec`s are resolved
/// into `re_render::Emitter`s and handed to `composite_dump_once`, which itself
/// runs ~24 deterministic particle ticks before splatting — so the user SEES the
/// emitter's particles spread out in the 3D view, and every focused-field edit
/// (rate/speed/spread/colour/…) is reflected the next frame. The single-shot
/// dump is deterministic, so the preview is a stable still rather than a moving
/// animation; a continuously-advancing pool is a follow-up (it would need the
/// editor to own a `ParticlePool` + frame clock across draws).
pub fn render_preview(
    state: &EditorState,
    fb: &mut Framebuffer,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
) {
    let mut sub = Framebuffer::new(w, h);
    let registry = registry_for_map(&state.map);
    let emitters: Vec<Emitter> = state
        .level
        .emitters
        .iter()
        .map(Emitter::from_spec)
        .collect();
    composite_dump_once(
        &state.map,
        &state.cam,
        &registry,
        &Fov::default(),
        &emitters,
        &mut sub,
    );
    // SOUND FIELDS view aid — a subtle marker for each field in the 3D preview:
    // gated on the ONE call below, so with `fields_aid` at its default OFF
    // this pass never runs and `PREVIEW_DIGEST_BEFORE` stays byte-identical.
    if state.fields_aid {
        overlay_field_markers(state, &mut sub, w, h);
    }
    for r in 0..h {
        for c in 0..w {
            if let Some(cell) = sub.get(c, r) {
                fb.set(x0 + c, y0 + r, *cell);
            }
        }
    }
}

/// Project each STANDALONE Sphere field's world position through the SAME
/// camera/projection the cast used (`re_render::lines::world_to_screen` — the
/// authored-line overlay's own projector) and, if it lands on-screen and is
/// not occluded, draw a `⊙` tinted by archetype family over the already-cast
/// `sub` framebuffer. Room-shaped fields (and inline-attached ones, which
/// `Level::fields()` still includes but which have no fixed cell an author
/// places via this editor — see `crate::field`'s module doc) have no single
/// world point to project, so they draw no marker here; the 2D plan's
/// `cell_tint` still washes their footprint where one exists.
///
/// Occlusion reuses the cast's OWN per-cell `Cell::depth` (the same
/// fisheye-free perpendicular distance `world_to_screen` returns): a marker
/// behind nearer geometry loses the depth compare and is skipped, so it never
/// draws through a wall. The `0.05`-unit bias avoids z-fighting flicker for a
/// field centred exactly on a surface.
fn overlay_field_markers(state: &EditorState, sub: &mut Framebuffer, w: usize, h: usize) {
    const OCCLUSION_BIAS: f32 = 0.05;
    let fov = Fov::default();
    for f in &state.level.resolved_fields {
        let FieldShape::Sphere { at } = f.shape else {
            continue;
        };
        let Some(sp) = world_to_screen(&state.cam, &fov, w, h, Vec3::new(at[0], at[1], at[2]))
        else {
            continue;
        };
        let (col, row) = (sp.col.round(), sp.row.round());
        if col < 0.0 || row < 0.0 {
            continue;
        }
        let (col, row) = (col as usize, row as usize);
        if col >= w || row >= h {
            continue;
        }
        let Some(existing) = sub.get(col, row) else {
            continue;
        };
        if existing.depth < sp.depth - OCCLUSION_BIAS {
            continue; // nearer geometry occludes the marker.
        }
        let fg = crate::field::family_color(&f.archetype);
        sub.set(
            col,
            row,
            Cell {
                glyph: '\u{2299}' as u32,
                fg,
                bg: existing.bg,
                ..Cell::default()
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use re_world::CameraPose;
    #[test]
    fn preview_is_nonblank_for_a_walled_room() {
        let map = re_world::test_room();
        let cam = CameraPose::spawn_in_test_room();
        let mut state = crate::state::EditorState::new_empty(1, 1, 1);
        state.map = map;
        state.cam = cam;
        let mut fb = Framebuffer::new(80, 24);
        render_preview(&state, &mut fb, 0, 0, 80, 24);
        let inked = fb.cells().iter().filter(|c| c.glyph != ' ' as u32).count();
        assert!(inked > 0, "parity preview must not be blank");
    }
}
