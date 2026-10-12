//! `face` — which of the four axis-aligned WALL FACES a surface decal is painted on.
//!
//! ## Why this type has to exist
//!
//! A material is bound to a CELL, not to a face: the renderer resolves a cell's
//! material from its `side_data` tag, and [`crate::Cell`]'s `surface` /
//! `side` fields carry no *sign* — `side = 0` means "an NS (y-normal) face",
//! `side = 1` means "an EW (x-normal) face", with no way to tell +y from −y.
//! For a texture that is symmetric (brick, grain, circuit) that is fine. For a
//! DECAL — a note placard, a sign, anything with a reading direction — it is not:
//! without a face sign the text renders on all four faces of the cell, including
//! the reverse side, where it appears mirrored in the adjoining room.
//!
//! So a decal recipe carries its bound face explicitly, as this plain-data enum,
//! and the shader gates on it (see `re_render`'s note-wall recipe).
//!
//! ## The two gating paths
//!
//! 1. **Preferred — the face NORMAL.** When per-cell lighting is present the
//!    renderer knows the true outward normal of the face being shaded (it points
//!    back toward the camera). [`FaceDir::normal`] gives the decal's own outward
//!    normal; the two must agree.
//! 2. **Fallback — the NS/EW pair.** With no lighting context (a focused unit
//!    test, or any caller that shades without a light) only `side` is available.
//!    [`FaceDir::side`] reduces the face to that one bit, so the decal renders on
//!    the matching axis PAIR (both +x and −x for an x-face decal). Coarser, but
//!    defined and never garbage.
//!
//! Engine-agnostic plain data: `re_world` carries it in a material recipe,
//! `re_level`/`re_game` synthesize it from authored `facing`, `re_render` gates on
//! it. No terminal, no render types.

/// One of the four axis-aligned vertical wall faces, named by its OUTWARD normal
/// — the direction a reader stands in to read what is painted on it.
///
/// `PosY` is the `+y` face (the authored `facing = +y`), and so on around the
/// compass. There is deliberately no floor/ceiling variant: a decal on a
/// horizontal surface has no reading direction to protect (and floor/ceiling
/// cells carry no `wall_v`), so the type stays exhaustive over the wall faces
/// only.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum FaceDir {
    /// Outward normal `+y`. Authored `facing = +y`; NS face (`side == 0`).
    #[default]
    PosY,
    /// Outward normal `−y`. Authored `facing = -y`; NS face (`side == 0`).
    NegY,
    /// Outward normal `+x`. Authored `facing = +x`; EW face (`side == 1`).
    PosX,
    /// Outward normal `−x`. Authored `facing = -x`; EW face (`side == 1`).
    NegX,
}

/// Minimum agreement between a cell's shaded face normal and a decal's own
/// outward normal for the decal to draw. The two are axis-aligned unit vectors,
/// so a match is `dot == 1` and a mismatch is `dot ∈ {0, −1}`; the generous
/// threshold only guards against a renormalisation wobble in the caster's
/// normal, never against a genuinely different face.
pub const FACE_MATCH_MIN_DOT: f32 = 0.5;

impl FaceDir {
    /// Every face, in authored-compass order. Handy for exhaustive tests and for
    /// tooling that enumerates the choices.
    pub const ALL: [FaceDir; 4] = [FaceDir::PosY, FaceDir::NegY, FaceDir::PosX, FaceDir::NegX];

    /// The face's OUTWARD unit normal in world space (`z` is always 0 — these are
    /// vertical faces).
    #[inline]
    pub fn normal(self) -> crate::Vec3 {
        match self {
            FaceDir::PosY => crate::Vec3::new(0.0, 1.0, 0.0),
            FaceDir::NegY => crate::Vec3::new(0.0, -1.0, 0.0),
            FaceDir::PosX => crate::Vec3::new(1.0, 0.0, 0.0),
            FaceDir::NegX => crate::Vec3::new(-1.0, 0.0, 0.0),
        }
    }

    /// The integer step from the wall cell TOWARD the reader (i.e. the outward
    /// normal as a cell delta). The wall body is in the opposite direction.
    #[inline]
    pub fn step(self) -> (i32, i32) {
        match self {
            FaceDir::PosY => (0, 1),
            FaceDir::NegY => (0, -1),
            FaceDir::PosX => (1, 0),
            FaceDir::NegX => (-1, 0),
        }
    }

    /// The renderer's coarse `side` bit for this face: `0` for an NS (y-normal)
    /// face, `1` for an EW (x-normal) face. This is all a caller with no lighting
    /// context can check, so it is the documented fallback gate — it cannot tell
    /// `PosX` from `NegX`.
    #[inline]
    pub fn side(self) -> u8 {
        match self {
            FaceDir::PosY | FaceDir::NegY => 0,
            FaceDir::PosX | FaceDir::NegX => 1,
        }
    }

    /// Whether a face whose outward normal is `face_n` is THIS face. Axis-aligned
    /// unit vectors, so this is a dot-product test against
    /// [`FACE_MATCH_MIN_DOT`] — exact for a clean normal, tolerant of a
    /// renormalisation wobble, and decisive against the reverse face (dot `−1`)
    /// and the perpendicular faces (dot `0`).
    #[inline]
    pub fn matches_normal(self, face_n: crate::Vec3) -> bool {
        self.normal().dot(face_n) >= FACE_MATCH_MIN_DOT
    }

    /// Parse the authored `.lvl` `facing` shortcut. Accepts the four compass
    /// tokens; anything else is `None` so the caller can decide the default
    /// (rather than silently picking a face).
    pub fn from_token(tok: &str) -> Option<FaceDir> {
        match tok.trim() {
            "+y" => Some(FaceDir::PosY),
            "-y" => Some(FaceDir::NegY),
            "+x" => Some(FaceDir::PosX),
            "-x" => Some(FaceDir::NegX),
            _ => None,
        }
    }

    /// The authored token for this face (round-trips [`from_token`](Self::from_token)).
    pub fn token(self) -> &'static str {
        match self {
            FaceDir::PosY => "+y",
            FaceDir::NegY => "-y",
            FaceDir::PosX => "+x",
            FaceDir::NegX => "-x",
        }
    }

    /// Map `re_level`'s legacy `facing` yaw-column encoding (`+y = 0`, `+x = 120`,
    /// `−y = 240`, `−x = 360`, on a 480-column circle) onto a face.
    ///
    /// Any other value is snapped to the NEAREST of the four cardinals rather than
    /// rejected: a hand-authored numeric `facing` is a legal (if unusual) level
    /// input, and a decal must always resolve to some defined face.
    pub fn from_yaw_col(yaw_col: i32) -> FaceDir {
        // Reduce onto [0, 480) then pick the nearest quadrant centre.
        let c = yaw_col.rem_euclid(YAW_COLS);
        // Quadrant boundaries at 60/180/300/420 (half-way between the cardinals);
        // 420..479 wraps back around to +y, so it shares the first arm.
        match c {
            60..=179 => FaceDir::PosX,
            180..=299 => FaceDir::NegY,
            300..=419 => FaceDir::NegX,
            _ => FaceDir::PosY,
        }
    }
}

/// Columns in the legacy full-circle yaw encoding `re_level`'s `facing` key uses
/// (`+y = 0`, `+x = 120`, `−y = 240`, `−x = 360`).
pub const YAW_COLS: i32 = 480;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normals_are_opposite_across_a_pair() {
        assert_eq!(FaceDir::PosY.normal(), -FaceDir::NegY.normal());
        assert_eq!(FaceDir::PosX.normal(), -FaceDir::NegX.normal());
    }

    #[test]
    fn face_matches_only_its_own_normal() {
        for f in FaceDir::ALL {
            for g in FaceDir::ALL {
                assert_eq!(
                    f.matches_normal(g.normal()),
                    f == g,
                    "{f:?} vs normal of {g:?}"
                );
            }
        }
    }

    #[test]
    fn side_bit_splits_ns_from_ew() {
        assert_eq!(FaceDir::PosY.side(), 0);
        assert_eq!(FaceDir::NegY.side(), 0);
        assert_eq!(FaceDir::PosX.side(), 1);
        assert_eq!(FaceDir::NegX.side(), 1);
    }

    #[test]
    fn yaw_col_encoding_round_trips_the_cardinals() {
        assert_eq!(FaceDir::from_yaw_col(0), FaceDir::PosY);
        assert_eq!(FaceDir::from_yaw_col(120), FaceDir::PosX);
        assert_eq!(FaceDir::from_yaw_col(240), FaceDir::NegY);
        assert_eq!(FaceDir::from_yaw_col(360), FaceDir::NegX);
        // Off-cardinal values snap to the nearest face; out-of-range wraps.
        assert_eq!(FaceDir::from_yaw_col(130), FaceDir::PosX);
        assert_eq!(FaceDir::from_yaw_col(-120), FaceDir::NegX);
        assert_eq!(FaceDir::from_yaw_col(480), FaceDir::PosY);
    }

    #[test]
    fn tokens_round_trip() {
        for f in FaceDir::ALL {
            assert_eq!(FaceDir::from_token(f.token()), Some(f));
        }
        assert_eq!(FaceDir::from_token("up"), None);
    }

    #[test]
    fn step_points_toward_the_reader() {
        assert_eq!(FaceDir::PosX.step(), (1, 0));
        assert_eq!(FaceDir::NegY.step(), (0, -1));
    }
}
