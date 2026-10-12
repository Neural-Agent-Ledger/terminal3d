//! Seamless-loop proofs for every [`AnimKind`] the shader ships.
//!
//! Every animation SHOULD loop. A time-monotone animation can never be cached, and
//! this engine's near-ring pan and ~30 Hz animation throttle both want "the surface
//! at time `t` is the surface at `t + period`".
//!
//! Two things this module deliberately does NOT do:
//!
//! 1. **It does not probe every kind through [`Material::anim_brightness`].** That
//!    function returns a literal `1.0` for `Shift`/`Zoom`/`Spin`/`SpinZoom` (see its
//!    `_ => 1.0` arm), so a `f(t) == f(t + period)` assertion routed through it would
//!    pass on a CONSTANT and measure nothing. Each kind is instead probed through the
//!    function that actually carries its motion — see [`motion_signal`] — and
//!    `anim_brightness_is_a_no_op_for_the_coord_and_phase_kinds` pins the no-op so a
//!    future refactor cannot silently make the routing wrong.
//! 2. **It does not assume the period.** Every period below is read off the
//!    implementation, not off a doc comment. In particular the three log-polar kinds
//!    loop `mod LN2` / `mod TAU` in their *phase variable*, but their phase is fed
//!    `speed · time · LN2` (resp. `· TAU`) — so in TIME they loop every `1/speed`,
//!    not every `LN2/speed`. Derivation is in [`loop_period`].
//!
//! Three of the ten shipped kinds have NO finite period, and a fourth ([`AnimKind::None`])
//! does not move at all. For `Shift` (an unbounded coord scroll) and `Flicker` (a hash of
//! a monotone tick index) that is a pre-existing property, not a regression; for `Tear` it
//! is a deliberate design choice with the same shape as `Flicker`'s. All three are pinned
//! by their own tests below so a later change cannot quietly widen or narrow the claim.
//! Per the plan's amended Global Constraint, looping is a LOOK requirement that binds only
//! where the signal has a visible SEAM — a hash and an unbounded scroll have none, and
//! forcing a wrap onto either would CREATE structure the unwrapped form does not have.

use super::*;

// ───────────────────────────── probe tunables ───────────────────────────────

/// Probe speed. Deliberately ≠ 1 so a period of `1/speed` cannot be confused with
/// a period of `speed`, `LN2/speed`, or a bare `1.0`.
const PROBE_SPEED: f32 = 0.5;
/// Probe amplitude. Deliberately ≠ 1 so `Shift`'s `speed · amplitude` scroll rate
/// cannot be confused with a bare `speed` rate.
const PROBE_AMPLITUDE: f32 = 0.3;
/// An off-lattice sample coordinate — avoids brick/circuit seams and integer
/// hash-cell boundaries, where a texel is one float wobble from flipping.
const PROBE_COORD: Vec2 = Vec2::new(0.37, 0.61);
/// How many samples span one probe window.
const PROBE_STEPS: usize = 32;
/// A signal must move at least this much across its window to count as "moving".
/// Well above float noise, well below every kind's real excursion.
const MOTION_EPS: f32 = 1.0e-2;
/// Two samples one period apart must agree to within this.
const REPEAT_EPS: f32 = 1.0e-4;
/// `anim_brightness`'s own `speed.max(0.01)` floor, mirrored so [`hold_period`]
/// reports the tick length the implementation actually uses.
const FLICKER_MIN_SPEED: f32 = 0.01;
/// How many ticks apart the generic `PROBE_STEPS` samples must sit for a sparse
/// stepped kind ([`AnimKind::Tear`]). See [`probe_window`] for the measurement that
/// forced it above 1.
const TEAR_PROBE_TICK_STRIDE: f32 = 32.0;

// ─────────────────────────────── the contract ───────────────────────────────

/// The period after which `a` repeats EXACTLY. `None` = no finite period.
///
/// `None` covers three distinct cases, all of which mean "do not assert
/// `f(t) == f(t + p)`":
/// * [`AnimKind::None`] / non-positive `speed` — nothing moves at all.
/// * [`AnimKind::Shift`] — an UNBOUNDED coord scroll (see below).
/// * [`AnimKind::Flicker`] and [`AnimKind::Tear`] — aperiodic by construction, both
///   hashing a monotone tick index (see below).
///
/// Derivations, each read off `ascii_shader.rs` rather than off a doc comment:
///
/// * **`Pulse`** — `anim_brightness`: `phase = TAU · speed · time`, value
///   `1 + amplitude · cos(phase)`. `cos` has period `TAU`, so `time` repeats every
///   `1/speed`. Amplitude does not enter the period.
/// * **`Scan`** — `anim_brightness`: `sweep = (speed · time).rem_euclid(1)`, and the
///   lifted band is a function of the WRAPPED distance `min(d, 1−d)` between `sweep` and
///   the cell's `u`. The fold is unit-periodic in its argument, so `time` repeats every
///   `1/speed`; the wrap means the band crosses the `sweep = 0` seam without a jump,
///   so the loop is seamless in the LOOK sense too, not merely in value.
/// * **`Breathe`** — `anim_brightness`: `p = (speed · time).rem_euclid(1)`, envelope
///   `0.5 − 0.5·cos(π·p/IN)` on `[0, IN)` and `0.5 + 0.5·cos(π·(p−IN)/(1−IN))` on
///   `[IN, 1)`. Same unit fold ⇒ period `1/speed`. The two segments meet at `env = 1`
///   at `p = IN` and both endpoints `p → 0⁺` / `p → 1⁻` give `env = 0`, so the
///   asymmetric envelope is CONTINUOUS across the wrap — an asymmetric shape that is
///   nevertheless seam-free, which is the whole point of the kind.
///
///   Both folds are `rem_euclid`, NOT the sign-preserving `fract`: see
///   `scan_stays_a_narrow_travelling_band_for_negative_speed_and_negative_coords` and
///   `breathe_stays_a_rise_then_fall_envelope_for_negative_speed_and_negative_time` for
///   what each kind does when a negative value reaches the fold.
/// * **`Zoom`** — `fractal_phase`: `t = time · speed`, `zoom = (t · LN2) mod LN2`.
///   Advancing `time` by `1/speed` advances the argument by exactly one `LN2`, so
///   the reduced phase is unchanged ⇒ period `1/speed`. The `mod LN2` in the doc
///   comment is the wrap of the PHASE VARIABLE (one zoom-octave), not of time; the
///   time period is `1/speed`, NOT `LN2/speed`.
/// * **`Spin`** — same shape with `TAU`: `spin = (t · TAU) mod TAU` ⇒ `1/speed`.
/// * **`SpinZoom`** — `fractal_phase` delegates to `fractal::spinzoom_phase(t · LN2,
///   twist)`, which reduces `z = (t · LN2) mod LN2` and derives
///   `spin = (z/LN2)·TAU − twist·z` — a pure function of `z`. So BOTH axes repeat
///   with `z`, i.e. every `1/speed`, and the two seams coincide (that is the point
///   of the coupling; `fractal.rs` proves the field is continuous across it).
/// * **`Shift`** — `anim_shift_coord`: `coord.y + speed · time · amplitude`, with no
///   wrap anywhere in the sampling path. The offset grows without bound, and the
///   recipes it feeds (fBm/Perlin, and a circuit lattice at a non-integer `scale`)
///   are not unit-periodic in `v`. There is therefore no `p` with
///   `f(t) == f(t + p)`. Pinned by `shift_scroll_is_monotone_and_never_repeats`.
/// * **`Flicker`** — `anim_brightness`: `hash(seed, floor(speed · time), salt)`. The
///   tick index is monotone and the hash is not periodic in it, so no `p` exists.
///   What Flicker DOES guarantee is a piecewise-constant HOLD — see [`hold_period`].
///   Pinned by `flicker_holds_within_a_tick_and_never_repeats`.
/// * **`Drift`** — `anim_shift_coord`: `coord + amplitude · (sin p, sin(RATIO · p))`
///   with `p = TAU · speed · time` and a [`DRIFT_AXIS_RATIO`] of `2`. Advancing `time` by
///   `1/speed` advances `p` by exactly `TAU`, and the second axis by `RATIO · TAU` — a
///   whole number of turns on BOTH axes, because the ratio is an integer. So the
///   Lissajous path closes ⇒ period `1/speed`. Note this is a genuinely different
///   derivation from `Shift`'s: `Drift` is bounded and returns, `Shift` is not and does
///   not, even though both live in `anim_shift_coord`. MEASURED, not assumed — see
///   `drift_closes_its_path_and_stays_bounded` and the repeat proof it joins.
/// * **`Tear`** — `anim_shift_coord`: `hash(seed, floor(speed · time · TICKS), row)`
///   thresholded at [`TEAR_DUTY`]. Same shape as `Flicker`: the tick index is monotone
///   and the hash is not periodic in it, so no `p` exists. This is a DESIGN choice, not
///   an oversight — wrapping the index would replay the identical burst sequence every
///   `1/speed`, and a glitch on a fixed rhythm reads as a mechanism rather than as
///   damage; there is no seam in a hash for the wrap to hide, so it would only add
///   structure. Like `Flicker` it offers a HOLD instead — see [`hold_period`]. Pinned
///   by `tear_bursts_briefly_holds_within_a_tick_and_never_repeats`.
///
/// Later tasks extend the match: a new kind that phases as `speed · t` returns
/// `Some(1.0 / a.speed)` and joins the `PERIODIC` list in the repeat test.
pub(crate) fn loop_period(a: &Animation) -> Option<f32> {
    if a.speed <= 0.0 {
        return None;
    }
    match a.kind {
        // Everything that phases as `speed · t` — the extension point: a new kind
        // whose phase advances linearly in `speed · time` joins this arm.
        AnimKind::Pulse
        | AnimKind::Scan
        | AnimKind::Breathe
        | AnimKind::Drift
        | AnimKind::Zoom
        | AnimKind::Spin
        | AnimKind::SpinZoom => Some(1.0 / a.speed),
        // No finite period: `None` is static; `Shift` is an unbounded scroll; `Flicker`
        // and `Tear` are hashes of a monotone tick index. See the derivations above —
        // an aperiodic kind here is CORRECT, not a defect to engineer around.
        AnimKind::None | AnimKind::Shift | AnimKind::Flicker | AnimKind::Tear => None,
    }
}

/// The interval over which a kind's output is CONSTANT (a piecewise hold), for kinds
/// that step rather than loop. `None` = the kind is continuous in time.
///
/// Two kinds hold, and both are the aperiodic-by-construction ones:
/// * [`AnimKind::Flicker`] — `anim_brightness` quantizes time into
///   `floor(speed.max(0.01) · time)` ticks, so the value is frozen for `1/speed`
///   seconds at a stretch.
/// * [`AnimKind::Tear`] — `anim_shift_coord` quantizes into
///   `floor(speed · time · TICKS)` ticks (see [`TEAR_TICKS_PER_UNIT`]), so the displacement is
///   frozen for `1/(speed · TICKS)` seconds. That hold is what makes a burst a burst:
///   a tick is either torn or not, never half-torn.
///
/// That hold — not a loop — is the cache-relevant property a stepped animation offers.
pub(crate) fn hold_period(a: &Animation) -> Option<f32> {
    if a.speed <= 0.0 {
        return None;
    }
    match a.kind {
        AnimKind::Flicker => Some(1.0 / a.speed.max(FLICKER_MIN_SPEED)),
        AnimKind::Tear => Some(1.0 / (a.speed * TEAR_TICKS_PER_UNIT)),
        _ => None,
    }
}

// ──────────────────────────────── probe rig ─────────────────────────────────

/// The minimal `SampleCtx` the crate's other unit tests use — copied verbatim from
/// `ascii_shader::tests::ctx(2.0, 1.0)` (that helper is private to a sibling test
/// module), with only `time` parameterised. `far_freeze = ∞` keeps the far-freeze
/// path out of the way, so `time` really does reach the animation.
fn probe_ctx(time: f32) -> SampleCtx {
    SampleCtx {
        distance: 2.0,
        foreshorten: 1.0,
        side: 0,
        screen_y: 4,
        wall_v: 0.5,
        tick: 0,
        time,
        fog: FogParams::DEFAULT,
        far_freeze: f32::INFINITY,
        light: None,
        lod_override: None,
        state: 0,
        pulse: 0.0,
        inset_rgb: None,
        wall_along: 0.0,
        wall_up: 0.0,
        on_wall: false,
        tex_footprint: None,
        subcell_backdrop: None,
    }
}

/// The same probe with the SHIPPED lighting environment attached.
///
/// FINDING (pre-existing, surfaced by Task 2): the light-LESS branch of
/// `Material::sample` (`ascii_shader.rs`, the `else` of `if let Some(lc) = ctx.light`)
/// applies the animation through `dim_rgb`, which **clamps its factor into `[0,1]`**.
/// That branch can therefore only ever DARKEN: an `anim_brightness` above `1.0` is
/// silently clipped to `1.0` there. `Pulse` survives a light-less texel probe only
/// because it also swings BELOW 1.0; the brighten-ONLY kinds ([`AnimKind::Scan`] and
/// [`AnimKind::Breathe`], both of which ride in `[1, 1+amplitude]`) are invisible in
/// that branch, so a texel-level assertion routed through it would be vacuous for
/// them. The SHIPPED path does not clamp — `apply_relief_lighting` computes
/// `albedo = unpack_rgb(out.fg) * anim_b` un-clamped — so the texel-level seam checks
/// use THIS ctx, i.e. the path the game actually renders through.
fn probe_ctx_lit(time: f32) -> SampleCtx {
    let view = Vec3::new(0.0, -1.0, 0.0);
    SampleCtx {
        light: Some(LightCtx::for_wall(0, -1.0, view, &LightEnv::builtin())),
        ..probe_ctx(time)
    }
}

/// Every kind the shader ships, in enum order. A new variant must be added here or
/// `all_shipped_kinds_are_covered` fails.
const SHIPPED_KINDS: [AnimKind; 10] = [
    AnimKind::Pulse,
    AnimKind::Shift,
    AnimKind::Flicker,
    AnimKind::Scan,
    AnimKind::Breathe,
    AnimKind::Drift,
    AnimKind::Tear,
    AnimKind::Zoom,
    AnimKind::Spin,
    AnimKind::SpinZoom,
];

/// The kinds MEASURED to carry a finite loop period. Membership is a finding, not a
/// preference: a kind belongs here only if `f(t) == f(t + p)` is demonstrated for it
/// (and `f(t) != f(t + p/2)`, so the "loop" cannot be a constant). Four shipped kinds
/// are deliberately absent — see [`loop_period`]'s `None` arm.
const PERIODIC_KINDS: [AnimKind; 7] = [
    AnimKind::Pulse,
    AnimKind::Scan,
    AnimKind::Breathe,
    AnimKind::Drift,
    AnimKind::Zoom,
    AnimKind::Spin,
    AnimKind::SpinZoom,
];

/// A material carrying `kind` at the probe tunables. The three log-polar kinds only
/// mean anything over a `Fractal` recipe (they drive its field phase), so they get
/// one — with the shipped `portal_spiral` arms/warp/twist, so the probe exercises
/// the coupling that actually ships.
fn probe_material(kind: AnimKind) -> Material {
    let animation = Animation {
        kind,
        speed: PROBE_SPEED,
        amplitude: PROBE_AMPLITUDE,
    };
    match kind {
        AnimKind::Zoom | AnimKind::Spin | AnimKind::SpinZoom => Material {
            recipe: TextureRecipe::Fractal {
                arms: 5,
                octave_warp: 0.30,
                twist: 0.9,
            },
            animation,
            ..DEFAULT_MATERIAL.clone()
        },
        _ => Material {
            animation,
            ..DEFAULT_MATERIAL.clone()
        },
    }
}

/// The twist the real call site threads into `fractal_phase` (it comes from the
/// recipe, not the animation) — mirrored here so the probe matches production.
fn probe_twist(m: &Material) -> f32 {
    match m.recipe {
        TextureRecipe::Fractal { twist, .. } => twist,
        _ => 0.0,
    }
}

/// The scalar that actually MOVES for `mat`'s kind, and the function it lives in:
///
/// | kind | carrier |
/// |---|---|
/// | `Pulse`, `Flicker`, `Scan`, `Breathe` | `Material::anim_brightness` |
/// | `Shift`, `Drift`, `Tear` | `Material::anim_shift_coord` (the moved coord) |
/// | `Zoom`, `Spin`, `SpinZoom` | `Material::fractal_phase` (the log-polar phase) |
///
/// For the log-polar kinds the two phase axes are independent (`Zoom` moves only
/// `zoom`, `Spin` only `spin`), so their SUM is a faithful motion detector; the
/// repeat test compares the axes separately. `Drift` moves BOTH coord axes, so it is
/// reduced by a weighted sum — the weight is irrational-ish rather than `1`, so the
/// two axes cannot cancel to a constant and make a repeat claim vacuous (`u + v`
/// would hide an implementation that drove `v = −u`).
const DRIFT_AXIS_WEIGHT: f32 = 1.7071;

fn motion_signal(mat: &Material, coord: Vec2, time: f32) -> f32 {
    let ctx = probe_ctx(time);
    match mat.animation.kind {
        AnimKind::None => 1.0,
        AnimKind::Pulse | AnimKind::Flicker | AnimKind::Scan | AnimKind::Breathe => {
            mat.anim_brightness(coord, &ctx)
        }
        AnimKind::Shift => mat.anim_shift_coord(coord, &ctx).y,
        // `Tear` displaces along `u` only, so `u` alone is its faithful carrier.
        AnimKind::Tear => mat.anim_shift_coord(coord, &ctx).x,
        AnimKind::Drift => {
            let c = mat.anim_shift_coord(coord, &ctx);
            c.x + DRIFT_AXIS_WEIGHT * c.y
        }
        AnimKind::Zoom | AnimKind::Spin | AnimKind::SpinZoom => {
            let p = mat.fractal_phase(&ctx, probe_twist(mat));
            p.zoom + p.spin
        }
    }
}

/// The time window over which a kind's motion is worth measuring: one loop for a
/// periodic kind, one full texture unit of scroll for `Shift`, and a long run of
/// ticks for `Flicker` (whose excursion only shows across tick boundaries).
fn probe_window(a: &Animation) -> f32 {
    if let Some(p) = loop_period(a) {
        return p;
    }
    match a.kind {
        // One whole texture unit of scroll — the only candidate period `Shift` has.
        AnimKind::Shift => 1.0 / (a.speed * a.amplitude),
        // Enough ticks that the hashed values spread out.
        AnimKind::Flicker => PROBE_STEPS as f32 * hold_period(a).expect("a stepped kind holds"),
        // `Tear` fires on only `TEAR_DUTY` of its ticks, and MEASUREMENT found quiet
        // runs of up to 38 CONSECUTIVE ticks (and, at the probe row, ticks 0..15 all
        // quiet). `PROBE_STEPS` adjacent ticks can therefore be entirely silent, which
        // would make `every_animation_kind_actually_moves` fail on a correct `Tear`.
        // Striding the same `PROBE_STEPS` samples across `TEAR_PROBE_TICK_STRIDE` times
        // as many ticks decorrelates them: the samples land in different quiet runs.
        AnimKind::Tear => {
            PROBE_STEPS as f32
                * TEAR_PROBE_TICK_STRIDE
                * hold_period(a).expect("a stepped kind holds")
        }
        _ => 1.0,
    }
}

/// `PROBE_STEPS` evenly spaced sample times covering `[0, window)`.
fn probe_times(window: f32) -> impl Iterator<Item = f32> {
    (0..PROBE_STEPS).map(move |i| window * i as f32 / PROBE_STEPS as f32)
}

// ─────────────────────────────────── tests ──────────────────────────────────

#[test]
fn all_shipped_kinds_are_covered() {
    // A new `AnimKind` variant must be added to `SHIPPED_KINDS` (and classified in
    // `loop_period`) or every proof below silently stops covering it. The match is
    // exhaustive, so adding a variant to the enum breaks THIS test at compile time.
    for kind in SHIPPED_KINDS {
        let covered = match kind {
            AnimKind::None => false,
            AnimKind::Pulse
            | AnimKind::Shift
            | AnimKind::Flicker
            | AnimKind::Scan
            | AnimKind::Breathe
            | AnimKind::Drift
            | AnimKind::Tear
            | AnimKind::Zoom
            | AnimKind::Spin
            | AnimKind::SpinZoom => true,
        };
        assert!(covered, "{kind:?} is not classified");
        assert!(
            probe_material(kind).is_animated(),
            "{kind:?} must mark its material animated"
        );
    }
    assert!(
        loop_period(&Animation::NONE).is_none(),
        "the static animation has no period"
    );
    for kind in SHIPPED_KINDS {
        // A non-positive speed is a stopped animation, whatever the kind.
        let stopped = Animation {
            kind,
            speed: 0.0,
            amplitude: PROBE_AMPLITUDE,
        };
        assert!(
            loop_period(&stopped).is_none(),
            "{kind:?} at speed 0 cannot loop"
        );
        assert!(
            hold_period(&stopped).is_none(),
            "{kind:?} at speed 0 cannot hold"
        );
    }
}

/// The anti-vacuity guard. `anim_brightness` is a literal `1.0` for the four kinds
/// whose motion lives elsewhere; probing them through it would assert nothing. If
/// this test ever fails, one of those kinds grew a brightness channel and
/// [`motion_signal`] must be re-routed for it.
#[test]
fn anim_brightness_is_a_no_op_for_the_coord_and_phase_kinds() {
    for kind in [
        AnimKind::Shift,
        AnimKind::Drift,
        AnimKind::Tear,
        AnimKind::Zoom,
        AnimKind::Spin,
        AnimKind::SpinZoom,
    ] {
        let mat = probe_material(kind);
        for t in probe_times(37.0) {
            assert_eq!(
                mat.anim_brightness(PROBE_COORD, &probe_ctx(t)),
                1.0,
                "{kind:?} must not touch brightness (its motion is elsewhere)"
            );
        }
    }
}

/// Every shipped kind must actually MOVE the quantity we probe it through — proven
/// BEFORE any "it repeats" claim, so no kind can pass a loop test on a constant.
#[test]
fn every_animation_kind_actually_moves() {
    for kind in SHIPPED_KINDS {
        let mat = probe_material(kind);
        let window = probe_window(&mat.animation);
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for t in probe_times(window) {
            let v = motion_signal(&mat, PROBE_COORD, t);
            lo = lo.min(v);
            hi = hi.max(v);
        }
        assert!(
            hi - lo > MOTION_EPS,
            "{kind:?} does not move: signal range {} over a {window}s window — the \
             probe is measuring a constant, so any repeat claim about it is vacuous",
            hi - lo
        );
    }
}

/// The core invariant: a periodic kind's motion signal AND its fully shaded texel are
/// identical one period later, and NOT identical half a period later (so the period
/// asserted is a real loop, not a constant).
#[test]
fn periodic_animations_repeat_after_their_loop_period() {
    let perlin = Perlin::new(SHADER_SEED);
    for kind in PERIODIC_KINDS {
        let mat = probe_material(kind);
        let period = loop_period(&mat.animation).expect("a periodic kind has a period");
        assert_eq!(
            period,
            1.0 / PROBE_SPEED,
            "{kind:?}: the derived period is 1/speed (see loop_period's derivation)"
        );
        let mut half_period_differs = false;
        let mut texel_half_period_differs = false;
        for t in probe_times(period) {
            let a = motion_signal(&mat, PROBE_COORD, t);
            let b = motion_signal(&mat, PROBE_COORD, t + period);
            assert!(
                (a - b).abs() < REPEAT_EPS,
                "{kind:?} does not repeat: f({t}) = {a}, f({}) = {b}",
                t + period
            );
            if (a - motion_signal(&mat, PROBE_COORD, t + 0.5 * period)).abs() > MOTION_EPS {
                half_period_differs = true;
            }
            // The whole shaded texel — glyph, fg, bg, rank — must come back too.
            // This is what "no seam" means to a viewer. Probed through the LIT ctx:
            // see `probe_ctx_lit` for why the light-less branch cannot see a
            // brighten-only kind.
            let here = mat.sample(&perlin, PROBE_COORD, &probe_ctx_lit(t));
            assert_eq!(
                here,
                mat.sample(&perlin, PROBE_COORD, &probe_ctx_lit(t + period)),
                "{kind:?}: the shaded texel differs across the loop seam at t={t}"
            );
            if here != mat.sample(&perlin, PROBE_COORD, &probe_ctx_lit(t + 0.5 * period)) {
                texel_half_period_differs = true;
            }
        }
        assert!(
            half_period_differs,
            "{kind:?} is the same at half a period — the 'loop' is a constant, not a loop"
        );
        assert!(
            texel_half_period_differs,
            "{kind:?}: the shaded TEXEL never changes within a period — the texel-level \
             seam check above is vacuous, so the animation is not reaching the output"
        );
    }
}

/// FINDING (pre-existing, not a regression): `Shift` has no finite period.
///
/// `anim_shift_coord` adds `speed · time · amplitude` to `coord.y` with no wrap, and
/// nothing downstream folds it back — so the sampled coordinate is strictly monotone
/// and unbounded. Even the one candidate period (`1/(speed·amplitude)`, one whole
/// texture unit of scroll) does not reproduce the texel, because the recipes are not
/// unit-periodic in `v` (fBm/Perlin, plus a circuit lattice at a non-integer scale).
/// Probed on the SHIPPED `wall_matrix` material, not a synthetic one.
#[test]
fn shift_scroll_is_monotone_and_never_repeats() {
    let mat = wall_matrix();
    assert_eq!(
        mat.animation.kind,
        AnimKind::Shift,
        "wall_matrix is the shipped Shift wall"
    );
    let (speed, amp) = (mat.animation.speed, mat.animation.amplitude);
    assert!(
        loop_period(&mat.animation).is_none(),
        "Shift must be classified aperiodic"
    );

    // 1. The offset is exactly `speed · time · amplitude`, and strictly increasing.
    let unit_period = 1.0 / (speed * amp);
    let mut prev = f32::NEG_INFINITY;
    for t in probe_times(unit_period) {
        let got = mat.anim_shift_coord(PROBE_COORD, &probe_ctx(t));
        assert_eq!(got.x, PROBE_COORD.x, "Shift scrolls v only");
        assert!(
            (got.y - (PROBE_COORD.y + speed * t * amp)).abs() < 1.0e-6,
            "Shift offset at t={t} is not speed·time·amplitude"
        );
        assert!(got.y > prev, "the Shift scroll must be strictly monotone");
        prev = got.y;
    }
    // 2. One full texture unit later the coord has advanced exactly 1.0 …
    let at0 = mat.anim_shift_coord(PROBE_COORD, &probe_ctx(0.0));
    let at1 = mat.anim_shift_coord(PROBE_COORD, &probe_ctx(unit_period));
    assert!((at1.y - at0.y - 1.0).abs() < 1.0e-5);
    // 3. … but the shaded texel does NOT come back, so that is not a period.
    let perlin = Perlin::new(SHADER_SEED);
    let mut any_differs = false;
    for t in probe_times(unit_period) {
        if mat.sample(&perlin, PROBE_COORD, &probe_ctx(t))
            != mat.sample(&perlin, PROBE_COORD, &probe_ctx(t + unit_period))
        {
            any_differs = true;
        }
    }
    assert!(
        any_differs,
        "if the Shift texel DID reproduce after one texture unit, Shift would be \
         periodic and loop_period must be updated to say so"
    );
}

/// FINDING (pre-existing, not a regression): `Flicker` has no period — it has a HOLD.
///
/// `anim_brightness` hashes `floor(speed · time)`, a monotone tick index, so the
/// value never returns. What it does guarantee is that the output is CONSTANT for a
/// whole tick and stays inside `[1 − amplitude, 1]`, which is the property a cache
/// (and the ~30 Hz throttle) can actually use.
#[test]
fn flicker_holds_within_a_tick_and_never_repeats() {
    let mat = probe_material(AnimKind::Flicker);
    let anim = mat.animation;
    assert!(
        loop_period(&anim).is_none(),
        "Flicker must be classified aperiodic"
    );
    let hold = hold_period(&anim).expect("Flicker holds for a tick");
    assert_eq!(hold, 1.0 / PROBE_SPEED);

    // 1. Constant within a tick: sample strictly inside tick 3 and compare.
    let inside_tick = |k: u32, frac: f32| (k as f32 + frac) * hold;
    let base = motion_signal(&mat, PROBE_COORD, inside_tick(3, 0.02));
    for frac in [0.17_f32, 0.5, 0.83, 0.98] {
        assert_eq!(
            motion_signal(&mat, PROBE_COORD, inside_tick(3, frac)),
            base,
            "Flicker must hold its value for the whole tick"
        );
    }
    // 2. Bounded — a defined output at every time, never out of range.
    let mut distinct = 0;
    let mut prev: Option<f32> = None;
    for k in 0..PROBE_STEPS as u32 {
        let v = motion_signal(&mat, PROBE_COORD, inside_tick(k, 0.5));
        assert!(
            (1.0 - anim.amplitude - 1.0e-6..=1.0 + 1.0e-6).contains(&v),
            "Flicker brightness {v} left [1-amplitude, 1] at tick {k}"
        );
        if prev != Some(v) {
            distinct += 1;
        }
        prev = Some(v);
    }
    assert!(
        distinct > PROBE_STEPS / 2,
        "Flicker must actually jitter across ticks"
    );
    // 3. It does NOT come back after one hold — that is the aperiodicity.
    let mut any_differs = false;
    for k in 0..PROBE_STEPS as u32 {
        let a = motion_signal(&mat, PROBE_COORD, inside_tick(k, 0.5));
        let b = motion_signal(&mat, PROBE_COORD, inside_tick(k, 0.5) + hold);
        if (a - b).abs() > REPEAT_EPS {
            any_differs = true;
        }
    }
    assert!(
        any_differs,
        "if Flicker DID repeat after one tick it would be periodic and loop_period \
         must be updated to say so"
    );
}

/// Defined output for the degenerate inputs: a stopped animation, a zero-amplitude
/// one, and a negative speed must all still produce a finite, in-range brightness
/// and an unmoved coord — never a NaN and never a panic.
#[test]
fn degenerate_animations_are_defined_and_static() {
    for kind in SHIPPED_KINDS {
        for (speed, amplitude) in [(0.0, 0.3), (PROBE_SPEED, 0.0), (-1.0, 0.3)] {
            let mat = Material {
                animation: Animation {
                    kind,
                    speed,
                    amplitude,
                },
                ..DEFAULT_MATERIAL.clone()
            };
            for t in [0.0_f32, 1.0, 97.5] {
                let ctx = probe_ctx(t);
                let b = mat.anim_brightness(PROBE_COORD, &ctx);
                assert!(
                    b.is_finite() && (0.0..=2.0).contains(&b),
                    "{kind:?}: brightness {b}"
                );
                let c = mat.anim_shift_coord(PROBE_COORD, &ctx);
                assert!(c.x.is_finite() && c.y.is_finite(), "{kind:?}: coord {c:?}");
                let p = mat.fractal_phase(&ctx, 0.9);
                assert!(p.zoom.is_finite() && p.spin.is_finite(), "{kind:?}: phase");
            }
        }
    }
}

// ───────────────────── Scan / Breathe behaviour (Task 2) ─────────────────────

/// How wide a `Scan` band is allowed to read, as a fraction of the surface. Mirrors
/// `anim_brightness`'s `BAND` only in spirit — the test measures the SHIPPED value,
/// it does not import it, so widening `BAND` in the shader fails this test.
const SCAN_PROBE_CELLS: usize = 40;
/// The brightness above which a `Scan` cell counts as "lifted" for the band-width
/// measurement. Comfortably above the at-rest `1.0`, comfortably below the peak.
const SCAN_LIT_THRESHOLD: f32 = 1.05;

/// A `Scan` material at the given `speed` (amplitude fixed at a strong 0.8 so the
/// lit/at-rest split is unambiguous).
fn scan_material(speed: f32) -> Material {
    Material {
        animation: Animation {
            kind: AnimKind::Scan,
            speed,
            amplitude: 0.8,
        },
        ..DEFAULT_MATERIAL.clone()
    }
}

/// Which of `SCAN_PROBE_CELLS` cells across one unit of `u` are lifted at time `t`.
/// `u_origin` shifts the probed span along `u` so a NEGATIVE world coord can be
/// probed by the same rig as a positive one.
fn scan_lit_cells(mat: &Material, t: f32, u_origin: f32) -> Vec<usize> {
    (0..SCAN_PROBE_CELLS)
        .filter(|i| {
            let u = u_origin + *i as f32 / SCAN_PROBE_CELLS as f32;
            mat.anim_brightness(Vec2::new(u, 0.5), &probe_ctx(t)) > SCAN_LIT_THRESHOLD
        })
        .collect()
}

/// `Scan` is a bright band crossing the surface: at any instant most of the
/// surface is at rest and a narrow window is lifted. A `Scan` that lifted half the
/// wall would be a `Pulse` with extra steps.
#[test]
fn scan_lifts_a_narrow_band_and_leaves_the_rest_at_rest() {
    let mat = scan_material(1.0);
    let t = 0.25;
    let lit = scan_lit_cells(&mat, t, 0.0);
    assert!(
        (1..=12).contains(&lit.len()),
        "expected a narrow band, {}/{SCAN_PROBE_CELLS} cells lifted",
        lit.len()
    );
    // Anti-vacuity: the band must be SOMEWHERE ELSE at another instant, or "narrow"
    // would be satisfied by a band that never moves (a static stripe).
    assert_ne!(
        lit,
        scan_lit_cells(&mat, t + 0.5, 0.0),
        "the Scan band must travel, not sit still"
    );
}

/// A NEGATIVE `speed`, and a NEGATIVE world coord, must each still give a narrow
/// travelling band — a negative speed sweeping the other way is the obvious reading,
/// and it is the one an author will try once a `.lvl` can name this kind.
///
/// This is why `anim_brightness`'s `Scan` arm uses `rem_euclid(1.0)` and not the
/// plan's literal `fract()`. `f32::fract` is SIGN-PRESERVING, so either negative
/// input lands the value in `(-1,0]`; the wrapped distance `min(d, 1−d)` then goes
/// NEGATIVE, `k = 1 − d/BAND` exceeds 1, and the arm lights a wide region (clamped at
/// 2.0) instead of a band. Folding the sweep alone is NOT enough — `here` reaches the
/// same subtraction by the same route — so both are folded.
#[test]
fn scan_stays_a_narrow_travelling_band_for_negative_speed_and_negative_coords() {
    for (speed, u_origin) in [(-1.0_f32, 0.0_f32), (1.0, -3.0), (-1.0, -3.0)] {
        let mat = scan_material(speed);
        // Narrow at EVERY instant, not at one lucky one: under the unhardened
        // `fract` the band is narrow at some phases and 31/40 wide at others, so a
        // single-time check would pass by accident.
        for t in [0.0_f32, 0.25, 0.5, 0.75, 0.9] {
            let lit = scan_lit_cells(&mat, t, u_origin);
            assert!(
                (1..=12).contains(&lit.len()),
                "speed {speed}, u origin {u_origin}, t {t}: expected a narrow band, \
                 {}/{SCAN_PROBE_CELLS} cells lifted",
                lit.len()
            );
        }
        assert_ne!(
            scan_lit_cells(&mat, 0.25, u_origin),
            scan_lit_cells(&mat, 0.75, u_origin),
            "speed {speed}, u origin {u_origin}: the band must still travel"
        );
    }
    // The negative speed sweeps the OTHER WAY: at the same instant its band sits
    // where the positive-speed band will be at the mirrored time, not where the
    // positive one is now.
    let (fwd, back) = (scan_material(1.0), scan_material(-1.0));
    assert_ne!(
        scan_lit_cells(&fwd, 0.25, 0.0),
        scan_lit_cells(&back, 0.25, 0.0),
        "a negative speed must reverse the sweep, not reproduce it"
    );
    assert_eq!(
        scan_lit_cells(&fwd, 0.25, 0.0),
        scan_lit_cells(&back, 0.75, 0.0),
        "reversing the speed must mirror the sweep in time"
    );
}

/// `Breathe` is asymmetric — slow in, fast out. Its rise occupies more of the
/// period than its fall, which is what makes it read as a sigh, not a sine.
#[test]
fn breathe_rises_slower_than_it_falls() {
    let mat = Material {
        animation: Animation {
            kind: AnimKind::Breathe,
            speed: 1.0,
            amplitude: 0.5,
        },
        ..DEFAULT_MATERIAL.clone()
    };
    let coord = Vec2::new(0.5, 0.5);
    let samples: Vec<f32> = (0..100)
        .map(|i| mat.anim_brightness(coord, &probe_ctx(i as f32 / 100.0)))
        .collect();
    let peak = samples
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    assert!(
        peak > 55,
        "the peak should sit late in the cycle, got index {peak}"
    );
    // The envelope must return to rest at the wrap, or the "loop" would be a value
    // jump the eye reads as a cut. (`periodic_animations_repeat_after_their_loop_period`
    // proves f(t) == f(t+p); this proves the SHAPE closes, i.e. the ends meet at rest.)
    assert!(
        (samples[0] - 1.0).abs() < 1.0e-5,
        "Breathe must start at rest, got {}",
        samples[0]
    );
    assert!(
        (samples[99] - 1.0).abs() < 0.02,
        "Breathe must come back to rest by the wrap, got {}",
        samples[99]
    );
}

/// FINDING (pre-existing, surfaced while adding `Scan`): a brighten-ONLY animation
/// reaches the output on the shipped lit path and is CLIPPED on the light-less
/// fallback, because that fallback applies the factor through `dim_rgb`, which clamps
/// it into `[0,1]`. Pinned here so (a) `probe_ctx_lit`'s existence is justified by a
/// test rather than by a comment, and (b) if someone ever makes the fallback brighten
/// too, this test fails and the harness can go back to the simpler ctx.
#[test]
fn a_brighten_only_animation_is_clipped_by_the_light_less_fallback() {
    let perlin = Perlin::new(SHADER_SEED);
    let mat = probe_material(AnimKind::Scan);
    // `t` chosen so the band sits on PROBE_COORD: sweep = (speed·t).fract() = 0.37.
    let t = 0.37 / PROBE_SPEED;
    let b = mat.anim_brightness(PROBE_COORD, &probe_ctx(t));
    assert!(
        b > 1.0,
        "the probe time must put the Scan band on the probe coord, got {b}"
    );

    let rest = 0.5 / PROBE_SPEED; // sweep = 0.5, half a surface away ⇒ at rest
    assert_eq!(mat.anim_brightness(PROBE_COORD, &probe_ctx(rest)), 1.0);

    assert_eq!(
        mat.sample(&perlin, PROBE_COORD, &probe_ctx(t)),
        mat.sample(&perlin, PROBE_COORD, &probe_ctx(rest)),
        "the light-less fallback clamps the anim factor to 1.0, so a brighten-only \
         kind cannot change its texel there"
    );
    assert_ne!(
        mat.sample(&perlin, PROBE_COORD, &probe_ctx_lit(t)),
        mat.sample(&perlin, PROBE_COORD, &probe_ctx_lit(rest)),
        "the SHIPPED lit path multiplies the albedo by the anim factor un-clamped, so \
         Scan must be visible there — if it is not, Scan renders as a static surface"
    );
}

/// A new enum variant must not change ANY existing material's output. The direct
/// proof: no shipped material — built-in or catalog — carries one of the four
/// newer kinds, so every existing material takes exactly the arm it took
/// before. (Adoption is a later step; when it happens, this test is the thing
/// that must be updated deliberately, and the goldens re-blessed with it.)
#[test]
fn no_shipped_material_adopts_the_new_kinds_yet() {
    let registry = MaterialRegistry::with_catalog();
    for i in 0..registry.len() {
        let kind = registry.get(MaterialId(i as u16)).animation.kind;
        assert!(
            !matches!(
                kind,
                AnimKind::Scan | AnimKind::Breathe | AnimKind::Drift | AnimKind::Tear
            ),
            "material slot {i} adopted {kind:?} — goldens must move with it"
        );
    }
    assert!(
        registry.len() > 8,
        "the probe must actually see the catalog"
    );
}

// ────────────────────── Drift / Tear behaviour (Task 3) ───────────────────────

/// How far along the loop the two `Drift` probe times sit. `1/8` and `3/8` of a period
/// put the Lissajous phase at `π/4` and `3π/4`: `sin` agrees on both (same `u`) while
/// `sin(2·)` is `+1` and `−1` (opposite `v`). One `u`, two `v`s — the crispest possible
/// statement that the path is a two-axis FIGURE, not a diagonal line.
const DRIFT_SAME_U_PHASES: (f32, f32) = (0.125, 0.375);
/// How far out in time the boundedness scan runs, in loop periods. Long enough that an
/// accidental unbounded term (a `Shift`-style `speed·time` added to the sine) would have
/// grown far past `amplitude` and be caught.
const DRIFT_BOUND_SCAN_PERIODS: usize = 100;
/// Samples per period in that scan.
const DRIFT_BOUND_SCAN_STEPS: usize = 40;

/// `Drift` is a CLOSED, BOUNDED two-axis path — the three properties that separate it
/// from [`AnimKind::Shift`], which shares its dispatch point but is a one-axis unbounded
/// scroll. `periodic_animations_repeat_after_their_loop_period` proves the closure in
/// value (`f(t) == f(t + p)`, and `!= f(t + p/2)` so it is not a constant); this proves
/// the SHAPE — that the excursion never escapes `amplitude`, that BOTH axes carry
/// motion, and that the two axes are independent rather than locked into a diagonal.
///
/// MEASURED before it was asserted: the offset ranges are exactly `[−0.3, 0.3]` on both
/// axes at `amplitude = 0.3` over a 40 s scan, and the worst `|f(t) − f(t + 1/speed)|`
/// over a 37 s horizon is `8.5e-6` (float rounding), against `0.60` at half a period.
#[test]
fn drift_closes_its_path_and_stays_bounded() {
    let mat = probe_material(AnimKind::Drift);
    let amp = mat.animation.amplitude;
    let period = loop_period(&mat.animation).expect("Drift is periodic");
    let offset = |t: f32| mat.anim_shift_coord(PROBE_COORD, &probe_ctx(t)) - PROBE_COORD;

    // 1. BOTH axes move, and neither is a constant. (A `Shift`-shaped implementation
    //    that only touched `v` would pass a mere "it moves" check; this fails it.)
    let (mut lo, mut hi) = (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY));
    for t in probe_times(period) {
        let d = offset(t);
        lo = lo.min(d);
        hi = hi.max(d);
    }
    assert!(
        hi.x - lo.x > MOTION_EPS && hi.y - lo.y > MOTION_EPS,
        "Drift must move BOTH axes: u range {}, v range {}",
        hi.x - lo.x,
        hi.y - lo.y
    );

    // 2. BOUNDED by `amplitude` on each axis, far past one period — this is the
    //    property `Shift` does not have, and the reason `Drift` never walks its texel
    //    off to a coordinate where float precision has decayed.
    for i in 0..DRIFT_BOUND_SCAN_PERIODS * DRIFT_BOUND_SCAN_STEPS {
        let t = period * i as f32 / DRIFT_BOUND_SCAN_STEPS as f32;
        let d = offset(t);
        assert!(
            d.x.abs() <= amp + 1.0e-5 && d.y.abs() <= amp + 1.0e-5,
            "Drift left its amplitude box at t={t}: {d:?} (amplitude {amp})"
        );
    }

    // 3. The path CLOSES at the origin: zero offset at `t = 0` and again at `t = p`.
    assert!(
        offset(0.0).length() < 1.0e-6,
        "Drift must start un-displaced"
    );
    assert!(
        offset(period).length() < REPEAT_EPS,
        "Drift must return to its origin after one period, got {:?}",
        offset(period)
    );

    // 4. The two axes are INDEPENDENT: one `u`, two different `v`s. A diagonal
    //    `(sin p, sin p)` — the obvious wrong implementation, and the one that would
    //    still pass every check above — makes `v` a function of `u` and fails here.
    let (a, b) = DRIFT_SAME_U_PHASES;
    let (da, db) = (offset(a * period), offset(b * period));
    assert!(
        (da.x - db.x).abs() < 1.0e-5,
        "the probe phases must share a u offset: {} vs {}",
        da.x,
        db.x
    );
    assert!(
        (da.y - db.y).abs() > amp,
        "Drift's axes are locked together — the same u {:.6} gave v {:.6} and {:.6}, so \
         the path is a line, not a two-axis figure",
        da.x,
        da.y,
        db.y
    );
}

/// How many ticks the `Tear` statistics are gathered over. Large enough that the duty
/// cycle is a measurement rather than a coin flip: MEASUREMENT found that a single
/// `1/speed` window (16 ticks) ranges from 0 to 43 samples-moved out of 100 across 64
/// consecutive windows, so the plan's draft "5..=35 over one window" assertion would
/// have failed outright on a correct implementation.
const TEAR_STAT_TICKS: i32 = 8192;
/// Bounds on the measured torn-tick fraction. Measured 0.185 / 0.185 / 0.176 on three
/// rows against a [`TEAR_DUTY`] of 0.18. The band is wide enough to survive a modest
/// retune of the duty knob and narrow enough that "never tears" (0.0) and "always
/// tears" (1.0) both fail.
const TEAR_DUTY_BOUNDS: (f32, f32) = (0.05, 0.40);
/// A burst must stay SHORT. Measured longest torn run: 5 ticks.
const TEAR_MAX_BURST_TICKS: usize = 12;
/// … and there must be MANY of them, i.e. the torn ticks are scattered bursts rather
/// than one long tear. Measured ~1200 distinct bursts per [`TEAR_STAT_TICKS`].
const TEAR_MIN_BURSTS: usize = 200;

/// Whether `mat` tears at row `row` on tick `k`, sampled mid-tick.
fn tear_torn_at(mat: &Material, row: f32, k: i32) -> bool {
    let hold = hold_period(&mat.animation).expect("Tear holds");
    let c = Vec2::new(PROBE_COORD.x, row);
    let t = (k as f32 + 0.5) * hold;
    (mat.anim_shift_coord(c, &probe_ctx(t)) - c).length() > 1.0e-6
}

/// FINDING (measured, not assumed): `Tear` has NO finite period — it has a HOLD, the
/// same contract [`AnimKind::Flicker`] carries, and for the same reason: the tick index
/// it hashes is monotone. See [`AnimKind::Tear`] for why that is the intended LOOK
/// (a glitch that replays on a fixed rhythm reads as a mechanism, not as damage) and
/// the amended Global Constraint in the plan for why a coordinate animation that cannot
/// loop is correct rather than a defect.
///
/// Measured at `speed = 0.5`, `amplitude = 0.3`: torn on 18.5% of ticks, longest torn
/// run 5 ticks, longest quiet run 36–38 ticks, `du ∈ [−0.1492, 0.1486]` (i.e. inside
/// `±amplitude/2`), `dv` identically 0, and the worst `|f(t) − f(t + p)|` is ≈ 0.2 for
/// every candidate period `p ∈ {0.25, 0.5, 1, 2, 4}/speed`.
#[test]
fn tear_bursts_briefly_holds_within_a_tick_and_never_repeats() {
    let mat = probe_material(AnimKind::Tear);
    let anim = mat.animation;
    let amp = anim.amplitude;
    assert!(
        loop_period(&anim).is_none(),
        "Tear must be classified aperiodic"
    );
    let hold = hold_period(&anim).expect("Tear holds for a tick");
    assert_eq!(hold, 1.0 / (PROBE_SPEED * TEAR_TICKS_PER_UNIT));

    // 1. QUIET most of the time, and BURSTING some of the time. Both halves asserted,
    //    so neither a no-op nor a permanent displacement can pass.
    let seq: Vec<bool> = (0..TEAR_STAT_TICKS)
        .map(|k| tear_torn_at(&mat, PROBE_COORD.y, k))
        .collect();
    let torn = seq.iter().filter(|b| **b).count();
    let duty = torn as f32 / TEAR_STAT_TICKS as f32;
    let (duty_lo, duty_hi) = TEAR_DUTY_BOUNDS;
    assert!(
        (duty_lo..=duty_hi).contains(&duty),
        "Tear tore on {torn}/{TEAR_STAT_TICKS} ticks (duty {duty:.4}) — outside \
         [{duty_lo}, {duty_hi}], so it is either silent or no longer quiet"
    );

    // 2. The bursts are SHORT and MANY — scattered damage, not one long smear. A
    //    thresholded smooth signal would give few, long runs and fail here.
    let (mut bursts, mut longest, mut run) = (0usize, 0usize, 0usize);
    for t in &seq {
        run = if *t { run + 1 } else { 0 };
        if run == 1 {
            bursts += 1;
        }
        longest = longest.max(run);
    }
    assert!(
        longest <= TEAR_MAX_BURST_TICKS,
        "a Tear burst ran {longest} ticks — a burst must be brief"
    );
    assert!(
        bursts >= TEAR_MIN_BURSTS,
        "only {bursts} distinct bursts in {TEAR_STAT_TICKS} ticks — the tears have \
         merged into a few long ones instead of scattering"
    );

    // 3. HOLD: the displacement is constant across a whole tick, so a tick is either
    //    torn or not, never half-torn. Probed on a tick that MEASURABLY tears, and the
    //    displacement asserted non-zero first, or "constant" would be satisfied by the
    //    un-torn zero.
    let torn_tick = (0..TEAR_STAT_TICKS)
        .find(|k| tear_torn_at(&mat, PROBE_COORD.y, *k))
        .expect("some tick must tear");
    let at =
        |frac: f32| mat.anim_shift_coord(PROBE_COORD, &probe_ctx((torn_tick as f32 + frac) * hold));
    let base = at(0.02);
    assert!(
        (base - PROBE_COORD).length() > 1.0e-6,
        "the probed tick must actually be torn"
    );
    for frac in [0.17_f32, 0.5, 0.83, 0.98] {
        assert_eq!(
            at(frac),
            base,
            "Tear must hold its displacement for the whole tick"
        );
    }

    // 4. BOUNDED, and HORIZONTAL ONLY — the displacement is a torn line, not a shake.
    for k in 0..TEAR_STAT_TICKS {
        let d =
            mat.anim_shift_coord(PROBE_COORD, &probe_ctx((k as f32 + 0.5) * hold)) - PROBE_COORD;
        assert!(
            d.x.abs() <= 0.5 * amp + 1.0e-6,
            "Tear displaced u by {} at tick {k} — the burst is bounded by amplitude/2",
            d.x
        );
        assert_eq!(
            d.y, 0.0,
            "Tear displaces along u only, got dv {} at tick {k}",
            d.y
        );
    }

    // 5. Rows tear INDEPENDENTLY. Without the per-row salt the whole surface would jump
    //    together, which reads as camera shake rather than as a torn band.
    let split = (0..TEAR_STAT_TICKS).any(|k| {
        let rows: Vec<bool> = (0..8)
            .map(|r| tear_torn_at(&mat, r as f32 + PROBE_COORD.y, k))
            .collect();
        rows.iter().any(|b| *b) && rows.iter().any(|b| !*b)
    });
    assert!(
        split,
        "no tick tore some rows and not others — Tear is displacing the whole surface \
         as one, so the per-row salt is not reaching the hash"
    );

    // 6. It does NOT come back. Checked against every candidate period a wrapped
    //    implementation would produce, over a long horizon — if `Tear` DID repeat, it
    //    would belong in `loop_period`'s periodic arm and this test must be rewritten,
    //    not deleted.
    for mul in [0.25_f32, 0.5, 1.0, 2.0, 4.0] {
        let p = mul / PROBE_SPEED;
        let mut worst = 0.0_f32;
        for k in 0..TEAR_STAT_TICKS {
            let t = (k as f32 + 0.5) * hold;
            let a = mat.anim_shift_coord(PROBE_COORD, &probe_ctx(t));
            let b = mat.anim_shift_coord(PROBE_COORD, &probe_ctx(t + p));
            worst = worst.max((a - b).length());
        }
        assert!(
            worst > MOTION_EPS,
            "Tear reproduced itself after {mul}/speed = {p}s (worst difference {worst}) \
             — it is periodic after all and loop_period must be updated to say so"
        );
    }
}

// ──────────────── Breathe negative-phase hardening (follow-up) ────────────────

/// How many phases each `Breathe` probe sweeps across one cycle. A POWER OF TWO on
/// purpose: `i / 256` is exactly representable in `f32`, so `1 − i/256` and
/// `(256 − i)/256` are the same float and the mirror comparison below is exact
/// arithmetic rather than a tolerance that might be hiding a small real error.
///
/// It is a full sweep, not a spot check, because Task 2's first negative-speed draft
/// PASSED against unhardened code by probing a single `t` — a broken envelope is broken
/// over a window, and at some instants it coincides with the correct one.
const BREATHE_PROBE_PHASES: usize = 256;
/// `Breathe`'s amplitude for these probes. Strong, so "at rest" and "at peak" are far
/// apart and a wrong envelope cannot hide inside the tolerance.
const BREATHE_PROBE_AMPLITUDE: f32 = 0.5;
/// Where the peak must sit, as a fraction of the swept cycle, when the envelope runs
/// FORWARD (positive speed). The shipped `IN` is `0.7`, but it is declared inside the
/// match arm and is not importable, so this band MEASURES the peak rather than
/// asserting a copy of the constant — widening `IN` past the band fails this test.
const BREATHE_FWD_PEAK_BAND: (f32, f32) = (0.55, 0.85);
/// … and where it must sit when the envelope runs BACKWARD (negative speed): the mirror
/// of the band above, because a peak at `q` forward lands at `1 − q` backward. The two
/// bands are disjoint, which is what makes "reversed" distinguishable from "reproduced".
const BREATHE_BACK_PEAK_BAND: (f32, f32) = (0.15, 0.45);

/// A `Breathe` material at the given (possibly negative) `speed`.
fn breathe_material(speed: f32) -> Material {
    Material {
        animation: Animation {
            kind: AnimKind::Breathe,
            speed,
            amplitude: BREATHE_PROBE_AMPLITUDE,
        },
        ..DEFAULT_MATERIAL.clone()
    }
}

/// The envelope sampled at `BREATHE_PROBE_PHASES` points across one cycle beginning at
/// `t0`. `t0` is a free parameter so the same rig can probe a NEGATIVE time origin —
/// the other way a negative value reaches the phase expression.
fn breathe_cycle(mat: &Material, t0: f32) -> Vec<f32> {
    let period = 1.0 / mat.animation.speed.abs();
    (0..BREATHE_PROBE_PHASES)
        .map(|i| {
            let t = t0 + period * i as f32 / BREATHE_PROBE_PHASES as f32;
            mat.anim_brightness(PROBE_COORD, &probe_ctx(t))
        })
        .collect()
}

/// Index of the largest sample.
fn peak_index(samples: &[f32]) -> usize {
    samples
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).expect("no NaN in an envelope"))
        .map(|(i, _)| i)
        .expect("a non-empty sweep")
}

/// A NEGATIVE `speed`, and a NEGATIVE `time`, must each still give a rise-then-fall
/// envelope that returns to rest at the wrap — and reversing the speed must MIRROR the
/// envelope in time, not reproduce it.
///
/// This is `Scan`'s hardening applied to the other half of the pair. `f32::fract` is
/// SIGN-PRESERVING, so a negative product lands `p` in `(−1, 0]`; then `p < IN` holds
/// for the WHOLE cycle, the fall segment is never reached, and — because `cos` is even —
/// the arm computes `env(|p|)`, i.e. it silently REPLAYS the forward envelope instead of
/// reversing it, and lands the wrap at `env ≈ 0.38` instead of `0` (a value jump the eye
/// reads as a cut).
///
/// VERIFIED, not assumed, that folding the phase is the COMPLETE fix here: unlike
/// `Scan`, whose `coord.x` reached the same subtraction independently and therefore
/// needed folding too, the `Breathe` arm reads no coordinate at all — `p` is its sole
/// operand. But `p` has TWO negative-capable inputs, `speed` and `ctx.time`, so both
/// routes are swept below.
#[test]
fn breathe_stays_a_rise_then_fall_envelope_for_negative_speed_and_negative_time() {
    let n = BREATHE_PROBE_PHASES as f32;
    // Both signs of `speed` crossed with a positive, zero and negative time origin, so
    // neither input route into the phase can be broken while the other masks it.
    for (speed, t0) in [
        (-1.0_f32, 0.0_f32),
        (-1.0, 37.0),
        (-1.0, -37.0),
        (1.0, 0.0),
        (1.0, 37.0),
        (1.0, -37.0),
    ] {
        let mat = breathe_material(speed);
        let samples = breathe_cycle(&mat, t0);

        // 1. Bounded in `[1, 1+amplitude]` at EVERY phase — `Breathe` only brightens.
        for (i, v) in samples.iter().enumerate() {
            assert!(
                (1.0 - 1.0e-5..=1.0 + BREATHE_PROBE_AMPLITUDE + 1.0e-5).contains(v),
                "speed {speed}, t0 {t0}, phase {i}: brightness {v} left [1, 1+amplitude]"
            );
        }

        // 2. The envelope RETURNS TO REST at the wrap. This is the seam: an unfolded
        //    negative phase never reaches the fall segment, so it ends the cycle part
        //    way up and jumps back down at the wrap.
        assert!(
            (samples[0] - 1.0).abs() < 1.0e-5,
            "speed {speed}, t0 {t0}: Breathe must start at rest, got {}",
            samples[0]
        );
        assert!(
            (samples[BREATHE_PROBE_PHASES - 1] - 1.0).abs() < 0.02,
            "speed {speed}, t0 {t0}: Breathe must come back to rest by the wrap, got {} \
             — the cycle ends part way up, so the wrap is a visible jump",
            samples[BREATHE_PROBE_PHASES - 1]
        );

        // 3. The peak sits late in the cycle running forward, and — mirrored — early
        //    running backward. The two bands are disjoint, so an implementation that
        //    replays the forward shape for a negative speed cannot satisfy both.
        let (lo, hi) = if speed > 0.0 {
            BREATHE_FWD_PEAK_BAND
        } else {
            BREATHE_BACK_PEAK_BAND
        };
        let peak = peak_index(&samples) as f32 / n;
        assert!(
            (lo..=hi).contains(&peak),
            "speed {speed}, t0 {t0}: the peak sits at {peak:.3} of the cycle, outside \
             [{lo}, {hi}] — the envelope is not running in the direction the speed asks"
        );
    }

    // 4. Reversing the speed MIRRORS the envelope in time. `back[i]` must equal
    //    `fwd[(N-i) % N]` — exactly, since the sample times are exact binary fractions.
    let fwd = breathe_cycle(&breathe_material(1.0), 0.0);
    let back = breathe_cycle(&breathe_material(-1.0), 0.0);
    for i in 0..BREATHE_PROBE_PHASES {
        let mirrored = fwd[(BREATHE_PROBE_PHASES - i) % BREATHE_PROBE_PHASES];
        assert!(
            (back[i] - mirrored).abs() < 1.0e-5,
            "phase {i}: reversing the speed must mirror the envelope in time — \
             back {} vs forward-mirrored {mirrored}",
            back[i]
        );
    }

    // 5. Anti-vacuity for step 4: the mirrored sweep must actually DIFFER from the
    //    un-mirrored one, or "mirror" would be satisfied by a symmetric envelope — and
    //    `Breathe`'s whole point is that it is asymmetric. Under a sign-preserving
    //    `fract` these two are IDENTICAL, which is precisely the defect.
    let worst = (0..BREATHE_PROBE_PHASES)
        .map(|i| (back[i] - fwd[i]).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        worst > MOTION_EPS,
        "a negative speed reproduced the forward envelope instead of reversing it \
         (worst difference {worst})"
    );
}
