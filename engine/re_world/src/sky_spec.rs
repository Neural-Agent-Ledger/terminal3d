//! `re_world::sky_spec` — the per-level PROCEDURAL sky program: a small stack of
//! up to a few layers (gradient base + stars/aurora/clouds/storm/data-rain/void/
//! fractal/watcher/mirror/constellation) parsed from a level's `[sky]` section.
//!
//! This module is pure data + pure parsing — NO rendering, NO `re_render`
//! dependency, NO terminal. `re_render::sky::SkyProgram::compile` turns a
//! [`SkySpec`] into the time-aware sampler; this module only describes WHAT was
//! authored.
//!
//! Additive: a level with no `[sky]` `base =` / `layer =` / `kind =` key never
//! builds a [`SkySpec`] (`re_level` leaves [`Map::sky_spec`](../../re_world/struct.Map.html#method.sky_spec)
//! at `None`), so the renderer keeps sampling the legacy [`ProceduralSky`]/
//! [`ImageSky`] path exactly as before — byte-identical to today for every
//! existing level with no `[sky]` section.
//!
//! [`ProceduralSky`]: ../../re_render/sky/struct.ProceduralSky.html
//! [`ImageSky`]: ../../re_render/sky/struct.ImageSky.html

/// The floor of every sky: a vertical gradient (horizon → zenith) plus an
/// optional horizon glow band. `base = gradient horizon=#rrggbb zenith=#rrggbb
/// [glow=#rrggbb] [glow_width=0.0..1.0]` (spec §2.1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkyBase {
    /// Packed `0x00RRGGBB` horizon tone (bottom of the dome).
    pub horizon: u32,
    /// Packed `0x00RRGGBB` zenith tone (top of the dome / cap centre).
    pub zenith: u32,
    /// Optional horizon airglow/city-light tint, blended in near the horizon.
    pub glow: Option<u32>,
    /// Fraction of the dome height (from the horizon) the glow band covers.
    pub glow_width: f32,
}

impl Default for SkyBase {
    /// A dark, cool dome — a reasonable default for a `kind =` preset that names
    /// no explicit `base`.
    fn default() -> Self {
        SkyBase {
            horizon: 0x1A2238,
            zenith: 0x04060E,
            glow: None,
            glow_width: 0.12,
        }
    }
}

/// `layer = stars density=.. twinkle=.. milkyway=.. pole_tilt=.. wheel=.. meteors=..
/// palette=..` (spec §2.2).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StarsParams {
    /// Star fraction of the dome.
    pub density: f32,
    /// Twinkle amount `[0,1]`.
    pub twinkle: f32,
    /// Milky-way band brightness `[0,1]`.
    pub milkyway: f32,
    /// Celestial-pole tilt, degrees.
    pub pole_tilt_deg: f32,
    /// Sky-wheel rotation rate.
    pub wheel: f32,
    /// Meteor event rate.
    pub meteors: f32,
}

impl Default for StarsParams {
    fn default() -> Self {
        StarsParams {
            density: 0.025,
            twinkle: 0.5,
            milkyway: 0.35,
            pole_tilt_deg: 24.0,
            wheel: 0.01,
            meteors: 0.0,
        }
    }
}

/// `layer = constellation stars=(u,v) (u,v) (u,v) link=tag:name_a,name_b,name_c`
/// (spec §2.2 — the L6 lore hook). `stars` are three normalized `(u,v)` sky
/// positions (`u` = wrapped yaw fraction, `v` = dome height fraction, both
/// `[0,1)`); `link` names three `state_bus` tags whose state (0/1) gates each
/// star's "powered" look. Tag NAMES are hashed to the `u16` id `state_bus`
/// indexes by (`re_render::sky::stars::tag_id_for_name`) — the hook contract any
/// future writer (e.g. L6's puzzle wiring) must use the SAME hash to publish
/// state under the id this layer reads.
#[derive(Clone, Debug, PartialEq)]
pub struct ConstellationParams {
    /// Three normalized `(u,v)` sky positions (`u` = wrapped yaw fraction,
    /// `v` = dome height fraction, both `[0,1)`).
    pub stars: [(f32, f32); 3],
    /// The three `state_bus` tag names gating each star's "powered" look.
    pub link: [String; 3],
}

impl Default for ConstellationParams {
    fn default() -> Self {
        ConstellationParams {
            stars: [(0.31, 0.42), (0.34, 0.47), (0.29, 0.51)],
            link: [
                "star_a".to_string(),
                "star_b".to_string(),
                "star_c".to_string(),
            ],
        }
    }
}

/// `layer = aurora curtains=.. band=lo..hi height=.. decay=.. palette=.. activity=..
/// shimmer_hz=.. surge_rate=.. drift_period=..` (spec §2.4).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AuroraParams {
    /// Number of aurora curtains.
    pub curtains: u32,
    /// Band lower bound (dome-height fraction).
    pub band_lo: f32,
    /// Band upper bound (dome-height fraction).
    pub band_hi: f32,
    /// Curtain height.
    pub height: f32,
    /// Colour decay rate with altitude.
    pub decay: f32,
    /// Colour-by-altitude preset (see [`AuroraPalette`]).
    pub palette: AuroraPalette,
    /// Overall activity level.
    pub activity: f32,
    /// Shimmer rate, Hz.
    pub shimmer_hz: f32,
    /// Rate of activity surges.
    pub surge_rate: f32,
    /// Period of the slow curtain drift, seconds.
    pub drift_period: f32,
}

impl Default for AuroraParams {
    fn default() -> Self {
        AuroraParams {
            curtains: 3,
            band_lo: 0.18,
            band_hi: 0.55,
            height: 0.22,
            decay: 3.0,
            palette: AuroraPalette::Oxygen,
            activity: 0.6,
            shimmer_hz: 2.0,
            surge_rate: 0.15,
            drift_period: 180.0,
        }
    }
}

/// The aurora colour-by-altitude preset (spec §2.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AuroraPalette {
    /// N₂ purple/pink hem → O green body → O red top — the emission-line ramp.
    #[default]
    Oxygen,
    /// A cooler violet-dominant ramp.
    Violet,
    /// A single-hue (green) ramp — cheap / colour-gated levels.
    Mono,
}

/// `layer = clouds cover=.. soft=.. scale=.. height=.. wind=dx,dy layers=1|2
/// lit=#rrggbb shade=#rrggbb haze=.. bake=on|off` (spec §2.3).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CloudsParams {
    /// Cloud coverage fraction `[0,1]`.
    pub cover: f32,
    /// Edge softness.
    pub soft: f32,
    /// Noise-field scale.
    pub scale: f32,
    /// Cloud-deck height.
    pub height: f32,
    /// Wind drift `(dx, dy)`.
    pub wind: (f32, f32),
    /// Number of cloud layers (`1` or `2`).
    pub layers: u32,
    /// Lit-face colour, packed `0x00RRGGBB`.
    pub lit: u32,
    /// Shadowed-face colour, packed `0x00RRGGBB`.
    pub shade: u32,
    /// Horizon haze amount.
    pub haze: f32,
    /// Whether the cloud field is baked (static) rather than animated.
    pub bake: bool,
}

impl Default for CloudsParams {
    fn default() -> Self {
        CloudsParams {
            cover: 0.4,
            soft: 0.15,
            scale: 1.0,
            height: 60.0,
            wind: (0.05, 0.0),
            layers: 1,
            lit: 0xFFF0D0,
            shade: 0x202030,
            haze: 40.0,
            bake: false,
        }
    }
}

/// `layer = storm cover=.. rate=..` — a dark cloud deck + lightning event stream
/// (spec §2.5). `rate` is bolts/second (before the ≤ 3/s WCAG cap).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StormParams {
    /// Cloud-deck coverage fraction.
    pub cover: f32,
    /// Lightning bolts per second (before the WCAG cap).
    pub rate: f32,
}

impl Default for StormParams {
    fn default() -> Self {
        StormParams {
            cover: 0.85,
            rate: 0.3,
        }
    }
}

/// `layer = datarain source=binary|hex|pool|shards speed0=.. speed1=.. trail=..`
/// (spec §2.7).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DataRainParams {
    /// Glyph source the falling columns draw from.
    pub source: DataRainSource,
    /// Slowest column fall speed.
    pub speed0: f32,
    /// Fastest column fall speed.
    pub speed1: f32,
    /// Trailing-fade amount behind each column's head.
    pub trail: f32,
}

impl Default for DataRainParams {
    fn default() -> Self {
        DataRainParams {
            source: DataRainSource::Binary,
            speed0: 0.5,
            speed1: 1.5,
            trail: 0.35,
        }
    }
}

/// The glyph source a `datarain` layer's columns draw from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DataRainSource {
    /// `0`/`1` glyphs.
    #[default]
    Binary,
    /// Hexadecimal-digit glyphs.
    Hex,
    /// A pool of miscellaneous symbol glyphs.
    Pool,
    /// Fragmented shard glyphs.
    Shards,
}

/// `layer = void [drift=..]` — near-black dome with a sparse drifting `·` field
/// (spec §2.6).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoidParams {
    /// Drift speed of the sparse `·` field.
    pub drift: f32,
}

impl Default for VoidParams {
    fn default() -> Self {
        VoidParams { drift: 0.15 }
    }
}

/// `layer = fractal [speed=..]` — the cap becomes an infinite log-polar tunnel
/// (spec §2.6).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FractalParams {
    /// Log-polar tunnel zoom/spin speed.
    pub speed: f32,
}

impl Default for FractalParams {
    fn default() -> Self {
        FractalParams { speed: 0.12 }
    }
}

/// `layer = watcher [iris=#rrggbb]` — the splash eye rendered into the zenith cap,
/// pupil tracking the PLAYER'S yaw (spec §2.6).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WatcherParams {
    /// Iris colour, packed `0x00RRGGBB`.
    pub iris: u32,
}

impl Default for WatcherParams {
    fn default() -> Self {
        WatcherParams { iris: 0x2A3040 }
    }
}

/// `layer = mirror [dim=..]` — the level's own footprint, seen from below,
/// rotating the wrong way (spec §2.6).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MirrorParams {
    /// Reflection dimming factor.
    pub dim: f32,
}

impl Default for MirrorParams {
    fn default() -> Self {
        MirrorParams { dim: 0.4 }
    }
}

/// `layer = skyline blocks=.. density=.. height=lo..hi style=.. lit=..
/// landmark_rate=..` — a distant-city horizon. Each of `blocks` azimuth slots around the full turn
/// is a candidate tower: `density` decides whether it carries one at all, and (for
/// the ones that do) a seeded hash picks its height in `[height_lo, height_hi]`
/// (a dome-height fraction, so it is horizon-anchored, not world-scaled — see
/// `re_render::sky::dome_height` on the render side) and a hue. `style` reuses
/// [`crate::FacadeStyle`] — the SAME window-unit table
/// [`crate::MaterialRecipe::Facade`] uses — so a tower's windows read with the
/// exact rhythm a nearby facade wall would. `landmark_rate` is the fraction of
/// towers that carry a spire above the roofline with a slow blinking antenna.
///
/// PARALLAX-FREE: like every sky layer this is sampled by ray DIRECTION only
/// (`SkyCtx::yaw`/`pitch`), never by `SkyCtx::cam_xy` — a "distant" skyline does
/// not need real-world distance to read as far away; it only needs to never
/// scroll under camera translation the way `clouds`' plane-hit does.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkylineParams {
    /// Azimuth blocks around the full turn — coarser than `stars`' star grid;
    /// each block is one candidate tower slot.
    pub blocks: u32,
    /// Fraction of blocks that actually carry a tower; the rest show sky/gap.
    pub density: f32,
    /// Shortest authored tower height, a dome-height fraction `[0,1]`.
    pub height_lo: f32,
    /// Tallest authored tower height, a dome-height fraction `[0,1]`.
    pub height_hi: f32,
    /// The window-unit table a tower's facade draws from — reused from
    /// [`crate::MaterialRecipe::Facade`]'s catalog.
    pub style: crate::FacadeStyle,
    /// Fraction of a tower's windows that hash lit.
    pub lit: f32,
    /// Fraction of towers that are LANDMARKS (spire + blinking antenna).
    pub landmark_rate: f32,
}

impl Default for SkylineParams {
    fn default() -> Self {
        SkylineParams {
            blocks: 96,
            density: 0.75,
            height_lo: 0.12,
            height_hi: 0.55,
            style: crate::FacadeStyle::Office,
            lit: 0.3,
            landmark_rate: 0.08,
        }
    }
}

/// One authored `layer = <kind> ...` entry. See each params type's docs for its
/// authoring keys (spec §2, §6.1 B5/B3/B8/B14/B20/B21/B22).
#[derive(Clone, Debug, PartialEq)]
pub enum SkyLayerSpec {
    /// `layer = stars ...` (see [`StarsParams`]).
    Stars(StarsParams),
    /// `layer = constellation ...` (see [`ConstellationParams`]).
    Constellation(ConstellationParams),
    /// `layer = aurora ...` (see [`AuroraParams`]).
    Aurora(AuroraParams),
    /// `layer = clouds ...` (see [`CloudsParams`]).
    Clouds(CloudsParams),
    /// `layer = storm ...` (see [`StormParams`]).
    Storm(StormParams),
    /// `layer = datarain ...` (see [`DataRainParams`]).
    DataRain(DataRainParams),
    /// `layer = void ...` (see [`VoidParams`]).
    Void(VoidParams),
    /// `layer = fractal ...` (see [`FractalParams`]).
    Fractal(FractalParams),
    /// `layer = watcher ...` (see [`WatcherParams`]).
    Watcher(WatcherParams),
    /// `layer = mirror ...` (see [`MirrorParams`]).
    Mirror(MirrorParams),
    /// `layer = skyline ...` (see [`SkylineParams`]).
    Skyline(SkylineParams),
}

/// The full per-level sky program: a gradient `base` plus up to a handful of
/// `layers`, a `seed` for the layers' deterministic hashing, and the tick rate
/// (`hz`) an animated layer re-evaluates at (spec §2.0).
#[derive(Clone, Debug, PartialEq)]
pub struct SkySpec {
    /// The gradient dome floor every layer composites over.
    pub base: SkyBase,
    /// The authored layer stack, bottom to top.
    pub layers: Vec<SkyLayerSpec>,
    /// Seeds the layers' deterministic hashing (star placement, tower hashes, …).
    pub seed: u32,
    /// Tick rate an animated layer re-evaluates at, in Hz.
    pub hz: u32,
}

impl Default for SkySpec {
    fn default() -> Self {
        SkySpec {
            base: SkyBase::default(),
            layers: Vec::new(),
            seed: 0,
            hz: 12,
        }
    }
}

// ───────────────────────────── parsing ──────────────────────────────────────

/// Parse a `#rrggbb` colour literal into a packed `0x00RRGGBB` value.
fn parse_hex_color(v: &str) -> Result<u32, String> {
    let hex = v
        .strip_prefix('#')
        .ok_or_else(|| format!("colour '{v}' must start with '#'"))?;
    if hex.len() != 6 {
        return Err(format!("colour '{v}' must be '#rrggbb' (6 hex digits)"));
    }
    u32::from_str_radix(hex, 16).map_err(|_| format!("colour '{v}' is not valid hex"))
}

/// Parse a plain float value.
fn parse_f32(k: &str, v: &str) -> Result<f32, String> {
    v.parse::<f32>()
        .map_err(|_| format!("{k}: '{v}' is not a number"))
}

/// Parse a plain non-negative integer value.
fn parse_u32(k: &str, v: &str) -> Result<u32, String> {
    v.parse::<u32>()
        .map_err(|_| format!("{k}: '{v}' is not a non-negative integer"))
}

/// Parse a `lo..hi` range, e.g. `band=0.18..0.55`.
fn parse_range(k: &str, v: &str) -> Result<(f32, f32), String> {
    let (lo, hi) = v
        .split_once("..")
        .ok_or_else(|| format!("{k}: '{v}' must be 'lo..hi'"))?;
    Ok((parse_f32(k, lo)?, parse_f32(k, hi)?))
}

/// Parse a `dx,dy` pair, e.g. `wind=0.05,0.0`.
fn parse_pair(k: &str, v: &str) -> Result<(f32, f32), String> {
    let (a, b) = v
        .split_once(',')
        .ok_or_else(|| format!("{k}: '{v}' must be 'a,b'"))?;
    Ok((parse_f32(k, a)?, parse_f32(k, b)?))
}

/// Parse a `on`/`off` boolean flag (matches the rest of the level format's
/// keyword flags, e.g. `[texture] tile`).
fn parse_bool(k: &str, v: &str) -> Result<bool, String> {
    match v.trim() {
        "on" | "true" | "1" => Ok(true),
        "off" | "false" | "0" => Ok(false),
        other => Err(format!("{k}: '{other}' must be on/off")),
    }
}

/// Split a layer body into `key=value` tokens (whitespace-separated; every
/// non-constellation layer's values are single whitespace-free tokens).
fn kv_tokens(s: &str) -> Result<Vec<(&str, &str)>, String> {
    let mut out = Vec::new();
    for tok in s.split_whitespace() {
        let (k, v) = tok
            .split_once('=')
            .ok_or_else(|| format!("bad token '{tok}' (expected key=value)"))?;
        out.push((k, v));
    }
    Ok(out)
}

impl SkySpec {
    /// Parse a `[sky] base = <value>` line's value into a [`SkyBase`]. The first
    /// token names the base kind (only `gradient` today); the rest are
    /// `key=value` pairs. Unknown keys and a non-`gradient` kind are
    /// LINE-NUMBERED errors at the `re_level` call site (this function reports
    /// the message only — no line number, matching every other `parse_*` helper
    /// in this crate).
    pub fn parse_base(s: &str) -> Result<SkyBase, String> {
        let s = s.trim();
        let mut toks = s.split_whitespace();
        let kind = toks
            .next()
            .ok_or_else(|| "[sky] base: empty value".to_string())?;
        if kind != "gradient" {
            return Err(format!(
                "[sky] base: unknown kind '{kind}' (only 'gradient' is supported)"
            ));
        }
        let mut base = SkyBase::default();
        for tok in toks {
            let (k, v) = tok
                .split_once('=')
                .ok_or_else(|| format!("[sky] base: bad token '{tok}' (expected key=value)"))?;
            match k {
                "horizon" => base.horizon = parse_hex_color(v)?,
                "zenith" => base.zenith = parse_hex_color(v)?,
                "glow" => base.glow = Some(parse_hex_color(v)?),
                "glow_width" => base.glow_width = parse_f32(k, v)?,
                other => return Err(format!("[sky] base: unknown key '{other}'")),
            }
        }
        Ok(base)
    }

    /// Parse one `[sky] layer = <value>` line into a [`SkyLayerSpec`]. The first
    /// token names the layer kind; the rest are `key=value` pairs (special-cased
    /// for `constellation`, whose `stars=` value is itself space-separated
    /// `(u,v)` tuples).
    pub fn parse_layer(s: &str) -> Result<SkyLayerSpec, String> {
        let s = s.trim();
        if let Some(rest) = s.strip_prefix("constellation") {
            return Self::parse_constellation(rest.trim()).map(SkyLayerSpec::Constellation);
        }
        let mut split = s.splitn(2, char::is_whitespace);
        let kind = split
            .next()
            .ok_or_else(|| "[sky] layer: empty value".to_string())?;
        let rest = split.next().unwrap_or("");
        let kv = kv_tokens(rest).map_err(|e| format!("[sky] layer {kind}: {e}"))?;
        match kind {
            "stars" => Self::parse_stars(&kv).map(SkyLayerSpec::Stars),
            "aurora" => Self::parse_aurora(&kv).map(SkyLayerSpec::Aurora),
            "clouds" => Self::parse_clouds(&kv).map(SkyLayerSpec::Clouds),
            "storm" => Self::parse_storm(&kv).map(SkyLayerSpec::Storm),
            "datarain" => Self::parse_datarain(&kv).map(SkyLayerSpec::DataRain),
            "void" => Self::parse_void(&kv).map(SkyLayerSpec::Void),
            "fractal" => Self::parse_fractal(&kv).map(SkyLayerSpec::Fractal),
            "watcher" => Self::parse_watcher(&kv).map(SkyLayerSpec::Watcher),
            "mirror" => Self::parse_mirror(&kv).map(SkyLayerSpec::Mirror),
            "skyline" => Self::parse_skyline(&kv).map(SkyLayerSpec::Skyline),
            other => Err(format!("[sky] layer: unknown kind '{other}'")),
        }
    }

    fn parse_stars(kv: &[(&str, &str)]) -> Result<StarsParams, String> {
        let mut p = StarsParams::default();
        for &(k, v) in kv {
            match k {
                "density" => p.density = parse_f32(k, v)?,
                "twinkle" => p.twinkle = parse_f32(k, v)?,
                "milkyway" => p.milkyway = parse_f32(k, v)?,
                "pole_tilt" => p.pole_tilt_deg = parse_f32(k, v)?,
                "wheel" => p.wheel = parse_f32(k, v)?,
                "meteors" => p.meteors = parse_f32(k, v)?,
                "palette" => {} // colour-class palette is fixed today; accept + ignore.
                "seed" => {}    // per-layer seed override not yet supported; accept + ignore.
                other => return Err(format!("stars: unknown key '{other}'")),
            }
        }
        Ok(p)
    }

    fn parse_constellation(rest: &str) -> Result<ConstellationParams, String> {
        let mut p = ConstellationParams::default();
        let mut stars: Vec<(f32, f32)> = Vec::new();
        let mut toks = rest.split_whitespace().peekable();
        while let Some(tok) = toks.peek().copied() {
            if let Some(after) = tok.strip_prefix("stars=") {
                stars.push(Self::parse_uv_tuple(after)?);
                toks.next();
                // Consume further bare `(u,v)` tuples (no '=') as more stars.
                while let Some(next) = toks.peek().copied() {
                    if next.starts_with('(') && !next.contains('=') {
                        stars.push(Self::parse_uv_tuple(next)?);
                        toks.next();
                    } else {
                        break;
                    }
                }
            } else if let Some(after) = tok.strip_prefix("link=") {
                let names: Vec<String> = after
                    .trim_start_matches("tag:")
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                if names.len() != 3 {
                    return Err(format!(
                        "constellation: link must name exactly 3 tags, got {}",
                        names.len()
                    ));
                }
                p.link = [names[0].clone(), names[1].clone(), names[2].clone()];
                toks.next();
            } else {
                return Err(format!("constellation: bad token '{tok}'"));
            }
        }
        if stars.len() == 3 {
            p.stars = [stars[0], stars[1], stars[2]];
        } else if !stars.is_empty() {
            return Err(format!(
                "constellation: stars must name exactly 3 positions, got {}",
                stars.len()
            ));
        }
        Ok(p)
    }

    /// Parse one `(u,v)` tuple token (optionally comma-separated, parens
    /// required — matches the spec's `stars=(0.31,0.42) (0.34,0.47) ...`).
    fn parse_uv_tuple(tok: &str) -> Result<(f32, f32), String> {
        let inner = tok
            .strip_prefix('(')
            .and_then(|s| s.strip_suffix(')'))
            .ok_or_else(|| format!("constellation: '{tok}' must be '(u,v)'"))?;
        let (u, v) = inner
            .split_once(',')
            .ok_or_else(|| format!("constellation: '{tok}' must be '(u,v)'"))?;
        Ok((parse_f32("stars", u)?, parse_f32("stars", v)?))
    }

    fn parse_aurora(kv: &[(&str, &str)]) -> Result<AuroraParams, String> {
        let mut p = AuroraParams::default();
        for &(k, v) in kv {
            match k {
                "curtains" => p.curtains = parse_u32(k, v)?,
                "band" => {
                    let (lo, hi) = parse_range(k, v)?;
                    p.band_lo = lo;
                    p.band_hi = hi;
                }
                "height" => p.height = parse_f32(k, v)?,
                "decay" => p.decay = parse_f32(k, v)?,
                "palette" => {
                    p.palette = match v {
                        "oxygen" => AuroraPalette::Oxygen,
                        "violet" => AuroraPalette::Violet,
                        "mono" => AuroraPalette::Mono,
                        other => {
                            return Err(format!(
                                "aurora: unknown palette '{other}' (oxygen/violet/mono)"
                            ))
                        }
                    };
                }
                "activity" => p.activity = parse_f32(k, v)?,
                "shimmer_hz" => p.shimmer_hz = parse_f32(k, v)?,
                "surge_rate" => p.surge_rate = parse_f32(k, v)?,
                "drift_period" => p.drift_period = parse_f32(k, v)?,
                other => return Err(format!("aurora: unknown key '{other}'")),
            }
        }
        Ok(p)
    }

    fn parse_clouds(kv: &[(&str, &str)]) -> Result<CloudsParams, String> {
        let mut p = CloudsParams::default();
        for &(k, v) in kv {
            match k {
                "cover" => p.cover = parse_f32(k, v)?,
                "soft" => p.soft = parse_f32(k, v)?,
                "scale" => p.scale = parse_f32(k, v)?,
                "height" => p.height = parse_f32(k, v)?,
                "wind" => p.wind = parse_pair(k, v)?,
                "layers" => p.layers = parse_u32(k, v)?,
                "lit" => p.lit = parse_hex_color(v)?,
                "shade" => p.shade = parse_hex_color(v)?,
                "haze" => p.haze = parse_f32(k, v)?,
                "bake" => p.bake = parse_bool(k, v)?,
                other => return Err(format!("clouds: unknown key '{other}'")),
            }
        }
        Ok(p)
    }

    fn parse_storm(kv: &[(&str, &str)]) -> Result<StormParams, String> {
        let mut p = StormParams::default();
        for &(k, v) in kv {
            match k {
                "cover" => p.cover = parse_f32(k, v)?,
                "rate" => p.rate = parse_f32(k, v)?,
                other => return Err(format!("storm: unknown key '{other}'")),
            }
        }
        Ok(p)
    }

    fn parse_datarain(kv: &[(&str, &str)]) -> Result<DataRainParams, String> {
        let mut p = DataRainParams::default();
        for &(k, v) in kv {
            match k {
                "source" => {
                    p.source = match v {
                        "binary" => DataRainSource::Binary,
                        "hex" => DataRainSource::Hex,
                        "pool" => DataRainSource::Pool,
                        "shards" => DataRainSource::Shards,
                        other => {
                            return Err(format!(
                                "datarain: unknown source '{other}' (binary/hex/pool/shards)"
                            ))
                        }
                    };
                }
                "speed0" => p.speed0 = parse_f32(k, v)?,
                "speed1" => p.speed1 = parse_f32(k, v)?,
                "trail" => p.trail = parse_f32(k, v)?,
                other => return Err(format!("datarain: unknown key '{other}'")),
            }
        }
        Ok(p)
    }

    fn parse_void(kv: &[(&str, &str)]) -> Result<VoidParams, String> {
        let mut p = VoidParams::default();
        for &(k, v) in kv {
            match k {
                "drift" => p.drift = parse_f32(k, v)?,
                other => return Err(format!("void: unknown key '{other}'")),
            }
        }
        Ok(p)
    }

    fn parse_fractal(kv: &[(&str, &str)]) -> Result<FractalParams, String> {
        let mut p = FractalParams::default();
        for &(k, v) in kv {
            match k {
                "speed" => p.speed = parse_f32(k, v)?,
                other => return Err(format!("fractal: unknown key '{other}'")),
            }
        }
        Ok(p)
    }

    fn parse_watcher(kv: &[(&str, &str)]) -> Result<WatcherParams, String> {
        let mut p = WatcherParams::default();
        for &(k, v) in kv {
            match k {
                "iris" => p.iris = parse_hex_color(v)?,
                other => return Err(format!("watcher: unknown key '{other}'")),
            }
        }
        Ok(p)
    }

    fn parse_mirror(kv: &[(&str, &str)]) -> Result<MirrorParams, String> {
        let mut p = MirrorParams::default();
        for &(k, v) in kv {
            match k {
                "dim" => p.dim = parse_f32(k, v)?,
                other => return Err(format!("mirror: unknown key '{other}'")),
            }
        }
        Ok(p)
    }

    fn parse_skyline(kv: &[(&str, &str)]) -> Result<SkylineParams, String> {
        let mut p = SkylineParams::default();
        for &(k, v) in kv {
            match k {
                "blocks" => p.blocks = parse_u32(k, v)?,
                "density" => p.density = parse_f32(k, v)?,
                "height" => {
                    let (lo, hi) = parse_range(k, v)?;
                    p.height_lo = lo;
                    p.height_hi = hi;
                }
                "style" => {
                    p.style = match v {
                        "office" => crate::FacadeStyle::Office,
                        "residential" => crate::FacadeStyle::Residential,
                        "industrial" => crate::FacadeStyle::Industrial,
                        "server_rack" => crate::FacadeStyle::ServerRack,
                        "lockers" => crate::FacadeStyle::Lockers,
                        other => return Err(format!("skyline: unknown style '{other}'")),
                    };
                }
                "lit" => p.lit = parse_f32(k, v)?,
                "landmark_rate" => p.landmark_rate = parse_f32(k, v)?,
                other => return Err(format!("skyline: unknown key '{other}'")),
            }
        }
        Ok(p)
    }

    /// Expand a `[sky] kind = <name>` shorthand into its preset layer stack
    /// (spec §2.0: "`kind =` is a shorthand that expands to a preset stack").
    pub fn from_kind(name: &str) -> Result<SkySpec, String> {
        let name = name.trim();
        let layers = match name {
            "stars" => vec![SkyLayerSpec::Stars(StarsParams::default())],
            "aurora" => vec![
                SkyLayerSpec::Stars(StarsParams {
                    density: 0.015,
                    ..StarsParams::default()
                }),
                SkyLayerSpec::Aurora(AuroraParams::default()),
            ],
            "clouds" => vec![SkyLayerSpec::Clouds(CloudsParams::default())],
            "storm" => vec![SkyLayerSpec::Storm(StormParams::default())],
            "datarain" => vec![SkyLayerSpec::DataRain(DataRainParams::default())],
            "void" => vec![SkyLayerSpec::Void(VoidParams::default())],
            "fractal" => vec![SkyLayerSpec::Fractal(FractalParams::default())],
            "watcher" => vec![SkyLayerSpec::Watcher(WatcherParams::default())],
            "mirror" => vec![SkyLayerSpec::Mirror(MirrorParams::default())],
            "skyline" => vec![
                SkyLayerSpec::Stars(StarsParams {
                    density: 0.01,
                    ..StarsParams::default()
                }),
                SkyLayerSpec::Skyline(SkylineParams::default()),
            ],
            "constellation" => vec![
                SkyLayerSpec::Stars(StarsParams {
                    density: 0.012,
                    twinkle: 0.2,
                    ..StarsParams::default()
                }),
                SkyLayerSpec::Constellation(ConstellationParams::default()),
            ],
            other => return Err(format!("[sky] kind: unknown preset '{other}'")),
        };
        Ok(SkySpec {
            base: SkyBase::default(),
            layers,
            ..SkySpec::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_base_gradient_all_keys() {
        let b = SkySpec::parse_base(
            "gradient horizon=#1A2238 zenith=#04060E glow=#3A2A40 glow_width=0.12",
        )
        .unwrap();
        assert_eq!(b.horizon, 0x1A2238);
        assert_eq!(b.zenith, 0x04060E);
        assert_eq!(b.glow, Some(0x3A2A40));
        assert!((b.glow_width - 0.12).abs() < 1e-6);
    }

    #[test]
    fn parse_base_rejects_unknown_kind() {
        assert!(SkySpec::parse_base("skybox horizon=#000000").is_err());
    }

    #[test]
    fn parse_base_rejects_unknown_key() {
        assert!(SkySpec::parse_base("gradient bogus=1").is_err());
    }

    #[test]
    fn parse_layer_stars_all_keys() {
        let l = SkySpec::parse_layer(
            "stars density=0.025 twinkle=0.5 milkyway=0.35 pole_tilt=24 wheel=0.01 meteors=0.1",
        )
        .unwrap();
        match l {
            SkyLayerSpec::Stars(p) => {
                assert!((p.density - 0.025).abs() < 1e-6);
                assert!((p.twinkle - 0.5).abs() < 1e-6);
                assert!((p.milkyway - 0.35).abs() < 1e-6);
                assert!((p.pole_tilt_deg - 24.0).abs() < 1e-6);
                assert!((p.wheel - 0.01).abs() < 1e-6);
                assert!((p.meteors - 0.1).abs() < 1e-6);
            }
            other => panic!("expected Stars, got {other:?}"),
        }
    }

    #[test]
    fn parse_layer_aurora_band_and_palette() {
        let l = SkySpec::parse_layer(
            "aurora curtains=3 band=0.18..0.55 palette=oxygen activity=0.6 shimmer_hz=2",
        )
        .unwrap();
        match l {
            SkyLayerSpec::Aurora(p) => {
                assert_eq!(p.curtains, 3);
                assert!((p.band_lo - 0.18).abs() < 1e-6);
                assert!((p.band_hi - 0.55).abs() < 1e-6);
                assert_eq!(p.palette, AuroraPalette::Oxygen);
                assert!((p.activity - 0.6).abs() < 1e-6);
                assert!((p.shimmer_hz - 2.0).abs() < 1e-6);
            }
            other => panic!("expected Aurora, got {other:?}"),
        }
    }

    #[test]
    fn parse_layer_clouds_wind_and_bake() {
        let l = SkySpec::parse_layer("clouds cover=0.5 wind=0.1,-0.2 bake=on").unwrap();
        match l {
            SkyLayerSpec::Clouds(p) => {
                assert!((p.cover - 0.5).abs() < 1e-6);
                assert!((p.wind.0 - 0.1).abs() < 1e-6);
                assert!((p.wind.1 - (-0.2)).abs() < 1e-6);
                assert!(p.bake);
            }
            other => panic!("expected Clouds, got {other:?}"),
        }
    }

    #[test]
    fn parse_layer_constellation_three_stars_and_link() {
        let l = SkySpec::parse_layer(
            "constellation stars=(0.31,0.42) (0.34,0.47) (0.29,0.51) link=tag:star_a,star_b,star_c",
        )
        .unwrap();
        match l {
            SkyLayerSpec::Constellation(p) => {
                assert_eq!(p.stars[0], (0.31, 0.42));
                assert_eq!(p.stars[2], (0.29, 0.51));
                assert_eq!(
                    p.link,
                    [
                        "star_a".to_string(),
                        "star_b".to_string(),
                        "star_c".to_string()
                    ]
                );
            }
            other => panic!("expected Constellation, got {other:?}"),
        }
    }

    #[test]
    fn parse_layer_unknown_kind_is_an_error() {
        assert!(SkySpec::parse_layer("nonsense foo=1").is_err());
    }

    #[test]
    fn parse_layer_unknown_key_is_an_error() {
        assert!(SkySpec::parse_layer("stars bogus=1").is_err());
    }

    #[test]
    fn parse_layer_skyline_all_keys() {
        let l = SkySpec::parse_layer(
            "skyline blocks=48 density=0.6 height=0.1..0.4 style=server_rack lit=0.5 landmark_rate=0.2",
        )
        .unwrap();
        match l {
            SkyLayerSpec::Skyline(p) => {
                assert_eq!(p.blocks, 48);
                assert!((p.density - 0.6).abs() < 1e-6);
                assert!((p.height_lo - 0.1).abs() < 1e-6);
                assert!((p.height_hi - 0.4).abs() < 1e-6);
                assert_eq!(p.style, crate::FacadeStyle::ServerRack);
                assert!((p.lit - 0.5).abs() < 1e-6);
                assert!((p.landmark_rate - 0.2).abs() < 1e-6);
            }
            other => panic!("expected Skyline, got {other:?}"),
        }
    }

    #[test]
    fn parse_layer_skyline_rejects_unknown_style() {
        assert!(SkySpec::parse_layer("skyline style=chrome").is_err());
    }

    #[test]
    fn parse_layer_skyline_rejects_unknown_key() {
        assert!(SkySpec::parse_layer("skyline bogus=1").is_err());
    }

    #[test]
    fn from_kind_expands_known_presets() {
        for kind in [
            "stars",
            "aurora",
            "clouds",
            "storm",
            "datarain",
            "void",
            "fractal",
            "watcher",
            "mirror",
            "skyline",
            "constellation",
        ] {
            let spec = SkySpec::from_kind(kind).unwrap_or_else(|e| panic!("{kind}: {e}"));
            assert!(!spec.layers.is_empty(), "{kind} preset has no layers");
        }
    }

    #[test]
    fn from_kind_unknown_is_an_error() {
        assert!(SkySpec::from_kind("nonexistent").is_err());
    }

    #[test]
    fn default_spec_has_sane_hz() {
        // hz=0 would divide-by-zero downstream tick math; the default must be > 0.
        assert!(SkySpec::default().hz > 0);
    }
}
