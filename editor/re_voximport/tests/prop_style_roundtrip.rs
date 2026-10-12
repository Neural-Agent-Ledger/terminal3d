//! `[meta] prop_style` round-trip through `re_voximport::to_lvl_text`.
//!
//! `re_level` used to derive a level's default prop pack from a
//! `campaign.toml` shared between the engine and game trees — an approach
//! that could not survive the engine/game repository split. Levels are now
//! self-describing: a
//! `[meta] prop_style` key, parsed by `re_level` (default `lab` when absent).
//! This file proves the OTHER half of that contract: the writer re-emits an
//! authored key instead of silently dropping it (the exact "authored, parsed,
//! never written back" bug class `version`/`spec_rev` already guard against),
//! while an unauthored level — everything shipped before this key existed —
//! stays byte-identical across a save.

use re_level::props::PropStyle;

/// Path to a repo-root level file (the workspace root is two dirs above this
/// crate's manifest — same convention as `re_voximport::tests::lvl_path`).
fn lvl_path(name: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../levels")
        .join(name)
}

/// An authored `[meta] prop_style` survives `to_lvl_text` → reparse → re-emit,
/// and the second emit is byte-identical to the first (idempotent), for every
/// named pack.
#[test]
fn authored_prop_style_round_trips_for_every_pack() {
    for style in PropStyle::ALL {
        let src = format!(
            "[meta]\nname = style-rt\nprop_style = {}\n[room]\nsize = 4 4 4\n\
             [pedestal]\ncenter = 1,1,1\nvolume = 1,1,1\n\
             [door]\npos = 0 1 1\n[player]\nspawn = 1 1 1\n",
            style.as_str()
        );
        let lv = re_level::load_string(&src, "style-rt.lvl")
            .unwrap_or_else(|e| panic!("{}: must load: {e}", style.as_str()));
        assert_eq!(lv.prop_default_style, style, "{}", style.as_str());
        assert!(
            lv.prop_style_authored,
            "{}: must be marked authored",
            style.as_str()
        );

        let t1 = re_voximport::to_lvl_text(&lv);
        assert!(
            t1.lines()
                .any(|l| l == format!("prop_style = {}", style.as_str())),
            "{}: writer must re-emit the authored key:\n{t1}",
            style.as_str()
        );

        let lv2 = re_level::load_string(&t1, "style-rt.lvl")
            .unwrap_or_else(|e| panic!("{}: re-parse failed: {e}\n---\n{t1}\n---", style.as_str()));
        assert_eq!(
            lv2.prop_default_style,
            style,
            "{}: survives the round trip",
            style.as_str()
        );
        assert!(
            lv2.prop_style_authored,
            "{}: stays authored after a round trip",
            style.as_str()
        );

        let t2 = re_voximport::to_lvl_text(&lv2);
        assert_eq!(
            t1,
            t2,
            "{}: to_lvl_text must be idempotent for prop_style",
            style.as_str()
        );
    }
}

/// A level that never authored `[meta] prop_style` gets the struct default
/// (`Lab`) at load time, but the writer must NOT invent the key: emitting
/// `prop_style = lab` for a level that never asked for it would be exactly the
/// silent-mutation-on-save bug this key's design (§4.2 E2) exists to avoid, and
/// would perturb every one of the ~50 non-campaign shipped levels on their next
/// editor save.
#[test]
fn unauthored_level_writer_omits_the_key() {
    let src = "[meta]\nname = no-style\n[room]\nsize = 4 4 4\n\
               [pedestal]\ncenter = 1,1,1\nvolume = 1,1,1\n\
               [door]\npos = 0 1 1\n[player]\nspawn = 1 1 1\n";
    let lv = re_level::load_string(src, "no-style.lvl").expect("must load");
    assert_eq!(
        lv.prop_default_style,
        PropStyle::Lab,
        "absent key ⇒ struct default"
    );
    assert!(!lv.prop_style_authored);

    let t1 = re_voximport::to_lvl_text(&lv);
    assert!(
        !t1.lines().any(|l| l.trim_start().starts_with("prop_style")),
        "unauthored level must not gain a prop_style line:\n{t1}"
    );

    // Idempotent: reparsing the writer's own output changes nothing further.
    let lv2 = re_level::load_string(&t1, "no-style.lvl").expect("re-parse must succeed");
    assert!(!lv2.prop_style_authored);
    assert_eq!(
        t1,
        re_voximport::to_lvl_text(&lv2),
        "to_lvl_text must be idempotent"
    );
}

/// A real shipped level that predates `prop_style` (an ARCHIVE room, not one of
/// the twelve campaign levels) round-trips with the key still absent — the
/// A1 fix must not perturb a single byte of the ~50 levels it does not touch.
#[test]
fn a_pre_existing_shipped_level_stays_unauthored_across_a_save() {
    let lv = re_level::load(lvl_path("01.lvl")).expect("01.lvl must load");
    assert!(
        !lv.prop_style_authored,
        "01.lvl predates prop_style and must not be retroactively marked authored"
    );
    assert_eq!(lv.prop_default_style, PropStyle::Lab);
    let t1 = re_voximport::to_lvl_text(&lv);
    assert!(!t1.lines().any(|l| l.trim_start().starts_with("prop_style")));
}

// `every_campaign_level_authors_and_round_trips_prop_style` MOVED / REMOVED.
// It loaded the twelve campaign `L*.lvl` files from `../../levels` relative
// to this crate, which held while campaign and engine levels shared one
// directory; after the engine/game repository split those files are not
// under `terminal3d/` at all. Its coverage is not lost: the WRITER path this
// test exercised (authored `prop_style` survives `to_lvl_text` → reparse →
// re-emit, idempotently) is already proven generically for every `PropStyle`
// value by `authored_prop_style_round_trips_for_every_pack` above, over
// synthetic fixtures — the campaign levels only ever used those same four
// enum values. The CAMPAIGN-specific fact (that all twelve author the key and
// the value agrees with the campaign manifest) is asserted, with a real path
// to that manifest, by the host game's own
// `campaign_prop_style_agrees_with_campaign_toml_movement` test — which
// already performs its own save/reload round trip via
// `re_voximport::to_lvl_text`, so nothing this test checked is now unchecked.
