//! `cargo xtask layers` — command-line entry point for the layering guard.
//!
//! Thin by design: all the logic lives in `xtask::check_workspace`, which is
//! also what `cargo test -p xtask` exercises directly (see `src/lib.rs`), so
//! this file only parses a subcommand and turns a [`xtask::Report`] into
//! process exit status and stdout/stderr text.
//!
//! Invoke via the workspace alias (`.cargo/config.toml`'s `[alias] xtask`):
//!
//! ```text
//! cargo xtask layers
//! ```

use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("layers") => run_layers(),
        Some(other) => {
            eprintln!("xtask: unknown subcommand `{other}` (expected `layers`)");
            ExitCode::FAILURE
        }
        None => {
            eprintln!("xtask: expected a subcommand — try `cargo xtask layers`");
            ExitCode::FAILURE
        }
    }
}

/// Fetch the workspace's resolved dependency graph and check it against the
/// engine/editor/apps/xtask direction table plus the terminal-isolation rule.
fn run_layers() -> ExitCode {
    let metadata = xtask::workspace_metadata();
    let report = xtask::check_workspace(&metadata);
    println!("{}", report.describe());
    if report.is_ok() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
