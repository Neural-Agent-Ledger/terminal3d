//! Dev-only ECS inspection tooling (Bevy Remote Protocol). Compiled only under
//! the `dev` feature; never part of the shipped build. No seams: a host adds
//! [`DevToolingPlugin`] directly under its own `#[cfg(feature = "dev")]` gate.

use bevy::app::{App, Plugin};
use bevy::remote::http::RemoteHttpPlugin;
use bevy::remote::RemotePlugin;

/// Adds the Bevy Remote Protocol server (HTTP transport on the default port,
/// 15702) and registers the game's inspectable types so an external client can
/// LIST/GET/mutate them at runtime. Dev-only.
pub struct DevToolingPlugin;

impl Plugin for DevToolingPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(RemotePlugin::default())
            .add_plugins(RemoteHttpPlugin::default())
            .register_type::<re_world::CameraPose>()
            .register_type::<re_core::FrameTimingMetrics>();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::prelude::*;

    #[test]
    fn dev_plugin_registers_inspectable_types() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.add_plugins(DevToolingPlugin);
        let registry = app.world().resource::<AppTypeRegistry>().read();
        assert!(
            registry
                .get(std::any::TypeId::of::<re_world::CameraPose>())
                .is_some(),
            "CameraPose must be registered for BRP"
        );
        assert!(
            registry
                .get(std::any::TypeId::of::<re_core::FrameTimingMetrics>())
                .is_some(),
            "FrameTimingMetrics must be registered for BRP"
        );
    }
}
