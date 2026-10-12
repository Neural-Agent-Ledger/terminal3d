//! renet/UDP transport setup for the testbed (server + client).
//!
//! Thin wrappers over `bevy_replicon_renet`'s netcode transport, following the
//! exact pattern verified from the crate's own `tests/netcode.rs`. NO terminal,
//! NO game logic — just socket + transport construction and resource insertion.
//!
//! The renet resources MUST be inserted AFTER plugins are built and `finish()`
//! has run, because `RepliconChannels` (which sizes the renet channels) is only
//! complete once every replication rule + client message is registered.

use std::fmt;
use std::net::{Ipv4Addr, SocketAddr, UdpSocket};
use std::time::SystemTime;

use bevy::prelude::*;
use bevy_replicon::prelude::RepliconChannels;
use bevy_replicon_renet::netcode::{
    ClientAuthentication, NetcodeClientTransport, NetcodeServerTransport, ServerAuthentication,
    ServerConfig,
};
use bevy_replicon_renet::renet::ConnectionConfig;
use bevy_replicon_renet::{RenetChannelsExt, RenetClient, RenetServer};

/// Shared netcode protocol id. The testbed uses a single fixed value; client and
/// server must agree or the handshake silently fails.
pub const PROTOCOL_ID: u64 = 0;

/// A recoverable transport-setup failure. The testbed prints this and exits
/// non-zero rather than panicking when (e.g.) the requested UDP port is already
/// in use or the system clock predates the UNIX epoch.
#[derive(Debug)]
pub enum TransportError {
    /// The system clock is before the UNIX epoch (`SystemTime` underflow).
    Clock,
    /// Binding the UDP socket failed (e.g. the port is already in use).
    Bind(std::io::Error),
    /// Reading the bound socket's local address failed.
    LocalAddr(std::io::Error),
    /// renet rejected the netcode transport configuration.
    Netcode(String),
}

impl fmt::Display for TransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransportError::Clock => write!(f, "system clock is before the UNIX epoch"),
            TransportError::Bind(e) => write!(f, "could not bind UDP socket: {e}"),
            TransportError::LocalAddr(e) => write!(f, "could not read socket local address: {e}"),
            TransportError::Netcode(e) => write!(f, "could not create netcode transport: {e}"),
        }
    }
}

impl std::error::Error for TransportError {}

/// Seconds since the UNIX epoch, or [`TransportError::Clock`] on underflow.
fn current_time() -> Result<std::time::Duration, TransportError> {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_err(|_| TransportError::Clock)
}

/// Build a [`NetcodeServerTransport`] bound to `0.0.0.0:port` and return it with
/// the actually-bound port (so passing `port = 0` yields an OS-assigned port —
/// used by the integration test for a collision-free ephemeral port).
pub fn server_transport(
    port: u16,
    max_clients: usize,
) -> Result<(NetcodeServerTransport, u16), TransportError> {
    let current_time = current_time()?;
    let server_addr = SocketAddr::new(Ipv4Addr::UNSPECIFIED.into(), port);
    let socket = UdpSocket::bind(server_addr).map_err(TransportError::Bind)?;
    let public_addr = socket.local_addr().map_err(TransportError::LocalAddr)?;
    let bound_port = public_addr.port();
    let server_config = ServerConfig {
        current_time,
        max_clients,
        protocol_id: PROTOCOL_ID,
        public_addresses: vec![public_addr],
        authentication: ServerAuthentication::Unsecure,
    };
    let transport = NetcodeServerTransport::new(server_config, socket)
        .map_err(|e| TransportError::Netcode(e.to_string()))?;
    Ok((transport, bound_port))
}

/// Build a [`NetcodeClientTransport`] connecting to `addr:port` with `client_id`.
pub fn client_transport(
    client_id: u64,
    addr: Ipv4Addr,
    port: u16,
) -> Result<NetcodeClientTransport, TransportError> {
    let current_time = current_time()?;
    let server_addr = SocketAddr::new(addr.into(), port);
    let socket = UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0)).map_err(TransportError::Bind)?;
    let authentication = ClientAuthentication::Unsecure {
        client_id,
        protocol_id: PROTOCOL_ID,
        server_addr,
        user_data: None,
    };
    NetcodeClientTransport::new(current_time, authentication, socket)
        .map_err(|e| TransportError::Netcode(e.to_string()))
}

/// A [`ConnectionConfig`] sized from the app's [`RepliconChannels`]. Must be
/// called AFTER `app.finish()` so the channel set is frozen.
pub fn connection_config(app: &App) -> ConnectionConfig {
    let channels = app.world().resource::<RepliconChannels>();
    ConnectionConfig {
        server_channels_config: channels.server_configs(),
        client_channels_config: channels.client_configs(),
        ..Default::default()
    }
}

/// Insert the renet SERVER resources (`RenetServer` + transport) into a finished
/// server app. Returns the bound port. Call after `app.finish()`.
pub fn insert_server(app: &mut App, port: u16, max_clients: usize) -> Result<u16, TransportError> {
    let config = connection_config(app);
    let server = RenetServer::new(config);
    let (transport, bound_port) = server_transport(port, max_clients)?;
    app.insert_resource(server).insert_resource(transport);
    Ok(bound_port)
}

/// Insert the renet CLIENT resources (`RenetClient` + transport) into a finished
/// client app. Call after `app.finish()`.
pub fn insert_client(
    app: &mut App,
    client_id: u64,
    addr: Ipv4Addr,
    port: u16,
) -> Result<(), TransportError> {
    let config = connection_config(app);
    let client = RenetClient::new(config);
    let transport = client_transport(client_id, addr, port)?;
    app.insert_resource(client).insert_resource(transport);
    Ok(())
}
