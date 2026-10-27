//! Pong 3D, in the Orange Vélodrome: the port of the engine's `pong`.
//!
//! ```text
//! cargo run --bin pong              # two players on one keyboard
//! cargo run --bin pong -- --demo    # it plays itself, the camera orbits
//! cargo run --bin pong -- --demo2   # the camera chases the ball
//! cargo run --bin pong -- --server  # headless, authoritative
//! cargo run --bin pong -- --client 192.168.1.10
//! ```
//!
//! Left player: W/A/S/D, Left Shift to lob, Left Ctrl to boost, Space to dash.
//! Right player: the arrows, Right Shift, Enter and Numpad 0. The first two
//! gamepads play too: stick or d-pad, a right trigger or B to lob, a left
//! trigger to boost, A to dash, Start to pause. Escape pauses.
//!
//! As in the engine the game opens borderless fullscreen (`--windowed` not to)
//! and reads `config.json` from the working directory if there is one:
//! `vsync` and `hdr_mode` (`auto`, `enabled` or `disabled`).
//!
//! The modules, in the order a frame meets them: `input` turns keys into
//! `state::PlayerInput`, `state` runs the rules (shared with `net`'s server),
//! `game` moves the `stadium`'s models and the effects and plays the `sound`,
//! and `ui` draws the menus over the scene.

mod demo;
mod game;
mod input;
mod net;
mod sound;
mod stadium;
mod state;
mod ui;

use anyhow::{Context as _, Result};
use clap::Parser;
use etib::{Hdr, Vsync};
use log::info;
use serde::Deserialize;

use crate::demo::DemoMode;
use crate::game::PongApp;
use crate::net::Client;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Act as authoritative server
    #[arg(long, default_value_t = false)]
    server: bool,

    /// IP Address of server to connect to
    #[arg(long)]
    client: Option<String>,

    /// Port to listen (server) or send (client) on
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// Export the generated static stadium geometry to a `.model` file and exit
    #[arg(long, value_name = "PATH")]
    export_static: Option<String>,

    /// Run an attract/demo mode: no menus or HUD, paddles play themselves while the
    /// camera slowly orbits the pitch
    #[arg(long, default_value_t = false)]
    demo: bool,

    /// Like `--demo`, but the camera chases the ball from behind
    #[arg(long, default_value_t = false)]
    demo2: bool,

    /// Open in a window rather than borderless fullscreen
    #[arg(long, default_value_t = false)]
    windowed: bool,
}

/// What the engine's `config.json` held. A missing file, or a missing field,
/// is the default.
#[derive(Debug, Deserialize)]
#[serde(default)]
struct EngineConfig {
    vsync: bool,
    hdr_mode: HdrMode,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            vsync: true,
            hdr_mode: HdrMode::Auto,
        }
    }
}

#[derive(Debug, Deserialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
enum HdrMode {
    Auto,
    Enabled,
    Disabled,
}

impl EngineConfig {
    fn load(path: &str) -> Self {
        let Ok(content) = std::fs::read_to_string(path) else {
            info!("no {path}; using the defaults");
            return Self::default();
        };
        serde_json::from_str(&content).unwrap_or_else(|err| {
            log::warn!("{path} could not be read ({err}); using the defaults");
            Self::default()
        })
    }
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();

    let config = EngineConfig::load("config.json");
    let args = Args::parse();

    if let Some(path) = args.export_static {
        let (walls, _seats) = stadium::make_static_geometry();
        etib_scene::save_model(&path, &walls).with_context(|| format!("writing {path}"))?;
        println!("Exported {} static cubes to {path}", walls.len());
        return Ok(());
    }

    if args.server {
        return net::run_server(args.port);
    }

    let demo = if args.demo2 {
        DemoMode::Follow
    } else if args.demo {
        DemoMode::Orbit
    } else {
        DemoMode::Off
    };
    let server = args.client.map(|ip| format!("{ip}:{}", args.port));

    etib::run(
        etib::Config {
            title: "Pong 3D".to_owned(),
            // Escape pauses the game rather than closing it.
            exit_on_escape: false,
            hdr: match config.hdr_mode {
                HdrMode::Auto => Hdr::Allow,
                HdrMode::Enabled => Hdr::Force,
                HdrMode::Disabled => Hdr::Off,
            },
            fullscreen: !args.windowed,
            vsync: if config.vsync { Vsync::On } else { Vsync::Off },
            ..Default::default()
        },
        // Built again if the platform recreates the window, connection and all.
        move |gpu| {
            let client = server.clone().map(Client::connect).transpose()?;
            PongApp::new(gpu, demo, client)
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_engine_config_reads_as_it_did() {
        let config: EngineConfig =
            serde_json::from_str(r#"{ "vsync": false, "hdr_mode": "enabled", "culling": true }"#)
                .unwrap();
        assert!(!config.vsync);
        assert!(matches!(config.hdr_mode, HdrMode::Enabled));

        let empty: EngineConfig = serde_json::from_str("{}").unwrap();
        assert!(empty.vsync);
        assert!(matches!(empty.hdr_mode, HdrMode::Auto));
    }

    #[test]
    fn the_arguments_are_the_engines() {
        let args = Args::parse_from(["pong", "--client", "10.0.0.2", "--port", "9000", "--demo2"]);
        assert_eq!(args.client.as_deref(), Some("10.0.0.2"));
        assert_eq!(args.port, 9000);
        assert!(args.demo2 && !args.demo && !args.server);
    }
}
