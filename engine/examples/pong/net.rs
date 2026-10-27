//! Playing over the network: an authoritative headless server, and the client
//! half that runs inside the game.
//!
//! The protocol is the engine's, byte for byte — bincode 1 over UDP, a
//! [`PlayerInput`] from each client every frame and a [`GameStateSnapshot`]
//! back from the server every tick — so either side can talk to the other's
//! engine build.

use std::io::ErrorKind;
use std::net::{SocketAddr, UdpSocket};
use std::time::Duration;

use log::{error, info};
use serde::{Deserialize, Serialize};

use crate::state::{Ball, Player, PlayerInput, step_physics};

/// Big enough for either message, with room to spare.
const DATAGRAM: usize = 1024;
/// The server's tick.
const TICK: f32 = 1.0 / 60.0;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct GameStateSnapshot {
    pub ball_pos: [f32; 3], // X, Y, Z
    pub ball_vel: [f32; 3], // X, Y, Z - for client prediction
    pub left_pos: [f32; 2], // X, Y
    pub left_boost_active: bool,
    pub left_dash_active: bool,
    pub right_pos: [f32; 2], // X, Y
    pub right_boost_active: bool,
    pub right_dash_active: bool,
    pub left_score: u32,
    pub right_score: u32,
    pub auth_sequence_left: u32,
    pub auth_sequence_right: u32,
}

impl GameStateSnapshot {
    /// Overwrites the local state with the server's.
    pub fn apply(&self, ball: &mut Ball, left: &mut Player, right: &mut Player) {
        ball.p = self.ball_pos.into();
        ball.v = self.ball_vel.into();
        [left.x, left.y] = self.left_pos;
        [right.x, right.y] = self.right_pos;
        left.boost_active = self.left_boost_active;
        right.boost_active = self.right_boost_active;
        // Only "dashing or not" crosses the wire; enough to colour the paddle.
        left.dash_timer = if self.left_dash_active { 0.1 } else { 0.0 };
        right.dash_timer = if self.right_dash_active { 0.1 } else { 0.0 };
        left.score = self.left_score;
        right.score = self.right_score;
    }
}

/// The game's side of the connection.
pub struct Client {
    socket: UdpSocket,
    server: String,
    sequence: u32,
}

impl Client {
    /// A client sending to `server`, an `ip:port` string.
    pub fn connect(server: String) -> anyhow::Result<Self> {
        let socket = UdpSocket::bind("0.0.0.0:0")?;
        socket.set_nonblocking(true)?;
        info!("Pong Client initialized. Target: {server}");
        Ok(Self {
            socket,
            server,
            sequence: 0,
        })
    }

    /// Sends this frame's input, numbered, and returns the newest snapshot
    /// that has arrived since the last call, if any.
    pub fn exchange(&mut self, mut input: PlayerInput) -> Option<GameStateSnapshot> {
        self.sequence += 1;
        input.sequence_number = self.sequence;
        if let Ok(bytes) = bincode::serialize(&input) {
            let _ = self.socket.send_to(&bytes, &self.server);
        }

        let mut buf = [0u8; DATAGRAM];
        let mut latest = None;
        while let Ok(size) = self.socket.recv(&mut buf) {
            if let Ok(snapshot) = bincode::deserialize::<GameStateSnapshot>(&buf[..size]) {
                latest = Some(snapshot);
            }
        }
        latest
    }
}

struct ServerPlayerState {
    addr: SocketAddr,
    latest_input: PlayerInput,
    last_seq_processed: u32,
}

/// The authoritative game: it runs the rules, the clients only send input.
pub struct PongServer {
    socket: UdpSocket,
    ball: Ball,
    left_player: Player,
    right_player: Player,
    left_net: Option<ServerPlayerState>,
    right_net: Option<ServerPlayerState>,
}

impl PongServer {
    pub fn new(port: u16) -> anyhow::Result<Self> {
        let socket = UdpSocket::bind(format!("0.0.0.0:{port}"))?;
        socket.set_nonblocking(true)?;
        Ok(Self {
            socket,
            ball: Ball::default(),
            left_player: Player::left(),
            right_player: Player::right(),
            left_net: None,
            right_net: None,
        })
    }

    /// The port it is listening on, which is the one asked for unless that
    /// was 0.
    pub fn port(&self) -> u16 {
        self.socket.local_addr().map_or(0, |addr| addr.port())
    }

    /// Takes every input that has arrived. The first two addresses heard from
    /// become the left and right players; an input older than one already
    /// used is dropped.
    pub fn receive_inputs(&mut self) {
        let mut buf = [0u8; DATAGRAM];
        loop {
            let (size, src) = match self.socket.recv_from(&mut buf) {
                Ok(received) => received,
                Err(err) if err.kind() == ErrorKind::WouldBlock => break,
                Err(err) => {
                    error!("UDP Recv error: {err}");
                    break;
                }
            };
            let Ok(input) = bincode::deserialize::<PlayerInput>(&buf[..size]) else {
                continue;
            };

            let slot = if self.left_net.as_ref().is_some_and(|p| p.addr == src) {
                &mut self.left_net
            } else if self.right_net.as_ref().is_some_and(|p| p.addr == src) {
                &mut self.right_net
            } else if self.left_net.is_none() {
                info!("Left player registered: {src}");
                &mut self.left_net
            } else if self.right_net.is_none() {
                info!("Right player registered: {src}");
                &mut self.right_net
            } else {
                continue;
            };

            match slot {
                Some(player) => {
                    if input.sequence_number > player.last_seq_processed {
                        player.latest_input = input;
                    }
                }
                None => {
                    *slot = Some(ServerPlayerState {
                        addr: src,
                        last_seq_processed: input.sequence_number,
                        latest_input: input,
                    });
                }
            }
        }
    }

    pub fn step_physics(&mut self, dt: f32) {
        let input = |net: &Option<ServerPlayerState>| {
            net.as_ref()
                .map(|p| p.latest_input.clone())
                .unwrap_or_default()
        };
        let (left_input, right_input) = (input(&self.left_net), input(&self.right_net));

        step_physics(
            &mut self.ball,
            &mut self.left_player,
            &mut self.right_player,
            &left_input,
            &right_input,
            dt,
        );

        for p in [&mut self.left_net, &mut self.right_net]
            .into_iter()
            .flatten()
        {
            p.last_seq_processed = p.latest_input.sequence_number;
        }
    }

    pub fn snapshot(&self) -> GameStateSnapshot {
        let seq =
            |net: &Option<ServerPlayerState>| net.as_ref().map_or(0, |p| p.last_seq_processed);
        GameStateSnapshot {
            ball_pos: self.ball.p.into(),
            ball_vel: self.ball.v.into(),
            left_pos: [self.left_player.x, self.left_player.y],
            left_boost_active: self.left_player.boost_active,
            left_dash_active: self.left_player.dash_timer > 0.0,
            right_pos: [self.right_player.x, self.right_player.y],
            right_boost_active: self.right_player.boost_active,
            right_dash_active: self.right_player.dash_timer > 0.0,
            left_score: self.left_player.score,
            right_score: self.right_player.score,
            auth_sequence_left: seq(&self.left_net),
            auth_sequence_right: seq(&self.right_net),
        }
    }

    pub fn broadcast_state(&mut self) -> anyhow::Result<()> {
        let bytes = bincode::serialize(&self.snapshot())?;
        for p in [&self.left_net, &self.right_net].into_iter().flatten() {
            let _ = self.socket.send_to(&bytes, p.addr);
        }
        Ok(())
    }
}

/// Runs the headless server until the process is stopped.
pub fn run_server(port: u16) -> anyhow::Result<()> {
    let mut server = PongServer::new(port)?;
    info!("Starting Headless Pong Server on port {}", server.port());

    info!("Listening for clients...");
    loop {
        server.receive_inputs();
        server.step_physics(TICK);
        server.broadcast_state()?;
        std::thread::sleep(Duration::from_millis(16));
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;

    /// Polls `f` until it is true or a second has gone by: the sockets are
    /// non-blocking, and loopback still takes a moment.
    fn eventually(mut f: impl FnMut() -> bool) -> bool {
        let start = Instant::now();
        while start.elapsed() < Duration::from_secs(1) {
            if f() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        false
    }

    #[test]
    fn a_snapshot_round_trips_and_applies() {
        let mut server = PongServer::new(0).unwrap();
        server.ball.p.x = 5.0;
        server.left_player.score = 3;
        server.right_player.dash_timer = 0.05;
        let snapshot = server.snapshot();
        let bytes = bincode::serialize(&snapshot).unwrap();
        let back: GameStateSnapshot = bincode::deserialize(&bytes).unwrap();
        assert_eq!(back, snapshot);

        let (mut ball, mut left, mut right) = (Ball::default(), Player::left(), Player::right());
        back.apply(&mut ball, &mut left, &mut right);
        assert_eq!(ball.p.x, 5.0);
        assert_eq!(left.score, 3);
        assert!(right.dash_timer > 0.0);
    }

    #[test]
    fn clients_are_registered_and_sent_the_state() {
        let mut server = PongServer::new(0).unwrap();
        let address = format!("127.0.0.1:{}", server.port());
        let mut first = Client::connect(address.clone()).unwrap();
        let mut second = Client::connect(address).unwrap();

        let up = PlayerInput {
            move_y: 1.0,
            ..PlayerInput::default()
        };
        first.exchange(up.clone());
        assert!(eventually(|| {
            server.receive_inputs();
            server.left_net.is_some()
        }));
        second.exchange(PlayerInput::default());
        assert!(eventually(|| {
            server.receive_inputs();
            server.right_net.is_some()
        }));

        let before = server.left_player.y;
        server.step_physics(TICK);
        assert!(
            server.left_player.y > before,
            "the left client's input moved its paddle"
        );
        server.broadcast_state().unwrap();

        let mut got = None;
        assert!(eventually(|| {
            got = first.exchange(up.clone()).or(got.take());
            got.is_some()
        }));
        assert_eq!(got.unwrap().auth_sequence_left, 1);
    }

    #[test]
    fn stale_inputs_are_dropped() {
        let mut server = PongServer::new(0).unwrap();
        let client = UdpSocket::bind("127.0.0.1:0").unwrap();
        let address = format!("127.0.0.1:{}", server.port());
        let send = |seq: u32, move_y: f32| {
            let input = PlayerInput {
                move_y,
                sequence_number: seq,
                ..PlayerInput::default()
            };
            client
                .send_to(&bincode::serialize(&input).unwrap(), &address)
                .unwrap();
        };

        send(5, 1.0);
        assert!(eventually(|| {
            server.receive_inputs();
            server.left_net.is_some()
        }));
        send(3, -1.0);
        send(9, 0.5);
        assert!(eventually(|| {
            server.receive_inputs();
            server
                .left_net
                .as_ref()
                .unwrap()
                .latest_input
                .sequence_number
                == 9
        }));
        assert_eq!(server.left_net.as_ref().unwrap().latest_input.move_y, 0.5);
    }
}
